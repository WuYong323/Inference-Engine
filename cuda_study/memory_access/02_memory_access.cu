// ============================================================================
// 02_memory_access.cu   访存模式 与 有效带宽
//
// 目的：用同一份数据、同样的搬运量，只改「访问模式」，实测有效带宽差多少倍。
//
// 编译（一定要带 -lineinfo，否则 ncu 无法把计数器对回源码行）：
//   nvcc -O2 -arch=sm_120 -lineinfo -Xptxas -v -o 02_memory_access 02_memory_access.cu(50系 sm_12) (h100 sm_90)
//   （-Xptxas -v 会打印每个 kernel 用了多少寄存器、有没有 spill）
// 运行：
//   ./02_memory_access                # 跑全部模式，打印有效带宽表
//   ./02_memory_access 26             # 自定义规模：n = 1<<26 个 float（小显存卡用这个）
// Profile：
//   ncu --set full -o 02_memory_access_rep ./02_memory_access
//   ncu --metrics \
//     l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
//     dram__bytes_read.sum,\
//     dram__throughput.avg.pct_of_peak_sustained_elapsed ./02_memory_access
//   nsys profile --stats=true -t cuda -o 02_memory_access_nsys ./02_memory_access
// 检查越界：
//   compute-sanitizer ./02_memory_access
// ============================================================================


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


// ---------------------------------------------------------------------------
// 0. 错误检查宏：CUDA 的 API 全部返回错误码
//    为什么写成 do{}while(0)：让宏在 if/else 里也能像普通语句一样安全使用
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                            \
    do{                                                             \
        cudaError_t err=(call);                                     \
        if(err!=cudaSuccess){                                       \
            fprintf(stderr,"[CUDA ERROR] %s:%d  %s\n",              \
                    __FILE__,__LINE__,cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    }while(0)

// ===========================================================================
// 1. 五个 copy kernel —— 算的事完全一样（把 in 搬到 out），只有「地址怎么算」不同
// ===========================================================================

// (A) 合并访问：warp 内 32 个线程读连续的 32 个 float = 连续 128 字节
//     __restrict__ 告诉编译器 in/out 不重叠，允许它把 load 提前发射（更好的访存重叠）
__global__ void copy_coalesced(const float* __restrict__ in,float* __restrict__ out,size_t n){
    size_t idx=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(idx<n) out[idx]=in[idx];         // 线程0读in[0]、线程1读in[1]... 硬件可合并
}

// (B) 错位访问：整体平移 offset 个 float，破坏 128B 对齐
//     现实来源：KV Cache 从第 k 个 token 开始读、图像 ROI 裁剪、padding 没对齐
__global__ void copy_offset(const float* __restrict__ in,float* __restrict__ out,size_t n,int offset){
    size_t idx=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(idx+offset<n) out[idx]=in[idx+offset];     // 一个 warp 的 128B 请求跨了 5 个 32B sector
}


// (C) 跨步访问：warp 内相邻线程读相隔 (n/stride) 个元素的地址
//       关键设计：src 是 idx 的「双射」(bijection)，每个元素恰好被读一次。
//       这样读到的数据总量和 (A) 完全一致，差异 100% 来自访问模式，
//       而不是「数据集变小、全落 L2」造成的假象。
__global__ void copy_strided(const float* __restrict__ in,float* __restrict__ out,size_t n,int stride){
    size_t idx=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(idx<n){
        size_t chunk=n/stride;              // 把数组看成 stride 列 × chunk 行
        size_t src=(idx%stride)*chunk+idx/stride;       // 转置式读取
        out[idx]=in[src];           // 写仍然连续 —— 单独隔离「读」的合并度
    }
}


// (D) 向量化访问：一个线程一次搬 16 字节（float4），编译成 LDG.E.128
//     工业界默认操作：同样的带宽下指令数 ÷4，更容易把访存流水线灌满
//     前提：指针必须 16B 对齐（cudaMalloc 返回的地址天然 256B 对齐，安全）
__global__ void copy_vec4(const float4* __restrict__ in,float4* __restrict__ out,size_t n4){
    size_t idx=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(idx<n4) out[idx]=in[idx];
}


// (E) grid-stride loop：工业标准写法，grid 大小与数据规模解耦
//     好处：kernel 只 launch 固定份数的 block（按 SM 数量调），
//           n 再大也不用改 grid；也便于持久化 kernel / 巨核风格的调度
__global__ void copy_grid_stride(const float* __restrict__ in,float* __restrict__ out,size_t n){
    size_t gsize=(size_t)gridDim.x*blockDim.x;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=gsize){
        out[i]=in[i];            // 每一步内部仍然是连续的 —— 合并度不受影响(在这一次迭代中，所有线程都在同时工作)
    }
}



// ===========================================================================
// 2. 计时工具：cudaEvent（GPU 侧时间戳，不受 CPU 调度抖动影响）
// ===========================================================================
struct Timer{
    cudaEvent_t s,e;
    //构造函数 Timer()
    //当创建 Timer t; 时自动调用，用来初始化（比如创建 CUDA 事件 cudaEventCreate）。
    Timer() {
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
    }
    //~ 是析构函数（Destructor）的专用语法标记。
    //当对象 t 离开作用域（比如 bench 函数执行结束）时自动调用，用来做“收尾清理”工作（比如销毁 CUDA 事件 cudaEventDestroy）。
    ~Timer() {
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    void tic() {
        CUDA_CHECK(cudaEventRecord(s));
    }
    float toc() {
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
        return ms;
    }
};


// 把「预热 + 多次取平均 + 同步」这套三铁律封起来，避免每个 case 重复写错
template <typename F>
static float bench(F&& launch,int warmup=5,int iters=20){
    for(int i=0;i<warmup;++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    Timer t;
    t.tic();
    for(int i=0;i<iters;++i) launch();
    float ms=t.toc()/iters;
    CUDA_CHECK(cudaGetLastError());
    return ms;
}


// ===========================================================================
// 3. 正确性校验：性能实验也必须先证明「算对了」，否则测的是垃圾
//    （W8 三尺子纪律的 C++ 版：这里数据是精确 copy，直接比 bit 相等）
// ===========================================================================
static void verify(const float* h_out,size_t n,int mode,int stride,int offset){
    size_t bad=0;
    for(size_t i=0;i<n;++i){
        float expect;
        if(mode==0||mode==3||mode==4) expect=(float)i;
        else if(mode==1) expect=(i+offset<n)?(float)(i+offset):h_out[i];
        else {
            size_t chunk=n/stride;
            expect=(float)((i%stride)*chunk+i/stride);
        }
        if(h_out[i]!=expect&&++bad<=5){
            fprintf(stderr,"mismatch @%zu: got %.1f expect %.1f\n",i,h_out[i],expect);
        }
    }
    printf(" correctness: %s (%zu mismatches)\n",bad ? "FAIL" : "OK",bad);
}



// argc: Argument Count（参数个数）。它是一个整数，记录你启动程序时，在命令行敲了几个“单词”（即参数）。
// argv: Argument Vector（参数向量）。它是一个字符串数组，里面存放着你敲的每一个具体的“单词”。
int main(int argc,char** argv){
    // ---------------- 设备信息：先把「理论峰值」这把尺子拿到手 ----------------
    int dev=0;
    cudaDeviceProp p;           // CUDA 运行时 API 中定义的一个结构体（Struct）,存储当前 GPU 硬件的全部静态属性信息。
    CUDA_CHECK(cudaGetDevice(&dev));         //获取当前上下文正在使用的设备 ID
    CUDA_CHECK(cudaGetDeviceProperties(&p,dev));         //把 dev 号 GPU 的所有硬件属性填充进结构体 p 中
    printf("GPU: %s | SM=%d | L2=%.1f MB | sharedMem/SM(opt-in)=%.0f KB\n",     //打印设备信息
            p.name,p.multiProcessorCount,p.l2CacheSize/1048576.0,p.sharedMemPerMultiprocessor/1024.0);

    // 规模选择： 缓冲区必须远大于 L2，否则测的是 L2 带宽而不是 HBM 带宽
    // H100 的 L2 有 50 MB —— 教程里常见的 64MB 数组在 H100 上会被 L2 严重污染
    int shift=(argc>1) ? atoi(argv[1]):27;      // 默认 n = 1<<27 = 512 MB / buffer
    size_t n=(size_t)1<<shift;
    size_t bytes=n*sizeof(float);
    printf("n = %zu floats, buffer = %.0f MB each (L2 = %.0f MB) -> %s\n\n",
            n, bytes/1048576.0,p.l2CacheSize/1048576.0,
            bytes > 4.0*p.l2CacheSize ? "OK, HBM-bound" : "WARNING: 太小，会被 L2 污染");

    // ---------------- 分配 + 初始化 ----------------
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for(size_t i=0;i<n;++i){
        h_in[i]=(float)i;          // 可校验的模式：值 == 下标
    }

    float *d_in=nullptr;
    float *d_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,bytes));
    CUDA_CHECK(cudaMalloc(&d_out,bytes));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice));      // 唯一一次过 PCIe

    const int block=256;                          // 每个线程块（Block）包含 256 个线程, 256 = 8 个 warp，通用稳妥值
    const size_t grid=(n+block-1)/block;          // 如果 n 大到超过了 CUDA 规定的最大网格（Grid）尺寸限制，这个 Kernel 启动会直接报错（cudaErrorInvalidConfiguration）
    const int stride=32;
    const int offset=1;

    struct Row{
        const char* name;
        float ms;
    }rows[5];

    // ---- (A) coalesced ----
    rows[0]={"coalesced (baseline)", bench([&]{ copy_coalesced<<<grid,block>>>(d_in,d_out,n);})};        //C++ 的 Lambda（匿名函数）写法
    CUDA_CHECK(cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost));
    //在 GPU 里直接比较（写一个 Check Kernel）是完全可行的，但在这段代码里，作者选择放回 CPU 比较，是基于“工程简洁性”和“实验纯净度”的双重考量。
    verify(h_out,n,0,stride,offset);

    // ---- (B) offset(misaligned) ----
    rows[1]={"offset=1 (misaligned)", bench([&]{ copy_offset<<<grid,block>>>(d_in,d_out,n,offset);})};
    CUDA_CHECK(cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost));
    verify(h_out,n,1,stride,offset);

    // ---- (C) strided ----
    rows[2]={"strided (scatter read)", bench([&]{ copy_strided<<<grid,block>>>(d_in,d_out,n,stride);})};
    CUDA_CHECK(cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost));
    verify(h_out,n,2,stride,offset);

    // ---- (D) float4 ----
    size_t n4=n/4;
    size_t grid4=(n4+block-1)/block;
    rows[3]={"coalesced + float4",bench([&]{copy_vec4<<<grid4,block>>>((const float4*)d_in,(float4*)d_out,n4);})};
    CUDA_CHECK(cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost));
    verify(h_out,n,3,stride,offset);

    // ---- (E) grid-stride loop：grid 固定为 SM 数 × 每 SM 若干 block ----
    // p.multiProcessorCount 代表 GPU 芯片上物理集成的“流式多处理器（SM，Streaming Multiprocessor）”的数量
    size_t gridP=(size_t)p.multiProcessorCount*8;           //目标就是让每个 SM 始终处于忙碌状态，从而“使用”到所有 SM 的计算资源。
    rows[4]={"grid-stride loop",bench([&]{copy_grid_stride<<<gridP,block>>>(d_in,d_out,n);})};
    CUDA_CHECK(cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost));
    verify(h_out,n,4,stride,offset);

    // ---------------- 结果表 ----------------
    // 有效带宽 = 「有用」字节数 / 时间 = (读 n*4 + 写 n*4) / t
    // 注意：跨步版硬件实际从 DRAM 搬的字节远多于此 —— 差值就是浪费，用 ncu 的
    //       dram__bytes_read.sum 才能看到真相
    //这是 ncu 提供的一个硬件性能指标，它统计的是 Kernel 执行期间，GPU 从显存（DRAM）实际读取的总字节数
    double gb=2.0*bytes/1e9;
    printf("\n%-26s %10s %14s %10s\n", "pattern", "time(ms)", "eff.BW(GB/s)", "slowdown");
    for (auto& r : rows)
        printf("%-26s %10.3f %14.0f %9.2fx\n", r.name, r.ms, gb / (r.ms / 1e3), r.ms / rows[0].ms);

    printf("\nCSV,pattern,ms,eff_gbs\n");            // 方便直接贴进 python 画图
    for (auto& r : rows) printf("CSV,%s,%.4f,%.1f\n", r.name, r.ms, gb / (r.ms / 1e3));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
}