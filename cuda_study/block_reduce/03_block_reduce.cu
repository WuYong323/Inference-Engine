// ============================================================================
// block_reduce/03_block_reduce.cu
//
// 目的：用「把 n 个数加成 1 个数」这件最简单的事，把三件真东西钉死：
//        ① shared memory 怎么用（声明 / 生命周期 / 作用域）
//        ② __syncthreads() 为什么非有不可（漏了会怎样，亲眼看）
//        ③ 规约的正确性不能用 == 判，float 加法不满足结合律
//
// 本文件包含 5 个 kernel，其中 2 个是【故意写错的反面教材】：
//        (0) reduce_atomic_per_element  —— 反面教材A：能跑对但极慢 + 精度崩塌
//        (1) reduce_shared_serial       —— 今天的保底目标：shared + 1 次 __syncthreads
//        (2) reduce_shared_nosync       —— 反面教材B：故意删掉 __syncthreads
//        (3) reduce_shared_tree         —— 树形规约（Day4 的正主，今天先见一面）
//        (4) reduce_tree_atomic_block   —— 树形 + 每 block 一次 atomicAdd（工业常见写法）
//
// 环境要求：
//   H100 (sm_90) + CUDA 12.4；显存 ≥ 2 GB（默认 n = 1<<27 = 512 MB 输入）
//   rtx5060 (sm_120) + CUDA 12.4；显存 ≥ 2 GB（默认 n = 1<<27 = 512 MB 输入）
// 编译：
//   nvcc -O3 -arch=sm_120 -lineinfo -Xptxas -v -Xcompiler="/utf-8" -o 03_block_reduce 03_block_reduce.cu
//     -lineinfo : 让 ncu / racecheck 能把问题对回源码行
//     -Xptxas -v: 打印寄存器用量 + 每个 kernel 用了多少 shared memory
// 运行(linux)：
//   ./03_block_reduce            # 默认 n=1<<27，数据全 1.0f（能一眼看出精度问题）
//   ./03_block_reduce 24         # 自定义规模 n = 1<<24
//   ./03_block_reduce 27 rand    # 换成随机数据，看相对误差而不是绝对值
//   今天最重要的两条调试命令（比看结果对不对更能建立肌肉记忆）：
//   compute-sanitizer --tool racecheck  ./03_block_reduce   # 抓 shared memory 数据竞争
//   compute-sanitizer --tool synccheck  ./03_block_reduce   # 抓 divergent __syncthreads
//   compute-sanitizer --tool memcheck   ./03_block_reduce   # 抓越界（含 shared 越界）
// Profile：
//   ncu --set full -o 03_block_reduce_rep ./03_block_reduce
//   ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
//                 smsp__inst_executed_op_shared_ld.sum,\
//                 l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./03_block_reduce
// ============================================================================


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 0. 错误检查宏（沿用 Day2）：CUDA API 全部返回错误码，不查 = 错了也不知道
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                            \
    do{                                                             \
        cudaError_t err_=(call);                                    \
        if(err_!=cudaSuccess){                                      \
            fprintf(stderr,"[CUDA ERROR] %s:%d  %s\n",              \
                    __FILE__,__LINE__,cudaGetErrorString(err_));    \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    }while(0)

// 一个 block 多少线程。必须是 2 的幂，否则树形规约的折半会漏元素（见 kernel 3 注释）
#define BLOCK 256

// ===========================================================================
// (0) 反面教材 A：每个元素直接 atomicAdd 到全局同一个地址
//     它是对的 —— atomicAdd 保证读-改-写不被打断，所以不会丢更新。
//     但它有两个致命问题，今天要亲眼看到：
//       慢：几千万个线程排队改同一个地址，并行度被彻底摧毁
//       不准：加法顺序上等价于「串行累加」，float 会在 2^24 处彻底加不动
// ===========================================================================
__global__ void reduce_atomic_per_element(const float* __restrict__ in,float* out,size_t n){
    size_t idx=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(idx<n) atomicAdd(out,in[idx]);        // 全局唯一热点地址 = 硬件级排队
}

// ===========================================================================
// (1) shared memory + 一次 __syncthreads + thread0 串行合并
//
//     三段式结构（后面所有规约 kernel 都是这个骨架）：
//       段一 grid-stride 私有累加 → 每线程在【寄存器】里先攒一个局部和
//       段二 写进 shared memory  → 让同 block 的 256 个线程能互相看见
//       段三 __syncthreads 之后合并 → 由 thread0 把 256 个数加起来
//
//     为什么段一要先在寄存器里攒：
//        寄存器 ~1 周期，shared ~25 周期，global ~500 周期。
//        先把 n/线程数 个元素在最便宜的地方加完，shared 只碰一次。
//     为什么不用 if (idx < n) return：
//        规约的边界处理和 vector_add 完全不同！提前 return 的线程不会到达
//        __syncthreads()，整个 block 的 barrier 语义就崩了。
//        正确做法是用【单位元】0 初始化 sum，让所有线程都活到最后一起同步。
// ===========================================================================
__global__ void reduce_shared_serial(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    __shared__ float s[BLOCK];              // block 私有、block 生命周期，SM 上的 SRAM
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    // 段一：私有累加。相邻线程读相邻地址 → 完全合并访问
    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }

    // 段二：把私有结果暴露给同 block 的其他线程
    s[tid]=sum;

    // 段三： 同步 —— "都写完了，才开始加"
    __syncthreads();

    // 朴素合并：只有 thread0 干活，其余 255 个线程干等 → 并行度 1/256
    // 故意先写这个「笨版本」，之后升级成树形，才能量出差多少
    if(tid==0){
        float acc=0.0f;
        for(int i=0;i<BLOCK;++i) acc+=s[i];
        partial[blockIdx.x]=acc;          // 每个 block 输出一个数，第二趟再合并
    }
}

// ===========================================================================
// (2) 反面教材 B：和 (1) 一模一样，只删掉 __syncthreads()
//     关键认知：它【可能大部分时候算对】—— 这正是数据竞争最可怕的地方。
//     warp0 里的 thread0 常常跑在前面，读到 warp1..7 还没写的 shared（未初始化内存）。
//     所以：不要用「跑一遍对了」来验证同步，要用 racecheck 工具确定性地抓。
// ===========================================================================
__global__ void reduce_shared_nosync(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    __shared__ float s[BLOCK];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    s[tid]=sum;

    // __syncthreads();   ← 故意注释掉。compute-sanitizer --tool racecheck 会报
    //                       "Race reported between Write ... and Read ..."

    if(tid==0){
        float acc=0.0f;
        for(int i=0;i<BLOCK;++i){
            acc+=s[i];
        }
        partial[blockIdx.x]=acc;
    }
}

// ===========================================================================
// (3) 树形规约（sequential addressing 版）
//     每一轮把后一半加到前一半，活跃线程折半，log2(256)=8 轮结束。
//     为什么 __syncthreads() 写在 if 外面：
//        barrier 必须被 block 内【所有】线程一致到达。写进 if(tid<off) 里
//        = divergent barrier = 未定义行为（很可能挂死）。这是今天最贵的一条规矩。
//     为什么用 s[tid] += s[tid+off] 而不是教科书的 s[2*off*tid] += s[2*off*tid+off]：
//        前者（顺序寻址）活跃线程是连续的 tid，天然无 bank conflict；
//        后者（交错寻址）步长 2/4/8... 会踩到同一个 bank 上。
// ===========================================================================
__global__ void reduce_shared_tree(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    __shared__ float s[BLOCK];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    s[tid]=sum;
    __syncthreads();

    // BLOCK 必须是 2 的幂：否则 off 折半时会漏掉奇数部分
    for(unsigned off=BLOCK/2;off>0;off>>=1){
        if(tid<off) s[tid]+=s[tid+off];   // 只有前半段线程干活
        __syncthreads();                          // 每轮都要同步：这一轮读的是上一轮写的
    }
    if(tid==0) partial[blockIdx.x]=s[0];
}

// ===========================================================================
// (4) 树形 + 每 block 一次 atomicAdd —— 工业里最常见的「一趟搞定」写法
//     优点：省掉第二趟 kernel launch（本例只有 1056 次 atomic，不构成瓶颈）
//     代价： 结果不可复现。atomic 的到达顺序由硬件调度决定，每次跑加法顺序都不同，
//           float 不满足结合律 → 每次跑最后几位可能不一样。
//           这就是 PyTorch 里 torch.use_deterministic_algorithms(True) 要管的事。
// ===========================================================================
__global__ void reduce_tree_atomic_block(const float* __restrict__ in,float* out,size_t n){
    __shared__ float s[BLOCK];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    s[tid]=sum;
    __syncthreads();

    for(unsigned off=BLOCK/2;off>0;off>>=1){
        if(tid<off) s[tid]+=s[tid+off];
        __syncthreads();
    }
    if(tid==0) atomicAdd(out,s[0]);      // 每 block 只打一次，热点压力降到 1/256
}

// ---------------------------------------------------------------------------
// 附：动态 shared memory 版本（block 大小运行时才定的场景，比如按 hidden_dim 开）
//     声明成 extern 无长度数组，大小在 <<<grid, block, nbytes>>> 的第三个参数给。
//       陷阱：第三个参数是【字节数】不是元素个数；忘乘 sizeof(float) 是经典 bug。
//       超过 48 KB 必须先 cudaFuncSetAttribute 显式 opt-in。
// ---------------------------------------------------------------------------
__global__ void reduce_dynshared_tree(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    extern __shared__ float sdyn[];
    const unsigned tid=threadIdx.x;
    const unsigned bs=blockDim.x;
    const size_t gsize=(size_t)gridDim.x*bs;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)bs+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    sdyn[tid]=sum;
    __syncthreads();

    for(unsigned off=bs/2;off>0;off>>=1){
        if(tid<off) sdyn[tid]+=sdyn[tid+off];
        __syncthreads();
    }
    if(tid==0) partial[blockIdx.x]=sdyn[0];
}

// ===========================================================================
// 计时工具（预热 / 多次取平均 / 显式同步）
// ===========================================================================
struct Timer{
    cudaEvent_t s,e;
    Timer() {
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
    }
    ~Timer(){
        CUDA_CHECK(cudaEventDestroy(s));
        CUDA_CHECK(cudaEventDestroy(e));
    }
    void tic(){
        CUDA_CHECK(cudaEventRecord(s));
    }
    float toc(){
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms,s,e));
        return ms;
    }
};

template<typename F>
static float bench(F&& launch,int warmup,int iters){
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

int main(int argc,char** argv){
    // ---------------- 设备信息 ----------------
    int dev=0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p,dev));
    printf("GPU: %s | SM=%d | sharedMem/SM=%.0f KB | sharedMem/block(default)=%.0f KB\n",
           p.name, p.multiProcessorCount,
           p.sharedMemPerMultiprocessor / 1024.0, p.sharedMemPerBlock / 1024.0);

    int shift=(argc>1)?atoi(argv[1]):27;          // 默认 n = 1<<27 = 512 MB
    bool use_rand=(argc>2)&&(strcmp(argv[2],"rand")==0);
    size_t n=(size_t)1<<shift;
    size_t bytes=n*sizeof(float);
    printf("n = %zu floats (%.0f MB), data = %s\n\n",
           n, bytes / 1048576.0, use_rand ? "uniform random [0,1)" : "all 1.0f");

    // ---------------- 数据准备 + 双精度参考值 ----------------
    float* h_in=(float*)malloc(bytes);
    double ref_double=0.0;
    float ref_float_serial=0;
    srand(1234);
    for(size_t i=0;i<n;++i){
        h_in[i]=use_rand?(float)rand()/(float)RAND_MAX:1.0f;
        ref_double+=(double)h_in[i];
        ref_float_serial+=h_in[i];
    }
    printf("[CPU 参考] double 串行 = %.6f\n", ref_double);
    printf("[CPU 参考] float  串行 = %.6f   (相对误差 %.3e)\n",
           ref_float_serial, fabs(ref_float_serial - ref_double) / ref_double);
    if(!use_rand&&n>(1u<<24)){
        printf("           注意卡在 16777216 = 2^24 附近了：float 尾数只有 24 位，\n"
               "             当 acc 大到 2^24 时，acc + 1.0f 舍入回 acc —— 串行加法,加不动了。\n");
    }
    printf("\n");

    // ---------------- 显存分配 ----------------
    const size_t gridP=(size_t)p.multiProcessorCount*8;   // 每 SM 8 个 block，够铺满
    float* d_in=nullptr;
    float* d_partial=nullptr;
    float* d_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,bytes));
    CUDA_CHECK(cudaMalloc(&d_partial,gridP*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice));       // 唯一一次过 PCIe

    // 规约是纯 memory-bound：有用字节 = 读一遍输入。用它算有效带宽和 %peak
    const double gb=(double)bytes/1e9;
    const double peak_bw_gbs=384.0;        // H100 SXM HBM3 理论峰值，换卡请改  (H100 SXM HBM3 理论峰值为3350.0)

    struct Row{
        const char* name;
        float ms;
        double val;
        double relerr;
    };
    Row rows[5];
    float h_res=0.0f;
    auto grab=[&](float* dev_scalar){                 // 取回单个标量结果
        CUDA_CHECK(cudaMemcpy(&h_res,dev_scalar,sizeof(float),cudaMemcpyDeviceToHost));
        return (double)h_res;
    };
    auto relerr=[&](double v){
        return fabs(v-ref_double)/ref_double;
    };

    // 把 gridP 个 partial 合成 1 个（复用同一个 kernel，1 个 block 足够）
    auto pass2=[&](){
        reduce_shared_tree<<<1,BLOCK>>>(d_partial,d_out,gridP);
    };

    // ---- (0) atomic per element：慢，iters 调小防止等太久 ----
    {
        size_t gridA=(n+BLOCK-1)/BLOCK;
        auto run=[&]{
            CUDA_CHECK(cudaMemsetAsync(d_out,0,sizeof(float)));      // atomic 前必须清零
            reduce_atomic_per_element<<<gridA,BLOCK>>>(d_in,d_out,n);
        };
        rows[0]={"atomic per element (BAD)",bench(run,1,3),0,0};
        run();
        CUDA_CHECK(cudaDeviceSynchronize());
        rows[0].val=grab(d_out);
        rows[0].relerr=relerr(rows[0].val);
    }

    // ---- (1) shared + serial merge 保底 ----
    {
        auto run=[&]{
            reduce_shared_serial<<<gridP,BLOCK>>>(d_in,d_partial,n);
            pass2();
        };
        rows[1]={"shared + serial merge",bench(run,5,20),0,0};
        run();
        CUDA_CHECK(cudaDeviceSynchronize());
        rows[1].val=grab(d_out);
        rows[1].relerr=relerr(rows[1].val);
    }

    // ---- (2) 漏掉 __syncthreads 的版本：看错没错（可能对） ----
    {
        auto run=[&]{
            reduce_shared_nosync<<<gridP,BLOCK>>>(d_in,d_partial,n);
            pass2();
        };
        rows[2]={"NO __syncthreads (BAD)",bench(run,5,20),0,0};
        run();
        CUDA_CHECK(cudaDeviceSynchronize());
        rows[2].val=grab(d_out);
        rows[2].relerr=relerr(rows[2].val);
    }

    // ---- (3) 树形规约 + 确定性第二趟 ----
    {
        auto run=[&]{
            reduce_shared_tree<<<gridP,BLOCK>>>(d_in,d_partial,n);
            pass2();
        };
        rows[3]={"tree + 2nd pass (det.)",bench(run,5,20),0,0};
        run();
        CUDA_CHECK(cudaDeviceSynchronize());
        rows[3].val=grab(d_out);
        rows[3].relerr=relerr(rows[3].val);
    }

    // ---- (4) 树形 + 每 block 一次 atomicAdd（不可复现） ----
    {
        auto run=[&]{
            CUDA_CHECK(cudaMemsetAsync(d_out,0,sizeof(float)));
            reduce_tree_atomic_block<<<gridP,BLOCK>>>(d_in,d_out,n);
        };
        rows[4]={"tree + block atomic",bench(run,5,20),0,0};
        run();
        CUDA_CHECK(cudaDeviceSynchronize());
        rows[4].val=grab(d_out);
        rows[4].relerr=relerr(rows[4].val);
    }

    // ---------------- 结果表 ----------------
    printf("%-28s %10s %12s %10s %18s %12s\n",
           "kernel", "time(ms)", "eff.BW", "%peak", "result", "rel.err");
    for (auto& r : rows) {
        double bw = gb / (r.ms / 1e3);
        printf("%-28s %10.3f %10.0f GB/s %9.1f%% %18.4f %12.3e\n",
               r.name, r.ms, bw, 100.0 * bw / peak_bw_gbs, r.val, r.relerr);
    }

    // ---------------- 不可复现性实验：同一个 kernel 连跑 5 次，看结果变不变 ----------------
    printf("\n[determinism] tree+block atomic 连跑 5 次（观察最后几位是否抖动）:\n");
    for (int k = 0; k < 5; ++k) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        reduce_tree_atomic_block<<<gridP, BLOCK>>>(d_in, d_out, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("   run %d: %.8f\n", k, grab(d_out));
    }
    printf("[determinism] tree+2nd pass 连跑 5 次（应当每次完全一致）:\n");
    for (int k = 0; k < 5; ++k) {
        reduce_shared_tree<<<gridP, BLOCK>>>(d_in, d_partial, n);
        pass2();
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("   run %d: %.8f\n", k, grab(d_out));
    }

    // ---------------- 动态 shared memory 版本 sanity ----------------
    reduce_dynshared_tree<<<gridP,BLOCK,BLOCK*sizeof(float)>>>(d_in,d_partial,n);
    pass2();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n[dynamic shared] result = %.4f (rel.err %.3e)\n",
           grab(d_out), relerr(grab(d_out)));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    return 0;
}

// ============================================================================
// 附录：两个死锁样例（看懂即可）
//
// 样例一：divergent barrier —— barrier 写进了不是所有线程都会进的分支
//   __global__ void deadlock_v1(float* a) {
//       __shared__ float s[BLOCK];
//       int tid = threadIdx.x;
//       if (tid < 128) {          // 只有一半线程进来
//           s[tid] = a[tid];
//           __syncthreads();      // ← 另一半线程永远不会到达这个 barrier
//       }
//   }
//   官方原话（CUDA C++ Programming Guide）：__syncthreads() 允许出现在条件分支里，
//   但【条件必须在整个 block 内取值一致】，否则「很可能挂死或产生意外副作用」。
//
// 样例二：early return —— 边界检查用 return，等价于样例一
//   __global__ void deadlock_v2(const float* in, float* out, size_t n) {
//       __shared__ float s[BLOCK];
//       size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//       if (idx >= n) return;     // ← 尾块里越界的线程直接退出
//       s[threadIdx.x] = in[idx];
//       __syncthreads();          // ← 语义上已经不完整了
//       ...
//   }
// 正确写法就是本文件 kernel(1) 的做法：
//      用单位元占位（sum = 0.0f），让所有线程都走完全程，边界只影响「加了什么」，
//      不影响「谁到达 barrier」。规约里，边界处理 = 补单位元，不是 return。
// ============================================================================















