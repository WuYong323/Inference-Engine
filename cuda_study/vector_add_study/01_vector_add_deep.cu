// ============================================================================
// 01_vector_add_deep.cu   CUDA 线程模型 ·  kernel(深化版)
// 环境: NVIDIA GPU + CUDA 12.x
// 编译: nvcc -O2 -arch=sm_120 -Xcompiler="/utf-8" 01_vector_add_deep.cu -o 01_vector_add_deep    (H100=sm_90)
// 运行: (Linux) ./01_vector_add_deep            (Windows) .\01_vector_add_deep.exe 或 01_vector_add_deep
// 调试: compute-sanitizer ./01_vector_add_deep
// ============================================================================

// nvcc -O3 -arch=sm_90 -Xptxas -v -c 02_memory_access.cu
// 日常开发与调试：建议使用 -O0（默认）或 -O1  性能测试与生产部署：-O2 通常是最佳且最安全的选择    追求极致性能：如果对性能有极致要求，并且已经过充分测试，可以尝试 -O3



#include <cstdio>       //标准输入输出库
#include <cstdlib>      //C 标准通用工具库
#include <cmath>
#include <cuda_runtime.h>

// ---- 工业标配: CUDA 错误检查宏----
// 必须:CUDA 的错误默认"沉默",不查返回码,越界了还以为跑对了。
#define CUDA_CHECK(call) do{                                \
    cudaError_t err=(call);                                 \
    if(err!=cudaSuccess){                                   \
        fprintf(stderr,"CUDA error %s:%d: %s\n",            \
                __FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(1);                                            \
    }                                                       \
}while(0)



// ---- 版本 A: 一线程一元素----
__global__ void vector_add(const float* a,const float* b,float* c,int n){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n)
        c[idx]=a[idx]+b[idx];
}


// ---- 教学版: 让前 8 个线程打印, 看见分工(仅调试用)----
__global__ void vector_add_verbose(const float* a,const float* b,float* c,int n){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<8){
        printf(" block=%2d thread=%3d -> 全局 idx=%d, 负责 c[%d]=a[%d]+b[%d]\n",
        blockIdx.x,threadIdx.x,idx,idx,idx,idx);
    }
    if(idx<n){
        c[idx]=a[idx]+b[idx];
    }
}


// ---- 版本 B: grid-stride loop(工业深化版,库里真实这么写)----
// 思想: 不追求一线程一元素,而是开固定数量的线程,每个线程跨步处理多个元素。
// stride = 整个 grid 的线程总数,保证每轮相邻线程仍访问相邻地址(访存友好)。
__global__ void vector_add_gridstride(const float* a,const float* b,float* c,int n){
    int stride=gridDim.x*blockDim.x;                // 整个 grid 一共有多少线程
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=stride) c[i]=a[i]+b[i];
}

// ---- CPU 参考实现(可信基线,用来验证 GPU 算得对不对)----
void cpu_reference(const float*a ,const float* b,float* c,int n){
    for(int i=0;i<n;++i) c[i]=a[i]+b[i];
}


int main(){
    // ============ 第 0 步: 先用小 n 跑 verbose 版, "看见"线程分工 ============
    {
        int n_small=1000;
        size_t bts=n_small*sizeof(float);
        float *ha=(float*)malloc(bts),*hb=(float*)malloc(bts),*hc=(float*)malloc(bts);
        for(int i=0;i<n_small;++i) {ha[i]=(float)i;hb[i]=2.0f*i;}
        float *da,*db,*dc;
        CUDA_CHECK(cudaMalloc(&da,bts));
        CUDA_CHECK(cudaMalloc(&db,bts));
        CUDA_CHECK(cudaMalloc(&dc,bts));
        CUDA_CHECK(cudaMemcpy(da,ha,bts,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db,hb,bts,cudaMemcpyHostToDevice));

        int block=256;
        int grid=(n_small+block-1)/block;
        printf("=== 教学演示: n=%d, block=%d, grid=%d(共 %d 线程, 多派 %d 个)===\n",
               n_small, block, grid, grid*block, grid*block - n_small);
        vector_add_verbose<<<grid,block>>>(da,db,dc,n_small);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());        // 等 GPU 打印完再往下走
        printf("  ...(只打印前 8 个线程)...\n");

        cudaFree(da);
        cudaFree(db);
        cudaFree(dc);
        free(ha);
        free(hb);
        free(hc);
    }

    // ============ 下面是正式跑大 n、验证正确性、计时 ============
    int n=1<<24;
    size_t bytes=n*sizeof(float);

    // ①: host 准备数据(h_ 前缀 = host,工业约定)
    float *h_a=(float*)malloc(bytes);
    float *h_b=(float*)malloc(bytes);
    float *h_c=(float*)malloc(bytes);
    float *h_ref=(float*)malloc(bytes);     // CPU 参考结果
    for(int i=0;i<n;++i){h_a[i]=1.0f*i;h_b[i]=2.0f*i;}

    // ②: device 开显存(d_ 前缀 = device)
    float *d_a,*d_b,*d_c;
    CUDA_CHECK(cudaMalloc(&d_a,bytes));
    CUDA_CHECK(cudaMalloc(&d_b,bytes));
    CUDA_CHECK(cudaMalloc(&d_c,bytes));

    // ③: 输入 CPU→GPU(H2D)
    CUDA_CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

    // ④: 配启动参数并发射(两个版本各发一次)
    int blocksize=256;                              // 常用值, 32(warp)的倍数
    int gridsize=(n+blocksize-1)/blocksize;        // 向上取整, 覆盖所有元素

    // 计时工具: cudaEvent(GPU 侧的秒表)
    cudaEvent_t t0,t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    int iters=50;
    float ms;

    // ---- 跑版本 A: 一线程一元素 ----
    for(int i=0;i<5;++i){
         vector_add<<<gridsize,blocksize>>>(d_a,d_b,d_c,n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t0));
    for(int i=0;i<iters;++i){
        vector_add<<<gridsize,blocksize>>>(d_a,d_b,d_c,n);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));
    float ms_A=ms/iters;

    // ---- 跑版本 B: grid-stride(开固定数量的块, 不再随 n 膨胀)----
    int numSMs=0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs,cudaDevAttrMultiProcessorCount,0));    //查询指定 GPU 设备的某个硬件属性，并将结果存入你提供的变量中。
    int gridB=numSMs*8;             // 经验值: 每个 SM 派 8 块, 把机器填满
    for(int i=0;i<5;++i){
        vector_add_gridstride<<<gridB,blocksize>>>(d_a,d_b,d_c,n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t0));
    for(int i=0;i<iters;++i){
        vector_add_gridstride<<<gridB,blocksize>>>(d_a,d_b,d_c,n);
    }
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));
    float ms_B=ms/iters;

    // ⑤: 结果 GPU→CPU(D2H), 才能读出来验证
    CUDA_CHECK(cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost));
    cpu_reference(h_a,h_b,h_ref,n);

    // ---- 正确性: 和 CPU 基线比, 允许浮点微小误差(同 PyTorch allclose )----
    double max_err=0.0;
    for(int i=0;i<n;++i){
        double e=fabs((double)h_c[i]-(double)h_ref[i]);
        if(e>max_err) max_err=e;
    }
    printf("\n=== 正确性 ===\n");
    printf("c[%d] = %.1f (期望 %.1f)\n", n-1, h_c[n-1], 3.0*(n-1));
    printf("最大逐元素误差 = %.3e  %s\n", max_err, max_err < 1e-4?"通过":"不通过");

    // ---- 性能: 算有效带宽(读 a + 读 b + 写 c = 3 次 n 个 float 的搬运)----
    double moved=3.0*n*sizeof(float);                // 总搬运字节数
    printf("\n=== 性能(memory-bound, 看带宽而非算力)===\n");
    printf("%-22s %10s %16s\n", "版本", "耗时(ms)", "有效带宽(GB/s)");
    printf("A 一线程一元素  %14.4f %16.1f\n", ms_A, moved / (ms_A*1e-3) / 1e9);
    printf("B grid-stride   %14.4f %16.1f\n", ms_B, moved / (ms_B*1e-3) / 1e9);
    printf("(H100 HBM 约 3.35 TB/s; 本卡跑到几百~上千 GB/s 即正常, 越接近峰值越好)\n");

    // 收尾: 显存/内存各自释放, 否则泄漏
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_ref);
    return 0;
}














