#include<cstdio>
#include<cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do{\
    cudaError_t err=(call);\
    if(err!=cudaSuccess){\
        fprintf(stderr,"CUDA err %s:%d: %s\n",\
                __FILE__,__LINE__,cudaGetErrorString(err));\
        exit(1);\
    }\
}while(0)

__global__ void vector_add(const float* a,const float* b,float* c,int n){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<n){
        c[idx]=a[idx]+b[idx];
    }
}

int main(){
    int n=1000;
    size_t bytes=n*sizeof(float);

    // 步骤① 在 CPU(host) 上准备数据。h_ 前缀 = host, 是工业界约定俗成的命名
    float *h_a=(float*)malloc(bytes);
    float *h_b=(float*)malloc(bytes);
    float *h_c=(float*)malloc(bytes);
    for(int i=0;i<n;++i){h_a[i]=i;h_b[i]=2*i;}

    // 步骤② 在 GPU(device) 上开显存。d_ 前缀 = device
    float *d_a,*d_b,*d_c;
    CUDA_CHECK(cudaMalloc(&d_a,bytes));
    CUDA_CHECK(cudaMalloc(&d_b,bytes));
    CUDA_CHECK(cudaMalloc(&d_c,bytes));

    // 步骤③ 把输入从 CPU 搬到 GPU (HostToDevice)
    CUDA_CHECK(cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice));

    // 步骤④ 配置启动参数并发射 kernel
    int blockSize=256;          // 每 block 256 线程, 常用值(32的倍数)
    int gridSize=(n+blockSize-1)/blockSize;     // 向上取整, 见下方深挖
    vector_add<<<gridSize,blockSize>>>(d_a,d_b,d_c,n);

    // kernel 启动不返回错误码, 要专门查两类错误:
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 步骤⑤ 把结果从 GPU 搬回 CPU (DeviceToHost) 才能读
    CUDA_CHECK(cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost));
    // cudaMemcpy 自带同步语义, 会等 kernel 跑完再拷, 所以这里不必额外 synchronize

    printf("c[999] = %.1f (期望 2997)\n", h_c[999]);  // 999 + 2*999 = 2997

    // 收尾: 显存和内存都要各自释放, 否则泄漏
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}













