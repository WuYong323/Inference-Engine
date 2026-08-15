#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// <cuda_runtime.h> 声明了调用的一切底层 CUDA 函数内存管理：cudaMalloc、cudaFree;数据搬运：cudaMemcpy;类型定义：cudaError_t（错误码类型）
// <c10/cuda/CUDAException.h> 专门用来帮助在 C++ 层优雅地捕获 CUDA 错误

// simple vector add CUDA kernel
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// kernel launcher (raw pointers only, no torch types -> no heavy torch headers here)
void launch_vector_add(const float* a, const float* b, float* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(a, b, c, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
