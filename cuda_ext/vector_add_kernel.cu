#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

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
