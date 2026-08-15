#include<cstdio>
#include<cstdlib>

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
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);
    
}