// ============================================================================
// 04_tree_reduction.cu   树形归约 + bank conflict + warp shuffle
//
// 四个问题，每个问题对应一组实验：
//   Q1 把 256 个数加成 1 个，从「1 个线程串 256 步」升级到「8 步树形」，到底快多少?
//      > 实验 A（厚归约，每线程处理很多元素）+ 实验 B（薄归约，每线程只处理 1 个元素）
//      > 两个实验的结论完全不同，优化的收益取决于占多大比例
//   Q2 教科书上的「交错寻址」为什么慢？bank conflict 到底能不能被量出来?
//      > 实验 C（bank_probe：把 bank conflict 从访存噪声里隔离出来）
//   Q3 warp 内的最后 5 轮，能不能连 shared memory 都不用？
//      > kernel (E)(F)：__shfl_down_sync 两级归约（工业标准结构）
//   Q4 学的东西怎么变成 CUDA RMSNorm ?
//      > 实验 D（rmsnorm 三个版本，行归约形状）
//
// 文件的 kernel 清单（(B)(C) 是【反面教材】，故意保留）：
//   (A) reduce_serial_merge     基线：shared + thread0 串行合并
//   (B) reduce_interleave_div   NVIDIA reduction.pdf 的 reduce#1：交错寻址 + warp 分化
//   (C) reduce_interleave_conf  reduce#2：修掉分化，但踩满 bank conflict
//   (D) reduce_tree_seq         reduce#3：顺序寻址树形归约 ← 正主
//   (E) reduce_tree_unroll      顺序寻址 + 最后一个 warp 用 shuffle 收尾（模板全展开）
//   (F) reduce_warp_2stage      纯 warp shuffle 两级归约 ← 工业里最常见的写法
//   (G) reduce_cub              CUB 官方实现：你手写是为了懂，上线用这个
//
// 环境要求：CUDA 12.4；显存 ≥ 2 GB（默认输入 n = 1<<27 = 512 MB）
// 编译：
//   nvcc -O3 -arch=sm_120 -lineinfo -Xptxas -v -Xcompiler="/utf-8 /Zc:preprocessor /std:c++17" -o 04_tree_reduction 04_tree_reduction.cu
//     -lineinfo  : ncu / compute-sanitizer 能把结果对回源码行
//     -Xptxas -v : 打印每个 kernel 的寄存器用量 + shared memory 用量
//     -Xcompiler="/utf-8 /Zc:preprocessor /std:c++17" :
//          /utf-8   源文件和执行字符集都采用 UTF-8 编码。
//          /Zc:preprocessor   强制 MSVC 使用 符合 C++ 标准的现代预处理器
//          /std:c++17   让 MSVC 在编译主机代码时采用 C++17 标准
// 运行：
//   ./04_tree_reduction              # 默认 n=1<<27，全 1.0f 数据
//   ./04_tree_reduction 24 rand      # n=1<<24，随机数据（看相对误差而不是整数值）
//
//  三条 profiling 命令（比时间更能建立认知）：
//   # 1) 看 bank conflict：对比 (C) 和 (D)，这个计数器会差几个数量级
//   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
//                l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./04_tree_reduction 24
//   # 2) 看 warp 分化：对比 (B) 和 (D)，「每条指令平均有几个活跃线程」（满分 32）
//   ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./04_tree_reduction 24
//   # 3) 看 shared memory 访问指令数：对比 (D) 和 (F)，shuffle 版应当断崖式下降
//   ncu --metrics smsp__inst_executed_op_shared_ld.sum,\
//                smsp__inst_executed_op_shared_st.sum ./04_tree_reduction 24
//   # 附：正确性护栏（Day3 养成的肌肉记忆，改归约代码后必跑）
//   compute-sanitizer --tool racecheck  ./04_tree_reduction 20
//   compute-sanitizer --tool synccheck  ./04_tree_reduction 20
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>         // CUB 随 CUDA Toolkit 分发，不需要额外装

// ---------------------------------------------------------------------------
// 错误检查宏（沿用 Day2/Day3）：CUDA API 全都靠返回值报错，不查 = 错了也不知道
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err_=(call);                                    \
        if(err_!=cudaSuccess){                                      \
            fprintf(stderr,"[CUDA ERROR] %s:%d   %s\n",             \
                    __FILE__,__LINE__,cudaGetErrorString(err_));    \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while(0);

// 一个 block 多少线程。必须是 2 的幂：树形折半靠它才不会漏元素
#define BLOCK 256

// ===========================================================================
// 0. warp 级归约：
//    __shfl_down_sync(mask, v, off) 的语义：
//      「把 lane (myLane + off) 手里的 v 值，直接搬到我的寄存器里」
//    它不经过任何内存 —— 不是 global，也不是 shared，是 warp 内的寄存器直连通路
//    （PTX 里就是一条 shfl.sync.down.b32）。
//
//    > 为什么 off 从 16 开始而不是 32：
//       一个 warp 只有 32 个 lane，off=16 就已经是后半个 warp 加到前半个 warp。
//       循环 16→8→4→2→1 共 5 步 = log2(32)，结束后总和在 lane 0 手里。
//    > mask 为什么是 0xffffffff：
//       它是「哪些 lane 参与这次交换」的位图，0xffffffff = 全部 32 个。
//       Volta 之后每个线程有独立的程序计数器（independent thread scheduling），
//       编译器无法自己推断谁会到场，所以必须由你显式声明 —— 这就是 CUDA 9 给所有
//       warp 原语加 `_sync` 后缀的原因。
//    > 陷阱：如果调用点在一个 warp 内部分化的分支里（比如 if (tid < 20)），
//       mask 写 0xffffffff 未定义行为。本文件所有调用点都保证
//       「整个 warp 要么全进要么全不进」。
// ===========================================================================
__device__ __forceinline__ float warp_reduce_sum(float v){
#pragma unroll                  //编译器指令，让 CUDA 编译器完全展开紧随其后的 for 循环。
    for(int off=16;off>0;off>>=1){
        v+=__shfl_down_sync(0xffffffff,v,off);
    }
    return v;                   // 有效值只在 lane 0, 其余 lane 拿到的是部分和
}

// ---------------------------------------------------------------------------
// 两级 block 归约：工业里最标准的结构（vLLM / FasterTransformer / CUB 都是这个形状）
//   第一级：每个 warp 内部用 shuffle 归约   → 32 个数变 1 个，0 次 shared 访问
//   第二级：8 个 warp 的结果写进一个 8 元素的 shared 数组，再由 warp 0 归约一次
//   收益：shared memory 访问量从「BLOCK 次写 + BLOCK×2 次读」降到「8 次写 + 32 次读」，
//   __syncthreads() 从 log2(BLOCK)=8 次降到 1 次。
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_sum(float v){
    __shared__ float warp_sums[32];              // 32 是上限：一个 block 最多 1024 线程 = 32 warp
    const unsigned lane=threadIdx.x&31;         // 在 warp 里的编号 (0..31)，等价 %32
    const unsigned wid=threadIdx.x>>5;          // 第几个 warp，等价 /32

    v=warp_reduce_sum(v);                       // 第一级
    if(lane==0) warp_sums[wid]=v;               // 每个 warp 的第一个
    __syncthreads();                            // 全文件唯一一次 barrier

    const unsigned nwarps=blockDim.x>>5;
    //   这一句必须让【整个 warp 0】都执行到（而不是 if (tid < nwarps)），
    //   否则下面的 warp_reduce_sum 里 mask=0xffffffff 就未定义了。
    //   多余的 lane 用加法单位元 0 占位。
    v=(threadIdx.x<nwarps) ? warp_sums[lane]:0.0f;
    if(wid==0) v=warp_reduce_sum(v);         // 第二级：分支对整个 warp 一致，安全
    return v;                                // 有效值只在 thread 0
}

// ===========================================================================
// (A) 基线：shared + thread0 串行合并
//     为了作为分母 —— 所有加速都要和它比
// ===========================================================================
__global__ void reduce_serial_merge(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    __shared__ float s[BLOCK];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    // 段一：grid-stride 私有累加（相邻线程读相邻地址 → 合并访问）
    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }

    s[tid]=sum;
    __syncthreads();

    if(tid==0){                            // 并行度 1/256，255 个线程干等
        float acc=0.0f;
        for(int i=0;i<BLOCK;++i) acc+=s[i];
        partial[blockIdx.x]=acc;
    }
}

// ===========================================================================
// (B) 反面教材一：交错寻址 + warp 分化（NVIDIA reduction.pdf 的 reduce#1）
//     这是最多教科书用来「介绍树形归约」的版本，但有两个错误：
//       1) warp 分化：活跃线程是 tid=0,2,4,... 隔一个活一个，
//           同一个 warp 里一半干活一半空转 → 有效并行度直接砍半
//       2) 取模慢：整数取模在 GPU 上要展开成十几条指令（无硬件除法器）
//       注意它【没有】bank conflict —— 活跃线程访问的还是 0,2,4..30 这些不同 bank。
//       很多人把 (B) 的慢归因于 bank conflict，那是错的。(B) 慢在分化，(C) 才慢在 bank。
// ===========================================================================
__global__ void reduce_interleave_div(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    __shared__ float s[BLOCK];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    s[tid]=sum;
    __syncthreads();

    for(unsigned off=1;off<blockDim.x;off<<=1){
        if(tid%(2*off)==0){
            s[tid]+=s[tid+off];
        }
        __syncthreads();
    }
    if(tid==0) partial[blockIdx.x]=s[0];
}

// ===========================================================================
// (C) 反面教材二：修掉分化，但踩满 bank conflict（reduce#2）
//     改法：让活跃线程变成连续的 tid=0..127，用 index = 2*off*tid 去访问。
//     分化没了，但访问地址的步长变成 2,4,8,16... → 同一个 warp 的线程挤在同一个 bank。
//     手算第一轮（off=1）：warp0 的 tid=0..31 访问 index=0,2,...,62，
//     bank = (index) % 32 = 0,2,...,30,0,2,...,30 → 每个 bank 被撞 2 次 = 2-way conflict。
//     off=2 → 4-way，off=4 → 8-way，off=8 → 16-way……
// ===========================================================================
__global__ void reduce_interleave_conf(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    __shared__ float s[BLOCK];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    s[tid]=sum;
    __syncthreads();

    for(unsigned off=1;off<blockDim.x;off<<=1){
        unsigned index=2*off*tid;
        if(index<blockDim.x){
            s[index]+=s[index+off];
        }
        __syncthreads();
    }
    if(tid==0) partial[blockIdx.x]=s[0];
}

// ===========================================================================
// (D)   正主：顺序寻址树形归约（reduce#3）
//     一行之差，两个错误一起治好：
//       治分化：活跃线程永远是 tid < off 的【连续一段】→ 整 warp 要么全活要么全死
//       治 bank：warp 内 32 个线程访问 s[tid] 和 s[tid+off]，都是连续 32 个地址
//                → 落在 32 个不同 bank 上，天然零冲突
//       这就是「为什么 stride 从大到小」的完整答案 —— 不只是习惯，是同时躲掉两个坑。
// ===========================================================================
__global__ void reduce_tree_seq(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    __shared__ float s[BLOCK];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*blockDim.x;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    s[tid]=sum;
    __syncthreads();

    // log2(256) = 8 轮。每轮活跃线程折半，但 barrier 是全 block 的（下面 (E) 会优化掉一半）
    for(unsigned off=blockDim.x>>1;off>0;off>>=1){
        if(tid<off) s[tid]+=s[tid+off];
        __syncthreads();                       //  这一轮读的是上一轮写的，必须同步
    }
    if(tid==0) partial[blockIdx.x]=s[0];
}

// ===========================================================================
// (E) 顺序寻址 + 最后一个 warp 用 shuffle 收尾（模板参数让循环编译期全展开）
//     两个优化点：
//       ① off <= 32 之后，只剩 warp 0 在干活 —— 全 block barrier 变成纯浪费，
//          改成 warp 内 shuffle，省掉 5 次 __syncthreads() + 5 轮 shared 读写。
//       ② BS 是模板参数（编译期常量）→ 折半循环被完全展开，省掉循环计数和分支。
//          Day3 的 `blockDim.x` 是运行期变量，编译器展不开。这是模板在 kernel 里
//          最常见、最实用的用途（CUB / CUTLASS 全靠这个）。
//   注意：这里【绝对不能】沿用 2017 年前教程里的 `volatile float* vs = s;` 写法。
//   那套写法依赖「warp 内 32 线程锁步」的旧假设，Volta 起独立线程调度已经打破它。
//   现代唯一正确解法就是显式 shuffle（或 __syncwarp()）。
// ===========================================================================
template <unsigned BS>
__global__ void reduce_tree_unroll(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    static_assert(BS>=64 && (BS&(BS-1))==0,"BS 必须是 >=64 的 2 的幂");
    __shared__ float s[BS];
    const unsigned tid=threadIdx.x;
    const size_t gsize=(size_t)gridDim.x*BS;

    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)BS+tid;i<n;i+=gsize){
        sum+=in[i];
    }
    s[tid]=sum;
    __syncthreads();

    // 只折到 off=32 为止（编译期已知次数，会被完全展开）
#pragma unroll
    for(unsigned off=BS>>1;off>32;off>>=1){
        if(tid<off) s[tid]+=s[tid+off];
        __syncthreads();
    }

    // 收尾：warp 0 独立完成最后 6 步（1 次 shared 读 + 5 次 shuffle）
    // 分支条件 tid < 32 对 warp 0 是整体成立、对别的 warp 是整体不成立 → 不是分化
    if(tid<32){
        float v=s[tid]+s[tid+32];
        v=warp_reduce_sum(v);
        if(tid==0) partial[blockIdx.x]=v;
    }
}

// ===========================================================================
// (F)   工业标准写法：纯 warp shuffle 两级归约，shared 只用 32 个 float
//     和 (D) 相比：shared 流量降两个数量级，barrier 从 8 次降到 1 次。
//     和 (E) 相比：连「把 BLOCK 个数写进 shared」这一步都省了（直接在寄存器里归约）。
//       这就是 vLLM `blockReduceSum` / FasterTransformer / Triton 编译产物的形状。
// ===========================================================================
__global__ void reduce_warp_2stage(const float* __restrict__ in, float* __restrict__ partial,size_t n){
    const size_t gsize=(size_t)gridDim.x*blockDim.x;
    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=gsize){
        sum+=in[i];
    }

    sum=block_reduce_sum(sum);               // 全部归约逻辑在这一行里
    if(threadIdx.x==0) partial[blockIdx.x]=sum;
}

// ===========================================================================
// (G) CUB：官方模板库的实现。手写是为了懂，上线用这个。
//     CUB 会按 arch 自动挑算法（raking / warp shuffle）、自动处理 bank 布局，
//     并且它的 TempStorage 大小是编译期算出来的最小值。
// ===========================================================================
__global__ void reduce_cub(const float* __restrict__ in,float* __restrict__ partial,size_t n){
    using BlockReduce=cub::BlockReduce<float,BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;         // 大小由 CUB 决定，别自己猜

    const size_t gsize=(size_t)gridDim.x*blockDim.x;
    float sum=0.0f;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=gsize){
        sum+=in[i];
    }

    float total=BlockReduce(temp).Sum(sum);                 // 结果只在 thread 0 有效
    if(threadIdx.x==0) partial[blockIdx.x]=total;
}

// ===========================================================================
// 实验 C 用：bank_probe —— 把 bank conflict 从访存噪声里隔离出来
//
// 为什么需要它：真实的全数组归约是 memory-bound（时间全花在读 HBM 上），
// bank conflict 藏在小数点后面，用秒表根本量不出来。要看清一个机制，
// 就得设计一个【只有这个机制在起作用】的实验 —— 这是做性能实验的基本功。
//
// 这个 kernel 里没有 global 访存（数据全在 shared），只有一个纯 shared 读循环：
//   STRIDE=1  → warp 内 32 线程读连续地址 → 32 个不同 bank → 无冲突（理想）
//   STRIDE=32 → 地址间隔 32 个 float = 128 B → 全部落在同一个 bank → 32-way 冲突
//   STRIDE=0  → 32 线程读【同一个】地址 → 触发广播（broadcast）→ 也是无冲突！
//                  这个反直觉结论很重要：bank conflict 的定义是
//                 「同一 bank 的【不同】地址」，同地址是硬件专门优化过的。
// ===========================================================================
template <int STRIDE>
__global__ void bank_probe(float* __restrict__ out,int iters) {
    __shared__ float s[1024];                      // 1024 float = 4 KB，跨 32 个 bank 各 32 行
    const int tid=threadIdx.x;

    for(int i=tid;i<1024;i+=blockDim.x){
        s[i]=(float)i;
    }
    __syncthreads();

    float acc=0.0f;
    for(int it=0;it<iters;++it){
        // & 1023 代替 % 1024：2 的幂取模用位运算，编译器其实也会帮你做，写清楚更好读
        int idx=(tid*STRIDE+it)&1023;
        acc+=s[idx];
    }
    // 必须把结果写出去，否则整个循环会被编译器判定为死代码删掉
    // （这是写 micro-benchmark 最经典的翻车点：测出来快得离谱，其实 kernel 是空的）
    out[blockIdx.x*blockDim.x+tid]=acc;
}

// ===========================================================================
// 实验 D 用：RMSNorm 的行归约三版本
//
// RMSNorm 的定义：y = x / sqrt(mean(x?) + eps) * w
// 拆成三步：① 对一行的 H 个元素求平方和（归约）
//         ② 由平方和算出缩放系数，广播给全行所有线程
//         ③ 每个元素乘缩放系数和权重
// 排布：一个 block 负责一行（一个 token 的 hidden 向量），grid = 行数
//   为什么这个形状里「树形 vs 串行」的差距远大于全数组归约：
//      全数组归约里每个线程要读几万个元素，归约只占 1%；
//      这里每个线程只读 H/BLOCK = 16 个元素，归约占了相当大的比例。
// ===========================================================================

// 版本 0：串行合并
__global__ void rmsnorm_serial(const float* __restrict__ x,const float* __restrict__ w,float* __restrict__ y,int H,float eps){
    __shared__ float s[BLOCK];
    const int tid=threadIdx.x;
    const float* xr=x+(size_t)blockIdx.x*H;
    float * yr=y+(size_t)blockIdx.x*H;

    float acc=0.0f;
    for(int i=tid;i<H;i+=BLOCK){
        float v=xr[i];
        acc+=v*v;
    }
    s[tid]=acc;
    __syncthreads();

    if(tid==0){
        float t=0.0f;
        for(int i=0;i<BLOCK;++i){
            t+=s[i];
            s[0]=t;
        }
    }
    __syncthreads();                           // thread0 写完 s[0]，全体才能读

    const float scale=rsqrtf(s[0]/H+eps);     // 32 个 lane 读同一个 s[0] → 广播，不是冲突
    for(int i=tid;i<H;i+=BLOCK){
        yr[i]=xr[i]*scale*w[i];
    }
}

// 版本 1：树形归约
__global__ void rmsnorm_tree(const float* __restrict__ x,const float* __restrict__ w,float* __restrict__ y,int H,float eps){
    __shared__ float s[BLOCK];
    const int tid=threadIdx.x;
    const float* xr=x+(size_t)blockIdx.x*H;
    float* yr=y+(size_t)blockIdx.x*H;

    float acc=0.0f;
    for(int i=tid;i<H;i+=BLOCK){
        float v=xr[i];
        acc+=v*v;
    }
    s[tid]=acc;
    __syncthreads();

#pragma unroll
    for(unsigned off=BLOCK>>1;off>0;off>>=1){
        if(tid<(int)off) s[tid]+=s[tid+off];
        __syncthreads();
    }
    // 循环最后一轮的 __syncthreads() 已经保证 s[0] 对全体可见，这里不用再同步
    const float scale=rsqrtf(s[0]/H+eps);
    for(int i=tid;i<H;i+=BLOCK){
        yr[i]=xr[i]*scale*w[i];
    }
}

// 版本 2：warp shuffle 两级归约（W2 要接进引擎的目标形态）
__global__ void rmsnorm_shfl(const float* __restrict__ x,const float* __restrict__ w,float* __restrict__ y,int H,float eps){
    __shared__ float s_scale;                   // 只需要 1 个 float 做广播
    const int tid=threadIdx.x;
    const float* xr=x+(size_t)blockIdx.x*H;
    float* yr=y+(size_t)blockIdx.x*H;

    float acc=0.0f;
    for(int i=tid;i<H;i+=BLOCK){
        float v=xr[i];
        acc+=v*v;
    }

    acc=block_reduce_sum(acc);                // 有效值只在 thread 0
    if(tid==0) s_scale=rsqrtf(acc/H+eps);
    __syncthreads();                            // 广播必须的那一次同步

    const float scale=s_scale;
    // 注意：这里第二次读 xr[i]，靠 L1/L2 命中（一行 4096 float = 16 KB，通常还在 cache）。
    // W2 的一个真实设计决策：H 小时可以把 x 暂存在寄存器/shared 里避免二次读，
    // H 大时寄存器不够只能重读 —— 这个权衡到时候要实测，别拍脑袋。
    for(int i=tid;i<H;i+=BLOCK){
        yr[i]=xr[i]*scale*w[i];
    }
}

// ===========================================================================
// 计时工具（三铁律：预热 / 多次取平均 / 显式同步）
// ===========================================================================
struct Timer{
    cudaEvent_t s,e;
    Timer(){
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

template <typename F>
static float bench(F&& launch,int warmup,int iters){
    for(int i=0;i<warmup;++i){
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    Timer t;
    t.tic();
    for(int i=0;i<iters;++i){
        launch();
    }
    float ms=t.toc()/iters;
    CUDA_CHECK(cudaGetLastError());
    return ms;
}


// ===========================================================================
int main(int argc,char** argv){
    int dev=0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p,dev));
    int clock_khz=0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz,cudaDevAttrClockRate,dev));
    printf("GPU: %s | SM=%d | shared/SM=%.0f KB | clock=%.2f GHz\n",
           p.name, p.multiProcessorCount,
           p.sharedMemPerMultiprocessor / 1024.0, clock_khz / 1e6);

    const int shift=(argc>1)?atoi(argv[1]):27;
    const bool use_rand=(argc>2) && (strcmp(argv[2],"rand")==0);
    const size_t n=(size_t)1<<shift;
    const size_t bytes=n*sizeof(float);
    printf("n = %zu floats (%.1f MB), data = %s\n\n",
           n, bytes / 1048576.0, use_rand ? "uniform random [0,1)" : "all 1.0f");

    // ---------------- 数据 + double 参考真值（参考值必须更高精度）----------
    float* h_in=(float*)malloc(bytes);
    double ref=0.0;
    srand(1234);
    for(size_t i=0;i<n;++i){
        h_in[i]=use_rand?(float)rand()/(float)RAND_MAX:1.0f;
        ref+=(double)h_in[i];
    }
    printf("[参考真值] double 串行 = %.6f\n\n", ref);

    const size_t gridP=(size_t)p.multiProcessorCount*8;   // 厚归约：每 SM 8 个 block
    const size_t gridT=n/BLOCK;
    const size_t maxGrid=(gridP>gridT)?gridP:gridT;

    float* d_in=nullptr;
    float* d_partial=nullptr;
    float* d_out=nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,bytes));
    CUDA_CHECK(cudaMalloc(&d_partial,maxGrid*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice));

    const double gb=(double)bytes/1e9;
    const double peak_bw=384.0;               // RTX 5060 Laptop 理论峰值 384 GB/s; H100 SXM HBM3 理论峰值 3350GB/s，换卡请改

    // 第二趟合并：把 grid 个 partial 加成 1 个（1 个 block 足够，固定顺序 → 可复现）
    auto pass2=[&](size_t g){
        reduce_warp_2stage<<<1,BLOCK>>>(d_partial,d_out,g);
    };
    float h_res=0.0f;
    auto grab=[&](){
        CUDA_CHECK(cudaMemcpy(&h_res,d_out,sizeof(float),cudaMemcpyDeviceToHost));
        return (double)h_res;
    };
    auto relerr=[&](double v){
        return fabs(v-ref)/fabs(ref);
    };

    struct Row{
        const char* name;
        float ms;
        double val,relerr;
    };

    // 统一的「跑一个 kernel + 计时 + 验证」流程
    auto run_one=[&](const char* name,size_t grid,auto&& kernel_launch){
        auto once=[&]{
            kernel_launch(grid);
            pass2(grid);
        };
        Row r{
            name,bench(once,5,20),0,0
        };
        once();
        CUDA_CHECK(cudaDeviceSynchronize());
        r.val=grab();
        r.relerr=relerr(r.val);
        return r;
    };

    auto print_table=[&](const char* title,Row* rows,int k,double moved_gb){
        printf("=== %s ===\n", title);
        printf("%-30s %10s %14s %8s %16s %11s\n",
               "kernel", "time(ms)", "eff.BW", "%peak", "result", "rel.err");
        for (int i = 0; i < k; ++i) {
            double bw = moved_gb / (rows[i].ms / 1e3);
            printf("%-30s %10.4f %11.0f GB/s %7.1f%% %16.2f %11.3e\n",
                   rows[i].name, rows[i].ms, bw, 100.0 * bw / peak_bw,
                   rows[i].val, rows[i].relerr);
        }
        printf("\n");
    };

    // =======================================================================
    // 实验 A：厚归约（grid-stride，每线程处理 n/(gridP*BLOCK) 个元素）
    //     预期结论：所有版本几乎一样快，都贴着 HBM 带宽上限。
    //     因为时间 = 读一遍 512 MB 的时间，归约那 8 步在里面占比极小。
    //     这个「负面结果」是今天最重要的收获之一：优化要先看它占多大比例。
    // =======================================================================
    {
        Row rows[7];
        rows[0]=run_one("(A) serial merge",gridP,[&](size_t g){ reduce_serial_merge<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[1]=run_one("(B) interleave+divergent",gridP,[&](size_t g){ reduce_interleave_div<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[2]=run_one("(C) interleave+bankconf",gridP,[&](size_t g){ reduce_interleave_conf<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[3]=run_one("(D) tree sequential",gridP,[&](size_t g){ reduce_tree_seq<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[4]=run_one("(E) tree+shfl unrolled",gridP,[&](size_t g){ reduce_tree_unroll<BLOCK><<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[5]=run_one("(F) warp shuffle 2stage",gridP,[&](size_t g){ reduce_warp_2stage<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[6]=run_one("(G) GUB BlockReduce",gridP,[&](size_t g){ reduce_cub<<<g,BLOCK>>>(d_in,d_partial,n);});
        char t[128];
        snprintf(t,sizeof t,"Exp A thick reduce: grid=%zu, %.0f elems/thread",gridP,(double)n/(gridP*BLOCK));
        print_table(t,rows,7,gb);
        printf("解读：几乎全部贴着 HBM 带宽 → 这是 memory-bound kernel，"
               "      归约算法的优劣被访存时间淹没了。\n");
    }

    // =======================================================================
    // 实验 B：薄归约（每线程恰好 1 个元素，归约成为主要成本）
    //    这里才能看出树形的真实价值。同时 (B)(C) 的两个错误也会显形。
    // =======================================================================
    {
        Row rows[7];
        rows[0]=run_one("(A) serial merge",gridT,[&](size_t g){ reduce_serial_merge<<<g,BLOCK>>>(d_in,d_partial,n);}); 
        rows[1]=run_one("(B) interleave+divergent",gridT,[&](size_t g){ reduce_interleave_div<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[2]=run_one("(C) interleave+bankconf",gridT,[&](size_t g){ reduce_interleave_conf<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[3]=run_one("(D) tree sequential",gridT,[&](size_t g){ reduce_tree_seq<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[4]=run_one("(E) tree+shfl unrolled",gridT,[&](size_t g){ reduce_tree_unroll<BLOCK><<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[5]=run_one("(F) warp shuffle 2stage",gridT,[&](size_t g){ reduce_warp_2stage<<<g,BLOCK>>>(d_in,d_partial,n);});
        rows[6]=run_one("(G) GUB BlockReduce",gridT,[&](size_t g){ reduce_cub<<<g,BLOCK>>>(d_in,d_partial,n);});
        char t[128];
        snprintf(t, sizeof t, "Exp B thin reduce: grid=%zu, exactly 1 elem/thread", gridT);
        print_table(t,rows,7,gb);
        printf("解读：(A) 应当明显最慢（串行 256 步）；(D)/(E)/(F) 拉开差距；"
               "      (B) 的错误是 warp 分化，(C) 的错误是 bank conflict\n");
    }

    // =======================================================================
    // 实验 C：bank_probe —— 隔离出 bank conflict 本身
    // =======================================================================
    {
        const int iters=4096;
        const size_t gridB=p.multiProcessorCount;   // 每 SM 一个 block，避免占用率干扰
        float* d_probe=nullptr;
        CUDA_CHECK(cudaMalloc(&d_probe,gridB*BLOCK*sizeof(float)));

        float t0=bench([&]{ bank_probe<0><<<gridB,BLOCK>>>(d_probe,iters);},5,20);
        float t1=bench([&]{ bank_probe<1><<<gridB,BLOCK>>>(d_probe,iters);},5,20);
        float t2=bench([&]{ bank_probe<2><<<gridB,BLOCK>>>(d_probe,iters);},5,20);
        float t4=bench([&]{ bank_probe<4><<<gridB,BLOCK>>>(d_probe,iters);},5,20);
        float t32=bench([&]{ bank_probe<32><<<gridB,BLOCK>>>(d_probe,iters);},5,20);
        CUDA_CHECK(cudaDeviceSynchronize());

        printf("=== 实验 C bank_probe: 纯 shared 读，%d 次/线程 ===\n", iters);
        printf("%-34s %10s %10s\n", "访问模式", "time(ms)", "相对 stride=1");
        printf("%-34s %10.4f %9.2fx\n", "STRIDE=0  (同地址 → 广播)",     t0,  t0  / t1);
        printf("%-34s %10.4f %9.2fx\n", "STRIDE=1  (连续 → 无冲突)",   t1,  1.0);
        printf("%-34s %10.4f %9.2fx\n", "STRIDE=2  (预期 2-way)",        t2,  t2  / t1);
        printf("%-34s %10.4f %9.2fx\n", "STRIDE=4  (预期 4-way)",        t4,  t4  / t1);
        printf("%-34s %10.4f %9.2fx\n", "STRIDE=32 (全同 bank → 32-way)", t32, t32 / t1);
        printf("解读：STRIDE=0 应当和 =1 一样快（广播不是冲突）；\n"
               "      STRIDE=2/4/32 的耗时比应当接近 2/4/32 —— 这就是「串行化」的字面含义。\n"
               "      用 ncu 的 bank_conflicts 计数器对一下，两者应当高度吻合。\n\n");
        CUDA_CHECK(cudaFree(d_probe));
    }

    // =======================================================================
    // 实验 D：RMSNorm 行归约
    // =======================================================================
    {
        const int H=4096;               // Llama-7B 的 hidden size
        const int R=8192;               // 8192 行 ≈ 一个长 prompt 的 token 数
        const size_t xn=(size_t)R*H;
        float* h_x=(float*)malloc(xn*sizeof(float));
        float* h_w=(float*)malloc(H*sizeof(float));
        for(size_t i=0;i<xn;++i){
            h_x[i]=(float)rand()/RAND_MAX-0.5f;
        }
        for(int i=0;i<H;++i){
            h_w[i]=1.0f+0.01f*i/H;
        }

        float *d_x,*d_w,*d_y;
        CUDA_CHECK(cudaMalloc(&d_x,xn*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w,H*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y,xn*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_x,h_x,xn*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w,h_w,H*sizeof(float),cudaMemcpyHostToDevice));

        const float eps=1e-6f;
        // CPU 参考：只算第 0 行，用 double
        double ss=0.0;
        for(int i=0;i<H;++i){
            ss+=(double)h_x[i]*h_x[i];
        }
        double sc=1.0/sqrt(ss/H+eps);

        float* h_y=(float*)malloc(xn*sizeof(float));
        auto check_row0=[&](){
            CUDA_CHECK(cudaMemcpy(h_y,d_y,(size_t)H*sizeof(float),cudaMemcpyDeviceToHost));
            double maxrel=0.0;
            for(int i=0;i<H;++i){
                double want=(double)h_x[i]*sc*h_w[i];
                double got=h_y[i];
                double d=fabs(got-want)/(fabs(want)+1e-12);
                if(d>maxrel) maxrel=d;
            }
            return maxrel;
        };

        // RMSNorm 的访存下界：读 x + 写 y（w 很小，忽略）→ 2 × xn × 4 B
        const double rms_gb=2.0*xn*sizeof(float)/1e9;
        struct RRow{
            const char* name;
            float ms;
            double err;
        };

        RRow rr[3];
        rr[0]={"rmsnorm serial merge",bench([&]{rmsnorm_serial<<<R,BLOCK>>>(d_x,d_w,d_y,H,eps);},5,20),0};
        rmsnorm_serial<<<R,BLOCK>>>(d_x,d_w,d_y,H,eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        rr[0].err = check_row0();

        rr[1]={"rmsnorm tree",bench([&]{rmsnorm_tree<<<R,BLOCK>>>(d_x,d_w,d_y,H,eps);},5,20),0};
        rmsnorm_tree<<<R,BLOCK>>>(d_x,d_w,d_y,H,eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        rr[1].err = check_row0();

        rr[2]={"rmsnorm warp shfl",bench([&]{rmsnorm_shfl<<<R,BLOCK>>>(d_x,d_w,d_y,H,eps);},5,20),0};
        rmsnorm_shfl<<<R,BLOCK>>>(d_x,d_w,d_y,H,eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        rr[2].err = check_row0();

        printf("=== 实验 D RMSNorm 行归约 (R=%d 行 × H=%d) ===\n", R, H);
        printf("%-24s %10s %14s %8s %12s\n", "kernel", "time(ms)", "eff.BW", "%peak", "max rel.err");
        for (auto& r : rr) {
            double bw = rms_gb / (r.ms / 1e3);
            printf("%-24s %10.4f %11.0f GB/s %7.1f%% %12.3e\n",
                   r.name, r.ms, bw, 100.0 * bw / peak_bw, r.err);
        }
        printf("解读：这里归约占比高，串行版会明显吃亏；shuffle 版应当最接近访存下界。\n"
               "      注意 eff.BW 用的是「读 x + 写 y」的理论下界，超过 100%% 说明 cache 帮了忙。\n\n");

        CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_y));
        free(h_x); free(h_w); free(h_y);
    }

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_partial)); CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    printf("done.\n");
    return 0;
}

// ============================================================================
// 附录一：三个错误的归约样例
//
//   样例 1：沿用 Volta 之前的 volatile warp 展开
//   if (tid < 32) {
//       volatile float* vs = s;
//       vs[tid] += vs[tid + 32];   // 假设「warp 内 32 线程锁步」
//       vs[tid] += vs[tid + 16];   // Volta 起独立线程调度已打破这个假设
//       ...                         // → 偶发错误，且只在特定 occupancy 下出现
//   }
//    正确：本文件 (E) 的写法，用 __shfl_down_sync 显式指定参与线程。
//
//   样例 2：把 barrier 放进折半分支里
//   for (off = BS/2; off > 0; off >>= 1) {
//       if (tid < off) { s[tid] += s[tid + off]; __syncthreads(); }  // divergent barrier
//   }
//    正确：barrier 必须在 if 外面。synccheck 能抓。
//
//   样例 3：只在循环外加一次 barrier
//   s[tid] = sum; __syncthreads();
//   for (off = BS/2; off > 0; off >>= 1) if (tid < off) s[tid] += s[tid + off];
//   → 结果偶尔偏小：第 k 轮读的是第 k-1 轮写的，不同步就读到旧值。
//    正确：每一轮都要同步；只有「同一个 warp 内部」才能靠 shuffle 免同步。
//













