// =============================================================================
// 05_tiled_matmul.cu  ——  分块矩阵乘（tiled matmul）+ cuBLAS 差距分析
//
// 上游    ：04/04_tree_reduction.cu（shared memory / bank conflict / barrier）
//           02/02_memory_access.cu（合并访问 / 内存层级）
// 下游    ：CUDA RMSNorm、FlashAttention（分块 + online softmax）
//
// -----------------------------------------------------------------------------
// 环境 / 编译 / 运行
// -----------------------------------------------------------------------------
//   环境： CUDA 12.x
//         需要 sm_80 以上（用到 TF32 WMMA）。
//
//   编译（-Xptxas -v 看：寄存器数 / shared 用量 / 有没有 spill）：
//     nvcc -O3 -arch=sm_120 -lineinfo -Xptxas -v -Xcompiler="/utf-8 /Zc:preprocessor /std:c++17" -o 05_tiled_matmul.cu 05_tiled_matmul.cu -lcublas
//
//   运行：
//     ./05_tiled_matmul.cu                     # 默认 M=N=K=4096，跑全部 5 组实验
//     ./05_tiled_matmul.cu 2048 2048 2048      # 自定义规模
//     ./05_tiled_matmul.cu 4096 4096 4096 20   # 第 4 个参数 = 计时迭代次数
//
//   正确性检验：
//     compute-sanitizer --tool memcheck   ./05_tiled_matmul.cu 512 512 512
//     compute-sanitizer --tool racecheck  ./05_tiled_matmul.cu 512 512 512
//     compute-sanitizer --tool synccheck  ./05_tiled_matmul.cu 512 512 512
//
//   profiling：
//     # ① 在 Roofline 上的位置：计算吞吐 vs 访存吞吐
//     ncu --kernel-name-base demangled \
//         --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
//dram__throughput.avg.pct_of_peak_sustained_elapsed,\
//dram__bytes_read.sum,\
//l1tex__t_sector_hit_rate.pct,\
//lts__t_sector_hit_rate.pct ./tmm 4096 4096 4096 3
//
//     # ② shared memory ：访问次数 + bank 冲突
//     ncu --metrics smsp__inst_executed_op_shared_ld.sum,\
//smsp__inst_executed_op_shared_st.sum,\
//l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
//l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./tmm 4096 4096 4096 3
//
//     # ③ 占用率 + 分化
//     ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
//smsp__thread_inst_executed_per_inst_executed.ratio,\
//launch__occupancy_limit_registers,\
//launch__occupancy_limit_shared_mem ./tmm 4096 4096 4096 3
//
//     # ④ 有没有用到 Tensor Core（区分 fp32 CUDA core 和 TF32 tensor core）
//     ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active \
//         ./tmm 4096 4096 4096 3
//
// -----------------------------------------------------------------------------
// 本文件的 8 个 kernel = 一条"从 0.25 到 32 FLOP/Byte"的算术强度阶梯
// -----------------------------------------------------------------------------
//   (A) mm_naive_uncoalesced : 朴素，且 row 绑 threadIdx.x —— 破坏合并访问
//   (B) mm_naive_coalesced   : 朴素，col 绑 threadIdx.x   —— 朴素合并，快好几倍
//   (C) mm_tiled<TILE>       : 经典 shared memory 分块
//   (D) mm_reg1d             : 一维寄存器分块（每线程 8 个输出）
//   (E) mm_reg2d             : 二维寄存器分块（每线程 8x8 个输出）  质变
//   (F) mm_reg2d_vec<PAD>    : float4 向量化访存 + shared padding 消 bank 冲突
//   (G) cuBLAS SGEMM         : 工业基准（FP32 CUDA core）
//   (H) cuBLAS SGEMM + TF32  : 工业基准（TF32 Tensor Core，另一个量级）
//   (I) mm_wmma_tf32_naive   : 手写 Tensor Core，但没有 tiling
// =============================================================================


//----------C 标准库-----------
#include <cstdio>
#include <cstdlib>          // 通用工具。malloc/free、rand、atoi、exit
#include <cstring>          // C 字符串和内存操作。memcpy、memset、strcmp
#include <cmath>            // 数学函数。sqrt、pow、fabs、exp
//--------c++ 标准库----------
#include <vector>           // 动态数组容器 std::vector
#include <string>           // 字符串类 std::string
#include <random>           // C++11 随机数库
#include <algorithm>        // 算法库。std::sort、std::min、std::max、std::fill

//--------CUDA 相关库----------
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

// -----------------------------------------------------------------------------
// 0. 错误检查 + 计时
// -----------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                                        \
    do {                                                                                        \
        cudaError_t err=(expr);                                                                 \
        if(err!=cudaSuccess){                                                                   \
            fprintf(stderr,"[CUDA] %s:%d   %s\n",__FILE__,__LINE,cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    }while(0)

#define CUBLAS_CHECK(expr)                                                                      \
    do {                                                                                        \
        cublasStatus_t err=(expr);                                                              \
        if(err!=CUBLAS_STATUS_SUCCESS){                                                         \
            fprintf(stderr,"[cuBLAS] %s:%d  status=%d\n",__FILE__,__LINE__,(int)err);           \
            exit(EXIT_FAILURE);                                                                 \
        }                                                                                       \
    }while(0)

template <typename Fn>
static double time_ms(Fn&& fn,int warmup,int iters){
    for(int i=0;i<warmup;++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t beg,end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for(int i=0;i<iters;++i) fn();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms=0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms,beg,end));
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return (double)ms/iters;
}


// =============================================================================
// (A)(B) 朴素 matmul —— 一行之差，带宽利用率天壤之别
// =============================================================================
//
// 两个版本的数学完全一样：C[row][col] = Σ_k A[row][k] * B[k][col]
// 唯一区别是"哪个维度绑到 threadIdx.x"。threadIdx.x 是 warp 内变化最快的维度
// （lane = threadIdx.x % 32），所以它决定了一个 warp 的 32 个线程访问的地址是连续的还是散的

// 反面教材：row 绑 threadIdx.x
__global__ void mm_naive_uncoalesced(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;          // ← 变化最快的是 row
    const int col=blockIdx.y*blockDim.y+threadIdx.y;
    if(row>=M||col>=N) return;

    float acc=0.0f;
    for(int k=0;k<K;++k){
        // 同一个 warp 的 32 个线程：row 连续、col 相同
        //   读 A[row*K+k] → 地址相隔 K*4 = 16KB，32 个线程散在 32 个不同的 128B 段
        //   读 B[k*N+col] → 32 个线程读同一个地址（广播，其实没问题）
        //   写 C[row*N+col] → 同样相隔 16KB，散
        acc+=A[(size_t)row*K+k]*B[(size_t)k*N+col];
    }
    C[(size_t)row*N+col]=acc;
}


__global__ void mm_naive_coalesced(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    const int row=blockIdx.y*blockDim.y+threadIdx.y;
    if(row>=M||col>=N) return;

    float acc=0.0f;
    for(int k=0;k<K;++k){
        // 同一个 warp：col 连续、row 相同
        //   读 A[row*K+k] → 32 个线程同一个地址 → 广播（1 个 sector 搞定）
        //   读 B[k*N+col] → 32 个连续 float = 128 B → 正好一次完整事务
        //   写 C[row*N+col] → 同上，完美合并
        acc+=A[(size_t)row*K+k]*B[(size_t)k*N+col];
    }
    C[(size_t)row*N+col]=acc;
}

// =============================================================================
// (C) 经典 shared memory 分块矩阵乘
// =============================================================================
//
// 朴素版每算一次乘加就去 global显存 搬两个数；分块版把
// 一小块 A 和一小块 B 搬进 shared memory，让这一小块被 block 内
// 所有线程复用 TILE 次，然后才换下一块。
//
// 算术强度：
//   每个 C 的 TILE×TILE 块，需要从 global 读 (K/TILE) 对 tile，每对 2*TILE²个 float
//     → 访存量 = 2 * K * TILE * 4 Byte
//     → 计算量 = 2 * TILE² * K FLOP
//     → AI = 2*TILE²*K / (8*K*TILE) = TILE/4  FLOP/Byte
//   TILE=32 → 8 FLOP/Byte。H100 FP32 的 ridge point 约 20 → 仍然 memory-bound
//   这个"还不够"就是 (D)(E) 存在的理由。
//
// 边界安全：允许任意 M/N/K（越界填 0，不影响乘加结果）
template <int TILE>
__global__ void mm_tiled(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int tx=threadIdx.x;           // 0..TILE-1（变化最快）
    const int ty=threadIdx.y;
    const int row=blockIdx.y*TILE+ty;               // 本线程负责 C 的哪一行
    const int col=blockIdx.x*TILE+tx;               // 本线程负责 C 的哪一列

    float acc=0.0f;
    const int nTiles=(K+TILE-1)/TILE;

    for(int t=0;t<nTiles;++t){
        const int aCol=t*TILE+tx;                // A 这一块的列偏移
        const int bRow=t*TILE+ty;                // B 这一块的行偏移

        // ── 阶段①：搬运。每个线程搬一个元素，全 block 协作搬两个 tile ──
        // 注意 tx 在两次 load 里都出现在"列"的位置 → 从 global 读是连续的（合并）
        sA[ty][tx]=(row<M&&aCol<K)?A[(size_t)row*K+aCol]:0.0f;
        sB[ty][tx]=(bRow<K&&col<N)?B[(size_t)bRow*N+col]:0.0f;

        __syncthread();        // 栅栏 1：等全 block 搬完，才能开始用别人搬的数据

        // ── 阶段②：计算。这段完全不碰 global memory ──
        // bank 分析：
        //   sA[ty][k]：一个 warp 内 ty 只有 1~2 个取值 → 同地址广播，不冲突
        //   sB[k][tx]：tx 连续 → 落在 32 个不同 bank，不冲突
        //   所以这个"最教科书"的写法恰好是 bank-conflict-free 的
#pragma unroll
        for(int k=0;k<TILE;++k){
            acc+=sA[ty][k]*sB[k][tx];
        }

        __syncthread();    //   栅栏 2：等全 block 算完，才能覆盖 shared 迎接下一块
                           //   少了这个 → 跑得快的线程把 sA 覆盖掉，慢的线程读到下一块
                           //   的数据。典型症状：小规模"碰巧对"，大规模偶发错。
    }

    if(row<M&&col<N) C[(size_t)row*N+col]=acc;
}


// =============================================================================
// (D) 一维寄存器分块：每个线程负责 TM 个输出元素
// =============================================================================
//
// (C) 的新瓶颈不在 global，而在 shared：内层每做 1 次 FMA 要从 shared 读 2 个
// float（8 Byte）。H100 一个 SM 的 shared 带宽是 128 B/cycle，FP32 FMA 吞吐是
// 128 次/cycle → 要吃满算力需要 1 FMA/Byte，而 (C) 只有 0.125 → 天花板 12.5%。
//
// 解法：让一个线程算 TM 个输出。读 1 个 B 值 + TM 个 A 值（TM+1 个 float），
// 做 TM 次 FMA → TM=8 时是 8 FMA / 36 Byte ≈ 0.22 FMA/Byte，好了近 2 倍。
// 同时 block tile 变大（64×64），global 层面的算术强度也从 8 涨到 16。
//
// 约束：M%BM==0, N%BN==0, K%BK==0（工业库也是这么干的：aligned kernel 走快路，
//       非对齐规模走另一个 generic kernel。）
template <int BM,int BN,int BK,int TM>
__global__ void mm_reg1d(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    static_assert(BM*BN%TM==0,"thread count must divide evenly");
    constexpr int THREADS=(BM*BN)/TM;
    static_assert(BM*BK==THREADS,"A tile load needs exactly one pass");
    static_assert(BK*BN==THREADS,"B tile load needs exactly one pass");

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    // 本 block 负责 C 的哪一块
    A+=(size_t)blockIdx.y*BM*K;
    B+=(size_t)blockIdx.x*BN;
    C+=(size_t)blockIdx.y*BM*N+(size_t)blockIdx.x*BM;

    // 计算时的分工：本线程负责 C 块内的 TM 行 × 1 列
    const int tCol=threadIdx.x%BN;          // 0..BN-1（连续 → 访存友好）
    const int tRow=threadIdx.x/BN;          // 0..BM/TM-1

    // 搬运时的分工（和计算时的分工可以不一样。 这是分块 kernel 的常见做法：
    // 搬运追求"合并访问"，计算追求"寄存器复用"，两者最优的索引映射不同）
    const int irA=threadIdx.x/BK,icA=threadIdx.x%BK;
    const int irB=threadIdx.x/BN,icA=threadIdx.x%BN;

    float acc[TM]={};

    for(int bk=0;bk<K;bk+=BK){
        sA[irA][icA]=A[(size_t)irA*K+icA];
        sB[irB][icB]=B[(size_t)irB*N+icB];
        __syncthread();

        A+=BK;            // 指针沿 K 维滑动（比每次重算 index 省寄存器和指令）
        B+=(size_t)BK*N;
    }
}

















































