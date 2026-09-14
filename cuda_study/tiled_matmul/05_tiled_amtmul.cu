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

        __syncthreads();        // 栅栏 1：等全 block 搬完，才能开始用别人搬的数据

        // ── 阶段②：计算。这段完全不碰 global memory ──
        // bank 分析：
        //   sA[ty][k]：一个 warp 内 ty 只有 1~2 个取值 → 同地址广播，不冲突
        //   sB[k][tx]：tx 连续 → 落在 32 个不同 bank，不冲突
        //   所以这个"最教科书"的写法恰好是 bank-conflict-free 的
#pragma unroll
        for(int k=0;k<TILE;++k){
            acc+=sA[ty][k]*sB[k][tx];
        }

        __syncthreads();    //   栅栏 2：等全 block 算完，才能覆盖 shared 迎接下一块
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
    C+=(size_t)blockIdx.y*BM*N+(size_t)blockIdx.x*BN;

    // 计算时的分工：本线程负责 C 块内的 TM 行 × 1 列
    const int tCol=threadIdx.x%BN;          // 0..BN-1（连续 → 访存友好）
    const int tRow=threadIdx.x/BN;          // 0..BM/TM-1

    // 搬运时的分工（和计算时的分工可以不一样。 这是分块 kernel 的常见做法：
    // 搬运追求"合并访问"，计算追求"寄存器复用"，两者最优的索引映射不同）
    const int irA=threadIdx.x/BK,icA=threadIdx.x%BK;
    const int irB=threadIdx.x/BN,icB=threadIdx.x%BN;

    float acc[TM]={};

    for(int bk=0;bk<K;bk+=BK){
        sA[irA][icA]=A[(size_t)irA*K+icA];
        sB[irB][icB]=B[(size_t)irB*N+icB];
        __syncthreads();

        A+=BK;            // 指针沿 K 维滑动（比每次重算 index 省寄存器和指令）
        B+=(size_t)BK*N;
#pragma unroll
        for(int k=0;k<BK;++k){
            const float bv=sB[k][tCol];      //  读一次，复用 TM 次
#pragma unroll
            for(int i=0;i<TM;++i){
                acc[i]+=sA[tRow*TM+i][k]*bv;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(int i=0;i<TM;++i){
        C[(size_t)(tRow*TM+i)*N+tCol]=acc[i];
    }
}

// =============================================================================
// (E) 二维寄存器分块：每个线程负责 TM×TN 个输出 —— 质变发生在这一步
// =============================================================================
//
// 关键的账：
//   每个 k 步，读 TM 个 A 值 + TN 个 B 值 = (TM+TN) 个 float，
//   却能做 TM*TN 次 FMA。TM=TN=8 → 16 个 float（64 B）换 64 次 FMA
//   = 1.0 FMA/Byte —— 正好等于 H100 "128 FMA/cycle : 128 B/cycle" 的比例。
//   这就是为什么工业库的 thread tile 几乎都是 8×8 或 8×4，不是随便选的。
//
// 同时 block tile 128×128 → global 算术强度 = 128*128/(2*256) = 32 FLOP/Byte
//   > H100 FP32 ridge point (~20) → 第一次真正跨进 compute-bound 区域。
//
// 实现细节：A 的 tile 转置存进 shared（sA[BK][BM]），这样内层循环读 A 的一"列"
// 变成读连续地址，才能用 float4 一次取 4 个（见 (F)）。
template <int BM,int BN,int BK,int TM,int TN>
__global__ void mm_reg2d(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    constexpr int THREADS=(BM*BN)/(TM*TN);
    static_assert(BM*BK%THREADS==0,"A tile must be loadable in whole passes");
    static_assert(BK*BN%THREADS==0,"B tile must be loadable in whole passes");

    __shared__ float sA[BK][BM];      // ← 转置存：sA[k][m]
    __shared__ float sB[BK][BN];      // ← 正常存：sB[k][n]

    A+=(size_t)blockIdx.y*BM*K;       //第 blockIdx.y 块 BM*BN tile 块 的第一行
    B+=(size_t)blockIdx.x*BN;         //第 blockIdx.x 块 BM*BN tile 块 的第一行
    C+=(size_t)blockIdx.y*BM*N+(size_t)blockIdx.x*BN;      // C的第 blockIdx.y 行, 第 blockIdx.x 列

    // 计算分工：thread tile 网格是 (BM/TM) × (BN/TN)
    const int tRow=threadIdx.x/(BN/TN);
    const int tCol=threadIdx.x%(BN/TN);

    // 搬运分工
    const int irA=threadIdx.x/BK,icA=threadIdx.x%BK;
    const int irB=threadIdx.x/BN,icB=threadIdx.x%BN;
    constexpr int strideA=THREADS/BK;   // 一趟能覆盖多少行 A
    constexpr int strideB=THREADS/BN;   // 一趟能覆盖多少行 B

    float acc[TM][TN]={};
    float regM[TM],regN[TN];

    for(int bk=0;bk<K;bk+=BK){
#pragma unroll
        for(int off=0;off<BM;off+=strideA){
            // 转置写入：读 A 是行连续（合并），写 shared 是跨行（这里会有 bank
            // 冲突，(F) 用 padding 改善）
            sA[icA][irA+off]=A[(size_t)(irA+off)*K+icA];
        }
#pragma unroll
        for(int off=0;off<BK;off+=strideB){
            sB[irB+off][icB]=B[(size_t)(irB+off)*N+icB];
        }
        __syncthreads();
        A+=BK;
        B+=(size_t)BK*N;

#pragma unroll
        for(int k=0;k<BK;++k){
            // 先把这一列 A / 这一行 B 拉进寄存器（每个只读一次）
#pragma unroll
            for(int i=0;i<TM;++i) regM[i]=sA[k][tRow*TM+i];
#pragma unroll
            for(int j=0;j<TN;++j) regN[j]=sB[k][tCol*TN+j];
            // 然后在寄存器里做 TM*TN 次 FMA —— 一次 shared 读，多次复用
            // 这 64 条 FADD/FFMA 互相独立 → 指令级并行(ILP)拉满，掩盖 FMA 延迟
#pragma unroll
            for(int i=0;i<TM;++i){
#pragma unroll
                for(int j=0;j<TN;++j){
                    acc[i][j]+=regM[i]*regN[j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(int i=0;i<TM;++i){
#pragma unroll
        for(int j=0;j<TN;++j){
            C[(size_t)(tRow*TM+i)*N+tCol*TN+j]=acc[i][j];
        }
    }
}

// =============================================================================
// (F) + float4 向量化访存 + shared memory padding（PAD 模板参数用来做 A/B 对照）
// =============================================================================
//
// 两个新技巧：
//  1) float4：一条 LDG.128 / STS.128 / LDS.128 顶四条标量指令。
//     省的是"指令发射带宽"和"访存请求数"——不是 DRAM 字节数。
//  2) PAD：sA 声明成 [BK][BM+PAD]。手算（路数取决于一个 warp 内
//     irA 有几个不同取值，改搬运分工路数就变，只能算）：
//       标量版 (E)：irA = tid/8 → 一个 warp 只有 4 个 irA
//               PAD=0 → bank = (icA*128 + irA) % 32 = irA % 32
//                       → 只用到 4 个 bank，每 bank 8 个 lane → 8-way 冲突
//       float4 版 (F)：每线程搬 4 个，一个 warp 覆盖更多行，路数更低
//       PAD=4 → bank = (icA*(BM+4) + irA) % 32 = (icA*4 + irA) % 32
//               → 4*icA 取 {0,4,...,28}，irA 取 {0..3} 填满间隔 → 冲突归零
//     注意 sB 侧没修：bank = (tCol*8 + j) % 32，tCol*8 只有 4 个取值 → 仍 8-way，
//     且病因是 tCol 乘了 8，padding 解决不了，要 swizzle（交错寻址）。
//     PAD=4 而不是 PAD=1 的原因：要保持 16 Byte 对齐，float4 读取才不会崩。
//     （4 floats = 16 B，(128+4)*4 = 528 B，528 % 16 == 0）
template <int BM,int BN,int BK,int TM,int TN,int PAD>
__global__ void mm_reg2d_vec(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    constexpr int THREADS=(BM*BN)/(TM*TN);
    static_assert(TM==8&&TN==8,"手写展开假设 8x8 thread tile");
    static_assert(BM*BK/4==THREADS,"A tile: 每线程恰好一个 float4");
    static_assert(BK*BN/4==THREADS,"B tile: 每线程恰好一个 float4");
    static_assert(PAD%4==0,"PAD 必须是 4 的倍数，否则 float4 读取不对齐");

    __shared__ float sA[BK][BM+PAD];
    __shared__ float sB[BK][BN];

    A+=(size_t)blockIdx.y*BM*K;
    B+=(size_t)blockIdx.x*BN;
    C+=(size_t)blockIdx.y*BM*N+(size_t)blockIdx.x*BN;

    const int tRow=threadIdx.x/(BN/TN);
    const int tCol=threadIdx.x%(BN/TN);

    // float4 搬运分工：每线程一个 float4
    const int irA=threadIdx.x/(BK/4);            // A 的行
    const int icA=(threadIdx.x%(BK/4))*4;        // A 的列（4 的倍数）
    const int irB=(threadIdx.x/(BN/4));          // B 的行
    const int icB=(threadIdx.x%(BN/4))*4;        // B 的列

    float acc[TM][TN]={};
    float regM[TM],regN[TN];

    for(int bk=0;bk<K;bk+=BK){
        // A：一次读 4 个连续的 K 元素，然后拆成 4 个标量转置写进 shared
        {
            const float4 t=*reinterpret_cast<const float4*>(&A[(size_t)irA*K+icA]);
            sA[icA+0][irA]=t.x;
            sA[icA+1][irA]=t.y;
            sA[icA+2][irA]=t.z;
            sA[icA+3][irA]=t.w;
        }
        // B：读连续 4 个 N 元素，直接 float4 写进 shared（行方向，无需转置）
        {
            const float4 t=*reinterpret_cast<const float4*>(&B[(size_t)irB*N+icB]);
            *reinterpret_cast<float4*>(&sB[irB][icB])=t;
        }
        __syncthreads();

        A+=BK;
        B+=(size_t)BK*N;

#pragma unroll
        for(int k=0;k<BK;++k){
            // 用 float4 从 shared 取：2 条 LDS.128 代替 8 条 LDS.32
            // 注意：不要对 regM 取地址后 reinterpret_cast —— 那会把局部数组
            // 逼进 local memory（=显存）。逐成员拷贝才能保证留在寄存器里。
            {
                const float4 a0=*reinterpret_cast<const float4*> (&sA[k][tRow*TM]);
                const float4 a1=*reinterpret_cast<const float4*> (&sA[k][tRow*TM+4]);
                regM[0]=a0.x;regM[1]=a0.y;regM[2]=a0.z;regM[3]=a0.w;
                regM[4]=a1.x;regM[5]=a1.y;regM[6]=a1.z;regM[7]=a1.w;
            }
            {
                const float4 b0=*reinterpret_cast<const float4*>(&sB[k][tCol*TN]);
                const float4 b1=*reinterpret_cast<const float4*>(&sB[k][tCol*TN+4]);
                regN[0]=b0.x;regN[1]=b0.y;regN[2]=b0.z;regN[3]=b0.w;
                regN[4]=b1.x;regN[5]=b1.y;regN[6]=b1.z;regN[7]=b1.w;
            }
#pragma unroll
            for(int i=0;i<TM;++i){
#pragma unroll
                for(int j=0;j<TN;++j) acc[i][j]+=regM[i]*regN[j];
            }
        }
        __syncthreads();
    }

    // 写回也用 float4：连续 4 列一次写出
#pragma unroll
    for(int i=0;i<TM;++i){
        float* dst=&C[(size_t)(tRow*TM+i)*N+tCol*TN];
        *reinterpret_cast<float4*>(dst)=make_float4(acc[i][0],acc[i][1],acc[i][2],acc[i][3]);
        *reinterpret_cast<float4*>(dst+4)=make_float4(acc[i][4],acc[i][5],acc[i][6],acc[i][7]);
    }
}

// =============================================================================
// (I) 手写 Tensor Core（WMMA + TF32）—— 故意不做 tiling，用来证明一件事
// =============================================================================
//
// Tensor Core 不是使用就会快的。这个 kernel 每算一个 16×16 输出块都直接
// 从 global memory 读 A/B，算术强度和朴素版一个数量级 → 算力再强也被访存卡死。
// 预期：它可能还不如 (E)/(F)。这正是笔记 §6.3 的论点：
//   "Tensor Core 提供的是更高的计算屋顶，tiling 提供的是爬上去的梯子。"
//
// 精度提醒：TF32 = 8 位指数 + 10 位尾数（FP32 是 23 位尾数）。
// 所以这个 kernel 和 FP32 版本的结果差异会明显大得多（相对误差 ~1e-3 量级），
// 这不是 bug，是 TF32 的定义。工业上判断"能不能用 TF32"要看下游任务容不容忍。
namespace wmma=nvcuda::wmma;

__global__ void mm_wmma_tf32_naive(const float* __restrict__ A,const float* __restrict__ B,float* __restrict__ C,int M,int N,int K){
    // 一个 warp 负责一个 16×16 的 C 块；一个 block 4 个 warp 沿 N 排开
    const int warpN=blockIdx.x*4+threadIdx.y;
    const int warpM=blockIdx.y;
    if(warpM*16>=M||warpN*16>=N) return;

    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> fb;
    wmma::fill_fragment(fc, 0.0f);

    for (int k = 0; k < K; k += 8) {
        wmma::load_matrix_sync(fa, A + (size_t)warpM * 16 * K + k, K);
        wmma::load_matrix_sync(fb, B + (size_t)k * N + warpN * 16, N);
        // TF32 fragment 必须显式做一次舍入（CUDA 的要求，不是可选优化）
#pragma unroll
        for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
#pragma unroll
        for (int i = 0; i < fb.num_elements; ++i) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
        wmma::mma_sync(fc, fa, fb, fc);
    }
    wmma::store_matrix_sync(C + (size_t)warpM * 16 * N + warpN * 16, fc, N,wmma::mem_row_major);
}






// =============================================================================
// 1. 主机侧：结果表 + 正确性验证 + Roofline 模型
// =============================================================================

struct DeviceInfo {
    std::string name;
    int sms = 0;
    double fp32_peak = 0.0;   // FLOP/s，CUDA core（非 tensor）
    double bw_peak = 0.0;     // Byte/s
    size_t smem_per_sm = 0;
};

static DeviceInfo query_device() {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    DeviceInfo d;
    d.name = p.name;
    d.sms = p.multiProcessorCount;
    d.smem_per_sm = p.sharedMemPerMultiprocessor;
    // 假设：每个 SM 128 个 FP32 core，每 core 每周期 1 次 FMA = 2 FLOP
    // （Ampere/Hopper/Ada 的消费级与数据中心卡都成立；老架构需要改这个常数）
    d.fp32_peak = 2.0 * 128.0 * d.sms * (p.clockRate * 1e3);
    // HBM/GDDR 都是 DDR，所以乘 2
    d.bw_peak = 2.0 * (p.memoryClockRate * 1e3) * (p.memoryBusWidth / 8.0);
    if (d.bw_peak <= 0) d.bw_peak = 3.35e12;   // 驱动读不到时退回 H100 SXM 的标称值
    return d;
}

struct Row {
    std::string name;
    double ms = 0.0;
    double tflops = 0.0;
    double model_ai = -1.0;     // 模型算术强度（FLOP/Byte），-1 表示不适用
    double max_rel_err = -1.0;
    std::string note;
};

// 用 double 在 CPU 上抽样重算若干个 C 元素 —— 全量重算 4096³ 太慢（1.4e11 次
// 乘加），抽样是工业上标准做法：既能抓住"整体算错"，也能抓住"某个 tile 边界错"。
// 关键是抽样要覆盖边角（第一行/最后一行/最后一列）而不是纯随机。
static double sampled_max_rel_err(const std::vector<float>& hA,
                                  const std::vector<float>& hB,
                                  const std::vector<float>& hC,
                                  int M, int N, int K, int nSamples) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> di(0, M - 1), dj(0, N - 1);

    std::vector<std::pair<int,int>> pts;
    pts.push_back({0, 0});
    pts.push_back({0, N - 1});
    pts.push_back({M - 1, 0});
    pts.push_back({M - 1, N - 1});
    for (int s = (int)pts.size(); s < nSamples; ++s) pts.push_back({di(rng), dj(rng)});

    double worst = 0.0;
    for (auto& pr : pts) {
        const int i = pr.first, j = pr.second;
        double ref = 0.0;
        for (int k = 0; k < K; ++k) ref += (double)hA[(size_t)i * K + k] *
                                          (double)hB[(size_t)k * N + j];
        const double got = (double)hC[(size_t)i * N + j];
        const double den = std::max(1e-8, std::fabs(ref));
        worst = std::max(worst, std::fabs(got - ref) / den);
    }
    return worst;
}

// 和参考结果（cuBLAS）逐元素比：返回 max|diff| / max|ref|
static double max_rel_err_vs(const std::vector<float>& got,
                             const std::vector<float>& ref) {
    double maxAbs = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        maxAbs = std::max(maxAbs, (double)std::fabs(got[i] - ref[i]));
        maxRef = std::max(maxRef, (double)std::fabs(ref[i]));
    }
    return maxAbs / std::max(1e-8, maxRef);
}

// -----------------------------------------------------------------------------
// cuBLAS 的行主序陷阱
// -----------------------------------------------------------------------------
// cuBLAS 是列主序（Fortran 传统）。而 C/C++ 里我们用行主序。
// 关键洞察：**一块行主序的 M×N 数据，被列主序的眼睛看过去，就是 N×M 的转置。**
// 所以：把 (B, A) 按列主序传进去算 B^T · A^T = (A·B)^T = C^T，
//       而 C^T 用行主序的眼睛看回来，正好就是我们想要的 C。
// 于是参数是：m=N, n=M, k=K, A_ptr=B(lda=N), B_ptr=A(ldb=K), C_ptr=C(ldc=N)。
static void cublas_rowmajor_sgemm(cublasHandle_t h, const float* dA, const float* dB,
                                  float* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, N,       // 列主序看到的是 B^T (N×K)
                             dA, K,       // 列主序看到的是 A^T (K×M)
                             &beta,
                             dC, N));
}

// =============================================================================
// 2. 实验
// =============================================================================

int main(int argc, char** argv) {
    const int M = (argc > 1) ? atoi(argv[1]) : 4096;
    const int N = (argc > 2) ? atoi(argv[2]) : 4096;
    const int K = (argc > 3) ? atoi(argv[3]) : 4096;
    const int ITERS = (argc > 4) ? atoi(argv[4]) : 10;
    const int WARMUP = 3;

    const DeviceInfo dev = query_device();
    const double totalFlop = 2.0 * M * N * K;
    // 理论最小访存量：A、B、C 各只读/写一次
    const double minBytes = 4.0 * ((double)M * K + (double)K * N + (double)M * N);

    printf("=============================================================\n");
    printf(" 05_tiled_matmul  ——  W1 Day5-6 分块矩阵乘 + cuBLAS 差距分析\n");
    printf("=============================================================\n");
    printf("GPU              : %s (%d SM, shared/SM = %.0f KB)\n",
           dev.name.c_str(), dev.sms, dev.smem_per_sm / 1024.0);
    printf("FP32 峰值(CUDA core): %.1f TFLOP/s   HBM 峰值: %.0f GB/s\n",
           dev.fp32_peak / 1e12, dev.bw_peak / 1e9);
    printf("ridge point (FP32)  : %.1f FLOP/Byte  ← 低于这个值 = memory-bound\n",
           dev.fp32_peak / dev.bw_peak);
    printf("问题规模         : M=%d N=%d K=%d   总计算量 %.1f GFLOP\n", M, N, K,
           totalFlop / 1e9);
    printf("理论最小访存     : %.1f MB  →  问题本身的算术强度 = %.0f FLOP/Byte\n",
           minBytes / 1e6, totalFlop / minBytes);
    printf("               （问题本身极度 compute-bound；朴素实现却是 memory-bound，\n");
    printf("                 这个反差就是今天全部内容的起点）\n\n");

    // ── 数据准备（固定 seed：可复现是你自己定的产出规范）──────────────
    std::mt19937 rng(20260727);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * hA.size()));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * hB.size()));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * (size_t)M * N));
    CUDA_CHECK(cudaMalloc(&dRef, sizeof(float) * (size_t)M * N));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * hA.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * hB.size(), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<float> hRef((size_t)M * N), hC((size_t)M * N);

    // ── 先拿 cuBLAS 当参考，并用 double 抽样验证 cuBLAS 本身 ───────────
    cublas_rowmajor_sgemm(handle, dA, dB, dRef, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, sizeof(float) * hRef.size(),
                          cudaMemcpyDeviceToHost));
    const double refErr = sampled_max_rel_err(hA, hB, hRef, M, N, K, 64);
    printf("[参考校验] cuBLAS SGEMM vs CPU double（抽样 64 点）：max rel err = %.3e\n",
           refErr);
    printf("           K=%d 的 fp32 累加，1e-6~1e-5 量级是正常的（误差 ~ sqrt(K)*eps）\n\n",
           K);

    std::vector<Row> rows;

    auto run_and_record = [&](const std::string& name, double model_ai,
                              const std::string& note, auto launch) {
        CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * (size_t)M * N));
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("  [skip] %-28s launch 失败: %s\n", name.c_str(),
                   cudaGetErrorString(e));
            return;
        }
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, sizeof(float) * hC.size(),
                              cudaMemcpyDeviceToHost));
        Row r;
        r.name = name;
        r.model_ai = model_ai;
        r.note = note;
        r.max_rel_err = max_rel_err_vs(hC, hRef);
        r.ms = time_ms(launch, WARMUP, ITERS);
        r.tflops = totalFlop / (r.ms * 1e-3) / 1e12;
        rows.push_back(r);
        printf("  %-28s %9.3f ms  %8.2f TFLOP/s  %5.1f%% peak   relerr %.2e\n",
               r.name.c_str(), r.ms, r.tflops,
               100.0 * (r.tflops * 1e12) / dev.fp32_peak, r.max_rel_err);
    };

    // =========================================================================
    // 实验 1：一行之差 —— 合并访问对朴素 matmul 的影响
    // =========================================================================
    printf("--- 实验 1：朴素 matmul，row/col 谁绑 threadIdx.x ---------------\n");
    {
        dim3 blk(32, 32);
        dim3 grdBad((M + 31) / 32, (N + 31) / 32);
        dim3 grdGood((N + 31) / 32, (M + 31) / 32);
        run_and_record("A naive (uncoalesced)", 0.25, "row 绑 tx，访存散",
                       [&] { mm_naive_uncoalesced<<<grdBad, blk>>>(dA, dB, dC, M, N, K); });
        run_and_record("B naive (coalesced)", 0.25, "col 绑 tx，访存合并",
                       [&] { mm_naive_coalesced<<<grdGood, blk>>>(dA, dB, dC, M, N, K); });
    }

    // =========================================================================
    // 实验 2：TILE 大小扫描 —— 算术强度 = TILE/4，但 TILE 越大不总是越好
    // =========================================================================
    printf("\n--- 实验 2：shared memory 分块，TILE 大小扫描 ------------------\n");
    {
        auto launch_tiled = [&](auto tile_tag) {
            constexpr int T = decltype(tile_tag)::value;
            dim3 blk(T, T);
            dim3 grd((N + T - 1) / T, (M + T - 1) / T);
            char nm[64];
            snprintf(nm, sizeof(nm), "C tiled TILE=%-2d", T);
            char nt[64];
            snprintf(nt, sizeof(nt), "shared %zu KB/block, %d 线程",
                     (size_t)(2 * T * T * sizeof(float)) / 1024, T * T);
            run_and_record(nm, T / 4.0, nt,
                           [&] { mm_tiled<T><<<grd, blk>>>(dA, dB, dC, M, N, K); });
        };
        launch_tiled(std::integral_constant<int, 8>{});
        launch_tiled(std::integral_constant<int, 16>{});
        launch_tiled(std::integral_constant<int, 32>{});
    }

    // =========================================================================
    // 实验 3：寄存器分块阶梯 —— 从 shared 带宽受限爬到接近算力上限
    // =========================================================================
    printf("\n--- 实验 3：寄存器分块阶梯（本文件的核心）---------------------\n");
    {
        const bool ok1d = (M % 64 == 0 && N % 64 == 0 && K % 8 == 0);
        const bool ok2d = (M % 128 == 0 && N % 128 == 0 && K % 8 == 0);
        if (ok1d) {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
            dim3 blk((BM * BN) / TM);
            dim3 grd(N / BN, M / BM);
            run_and_record("D reg1d 64x64, TM=8", (double)BM * BN / (2.0 * (BM + BN)),
                           "每线程 8 个输出",
                           [&] { mm_reg1d<BM, BN, BK, TM><<<grd, blk>>>(dA, dB, dC, M, N, K); });
        } else {
            printf("  [skip] reg1d 需要 M,N%%64==0 且 K%%8==0\n");
        }
        if (ok2d) {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            dim3 blk((BM * BN) / (TM * TN));
            dim3 grd(N / BN, M / BM);
            run_and_record("E reg2d 128x128, 8x8", (double)BM * BN / (2.0 * (BM + BN)),
                           "★ 算术强度跨过 ridge point",
                           [&] { mm_reg2d<BM, BN, BK, TM, TN><<<grd, blk>>>(dA, dB, dC, M, N, K); });
            run_and_record("F reg2d+vec, PAD=0", (double)BM * BN / (2.0 * (BM + BN)),
                           "float4，转置写有 2-way bank 冲突",
                           [&] { mm_reg2d_vec<BM, BN, BK, TM, TN, 0><<<grd, blk>>>(dA, dB, dC, M, N, K); });
            run_and_record("F reg2d+vec, PAD=4", (double)BM * BN / (2.0 * (BM + BN)),
                           "★ padding 消掉 bank 冲突",
                           [&] { mm_reg2d_vec<BM, BN, BK, TM, TN, 4><<<grd, blk>>>(dA, dB, dC, M, N, K); });
        } else {
            printf("  [skip] reg2d 需要 M,N%%128==0 且 K%%8==0\n");
        }
    }

    // =========================================================================
    // 实验 4：工业基准 —— cuBLAS FP32 / cuBLAS TF32 / 裸 Tensor Core
    // =========================================================================
    printf("\n--- 实验 4：工业基准 + Tensor Core ----------------------------\n");
    {
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        run_and_record("G cuBLAS SGEMM (fp32)", -1.0, "工业基准",
                       [&] { cublas_rowmajor_sgemm(handle, dA, dB, dC, M, N, K); });

        // 注意：cublasSgemm 默认【不会】偷偷用 TF32，必须显式开。
        // 这也是一个真实陷阱：拿手写 fp32 kernel 去比一个开了 TF32 的 torch.matmul，
        // 等于拿自行车比汽车。对标前先确认双方精度模式一致。
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
        run_and_record("H cuBLAS SGEMM (TF32)", -1.0, "Tensor Core，尾数只有 10 位",
                       [&] { cublas_rowmajor_sgemm(handle, dA, dB, dC, M, N, K); });
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

        if (M % 16 == 0 && N % 64 == 0 && K % 8 == 0) {
            dim3 blk(32, 4);
            dim3 grd(N / 64, M / 16);
            run_and_record("I wmma TF32, 无 tiling", 0.25,
                           "Tensor Core 但直读 global",
                           [&] { mm_wmma_tf32_naive<<<grd, blk>>>(dA, dB, dC, M, N, K); });
        } else {
            printf("  [skip] wmma demo 需要 M%%16==0, N%%64==0, K%%8==0\n");
        }
    }

    // =========================================================================
    // 实验 5：形状研究 —— 为什么 decode 阶段 tiling 救不了你
    // =========================================================================
    // 这是今天最直接对接推理引擎的一段。M 就是 batch×seq：
    //   prefill：M 很大（几千个 token 一起算）→ 方阵 GEMM → compute-bound
    //   decode ：M = batch（1~几十）        → 瘦长 GEMM/GEMV → memory-bound
    // 同一个算子、同一块卡，两个阶段落在 Roofline 的两侧，优化手段完全不同。
    printf("\n--- 实验 5：形状研究（N=K=%d 固定，扫 M）----------------------\n", N);
    printf("  %6s %10s %12s %10s %10s  %s\n", "M", "ms", "TFLOP/s", "GB/s", "AI", "判定");
    {
        for (int m : {1, 2, 8, 32, 128, 512, 2048}) {
            if (m > M) break;
            const double flop = 2.0 * m * N * K;
            const double bytes = 4.0 * ((double)m * K + (double)K * N + (double)m * N);
            const double ai = flop / bytes;
            const double ms = time_ms(
                [&] { cublas_rowmajor_sgemm(handle, dA, dB, dC, m, N, K); }, WARMUP, ITERS);
            const double tflops = flop / (ms * 1e-3) / 1e12;
            const double gbs = bytes / (ms * 1e-3) / 1e9;
            const char* verdict = (ai < dev.fp32_peak / dev.bw_peak) ? "memory-bound"
                                                                     : "compute-bound";
            printf("  %6d %10.4f %12.2f %10.0f %10.2f  %s\n", m, ms, tflops, gbs, ai,
                   verdict);
        }
    }
    printf("  读法：M 小的时候 TFLOP/s 惨不忍睹，但 GB/s 接近 HBM 峰值 —— 说明卡没偷懒，\n");
    printf("        是这个形状本身没有可复用的数据。这就是 decode 阶段的处境：\n");
    printf("        唯一的解法是 ①提高 batch（continuous batching）②减少权重字节数（量化）。\n");

    // ── 汇总表 ────────────────────────────────────────────────────────
    printf("\n=============================================================\n");
    printf(" 汇总：算术强度阶梯（ridge point = %.1f FLOP/Byte）\n",
           dev.fp32_peak / dev.bw_peak);
    printf("=============================================================\n");
    printf(" %-28s %9s %10s %8s %9s %s\n", "kernel", "ms", "TFLOP/s", "%peak",
           "模型AI", "备注");
    double best = 0.0;
    for (auto& r : rows) best = std::max(best, r.tflops);
    for (auto& r : rows) {
        char ai[16];
        if (r.model_ai < 0) snprintf(ai, sizeof(ai), "%9s", "-");
        else snprintf(ai, sizeof(ai), "%9.2f", r.model_ai);
        printf(" %-28s %9.3f %10.2f %7.1f%% %s %s\n", r.name.c_str(), r.ms, r.tflops,
               100.0 * (r.tflops * 1e12) / dev.fp32_peak, ai, r.note.c_str());
    }
    printf("\n 我的最快 FP32 kernel = %.2f TFLOP/s\n", best);
    for (auto& r : rows) {
        if (r.name.rfind("G cuBLAS", 0) == 0) {
            printf(" cuBLAS SGEMM         = %.2f TFLOP/s  →  差距 %.2fx\n", r.tflops,
                   r.tflops / std::max(1e-9, best));
        }
    }
    printf("\n 差距不是 bug，是四条各自有名字的优化（笔记 §6）：\n");
    printf("   ① warp 级 tile 划分 + 更深的寄存器分块\n");
    printf("   ② double buffering / cp.async（H100 上是 TMA）—— 搬与算重叠\n");
    printf("   ③ Tensor Core（TF32/FP16）—— 换一个更高的计算屋顶\n");
    printf("   ④ shared memory swizzle + L2 感知的 block 调度顺序\n");

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dRef));
    return 0;
}


































