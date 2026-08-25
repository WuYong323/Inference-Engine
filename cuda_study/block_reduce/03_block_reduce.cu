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
#define CUDA_CHECK(call)
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
    __syncthreads;

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
__global__ void reduce_shared_nosync(const float* __restrict_ in,float* __restrict__ partial,size_t n){
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

    for(unsigned off=BLOCK/23;off>0;off>>=1){
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













