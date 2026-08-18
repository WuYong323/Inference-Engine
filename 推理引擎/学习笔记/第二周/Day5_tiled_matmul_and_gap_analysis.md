# 分块矩阵乘（tiled matmul）与 cuBLAS 差距分析

> 阶段一 W1 Day 5-6 · 配套代码 `05/05_tiled_matmul.cu`
> 上游：`04/reduction_cuda_vs_triton.md`（shared memory / bank conflict / 栅栏）
> 下游：W2 CUDA RMSNorm、W3 FlashAttention（分块 + online softmax）

本周最重要的一个 kernel。不是因为它最难，是因为**后面所有东西都是它的放大版**：FlashAttention 的分块、megakernel 里的 GEMM、FFN 的三个矩阵乘，骨架都是今天这一套。tiling 想通了，FlashAttention 的分块看一眼就懂；tiling 没想通，后面会一直在"我知道它在分块，但不知道为什么这么分"的状态里卡着。

---

## §0 电梯答案

先把今天要回答的问题列出来，每个问题后面是答案所在的小节。读完全文之后，你应该能不看笔记把这一列讲一遍。

| # | 问题 | 一句话答案 | 详见 |
|---|---|---|---|
| 1 | 矩阵乘明明是"计算密集"的典型，为什么朴素实现是 memory-bound？ | 计算量 O(n³)、数据量 O(n²)，理论算术强度极高（4096³ 时约 683 FLOP/Byte）；但朴素实现**不复用**任何数据，把强度砸到 0.25，多搬了 2730 倍的字节 | §1 §2.2 |
| 2 | tiling 到底省了什么？ | 省的不是"计算"，是"同一个数从 global 被读的次数"。搬一块进 shared，让它被 block 内所有线程用 TILE 次 | §2.3 §3 |
| 3 | 为什么要**两个** `__syncthreads()`？ | 第一个防"还没搬完就开始算"，第二个防"还没算完就被覆盖"。少了第二个的典型症状是小规模碰巧对、大规模偶发错 | §3.3 |
| 4 | 经典 tiled 已经是教科书答案了，为什么还差 cuBLAS 好几倍？ | 瓶颈换位置了：global 不再是瓶颈，shared 成了瓶颈。经典写法每 1 次 FMA 要发 2 条 shared 读指令，发射槽最多只有 1/3 给到 FMA | §4.1 §4.2 |
| 5 | 为什么工业库的 thread tile 几乎都是 8×8？ | 8×8 让"每 k 步读 16 个数、做 64 次乘加"，shared 读指令 : FMA 指令从 2:1 掉到 0.25:1，配上 `float4` 再掉到 0.0625:1 | §4.2 §5.5 |
| 6 | Day 2 要"聚"（合并访问）、Day 4 要"散"（避 bank 冲突），矛盾吗？ | 不矛盾，是两层内存的不同规则。tiled matmul 是第一个必须**同时**满足两者的 kernel，而且搬运时的索引映射可以和计算时的不一样 | §5.1 §5.4 |
| 7 | 打不过 cuBLAS，差距具体差在哪四件事上？ | ① warp 级 tile + 更深寄存器分块 ② double buffering / `cp.async` / TMA ③ Tensor Core ④ shared swizzle + L2 感知调度 | §6 |
| 8 | Tensor Core 是不是打开就快？ | 不是。它抬高的是"计算屋顶"，tiling 才是爬上去的梯子。没有 tiling 的 WMMA kernel 可能还不如手写 fp32 分块版 | §6.3 |
| 9 | W0 Day5 遗留问题 3：`if (idx < n)` 这种分化在 tiled matmul 里要紧吗？ | 只在**边界 tile** 里真分化，占比 O((M+N)/TILE) / O(MN/TILE²)，可以忽略；而且底层是谓词执行不是跳转。Triton 的 `mask=` 编译下去是同一套东西 | §7.4 |
| 10 | 今天的东西怎么接到推理引擎？ | prefill 是大 M 的方阵 GEMM（compute-bound），decode 是 M=batch 的瘦长 GEMM（memory-bound，AI≈0.5）。同一个算子落在 Roofline 两侧，优化手段完全不同 | §9.2 §9.3 |

---

## §1 问题背景：一个"应该很快"却很慢的算子

### 1.1 先把账算清楚

矩阵乘 `C = A · B`，A 是 M×K，B 是 K×N，C 是 M×N。

**计算量**：C 有 M·N 个元素，每个要做 K 次乘 + K 次加 = 2K 次浮点运算。

```
总计算量 = 2 · M · N · K   FLOP
```

**理论最小访存量**：A、B 各读一次，C 写一次，一个 float 4 字节。

```
理论最小访存 = 4 · (M·K + K·N + M·N)   Byte
```

代入 M=N=K=4096：

| 项 | 数值 |
|---|---|
| 总计算量 | 2 × 4096³ ≈ **1.374 × 10¹¹ FLOP** = 137 GFLOP |
| 理论最小访存 | 4 × 3 × 4096² ≈ **201 MB** |
| 问题本身的算术强度 | 137e9 / 201e6 ≈ **683 FLOP/Byte** |

683 FLOP/Byte 是个什么概念？H100 的 ridge point（脊点，后面 §2.1 讲）大约是 18 FLOP/Byte。**683 远远大于 18，说明矩阵乘这个问题本身极度 compute-bound（计算受限）**，它就该跑在算力峰值附近。

### 1.2 但朴素实现不是

朴素实现长这样（`05_tiled_matmul.cu` 的 kernel (B)）：

```cpp
__global__ void mm_naive_coalesced(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    }
    C[(size_t)row * N + col] = acc;
}
```

每个线程算一个 C 元素，从 global memory 读 2K 个 float。全局看：

```
朴素实现访存量 = M · N · 2K · 4 Byte = 4096² × 8192 × 4 ≈ 5.5 × 10¹¹ Byte = 550 GB
```

**550 GB vs 理论最小 201 MB —— 多搬了 2730 倍。**

按 H100 的 HBM3 标称带宽 3.35 TB/s 算，光搬这 550 GB 就要 164 ms。而 137 GFLOP 的计算在 60 TFLOP/s 下只需要 2.3 ms。**搬运时间是计算时间的 70 倍**，卡的算力单元 98.6% 的时间在等数据。

> 实际测出来会比 164 ms 好一些，因为 L1/L2 cache 会顺手接住一部分复用（同一个 block 里相邻线程读的 A 是同一行）。但那是"碰巧被缓存捡到"，不是你设计出来的。**优化的第一原则是：不要指望缓存替你做本该显式做的事。** 这条在 §2.2 会再说一次。

### 1.3 所以今天的任务是什么

把这 2730 倍的浪费，用**显式的数据复用**压下去。手段就是分块（tiling）：不是让每个线程独自去 global 拿数据，而是让一个 block 的线程**协作**把一小块数据搬到片上，然后在片上反复用。

这句话你可能已经在很多博客里看过。今天要做到的是：**能把"复用几次"和"算术强度变成多少"用纸笔算出来，并且用 ncu 验证算得对。** 会背结论和会推导之间的差距，就是 W1 和 W8 的差距。

---

## §2 算术强度：一把能提前预判性能的尺子

### 2.1 什么是算术强度和 ridge point

**算术强度（Arithmetic Intensity, AI）**：一个 kernel 每从内存搬 1 个字节，能做多少次浮点运算。

```
AI = 总计算量(FLOP) / 总访存量(Byte)      单位 FLOP/Byte
```

**类比**：你在家做饭。冰箱在楼下（global memory），厨房台面在楼上（shared memory / register）。AI 就是"每下一趟楼拿的食材，能做多少道菜"。

- AI 低 = 拿一根葱上楼，切完又下楼拿第二根葱。你的刀（算力）再快也没用，全部时间在爬楼梯 → **memory-bound（访存受限）**
- AI 高 = 一次拿一篮子菜上楼，切半小时。这时候刀快不快才决定速度 → **compute-bound（计算受限）**

**ridge point（脊点）** 就是"刚好不爬楼梯浪费时间"的那个平衡值：

```
ridge point = 峰值算力(FLOP/s) / 峰值带宽(Byte/s)
```

H100 SXM 的标称推导值（**以 `05_tiled_matmul.cu` 在你机器上打印的为准**，下面这些是标称值推的，我没能联网核实，别当成规格书）：

```
FP32 峰值（CUDA core，非 Tensor Core）
  = 2 FLOP/FMA × 128 core/SM × 132 SM × 1.755 GHz ≈ 59.3 TFLOP/s
HBM3 带宽 ≈ 3.35 TB/s
ridge point ≈ 59.3e12 / 3.35e12 ≈ 17.7 ≈ 18 FLOP/Byte
```

代码里这段就是干这个的，它自己读 `cudaDeviceProp` 算，比背数字可靠：

```cpp
// 假设：每个 SM 128 个 FP32 core，每 core 每周期 1 次 FMA = 2 FLOP
// （Ampere/Hopper/Ada 都成立；老架构需要改这个常数）
d.fp32_peak = 2.0 * 128.0 * d.sms * (p.clockRate * 1e3);
// HBM/GDDR 都是 DDR（double data rate），所以乘 2
d.bw_peak = 2.0 * (p.memoryClockRate * 1e3) * (p.memoryBusWidth / 8.0);
```

> ⚠️ 一个真实陷阱：`p.clockRate` 报的是 base clock 还是 boost clock，跨架构不一致；`cudaDeviceProp` 在部分 CUDA 版本上对 HBM 卡的 `memoryClockRate` 会报 0。代码里已经加了 `if (d.bw_peak <= 0) d.bw_peak = 3.35e12;` 的兜底。**所以 ridge point 请当成"±20% 的量级参考"，不要当成精确阈值。** 结论只依赖"18 是十几，不是几百"这个量级判断，不依赖它到底是 17.7 还是 21。

**判定规则**：

```
AI < ridge point  →  memory-bound  →  优化访存（复用、合并、减少字节数）
AI > ridge point  →  compute-bound →  优化计算（用更快的指令、Tensor Core、提高 ILP）
```

这把尺子的价值在于：**它让你在写代码之前就知道该优化什么。** W7 学 Roofline 时你已经见过它，今天是第一次用它来指导设计，而不是事后解释现象。

### 2.2 朴素版的 AI = 0.25

内层循环一次迭代：

```cpp
acc += A[row * K + k] * B[k * N + col];
```

- 计算：1 次 FMA = **2 FLOP**
- 访存：读 2 个 float = **8 Byte**

```
AI_naive = 2 / 8 = 0.25 FLOP/Byte
```

0.25 对 18 —— **差了 71 倍**。这不是"有点慢"，是"卡上 98.6% 的算力在空转"。

反过来推一个有用的上界：memory-bound 的 kernel，速度上限 = 带宽 × AI。

```
朴素版上限 = 3.35 TB/s × 0.25 FLOP/Byte ≈ 0.84 TFLOP/s
59.3 TFLOP/s 的峰值只能用到 1.4%
```

这个 1.4% 你可以在实验 1 的输出里直接对照。**能提前算出上限，再看实测有没有贴着上限，是判断"我是被访存卡住"还是"我另外还写错了什么"的标准方法。**

> **为什么不能把功劳交给 cache？**
> 有人会说：A 的同一行被同一个 block 的多个线程读，L1 会命中啊。会，但有三个问题：
> ① 命中率不可控，取决于 block 大小、调度顺序、别的 kernel 有没有把你的 cache 冲掉；
> ② L1 容量有限（H100 每 SM 256 KB 的 L1/shared 合并空间），4096 长的一行 A 是 16 KB，几十行就满了；
> ③ 就算命中 L1，你付的仍然是"一条访存指令 + 一次 L1 查询"的代价，而 shared memory 是你**明确知道数据在哪、不需要查 tag** 的。
> 工业上的说法是：**cache 是兜底，shared memory 是设计。** 你手动搬进 shared，是把"希望它命中"变成"保证它命中"。

### 2.3 tiled 版的 AI = TILE/4（★ 核心推导）

这是今天最重要的一次推导，纸笔跟着算一遍。

设 block tile 大小是 TILE×TILE，一个 block 负责算 C 里一个 TILE×TILE 的小方块。

C 的这个小方块需要 A 的 TILE 行全部 K 列、B 的 TILE 列全部 K 行。把 K 维切成 `K/TILE` 段，一段一段处理：

```
第 t 步：搬 A 的 (TILE × TILE) 一块 + B 的 (TILE × TILE) 一块 进 shared
        然后算这一段对 C 小方块的贡献
```

**每一步的访存量**：2 个 tile × TILE² 个 float × 4 Byte = `8 · TILE²` Byte
**一共多少步**：`K / TILE`
**这个 block 的总访存量**：`(K/TILE) × 8·TILE² = 8 · K · TILE` Byte

**这个 block 的总计算量**：TILE² 个输出，每个 2K FLOP = `2 · TILE² · K` FLOP

```
        2 · TILE² · K       TILE
AI  =  ───────────────  =  ──────   FLOP/Byte
         8 · K · TILE          4
```

**AI_tiled = TILE / 4**。干净得不像话。

| TILE | shared 用量/block | AI (FLOP/Byte) | vs ridge point ≈ 18 | 带宽给出的速度上限 |
|---|---|---|---|---|
| 8 | 0.5 KB | 2 | 远低于 → memory-bound | 6.7 TFLOP/s |
| 16 | 2 KB | 4 | 低于 → memory-bound | 13.4 TFLOP/s |
| 32 | 8 KB | 8 | 低于 → memory-bound | 26.8 TFLOP/s |

**读出三件事：**

1. **AI 从 0.25 涨到 8，是 32 倍的提升。** 这就是 tiling 的全部功劳，也是为什么它值得叫"本周最重要的 kernel"。
2. **但 TILE=32 的 AI=8 仍然小于 ridge point 18 —— 还是 memory-bound。** 教科书答案不是终点。这个"还不够"，就是 §4 存在的理由，也是绝大多数教程停下来的地方。
3. **TILE 不能无限大。** TILE=64 需要 2×64²×4 = 32 KB shared/block，且 block 要 4096 线程（超过 1024 上限）。**这是硬约束逼出来的转折点：要继续提高 AI，必须让"一个线程算多个输出"，而不是继续放大 block。** 这句话直接推出 §4 的二维寄存器分块。

**换个等价的写法**，后面会一直用：block tile 是 BM×BN、K 方向步长 BK 时，

```
        2 · BM · BN · K              BM · BN
AI  =  ─────────────────────  =  ────────────────
        4 · K · (BM + BN)          2 · (BM + BN)
```

（注意 **BK 被约掉了**，AI 只由 BM、BN 决定。BK 影响的是 shared 用量和流水线深度，不影响算术强度 —— 这是一个很多人搞错的点。）

验算：
- BM=BN=TILE → TILE²/(4·TILE) = TILE/4 ✓ 和上面一致
- BM=BN=64 → 4096/256 = **16**（还是差一点）
- BM=BN=128 → 16384/512 = **32** > 18 ✓ **第一次跨过 ridge point**

128×128 的 block tile 是分水岭。这就是为什么 CUTLASS、cuBLAS 的默认 SGEMM tile 就在 128×128 这一档。

---

## §3 分块的机制：纸上走一遍，再看代码

### 3.1 手算一个 4×4，TILE=2

不要跳过这一步。索引写错是这个 kernel 最常见的 bug，而纸上走一遍是唯一能让索引"长在脑子里"的办法。

M=N=K=4，TILE=2 → grid 是 2×2 个 block，每个 block 4 个线程算 C 的一个 2×2 块。

看 **block (0,0)**，它负责 `C[0..1][0..1]`。

数学上：

```
C[0][0] = A[0][0]B[0][0] + A[0][1]B[1][0] + A[0][2]B[2][0] + A[0][3]B[3][0]
          └────── t=0 这一步能算的 ──────┘ └────── t=1 这一步能算的 ──────┘
```

**关键洞察：K 方向的求和是可以拆开、分批累加的。** 这是 tiling 成立的数学基础 —— 加法满足结合律。（W3 的 FlashAttention 之所以要搞 online softmax，正是因为 softmax **不**满足这种朴素的分批性质，必须额外维护 running max 和 running sum 来修正。今天先记住这个对照。）

**t = 0**：搬 A 的左上 2×2、B 的左上 2×2 进 shared

```
sA = A[0..1][0..1] = ┌ a00 a01 ┐     sB = B[0..1][0..1] = ┌ b00 b01 ┐
                     └ a10 a11 ┘                          └ b10 b11 ┘
```

四个线程各自累加（tx=列, ty=行）：

```
线程(0,0): acc += sA[0][0]*sB[0][0] + sA[0][1]*sB[1][0]   = a00·b00 + a01·b10
线程(1,0): acc += sA[0][0]*sB[0][1] + sA[0][1]*sB[1][1]   = a00·b01 + a01·b11
线程(0,1): acc += sA[1][0]*sB[0][0] + sA[1][1]*sB[1][0]   = a10·b00 + a11·b10
线程(1,1): acc += sA[1][0]*sB[0][1] + sA[1][1]*sB[1][1]   = a10·b01 + a11·b11
```

**数一下 `a00` 被用了几次：2 次**（线程(0,0) 和 线程(1,0)）。但它只从 global 被读了 **1 次**。

这就是复用。TILE=2 时复用 2 次，TILE=32 时复用 32 次。**AI = TILE/4 里那个 TILE，就是这个复用次数。**

**t = 1**：搬 A 的右上 2×2、B 的左下 2×2，继续往同一个 `acc` 上累加。两步之后 `acc` 就是完整的 C 元素。

### 3.2 代码：经典 tiled kernel

```cpp
template <int TILE>
__global__ void mm_tiled(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int tx = threadIdx.x;                     // 0..TILE-1（warp 内变化最快）
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;         // 本线程负责 C 的哪一行
    const int col = blockIdx.x * TILE + tx;         // 本线程负责 C 的哪一列

    float acc = 0.0f;                               // 累加器在 register 里（最快）
    const int nTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < nTiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        // ── 阶段①：搬运。每个线程搬一个元素，全 block 协作搬两个 tile ──
        sA[ty][tx] = (row < M && aCol < K) ? A[(size_t)row * K + aCol] : 0.0f;
        sB[ty][tx] = (bRow < K && col < N) ? B[(size_t)bRow * N + col] : 0.0f;

        __syncthreads();   // ★ 栅栏 1

        // ── 阶段②：计算。这段完全不碰 global memory ──
#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();   // ★ 栅栏 2
    }

    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}
```

**逐条说为什么这么写：**

**① `tx` 出现在两次 load 的"列"位置**

```cpp
sA[ty][tx] = A[row * K + aCol];   // aCol = t*TILE + tx
sB[ty][tx] = B[bRow * N + col];   // col  = blockIdx.x*TILE + tx
```

`threadIdx.x` 是 warp 内变化最快的维度（lane = `threadIdx.x % 32`）。让它落在**行内偏移**上，一个 warp 的 32 个线程读的就是连续地址 → 合并访问。这是 Day 2 的规矩，在这里被复用。

如果你把 `sA[ty][tx] = A[aCol * K + row]` 这样写反了，功能上可能还对（只是转置了），但 32 个线程的地址会相隔 `K*4 = 16 KB`，一个 warp 打出 32 个独立事务，实验 1 里那个"一行之差差好几倍"就是这个现象。

**② 越界填 0，而不是 `return`**

```cpp
sA[ty][tx] = (row < M && aCol < K) ? A[...] : 0.0f;
```

为什么不能像朴素版那样 `if (row >= M) return;`？因为**这个线程即使不负责有效输出，也要参与搬运和 `__syncthreads()`**。让它提前 `return`，剩下的线程会在栅栏上永远等一个不会来的线程 —— 死锁或未定义行为。

> 这是一个非常值得记住的规则：**只要 kernel 里有 `__syncthreads()`，就不能让部分线程提前退出。** CUDA 的 `__syncthreads()` 要求 block 内所有活跃线程都到达。填 0 是标准做法：0 乘任何数是 0，加到 `acc` 上不影响结果 —— 这正是 Day 3 学的"补单位元"技巧（reduction 补 0，max 补 -inf，乘法补 1）。

**③ `acc` 是标量，在寄存器里**

编译器会把 `acc` 分配到寄存器。整个内层 K 循环里，累加器一次都不落到内存。这是 Day 1 那张延迟表的直接应用：register ~1 cycle，shared ~30 cycle，L2 ~200 cycle，HBM ~400+ cycle。**能待在寄存器里的东西，绝不让它下楼。**

**④ `#pragma unroll`**

TILE 是模板参数（编译期常量），所以循环可以完全展开。展开的收益不是"省掉循环判断"这么表面，而是：展开后编译器能看到 32 条互相独立的 FMA，可以重排指令来掩盖 shared memory 的访问延迟（ILP，指令级并行）。**不展开的话，内层循环的索引算术和分支判断会吃掉相当一部分收益** —— 这在坑表 §10 里有一条。

### 3.3 ★ 两道栅栏，各防一个 bug

这是 Day 5 最容易"以为懂了其实没懂"的地方。两个 `__syncthreads()` 防的是**方向相反**的两个竞态。

**栅栏 1：防"读到还没写的数据"（read-before-write）**

```cpp
sA[ty][tx] = ...;   // 线程 T0 写 sA[0][0]
__syncthreads();    // ← 没有这一行会怎样
acc += sA[ty][k] * sB[k][tx];   // 线程 T5 要读 sA[0][0]
```

`sA[0][0]` 是线程 (0,0) 写的，但线程 (5,0) 也要读它。如果没有栅栏，线程 (5,0) 所在的 warp 可能跑得快，在 (0,0) 那个 warp 还没执行到 store 的时候就来读了 → 读到上一轮的旧值或未初始化的垃圾。

**栅栏 2：防"数据还在被读就被覆盖"（write-before-read / WAR）**

```cpp
acc += sA[ty][k] * sB[k][tx];   // 慢的 warp 还在读第 t 块
__syncthreads();                // ← 没有这一行会怎样
// 循环回到开头
sA[ty][tx] = ...;               // 快的 warp 已经在写第 t+1 块了
```

快的 warp 转下一轮，把 `sA` 覆盖成第 t+1 块的数据；慢的 warp 还在算第 t 块，读到的却是 t+1 块的数 → **结果错，而且是数据相关的、非确定性的错**。

**为什么这个 bug 特别阴险（重点）：**

| 规模 | 现象 |
|---|---|
| M=N=K=64，TILE=32 | 只有 2 个 tile，warp 少，调度几乎同步 → **结果正确**，你以为没问题 |
| M=N=K=4096 | 128 个 tile、大量 warp 竞争 SM 资源、调度差异被放大 → **偶发错误，每次跑的错误位置还不一样** |

这就是"小规模碰巧对，大规模偶发错"。你会先怀疑是浮点误差，然后怀疑是 cuBLAS 参考值不对，最后才想到栅栏 —— 中间可能已经浪费两小时。

**工业做法：不靠人眼看，靠工具。**

```bash
compute-sanitizer --tool racecheck ./tmm 512 512 512
```

`racecheck` 专门检测 shared memory 的竞态，删掉栅栏 2 它会直接报出 "Race reported between Write access ... and Read access ..." 并给出源码行号（这就是为什么编译一定要带 `-lineinfo`）。

**改完 kernel 就跑一遍 sanitizer，这是从 Day 3 就该养成的肌肉记忆。** 它比"跑一下看结果对不对"可靠得多，因为它检测的是**是否存在竞态的可能**，不是"这一次有没有出错"。

**三个 sanitizer 各管什么：**

| 工具 | 抓什么 | 典型报错 |
|---|---|---|
| `memcheck` | 越界读写、非法地址、未对齐访问 | `Invalid __shared__ write of size 4 bytes` |
| `racecheck` | shared memory 数据竞态 | `Race reported between Write and Read` |
| `synccheck` | 栅栏使用错误（部分线程到不了） | `Barrier error detected. Divergent thread(s) in warp` |

**顺带一个 `__syncthreads()` 的常见误解**：它同步的是**一个 block**，不是整个 grid。跨 block 没有轻量同步手段，这就是为什么 AMK 那种 megakernel 要费大力气把多个 kernel 融进一个 kernel —— 详见 §9.4。三级同步的成本量级（Day 4 已量过）：

```
warp 内 (__syncwarp)        ~ 几个 cycle，常常是免费的
block 内 (__syncthreads)    ~ 几十个 cycle
grid 级 (kernel 边界 / grid.sync())  ~ 微秒级 —— 比前两者贵 3 个数量级
```

---

## §4 为什么"教科书答案"还差 cuBLAS 好几倍

### 4.1 瓶颈换了位置

TILE=32 的 tiled kernel 把 global 访存降了 32 倍，AI 从 0.25 到 8。global memory 不再是主要瓶颈了。

那新瓶颈在哪？看内层循环：

```cpp
for (int k = 0; k < TILE; ++k) {
    acc += sA[ty][k] * sB[k][tx];   // 2 次 shared 读 → 1 次 FMA
}
```

**每做 1 次 FMA，要发 2 条 shared memory 读指令。**

要把这件事量化，有两种记账方式。它们指向同一个方向，但精确程度不同，**都值得会**：

**记账法 A：按字节（粗，用来估量级）**

每次 FMA 从 shared 取 2 个 float = 8 Byte → 0.125 FMA/Byte。
H100 每 SM 的 shared memory 带宽 = 32 bank × 4 Byte/cycle = **128 Byte/cycle**；FP32 FMA 吞吐 = **128 次/cycle**（128 个 core）。
要喂满算力需要 `128 FMA / 128 Byte = 1 FMA/Byte`。现在只有 0.125 → **理论天花板 12.5% 峰值**。

这个算法的**缺陷**是它忽略了广播：`sA[ty][k]` 在一个 warp 内所有 lane 读的是**同一个地址**，硬件一次广播就服务 32 个 lane，并没有真的搬 32×4 字节。所以 12.5% 是个偏悲观的估计。

**记账法 B：按指令（准，而且 ncu 能直接验）**

数**指令条数**，因为一个 SM sub-partition 每 cycle 只能发射 1 条指令 —— shared 读指令和 FMA 指令**抢同一个发射槽**。

| kernel | 每 k 步 per warp 的 shared 读指令 | 每 k 步 per warp 的 FFMA 指令 | LDS : FFMA | FFMA 能占到的发射槽 |
|---|---|---|---|---|
| (C) 经典 tiled | 2 | 1 | **2 : 1** | 1/(1+2) = **33%** |
| (D) 一维分块 TM=8 | 9（1 个 B + 8 个 A） | 8 | 1.125 : 1 | 8/(8+9) = **47%** |
| (E) 二维分块 8×8 | 16（8 个 A + 8 个 B） | 64 | **0.25 : 1** | 64/(64+16) = **80%** |
| (F) 二维分块 + float4 | 4（4 条 LDS.128） | 64 | **0.0625 : 1** | 64/(64+4) = **94%** |

**这张表是 §4 的全部内容。** 记账法 B 的好处是每一行都可以用 ncu 的 `smsp__inst_executed_op_shared_ld.sum` 除以 FFMA 数验证 —— §7 会教怎么验。

结论：**经典 tiled kernel 的天花板在 33% 左右（记账法 A 给的 12.5% 是下界，真实值在两者之间）。** 就算 global 访存完全免费，它也上不去。所以差 cuBLAS 好几倍是**结构性的**，不是"调调参数就好"。

> 这里有个思维方式值得提炼：**优化是"搬瓶颈"，不是"变快"。** 每解决一个瓶颈，下一个瓶颈就浮出来。global → shared → 发射槽 → 计算屋顶。W7 学 Roofline 时这叫"沿着屋顶往右走"，今天是第一次亲手走两步。

### 4.2 ★ 二维寄存器分块：为什么工业库都选 8×8

**核心手法**：让一个线程负责 **TM×TN** 个输出，而不是 1 个。

**类比先行**。经典 tiled 像这样干活：你（一个线程）从工作台（shared）拿一个零件 A、一个零件 B，拧一颗螺丝（1 次 FMA），然后再去拿下两个。跑腿次数和干活次数一样多。

二维寄存器分块：你一次从工作台抓 **8 个 A 零件 + 8 个 B 零件**，摊在手边（寄存器），然后**两两配对拧 64 颗螺丝**。跑腿 16 次，干活 64 次。

**这就是关键的不对称：读 (TM+TN) 个数，能做 TM×TN 次乘加。** 一个是加法，一个是乘法 —— tile 越大，比值越好。

```
每 k 步：读 (TM + TN) 个 float，做 TM × TN 次 FMA
复用率 = TM·TN / (TM+TN)
```

| TM×TN | 每 k 步读 | 每 k 步 FMA | 复用率 | 每线程 acc 寄存器 |
|---|---|---|---|---|
| 1×1（经典） | 2 | 1 | 0.5 | 1 |
| 4×4 | 8 | 16 | 2.0 | 16 |
| **8×8** | **16** | **64** | **4.0** | **64** |
| 16×16 | 32 | 256 | 8.0 | 256 ✗ 超过 255 寄存器上限 |

**为什么停在 8×8 而不是继续放大？** 两个上限同时逼近：

1. **寄存器上限**：每线程最多 255 个寄存器。8×8 已经用掉 `acc[8][8]` = 64 个，加上 `regM[8]`、`regN[8]` = 16 个，加上指针和索引，实测大约 **90~110 个寄存器/线程**（`-Xptxas -v` 会告诉你确切数字）。16×16 需要 256 个 acc，直接爆表 → 编译器把数组挪到 **local memory**，而 local memory 物理上就在**显存**里 → 性能悬崖。
2. **收益递减**：从 1×1 到 8×8，LDS:FFMA 从 2:1 降到 0.25:1，FFMA 发射槽占比从 33% 升到 80%。再往上从 80% 到 89%，只剩 9 个百分点可捞，代价却是寄存器翻 4 倍、occupancy 腰斩。

**8×8 是这两条曲线的交点。** 这就是为什么你在 CUTLASS、cuBLAS、triton 的 autotune 配置里到处看到 8×8 和 8×4 —— 不是玄学，是算出来的。

**同时 block tile 也放大了**，两个收益一起拿：

```
BM=BN=128, TM=TN=8 → 线程数 = 128×128/64 = 256 线程/block
global AI = 128×128 / (2×256) = 32 FLOP/Byte  >  ridge point ≈ 18  ✓
```

**这是今天第一次真正跨进 compute-bound 区域。** 一步同时解决了两层瓶颈：global 层（AI 32 > 18）和发射槽层（80%）。

**代码**（kernel (E)，`05_tiled_matmul.cu`）：

```cpp
template <int BM, int BN, int BK, int TM, int TN>
__global__ void mm_reg2d(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
    constexpr int THREADS = (BM * BN) / (TM * TN);

    __shared__ float sA[BK][BM];      // ← 转置存：sA[k][m]，理由见下
    __shared__ float sB[BK][BN];      // ← 正常存：sB[k][n]

    A += (size_t)blockIdx.y * BM * K;             // 指针先偏到本 block 的地盘
    B += (size_t)blockIdx.x * BN;
    C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;

    // 计算分工：thread tile 网格是 (BM/TM) × (BN/TN)
    const int tRow = threadIdx.x / (BN / TN);
    const int tCol = threadIdx.x % (BN / TN);

    // 搬运分工（和计算分工不一样！见下面第 ② 点）
    const int irA = threadIdx.x / BK, icA = threadIdx.x % BK;
    const int irB = threadIdx.x / BN, icB = threadIdx.x % BN;
    constexpr int strideA = THREADS / BK;
    constexpr int strideB = THREADS / BN;

    float acc[TM][TN] = {};           // 64 个寄存器，整个 K 循环都不落内存
    float regM[TM], regN[TN];

    for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
        for (int off = 0; off < BM; off += strideA)
            sA[icA][irA + off] = A[(size_t)(irA + off) * K + icA];   // 转置写入
#pragma unroll
        for (int off = 0; off < BK; off += strideB)
            sB[irB + off][icB] = B[(size_t)(irB + off) * N + icB];
        __syncthreads();

        A += BK;                      // 指针滑动，比每次重算 index 省寄存器和指令
        B += (size_t)BK * N;

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            // ① 先把这一列 A / 这一行 B 拉进寄存器（每个只读一次）
#pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = sA[k][tRow * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = sB[k][tCol * TN + j];
            // ② 然后在寄存器里做 TM*TN 次 FMA —— 16 次读换 64 次算
            //    这 64 条 FFMA 互相独立 → ILP 拉满，掩盖 FMA 延迟
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    // 写回略
}
```

**三个必须理解的设计决策：**

**① 为什么 A 要转置存进 shared（`sA[BK][BM]` 而不是 `sA[BM][BK]`）**

内层要读 `regM[i] = sA[k][tRow*TM + i]`，也就是固定 k、连续的 m。

- 存成 `sA[BM][BK]`（正常）→ 读 `sA[m][k]`，i 变化时地址跨 `BK` 个 float = 32 Byte → 不连续，**没法用 `float4` 一次取 4 个**
- 存成 `sA[BK][BM]`（转置）→ 读 `sA[k][m]`，i 变化时地址连续 → **可以 `float4`**（kernel (F) 就靠这个）

代价是**写入时**变成跨行写（`sA[icA][irA]`，icA 变化跨 BM 个 float），这会引入 bank 冲突 —— §5.4 会手算它（是 8-way，不是很多博客里说的 2-way）并用 padding 治好。

**这是一个经典的权衡：把不规则的代价放在"搬运一次"上，换来"计算 BK 次"的规整。** 因为计算发生的次数远多于搬运，这笔交易划算。工业库全都这么干。

**② 为什么搬运的索引映射和计算的索引映射故意不一样**

```cpp
const int tRow = threadIdx.x / (BN / TN);   // 计算时的身份
const int irA  = threadIdx.x / BK;          // 搬运时的身份
```

同一个线程，在两个阶段扮演不同角色。这不是代码写乱了，是**两个阶段的最优目标不同**：

| 阶段 | 追求什么 | 对索引的要求 |
|---|---|---|
| 搬运 | 从 global 合并访问 | `threadIdx.x` 要落在行内偏移上 |
| 计算 | 寄存器复用最大化 | `threadIdx.x` 要按 thread tile 网格切 |

强行让两者用同一套索引，必然牺牲一边。**"一个线程在不同阶段有不同身份"是所有高性能分块 kernel 的共同特征**，第一次看会觉得乱，看懂之后会觉得理所当然。

**③ 为什么 64 条 FFMA 要写成两层展开的形式**

```cpp
for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) acc[i][j] += regM[i] * regN[j];
```

全部展开后是 64 条 `FFMA acc[i][j], regM[i], regN[j], acc[i][j]`。关键是**这 64 条互相没有数据依赖**（写的是 64 个不同的累加器）。

FMA 指令有几个 cycle 的延迟。如果 64 条指令首尾相接（比如都累加到同一个变量），每条都要等前一条的结果，延迟完全暴露。**互相独立意味着硬件可以流水发射，用后面的指令填前面的延迟空隙。** 这就是 ILP（Instruction-Level Parallelism，指令级并行）。

> **这解释了一个反直觉的现象**：这个 kernel 的 occupancy（占用率）很低。算一下：约 100 寄存器/线程 × 256 线程 = 25600 寄存器/block，H100 每 SM 65536 个寄存器 → 只能放 2 个 block = 512 线程，而 SM 上限 2048 线程 → **occupancy ≈ 25%**。
>
> Day 1 学的"提高 occupancy 来掩盖延迟"在这里**不适用**。因为掩盖延迟有两条路：
> - **TLP**（线程级并行）：warp 多，一个 warp 卡住就换另一个 → 需要高 occupancy
> - **ILP**（指令级并行）：单个 warp 内就有一堆独立指令可以流水 → 不需要高 occupancy
>
> 高性能 GEMM 走的是 ILP 路线。**看到 25% occupancy 不要慌，那是这个 kernel 的正常状态。** 判断标准永远是 `sm__throughput` 和 `dram__throughput` 这两个吞吐指标，不是 occupancy 本身。occupancy 是手段，吞吐才是目的 —— 这是新手最容易搞反的一件事，也是面试常问点。

---

## §5 两层内存的两套规矩：global 要"聚"，shared 要"散"

### 5.1 为什么规则是相反的

Day 2 学合并访问（coalesced access）：一个 warp 的 32 个线程访问 global 时，地址越集中越好，最好挤在同一个 128 Byte 段里 → 一次事务搞定。

Day 4 学 bank 冲突（bank conflict）：一个 warp 访问 shared 时，地址越分散越好，最好落在 32 个不同的 bank 上 → 一个 cycle 全部服务完。

**看起来矛盾，其实是两种硬件结构的必然结果：**

| | global memory | shared memory |
|---|---|---|
| 物理结构 | DRAM，按 **段**（sector，32 B）搬运 | 32 个 **bank**，每个 bank 独立、每 cycle 出 4 B |
| 代价来自 | 搬了不用的字节（浪费带宽） | 同一 bank 被多个 lane 要求不同地址（串行化） |
| 所以要 | **聚** —— 挤进最少的段 | **散** —— 铺开到最多的 bank |

**类比**：global 是"从仓库调货"，一次派一辆卡车拉一整箱，你要的东西越集中在一箱里越省车次。shared 是"32 个窗口的柜台"，32 个人要越分散到不同窗口越快，全挤一个窗口就得排队。

两句话记住：**卡车怕散，柜台怕挤。**

**bank 编号公式**（32 个 bank，每 bank 宽 4 Byte）：

```
bank_id = (字节地址 / 4) % 32 = (float 数组下标) % 32
```

**一个必须记住的例外：广播（broadcast）。** 如果一个 warp 内多个 lane 读的是**同一个地址**，硬件一次广播全部服务完，**不算冲突**。冲突的定义是"同一个 bank 的**不同**地址"。这个例外在 §5.3 直接救了经典 tiled kernel。

### 5.2 tiled matmul 是第一个必须同时满足两者的 kernel

前四天的 kernel 只需要顾一头：`01` 的 vector add 只有 global，`03`/`04` 的 reduction 主要顾 shared。tiled matmul 有明确的两个阶段，**搬运阶段考 global 合并，计算阶段考 shared 无冲突**，两关都要过。

这也是为什么它是"本周最重要的 kernel"—— 它是第一个把前面所有零散知识**同时**用上的地方。

### 5.3 经典 tiled kernel 的 bank 分析：恰好无冲突

```cpp
acc += sA[ty][k] * sB[k][tx];    // TILE = 32
```

一个 warp 的 32 个线程，`blockDim = (32, 32)` → **同一个 warp 内 `ty` 相同、`tx` = 0..31**（因为 `threadIdx.x` 变化最快，一个 warp 正好是一行）。

**读 `sA[ty][k]`：** ty 和 k 在这条指令里对全 warp 都是同一个值 → 32 个 lane 读**同一个地址** → **广播，1 个 cycle** ✓

**读 `sB[k][tx]`：** 下标 = `k*32 + tx`

```
bank = (k*32 + tx) % 32 = (0 + tx) % 32 = tx        ← k*32 是 32 的整数倍，模掉了
tx = 0..31 → bank = 0..31，32 个不同 bank
```

**无冲突** ✓

所以最教科书的写法**恰好**是 bank-conflict-free 的。

> **但"恰好"三个字要划重点。** 把 `sB[k][tx]` 改成 `sB[tx][k]`（比如你想"优化"成让 A 转置），就变成 bank = `(tx*32 + k) % 32 = k` —— **全 warp 撞同一个 bank，32-way 冲突，这条指令慢 32 倍。**
>
> 一个下标顺序的调换，性能差 32 倍。这就是为什么 shared memory 的每一次索引改动都要重新手算一遍 bank。**bank 冲突不是"编译器会帮你处理"的东西，它 100% 由你写的下标决定。**

**验证方法**（Day 4 已经用过）：

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./tmm 4096 4096 4096 3
```

kernel (C) 这一项应该接近 **0**。如果不是 0，你的索引和你以为的不一样。

### 5.4 ★ 转置写入的 bank 冲突：手算 + padding 修复

现在看 kernel (E) 的转置写入，这是今天唯一一处真有冲突的地方。

```cpp
__shared__ float sA[BK][BM];              // BK=8, BM=128
const int irA = threadIdx.x / BK;         // BK=8  → irA = tid/8
const int icA = threadIdx.x % BK;         // icA = tid%8
sA[icA][irA + off] = A[(irA + off) * K + icA];
```

**先看 global 读（必须先确认这头没坏）：**

```
地址 = (irA+off)*K + icA
tid=0..7  → irA=0, icA=0..7  → 读 A 第 0 行的第 0..7 个 → 连续 8 个 float = 32 B ✓
tid=8..15 → irA=1, icA=0..7  → 读 A 第 1 行的第 0..7 个 → 另一段 32 B
```

一个 warp（32 lane）覆盖 4 行 × 每行 32 B = **4 个 sector**。不是完美的单段 128 B，但每个 sector 都被完整用掉，**没有浪费字节** —— 这是可接受的合并度。（BK=8 决定了这个形状；BK 更大会更好，但 shared 用量和寄存器压力会上去。）

**再看 shared 写（问题在这）：**

```
sA[icA][irA]  的一维下标 = icA * BM + irA = icA * 128 + irA
bank = (icA * 128 + irA) % 32
```

`128 % 32 = 0`，所以 `icA * 128` 整个被模掉：

```
bank = irA % 32
```

一个 warp 内 tid = 0..31 → `irA = tid/8` = 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2×8, 3×8

```
lane  0-7  : irA=0 → bank 0    ← 8 个 lane 挤 bank 0，但地址各不相同！
lane  8-15 : irA=1 → bank 1
lane 16-23 : irA=2 → bank 2
lane 24-31 : irA=3 → bank 3
```

**只用到 4 个 bank，每个 bank 上有 8 个 lane 要写 8 个不同的地址 → 8-way bank conflict。** 这条 store 指令要 8 个 cycle 而不是 1 个。

> **这里纠正一处：`05_tiled_matmul.cu` 里 (F) 的注释写的是"2-way 冲突"，那个数字对应的是 (F) 的 float4 版本（每线程搬 4 个、warp 覆盖更多行），(E) 的标量版本按上面这个算法是 8-way。** 手算结果以本节为准，ncu 会给出最终裁决 —— §7.3 就是去验这个数的。这也正好演示了一件事：**bank 冲突的路数完全取决于"一个 warp 内 irA 有几个不同取值"，改一下搬运分工，路数就变了。** 不能背，只能算。

**padding 修复。** 把声明改成 `sA[BK][BM + PAD]`，行长从 128 变成 132：

```
bank = (icA * 132 + irA) % 32
132 % 32 = 4          ← 不再是 0！这是关键
bank = (icA * 4 + irA) % 32
```

重新列表（tid = 0..31，`irA = tid/8`，`icA = tid%8`）：

| lane | irA | icA | bank = (icA·4 + irA) % 32 |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 4 |
| 2 | 0 | 2 | 8 |
| 3 | 0 | 3 | 12 |
| 4 | 0 | 4 | 16 |
| 5 | 0 | 5 | 20 |
| 6 | 0 | 6 | 24 |
| 7 | 0 | 7 | 28 |
| 8 | 1 | 0 | 1 |
| 9 | 1 | 1 | 5 |
| … | … | … | … |
| 31 | 3 | 7 | 31 |

通式：`bank = 4·icA + irA`，其中 `icA ∈ [0,7]`、`irA ∈ [0,3]`。`4·icA` 取 {0,4,8,...,28}，`irA` 取 {0,1,2,3} 填满每个间隔 → **0..31 全覆盖，一一对应，冲突归零** ✓

**为什么 PAD 选 4 而不是 1？**

PAD=1 也能消冲突（`129 % 32 = 1` → `bank = icA + irA`，也够散）。但 kernel (F) 要用 `float4` 从 shared 读：

```cpp
const float4 a0 = *reinterpret_cast<const float4*>(&sA[k][tRow * TM]);
```

**`float4` 要求 16 Byte 对齐，否则直接 `misaligned address` 崩掉。**

```
PAD=1 → 行长 129 float = 516 Byte，516 % 16 = 4  ✗ 每换一行就错位 4 字节
PAD=4 → 行长 132 float = 528 Byte，528 % 16 = 0  ✓ 每行起点都对齐
```

代码里这个 `static_assert` 就是把这条规则钉死，让违反它变成**编译错误**而不是运行时崩溃：

```cpp
static_assert(PAD % 4 == 0, "PAD 必须是 4 的倍数，否则 float4 读取不对齐");
```

> **工业规范**：约束能在编译期检查就别留到运行期。`static_assert` 零成本、报错信息可读、改配置时立刻拦住你。CUTLASS 里满屏都是这个。

**代价核算**：`sA` 从 `8×128×4 = 4 KB` 变成 `8×132×4 = 4.125 KB`，多 128 Byte。**用 3% 的 shared 空间换掉 8 倍的 store 延迟。**

这也正是 Day 4 笔记 §7.6 预告过的手法（那里用的例子是 `__shared__ float tileB[32][33]`，bank = `(k*33+tx)%32 = (k+tx)%32`）。**同一个技巧，第二次见面，这次是在真实 kernel 里自己算出来的。**

### 5.5 一处我**没有**修的冲突：sB 侧的读

诚实交代，因为 ncu 会把它抖出来，而且它正是 §6.4 那条优化存在的理由。

```cpp
regN[j] = sB[k][tCol * TN + j];       // BN=128, TN=8, tCol = tid % 16
```

`sB` 声明是 `sB[BK][BN]` = `sB[8][128]`，**没加 padding**。一维下标 = `k*128 + tCol*8 + j`：

```
bank = (k*128 + tCol*8 + j) % 32 = (tCol*8 + j) % 32        (128 % 32 = 0)
```

一个 warp 内 tid = 0..31 → `tCol = tid % 16` = 0..15,0..15。固定 j：

```
tCol =  0 → bank = j
tCol =  1 → bank = 8+j
tCol =  2 → bank = 16+j
tCol =  3 → bank = 24+j
tCol =  4 → bank = (32+j)%32 = j      ← 和 tCol=0 撞了
```

`tCol*8 % 32` 只有 **4 个取值** {0,8,16,24} → 32 个 lane 挤在 4 个 bank 上（每 bank 8 个 lane，地址不同）→ **8-way 冲突**。

**为什么 padding 治不了它？** 因为冲突不是来自"行长是 32 的倍数"（那是 §5.4 的病因），而是来自 **`tCol` 本身乘了 8**。行长加 PAD 只改 `k*BN` 那一项，而那一项已经被模掉了。**病因不同，药也不同。**

真正的解法是 **swizzle（异或置换）**：给地址再做一次 `^` 变换，把挤在一起的 lane 打散。

```cpp
// 思路示意（不是可直接编译的完整实现）：
// 原始     sB[k][c]
// swizzle  sB[k][c ^ ((k & 3) << 3)]
// 每个 k 把列偏移异或一个不同的量，同一个 bank 上的 lane 被换到不同 bank。
// 读写两侧必须用同一个变换，否则数据对不上。
```

XOR 的好处是**它是自己的逆**（`(c^x)^x = c`），不需要额外的表，也不浪费 shared 空间（padding 要浪费）。代价是索引算术变复杂、可读性变差。

**所以差距表 §6.4 里"shared memory swizzle"这一条，在我们自己的代码里就有一个活标本。** 这不是我偷懒留的坑，而是这一层优化本来就属于 Day 6 之后的内容：先看到它、量到它、知道它叫什么名字，比现在就写对更重要。Day 6 的目标是**认识差距**，不是消灭差距。

---

## §6 ★ 对标 cuBLAS：差距的四个名字

Day 6 的核心产出。先把预期立住：**打不过 cuBLAS 是预期结果，重点是理解不是超越。** 这句话是你自己在暑假计划的风险表里写的，今天正好兑现。

典型量级（你机器上跑出来的数字为准）：

| 实现 | 大致水平 | 相对 cuBLAS |
|---|---|---|
| (A) 朴素未合并 | ~0.2 TFLOP/s | 200× 慢 |
| (B) 朴素合并 | ~0.8 TFLOP/s | 50× 慢 |
| (C) 经典 tiled TILE=32 | ~5 TFLOP/s | 8× 慢 |
| (E)(F) 二维寄存器分块 | ~25-35 TFLOP/s | **1.5-2× 慢** |
| (G) cuBLAS SGEMM fp32 | ~45-55 TFLOP/s | 1× |
| (H) cuBLAS SGEMM TF32 | ~200-400 TFLOP/s | 另一个量级 |

**从 (A) 到 (F) 你自己走了 100 多倍。** 剩下的 1.5-2× 是下面四件事。

### 6.1 ① warp 级 tile 划分 + 更深的寄存器分块

我们的 (E)/(F) 只有两级：block tile（128×128）→ thread tile（8×8）。

工业库有**三级**：block tile → **warp tile** → thread tile。

```
CUTLASS 的层次（cuBLAS 内部同构）：
  ThreadBlock tile   128×128×8     ← 决定 global AI（我们有）
    Warp tile         64× 32×8     ← 决定 shared 读的组织方式（我们没有）
      Thread tile      8×  8        ← 决定寄存器复用（我们有）
```

**warp tile 为什么重要？** 一个 warp 的 32 个 lane 如果在 shared 里读的是**规整的一片矩形**，就可以用 `ldmatrix`（Ampere+ 的指令，一条指令为整个 warp 加载一个矩阵片段）；我们现在是 256 个线程各按 `threadIdx.x` 算自己的位置，warp 内的访问模式是散的，用不上这类指令，也更容易撞 bank（§5.5 就是这个后果）。

**这一级也是 Tensor Core 的入场券**：`wmma`/`mma` 指令的操作单位就是 warp，没有 warp tile 这一层就没法自然地接上 Tensor Core。

### 6.2 ② double buffering / `cp.async` / TMA —— 让搬运和计算重叠

**我们现在的时间线（同步、串行）：**

```
搬第0块 ──栅栏── 算第0块 ──栅栏── 搬第1块 ──栅栏── 算第1块 ──栅栏── ...
   ↑ 算力单元闲着      ↑ 访存单元闲着
```

两个单元轮流干活、轮流闲着。搬运占的那段时间是**纯损失**。

**double buffering（双缓冲）**：开两块 shared buffer，算 buffer0 的同时搬 buffer1。

```cpp
__shared__ float sA[2][BK][BM+PAD];      // 两份
__shared__ float sB[2][BK][BN];

load(sA[0], sB[0], 0);                    // 预取第 0 块
__syncthreads();
for (int t = 0; t < nTiles; ++t) {
    int cur = t & 1, nxt = cur ^ 1;
    if (t + 1 < nTiles) load_async(sA[nxt], sB[nxt], t + 1);   // 发出搬运，不等
    compute(sA[cur], sB[cur]);                                 // 同时算当前块
    wait_async();                                              // 等搬运到位
    __syncthreads();
}
```

**类比**：单缓冲是"洗一个菜、切一个菜、再洗下一个"；双缓冲是"切这个菜的同时，水龙头在洗下一个"。厨师的手（算力）不再有空档。

**关键是"发出搬运但不等它"。** 三代硬件支持：

| 手法 | 架构 | 做法 | 代价 |
|---|---|---|---|
| 手工双缓冲 | 全部 | 先 `LDG` 到寄存器，算完再 `STS` 进 shared | 占大量寄存器，挤压 occupancy |
| **`cp.async`** | Ampere (sm_80+) | global → shared **直通**，不过寄存器；`cp.async.commit_group` / `wait_group` 控制等待 | 不占寄存器 ✓ |
| **TMA** | Hopper (sm_90+) | Tensor Memory Accelerator，一条指令描述整个多维 tile 的搬运，硬件 DMA 引擎异步执行 | 几乎不占指令发射槽 ✓✓ |

**H100 上就是 TMA。** 这也是 Hopper 上 FlashAttention-3 相对 FA-2 提速的主要来源之一（另一个是 warp specialization：让一部分 warp 专门搬运、一部分专门计算，生产者-消费者模式）。

**收益量级**：搬运和计算完全重叠的话，理想情况能省掉搬运那部分时间。对我们的 (F) 来说这大约是 20-30% 的提升 —— 是这四条里**性价比最高**的一条，也是你自己动手最可能拿到的一条。

> **调试提示**：`cp.async` 的等待逻辑写错了，症状是**偶发的结果错误**，和忘记栅栏一模一样。同样用 `compute-sanitizer --tool racecheck` 抓。

### 6.3 ★ ③ Tensor Core：更高的屋顶，但你得自己爬上去

**Tensor Core** 是专门做小矩阵乘加的硬件单元：一条指令完成 `D = A·B + C`，其中 A、B 是 16×16 之类的小块。相比 CUDA core 一条指令做一次标量 FMA，吞吐高一个数量级。

**TF32（TensorFloat-32）** 是喂给它的一种格式：

| 格式 | 指数位 | 尾数位 | 总位数 | 相对精度 |
|---|---|---|---|---|
| FP32 | 8 | 23 | 32 | ~1e-7 |
| **TF32** | **8** | **10** | 19（存储占 32） | **~1e-3** |
| FP16 | 5 | 10 | 16 | ~1e-3，且**动态范围小**，易溢出 |
| BF16 | 8 | 7 | 16 | ~1e-2，动态范围同 FP32 |

**TF32 的设计意图**：指数位和 FP32 一样是 8 位 → **动态范围完全相同，不会因为数值太大太小而溢出**；只砍尾数 → 精度降到 ~1e-3。对深度学习来说，梯度和激活值本来就有噪声，1e-3 通常够用。**"保范围、砍精度"是这一代混合精度格式的共同思路**（BF16 也是同一个哲学）。

**关键实验：kernel (I) 证明 Tensor Core 不是开关。**

```cpp
// 一个 warp 负责一个 16×16 的 C 块，直接从 global 读 A/B —— 故意不做 tiling
for (int k = 0; k < K; k += 8) {
    wmma::load_matrix_sync(fa, A + warpM * 16 * K + k, K);
    wmma::load_matrix_sync(fb, B + k * N + warpN * 16, N);
    // TF32 fragment 必须显式舍入（CUDA 的要求，不是可选优化）
    for (int i = 0; i < fa.num_elements; ++i) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
    for (int i = 0; i < fb.num_elements; ++i) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
    wmma::mma_sync(fc, fa, fb, fc);
}
```

它的算术强度和朴素版一个量级（每个 16×16 块的数据都从 global 重新读）→ **算力屋顶抬高了 10 倍，但它被访存地板锁在下面，可能还不如我们手写的 fp32 (F)。**

> **这就是 §6.3 要立的论点：Tensor Core 提供的是更高的计算屋顶，tiling 提供的是爬上去的梯子。没有梯子，屋顶多高都没用。**
>
> 推论：**今天学的 tiling 不会因为"以后都用 Tensor Core"而过时，恰恰相反，它是用好 Tensor Core 的前提。** 这是今天最该带走的一句话。

**两个真实陷阱：**

**陷阱 1：TF32 fragment 必须显式舍入。** 上面那两个 `__float_to_tf32` 循环不是可选优化，是 CUDA 编程模型的要求。漏掉的话结果错（喂进 Tensor Core 的位模式不对），而且不报错。

**陷阱 2（更常踩）：对标时精度模式必须一致。**

```cpp
CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));       // fp32
run("G cuBLAS SGEMM (fp32)", ...);
CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)); // TF32
run("H cuBLAS SGEMM (TF32)", ...);
```

`cublasSgemm` 默认**不会**偷偷用 TF32，要显式开。但 PyTorch 那边情况不同 —— 而且这是个**版本相关**的坑：

- `torch.backends.cudnn.allow_tf32` 长期默认 **True**（卷积走 TF32）
- `torch.backends.cuda.matmul.allow_tf32` 默认 **False**（矩阵乘不走 TF32）
- 较新版本还引入了 `torch.backends.cuda.matmul.fp32_precision` / `torch.set_float32_matmul_precision("highest"|"high"|"medium")` 这套更细的开关

> ⚠️ 我没能联网核实你当前 PyTorch 版本的确切默认值（域名被网络策略拦了）。**别信任何博客上的记忆值，包括这份笔记 —— 直接在你的环境里打印：**
>
> ```python
> import torch
> print(torch.__version__)
> print("matmul.allow_tf32 =", torch.backends.cuda.matmul.allow_tf32)
> print("cudnn.allow_tf32  =", torch.backends.cudnn.allow_tf32)
> print("float32_matmul_precision =", torch.get_float32_matmul_precision())
> ```

**为什么这条值得单独强调**：拿你手写的 fp32 kernel 去比一个开了 TF32 的 `torch.matmul`，等于**拿自行车比汽车**。你会得出"我慢 20 倍"的错误结论，然后花一周去优化一个本来就不该那么比的东西。

**对标三条纪律（写死在流程里）：**
1. **精度模式对齐** —— 双方都是 fp32，或都是 TF32
2. **计时方式对齐** —— 都用 `cudaEvent`，都预热、都同步、都取多次平均
3. **规模对齐** —— 同一个 M/N/K，别拿 512 的结果去推 4096 的结论

### 6.4 ④ shared memory swizzle + L2 感知的 block 调度

**swizzle** 在 §5.5 讲过了：用 XOR 置换消掉 padding 治不了的那类冲突，我们代码里 `sB` 侧就有一个活标本。

**L2 感知的 block 调度**是另一件事，讲一下因为它非常"工业"、而且几乎零成本。

默认的 block 编号是行主序展开的：`blockIdx.x` 先变。4096/128 = 32×32 = 1024 个 block。

```
默认顺序：block 0,1,2,...,31 是 C 的第一"行" block
→ 它们共用 A 的同一块行条带（好），但要读 B 的全部 32 个列条带（差）
→ 同时活跃的 block 加起来要碰的 B 数据量很大 → L2 装不下 → 反复从 HBM 读
```

**Thread Block Swizzle / grouped ordering**：把 block 重排成一块一块的方形分组（比如 8×8 一组），让同时在跑的 block 尽量共享 A 和 B 的条带。

```cpp
// 思路示意：把线性 blockIdx 重映射成分组后的 (bx, by)
const int GROUP = 8;
int bid = blockIdx.x;                          // grid 用一维启动
int blocks_per_group = GROUP * gridDim_n;      // gridDim_n = N/BN
int group_id  = bid / blocks_per_group;
int inner     = bid % blocks_per_group;
int by = group_id * GROUP + inner % GROUP;     // 先填满 GROUP 行
int bx = inner / GROUP;
```

**收益**：纯改索引，不改任何算法，L2 命中率能明显提升，大规模 GEMM 上拿到 5-15% 不稀奇。Triton 的官方 matmul 教程里那段 `pid_m`/`pid_n` 重排（`GROUP_SIZE_M`）就是这个 —— **W8 你写 Triton 时会再见到它，届时你已经知道它为什么存在。**

**验证指标**：

```bash
ncu --metrics lts__t_sector_hit_rate.pct ./tmm 4096 4096 4096 3
```

### 6.5 差距表（Day 6 的正式产出）

| # | 优化 | 治什么瓶颈 | 预期收益 | 我们做了吗 | 难度 |
|---|---|---|---|---|---|
| 0 | shared 分块 | global 访存量（AI 0.25→8） | **30×+** | ✓ (C) | ★★ |
| 0.5 | 二维寄存器分块 | 发射槽 + global AI（→32） | **5×+** | ✓ (E) | ★★★ |
| 0.6 | `float4` 向量化 | 指令发射数 | 10-20% | ✓ (F) | ★★ |
| 0.7 | shared padding | 转置写的 8-way 冲突 | 5-15% | ✓ (F, PAD=4) | ★★ |
| ① | warp 级 tile | shared 访问组织 + Tensor Core 入口 | 10-20% | ✗ | ★★★★ |
| ② | double buffering / TMA | 搬运与计算不重叠 | **20-30%** | ✗ | ★★★★ |
| ③ | Tensor Core | 计算屋顶太低 | **5-10×**（换精度） | 半个 (I) 反面例子 | ★★★★ |
| ④ | swizzle + L2 调度 | `sB` 侧冲突 + L2 命中率 | 5-15% | ✗（§5.5 标本） | ★★★★★ |

**读法**：0 和 0.5 两级拿走了大头（150 倍量级），①②④ 三条合起来是剩下的 1.5-2×，③ 是换赛道。**这个分布本身就是一条经验：优化的前 20% 工作量拿走 80% 收益，剩下 20% 收益要花 80% 工作量。** 知道自己站在曲线的哪一段，比闷头优化重要。

---

## §7 实测与 profiling：让数字裁决手算

### 7.1 计时的三条铁律

```cpp
template <typename Fn>
static double time_ms(Fn&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();        // 铁律1：预热
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(beg));
    for (int i = 0; i < iters; ++i) fn();          // 铁律3：多次取平均
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));         // 铁律2：同步
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
    // ...
    return (double)ms / iters;
}
```

| 铁律 | 为什么 | 不做的后果 |
|---|---|---|
| **预热** | 首次调用含 JIT/模块加载/显存首次触碰/时钟从省电档爬升 | 第一次的数字可能慢 2-10 倍 |
| **同步** | kernel launch 是**异步**的，CPU 秒表只量到"提交" | 测出接近 0 的荒谬数字 |
| **多次平均** | 单次受调度抖动、别的进程干扰影响 | 波动大到无法比较 |

**为什么必须用 `cudaEvent` 而不是 `std::chrono`**：event 记录在 **GPU 的流**上，量的是 GPU 上真实的执行区间。CPU 计时器量的是 CPU 这边的墙上时间，中间隔着异步队列。

> 还有一个隐蔽点：`cudaEventRecord(beg)` 之前那个 `cudaDeviceSynchronize()` 不能省。否则 warmup 的 kernel 可能还在跑，`beg` 事件排在它们后面，测出来的时间把 warmup 的尾巴也算进去了。

### 7.2 正确性怎么验（比性能更重要）

**跑得快的错答案毫无价值。** 代码里用了两层验证：

**第一层：和 cuBLAS 逐元素比**

```cpp
static double max_rel_err_vs(const std::vector<float>& got,
                             const std::vector<float>& ref) {
    double maxAbs = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        maxAbs = std::max(maxAbs, (double)std::fabs(got[i] - ref[i]));
        maxRef = std::max(maxRef, (double)std::fabs(ref[i]));
    }
    return maxAbs / std::max(1e-8, maxRef);
}
```

用 `max|diff| / max|ref|` 而不是逐元素相对误差 —— 因为 C 里某些元素可能接近 0（正负项抵消），逐元素相对误差会被这些点炸成天文数字，**是个假警报**。这是数值比较的标准做法。

**第二层：用 CPU `double` 抽样验证 cuBLAS 本身**

```cpp
std::vector<std::pair<int,int>> pts;
pts.push_back({0, 0});
pts.push_back({0, N - 1});
pts.push_back({M - 1, 0});
pts.push_back({M - 1, N - 1});          // ★ 四个角必须验
for (int s = pts.size(); s < nSamples; ++s) pts.push_back({di(rng), dj(rng)});
```

**为什么必须包含四个角**：分块 kernel 的 bug 极大概率出在**边界 tile**（越界判断写错、最后一块没处理干净）。纯随机抽样 64 个点，落在最后一行/最后一列的概率极低 → **漏掉最典型的 bug**。

**为什么用 `double` 重算**：fp32 累加 4096 项本身有误差，用 fp32 算参考值就没有"参考"的意义了。

**误差量级的判断标准**：

```
fp32 的 machine epsilon ≈ 1.2e-7
K 项朴素顺序累加的误差 ~ sqrt(K) · eps ≈ 64 × 1.2e-7 ≈ 8e-6
```

| max rel err | 判断 |
|---|---|
| < 1e-5 | ✓ 正常的 fp32 累加误差 |
| 1e-3 量级 | TF32/FP16 的正常水平；如果你以为在跑 fp32，说明**精度模式搞错了** |
| 1e-2 以上 或 nan | ✗ 真 bug：索引错、栅栏错、边界错 |

> **一个容易误判的现象**：如果 (E)/(F) 的误差**比** (C) **小**，不是 bug。因为寄存器分块把 K 维累加拆成了多个部分和（每个 acc 累加 K 次，但 tile 内部顺序不同），**分批累加的数值稳定性通常比长串顺序累加更好**。cuBLAS 更是刻意用 split-K 和树形累加来控误差。**"精度不同"和"算错了"要分清。**

### 7.3 ncu 四组指标，每组回答一个问题

```bash
# ① 我在 Roofline 上的哪里？
ncu --kernel-name-base demangled \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_read.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct ./tmm 4096 4096 4096 3
```

**读法（这是 profiling 的核心判断）：**

| `sm__throughput` | `dram__throughput` | 诊断 | 下一步 |
|---|---|---|---|
| 低 | **高** | memory-bound | 提高 AI：加大 tile、复用 |
| **高** | 低 | compute-bound | 换更快指令（Tensor Core）、提高 ILP |
| 低 | 低 | **延迟受限或并行度不足** | 查 occupancy、查依赖链、查栅栏 |
| 高 | 高 | 接近极限 | 收工 |

**"两个都低"是最需要警惕的情况**：说明卡在等，既不是算力不够也不是带宽不够。经典原因是同步太多、依赖链太长、或者 block 数不够填满 SM。

**预期**：(C) 落在第一行（dram 高），(F) 落在第二行（sm 高）。**这两行就是 §2 那个"AI 从 8 到 32 跨过 ridge point"的实测证据。手算和实测对上了，这一天才算真的学会。**

```bash
# ② shared memory 两笔账：访问次数 + bank 冲突（验证 §5.4 / §5.5 的手算）
ncu --metrics smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./tmm 4096 4096 4096 3
```

**这组是今天最该认真看的。三个预期，逐个核对手算：**

1. **(C) 的 `bank_conflicts` ≈ 0** → 验证 §5.3（经典写法恰好无冲突）
2. **`PAD=0` 的 `..._op_st` 冲突数 >> `PAD=4` 的** → 验证 §5.4（padding 治好了转置写）
3. **`PAD=4` 的 `..._op_ld` 仍然不为 0** → 验证 §5.5（`sB` 侧的 8-way 我没修，padding 也治不了）

**如果第 3 条你测出来是 0**，说明我的手算或编译器的实际行为和我想的不一样（比如编译器把 `sB` 的读优化成了别的形式）—— **那就以 ncu 为准，并且回头重算一遍索引。这才是 profiling 的意义：它是裁判，不是装饰。**

`smsp__inst_executed_op_shared_ld.sum` 还能直接验 §4.1 那张 LDS:FFMA 表：(F) 用了 `float4`，这个数应该约是 (E) 的 1/4。

```bash
# ③ 占用率 + 分化（连回 Day 1）
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem ./tmm 4096 4096 4096 3
```

- `sm__warps_active...`：真实 occupancy。(F) 大约 25%，**这是正常的**（§4.2 讲过原因）
- `thread_inst_executed_per_inst_executed.ratio`：**平均每条指令有几个线程真在干活，满分 32**。这是分化的直接度量
- 后两个：告诉你 occupancy 到底是被**寄存器**还是被 **shared** 限住的 —— 决定你该调哪个参数

```bash
# ④ 有没有真用上 Tensor Core
ncu --metrics sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active ./tmm 4096 4096 4096 3
```

(A)-(G) 应该是 **0**（纯 CUDA core），(H)(I) 明显非 0。**这是区分"我以为用了 Tensor Core"和"真的用了"的唯一可靠办法** —— 很多人开了 TF32 开关但因为对齐/形状不满足，实际 fallback 回了 CUDA core，只有这个指标能抓到。

### 7.4 ★ 兑现 W0 Day5 遗留问题 3：分化在 tiled matmul 里要紧吗

**原问题**：`if (idx < n)` 这种边界判断造成的 warp divergence（warp 分化），在 tiled matmul 里影响大吗？

**答案：几乎不影响。三层理由，从粗到细。**

**第一层：算占比。**

分化只发生在**边界 tile**（不能被 TILE 整除的最后一行/最后一列）。

```
总 block 数        = (M/TILE) × (N/TILE)        = O(MN/TILE²)
含边界的 block 数  = (M/TILE) + (N/TILE)        = O((M+N)/TILE)
占比 = (M+N)/TILE / (MN/TILE²) = TILE(M+N)/(MN)
```

M=N=4096, TILE=32：`32 × 8192 / 4096² ≈ 1.6%`。而且 M/N 是 32 的倍数时，**占比恰好是 0**。

**Amdahl 定律的直接应用**（Day 4 已经吃过一次教训）：**优化的价值 = 优化的部分 × 它的占比。** 1.6% 的部分就算完全消除也只省 1.6%，不值得为它把代码写复杂。

**第二层：底层根本不是"跳转"。** 看 PTX：

```ptx
// sA[ty][tx] = (row < M && aCol < K) ? A[row*K + aCol] : 0.0f;
setp.lt.s32   %p1, %r_row, %r_M;        // p1 = (row < M)
setp.lt.s32   %p2, %r_aCol, %r_K;       // p2 = (aCol < K)
and.pred      %p3, %p1, %p2;            // p3 = p1 && p2
mov.f32       %f1, 0f00000000;          // f1 = 0.0
@%p3 ld.global.nc.f32  %f1, [%rd_addr]; // ★ 谓词执行：p3 为真才真的 load
st.shared.f32 [%rd_s], %f1;
```

**`@%p3` 是谓词（predicate）执行**：指令照样发射，但谓词为假的 lane 不产生效果。**没有分支跳转，没有两条路径串行执行。**

对应的 SASS（Hopper 上大致形态）：

```
      ISETP.GE.AND  P0, PT, R_row, R_M, PT
      ISETP.GE.OR   P0, PT, R_aCol, R_K, P0
      MOV           R4, RZ
 @!P0 LDG.E         R4, [R2.64]          ← 谓词化的 load
      STS           [R6], R4
```

**为什么这很重要**：真正昂贵的 divergence 是**两条路径都有大量指令、必须串行跑完**（比如 `if` 里 100 条、`else` 里 100 条 → 一个 warp 花 200 条的时间）。而 `a ? b : 0` 这种，编译器一律用谓词执行，代价是**几条额外的 `setp` 指令**，不是路径串行。

**第三层：不做边界判断反而更贵。** 备选方案是把矩阵 pad 到 TILE 的倍数 —— 那要多分配显存、多一次 memset、多读多写。**为 1.6% 的分化付 100% 的额外访存，明显不划算。**

**连到 Triton（W8 会用到）**：

```python
mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
a = tl.load(a_ptrs, mask=mask, other=0.0)
```

Triton 的 `mask=` + `other=0.0`，编译下去就是上面那套 `setp` + `@%p` 谓词化 load + 填 0。**语法糖不同，机器码同源。** 这也是为什么先手写 CUDA 再学 Triton 值得 —— **你会知道每个 Triton 参数底下对应哪条 PTX，而不是把它当黑盒调参。**

**总结成一条可迁移的判断法**：看到分化先别急着优化，先问两件事 —— **① 它占多大比例（Amdahl）② 它是谓词执行还是真跳转（看 PTX/SASS）。** 两个答案决定值不值得动手。这条方法论比"tiled matmul 的分化不要紧"这个具体结论有用得多。

---

## §8 工业实践：真实的库长什么样

### 8.1 cuBLAS / CUTLASS 的分层结构

我们的 (F) 和工业库的差别不在"有没有用某个技巧"，而在**分层的完整度**。CUTLASS（NVIDIA 开源的 GEMM 模板库，cuBLAS 内部是同构的）的层次：

```
Device       ── 选 kernel、切 split-K、算 grid
  Kernel     ── block tile 循环、L2 感知的 block 重排（§6.4）
    Mainloop ── double buffering / cp.async / TMA（§6.2）
      Warp   ── warp tile、ldmatrix、mma（§6.1）
        Thread ── thread tile 的寄存器 FMA（我们有 ✓）
    Epilogue ── 写回 + 融合的后处理（见下）
```

**Epilogue（尾声）这一层值得单独说**，因为它和推理引擎直接相关。GEMM 算完 `acc` 之后不急着写回，而是在寄存器里顺手做完后续操作：

```
C = act(alpha · A·B + beta · C + bias)
```

`bias` 加法、激活函数（ReLU/GELU/SiLU）、残差加法、量化缩放，全都在 `acc` 还在寄存器里的时候做掉。**省掉的是一整轮"写回 global + 再读回来"** —— 对 memory-bound 的逐元素算子来说，这等于把它的成本降到接近 0。

这就是 **kernel fusion（算子融合）**，也是推理引擎最主要的优化手段之一。你在 §9.2 会亲手用到。

### 8.2 aligned fast path vs generic kernel

我们的 (E)(F) 有硬约束：

```cpp
static_assert(BM * BK / 4 == THREADS, "A tile: 每线程恰好一个 float4");
// 运行时：
const bool ok2d = (M % 128 == 0 && N % 128 == 0 && K % 8 == 0);
if (ok2d) { ... } else { printf("  [skip] reg2d 需要 M,N%%128==0 且 K%%8==0\n"); }
```

**这不是偷懒，这就是工业做法。** cuBLAS 内部有几十甚至上百个 kernel 变体，运行时按 (M, N, K, 数据类型, 对齐情况, 架构) 挑一个：

- 规模对齐、够大 → 走 `float4`/TMA 的**快路径**
- 规模零碎 → 走带完整边界处理的 **generic kernel**（慢一些但通用）
- M 很小 → 走 **GEMV 专用 kernel**（完全不同的并行策略，见 §9.3）
- K 特别大而 M/N 小 → 走 **split-K**（把 K 维切开给多个 block，最后再规约）

**这解释了一个你迟早会遇到的现象**：cuBLAS 在 4096×4096×4096 上飞快，换成 4097×4097×4097 可能掉 30% —— 因为掉到了 generic 路径。**"性能随规模不连续跳变"是分块 kernel 的固有特征，不是 bug。** 这也是为什么模型里的维度总喜欢取 128 的倍数（hidden_size=4096、8192 之类），不是审美，是为了踩在快路径上。

> **给你的实践建议**：写自己的 kernel 时，与其追求一个 kernel 通吃所有规模，不如**写一个对齐快路径 + 兜底调用 cuBLAS**。这是工程上最优的选择，也是所有推理引擎实际在做的事（vLLM/TensorRT-LLM 都大量直接调 cuBLAS/CUTLASS，只在有融合收益的地方才手写）。

### 8.3 cuBLAS 的行主序陷阱

**Day 6 最容易白扔一小时的地方**，值得单独一节。

cuBLAS 是**列主序**（column-major，Fortran 传统：一列连续存放）。C/C++ 的二维数组是**行主序**（row-major：一行连续存放）。

**关键洞察（想通这一句就不用记公式）：**

> **一块行主序的 M×N 数据，被列主序的眼睛看过去，就是 N×M 的转置。**

内存里 `[a00 a01 a02 a10 a11 a12]`：
- 行主序的眼睛：2 行 3 列，`A = [[a00,a01,a02],[a10,a11,a12]]`
- 列主序的眼睛：3 行 2 列，第一列是 `a00,a01,a02` → 这正是 `Aᵀ`

所以，我们想算 `C = A·B`（都是行主序）。利用转置恒等式：

```
Cᵀ = (A·B)ᵀ = Bᵀ · Aᵀ
```

把 B、A 的指针按**列主序**传给 cuBLAS（列主序看到的就是 `Bᵀ`、`Aᵀ`），让它算 `Bᵀ·Aᵀ = Cᵀ`。结果 `Cᵀ` 是列主序的 N×M，**再用行主序的眼睛读回来，就是 M×N 的 C** —— 正是我们要的，而且**一次转置都没有真的发生**。

```cpp
static void cublas_rowmajor_sgemm(cublasHandle_t h, const float* dA, const float* dB,
                                  float* dC, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,     // ← 注意 m,n 互换了
                             &alpha,
                             dB, N,       // 列主序看到的是 B^T (N×K)，lda = N
                             dA, K,       // 列主序看到的是 A^T (K×M)，ldb = K
                             &beta,
                             dC, N));     // 结果 C^T (N×M)，ldc = N
}
```

**记法**：**"要行主序的 A·B，就把 (B, A) 按 (N, M, K) 传进去。"** `OP_N` 两个都不用转置。

> **踩坑的典型症状**：结果不是完全错，而是**看起来像转置了**、或者只有对角线附近对。看到这个症状第一反应就该是主序问题。**另外，`M == N` 时这个 bug 会被掩盖一半**（形状对得上，只是数值转置了），所以调试时故意用 `M != N`（比如 512×256×128）能更快暴露问题。

### 8.4 编译选项的工业惯例

```bash
nvcc -O3 -arch=sm_90 -lineinfo -Xptxas -v -o tmm 05_tiled_matmul.cu -lcublas
```

| 选项 | 作用 | 为什么必须有 |
|---|---|---|
| `-O3` | 优化级别 | 少了它循环不展开，性能结论全废 |
| `-arch=sm_90` | 目标架构（Hopper） | 用 `sm_90` 才能生成 TMA/新指令；写 `sm_70` 会退化 |
| **`-lineinfo`** | 保留行号信息 | **ncu 和 sanitizer 的报错能指到源码行**，没它只能看 SASS 地址 |
| **`-Xptxas -v`** | 打印寄存器/shared 用量 | 见下，这是每次改完 kernel 必看的体检报告 |

`-Xptxas -v` 的输出长这样：

```
ptxas info : Compiling entry function '_Z9mm_reg2dILi128ELi128ELi8ELi8ELi8EEvPKfS1_Pfiii' for 'sm_90'
ptxas info : Used 104 registers, 8448 bytes smem, 384 bytes cmem[0]
```

**三个数各自的意义：**

- **`104 registers`** → 算 occupancy：`65536 / 104 ≈ 630` 线程/SM 的寄存器预算... 实际按 block 粒度算：256 线程 × 104 = 26624，`65536/26624 = 2` 个 block/SM
- **`8448 bytes smem`** → `sA[8][132] + sB[8][128]` = `(1056 + 1024) × 4 = 8320`，加上对齐凑到 8448 ✓ **和手算对上说明 PAD 生效了**
- **`spill stores` / `spill loads`** → **如果出现这两行，立刻停下来处理。** 意思是寄存器不够，编译器把变量挪到了 local memory（物理上在**显存**里）。内层循环里一次 spill 就能吃掉一半性能

> **工业规范**：CI 里把 `-Xptxas -v` 的输出解析出来，**寄存器数超过阈值或出现 spill 就让构建失败**。这是 kernel 库的标准防回归手段。

### 8.5 sanitizer 是护栏，不是可选项

```bash
compute-sanitizer --tool memcheck   ./tmm 512 512 512
compute-sanitizer --tool racecheck  ./tmm 512 512 512
compute-sanitizer --tool synccheck  ./tmm 512 512 512
```

**用小规模跑**（512 而不是 4096）—— sanitizer 会让程序慢 10-100 倍，4096 要等到你怀疑机器死了。

**为什么小规模也有效**：sanitizer 检测的是**"存在竞态的可能"**和**"这次访问是否越界"**，不依赖"错误是否真的发生"。这正是它比"跑一遍看结果对不对"强的地方 —— §3.3 那个"小规模碰巧对"的 bug，racecheck 在 512 规模就能抓到。

**改完 kernel 就跑，形成肌肉记忆。** 这三条命令的总耗时不到一分钟，能省下的调试时间是以小时计的。

---

## §9 落到我自己的两条线上

### 9.1 【造】推理引擎：FFN 的演进路线

Transformer 的 FFN（Feed-Forward Network，前馈网络）就是连着的两三个矩阵乘。LLaMA 系的 SwiGLU 版本（W0 已经实现过）：

```python
# hidden = 4096, intermediate = 11008 (LLaMA-7B)
def ffn(x):                          # x: [batch·seq, 4096]
    gate = x @ W_gate                # [T, 4096] @ [4096, 11008]
    up   = x @ W_up                  # [T, 4096] @ [4096, 11008]
    h    = silu(gate) * up           # 逐元素
    return h @ W_down                # [T, 11008] @ [11008, 4096]
```

**这三个 matmul 就是今天写的东西的放大版。** 具体的对应关系：

| 今天的概念 | FFN 里的对应 |
|---|---|
| M | `batch × seq_len`（prefill 时几千，decode 时等于 batch） |
| K, N | `hidden_size`, `intermediate_size`（4096 / 11008，都是好对齐的数） |
| block tile 128×128 | cuBLAS 内部实际用的 tile |
| §8.1 的 epilogue 融合 | `silu(gate) * up` 应该融进 gate 的 GEMM 尾声 |

**今天该做的三件具体事（Day 6 的【造】）：**

**① 在 `ARCHITECTURE.md` 里写下 FFN 的演进路线**（不写代码，只定路线）：

```markdown
## FFN 的四步演进（W1 Day6 定，逐周兑现）
v0 (W0 已完成)  torch.matmul × 3 + 逐元素 silu/mul     ← 正确性基线
v1 (W2)         gate/up 两个 GEMM 合并成一个            ← 少一次 kernel launch、
                W_gate 和 W_up 拼成 [4096, 22016]        A 只读一次
v2 (W3)         silu(gate)*up 融进 GEMM 的 epilogue     ← 省一轮 global 往返
v3 (W5+)        权重 INT8/INT4 量化                     ← decode 阶段唯一的真解法（§9.3）
```

**为什么这个顺序**：先拿正确性基线，再做不改数值语义的合并（v1），再做融合（v2，要小心数值），最后才动精度（v3，要评估质量）。**风险从低到高，每一步都可回退。这是工程纪律，不是保守。**

**② 用 ncu profile 引擎里现有的 `torch.matmul`**，把它填进一张表：

```bash
ncu --kernel-name-base demangled --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active \
python -m minillm.bench --stage prefill
```

要记录的：**PyTorch 选了哪个 cuBLAS kernel（名字里有 tile 尺寸）、有没有走 Tensor Core、落在 Roofline 哪一侧。**

**③ 记下 prefill 和 decode 两组数字**，这是 W4 做 continuous batching 的基线。没有基线的优化是没法验证的。

### 9.2 实验 5 的读法：为什么 decode 阶段 tiling 救不了你

代码里的实验 5 固定 N=K=4096，扫 M：

```
   M         ms      TFLOP/s       GB/s         AI  判定
   1     0.0xxx        ~0.03      ~3200       0.50  memory-bound
   2                   ~0.07                  1.00  memory-bound
   8                   ~0.27                  3.99  memory-bound
  32                   ~1.05                 15.4   memory-bound
 128                   ~3.9                  56.9   compute-bound
 512                  ~20                   186     compute-bound
2048                  ~45                   585     compute-bound
```

（数值形态示意，实际以你机器上的输出为准。）

**M=1 时的 AI 手算**：

```
FLOP  = 2 · 1 · 4096 · 4096 = 3.36e7
Byte  = 4 · (1·4096 + 4096·4096 + 1·4096) = 4 · 16.78e6 ≈ 6.71e7
AI    = 3.36e7 / 6.71e7 ≈ 0.5 FLOP/Byte
```

**0.5！比朴素 matmul 的 0.25 只好一倍，比 ridge point 18 差 36 倍。**

**关键的读法（这一段是今天最该带走的工程洞察）：**

> **M=1 时 TFLOP/s 惨不忍睹（可能只有峰值的 0.05%），但 GB/s 接近 HBM 峰值。**
>
> **这说明卡没有偷懒 —— 它已经在以最快速度搬数据了。是这个形状本身没有可复用的数据。**

为什么没得复用？M=1 时，权重矩阵 B 的每个元素**只被用一次**（乘上 x 的一个元素）。tiling 的全部前提是"搬进来的数据能被多个线程复用多次"，而这里**根本不存在复用机会**。这不是实现问题，是**问题本身的数学性质**。

**这正是 LLM 推理 decode 阶段的处境：**

| | prefill（首 token） | decode（后续每个 token） |
|---|---|---|
| M = batch × seq | 几千（整个 prompt 一起算） | **= batch（1~几十）** |
| 形状 | 接近方阵 GEMM | 瘦长 GEMM / GEMV |
| AI | 几百 → **compute-bound** | **~0.5 → memory-bound** |
| 瓶颈 | 算力 | **HBM 带宽（搬权重）** |
| 有效优化 | tiling、Tensor Core、融合 | **提高 batch、减少权重字节数** |
| 无效优化 | — | **tiling（没得复用）、Tensor Core（喂不满）** |

**同一个算子、同一块卡、同一份权重，两个阶段落在 Roofline 的两侧，优化手段几乎不重叠。**

**所以 decode 只有两条真解法：**

**① 提高 batch —— continuous batching（连续批处理）**
把多个请求的 decode 拼在一起，M 从 1 变成 32/64。权重读一次，服务 32 个请求 → **AI 直接乘 32**。看上面的表：M=32 时 AI=15.4，已经接近 ridge point；M=128 时就跨过去了。
**这就是 vLLM 最核心的收益来源。** 它不是靠把 kernel 写得更快，而是靠**改变形状**让 kernel 有机会跑快。W4 你会自己实现它。

**② 减少权重字节数 —— 量化**
AI = FLOP/Byte，分母里绝大部分是权重字节。INT8 权重 → 字节减半 → **AI 翻倍、搬运时间减半**。INT4 → 再翻倍。
注意这里的逻辑：**量化在 decode 阶段的收益主要来自"少搬字节"，不是"算得快"。** 很多人以为量化是为了用更快的整数运算单元，在 decode 场景这是次要的。这个区分很重要，它决定了你该优化哪一头。

> **这一段的价值**：它把"矩阵乘优化"从一个 CUDA 练习变成了**引擎架构决策的依据**。你现在能回答"为什么 vLLM 要做 continuous batching""为什么推理要量化而训练可以不量化"这类问题，而且是从 Roofline 推出来的，不是背来的。**这是面试里能明显区分层次的一类回答。**

### 9.3 【研】AMK / 小米项目：今天的三个可用动作

对接你手上的 AutoMegaKernel（AMK）在 H100 上的 profiling 工作：

**① 看 AMK 的 GEMM 落在 Roofline 哪里**
用 §7.3 的第 ① 组指标跑 AMK 的 kernel。**如果 `dram__throughput` 高而 `sm__throughput` 低，说明它的 tile 配置在 H100 上偏小** —— H100 的算力/带宽比（ridge point ≈ 18）和它原本调优的目标卡（A100 ridge point ≈ 19.5/1.56 ≈ 12.5）不同，**在 A100 上刚好跨过 ridge point 的 tile 配置，搬到 H100 上可能就掉回 memory-bound 那一侧了。**

> 这是你"H100 是 AMK 反主场"这个判断的**定量表述**。原来只是直觉，现在可以拿 ridge point 的数字说话。

**② 查它有没有用 TMA / double buffering**
看 SASS 里有没有 `UTMALDG`（TMA 的 load 指令）或 `LDGSTS`（`cp.async`）：

```bash
cuobjdump -sass amk_kernel.cubin | grep -E "UTMALDG|LDGSTS|BAR"
```

只有 `LDG` + `STS` + `BAR.SYNC` 的规整交替 → **是同步的单缓冲，§6.2 那 20-30% 还在桌上。** 这是一个具体、可量化、可交付的发现。

**③ 查 Tensor Core 利用率**
用 §7.3 第 ④ 组。`sm__inst_executed_pipe_tensor` 明显偏低而 kernel 本该是 GEMM → 要么没走 Tensor Core，要么被访存卡得喂不满。**两种情况的处方完全不同**（前者查对齐和形状约束，后者查 tile 配置），这个指标能帮你分辨。

**关于 megakernel 和今天内容的关系**（值得写进你的科研笔记）：

AMK 这类 megakernel 的思路是把多个 kernel 融成一个，省掉 kernel launch 和中间结果的 global 往返 —— 本质上是**把 §8.1 的 epilogue 融合推到极致**。

但它有一个今天可以理解的根本代价：**融进一个 kernel 之后，原本"每个 kernel 各自选最优 tile 配置"的自由度就没了。** 一个 kernel 里所有 GEMM 共享同一套 block 形状和 shared 预算，而不同形状的 GEMM 最优 tile 是不同的（§8.2 讲了 cuBLAS 为此准备了几十个变体）。

**所以 megakernel 的收益是"省 launch 和往返"，代价是"tile 配置被迫妥协"。** 收益是否大于代价，取决于具体模型的算子构成 —— 这正是"自动"（Auto）那部分要解决的问题。

> **这个视角能让你在组会上问出有价值的问题**：AMK 在做 tile 配置搜索时，是按"全局统一"还是"分段允许不同"？它的搜索空间里有没有把 H100 的 ridge point 作为约束？

**还有一个跨 block 同步的连接**（§3.3 提过）：megakernel 把多个 kernel 融合后，原本靠 kernel 边界实现的全局同步，必须改用 grid 级同步（`cooperative_groups::grid_group::sync()`）或 Hopper 的 **thread block cluster + `cluster.sync()`**。

```
kernel 边界同步：微秒级，但顺带把 L2 也刷掉了
grid.sync()    ：微秒级，但数据能留在 L2 ✓ 这是 megakernel 的收益来源之一
cluster.sync() ：Hopper 新增，同一 cluster 内的 block 可同步 + 互访 shared
                （distributed shared memory），比 grid.sync 便宜得多
```

**H100 的 cluster 是 AMK 可能还没吃到的红利**，因为它是 sm_90 独有的。这是"H100 独特价值"的另一个具体抓手。

---

## §10 坑表：按踩到的概率排序

| # | 坑 | 症状 | 原因 | 处方 |
|---|---|---|---|---|
| 1 | 忘记**第二个** `__syncthreads()` | 小规模对、大规模偶发错，每次错的位置不同 | 快 warp 覆盖了慢 warp 还在读的 shared（WAR 竞态） | `racecheck`；§3.3 |
| 2 | cuBLAS **行主序/列主序**搞混 | 结果像被转置了；或 M=N 时只有数值不对 | cuBLAS 是列主序 | 用 §8.3 的 `(B, A, N, M, K)` 写法；调试用 M≠N |
| 3 | 边界线程提前 `return` | **死锁**或未定义行为 | 有 `__syncthreads()` 时不能让部分线程退出 | 越界填 0，不 return；§3.2 |
| 4 | 对标时**精度模式不一致** | "我慢 20 倍"的错误结论 | `torch.matmul` 可能走 TF32，你的是 fp32 | 打印 `allow_tf32`；§6.3 |
| 5 | 忘了 `-O3` 或 `#pragma unroll` | 比预期慢 2-3 倍，找不到原因 | 内层循环没展开，索引算术和分支吃掉收益 | 检查编译命令；看 SASS 有没有展开 |
| 6 | 没预热就计时 | 第一个 kernel 特别慢，结论混乱 | JIT、模块加载、时钟爬升 | `time_ms` 的 warmup；§7.1 |
| 7 | 用 CPU 秒表计时 | 时间接近 0，或数字荒谬 | kernel launch 是异步的 | `cudaEvent` + `cudaEventSynchronize` |
| 8 | shared 下标顺序改动引入 **32-way 冲突** | 慢 10-30 倍 | `sB[k][tx]` 改成 `sB[tx][k]` → 全 warp 同一个 bank | 每次改索引都手算 bank；§5.3 |
| 9 | `float4` 未对齐 | `misaligned address` 直接崩 | PAD 不是 4 的倍数，或起始地址没 16B 对齐 | `static_assert(PAD % 4 == 0)`；§5.4 |
| 10 | 对局部数组 `reinterpret_cast` 取 `float4` | 性能悬崖，`-Xptxas -v` 出现 spill | 取地址迫使数组落进 **local memory（=显存）** | 逐成员拷贝，别取地址；见 (F) 注释 |
| 11 | 看到 occupancy 25% 就去"优化" | 越改越慢 | 高性能 GEMM 靠 ILP 不靠 TLP | 看吞吐指标，不看 occupancy；§4.2 |
| 12 | 逐元素算相对误差 | 报告出 1e5 量级的"误差" | C 里有接近 0 的元素（正负抵消）做了分母 | `max|diff|/max|ref|`；§7.2 |
| 13 | 用 fp32 在 CPU 上算参考值 | 分不清是自己错还是精度差 | fp32 累加 4096 项本身误差就有 1e-5 | CPU 用 `double`；§7.2 |
| 14 | 抽样验证只用随机点 | 边界 tile 的 bug 漏检 | 随机 64 点几乎不会落在最后一行/列 | **四个角必验**；§7.2 |
| 15 | 以为 TF32 开了就用上了 | 性能没变化，以为 Tensor Core 没用 | 对齐/形状不满足，静默 fallback 回 CUDA core | 查 `sm__inst_executed_pipe_tensor`；§7.3 |
| 16 | 在 4096 规模跑 sanitizer | 以为机器死了 | sanitizer 慢 10-100 倍 | 用 512；§8.5 |
| 17 | `-Xptxas -v` 出现 spill 没在意 | 性能只有预期一半 | 寄存器不够，变量落进显存 | 减小 tile 或 `__launch_bounds__` |
| 18 | 只在一个规模测就下结论 | 换规模后结论翻转 | cuBLAS 会按规模切 kernel（§8.2） | 至少测 512/2048/4096 三档 |

**前 4 条是"必踩"级别的**，建议现在就在代码里搜一遍确认自己没犯。

---

## §11 交叉索引：今天和前后的关系

### 11.1 上游（今天用到了之前学的什么）

| 来源 | 概念 | 今天用在哪 |
|---|---|---|
| Day 1 | warp / SIMT / `threadIdx.x` 变化最快 | §3.2 为什么 `tx` 要放列位置 |
| Day 1 | 内存层级延迟表 | §3.2 `acc` 为什么必须在寄存器 |
| Day 1 | occupancy | §4.2 为什么 25% 是正常的（反例！） |
| Day 2 | **合并访问** | §5.1 §5.4 搬运阶段的规则 |
| Day 2 | sector / 128B 事务 | §5.4 global 读的合并度分析 |
| Day 3 | shared memory 基本用法 | §3.2 整个 kernel |
| Day 3 | **补单位元**（reduction 补 0） | §3.2 越界填 0 |
| Day 4 | **bank conflict** | §5.3 §5.4 §5.5 |
| Day 4 | **padding 消冲突**（§7.6 预告） | §5.4 PAD=4，这次自己算出来了 |
| Day 4 | 三级同步成本 | §3.3 §9.3 megakernel 的同步代价 |
| Day 4 | **Amdahl：优化价值 = 幅度 × 占比** | §7.4 分化只占 1.6%，不值得优化 |
| Day 4 | PTX/SASS 阅读 | §7.4 谓词执行的证据 |
| W0 | 计时三铁律 | §7.1 |
| W0 | Day5 遗留问题 3 | §7.4 **本笔记兑现** ✓ |
| W7 | **Roofline / ridge point** | §2.1 全文的分析框架 |
| W7 | ncu / nsys 三级 profiler | §7.3 |

### 11.2 下游（今天的东西后面在哪里用）

| 去向 | 怎么用到今天的内容 |
|---|---|
| **W2 CUDA RMSNorm** | 同样的"搬进 shared → 复用 → 写回"骨架；不过 RMSNorm 是 memory-bound，重点变成融合而非 tiling |
| **W3 FlashAttention** ★ | **今天的 tiling 直接放大**：Q/K/V 分块 + online softmax。把 `acc` 从标量换成"部分和 + running max/sum"就是 FA 的核心 |
| W3 | §3.1 提到的"加法可分批、softmax 不可" | 正是 online softmax 存在的理由 |
| **W4 continuous batching** | §9.2 的 M 从 1 变 32，AI 乘 32 |
| W5 量化 | §9.2 的"减少权重字节数 → 提高 AI" |
| **W8 Triton** | §7.4 的 `mask=`；§6.4 的 `GROUP_SIZE_M` block 重排；autotune 里的 `BLOCK_M/N/K` 和 `num_stages`（= double buffering 的级数！） |
| 保研面试 | §9.2 那张 prefill/decode 对照表；"为什么 vLLM 要 continuous batching"的 Roofline 推导 |

**特别标注 FlashAttention 这条**：Attention 是 `softmax(QKᵀ/√d)·V`，朴素实现要把 `S = QKᵀ` 这个 `seq×seq` 的中间矩阵写回 global（seq=4096 时是 64 MB！）再读回来。FlashAttention 的做法是**把 S 的分块留在 shared 里直接接着算**，永不落地 —— **这就是今天"搬进片上、在片上算完、只写最终结果"思想的直接应用**，只是多了一个 online softmax 来处理 softmax 的不可分批性。

**所以今天真想通了，W3 会轻松很多。** 这句话是本笔记开头那个论断的兑现。

---

## §12 自测与验收

### 12.1 自测题（不看笔记回答，答不出就回对应小节）

1. 矩阵乘的理论算术强度是几百 FLOP/Byte，为什么朴素实现只有 0.25？多搬了多少倍字节？→ §1.1 §2.2
2. 手推 tiled matmul 的 AI = TILE/4。为什么 BK 不出现在这个公式里？→ §2.3
3. TILE=32 的 AI=8，仍然低于 ridge point。为什么不能靠继续放大 TILE 解决？→ §2.3
4. 两个 `__syncthreads()` 各防什么？去掉第二个，为什么小规模测不出来？→ §3.3
5. 经典 tiled kernel 读 `sB[k][tx]` 为什么无 bank 冲突？改成 `sB[tx][k]` 会怎样？→ §5.3
6. 手算 `sA[icA][irA]`（`sA[8][128]`）的 bank 冲突路数，并说明 PAD=4 为什么能治。为什么不用 PAD=1？→ §5.4
7. 为什么工业库的 thread tile 是 8×8？两个上限分别是什么？→ §4.2
8. 为什么这个 kernel 的 occupancy 只有 25% 却是正常的？→ §4.2
9. 打不过 cuBLAS 的四个原因，各自治什么瓶颈？→ §6
10. Tensor Core 是不是打开就快？kernel (I) 证明了什么？→ §6.3
11. 手算行主序 A·B 要怎么调 `cublasSgemm`。为什么 M=N 时这个 bug 更难发现？→ §8.3
12. M=1、N=K=4096 时 AI 是多少？为什么 tiling 救不了它？两条真解法是什么？→ §9.2
13. `if (idx < n)` 的分化在 tiled matmul 里要紧吗？三层理由。→ §7.4
14. Triton 的 `mask=` 编译成什么？→ §7.4
15. 为什么误差要用 `max|diff|/max|ref|`，抽样为什么必须包含四个角？→ §7.2

**能答出 12 个以上，Day 5-6 算过关。**

### 12.2 硬指标（必须做到）

- [ ] `05_tiled_matmul.cu` 编译无警告，`-Xptxas -v` 的 shared 用量和手算对得上
- [ ] kernel (C) 与 cuBLAS 的 `max_rel_err < 1e-5`
- [ ] 三个 `compute-sanitizer` 全部干净
- [ ] 手算的 AI（0.25 / TILE/4 / 32）与实测的 memory/compute-bound 判定一致
- [ ] **能不看笔记推导 AI = TILE/4 和 AI = BM·BN/(2(BM+BN))**
- [ ] 拿到 cuBLAS 差距倍数，并能说出四个原因的名字
- [ ] 实验 5 的 M 扫描跑出来，能解释"TFLOP/s 低但 GB/s 高"

### 12.3 加分项

- [ ] 故意删掉第二个 `__syncthreads()`，用 `racecheck` 抓到它（**亲手踩一次，比读十遍都记得牢**）
- [ ] ncu 验证 §5.4 / §5.5 的三个 bank 冲突预期，特别是**第 3 条（`sB` 侧仍有冲突）**
- [ ] 给 (F) 加 double buffering，量一下 §6.2 说的 20-30%
- [ ] 把 §6.4 的 block 重排加上，看 `lts__t_sector_hit_rate` 变化
- [ ] 在 512/2048/4096 三档规模都测一遍，观察 cuBLAS 有没有切 kernel（§8.2）

### 12.4 卡壳规则

**单点超过 90 分钟没进展就记录下来跳过，晚上复盘时决定要不要回头。**

具体到今天，允许跳过的：(F) 的 `float4` 索引推导、WMMA 的 fragment 布局细节、swizzle 的完整实现。
**不允许跳过的**：§2.3 的 AI 推导、§3.3 的两道栅栏、§5.4 的 bank 手算、§9.2 的 prefill/decode 对照。这四个是后面几周的地基。

---

## 一句话总结

**tiling 的本质不是"把矩阵切小"，而是"让每个从 global 搬上来的字节，被复用尽可能多次"—— 算术强度是这件事的度量，Roofline 的 ridge point 是它的及格线，而当 AI 已经够高之后，瓶颈会依次搬到 shared 带宽、指令发射槽、和计算屋顶上，每一次搬家都对应一个有名字的优化。**

---

*产出于 阶段一 W1 Day 5-6 · 配套代码 `05/05_tiled_matmul.cu` · 上游 `04/reduction_cuda_vs_triton.md` · 下游 W2 CUDA RMSNorm、W3 FlashAttention*


