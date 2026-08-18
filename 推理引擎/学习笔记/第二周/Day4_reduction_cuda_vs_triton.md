# 阶段一 · W1 Day 4 —— 树形归约（parallel reduction）+ bank conflict + warp shuffle

> **今天的一句话**：Day 3 你让 1 个线程串行加了 256 次，另外 255 个线程在旁边看着。今天把这 255 个线程叫回来干活——顺便发现，**"叫回来"这件事有三种写法，两种是错的（慢的），而这三种写法的差别，正好把 warp 分化、bank conflict、warp shuffle 这三个 CUDA 核心概念一次串起来**。
>
> **本周主轴上的位置**：W8 你写 Triton 时敲下的 `tl.sum(x, axis=0)`，编译出来就是今天这些东西。今天之后，"Triton 帮我省了什么"这个问题会有一个完整的、能拿代码指着说的答案。
>
> 产出：`04/04_tree_reduction.cu`（7 个归约 kernel + 4 组实验）+ 本笔记。

---

## 0. 先给六个学习目标问题一个"电梯答案"

带着答案去读细节，比读完再总结高效得多。以下每条都标了正文位置。

| # | 问题 | 电梯答案 | 详见 |
|---|------|---------|------|
| 1 | 树形归约到底优化了什么？ | **不是减少加法次数（还是 n−1 次），是把"必须排队的步数"从 n 降到 log₂n。** 加法总量不变，关键路径变短了 | §2.1、§2.4 |
| 2 | 为什么 stride 必须从大到小？ | 一句话两件事：活跃线程是**连续一段** → 整 warp 要么全活要么全死（无分化）；访问地址是**连续 32 个** → 落在 32 个不同 bank（无冲突）。**一行代码同时躲掉两个坑** | §2.3 |
| 3 | bank conflict 是什么？ | shared memory 被切成 32 个"柜台"（bank），一个 warp 的 32 个线程若挤在同一柜台的**不同抽屉**，硬件只能排队 → 慢 k 倍。挤在**同一个抽屉**反而不慢（广播） | §3.1、§3.3 |
| 4 | `tl.sum` 底下是什么？ | warp 内 `shfl.sync` 蝶形交换 + 跨 warp 的一小块 shared + 一次 `bar.sync`。**Triton 还替你处理了 mask 补单位元、bank 布局、Volta 之后的 `_sync` 语义** | §7.1 |
| 5 | 树形归约到底能快多少？ | **在全数组求和上：几乎一点都不快**（memory-bound，被 HBM 带宽压死）；在 RMSNorm 这种"每线程只处理十几个元素"的形状上：显著快。**收益取决于归约在总时间里占多大比例** | §5.1、§5.2 |
| 6 | 这堂课明天怎么用？ | W2 的 CUDA RMSNorm = 今天的 `block_reduce_sum` 把 `sum` 换成 `sum of squares` + 一次广播。W3 的 FlashAttention = 把 softmax 的**两次**归约合成**一次** | §7.5、§7.6 |

> **今天最反直觉、也最值钱的一条**：第 5 条。你会亲手做出一个"教科书说该快 30 倍、实测只快 1.02 倍"的优化，然后搞清楚为什么。**这个经验比"学会写树形归约"重要得多**——它是你 W7 学 Roofline 之后第一次在自己写的代码上验证"先看瓶颈在哪，再决定优化什么"。

---

## 1. 问题背景：Day 3 那个 `if (tid == 0)` 浪费了什么

### 1.1 先把浪费算成数字

Day 3 的 `reduce_shared_serial` 最后一段：

```cuda
if (tid == 0) {
    float acc = 0.0f;
    for (int i = 0; i < BLOCK; ++i) acc += s[i];   // 256 次串行加法
    partial[blockIdx.x] = acc;
}
```

一个 block 有 256 个线程 = 8 个 warp。这段代码执行时：

- **warp 0** 里只有 lane 0 在干活，其余 31 个 lane 被谓词掉（Day 1 §4.2 讲的分化）；
- **warp 1–7** 共 224 个线程，早就到达函数末尾，但**它们不能退出**——block 要等全部 warp 结束才释放资源，所以它们占着寄存器和 shared memory 干等。

浪费的量级：**并行度 1/256**。而且这 256 次加法是**严格串行**的——`acc += s[i]` 里第 i 次加法依赖第 i−1 次的结果，这条依赖链没法打断。fp32 加法在 H100 上的**延迟**大约 4 个周期（不是吞吐，是延迟——从发出到结果可用），所以这条链至少要 `256 × 4 ≈ 1024` 个周期。

> **延迟（latency）vs 吞吐（throughput）** 这对概念今天要用到，先立住：
> - **吞吐**：单位时间能完成多少个操作。H100 一个 SM 每周期能做 128 次 fp32 加法。
> - **延迟**：单个操作从开始到结果可用要多久。fp32 加法约 4 周期。
>
> **类比**：一条流水线上组装手机，吞吐 = 每分钟出 100 台，延迟 = 一台手机从上线到下线要 30 分钟。**串行依赖链吃的是延迟，并行任务吃的是吞吐。**
> 这就是为什么串行归约特别惨——它把一个"吞吐问题"变成了"延迟问题"，而 GPU 的强项恰恰是吞吐、弱项恰恰是延迟。

### 1.2 核心洞察：加法可以"两两配对、层层折半"

加法满足**结合律**（数学上；浮点上只是近似满足，见 §5.4）：`(a+b)+(c+d) = a+(b+(c+d))`。所以我不必按顺序加，可以：

```
初始:   a0  a1  a2  a3  a4  a5  a6  a7      8 个数
        └───┼───┼───┼───┘   │   │   │
步1:    (a0+a4) (a1+a5) (a2+a6) (a3+a7)     4 个线程各加一对，off=4
        └───────┼───────┘
步2:    (a0+a4+a2+a6) (a1+a5+a3+a7)         2 个线程，off=2
        └─────────────┘
步3:    全部之和                             1 个线程，off=1
```

**3 步 = log₂8**。256 个数 → **8 步 = log₂256**。

**类比**：256 人的淘汰赛决冠军。
- 串行归约 = 一个擂主坐着，255 个人排队上来一个个打，打 255 场。
- 树形归约 = 128 场同时开打，赢家配对再打，**8 轮出冠军**。

**关键点在于：总场次没变（还是 255 场），但"必须等前一场结束才能打下一场"的场次从 255 降到了 8。** 前提是你有足够多的场地（线程）。

### 1.3 一个正规的说法：work（工作量）与 depth（深度）

并行算法有一套标准语言，值得知道，因为工业界讨论算法时用的就是它：

| 概念 | 中文 | 含义 | 串行归约 | 树形归约 |
|---|---|---|---|---|
| **work** | 工作量 | 一共要做多少次基本操作 | n−1 | n−1 |
| **depth** / span | 深度 / 关键路径 | 有依赖关系的最长链有多长 | n−1 | log₂n |
| 理想并行时间 | | ≈ work/处理器数 + depth | | |

**树形归约是"work-efficient（工作量最优）"的**——它没有为了并行而多做任何一次加法，只是重排了顺序。这是个很高的评价：很多并行算法为了降 depth 会付出 work 变大的代价（比如前缀和 scan 的某些版本），归约不用付这个代价。

> **为什么这个区分重要**：如果一个"并行优化"让 work 涨了 2 倍、depth 降了 100 倍，那它只在**处理器多到用不完**时才划算。GPU 上经常出现这种权衡（比如 FlashAttention 的反向传播就是"重算换省显存"，故意让 work 涨）。**看到一个并行算法，先问它的 work 和 depth 各是多少——这是判断它值不值的第一步。**

### 1.4 但是——先说一个诚实的坏消息

我可以现在就告诉你实验 A 的结果（`04_tree_reduction.cu` 实验 A）：**在 512 MB 数组求和上，串行合并版和树形版的速度几乎一样，差距在 2% 以内。**

为什么？回顾 Day 2 的账：

```
读 512 MB 数据 @ H100 HBM3 峰值 3350 GB/s = 0.157 ms 的理论下界
一个 block 内的归约（8 步 × 每步几十周期）≈ 几百个周期 ≈ 0.3 µs @ 1.7 GHz
```

**归约占总时间的比例：约 0.2%。** 把它优化到 0 也只快 0.2%。

这不是说树形归约没用，而是说：

> **一个优化的价值 = 它优化的那部分 × 那部分的占比。**（这就是 Amdahl 定律的实用版本。）

所以 `04_tree_reduction.cu` 里设计了两组实验：

- **实验 A（厚归约）**：`grid = SM数 × 8`，每个线程用 grid-stride 循环处理几万个元素。归约占比 0.2% → 看不出差别。**这是全数组求和的正确写法**，也是 CUB 实际采用的策略（Mark Harris 那份经典 slides 里叫 **algorithm cascading，算法级联**）。
- **实验 B（薄归约）**：`grid = n / BLOCK`，每个线程恰好处理 1 个元素。归约占比大幅上升 → 各版本差距显形。**这个形状虽然不是求和的最优写法，但它恰好是 RMSNorm / softmax / LayerNorm 的真实形状**（一行 4096 个元素，256 个线程，每线程 16 个）。

**这个实验设计本身是今天要学的方法论**：想看清一个机制，就要设计一个"只有这个机制在起作用"的场景。否则你测到的全是噪声，然后得出"树形归约没用"这种错误结论。

---

## 2. 核心原理一：树形归约的三个版本，和它们各自的病

历史上 NVIDIA 有一份著名的 slides（Mark Harris, *Optimizing Parallel Reduction in CUDA*），把归约从 reduce#1 优化到 reduce#7，一路快了 30 倍。今天走它的前三步——**因为这三步恰好各自暴露一个不同的硬件机制**。

### 2.1 版本 1（reduce#1）：交错寻址 + 取模 → warp 分化

最直观的写法："相邻的两两配对"。

```cuda
// 04_tree_reduction.cu 的 kernel (B)
for (unsigned off = 1; off < blockDim.x; off <<= 1) {
    if (tid % (2 * off) == 0)          // ← 病灶
        s[tid] += s[tid + off];
    __syncthreads();
}
```

**执行图（off=1 那一轮，只画 warp 0 的 32 个 lane）**：

```
lane:   0  1  2  3  4  5  6  7 ... 30 31
活跃:   ✓  ✗  ✓  ✗  ✓  ✗  ✓  ✗ ...  ✓  ✗
        └─ 16 个活，16 个空转，同一个 warp 内 ─┘
```

**病灶：warp 分化（warp divergence）。** Day 1 §4.2 讲过：一个 warp 的 32 个 lane 共享一个指令发射，`if` 不成立的 lane 不是"跳过"，是**执行但结果被丢弃**（谓词执行）。所以这一轮的**有效并行度只有 50%**——付了 32 个 lane 的电，只干了 16 个 lane 的活。

越往后越惨：off=2 时 1/4 活跃，off=4 时 1/8……到 off=32 时，warp 0 里只有 lane 0 活跃，**其余 7 个 warp 整体空转但仍要走完 barrier**。

**第二个病：`%` 很慢。** GPU 没有硬件整数除法/取模单元。`tid % (2*off)` 会被编译器展开成一串乘法+移位（因为 `2*off` 是运行期值，编译器不能用"与 2^k−1"的快捷方式）。看一眼 PTX 就明白了：

```ptx
// tid % (2*off) 大致会编成这样（简化）：
// 无硬件取模 → 先算倒数近似 → 乘 → 取整 → 回乘 → 减
mul.lo.s32      %r10, %r5, 2;        // 2*off
// ... 十几条指令来算一个 %，包括 mul.hi.u32 / shr / sub
rem.u32         %r11, %r4, %r10;     // PTX 里是一条 rem，但 ptxas 会展开成多条 SASS
setp.eq.s32     %p1, %r11, 0;
```

> **一条通用经验**：GPU kernel 的内层循环里出现 `%` 或 `/`，且除数不是编译期常量的 2 的幂 —— 这几乎总是一个可以优化掉的点。改成位运算（`& (k-1)`）、或者干脆重排索引让它消失（下面版本 3 就是这么干的）。

**注意一个常见误解**：很多博客说 reduce#1 慢是因为 bank conflict。**这是错的。** 数一下：off=1 时活跃 lane 是 0,2,4,…,30，访问 `s[0], s[2], …, s[30]`，bank 号 = 0,2,…,30，**16 个不同的 bank，零冲突**。reduce#1 的病是分化和取模，**不是 bank**。（想验证？用 `04_tree_reduction.cu` 文件头的第 1 条 ncu 命令量它的 bank conflict 计数器，会接近 0。）

### 2.2 版本 2（reduce#2）：修掉分化，踩进 bank conflict

Mark Harris 的下一步：让活跃线程变成**连续的 tid**，用索引变换去访问原来的位置。

```cuda
// 04_tree_reduction.cu 的 kernel (C)
for (unsigned off = 1; off < blockDim.x; off <<= 1) {
    unsigned index = 2 * off * tid;      // 活跃线程连续了……
    if (index < blockDim.x)
        s[index] += s[index + off];      // ……但访问地址的步长炸了
    __syncthreads();
}
```

分化确实好了：off=1 时活跃的是 tid 0–127，warp 0–3 全活、warp 4–7 全死，**warp 内部没有分化**。取模也没了。

**但引入了新病：bank conflict。** off=1 时 warp 0 访问 `s[0], s[2], s[4], …, s[62]`，bank 号 = `0,2,…,30,0,2,…,30` —— **每个 bank 被两个线程撞上，且地址不同** → 2-way 冲突，这一轮的 shared 访问要花 2 倍时间。

具体的冲突度我在 §3.2 手算了完整的表，**结论是峰值 8-way**（不是很多博客说的 16 或 32——原因很有意思，见 §3.2 的"活跃线程上限"）。

### 2.3 版本 3（reduce#3）：顺序寻址——一行代码治两个病

```cuda
// 04_tree_reduction.cu 的 kernel (D)  ★ 今天的正主
for (unsigned off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) s[tid] += s[tid + off];   // 活跃线程 = 连续一段，地址 = 连续一段
    __syncthreads();
}
```

**这就是"stride 从大到小"的完整答案。** 它同时满足两个约束：

| | 活跃线程 | 访问地址 | 结果 |
|---|---|---|---|
| reduce#1（stride 小→大，取模） | 0,2,4,…（**散**） | 0,2,4,…（散在不同 bank） | ❌ 分化 ✅ 无冲突 |
| reduce#2（stride 小→大，索引变换） | 0,1,2,…（**连**） | 0,2,4,…（**挤在同 bank**） | ✅ 无分化 ❌ 冲突 |
| reduce#3（stride 大→小） | 0,1,2,…（**连**） | 0,1,2,…（**连**） | ✅ 无分化 ✅ 无冲突 |

> **为什么"连续的活跃线程"和"连续的地址"能同时满足？** 因为顺序寻址让这两件事变成了同一件事——`tid < off` 里活跃的 tid 是连续的，而它访问的地址就是 `s[tid]`，也就直接连续。**reduce#2 的问题在于它把"谁活跃"和"访问哪里"解耦了，解耦之后就没法同时优化两边。**
>
> **一个可迁移的直觉**：GPU 上"线程编号"和"数据地址"的映射关系，永远希望是**恒等映射或加个偏移**。任何形式的乘性变换（`2*off*tid`、`tid*stride`）都在制造麻烦——Day 2 的合并访问是这个道理（global 层面），今天的 bank conflict 也是这个道理（shared 层面），Day 5-6 的 tiling 还会再遇到（转置时的 padding trick）。

### 2.4 精确一点：reduce#3 真的完全没有分化吗？

**没有完全消除，只消除了 off ≥ 32 的那几轮。** 这个细节值得抠清楚，因为它直接引出版本 4。

BLOCK=256，8 轮循环：

| 轮 | off | 活跃线程 | 活跃 warp | warp 内部分化？ |
|---|---|---|---|---|
| 1 | 128 | tid 0–127 | warp 0–3 全活，4–7 全死 | 无 |
| 2 | 64 | tid 0–63 | warp 0–1 全活 | 无 |
| 3 | 32 | tid 0–31 | warp 0 全活 | 无 |
| 4 | 16 | tid 0–15 | warp 0 **一半活** | **有** ← 从这里开始 |
| 5 | 8 | tid 0–7 | warp 0 的 1/4 | 有 |
| 6 | 4 | tid 0–3 | | 有 |
| 7 | 2 | tid 0–1 | | 有 |
| 8 | 1 | tid 0 | 1/32 | 有 |

**后 5 轮里，只有 warp 0 在动，而且它内部一直在分化；同时另外 7 个 warp 什么都不干，却要陪着走 5 次 `__syncthreads()`。**

Day 3 §3.2 算过 barrier 的成本：指令本身很便宜（SM 内一个计数器），但"让快的等慢的"不便宜。这里的情况更糟——是**让 7 个闲着的 warp 陪 1 个 warp 走 5 次流程**。

这就是版本 4（kernel (E)）和版本 5（kernel (F)）要解决的：**当只剩一个 warp 时，用 warp 内部的通信原语，把 shared memory 和 barrier 一起省掉。** 见 §4。

### 2.5 看底层：`if (tid < off)` 编译成什么

Day 1 §4.3 已经讲过谓词执行，这里在归约的上下文里再确认一次（因为归约循环是短 if 的典型场景）。

```ptx
// nvcc -O3 -arch=sm_90 -ptx 04_tree_reduction.cu 里 (D) 的循环体，简化后：
$L__BB_loop:
    setp.ge.u32     %p1, %r_tid, %r_off;     // p1 = (tid >= off) —— 注意是取反的条件
    @%p1 bra        $L__SKIP;                // 若整个 warp 都满足，才真跳转
    // ↓ 下面三条只在 p1 为假（即 tid < off）的 lane 上生效
    shl.b32         %r10, %r_off, 2;         // off * sizeof(float)
    add.s32         %r11, %r_saddr, %r10;
    ld.shared.f32   %f1, [%r_saddr];         // 读 s[tid]
    ld.shared.f32   %f2, [%r11];             // 读 s[tid + off]
    add.f32         %f3, %f1, %f2;
    st.shared.f32   [%r_saddr], %f3;         // 写 s[tid]
$L__SKIP:
    bar.sync        0;                       // ★ __syncthreads()，在 if 外面
    shr.u32         %r_off, %r_off, 1;
    setp.ne.s32     %p2, %r_off, 0;
    @%p2 bra        $L__BB_loop;
```

对应的 SASS（`cuobjdump -sass ./treereduce` 可以自己看）大致长这样：

```
        ISETP.GE.U32 P0, PT, R_tid, R_off, PT
   @P0  BRA  `(.SKIP)                  // warp 整体不满足时才跳（省掉后面的执行）
        LDS   R4, [R_saddr]            // LDS = LoaD Shared
        LDS   R5, [R_saddr + off*4]
        FADD  R4, R4, R5
        STS   [R_saddr], R4            // STS = STore Shared
.SKIP:
        BAR.SYNC.DEFER_BLOCKING 0x0
```

**三个值得注意的地方**：

1. **`LDS` / `STS` 是 shared memory 专用指令**，和 global 的 `LDG`/`STG`、寄存器操作完全不同的指令。ncu 里 `smsp__inst_executed_op_shared_ld/st` 数的就是它们——**这是你量"shuffle 版省了多少 shared 访问"的直接依据**。
2. **每轮每个活跃线程：2 次 LDS + 1 次 STS。** 8 轮下来，一个 block 的 shared 访问总量约 `(128+64+…+1) × 3 ≈ 765` 次。§4 的 shuffle 版会把这个数降到 **8 次 STS + 32 次 LDS**——两个数量级。
3. **`BRA` 是真跳转，不是谓词。** 当整个 warp 都满足 `tid >= off` 时（比如 off=128 时的 warp 4–7），它们直接跳过循环体，**一条 LDS 都不执行**。这就是"整 warp 全死"比"warp 内一半死"好的地方——**全死的 warp 不消耗执行单元，半死的 warp 要执行完再丢弃结果**。

> **调试技巧**：想确认自己的 kernel 有没有分化，不要读代码猜，用 ncu 量：
> ```bash
> ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./treereduce 24
> ```
> 这个指标是"每条已执行指令平均有几个活跃线程"，**满分 32**。reduce#1 会明显低于 reduce#3。

---

## 3. 核心原理二：bank conflict（存储体冲突）

Day 3 §2.6 给它留了一句话，今天补完。这是**理解 shared memory 性能的唯一一件核心事**。

### 3.1 是什么：32 个柜台，不是 32 段内存

**shared memory 在硬件上被切成 32 个 bank（存储体），每个 bank 每个周期能服务一个 4 字节的访问。**

第一个要纠正的直觉：**bank 不是"把 shared memory 分成 32 块连续区域"，而是按 4 字节交错分配的。**

```
byte 地址:   0    4    8   12   ...  124  128  132  ...
float 下标:  0    1    2    3   ...   31   32   33  ...
bank 号:     0    1    2    3   ...   31    0    1  ...
             └────────── 一轮 32 个 bank ──────────┘

bank 号 = (字节地址 / 4) % 32 = (float 下标) % 32
```

**类比**：一个有 32 个窗口的银行，但**每个窗口只管特定尾号的账户**——尾号 0 的去 0 号窗口，尾号 1 的去 1 号窗口……
- 32 个客户尾号刚好是 0–31 → **32 个窗口同时办，一个周期搞定**（这是理想情况，也是 reduce#3 的情况）。
- 32 个客户尾号全是 0（比如账号 0、32、64、96…）→ **全挤在 0 号窗口，排队办 32 次** → 32-way conflict，慢 32 倍。
- **但如果 32 个客户是同一个账户**（同一个地址）→ 窗口喊一次号，32 个人一起听见 → **broadcast（广播），一个周期，不算冲突**。

第三条最反直觉，也最重要：

> **bank conflict 的准确定义是：一个 warp 内，两个或更多线程访问了「同一个 bank 的不同地址」。**
> 「同一个 bank 的同一个地址」不是冲突——硬件有专门的广播（broadcast）通路。写的时候更极端：多个线程写同一地址不会串行化，只是**只有一个能赢**（哪个赢是未定义的）。

这条为什么值钱？因为归约的收尾里到处是"全体读同一个 scalar"：

```cuda
const float scale = rsqrtf(s[0] / H + eps);   // 256 个线程都读 s[0]
```

如果广播算冲突，这一行就是 32-way 灾难。**实际上它是一个周期。** 所以 RMSNorm 里"把 scale 放 shared 让全体读"这个写法是完全合理的，不需要为它做任何优化。

### 3.2 手算：reduce#2 每一轮的冲突度

这是今天最值得亲手算一遍的东西（**能算出这张表，才算真懂 bank**）。

设 BLOCK=256，kernel (C) 的访问是 `s[index]` 和 `s[index+off]`，`index = 2*off*tid`，条件 `index < 256` 即 `tid < 128/off`。

| off | 活跃 tid 范围 | 活跃 lane 数（warp 0） | 访问 bank = (2·off·tid) % 32 | 不同 bank 数 | 冲突度 |
|---|---|---|---|---|---|
| 1 | 0–127 | 32 | 0,2,4,…,62 → 0,2,…,30 | 16 | **2-way** |
| 2 | 0–63 | 32 | 0,4,…,124 → 0,4,…,28 | 8 | **4-way** |
| 4 | 0–31 | 32 | 0,8,…,248 → 0,8,16,24 | 4 | **8-way** |
| 8 | 0–15 | 16 | 0,16,…,240 → 0,16 | 2 | **8-way** |
| 16 | 0–7 | 8 | 0,32,…,224 → 0 | 1 | **8-way** |
| 32 | 0–3 | 4 | 0 | 1 | 4-way |
| 64 | 0–1 | 2 | 0 | 1 | 2-way |
| 128 | 0 | 1 | 0 | 1 | 无 |

**两个观察**：

1. **峰值是 8-way，不是 32-way。** 很多博客说 reduce#2 有"最高 32 路冲突"，那是不对的——**冲突度上限受活跃 lane 数限制**。off=16 时全 warp 只有 8 个 lane 活跃，就算它们全挤在 bank 0，也只能是 8-way。**这就是"手算一遍"的价值：能纠正你读到的错误结论。**
2. **前 3 轮（off=1,2,4）是主要成本**——它们的活跃线程最多（32 个 lane 满编），且冲突度已经涨到 8。后面几轮线程都不满编了，绝对成本很小。

对照 reduce#3（顺序寻址）：访问 `s[tid]` 和 `s[tid+off]`，warp 0 的 lane 0–31 访问 `s[0..31]` → bank 0–31 各一个；`s[off..off+31]` → 也是 32 个连续地址 → 32 个不同 bank。**每一轮，读写全部零冲突。**

### 3.3 硬件为什么这样设计（为什么不做成"全交叉开关"）

shared memory 是 SM 内的 SRAM。要让 32 个线程同时访问任意 32 个地址，理论上需要一个 32×N 的全交叉开关（crossbar），面积和功耗都爆炸。**32 个独立 bank + 一个 32×32 的交换网络是面积/性能的折中点**——它能满足"32 个不同 bank"这个最常见的模式，代价是"撞同一个 bank"要排队。

> **和 Day 2 的合并访问对照，这是今天最漂亮的一个对称**：
>
> | | global memory | shared memory |
> |---|---|---|
> | 物理限制来源 | DRAM 的**突发传输**（一次至少搬 32 B 一个 sector） | SRAM 的**多端口并发**（32 个 bank 各一个端口） |
> | 想要的访问模式 | **聚**：32 个线程的地址挤在同一段 128 B 里 | **散**：32 个线程的地址分散到 32 个不同 bank |
> | 违反的代价 | 搬了没用的字节，带宽利用率降到 1/8、1/32 | 访问被串行化，延迟 ×k |
> | 特例 | 全都访问同一地址 → 只搬 1 个 sector，很好 | 全都访问同一地址 → 广播，很好 |
>
> **一个要聚、一个要散**——刚好相反，但根源都是硬件的物理约束。**记住这张表，Day 5-6 的 tiled matmul 会同时踩这两个坑**：从 global 读 tile 要合并（聚），在 shared 里按列访问 tile 要避冲突（散），经典解法是给 shared 数组多加一列 padding（`__shared__ float tile[32][33]`），让列访问的 bank 号错开。

### 3.4 实测：怎么把 bank conflict 从噪声里隔离出来

**问题**：在真实的全数组归约里，bank conflict 藏在 HBM 访存时间后面，秒表量不出来（实验 A 已经说明了这点）。

**做法**：设计一个**没有 global 访存、只有 shared 读**的 micro-benchmark。`04_tree_reduction.cu` 的 `bank_probe` 就是它：

```cuda
template <int STRIDE>
__global__ void bank_probe(float* __restrict__ out, int iters) {
    __shared__ float s[1024];
    const int tid = threadIdx.x;
    for (int i = tid; i < 1024; i += blockDim.x) s[i] = (float)i;
    __syncthreads();

    float acc = 0.0f;
    for (int it = 0; it < iters; ++it) {
        int idx = (tid * STRIDE + it) & 1023;   // STRIDE 决定 bank 分布
        acc += s[idx];
    }
    out[blockIdx.x * blockDim.x + tid] = acc;   // ★ 必须写出去
}
```

三个设计细节，每一个都是 micro-benchmark 的通用坑：

1. **`STRIDE` 是模板参数不是运行期参数**——保证编译期能把索引算式化简，测的是访存不是算术。
2. **`out[...] = acc` 这行不能省。** 如果不把结果写出去，编译器会判定整个循环是死代码，直接删掉——你会测出一个"快得不可思议"的空 kernel。**这是写 GPU micro-benchmark 最经典的翻车方式**，我见过的错误 benchmark 里一半是这个。
3. **`grid = SM 数`，每 SM 一个 block**——避免多 block 竞争 shared memory 容量导致 occupancy 变化，把变量控制住。

**预期与真实的诚实说明**：

| STRIDE | bank 分布 | 理论冲突度 | 预期耗时倍数 |
|---|---|---|---|
| 0 | 全体同地址 | 广播 | ≈ 1.0×（**和 STRIDE=1 一样快**） |
| 1 | 32 个不同 bank | 无 | 1.0×（基准） |
| 2 | 16 个 bank | 2-way | ~2× |
| 4 | 8 个 bank | 4-way | ~4× |
| 32 | 1 个 bank | 32-way | ~32× |

**但实测的时间比可能小于理论值**，原因要说清楚：这个 kernel 每 block 有 8 个 warp，当一个 warp 因为 bank 排队而挂起时，**SM 会切换到其他 warp 执行**（这正是 GPU 隐藏延迟的机制）。所以小的冲突度（2-way、4-way）可能被 warp 并发部分掩盖，时间比达不到 2×/4×；32-way 掩盖不住，会明显显形。

> **所以正确的验证姿势是：时间只作为参考，用 ncu 计数器做定论。**
> ```bash
> ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
>              ./treereduce 24
> ```
> 这个计数器数的是"因 bank 冲突而额外产生的 wavefront 数"，**它不受 warp 并发掩盖的影响**——冲突发生了就是发生了。拿它对比 (C) 和 (D) 两个 kernel，会看到差几个数量级；拿它对比 `bank_probe<1>` 和 `bank_probe<32>`，比值应当接近 31（32 路访问变成 32 个 wavefront，额外 31 个）。
>
> **这个"时间被掩盖、计数器不会被掩盖"的区别，是 profiling 的一个通用要点**：延迟类问题（冲突、依赖链）在高并发下会被隐藏，**必须用计数器而不是秒表来定位**。呼应 W7 你学 ncu 时的三级 profiler 观——nsys 看"谁慢"，ncu 看"为什么慢"。

---

## 4. 核心原理三：warp shuffle —— 连 shared memory 都不用

§2.4 留了个尾巴：归约的**最后 5 轮只有一个 warp 在动**，却要 7 个闲 warp 陪着走 barrier。现代 CUDA 的解法是 **warp 级原语（warp-level primitives）**。

### 4.1 是什么：让线程直接读另一个线程的寄存器

```cuda
float __shfl_down_sync(unsigned mask, float var, unsigned delta);
//                     ↑ 谁参与        ↑ 交换什么  ↑ 从我 +delta 的 lane 那里取
```

语义：**"把 lane `(myLane + delta)` 手里的 `var` 值，搬到我的寄存器里"**（超出 warp 边界的 lane 拿到的是自己的原值，不定义为有效数据）。

**它不经过任何内存。** 不是 global（~500 周期），不是 shared（~25 周期），是 warp 内部的**寄存器直连通路**——硬件上一个 warp 的 32 个 lane 的寄存器文件本来就在同一块 SRAM 的同一行上，shuffle 就是让读取端口跨 lane 取值。**延迟和一次普通算术指令同量级。**

**类比**：Day 3 的 shared memory 版是"每人把纸条放到共用白板上，然后大家去白板上抄"（要走出去、要等齐）。shuffle 版是"32 个人手拉手围一圈，直接把纸条递给旁边的人"——**不需要白板，也不需要喊'都写完了吗'**（同一个 warp 内，指令天然是一起发射的，`_sync` 只是保证汇合）。

```cuda
// 04_tree_reduction.cu 里的实现
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;               // 有效结果只在 lane 0
}
```

**5 步 = log₂32。** 数据流（画 8 个 lane 示意，实际 32 个）：

```
lane:      0    1    2    3    4    5    6    7
初始:      v0   v1   v2   v3   v4   v5   v6   v7
off=4:  v0+v4 v1+v5 v2+v6 v3+v7  --   --   --   --
off=2:  Σ0246 Σ1357  --   --
off=1:  Σall   --
           ↑ 结果在 lane 0
```

**注意 lane 4-7 在 off=4 之后拿到的是垃圾**（它们取的是 lane 8-11 的值）——所以**只有 lane 0 的结果可用**。这是 `_down` 变体的特点。想让所有 lane 都拿到总和，用**蝶形交换（butterfly）**：

```cuda
// 全 lane 都得到总和的版本（XOR 蝶形，也叫 all-reduce 形态）
for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, off);
//       ^^^^^^^^^^^^^^ 和 lane (myLane XOR off) 交换，双向对称
```

> **什么时候用哪个**：只有一个线程要用结果（比如写回 `partial[blockIdx.x]`）→ 用 `_down`，省一点。**全体线程都要用结果**（比如 RMSNorm 里每个线程都要乘 scale、softmax 里每个元素都要除 sum）→ 用 `_xor` 蝶形，或者用 `_down` 之后再 `__shfl_sync(mask, v, 0)` 广播回去。
> **工业代码里两种都常见**，vLLM 的 `warpReduceSum` 用的是 `_xor` 蝶形，因为它的调用点通常需要全体可见。

### 4.2 `_sync` 后缀的来历：Volta 打破了一个 15 年的假设

Day 3 §3.5 提过这件事，今天要讲透，因为它是**归约代码里最贵的一个历史坑**。

**Pascal（2016）及之前**：一个 warp 的 32 个线程共享**一个程序计数器（PC）**，永远锁步执行。所以下面这段"优化"是安全的：

```cuda
// ⚠️ Pascal 时代的写法，Volta（2017）之后【不安全】
if (tid < 32) {
    volatile float* vs = s;      // volatile 骗编译器别把它缓存到寄存器
    vs[tid] += vs[tid + 32];     // 省掉 __syncthreads()：反正一个 warp 锁步
    vs[tid] += vs[tid + 16];
    vs[tid] += vs[tid +  8];
    vs[tid] += vs[tid +  4];
    vs[tid] += vs[tid +  2];
    vs[tid] += vs[tid +  1];
}
```

这套做法有个名字：**warp-synchronous programming（warp 同步编程）**。NVIDIA 官方那份 `reduction.pdf`（2007 年的经典 slides，至今还在被无数博客抄）用的就是它。

**Volta（2017）引入 independent thread scheduling（独立线程调度）后，前提失效了**：每个线程有了自己的 PC 和调用栈，**warp 内的线程可以分头执行，且不保证自动重新汇合**。硬件为了功耗和 occupancy 的灵活性，可能让 lane 0-15 先跑一段、lane 16-31 后跑。上面那段代码于是变成了 warp 内的数据竞争——**在某些 occupancy 下偶发错误，且几乎无法复现调试**。

**为什么 `volatile` 救不了它**（Day 3 §3.3 说过一半，这里说完）：

| | `volatile` 管什么 | 不管什么 |
|---|---|---|
| 语义 | 每次访问都真读/真写内存，不许编译器缓存到寄存器 | **另一个线程执行到哪一行了** |

`volatile` 解决的是**可见性（visibility）**，不解决**时序（ordering / 汇合）**。而 warp-synchronous 假设需要的恰恰是时序保证。

**现代唯一正确的写法**就是显式原语：

```cuda
// ✅ 方案一：shuffle（最快，推荐）
v += __shfl_down_sync(0xffffffff, v, off);

// ✅ 方案二：如果必须用 shared，就显式 warp 同步
if (tid < 32) {
    s[tid] += s[tid + 32];  __syncwarp();   // 显式让这 32 个 lane 汇合
    s[tid] += s[tid + 16];  __syncwarp();
    ...
}
```

`__syncwarp(mask = 0xffffffff)` 就是 warp 级的 `__syncthreads()`——比 `bar.sync` 便宜，但仍然是一条真指令。**方案一更好，因为它连 shared 访问都省了。**

> **一个能立刻用上的代码考古技巧**：看到 `__shfl_down`（**没有** `_sync` 后缀）、看到 `volatile float* vs`、看到 `reduce#6` 这类命名——**这段代码是 2017 年之前写的，直接抄进 Hopper 项目有风险**。`_sync` 系列是 CUDA 9（2017）为 Volta 引入的，老 API 在 CUDA 9 就标了 deprecated，`sm_90` 上编译会报 warning。
>
> **`mask` 参数的真正含义**：它是"我保证这些 lane 都会到达这条指令"的**契约**，由你负责保证正确。写 `0xffffffff` 但实际只有一半 lane 到场 = 未定义行为。所以 `04_tree_reduction.cu` 里所有 shuffle 调用点都刻意保证"整个 warp 要么全进要么全不进"——`if (tid < 32)` 对 warp 0 整体成立、对 warp 1-7 整体不成立，**这不是分化**。

### 4.3 看底层：shuffle 编译成一条指令

```ptx
// warp_reduce_sum 的 PTX（nvcc -arch=sm_90 -ptx），5 步全展开：
mov.u32          %r2, 0xffffffff;         // full mask
shfl.sync.down.b32  %f2|%p1, %f1, 16, 31, %r2;   // ← 一条指令
add.f32          %f3, %f1, %f2;
shfl.sync.down.b32  %f4|%p2, %f3, 8,  31, %r2;
add.f32          %f5, %f3, %f4;
shfl.sync.down.b32  %f6|%p3, %f5, 4,  31, %r2;
add.f32          %f7, %f5, %f6;
shfl.sync.down.b32  %f8|%p4, %f7, 2,  31, %r2;
add.f32          %f9, %f7, %f8;
shfl.sync.down.b32  %f10|%p5, %f9, 1, 31, %r2;
add.f32          %f11, %f9, %f10;
```

对应 SASS：

```
SHFL.DOWN PT, R5, R4, 0x10, 0x1f     // 一条硬件指令，无内存访问
FADD      R4, R4, R5
SHFL.DOWN PT, R5, R4, 0x8,  0x1f
FADD      R4, R4, R5
... 共 5 组
```

**和 §2.5 的 shared 版对照，一眼看出省了什么**：

| | shared 树形（(D) 最后 5 轮） | shuffle（(E)/(F)） |
|---|---|---|
| 指令 | 5×(2 LDS + 1 FADD + 1 STS) = 20 条 | 5×(1 SHFL + 1 FADD) = 10 条 |
| shared 访问 | 15 次 | **0 次** |
| barrier | 5 次 `BAR.SYNC` | **0 次** |
| 参与的 warp | 8 个（7 个陪跑） | 1 个 |

`0x1f` 那个操作数是 **clamp 值**（=31）：它规定"取值的 lane 号超过 31 就不搬，保留原值"。这是 shuffle 的边界语义，硬件级实现，不用你管。

### 4.4 工业标准结构：两级 block 归约

把 warp 内 shuffle 和跨 warp 的 shared 组合起来，就是**工业界写 block reduce 的标准形状**（`04_tree_reduction.cu` 的 `block_reduce_sum`）：

```cuda
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];            // 32 = 一个 block 最多 32 个 warp
    const unsigned lane = threadIdx.x & 31;    // 我在 warp 里的编号（等价 %32）
    const unsigned wid  = threadIdx.x >> 5;    // 我是第几个 warp（等价 /32）

    v = warp_reduce_sum(v);                    // 第一级：warp 内 5 步 shuffle
    if (lane == 0) warp_sums[wid] = v;         // 每个 warp 的头儿交作业（8 次 STS）
    __syncthreads();                           // ★ 唯一一次 barrier

    const unsigned nwarps = blockDim.x >> 5;
    // ★ 让整个 warp 0 都执行这一句（而不是 if (tid < nwarps) 包起来），
    //   否则下面 warp_reduce_sum 的 mask=0xffffffff 就撒谎了。
    //   多余 lane 补加法单位元 0 —— Day3 §3.4 的「补单位元」范式又一次出现
    v = (threadIdx.x < nwarps) ? warp_sums[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);      // 第二级：warp 0 再来 5 步
    return v;                                  // 有效值只在 thread 0
}
```

**成本对照（BLOCK=256）**：

| | Day 3 串行 | (D) shared 树形 | (F) 两级 shuffle |
|---|---|---|---|
| shared 写（STS） | 256 | 256 + 255 ≈ 511 | **8** |
| shared 读（LDS） | 256 | ~510 | **32** |
| `__syncthreads()` | 1 | **8** | **1** |
| 关键路径步数 | 256 | 8 | 5 + 1 + 5 = 11（但每步更便宜） |
| shared memory 用量 | 1 KB | 1 KB | **128 B** |

最后一行别忽略：**shared 用量从 1 KB 降到 128 B**。Day 3 §2.5 讲过 shared 和 occupancy 是一对冤家——省下的 shared 直接换成更高的 occupancy，这在真实算子里往往比省几条指令更值钱。

> **两个"别造轮子"的提醒**：
> 1. **`__reduce_add_sync`**（Ampere sm_80+ 引入的硬件归约指令 `REDUX`）**只支持整型**（`int`/`unsigned`）。**fp32 没有硬件 redux 指令**，浮点归约在 Hopper 上仍然必须用 shuffle。别以为新架构给了你一条 float 归约指令。
> 2. **Cooperative Groups** 提供了更干净的封装：
>    ```cuda
>    #include <cooperative_groups.h>
>    #include <cooperative_groups/reduce.h>
>    namespace cg = cooperative_groups;
>    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
>    float total = cg::reduce(tile, v, cg::plus<float>());   // 编译出来就是 shuffle
>    ```
>    它的好处是 mask 由类型系统保证正确，不用你手写 `0xffffffff`。**工业项目里推荐用它或 CUB；手写 shuffle 是为了懂底层。**

---

## 5. 实测：把理论压成数字，并诚实解读

编译运行：

```bash
# 编译（-lineinfo 让 ncu/sanitizer 能对回源码行；-Xptxas -v 看寄存器和 shared 用量）
nvcc -O3 -arch=sm_90 -lineinfo -Xptxas -v -o treereduce 04_tree_reduction.cu

# 主实验（默认 n = 1<<27 = 512 MB）
./treereduce

# 小规模 + 随机数据（跑 sanitizer 和 ncu 时用这个，快得多）
./treereduce 24 rand

# 正确性护栏（改归约代码后必跑，Day3 养成的习惯）
compute-sanitizer --tool racecheck ./treereduce 20
compute-sanitizer --tool synccheck ./treereduce 20
```

`-Xptxas -v` 的输出要顺手看一眼：

```
ptxas info : Function properties for _Z16reduce_tree_seqPKfPfm
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 20 registers, 1024 bytes smem, 380 bytes cmem[0]
                  ↑ 无 spill（Day2 §2.4 的警戒线）   ↑ 1 KB = BLOCK×4，对得上

ptxas info : Used 22 registers, 128 bytes smem, ...   ← reduce_warp_2stage
                                 ↑ 只用 128 B，省下的 shared 换 occupancy
```

### 5.1 实验 A（厚归约）：一个"优化无效"的诚实结果

`grid = SM数 × 8 = 1056`，每线程 grid-stride 处理约 496 个元素。**预期所有版本几乎一样快。**

| kernel | 预期表现 | 为什么 |
|---|---|---|
| (A) serial merge | ~0.16 ms，贴带宽 | 归约只占 0.2%，串行的浪费被淹没 |
| (B) interleave+divergent | 同上 | 同理 |
| (C) interleave+bankconf | 同上 | 同理 |
| (D) tree sequential | 同上 | 同理 |
| (F) warp shuffle | 同上，可能略优 | 同理 |
| (G) CUB | 同上 | 同理 |

**理论下界**：`512 MB / 3350 GB/s = 0.157 ms`。实测应当在 0.16–0.19 ms，即 **80–95% of peak**——这个 kernel 是彻底的 memory-bound，**已经贴着 Roofline 的斜边了，归约算法怎么改都没用**。

> **这个结果不是失败，是今天最重要的一课。** 三层含义：
>
> 1. **验证了 W7 的 Roofline 观。** 你算出算术强度极低（读 4 字节做 1 次加法 = 0.25 FLOP/Byte）→ 必然 memory-bound → 优化必须冲着访存去，不是冲着计算去。今天亲手在自己的代码上确认了这条。
> 2. **解释了为什么 CUB 用"厚归约"策略。** 只要每个线程处理足够多的元素（grid-stride），归约那 log₂n 步的占比就被摊薄到可忽略。这个技巧叫 **algorithm cascading（算法级联）**：**先用高效的串行循环把 n 降到 gridDim×blockDim，再用树形处理剩下的**。Mark Harris 的 reduce#7 就是加了这一步，这也是他那 30× 提速里最大的一块。
> 3. **给了你一个可迁移的纪律**：**优化之前，先算这块占多少。** 别一头扎进"把归约从 8 步优化到 5 步"，先问"归约占总时间几个百分点"。这就是 W8 你立的"三尺子"之外，还需要一把"占比尺"。

### 5.2 实验 B（薄归约）：这里才看得见树形的价值

`grid = n / BLOCK`，每线程恰好 1 个元素。归约成为主要成本。

**预期的相对关系**（绝对数字取决于你的卡和数据规模，重点看排序和倍数）：

| kernel | 预期相对 (D) | 病因 |
|---|---|---|
| (A) serial merge | **明显最慢** | 关键路径 256 步串行 fp32 加法链 |
| (B) interleave+divergent | 慢于 (D) | warp 分化（有效并行度腰斩）+ `%` 指令开销 |
| (C) interleave+bankconf | 慢于 (D) | bank conflict（前 3 轮 2/4/8-way） |
| (D) tree sequential ★ | 基准 | 无分化、无冲突 |
| (E) tree+shfl unrolled | 略优于 (D) | 省掉后 5 轮的 shared 访问和 barrier |
| (F) warp shuffle 2stage | 优于 (D) | shared 流量降两个数量级，barrier 8→1 |
| (G) CUB | 与 (F) 同级 | CUB 内部就是这个结构 |

**别只看时间，要用计数器验证"病因"**——这是今天最该练的动作：

```bash
# ① 验证 (B) 的病是分化：这个指标是「每条指令平均活跃线程数」，满分 32
ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./treereduce 24
#    预期：(B) 明显低于 (D)

# ② 验证 (C) 的病是 bank conflict：这个计数器不受 warp 并发掩盖
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
             l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./treereduce 24
#    预期：(C) 比 (D) 高几个数量级，(D) 接近 0

# ③ 验证 (F) 省掉了 shared 访问
ncu --metrics smsp__inst_executed_op_shared_ld.sum,\
             smsp__inst_executed_op_shared_st.sum ./treereduce 24
#    预期：(F) 比 (D) 低约一个数量级
```

> **"用计数器指认病因"这个习惯，比会写树形归约更重要。** 理由很实在：(B) 和 (C) 都慢，但**病因完全不同，药方也完全不同**。如果你只看时间，只会得出"这两个版本都慢"，然后凭直觉猜原因——而网上一半的博客就是这么把 (B) 的慢误归因给 bank conflict 的（§2.1 提过）。**能拿计数器把"分化"和"冲突"分开，这是 profiling 能力从"会看总时间"升级到"会定位机制"的分界线。** 这也正是你 W7 三级 profiler 那套方法论在 kernel 内部的落地。

### 5.3 实验 C（bank_probe）：把冲突单独关进笼子

预期输出形状：

```
=== 实验 C bank_probe: 纯 shared 读，4096 次/线程 ===
访问模式                              time(ms)  相对 stride=1
STRIDE=0  (同地址 → 广播)               0.0xxx      ~1.0x     ← 和 =1 一样快
STRIDE=1  (连续 → 无冲突) ★             0.0xxx       1.00x
STRIDE=2  (预期 2-way)                  0.0xxx      1.x–2.0x
STRIDE=4  (预期 4-way)                  0.0xxx      2.x–4.0x
STRIDE=32 (全同 bank → 32-way)          0.xxxx      十几–32x
```

**两个必须看懂的现象**：

1. **STRIDE=0 和 STRIDE=1 一样快** → 亲眼确认"同地址是广播，不是冲突"（§3.1 那条反直觉的定义）。
2. **STRIDE=2/4 的时间比往往小于理论的 2×/4×** → 因为 SM 用其他 warp 填补了挂起 warp 的空档（延迟隐藏）。**STRIDE=32 掩盖不住**，会明显显形。

**结论**：小冲突度在高 occupancy 下可能被藏起来，**但这不代表它不存在**——一旦你的 kernel occupancy 低（比如寄存器用得多、或 shared 用得多），没有别的 warp 来填空档，冲突就会全额付账。**所以"能藏住"不等于"可以不管"，尤其在真实算子里 occupancy 通常没那么高。**

### 5.4 数值精度：树形归约顺带把精度也改善了

Day 3 §5.2 已经给出了结论表，今天在树形归约的语境下再确认一次（因为这是它的直接后果）：

| 求和方式 | 最坏误差界 | n = 2²⁷ 时的量级 |
|---|---|---|
| 串行累加 | O(n · ε · Σ\|xᵢ\|) | n = 1.34e8 |
| 树形 / pairwise | **O(log₂n · ε · Σ\|xᵢ\|)** | log₂n = 27 |
| Kahan 补偿求和 | O(ε · Σ\|xᵢ\|)，与 n 无关 | 1（代价：4 倍加法） |

**误差界差 500 万倍。** 用 `./treereduce`（全 `1.0f` 数据）能亲眼看到：CPU 上 float 串行累加会停在 `16777216 = 2²⁴`（尾数 24 位，`acc + 1.0f` 舍回原值），而所有 GPU 树形版都给出精确的 `134217728`。

> **一个少见的"性能和精度同向"的情况。** 为了并行而采用的树形结构，顺带把误差从线性累积降到了对数累积。
>
> **但要注意 (F) 两级 shuffle 版的一个细节**：段一的 grid-stride 私有累加**仍然是串行的**（每个线程串行加它负责的那 496 个元素）。所以整体误差界是 `O((n/P + log P) · ε)`，其中 P 是总线程数。**厚归约在提性能的同时略微牺牲了精度**——这是个真实的权衡，实验 A 的 `rel.err` 列能看出来（虽然仍远好于纯串行）。
>
> **工业上怎么处理**：需要极高精度时用 **fp32 输入 + fp64 累加器**，或 Kahan。PyTorch 的 `torch.sum` 对 fp16/bf16 输入默认就会**升到 fp32 累加**（`dtype` 参数可以控制）——这就是为什么 `x.half().sum()` 比你自己写的 fp16 累加准得多。**这个"低精度存储、高精度累加"的模式，是你 W7 学量化时会反复见到的核心手法。**

### 5.5 实验 D（RMSNorm）：今天的知识第一次变成引擎算子

`R = 8192` 行 × `H = 4096`（Llama-7B 的 hidden size），一个 block 负责一行。

**这个形状为什么重要**：每个线程处理 `H / BLOCK = 16` 个元素——**恰好是"归约占比不低、但也不是全部"的真实工作点**。它不是实验 A（归约占 0.2%），也不是实验 B（人造的极端），**它就是你 W2 要优化的那个真算子**。

预期：

| kernel | 预期 | 说明 |
|---|---|---|
| rmsnorm serial merge | 最慢 | 256 步串行链，每行都要付一次 |
| rmsnorm tree | 中 | 8 轮 shared + 8 次 barrier |
| rmsnorm warp shfl ★ | 最快，最接近访存下界 | 1 次 barrier，shared 只用 4 B |

**访存下界的算法**（这个要会算，是 W2 四方对标表的分母）：

```
必须搬的字节 = 读 x（R×H×4 B）+ 写 y（R×H×4 B）= 2 × 8192 × 4096 × 4 B = 268 MB
             （w 只有 16 KB，可忽略；它会常驻 L2）
理论下界时间 = 268 MB / 3350 GB/s ≈ 0.080 ms
```

代码里 `eff.BW` 就是按这个下界算的。**如果 %peak 超过 100%，不是算错了，是 cache 帮了忙**——`rmsnorm_shfl` 里第二个循环重读了 `xr[i]`，一行 16 KB 通常还在 L1/L2 里，这次重读没花 HBM 带宽。

> **这里藏着 W2 的第一个真实设计决策，今天先想清楚**：
>
> `rmsnorm_shfl` 读了 `x` **两遍**（第一遍算平方和，第二遍算输出）。有两种选择：
> - **方案 A（重读）**：靠 cache。H 大时唯一可行——H=4096 时一行 16 KB，一个 SM 上并发几个 block 就把 L1（H100 是 256 KB 共享 L1/shared）挤爆了，但 L2（50 MB）还接得住。
> - **方案 B（暂存）**：第一遍就把 `x` 存进寄存器或 shared，第二遍直接用。省一次读，但 `H/BLOCK = 16` 个 float 要占 16 个寄存器/线程——**寄存器压力上升 → occupancy 下降 → 延迟隐藏能力下降**。
>
> **哪个快？取决于 H、BLOCK、和卡。这必须实测，不能拍脑袋。** W2 Day1 的第一个实验就该是这个 A/B 对比。**今天在笔记里把这个问题立好，明天开工不用重新想。**
>
> 顺带一提：**PyTorch 的 eager RMSNorm 更惨**——`x.pow(2)` 一趟、`.mean()` 一趟、`* rsqrt` 一趟、`* weight` 一趟，每趟都要读写完整的中间张量，访存量是理论下界的 4–5 倍。**这就是 Day 2 §7 你在 benchmark 脚手架里看到的"eager vs compile"差距的根源，也是你手写 kernel 能赢 eager 的全部原因。**

---

## 6. 常见陷阱清单（按踩坑概率排序）

Day 3 那张表还全部有效，这是**归约进阶特有的**新增项：

| # | 陷阱 | 症状 | 正解 |
|---|---|---|---|
| 1 | 抄了 2017 年前的 `volatile` warp 展开 | Volta+ 上偶发错误，**换 occupancy 就变** | 用 `__shfl_down_sync` 或 `__syncwarp()`；看到没 `_sync` 后缀就警觉 |
| 2 | shuffle 的 `mask` 写 `0xffffffff` 但调用点在 warp 内分化的分支里 | 未定义行为，偶发错 | 保证整 warp 一致到达；用 `activemask()` 或 Cooperative Groups |
| 3 | `block_reduce_sum` 的第二级只让 `tid < nwarps` 进入 | mask 撒谎 → 结果偶错 | 让整个 warp 0 进入，多余 lane **补单位元 0** |
| 4 | 忘了 shuffle 的结果**只有 lane 0 有效** | 后续所有线程用了垃圾值 | `_down` 后要广播（`__shfl_sync(m,v,0)`），或直接用 `_xor` 蝶形 |
| 5 | BLOCK 不是 2 的幂却写折半 | **静默漏加**一部分元素 | BLOCK 用 2 的幂；或 pad 到 2 的幂并补单位元 |
| 6 | 树形循环里只加了一次 barrier | 结果偶尔偏小 | 每一轮都要 `__syncthreads()`（跨 warp 时） |
| 7 | 用 `blockDim.x` 而不是模板常量写折半 | 循环展不开，多余的分支和计数开销 | 性能关键路径用模板参数 `template<unsigned BS>` |
| 8 | micro-benchmark 忘了把结果写出去 | 测出"快到不可思议"的空 kernel | 结果必须写进 global（见 `bank_probe`） |
| 9 | 以为 `__reduce_add_sync` 能做 fp32 | 编译错误 | **它只支持整型**；fp32 必须 shuffle |
| 10 | 拿"同地址访问"当 bank conflict 优化 | 白改一通没提升 | 同地址是**广播**，本来就不冲突 |
| 11 | 只看时间就下结论"是 bank conflict" | 优化了错误的东西 | 用 ncu 计数器分辨分化 vs 冲突（§5.2） |
| 12 | 在实验 A 那种 memory-bound 形状上优化归约算法 | 花了一天，快了 0.2% | **先算占比**（§5.1） |

**第 3 条展开**，因为它是今天新代码里最隐蔽的一个：

```cuda
// ❌ 看起来更"干净"，实际是未定义行为
if (threadIdx.x < nwarps) {
    float v = warp_sums[lane];
    v = warp_reduce_sum(v);        // mask=0xffffffff 声称 32 个 lane 都在，
                                   // 但实际只有 nwarps(=8) 个 lane 进了这个分支
}

// ✅ 正确：整个 warp 0 都进来，多余 lane 补单位元
float v = (threadIdx.x < nwarps) ? warp_sums[lane] : 0.0f;
if (wid == 0) v = warp_reduce_sum(v);   // 这个条件对 warp 0 整体成立 → 不是分化
```

> **注意这两段的差别有多小，后果有多大。** 第一段在很多卡上"跑起来是对的"——因为 Volta 的独立线程调度**不保证**分头执行，但也**不保证**不分头。它可能在你的 H100 上、在这个 occupancy 下、这次编译中恰好正确，然后在换了 BLOCK 大小或升级了 CUDA 版本后崩掉。**"补单位元让所有线程走完全程"这个 Day 3 学的范式，在 warp 级原语这里第二次救了你**——同一个思想，两个层级。

---

## 7. 工业锚点：这块砖砌在哪

### 7.1 ★ 收口本周主轴：`tl.sum` 底下到底是什么

这是今天最该带走的一节。W8 你写过无数次 `tl.sum(x, axis=0)`，现在你能拿代码指着说它是什么了。

**Triton 的 `tl.sum` 编译产物（概念上的对应关系）**：

| 你在 Triton 里写的 | Triton 编译器生成的 CUDA/PTX 级操作 | 对应今天的哪一段 |
|---|---|---|
| `tl.sum(x, axis=0)` | ① warp 内 `shfl.sync.down` / 蝶形归约（寄存器级，零内存） | §4.1 `warp_reduce_sum` |
| | ② 跨 warp：`num_warps` 个部分和写进一小块 shared | §4.4 `warp_sums[32]` |
| | ③ 一次 `bar.sync` | §4.4 唯一的 `__syncthreads()` |
| | ④ 第一个 warp 再归约一次 | §4.4 第二级 |
| | ⑤ 自动选择无 bank conflict 的 shared 布局 | §3.2 顺序寻址 |
| `BLOCK` 常量 | 自动映射成 `num_warps × 32` 个线程，并按需展开循环 | §4.3 模板常量展开 |
| `mask=` 参数 | 自动补单位元（sum 补 0、max 补 −inf） | Day3 §3.4 + §6 第 3 条 |
| 不写 `_sync` mask | 编译器自己保证 warp 一致性，你不可能写错 | §4.2 那个 15 年的坑 |

**"Triton 帮我省了什么"——完整的三条答案（本周三天各答一条）**：

| 天 | Triton 省掉的 | 代价 |
|---|---|---|
| Day 1 | 索引拆分（`blockIdx*blockDim+threadIdx`）、边界 `if` | 失去对线程粒度的控制 |
| Day 2 | 访存布局（自动向量化、自动合并） | 失去对搬运时机的控制（无法手工 prefetch / double buffer） |
| **Day 4** | **同步的正确性负担 + warp 级优化 + bank 布局** | **失去对同步位置的控制** |

第三条的代价具体是什么？§6 那 12 条陷阱里，**第 1、2、3、5、6、9、10 条写 Triton 时根本遇不到**——它们发生在编译器生成的那一层。

> **但这个"省"是有天花板的，而天花板恰好是你课题的战场**：
>
> 当你要做 **warp specialization（warp 专业化分工）**——让 warp 0-3 专门搬数据、warp 4-7 专门算矩阵乘，两组用 `mbarrier` 异步握手（Hopper 上 FlashAttention-3 和 cuBLAS 的核心手法）——**Triton 的"一个 program 一块数据"抽象就挡路了**，因为它假设 block 内所有 warp 做同样的事。
>
> 这就是**为什么 cuBLAS、FlashAttention-3、以及 AMK 的巨核仍然要下到 CUDA/PTX**。你现在知道了这条边界在哪，也知道了跨过它需要什么（今天的 shuffle、Day 3 的 barrier 语义、Day 5-6 的 tiling）。**这正是暑假计划 §2.1 里"主线 3 巨核"要求的地基。**

### 7.2 别造轮子：三个层次的工业选择

**你手写是为了懂；上线用现成的。** 从高到低三个层次：

**① Cooperative Groups（最推荐，CUDA 自带，类型安全）**

```cuda
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__global__ void reduce_cg(const float* in, float* out, size_t n) {
    auto block = cg::this_thread_block();
    auto tile  = cg::tiled_partition<32>(block);      // 一个 warp 的抽象

    float v = 0.0f;
    for (size_t i = block.group_index().x * block.size() + block.thread_rank();
         i < n; i += (size_t)gridDim.x * block.size()) v += in[i];

    v = cg::reduce(tile, v, cg::plus<float>());        // 编译出来就是 shuffle
    // mask 由 tile 类型保证正确 —— §6 第 2 条陷阱在类型层面被消除了
    if (tile.thread_rank() == 0) atomicAdd(out, v);
}
```

**② CUB（NVIDIA 官方模板库，随 Toolkit 分发，性能最强）**

```cuda
#include <cub/cub.cuh>
using BlockReduce = cub::BlockReduce<float, 256>;
__shared__ typename BlockReduce::TempStorage temp;    // 大小由 CUB 算，别自己猜
float total = BlockReduce(temp).Sum(my_value);        // 结果只在 thread 0
```

CUB 会**按 arch 自动挑算法**（raking vs warp shuffle）、自动处理 bank 布局。整数组求和还有更省事的设备级 API：`cub::DeviceReduce::Sum`（内部就是 algorithm cascading + 两趟 kernel）。

**③ 框架层（真实推理引擎里长什么样）**

vLLM 的 `csrc/reduction_utils.cuh` 里就是你今天写的东西，形状几乎一模一样：

```cuda
// vLLM 的实际结构（简化引用，看它和你的 block_reduce_sum 多像）
template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += VLLM_SHFL_XOR_SYNC(val, mask);         // ← 蝶形，全 lane 都要结果
    return val;
}

template<typename T, int numLanes = WARP_SIZE>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);                          // 全 lane 都拿到结果
    return val;
}
```

它被用在哪里？**`rms_norm_kernel`、`layernorm_kernel`、`softmax`、`moe_topk`** ——全是归约。**你今天写的 `block_reduce_sum` 就是生产级推理引擎里被调用最频繁的 device 函数之一。**

> **注意 vLLM 用的是 `_XOR_` 蝶形而不是 `_DOWN_`**，原因在 §4.1 说过：它的调用点（RMSNorm、softmax）都需要**全体线程拿到结果**去做后续的缩放。**这个选择不是随意的，是被调用场景决定的**——这种"接口形态由使用方决定"的思维，就是你 W0 做 Backend 抽象时练的那件事在 kernel 层的版本。

### 7.3 大模型推理里，归约到底出现在哪

把今天学的东西定位到你的引擎上：

| 算子 | 归约类型 | 归约维度 | 在推理里的位置 |
|---|---|---|---|
| **RMSNorm / LayerNorm** | sum of squares（+ mean） | hidden_dim (4096) | 每层 2 次，prefill 和 decode 都有 |
| **softmax**（attention 内） | **max，然后 exp-sum**（两次！） | seq_len（decode 时 = KV 长度） | 每层每 head 一次 |
| **logits → 采样** | max（top-k / argmax） | vocab_size (128256) | 每步一次，vocab 很大 |
| **fused QKV / FFN 的 GEMM** | K 维累加（本质也是归约） | K (4096/11008) | 计算量的主体 |
| **张量并行的 AllReduce** | 跨 GPU 求和 | hidden_dim | 每层 2 次（多卡时） |

**两个值得琢磨的点**：

1. **softmax 需要两次归约，这正是 FlashAttention 的靶子。** 标准 softmax：先扫一遍求 max（数值稳定必需），再扫一遍求 `Σexp(x−max)`，第三遍做除法——**三次遍历，两次全局同步**。序列很长时这意味着中间结果（那个 `T×T` 的矩阵）必须落 HBM。**online softmax 的技巧就是把这两次归约合成一次**：边扫边维护 running max 和 running sum，每来一个新块就用 `exp(old_max − new_max)` 重缩放已有的和。**W3 你要精读的就是这个。今天先在这里埋一句：它之所以可能，正是因为 max 和 sum 都是"可结合的归约"，而可结合的东西就能增量更新。**

2. **decode 阶段的归约维度非常短，这是个真问题。** decode 时 batch=1、seq=1，RMSNorm 只对 4096 个元素归约一次——`4096/256 = 16` 个元素/线程，**kernel launch 开销（~3-5 µs）都可能比 kernel 本身长**。这就是为什么 decode 阶段的优化重点是**算子融合**（把 RMSNorm+QKV 投影融进一个 kernel，省 launch 和中间访存），而**巨核（megakernel）是这条路的极致形态**——把整个 decoder layer 甚至整个模型融成一个 kernel。**这直接就是你小米课题主线 3 和 AMK 要解决的问题。**

### 7.4 ★ 直通研线：为什么巨核在 H100 上会被同步拖累

Day 3 §7.6 讲过"block 之间不能同步"，今天有了 warp/block 两级同步的完整认知，可以把这个假设写得更精确了。

**同步成本的三个层级（H100 量级，理解到数量级即可）**：

| 层级 | 机制 | 成本 | 作用域 |
|---|---|---|---|
| warp 内 | `__syncwarp()` / shuffle 隐含 | **~几个周期**（一条指令） | 32 线程 |
| block 内 | `__syncthreads()` = `bar.sync` | **~几十周期**（SM 内计数器） | ≤1024 线程，同一 SM |
| **grid 级** | `cg::this_grid().sync()` | **~微秒级**（要走 global memory 上的原子计数 + L2 往返） | 全部 SM |

**差了三个数量级。** 而巨核的定义就是"把很多层融进一个 kernel"，层与层之间有真实的数据依赖（第 2 层要用第 1 层的全部输出），**所以它必须做 grid 级同步**——这是它省下 kernel launch 开销所付出的代价。

**H100 上为什么更痛？** 132 个 SM（H100 SXM），grid 级 barrier 要等**最慢的那个 SM**。SM 数越多，"最慢的那个"的期望值越差（这是统计上的必然：max 的期望随样本数增长）。**再加上 H100 的 HBM3 更快、计算更强，barrier 的绝对时间没变，占比就更大了。**

> **这就是给研线的一句话假设（可以写进你的 AMK 三问清单）**：
> **AMK 巨核在 H100 上相对 cuBLAS 只有 0.60–0.72× 的性能，一个可检验的假设是：grid 级同步（cross-SM sync）的开销随 SM 数量增长而恶化，而 H100 的 132 SM 是 A100（108）的 1.22 倍。**
>
> **怎么验证**（W3-W4 做）：ncu 里看 barrier 相关的 stall reason——`smsp__warp_issue_stalled_barrier_per_warp_active` 这类指标，或者在 nsys timeline 上看巨核内部的"活跃 SM 数随时间的波动"（同步点会表现为周期性的全体空闲）。**如果 stall_barrier 占比高，假设成立；如果不高，就要换方向找（比如寄存器压力、L2 抖动）。**
>
> **这个假设的价值在于它是可证伪的。** 带着"我猜是 X，我打算用指标 Y 验证"去和师兄讨论，比带着"我觉得同步很慢"专业得多。

### 7.5 归约的另一面：Hopper 的新硬件（知道门在哪）

H100 给归约相关的场景加了两样东西，今天只需要知道存在：

1. **Thread Block Cluster（线程块簇）+ Distributed Shared Memory（分布式共享内存）**：Day 3 §7.7 提过——**同一个 cluster（最多 16 个 block）内的 block 可以互相读写对方的 shared memory**，并且有 `cluster.sync()`。这是 CUDA 编程模型第一次在 block 和 grid 之间插入一个新层级。**意义：原来需要 grid 级同步的场景，如果规模能装进一个 cluster，就能用便宜得多的 cluster 级同步。** 这对巨核是一个真实的可用工具——**值得列进你 AMK 的"可改点清单"（W2 的任务）**。
2. **`__reduce_add_sync` 等 REDUX 指令（sm_80+）**：硬件级 warp 归约，**但只支持整型**。fp32 归约在 Hopper 上仍然只能用 shuffle（§4.4 提醒过）。

### 7.6 明天的接口：从归约到 tiling

Day 5-6 的 tiled matmul 会同时用到今天和 Day 2 的全部知识：

- 从 global 读 tile → **要合并访问**（Day 2，要"聚"）
- tile 存在 shared 里，按行按列访问 → **要避 bank conflict**（今天，要"散"）
- K 维累加 → **本质是归约**，只不过是在寄存器里做的（每个线程独立累加自己那个输出元素）
- 每个 tile 处理完要同步 → **`__syncthreads()`**（Day 3）

**一个具体的预告**：矩阵乘的 `B` 矩阵在 shared 里按列访问时，`tile[k][tx]` 中 k 变化会造成步长为 tile 宽度的访问 → **如果 tile 宽度是 32，全部落在同一 bank，32-way 冲突**。经典解法是 padding：

```cuda
__shared__ float tileB[32][33];   // ← 多一列！让每行的起始 bank 号错开 1
//                       ^^ 这一个字符的改动能带来可观的提速
```

**为什么有效**：`tileB[k][tx]` 的地址 = `k*33 + tx`，bank = `(k*33 + tx) % 32 = (k + tx) % 32` —— k 变化时 bank 号也变化，冲突消除。**今天算过 §3.2 那张表，明天看到这个 trick 就是"哦，原来如此"，而不是"这是什么黑魔法"。**

---

## 8. 【造 1.5h】+【研 1h】：今天的另外两条线

### 8.1 造：给 RMSNorm 立好 W2 的接口和分母

计划里今天的【造】只要求"确认 RMSNorm 在 PyTorch 后端跑通，为下周 CUDA 重写做接口准备"。做三件具体的事：

**① 在 `engine/backend.py` 里把换乘标记写死**（这是 W1→W2 的交接单）：

```python
# engine/backend.py
class TorchBackend(Backend):
    def rmsnorm(self, x, weight, eps=1e-6):
        """RMSNorm 参考实现（永远正确的 baseline）。

        ★ W2 CUDA 重写笔记（W1 Day4 写下）：
          核心 = 一次「平方和归约」+ 一次「广播缩放」，就是 Day4 的 block_reduce_sum
          把 sum 换成 sum of squares。
          排布：一个 block 负责一行（一个 token 的 hidden 向量），grid = 行数。
          归约实现用两级 warp shuffle（04_tree_reduction.cu 的 block_reduce_sum），
          不要用 shared 树形版 —— 实测 shared 流量差两个数量级。
          待实测的设计决策：x 读两遍（靠 cache）vs 第一遍暂存寄存器/shared。
          H=4096 时寄存器方案要 16 个 float/线程，可能压 occupancy —— A/B 实测决定。
          访存下界 = 2 × R × H × 4 B（读 x + 写 y），四方对标表的分母用这个。
        """
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return h * weight
```

**② 用 Day 2 的 benchmark 脚手架给 PyTorch RMSNorm 记一笔基线**（eager + compile 两个数字），存进 `engine/benchmark_baseline.json`。**这是 W2 四方对标表的前两列，今天先把分母立起来。**

**③ 确认 `tests/test_backend.py` 里 RMSNorm 的数值测试还是绿的**（三尺子：`allclose` → 相对误差 → 极端输入）。**接口稳定的定义是"测试还在过"，不是"签名没变"。**

> **为什么今天做这个而不是别的**：W2 Day1 你要的是"打开文件就知道从哪写第一行"。上面那段 docstring 就是给未来的自己的完整交接单——**排布、归约方案、待实测的决策、对标的分母，四件事全在里面。** 这比"明天再想"节省的时间，往往是半天。

### 8.2 研：给 AMK 补一条可证伪的假设

按计划今天的【研】是 1h，做两件轻的：

1. **用今天的眼光扫一遍 AMK 的 ncu 数据**：`amk_l8b_ncu_full.txt` 里找归约相关的迹象——`smsp__inst_executed_op_shared_ld/st`（shared 访问量）、bank conflict 计数器、以及 barrier stall。**巨核里的 RMSNorm 和 softmax 都是归约，如果它们的 shared 流量异常高，说明用的是 shared 树形而不是 shuffle。**
2. **把 §7.4 那条假设写进 W0 Day5 的三问清单**：

```markdown
## AMK 三问清单（W1 Day4 更新）

### 问题 2：跨 SM 同步在数据里的证据是哪几个指标？
**假设（W1 Day4 立）**：grid 级同步开销随 SM 数量增长恶化。
- 机制：grid barrier 走 global memory 原子计数 + L2 往返，~µs 级；
  而 block 内 `__syncthreads()` 只是 SM 内计数器，~几十周期。差 3 个数量级。
- H100 有 132 SM（A100 是 108），barrier 要等最慢的 SM，SM 越多期望越差。
- **待验证指标**：`smsp__warp_issue_stalled_barrier_per_warp_active`（barrier stall 占比）；
  nsys timeline 上巨核内部"活跃 SM 数"的周期性塌陷。
- **可证伪**：如果 barrier stall 占比不高，假设不成立 → 转向查寄存器压力 / L2 抖动。
- **可能的解法方向**：Hopper 的 thread block cluster + `cluster.sync()` 比 grid sync 便宜，
  如果巨核的同步规模能装进 16 个 block，这是一个真实可试的改动点（W2 选靶子时考虑）。
```

> **为什么要写成"假设 + 待验证指标 + 可证伪条件"这个格式**：这就是 measurement 类研究的基本单元。W7 的 `technical_report_v1.md` 就是把若干个这样的单元缝起来。**从今天开始按这个格式记，到 W7 就不用重写。**

---

## 9. 串联表 + 自测题 + 完成标准

### 9.1 和已有笔记的串联

| 今天的内容 | 关联笔记 | 换乘逻辑 |
|---|---|---|
| 树形归约 vs 串行 | Day 3 §4.1 三段式骨架 | Day 3 的 `if(tid==0)` 串行合并，今天升级成 log₂n 步 |
| stride 从大到小 | **Day 1 §4.2 warp 分化** | Day 1 学的分化，今天第一次成为**具体优化决策的依据** |
| bank conflict | **Day 3 §2.6**（留了一句话）+ **Day 2 §3 合并访问** | 一个要"散"、一个要"聚"，硬件约束不同（§3.3 那张对称表） |
| warp shuffle | **Day 3 §3.5**（Volta 的坑） | Day 3 说"Day 4 会写出来"，今天写了，并讲清 `_sync` 的来历 |
| 补单位元 | Day 3 §3.4 | 同一范式的第二次应用：这次是给 shuffle 的 mask 兜底 |
| `tl.sum` 底下是什么 | **W8 全周 Triton** | 本周主轴的收口第三条：Triton 省掉了同步的正确性负担 |
| 实验 A 的"优化无效" | **W7 Roofline** | 第一次在自己的代码上验证"先看瓶颈占比再优化" |
| 树形归约更准 | Day 3 §5.2 | 误差从 O(n·ε) 降到 O(log n·ε)，性能和精度同向 |
| RMSNorm 行归约 | W8 Day3 Triton RMSNorm + W0 Backend | **W2 的直接起点**：把 sum 换成 sum of squares |
| softmax 两次归约 | W6 attention + W8 Day7 online softmax | **W3 FlashAttention**：把两次归约合成一次的前提是"可结合" |
| grid 级同步很贵 | Day 3 §7.6 + W7 AMK report | 三级同步成本表 → 给 AMK 立了一条可证伪假设（§7.4） |
| shared padding trick | — | **Day 5-6 tiled matmul** 的预告（§7.6） |

### 9.2 自测六问（合上笔记能答才算过）

1. 树形归约把加法次数从 n−1 降到了多少？**（陷阱题，§1.3）**
2. 为什么 `stride` 从大到小同时解决了两个不同的问题？分别是什么问题、各自的硬件根源是什么？（§2.3、§3.3）
3. 一个 warp 的 32 个线程都读 `s[0]`，是几路 bank conflict？为什么？（§3.1）
4. 2015 年的教科书里用 `volatile` 省掉最后 5 次 `__syncthreads()`，为什么这段代码在 H100 上不安全？`volatile` 到底管什么、不管什么？（§4.2）
5. 实验 A 里树形归约几乎没有提速，这说明树形归约没用吗？该怎么正确表述这个结论？（§5.1）
6. `tl.sum` 底下有哪四步？Triton 因此让你写不出哪一类优化？（§7.1）

**参考答案位置**：1→§1.3（**还是 n−1，降的是 depth 不是 work**）；2→§2.3+§3.3；3→**0 路，是广播**，§3.1；4→§4.2；5→§5.1；6→§7.1。

### 9.3 完成标准 checklist

**硬指标（前 5 条必须打勾）**：

- [ ] `04_tree_reduction.cu` 编译通过（`-Xptxas -v` 确认无 spill、shared 用量符合预期），7 个 kernel 结果全部和 double 参考值对齐（相对误差 < 1e-6）
- [ ] 能**不看代码**写出树形归约骨架（折半循环 + barrier 在 if 外面），并说清为什么 stride 从大到小
- [ ] 能**不看代码**写出 `warp_reduce_sum`（5 步 shuffle）和两级 `block_reduce_sum` 的结构
- [ ] 用 ncu 分别**量出** (B) 的分化（`thread_inst_executed_per_inst_executed` 偏低）和 (C) 的 bank conflict（计数器高几个数量级）——**能拿数据指认病因，不是凭猜**
- [ ] 能讲清 `tl.sum` 底下的四步，以及"Triton 省了什么、代价是什么"（§7.1 那张表）

**加分项**：

- [ ] `bank_probe` 跑出 STRIDE=0/1/2/4/32 的时间表，并能解释"为什么 STRIDE=0 不慢"和"为什么 2×/4× 达不到理论倍数"
- [ ] 实验 D 的 RMSNorm 三版本跑通，`%peak` 数字记进 `benchmark_baseline.json`
- [ ] 手算一遍 §3.2 那张 reduce#2 冲突度表（**能算出峰值是 8-way 而不是 32-way**）
- [ ] `engine/backend.py` 的 W2 换乘标记写好；`tests/test_backend.py` 全绿
- [ ] AMK 三问清单里补上"grid 同步开销"那条可证伪假设

**卡壳规则**（你的老规矩）：任何一条卡超 1h → 记录问题、继续推进。今天最容易卡的是 ncu 权限/指标名（H100 上某些计数器需要 `--target-processes all` 或管理员放开 `NVreg_RestrictProfilingToAdminUsers`）——**这种环境问题直接记录跳过，不值得占用主线时间**。

---

### 附：今日一句话总结

> **树形归约的本质不是"加得更少"，而是"排队更短"——work 不变（n−1 次加法），depth 从 n 降到 log₂n。**
> 而让这个理论优势真正落地，要躲开两个硬件坑（warp 分化、bank conflict）、越过一个历史坑（Volta 之后 warp 不再锁步），最终形态是"warp 内 shuffle + 跨 warp 一小块 shared + 一次 barrier"——**这就是 `tl.sum` 底下的东西，也是明天要写进 CUDA RMSNorm 的东西。**
>
> **但今天最该记住的是实验 A 那个"没提速"的结果**：一个优化的价值 = 它优化的那部分 × 那部分的占比。**先算占比，再动手。** 这条纪律，比今天所有 kernel 加起来更值钱。

---

*产出于 阶段一 W1 Day 4 · 配套代码 `04/04_tree_reduction.cu` · 上游 Day3 `03/cuda_shared_memory_reduction.md` · 下游 Day5-6 tiled matmul + W2 CUDA RMSNorm*
