# Day 6 · 预研日：FlashAttention + PagedAttention 框架认知

> **一句话主题**：W3-W6 要造的东西，在领域地图上各自解决什么问题？
> 预研不是提前学，是把**框架认知**建好——到时候学的每一块，你都知道往哪放、为什么要学它。
> 今天读论文骨架，串联你已有的认知，而不是从零开始。

---

## 0. 先建全局地图：推理引擎的两道瓶颈

在进任何论文之前，先搞清楚 FlashAttention 和 PagedAttention 分别解决的是**哪一道瓶颈**。否则读论文时，满眼都是细节，不知道"这在解决什么"。

大模型推理的整条数据流，有两道墙：

```
输入 tokens
    ↓
[Attention 计算] ← 墙一：计算本身太慢（大矩阵 + 反复搬显存）
    ↓
[KV Cache 管理] ← 墙二：显存用不完整（碎片化 + 静态分配浪费）
    ↓
输出 tokens
```

> **FlashAttention** 砸的是**墙一**：让 attention 计算本身更快（算法层面重排计算顺序，减少显存搬运）。
>
> **PagedAttention** 砸的是**墙二**：让 KV Cache 的显存用得更充分（操作系统分页思路，消灭碎片）。

**这俩是正交的优化**，互不冲突，所以现代推理引擎（vLLM、TRT-LLM）通常把两者都用上。今天的目标，是把这两块的"为什么""是什么""解决了什么"搞清楚，不要求会实现。

---

## 1. FlashAttention —— 砸墙一：让 attention 计算变快

### 1.1 问题背景：标准 attention 到底慢在哪

先回忆标准 attention 的三步：

```
S = Q @ K^T          # 相似度分数矩阵, 形状 [N, N]  (N = 序列长度)
P = softmax(S)       # 归一化成概率, 还是 [N, N]
O = P @ V            # 加权求和, 得到输出
```

新手的直觉是"慢在矩阵乘法（计算量大）"。**这个直觉是错的**，这也是 FlashAttention 最反常识、最值钱的洞察：

> **attention 慢，主要不是慢在"算"，而是慢在"搬"。**

关键在中间那个 `[N, N]` 的矩阵 `S`（和 `P`）。序列长度 `N=2048` 时，这个矩阵有 `2048 × 2048 ≈ 400万` 个元素。标准做法把它**整个算出来、写进显存、再整个读回来**做 softmax，再写回、再读回做 `@V`。**这个巨大的中间矩阵在显存里反复来回搬，才是真正的时间黑洞。**

### 1.2 前置概念：GPU 的内存是分层的（HBM vs SRAM）

要理解"搬"为什么这么贵，必须先懂 GPU 内存不是铁板一块，而是**分层的**——这是理解 FlashAttention 的地基：

> **HBM（High Bandwidth Memory，高带宽显存）**：就是你说的"这张卡有 80GB 显存"里的那 80GB。**容量大、但相对慢**。它离计算单元"远"。类比：仓库——东西全放这，但每次取货要走一趟。

> **SRAM（Static RAM，片上高速缓存）**：紧贴计算核心（SM）的一小块超高速内存，H100 上每个 SM 只有几十 KB。**快得多（约 HBM 的 10-20 倍带宽），但极小**。类比：你手边的桌面——拿取瞬间完成，但只能放几样东西。

> **一个必须记住的数字直觉**：从 HBM 读数据，比在 SRAM 里算，慢一个数量级。所以现代 GPU kernel 的性能，**往往不由"算了多少次"决定，而由"从 HBM 搬了多少次"决定**。这类被搬运卡脖子的操作叫 **memory-bound（访存受限）**（呼应你 Day4 学的 decode 阶段瓶颈）。attention 恰恰是重度 memory-bound。

### 1.3 核心思想：不落地那个大矩阵（这就是你写过的 online softmax）

FlashAttention 的解法一句话：**别把 `[N,N]` 的大矩阵写进 HBM。** 把 Q、K、V 切成小块（tile），一块一块搬进 SRAM，在 SRAM 里把这一小块的 `S→softmax→@V` **一条龙算完**，只把最终的小块结果写回 HBM。中间那个巨大的 `S`/`P` 矩阵**从头到尾没有在 HBM 里完整存在过**。

> **这里有个真正的难点，也正是你 W8 亲手写过的东西**：softmax 需要"看到一整行所有的数"才能归一化（分母是整行求和）。可现在你一次只处理一小块，怎么在"没看全"的情况下算 softmax？
>
> 答案就是 **online softmax（在线/增量 softmax）**——你 Day7 笔记里那段 numpy 的心脏。它维护两个"运行状态"：目前见过的**最大值** `m` 和**指数和** `l`，每来一个新块就**增量修正**之前的结果。看下面这段你已经懂的代码，今天只是把它放回论文语境：

```python
# online softmax 的心脏 (你 W8 已亲手写过)。这就是 FlashAttention 免于落地大矩阵的关键
# 环境: numpy。演示"分块看到部分数据也能正确算 softmax 加权和"
import numpy as np

def online_softmax_attention(Q, K, V, block_size=64):
    N, d = Q.shape
    O = np.zeros((N, d))
    for i in range(N):                     # 对每个 query 行 (真实 FA 里 query 也分块, 这里简化)
        m = -np.inf                        # 运行最大值 (running max), 防指数爆炸
        l = 0.0                            # 运行指数和 (running sum of exp), softmax 的分母
        acc = np.zeros(d)                  # 运行加权和 (running weighted sum of V)
        for j in range(0, N, block_size):  # K/V 分块进来, 一次只看一小块 (这就是 tiling)
            K_blk = K[j:j+block_size]      # 这一小块能装进 SRAM
            V_blk = V[j:j+block_size]
            s = Q[i] @ K_blk.T             # 只算这一小块的分数, 不落地整行

            m_new = max(m, s.max())        # ① 更新全局最大值
            # ② 关键: 用新旧最大值的差, 把"旧的累积结果"重新缩放到统一基准
            #    这一步保证了"分块算"和"整行一次算"结果完全相同
            correction = np.exp(m - m_new)
            p = np.exp(s - m_new)          # 当前块相对新基准的指数
            l = l * correction + p.sum()   # 修正旧分母, 加上新块的贡献
            acc = acc * correction + p @ V_blk  # 修正旧加权和, 加上新块的贡献
            m = m_new
        O[i] = acc / l                     # 最后一次性归一化
    return O

# 验证: 和"整行一次算"的标准 softmax 数值对齐 (呼应 Day4/Day5 的正确性观)
def standard_attention(Q, K, V):
    S = Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P = P / P.sum(axis=1, keepdims=True)
    return P @ V

np.random.seed(0)
Q, K, V = np.random.randn(128, 32), np.random.randn(128, 32), np.random.randn(128, 32)
assert np.allclose(online_softmax_attention(Q, K, V), standard_attention(Q, K, V), atol=1e-6)
print("✅ online softmax 分块结果 与 标准 softmax 完全一致")
```

> **为什么那个 `correction = exp(m - m_new)` 是整个算法的命门？** 因为每来一个新块，全局最大值可能变大，之前用旧最大值算出来的所有 `exp` 都"基准错了"、偏大了。这个修正因子把旧结果**统一拉到新基准**上，保证分块累积的结果和"一次看全"在数学上**逐位相等**。没有它，分块 softmax 就是错的。**这一行，就是 FlashAttention 敢把大矩阵拆开算的底气。**

### 1.4 论文的核心表述：IO 复杂度分析（今天要读回论文的重点）

你已经懂算法了，今天读论文（摘要 + §1-2 + 图1）真正要"接回去"的，是论文**怎么用一套语言把这件事说清楚**——即 **IO 复杂度（IO complexity）分析**。

> **IO 复杂度**：不数"算了多少次浮点运算"（那是传统的计算复杂度），而是数**"从 HBM 读写了多少次数据"**。这是 FlashAttention 论文的核心视角转换——因为 attention 是 memory-bound，决定速度的是 IO 不是计算。

论文里那张关键对比（读图1和对应的表时重点看）：

| | 标准 attention | FlashAttention |
|---|---|---|
| HBM 访问量 | **O(N²)**：那个 `[N,N]` 大矩阵要写回、读出 | **O(N²·d² / M)**（M=SRAM大小），实际远小于标准 |
| 中间矩阵 `S`/`P` | 完整落地 HBM | **从不落地**，只在 SRAM 里存在 |
| 计算量（FLOPs） | 一样 | **一样**（算的次数没变！） |
| 实测速度 | 基准 | 快数倍 |

> **读到这里要抓住的"啊哈"**：FlashAttention **没有减少任何一次乘法加法**（FLOPs 完全一样），它只是**大幅减少了 HBM 访问**。速度提升几倍，全来自"少搬数据"。这颠覆了"优化 = 减少计算量"的朴素认知——**在 memory-bound 场景，减少访存比减少计算重要得多。** 这个认知你会在整个 AI Infra 生涯里反复用到（也直接关联你 AMK profiling 里 memory 相关指标为什么关键）。

### 1.5 【产出】在 W8"我 vs 官方"差距表上补一列

你 W8 已经有一张"我手写的 vs 官方 FlashAttention"对照表。今天读完论文，补上第三列"**论文原文怎么说**"：

| 维度 | 我 W8 手写版 | 官方实现 | 论文原文怎么说（今天补） |
|------|------------|---------|----------------------|
| online softmax | ✅ 手写了心脏 | CUDA kernel 融合 | §3 Algorithm 1 的 rescale 步骤 |
| tiling 分块 | 简化版/逐行 | Q/K/V 都分块 + 双层循环 | §3.1 图1 的 outer/inner loop |
| IO 复杂度 | 没分析 | —— | §3.2 Theorem，O(N²d²/M) |
| 反向传播 | 没实现 | 重算而非存 P | §3.3 recomputation 省显存 |

> 补这一列的意义：**把"我会写"升级成"我知道它在学术脉络里的准确位置和表述"**。面试或和张老师聊时，你能说"我手写过 online softmax，也读了原论文的 IO 复杂度分析，理解它为什么是 memory-bound 优化"——这是有深度的表达。

### 1.6 三句话讲清 FlashAttention（完成标准自测）

> 1. **问题**：注意力计算慢，不是慢在算，是慢在那个巨大的中间矩阵在慢速显存里反复搬来搬去。
> 2. **做法**：把数据切成小块塞进芯片上的高速小内存，用 online softmax 一块块增量算，中间大矩阵永不落地。
> 3. **结果**：计算量一点没减，但显存搬运大幅减少，注意力快了好几倍——尤其长序列。

---

## 2. PagedAttention —— 砸墙二：让 KV Cache 的显存不浪费

### 2.1 问题背景：你 W7 已经骂过的那个浪费

先接回你 W7 笔记里已经想明白的事：**用 `torch.cat` 拼 KV Cache 为什么浪费显存？** 今天就是看 vLLM 官方的答案，和你的理解对不对得上。

传统推理引擎给每个请求的 KV Cache 分配显存时，面临一个两难：**它不知道这个请求最终会生成多长。** 于是只能按**最大可能长度**（比如 2048）**一次性预留一整块连续显存**。问题来了：

- 实际大部分请求可能只生成 100 个 token，**剩下 1948 个位置的显存被占着、却空着**——这叫**内部碎片（internal fragmentation）**。
- 每个请求要一整块**连续**显存，来来去去之后，显存里留下很多"不够大又用不上"的空隙——这叫**外部碎片（external fragmentation）**。

> vLLM 论文里给了一个惊人的数字：传统方式下，**实际有效利用的 KV Cache 显存可能只有 20-40%**，其余全浪费在碎片上。显存是推理最贵的资源，浪费 60% 意味着你能同时服务的请求数直接砍到零头。**这就是墙二。**

### 2.2 灵魂类比：操作系统的虚拟内存分页

PagedAttention 的名字里 "Paged"（分页）直接暴露了它的思想来源——**操作系统管理内存的经典方案：分页（paging）**。这是本节最重要的类比，抓住它整节就通了。

> **操作系统的分页**：程序以为自己用的是一整块**连续**的内存（叫"虚拟地址"），但操作系统在背后把物理内存切成固定大小的小块（叫"页/page"），程序的连续虚拟地址，实际映射到**物理上东一块西一块、不连续**的页上。中间靠一张**页表（page table）**记录"虚拟第几页 → 物理哪一块"。

**好处**：物理内存不需要一大块连续的空闲区，任何零散的空页都能用上——**碎片问题被彻底解决**。PagedAttention 就是把这套原封不动搬到 KV Cache 上。

### 2.3 三个核心概念（论文 §1-3 要抓的直觉）

论文只需读摘要 + §1-3 的图，抓住这三个概念的**直觉**即可，别抠实现：

> **① KV block（KV 块）**：把一个请求的 KV Cache，切成固定大小的小块，比如**每块存 16 个 token 的 K/V**。不再要求整个 Cache 连续，只要求"块内"这 16 个连续。类比：操作系统的一个"页"。

> **② block table（块表）**：每个请求维护一张表，记录"我这个请求的第 0~15 个 token 在物理块 #7，第 16~31 个 token 在物理块 #3……"。类比：操作系统的"页表"，负责把逻辑顺序翻译成物理位置。

> **③ 逻辑连续 / 物理不连续**：这是全部精髓一句话——**从模型/请求的视角看，KV Cache 是一段连续的序列（逻辑连续）；但它在显存里实际存放，是零散分布在各个物理块里的（物理不连续）**。block table 就是这两者之间的翻译官。

**用一张图把三者串起来：**

```
请求 A 的逻辑视图 (它以为的连续序列):
  [tok0..15][tok16..31][tok32..47]
       │         │          │
   block table 翻译 (第几逻辑块 → 哪个物理块):
       ↓         ↓          ↓
物理显存 (实际零散存放, 和别的请求的块混在一起):
  [物理块#7] ... [物理块#3] ...... [物理块#9]
  (中间穿插着请求 B、C 的块, 但谁也不浪费)
```

### 2.4 为什么这样就解决了浪费

回到 2.1 的两个碎片：

- **内部碎片没了**：不再按 2048 预留，而是**用多少块、开多少块**。请求生成到 100 个 token，就只占用 `ceil(100/16)=7` 个块，一个块都不多占。**按需分配**。
- **外部碎片没了**：所有块**大小一样**，任何一个空闲块都能给任何请求用，显存里不会再有"不够大"的尴尬空隙。**统一规格，随取随用。**

> **额外的大红利：copy-on-write（写时复制）共享**。多个请求如果有相同的 prompt 前缀（比如同一个 system prompt），它们的这部分 KV block **可以物理上共享同一份**，只有当某个请求要修改时才复制——又省一大笔显存。这也是从操作系统 fork 进程那里借来的思想。**你 W7 骂 `torch.cat` 浪费时，大概率没想到还能玩到"多请求共享物理块"这一层——这就是读官方论文的价值。**

### 2.5 用最简代码把 block table 的机制摸清楚

论文不要你实现，但一段几十行的 Python 骨架，能让"逻辑连续/物理不连续"从抽象变具体。这就是你 W5-W6 要亲手造的东西的雏形：

```python
# 环境: 纯 Python, 演示 PagedAttention 的显存管理骨架 (不含真实 attention 计算)
# 目的: 把 "block table 如何把逻辑序列翻译到物理块" 摸清楚
class BlockAllocator:
    """管理物理块的分配与回收 —— 对应操作系统的物理内存管理器"""
    def __init__(self, num_blocks, block_size=16):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))  # 空闲物理块编号池

    def allocate(self):
        if not self.free_blocks:
            raise MemoryError("显存块用尽 —— 真实引擎此时会触发抢占/换出")
        return self.free_blocks.pop()               # 取一个空闲物理块

    def free(self, block_id):
        self.free_blocks.append(block_id)           # 请求结束, 归还物理块给池子

class Sequence:
    """一个请求的 KV Cache 视图 —— 只维护 block table, 不要求物理连续"""
    def __init__(self, allocator):
        self.allocator = allocator
        self.block_table = []       # ★ 核心: 逻辑块索引 -> 物理块编号 的映射表
        self.length = 0

    def append_token(self):
        # 当前最后一个块满了(或还没有块), 才申请新物理块 —— 这就是"按需分配"
        if self.length % self.allocator.block_size == 0:
            phys = self.allocator.allocate()
            self.block_table.append(phys)           # 逻辑上的新块, 落到某个零散物理块
        self.length += 1

    def locate(self, token_idx):
        """把逻辑 token 位置 翻译成 (物理块, 块内偏移) —— 这就是页表查询"""
        logical_block = token_idx // self.allocator.block_size
        offset = token_idx % self.allocator.block_size
        phys_block = self.block_table[logical_block]  # 查表: 逻辑->物理
        return phys_block, offset

# 演示: 两个请求交错申请, 物理块必然不连续, 但各自逻辑上都连续
alloc = BlockAllocator(num_blocks=8, block_size=16)
a, b = Sequence(alloc), Sequence(alloc)
for _ in range(20): a.append_token()   # A 生成 20 个 token -> 2 个块
for _ in range(20): b.append_token()   # B 也生成 20 个 -> 2 个块
print("请求A 的物理块:", a.block_table)  # 例如 [7, 6] —— 和 B 的块交错
print("请求B 的物理块:", b.block_table)  # 例如 [5, 4]
print("A 的第 17 个 token 在:", a.locate(17))  # (物理块, 偏移) —— 逻辑第2块的第1个
```

> 跑一下你会发现：A 和 B 的物理块编号是**交错、不连续**的，但通过各自的 `block_table`，每个请求都能把自己的 token **当成连续序列**来访问。**这几十行就是 PagedAttention 显存管理的灵魂**——真实的 vLLM 在这之上加了 CUDA kernel（让 attention 计算能直接吃这种分块的 KV）、抢占换出、前缀共享等，但骨架就是这个。

### 2.6 顺带一个概念：continuous batching（连续批处理）

论文和 landscape 里会一起出现的还有这个词，一并把直觉建好：

> **static batching（静态批处理）**：传统做法——凑齐一批请求一起算，**必须等批里最慢（最长）的那个请求全部生成完，整批才释放**。就像拼车必须等所有人都到目的地才算结束，先到的干等着，位置白白空占。

> **continuous batching（连续批处理，又叫 in-flight batching）**：某个请求生成完就**立刻**把它的位置腾出来，**马上塞进一个新的等待请求**，不等整批。就像出租车——谁到站谁下车，立刻接下一位。GPU 利用率因此大幅提升。

**为什么 continuous batching 依赖 PagedAttention？** 因为请求随时进出、长度各异，显存必须能**灵活地按块分配和回收**——这正是分页机制提供的能力。**两者是绝配**：PagedAttention 提供灵活显存，continuous batching 用它榨干 GPU。这也是为什么它俩在你的 landscape 里会被标成一组、都放在 W5-W6 实现。

### 2.7 三句话讲清 PagedAttention（完成标准自测）

> 1. **问题**：KV Cache 按最大长度预留连续显存，导致大量显存被空占和碎片浪费，有效利用率可能只有二三成。
> 2. **做法**：借用操作系统分页思想，把 KV Cache 切成固定大小的小块、按需分配，用一张块表把"逻辑上连续"翻译成"物理上零散"。
> 3. **结果**：碎片几乎消失，显存利用率大幅提升，还能让相同前缀的请求共享物理块，同时服务的请求数成倍增加。

---

## 3. 【产出】更新 `inference_optimization_landscape.md`

预研的落点：把今天读懂的两块，在你的领域地图上从"没听过"升级到"懂概念"。用一套状态标记法管理认知进度：

| 状态 | 含义 |
|------|------|
| ⬜ | 没接触 |
| 🔶 | 懂概念（知道解决什么问题、大致怎么做）—— **今天的目标** |
| 🟩 | 亲手实现过 |

今天要改的两行：

```markdown
## 推理优化技术地图
- FlashAttention        🟩 已手写 online softmax + 读原论文 IO 分析   [W8 已实现]
- PagedAttention        ⬜ → 🔶  懂了分页/blocktable/逻辑物理分离      [W5-W6 亲手实现]
- continuous batching   ⬜ → 🔶  懂了动态进出 batch, 依赖分页           [W5-W6 亲手实现]
- KV Cache 量化          ⬜  待预研
- 投机解码 (speculative) ⬜  待预研
```

> 这张地图的价值：**它让你随时知道"我在哪、下一步往哪走"**。预研的本质就是给地图上未知的格子先标个 🔶，等真正学到时，你不是从零开始，而是"把已知的框架填实"——学习速度天差地别。这正是今天开头说的"框架认知"。

---

## 4. 【副线·研】联络张老师/师兄：带着成果问，是今天杠杆最高的 1 小时

> **本周最重要的 1 小时在这里。** 这 1 小时的杠杆，比本周任何一个 3.5h 深块都高——因为它直接决定你整个暑假【研】线的加码程度。所以**先写 checklist，再发**，不打无准备之仗。

### 4.1 核心策略：为什么必须"带着成果问"

这是本节唯一最重要的道理，值得单独讲透：

> **空手问"我能不能署名"** → 显得功利、像在索取，老师会本能防备。
> **带着"论文做不到的真实硬件数据"问** → 你是在**展示已经创造的价值**，署名成了水到渠成的话题。

差别在于：前者是"我想要什么"，后者是"我已经贡献了什么、接下来能贡献更多"。**你手里正好有一张别人没有的牌**——AMK 论文是在别的硬件上做的，而你在 **H100 上真实跑通了 Llama-3.1-8B 并拿到了 ncu 数据**。这是"论文做不到的真实硬件 profiling"，是你独特的、已经产生的价值。**用价值开路，再谈回报，这是所有向上沟通的黄金结构。**

### 4.2 消息 checklist（先填好再发）

**第一部分 · 汇报成果（先给价值）：**
- [ ] AMK 在 H100 上**已跑通** Llama-3.1-8B-Instruct
- [ ] **已拿到 ncu profiling 数据**，附 report 的 2-3 个关键数字（如实测 cuBLAS 比值、最大热点 region 占比）
- [ ] 附上 Day5 列的**三个待深挖问题**（跨 SM 同步证据、热点 region 归因、和论文 0.60-0.72× 对标）——**展示你不只是跑了数据，还在主动思考**

**第二部分 · 三个要问清的问题（原计划 §12 点名发射周该问的）：**
- [ ] ① 这个课题**有没有出论文的计划**？（决定这条线的天花板）
- [ ] ② 我**能参与到什么程度、有没有署名可能**？（你的核心关切，但放在成果之后问）
- [ ] ③ 师兄对 AMK 这条线**接下来最想要什么数据**？（把自己嵌进团队真实需求，最能体现你"好用"）

### 4.3 一个可参考的消息骨架

```
张老师/师兄好：
汇报下 AMK 的进展——我已经在 H100 上把 Llama-3.1-8B-Instruct 跑通，
拿到了完整的 ncu profiling 数据。有几个数字挺有意思：[关键数字1]、[关键数字2]。

我注意到论文的实验是在 [别的硬件] 上做的，H100 作为 Hopper 新架构，
这份真实数据可能有独特的对照价值。我整理了三个想深挖的问题：
[三个问题]。

想请教三件事：
1. 这个课题后续有出论文的规划吗？
2. 如果我持续投入，有没有可能参与到论文里？
3. 接下来最需要我补哪方面的数据，我好安排下周的重点。
```

> **发出去就是今天的产出**（完成标准要求"已发出"，不是"写好了"）。回复内容直接决定你暑假【研】线要加多少码——所以别拖到"准备得完美"，成果已经够硬，今天就发。

---

## 5. 【整理块】GitHub profile 收口

把主页 README 收口，pin 好 4-5 个项目，让访客 30 秒看懂你的技术叙事线：

- [ ] **总 README** 写清一句话定位（如"AI Infra 方向，专注推理优化 / GPU kernel"）+ 学习主线
- [ ] **Pin 4-5 个项目**，按"从上层到底层"排序讲一个完整故事：
  - `nanoGPT`（Llama 化 + 三级验证）—— 会改架构、懂正确性验证
  - `week8_triton`（手写 FlashAttention/online softmax）—— 懂 kernel 层优化
  - `miniLLM-serve`（推理引擎，W5-W6 会加 PagedAttention）—— 懂服务层
  - 早期项目合集（micrograd / numpy 网络 / MNIST）—— 地基扎实
- [ ] 每个项目 README 顶部一句"**证明了什么**"（延续 Day4/Day5 一贯精神）

> 和前几天一脉相承：**portfolio 不是项目堆砌，是一条有方向的叙事线。** 今天读的 FlashAttention/PagedAttention，正好是 `week8_triton` 和 `miniLLM-serve` 两个 pin 项目的"理论注脚"——面试时你能把项目和论文串起来讲，深度立现。

---

## 6. 今日收尾 · 里程碑自测

**产出清单**：
- [x] 两篇论文的框架笔记（FlashAttention + PagedAttention，只读骨架）
- [x] W8"我 vs 官方"表补上"论文原文怎么说"一列
- [x] `inference_optimization_landscape.md`：PagedAttention / continuous batching 升级为 🔶
- [x] **已发出**给张老师/师兄的联络消息
- [x] GitHub profile README 收口 + pin 4-5 项目

**完成标准（今天硬指标）**：**能各用三句话向外行讲清 FlashAttention 和 PagedAttention 分别解决什么问题。** 见 1.6 和 2.7，合上笔记复述一遍。

**能过关的自测四问**：
1. FlashAttention 减少了计算量吗？它到底优化了什么？（没减 FLOPs，减的是 HBM 访存；memory-bound 优化）
2. online softmax 里的 `correction = exp(m - m_new)` 为什么是命门？（把旧累积结果拉到新最大值基准，保证分块=整行）
3. PagedAttention 的"逻辑连续、物理不连续"和操作系统的什么机制一一对应？（虚拟内存分页 + 页表）
4. continuous batching 为什么依赖 PagedAttention？（请求动态进出需要灵活的按块显存分配/回收）

> **一句话总结今天**：你今天没有"学新算法"，你在**给暑假要造的每一块，在领域地图上钉好坐标**——FlashAttention 砸计算墙、PagedAttention 砸显存墙，两者正交。更重要的是，你用手里那张"H100 真实数据"的牌，撬动了整条科研线的走向。**框架建好了，后面每学一块都知道往哪放。**

