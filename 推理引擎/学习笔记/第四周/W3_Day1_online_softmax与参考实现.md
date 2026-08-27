# 阶段二 · W3 Day 1 学习笔记 —— 论文精读 + online softmax 数学 + 参考实现

> **对应规划**：`阶段二W3_FlashAttention_逐日详细规划.md`（v1.1）→ W3 Day 1（8/27 周三）
> **今日目标**：回答"FA 的在线更新公式，凭什么数值稳定、凭什么不重算？"——把 m/l/acc 三行更新公式的每一行亲手推出来，并写死本周全部参考实现（母本）。
> **今日定位**：本周的数学地基日。今天推导错一行，后面六天的 kernel 全错。
> **前置**：W1 Day4 §7.3 的 online softmax numpy（你已重读）、W1 05_tiled_matmul（tiling 肌肉记忆）、W2 的 reduce.cuh 契约（"只有 thread 0 正确"）。

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `ref_attn.py` | **母本四件套**：attn_ref / attn_fa_style_ref / softmax_ref / softmax_online_ref | 复制进 `推理引擎/bench/` |
| `attn_harness.py` | 参考端封装：输入工厂 + 访存下界 + 三尺子 | 复制进 `推理引擎/bench/` |
| `test_attn_sanity.py` | attention 接口 sanity 测试（今天必须全绿） | 复制进 `推理引擎/tests/` |
| `reduce_max.cuh` | 追加到 reduce.cuh 的两个 max 归约 | 内容合并进 `推理引擎/csrc/reduce.cuh` |
| `softmax.cu` | 独立行 softmax 算子 v0（两遍版） | 复制进 `推理引擎/csrc/` |
| `backend_softmax_snippet.py` | CudaBackend.softmax + bindings/kernels 接入说明 | 合并进 `推理引擎/engine/backend.py` |

---

## 0. 今天的问题与全景图

### 0.1 一个问题的两种问法

- **凭什么数值稳定**：softmax 里有 exp，输入大一点（比如 logits=88）exp(88) 就溢出 fp32（>3.4e38）。标准解法是"减掉行最大值"——但**分块处理时，块 2 的最大值可能比块 1 大**，块 1 已经算好的结果怎么办？
- **凭什么不重算**：FlashAttention 的承诺是 S=QK^T 这个 T×T 矩阵**算完就扔**。但如果块 2 来了更大的 max，块 1 的 exp 值全都要按新 max 重新缩放——难道要把块 1 重算一遍？

答案就在三行更新公式里：**旧结果不用重算，乘一个 exp(差值) 就完成"整体缩水"**。今天把它推出来。

### 0.2 今天的完整闭环

```
① 前置核查（0.5h）：地基清单 + 引擎 attention 接口核对          §1
② 论文精读（1h）：FA §3 前向算法 + 两笔账 + IO 复杂度           §2
③ 手推（1.5h）：三行更新公式，本周最重要的一小时                §3
④ 母本（1h）：ref_attn.py 四件套 + 交叉验证                    §4
⑤ 造（2.5h）：attn_harness + sanity 测试（今天全绿）            §5
⑥ 研（1.5h）：softmax 算子 v0（两遍版）+ 接入 CudaBackend       §6
```

---

## 1. 前置核查（0.5h，别跳过）

### 1.1 地基清单

- [ ] `Desktop/01/05_tiled_matmul.cu` 存在，快速重读一遍"为什么分块"那一节
- [ ] W1 Day4 笔记 §7.3（online softmax 的 numpy 版）在手边——今天推导的每一步都和它互证
- [ ] W8 Day4 笔记（简化 fused attention）在手边——周末对比"简化版 vs 完整版"的差距在哪
- [ ] 引擎仓库 `git log --oneline -5` 干净，W2 收口已 commit

### 1.2 引擎 attention 接口核对（写进今天的笔记，后面六天都要用）

```powershell
# 在推理引擎仓库里执行，把答案抄进笔记：
grep -n "def attention" engine/backend.py        # 接口签名
grep -n "attention(" engine/model.py             # 调用点：看 q/k/v 的形状与 causal 用法
grep -n "n_embd\|n_head" engine/model.py         # head_dim = n_embd / n_head（大概率 64/768）
```

**要抄下来的四件事**（模板）：① `backend.attention(q,k,v,causal)` 的签名与默认值；② q/k/v 的形状约定（B,heads,T,D）；③ causal 在 prefill/decode 分别怎么传；④ KV Cache 拼接后 T_kv 与 T_q 的关系（decode 时 T_q=1、T_kv=start_pos+1）。**今天抄错一个，Day 4 接引擎时就要返工。**

---

## 2. 论文精读：FA §3 前向算法（1h）

### 2.1 先对齐符号（scaled dot-product attention，缩放点积注意力）

```
输入：Q (Tq×D)、K (Tk×D)、V (Tk×D)   ← 一个头的视角；D = head_dim（每头维度）
S   = Q·Kᵀ / √D                       ← 注意力分数矩阵 (Tq×Tk)；√D 防止点积方差随 D 膨胀
P   = softmax(S)（行方向）             ← 注意力权重：每行和为 1
O   = P·V                             ← 输出 (Tq×D)
```

**causal（因果）mask**：token t 只能看到 ≤t 的历史 → S 的第 t 行、列 > t 的位置置 −inf（exp(−inf)=0，被"看不见"）。注意引擎里 KV Cache 拼接后 **Tk ≥ Tq**，mask 的通用写法必须带偏移 `offset = Tk − Tq`：列 > 行 + offset 才掩（§4.1 代码里细讲）。

### 2.2 标准实现的两笔账（图 1 的解读）

标准实现是三个独立 kernel：S 写显存（T×T）→ softmax 读 S 写 P → O=P·V 再读 P。**S 和 P 这两个 T×T 矩阵各进出显存一次 = O(T²) 的 HBM 流量**。T=2048 时：S/P 各 2048²×4B ≈ 16.8 MB **每个头**——而 Q/K/V/O 加起来才 2048×64×4×4 ≈ 2 MB。**98% 的流量花在"路过"的中间矩阵上**。

FlashAttention 的答案：把 softmax 拆成"按 K 分块、在线合并"，S 的每一小块在 **SRAM（片上）里算完就扔**，永远不落 HBM——HBM 流量从 O(N²) 级掉回 O(N) 级。论文的 IO 复杂度结论：HBM 访问量从 Θ(Nd+N²) 降到 Θ(N²d²/M)（M = SRAM 容量）——今天只记住结论和直觉，推导是大二上的事。

### 2.3 论文 Algorithm 1 逐行对照（精读重点，~45min）

论文 §3 的伪代码（forward）翻译成中文加旁注：

```
1: 把 Q 按行分块（每块 BLOCK_M 行），K/V 按列分块（每块 BLOCK_N 列）
2: O 初始化为 0，l、m 初始化为 0、−inf          ← 三个状态量：和 / 最大值 / 加权输出
3: for 每个 Q 块 do
4:    加载 Q_i 到 SRAM；l_i=0，m_i=−inf
5:    for 每个 K/V 块 j do
6:       S_ij = Q_i·K_jᵀ / √D                   ← 小矩阵，只在 SRAM 里活一轮
7:       m̃_ij = rowmax(S_ij)                    ← 本块的局部最大值
8:       P̃_ij = exp(S_ij − m̃_ij)                ← 局部概率（相对本块 max）
9:       l̃_ij = rowsum(P̃_ij)                    ← 本块局部和
10:      m_i = max(m_i, m̃_ij)                   ← ★ 状态更新 1：跑动最大值
11:      l_i = l_i·exp(m_i_old − m_i_new) + l̃_ij·exp(m̃_ij − m_i_new)   ← ★ 状态更新 2：跑动和
12:      O_i = O_i·exp(m_i_old − m_i_new) + P̃_ij·V_j·exp(m̃_ij − m_i_new) ← ★ 状态更新 3：加权输出
13:   end for
14:   O_i = O_i / l_i                            ← 最后一步归一化
15: end for
```

对照 [Triton 教程 FA 章节](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)（只读不实现）：它的 `m_i, l_i, acc` 三个变量就是伪代码的 m/l/O——**编译器替你管理了片上循环，但状态更新的三行一模一样**。这就是"看懂编译器在替你干什么"的时刻。

### 2.4 论文没直说、但你必须想清楚的两件事

1. **为什么 softmax 要减 max**：exp 会溢出。减掉行最大值后 exp 的参数全部 ≤0，输出落在 (0,1]，数值稳定。**任何手写 softmax 的第一步都是 max 归约**。
2. **为什么分块后不能各算各的**：块 1 算的 P 用的是块 1 的 max，块 2 来了更大的 max 后，**正确的 softmax 分母要按全局 max 重新算**。要么重算块 1（违背"不重算"），要么用一个代数恒等式把块 1 的结果"换算"到新尺度——§3 推的就是这个恒等式。

---

## 3. ★ 手推 online softmax（1.5h，本周最重要的一小时）

### 3.1 两遍版的回顾（你 W1 的知识）

行 softmax 的标准写法要**两遍**：第一遍找行最大值 m；第二遍算 `l = Σ exp(x−m)` 并写 `y = exp(x−m)/l`。两遍之间必须等 m 出来——**这就是"流式读一遍"做不到标准 softmax 的原因**。

### 3.2 问题的精确表述

把一行 x 分成两块 A=[3,1]、B=[4,1]。块 A 的局部 max 是 3，块 B 的是 4。**全局 max 是 4，不是 3**——块 A 里按 3 算的 exp 值，拿到全局尺度下全部"虚高"了 e 倍，必须缩水。**缩水多少？能不能不重读 A 就完成缩水？**

### 3.3 推导（一步步，每一步都写清理由）

**引理（rescale 恒等式）**——整个推导只有这一个代数事实：

```
exp(x − m*) = exp(x − m)·exp(m − m*)     ← 指数相减拆开：先按旧尺 m 算，再乘一个因子换算到新尺 m*
```

**l 的合并**：设块 A 已按自己的 max m_A 算出局部和 l_A = Σ_{A} exp(x−m_A)，块 B 同理 l_B。全局 max m* = max(m_A, m_B)。按引理把两块都换算到 m* 尺度：

```
l* = Σ_{A∪B} exp(x−m*)
   = Σ_A exp(x−m*) + Σ_B exp(x−m*)
   = Σ_A [exp(x−m_A)·exp(m_A−m*)] + Σ_B [exp(x−m_B)·exp(m_B−m*)]
   = l_A·exp(m_A−m*) + l_B·exp(m_B−m*)          ★ 这就是状态更新 2
```

**acc（加权输出）的合并**：acc 的每一项是 exp(x−m*)·v，比 l 的每一项多乘一个 v——合并公式**完全同构**：

```
acc* = acc_A·exp(m_A−m*) + acc_B·exp(m_B−m*)    ★ 这就是状态更新 3
```

**为什么"每来一个新块"的增量形式成立**：把"已有累积"当作块 A、把"新块"当作块 B，m* = max(m_old, m_new)——归纳一下就得到伪代码第 10–12 行的三行更新。**每次 rescale 因子都是 exp(差值)，因为引理里只有差值出现。**

### 3.4 手算一遍（数字全列出来，亲手再算一次）

x = [3, 1, 4, 1]，分块 A=[3,1]、B=[4,1]：

```
块 A：m_A = 3；l_A = e^0 + e^(−2) = 1 + 0.1353 = 1.1353
块 B：m_B = 4；l_B = e^0 + e^(−3) = 1 + 0.0498 = 1.0498
合并：m* = 4
      l* = l_A·e^(3−4) + l_B·e^(4−4) = 1.1353×0.3679 + 1.0498×1
         = 0.4177 + 1.0498 = 1.4675
直接验证：l* = e^(3−4) + e^(1−4) + e^(4−4) + e^(1−4)
            = 0.3679 + 0.0498 + 1 + 0.0498 = 1.4675  ✓ 分毫不差
```

### 3.5 三个"为什么"的最终答案（今天的自测题答案）

1. **凭什么数值稳定**：每一步的 exp 参数都是"相对当前最大值的差" ≤ 0，永不溢出；
2. **凭什么不重算**：旧结果（l_old、acc_old）只需要**乘一个标量因子** exp(m_old − m_new) 就完成尺度换算——状态量 O(1)，不碰旧数据；
3. **凭什么 O = acc/l 正确**：acc/l = Σ exp(x−m*)·v / Σ exp(x−m*) = Σ softmax 权重·v，正是 P·V 的定义。

### 3.6 ★ 一个必须写清的辨析（防止概念混装）

**online 省的是"第二次归约遍"，不是"第二次读"。** 在 attention 里省读靠的是 tiling（S 不落 HBM）；而独立行 softmax 算子无论两遍版还是 online 版，**写回 y 时都要再读一次 x**（y_i = exp(x_i − m)/l 需要 x_i）。这和 W2 Day3 板斧四的"读两遍 vs 暂存"是同一个取舍——W3 Day2 研线的 softmax v1 会用"online 归约 + 寄存器暂存"把重读也收掉。**今天先把两个概念分清。**

> **历史注脚（联网核实）**：online softmax 不是 FlashAttention 发明的——2018 年 Milakov & Gimelshein 的论文 *[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)* 就提出了 running max/sum 的在线归一化，FA 的贡献是把它装进 tiling 框架、让 S 永不落 HBM。**"新论文 = 旧数学 + 新框架"是读论文的重要视角**：你学到的三行更新公式，比 FA 论文早四年。

---

## 4. 参考实现：母本四件套（1h）

配套 `ref_attn.py`。四个函数，从今天起就是后面六天所有 kernel 的"逐行翻译母本"（reference implementation）——**母本的每一行都要能被 CUDA 代码一一对应**。

### 4.1 `attn_ref`：永远正确的基线（重点看 causal mask 的通用写法）

```python
def attn_ref(q, k, v, causal=True):
    # q,k,v: (B, H, T, D)，fp32。引擎接口的形状约定（§1.2 抄来的那四件事）
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    s = torch.einsum("bhtd,bhsd->bhts", q, k) * scale       # (B,H,Tq,Tk)
    if causal:
        offset = Tk - Tq                     # ★ KV Cache 拼接后 Tk ≥ Tq：
        rows = torch.arange(Tq, device=q.device) + offset   #   查询的真实绝对位置
        cols = torch.arange(Tk, device=q.device)
        s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhts,bhsd->bhtd", p, v)
```

**offset 为什么必须有**：decode 时 T_q=1、T_kv=start_pos+1，查询的绝对位置是 start_pos 而不是 0——mask 若写死 `col > row` 会把整个历史误掩掉。`col > row + (Tk−Tq)` 在 Tq=Tk 时退化成标准因果掩码，在 decode 时正确放行全部历史。**这是 Day 4 接引擎前就该想明白的事，今天用测试锁死它。**

### 4.2 `attn_fa_style_ref`：CUDA 母本（逐行对应将来的 kernel）

```python
def attn_fa_style_ref(q, k, v, causal=True, block_n=64):
    B, H, Tq, D = q.shape
    Tk = k.shape[2]
    scale = 1.0 / math.sqrt(D)
    offset = Tk - Tq
    o = torch.zeros(B, H, Tq, D, device=q.device)            # ← 将来是寄存器里的 acc
    l = torch.zeros(B, H, Tq, device=q.device)               # ← 将来是寄存器里的 l
    m = torch.full((B, H, Tq), float("-inf"), device=q.device)  # ← 寄存器里的 m
    for j0 in range(0, Tk, block_n):                         # ← 外层循环 = CUDA 的 K 块循环
        kj = k[:, :, j0:j0 + block_n, :]                     # ← 将来 load 进 shared
        vj = v[:, :, j0:j0 + block_n, :]
        s = torch.einsum("bhtd,bhsd->bhts", q, kj) * scale   # ← S_ij 只在片上活一轮
        if causal:
            cols = torch.arange(j0, j0 + kj.shape[2], device=q.device)
            rows = torch.arange(Tq, device=q.device) + offset
            s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
        m_j = s.max(dim=-1).values                            # ← 本块 rowmax（可能 -inf）
        m_new = torch.maximum(m, m_j)                         # ★ 状态更新 1（先定新尺）
        # ★ 3D 张量加新轴必须 [:, :, :, None]（=unsqueeze(-1)）：
        #   [:,:,None] 会把轴插到中间（B,H,1,Tq）→ 沿错误轴相减/相乘/相除 = 静默错
        p = torch.exp(s - m_new[:, :, :, None])               # ← P 直接在新尺上算
        l_j = p.sum(dim=-1)                                   # ← rowsum（将来 warp 归约）
        alpha = torch.exp(m - m_new)                          # ★ 旧累积的 rescale 因子
        l = l * alpha + l_j                                   # ★ 状态更新 2（单因子形式）
        o = o * alpha[:, :, :, None] + torch.einsum("bhts,bhsd->bhtd", p, vj)  # ★ 更新 3
        m = m_new
    return o / l[:, :, :, None]
```

**为什么母本用"单因子形式"而不是论文的 alpha+beta 双因子**（重要工程决策，Day3 的 CUDA 也照此实现）：

- 论文形式：P 按本块 max（m_j）算，合并时新块还要乘 beta = exp(m_j−m_new)；
- 单因子形式：**先算 m_new = max(m_old, m_j)，让 P 直接在新尺度上算**——新块贡献不再需要 beta，旧累积只乘一个 alpha；
- **关键收益：全掩块免疫**。causal 下某个块可能对某些行完全被掩（m_j = −inf）——论文形式里 `exp(−inf − (−inf)) = NaN` 直接污染整行；单因子形式里 `exp(−inf − 有限值) = 0`，天然无 NaN（FA2 的 CUDA 实现需要专门的 isinf 分支处理这事，Triton 教程的单因子形式则完全规避）。
- 数学等价：把论文形式的新块项 `l_j·exp(m_j−m_new)` 展开，正是"按 m_new 重算的块和"。
- **来源核实**：这正是 [Triton 官方教程 06-fused-attention.py](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) 的真实写法（`m_ij = tl.maximum(m_i, tl.max(scores,1))` → `p = exp2((scores−m_ij)·log2e)` → `alpha = exp2((m_i−m_ij)·log2e)`，无 beta）——本母本是它的 torch 逐行版，Day 3 的 CUDA 再逐行翻译一次。

**逐行对应表**（写进笔记，Day 2/3 翻译时照这张表）：

| 母本这行 | 将来的 CUDA 结构 |
|---|---|
| 外层 for j0 | kernel 内的 K/V 块循环 |
| kj/vj 切片 | `__shared__` 加载 + `__syncthreads` |
| einsum QK^T | 每 lane 算 S 的一列 + warp 内部分积 |
| `.max(dim=-1)` / `.sum(dim=-1)` | `warp_reduce_max` / `warp_reduce_sum`（或 v0 的 block 级） |
| alpha/beta | 标量 exp 差，每个线程各自算 |
| o、l、m | 每行三组寄存器状态 |

### 4.3 交叉验证（今天必须全绿）

```python
# ref_attn.py 自测：attn_fa_style_ref 必须与 attn_ref 一致（这才是"母本可信"）
# 形状矩阵覆盖：Tq=Tk、decode 形状(Tq=1,Tk>1)、非对齐(63/65)、D=64 与 D=128、causal 开/关
# 判据：allclose(atol=1e-4, rtol=1e-4) + cosine ≥ 0.9999
```

**为什么 atol 给 1e-4 而不是 1e-6**：分块累加的顺序与 torch.softmax 不同，fp32 下尾数位差异 ~1e-6 正常，1e-4 是"算法等价"的合理容差（W2 的纪律：容差要按实现路径差异定，不是越紧越好）。

### 4.4 四个今天就会踩的坑（速查表 §7 有完整版）

① **mask 方向写反**（`col < row` 掩的是自己）；② **忘了 offset**（decode 形状全灭）；③ **m 的初值**——`-inf` 的 `exp(m−m_new)` 在 torch 里是 0（正确），但 CUDA 里 `expf(-inf−x)` 依赖写法，Day 3 会用 `__expf` 前先判断或直接用 fmaxf 组合规避（v1 的坑①，今天埋好意识）；④ **★ 3D 张量的 None 轴位**——`l[:,:,None]` 得到 (B,H,**1**,Tq) 而不是 (B,H,Tq,**1**)，`o/l` 会沿错误轴相除、**静默算错且不报错**（本文件夹母本初版就栽在这里，自测抓到后已修复）；⑤ **★ 全掩块 NaN**——causal 下某个块对某些行完全被掩时，论文双因子形式的 `exp(−inf−(−inf))` = NaN 污染整行；单因子形式（先定 m_new 再算 P）天然免疫，本母本已采用。**这五个坑里 ④⑤ 都是"母本必须当天交叉验证"的活证据。**

---

## 5. 【造】测试基建（2.5h）

### 5.1 `attn_harness.py`：参考端的统一封装

```python
# 设计理由（今天 kernel 还没进来，先立规矩）：
#   ① 输入工厂 make_inputs(seed)：所有实现吃同一批数据——"同一把尺子"（W2 三纪律）
#   ② bytes_moved = (B·H·Tq + 2·B·H·Tk + B·H·Tq)·D·4   ← 读 Q + 读 K,V + 写 O 的 HBM 下界
#      （注意：S/P 不算——FA 的承诺就是它们不落 HBM；用这个下界当分母，%peak 才有意义）
#   ③ compare()：allclose / cosine / max_rel 三尺子（引擎仓库里可换用 bench_harness.compare）
```

### 5.2 `test_attn_sanity.py`：今天全绿的测试

```python
SHAPES = [
    ((2, 4, 64, 64),  (2, 4, 64, 64)),    # prefill：Tq = Tk，对齐
    ((2, 4,  1, 64),  (2, 4, 63, 64)),    # decode：Tq=1，Tk=63（KV Cache 拼接后）
    ((2, 4, 63, 64),  (2, 4, 65, 64)),    # 非对齐尾巴
    ((2, 4, 16, 128), (2, 4, 16, 128)),   # ★ D=128：工业标准 head_dim（v1.1 修订）
]
@pytest.mark.parametrize("causal", [True, False])
def test_fa_style_matches_ref(qk_shape, causal):
    ...  # attn_fa_style_ref vs attn_ref，atol=rtol=1e-4
```

**每个用例的意图**：decode 形状专测 offset 逻辑；非对齐专测分块尾巴（block_n=64 整除不了 65）；D=128 保证"本周产出直接对标工业形状"（Llama-8B 的 head_dim）。CPU/GPU 都可跑（纯 torch 参考，无 kernel），本机无 GPU 也不阻塞。

### 5.3 为什么"今天 kernel 不进引擎"

接口先独立验干净：今天绿的是**参考与测试基建**，明天 CUDA v0 一进来，直接和 `attn_fa_style_ref` 对拍——**每一步都有母本兜底，调试从"找 bug"变成"找差异"**。

---

## 6. 【研】softmax 算子 v0（1.5h）

### 6.1 两遍版 kernel（配套 `softmax.cu`，结构 = W2 rmsnorm 的兄弟）

```cuda
template <int BLOCK>
__global__ void softmax_two_pass(const float* __restrict__ x, float* __restrict__ y, int H) {
    const int row = blockIdx.x;                          // 排布与 rmsnorm 一致：一 block 一行
    const float* xr = x + (size_t)row * H;
    float*       yr = y + (size_t)row * H;

    float m = -INFINITY;                                 // ★ 初值 -inf（W3 Day3 坑①的伏笔）
    for (int i = threadIdx.x; i < H; i += BLOCK)
        m = fmaxf(m, xr[i]);
    m = block_reduce_max<BLOCK>(m);                      // ★ 契约：只有 thread 0 正确

    __shared__ float sm;
    if (threadIdx.x == 0) sm = m;                        // 广播两步（W2 的标准姿势）
    __syncthreads();
    m = sm;

    float l = 0.f;
    for (int i = threadIdx.x; i < H; i += BLOCK)
        l += __expf(xr[i] - m);                          // exp 参数 ≤ 0，数值稳定
    l = block_reduce_sum<BLOCK>(l);
    if (threadIdx.x == 0) sm = l;
    __syncthreads();
    l = sm;

    for (int i = threadIdx.x; i < H; i += BLOCK)         // 写回：又读了一次 x（§3.6 的辨析）
        yr[i] = __expf(xr[i] - m) / l;
}
```

- **`__expf`**：硬件近似指数（MUFU.EX2 指令），相对误差 ~2⁻²² 量级——fp32 下与 torch.exp 的差落在 1e-4 容差内；工业惯例：推理热点里用近似 exp 换速度，训练精度敏感处才换 `expf`。
- **诚实标注**：v0 实际读 x **三次**（max 归约 / exp+sum 归约 / 写回重读）——"两遍版"指两次归约遍。第 3 遍与 Day2 研线的 v1（online + 寄存器暂存）形成对照组，这正是论文线"两变体"的意义。

### 6.2 `reduce_max.cuh`：追加到 reduce.cuh 的两个函数

```cuda
__inline__ __device__ float warp_reduce_max(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    return v;
}

template <int BLOCK>
__inline__ __device__ float block_reduce_max(float v) {
    static_assert(BLOCK % 32 == 0 && BLOCK / 32 <= 32, "BLOCK must be 32..1024");
    __shared__ float smem[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();
    v = (threadIdx.x < BLOCK / 32) ? smem[threadIdx.x] : -INFINITY;   // ★ 初值 -inf 不是 0！
    if (wid == 0) v = warp_reduce_max(v);
    return v;                                            // ★★ 契约同 reduce.cuh：只有 thread 0 正确
}
```

**和 `block_reduce_sum` 的唯一区别**：第二级里"不参与读槽位的线程"初值——sum 用 `0.0f`（加法单位元），max 用 `-INFINITY`（最大值运算的单位元）。**写错这个，含全负数的行会算出 0 当最大值——静默错。**

### 6.3 接入三件套（配套 `backend_softmax_snippet.py`）

1. `bindings.cpp` 加：`m.def("softmax", &softmax_cuda, "row softmax (CUDA)", py::arg("x"), py::arg("block") = 256);`
2. `kernels.py` 的 sources 加 `"softmax.cu"`（改完重跑即重编，Day 1 的红利）；
3. `CudaBackend.softmax`：

```python
def softmax(self, x):
    from .kernels import softmax as cuda_softmax
    if not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous():
        return torch.softmax(x, dim=-1)                  # ← 回退（Day 1 的分层思想）
    return cuda_softmax(x, self.block)
```

### 6.4 数值对照（今天就要做）

本机与 `softmax_ref` 对拍：形状 (rows, H) ∈ {(256,4096), (32,4096), (1,4095)}，`assert_close(atol=1e-4, rtol=1e-4)`。**（1,4095）是尾巴用例**——grid-stride 天然覆盖，但必须测过才算数（W2 纪律）。

---

## 7. 常见错误与调试速查表（Day 1 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| fa_style 与 ref 只在 decode 形状上差 | causal mask 忘了 offset=Tk−Tq | §4.1 的通用写法 + test 里 decode 用例锁死 |
| 全负数的行 softmax 结果全 0 或 NaN | max 归约初值用了 0 | `-INFINITY`（reduce_max.cuh 第二级也是） |
| cosine 好但 max_rel 差 | 容差按实现路径定：1e-4 合理 | 别把容差拧到 1e-6 折磨自己（§4.3） |
| 母本跑得奇慢 | einsum 在 CPU 上大矩阵 | 参考实现只求对不求快；本机跑小形状 |
| softmax v0 输出 NaN | l 为 0 / 广播漏 `__syncthreads` | 检查广播两步（写→屏障→读）；l>0 恒成立（exp>0） |
| 结果对但和 ≠1 | 写回那遍的 m/l 与中间值不一致 | 检查两次广播后是否全员读到新值 |
| 母本数值"看起来对"但差 1e-2 级 | ★ 3D 张量 `[:,:,None]` 轴插在中间（B,H,1,Tq）→ 沿错误轴相除 | 一律 `[:, :, :, None]` 或 `unsqueeze(-1)`；任何广播前先打印 shape 断言 |
| 非对齐形状（如 63/65）出现 NaN | ★ 全掩块：causal 下块内所有列被掩 → m_j=−inf → exp(−inf−(−inf))=NaN | 单因子形式：先定 m_new 再算 P（exp(−inf−有限值)=0）；母本已采用 |
| pytest 找不到模块 | 没从仓库根/正确 sys.path 跑 | `python -m pytest tests/test_attn_sanity.py` |
| git 状态不干净 | W2 收口没 commit | Day 1 早段核查清单第 3 条 |

---

## 8. 完成标准自测（先默写再对答案）

**规划题**：能不看纸推完 m/l/acc 三行更新公式，并说清"为什么每次 rescale 都是 exp(差值)"。
*答案要点*：引理 exp(x−m*) = exp(x−m)·exp(m−m*) 是唯一代数事实；l* = l_A·exp(m_A−m*) + l_B·exp(m_B−m*)（acc 同构）；增量形式 = 把"已有累积"当块 A；因子是 exp(差值) 因为引理中只有差值出现。

**附加题（今天的衍生自测）**：
1. 数值稳定从哪来？→ 所有 exp 参数 = "相对当前 max 的差" ≤ 0，永不溢出。
2. "online 省第二次归约遍、不省第二次读"——attention 里省读靠什么？→ tiling（S 不落 HBM）；standalone softmax 写回仍重读 x，Day2 研线 v1 用 stash 收掉。
3. 母本的哪一行对应将来的 warp_reduce_max？→ `s.max(dim=-1)` 与 `p.sum(dim=-1)`。

---

## 9. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] `W3_Day1/{学习笔记.md, ref_attn.py}`（母本四件套，交叉验证全绿）
- [ ] `bench/attn_harness.py` + `tests/test_attn_sanity.py`（今天全绿）
- [ ] `csrc/{reduce_max.cuh, softmax.cu}` + `CudaBackend.softmax`（v0，与参考对拍通过）
- [ ] 前置核查四件事抄进笔记（§1.2 模板）

**明日预告（Day 2）**：把 `attn_fa_style_ref` 逐行翻译成 CUDA——前向 v0（一个 block 一行 Q，block 级归约）。今天的母本对应表（§4.2）就是明天的翻译字典；softmax v1（online + 暂存）进研线。

---

## 附 A：术语速查表（Day 1）

| 名词 | 一句话解释 |
|---|---|
| scaled dot-product attention（缩放点积注意力） | S=QKᵀ/√D → softmax → ×V；√D 防止点积方差膨胀 |
| causal mask（因果掩码） | 列 > 行+offset 置 −inf；offset=Tk−Tq 是 KV Cache 形状下的通用写法 |
| online softmax（在线 softmax） | 分块流式算 softmax：状态 m/l/acc + 每块一次 rescale |
| rescale 因子（重缩放因子） | exp(m_old − m_new)：把旧累积换算到新尺度的标量 |
| running max / running sum | 跑动最大值 m / 跑动和 l——在线算法的两个状态量 |
| 母本（reference implementation） | 逐行可翻译成 CUDA 的 torch 参考实现 |
| 两笔账 | S、P 两个 T×T 中间矩阵进出 HBM 的 O(T²) 流量 |
| IO 复杂度 | HBM 访问量量级：标准 Θ(Nd+N²) vs FA Θ(N²d²/M) |
| 单位元（identity element） | 归约初值：sum 用 0、max 用 −inf——写错 = 静默错 |
| `__expf` | 硬件近似指数指令（MUFU.EX2），~2⁻²² 精度，推理热点标配 |
| 契约（thread 0） | 归约结果只在 thread 0 正确——下游必须先广播（W2 纪律） |
| 三遍 vs 两遍归约 | v0 读 x 三次（max/sum/写回）——"两遍"指归约遍数，Day2 v1 收掉重读 |

## 附 B：参考与延伸

- FlashAttention 论文（今天精读 §3）：https://arxiv.org/abs/2205.14135
- FlashAttention-2（周末对照）：https://arxiv.org/abs/2307.08691
- **Online normalizer calculation for softmax（Milakov & Gimelshein, 2018——online softmax 的源头，比 FA 早四年）**：https://arxiv.org/abs/1805.02867
- Triton 教程 FA 章节（编译器视角旁注）：https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- Triton 教程源码（单因子形式的原始出处，母本逐行对应）：https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
- W1 Day4 笔记 §7.3（online softmax numpy 版，互证对象）
- W2 笔记（reduce.cuh 契约 / 广播两步 / 1e-4 容差纪律）
