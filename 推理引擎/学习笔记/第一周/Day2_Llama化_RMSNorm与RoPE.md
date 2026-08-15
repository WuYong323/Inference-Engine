# Day 2 · Llama 化 ①：RMSNorm + RoPE（本周最深的一天）

> **今天只回答一个问题**：为什么现代 LLM（Llama / Qwen / DeepSeek……）**全都抛弃了"给每个位置加一个向量"的绝对位置编码，改用"旋转"？**
>
> 这是 W0 含金量最高的一天。它难，但难在"想通"，不难在"写完"——一旦你在纸上亲手推出"旋转后点积只含 (m−n)"，剩下的代码只是把这个几何事实翻译成张量运算。
>
> **本文沿用你的三段式**：原理 + 直觉类比 → 可运行代码（含底层视角）→ 工业锚点。**核心目标**：让你能脱稿在纸上推 2 维 RoPE 点积，并讲清"为什么 K 缓存的是旋转后的值"。

---

## 0. 今日地图

```
┌─ 主线深块 3.5h ──────────────────────────────────────────────┐
│ 先懂(1h)：RoPE 直觉 + 纸上推导 2 维点积只含 (m−n)               │
│ 再写(2h)：engine/rope.py（precompute_freqs_cis + apply_rope）  │
│           tests/test_rope.py 两个性质测试通过                   │
│           model.py：LayerNorm→RMSNorm(走 backend)、去绝对位置、  │
│                     attention 接入 RoPE                         │
│ 工业锚点(0.5h)：RoPE 与 KV Cache 的关系 + 长上下文外推占位       │
├─ 副线块 1.5h：tech_notes/rope_from_scratch.md（三段式串 W6 笔记）│
├─ 整理块 1h：nanoGPT 仓库 README 打磨（第一屏讲"证明了什么"）      │
└──────────────────────────────────────────────────────────────┘
```

本文重心：**原理推导 + rope.py 实现 + KV Cache 锚点**，这三块是今天真正的"新认知"。

---

## 第一部分：问题背景 —— 位置编码到底在解决什么问题？

### 1.1 先回到 W6 的一个事实：注意力是"位置盲"的

你 W6 已经知道：注意力（Attention）算的是 `softmax(QKᵀ/√d) · V`。**这里面没有任何一项跟"token 在第几个位置"有关。**

举个例子，把句子 `"猫 追 狗"` 打乱成 `"狗 追 猫"`，只要词还是那三个，注意力算出来的每一对 token 相似度**完全一样**——它是一个"词袋"运算，天生看不见顺序。

> **一句话**：注意力对输入做的是"集合"运算，不是"序列"运算。但语言是有顺序的（"猫追狗"和"狗追猫"意思相反）。所以必须额外**注入位置信息**，否则模型永远分不清语序。这就是**位置编码（Positional Encoding，PE）**要解决的唯一问题。

> **英文名词：Positional Encoding（位置编码，PE）**
> 给模型注入"token 排在第几"的机制。因为注意力本身位置盲，位置编码是把"顺序"这个信息塞进模型的唯一入口。

---

### 1.2 老办法：绝对位置编码（Absolute PE），以及它的三个痛点

> **英文名词：Absolute Positional Encoding（绝对位置编码）**
> 给"第 m 个位置"造一个固定的向量 `p_m`，直接加到该位置的 token 嵌入上：`x_m ← token_emb + p_m`。
> - **正弦版（Sinusoidal，原始 Transformer）**：`p_m` 用不同频率的 sin/cos 拼出来，是写死的公式。
> - **可学习版（Learned，BERT/GPT-2/nanoGPT）**：`p_m` 是一张 `[max_len, d]` 的可训练查找表（nanoGPT 里的 `wpe`）。

**类比**：绝对位置编码 = 给会场里每个人发一张写着**座位号**的牌子。模型想知道"3 号和 7 号隔多远"，得**自己从两张牌子的数字里学出"差 4"**这件事。

它的三个痛点，正是现代 LLM 抛弃它的原因：

1. **相对关系要"学"，学不牢。** 语言里真正重要的往往是**相对距离**（"这个词"和"上一个词"的关系），而不是绝对坐标。绝对 PE 把绝对坐标喂进去，逼模型自己去推导相对关系——这是"绕远路"，学出来的还不稳。

2. **外推能力差（Length Extrapolation）。** 可学习版最致命：训练时最长 1024，那第 1025 号位置的牌子**压根没造过**，模型一遇到更长的输入直接懵。

   > **英文名词：Length Extrapolation（长度外推）**：模型在比训练时更长的序列上还能不能正常работать。绝对 PE（尤其可学习版）外推几乎为 0，这是长上下文时代它出局的直接原因。

3. **位置和内容"焊死"在一起。** 位置向量直接加到 token 嵌入上，位置信息和语义信息从此揉在一个向量里，后面每一层都得带着这个"包袱"，不干净。

---

## 第二部分：核心原理 —— RoPE 的直觉与推导（今天的命门）

### 2.1 RoPE 是什么：把"加位置"换成"转角度"

> **英文名词：RoPE（Rotary Position Embedding，旋转位置编码）**
> 由苏剑林（追一科技）提出、被 Llama 系列发扬光大的位置编码方案。核心动作：**不给向量"加"任何东西，而是把 Q、K 向量按它所在的位置 m，在平面里"旋转"一个角度 m·θ。**

**类比（把上面座位号的类比升级）**：

- 绝对 PE：发座位号牌子，模型自己算"3 号和 7 号差 4"。
- **RoPE**：让每个人**按座位号转过一个角度**（3 号转 3θ，7 号转 7θ）。这样任意两人**面对面的夹角，天然就等于他们的座位号之差 ×θ**——不用学，几何上直接成立。

这句话是 RoPE 的全部灵魂：**把"相对位置"从"要模型去学的东西"，变成"数学上必然成立的几何事实"。** 下面我们把这个"必然成立"亲手推出来。

---

### 2.2 手把手推导：2 维情形，点积只含 (m−n)

这是今天你必须能在纸上复现的部分。别跳，跟着算一遍，你会亲眼看到 `(m−n)` 从代数里"长"出来。

**设定**：取 Q 向量的一对分量 `q = (q₁, q₂)`，它在位置 m。RoPE 对它做一个标准的 2 维旋转（旋转矩阵 `R`）：

```
        ┌ cos(mθ)   −sin(mθ) ┐
R(mθ) = │                    │
        └ sin(mθ)    cos(mθ) ┘
```

旋转后：
```
q'_m = R(mθ)·q = ( q₁cos(mθ) − q₂sin(mθ) ,  q₁sin(mθ) + q₂cos(mθ) )
```
同理，K 向量的一对分量 `k = (k₁, k₂)` 在位置 n，旋转后：
```
k'_n = R(nθ)·k = ( k₁cos(nθ) − k₂sin(nθ) ,  k₁sin(nθ) + k₂cos(nθ) )
```

**现在算注意力真正要的东西——它俩的点积 `q'_m · k'_n`**（把两个分量各自相乘再相加，耐心展开）：

```
q'_m · k'_n
= (q₁cos(mθ) − q₂sin(mθ))(k₁cos(nθ) − k₂sin(nθ))      ← 第一分量相乘
+ (q₁sin(mθ) + q₂cos(mθ))(k₁sin(nθ) + k₂cos(nθ))      ← 第二分量相乘
```

全部展开、按 `q_i k_j` 归类，用两个高中三角恒等式收拢：
- `cos(mθ)cos(nθ) + sin(mθ)sin(nθ) = cos(mθ − nθ)`
- `sin(mθ)cos(nθ) − cos(mθ)sin(nθ) = sin(mθ − nθ)`

最后得到（建议你自己合并一次，验证没错）：

```
q'_m · k'_n = (q₁k₁ + q₂k₂)·cos((m−n)θ) + (q₁k₂ − q₂k₁)·sin((m−n)θ)
```

**盯住这个结果**：等号右边，`q`、`k` 的分量是固定的，唯一和位置有关的是 `cos((m−n)θ)` 和 `sin((m−n)θ)`——**只含 `(m−n)`，绝对位置 m、n 各自消失了！**

这就是"魔法"的真相：**我们用绝对角度（m 转 mθ，n 转 nθ）分别旋转，但点积只依赖它们的角度差 (m−n)θ。** 相对位置不是设计出来强加的，是旋转的代数结构里天然掉出来的。

> **矩阵语言的一行证明（给想看本质的你）**：
> 因为旋转矩阵满足 `R(a)ᵀ = R(−a)` 且 `R(a)R(b) = R(a+b)`，所以
> `q'ᵀ k' = (R(mθ)q)ᵀ(R(nθ)k) = qᵀ R(−mθ)R(nθ) k = qᵀ R((n−m)θ) k`。
> 右边只有 `(n−m)`。三行结束。上面那一大段手工展开，就是这三行的"慢镜头回放"。

---

### 2.3 从 2 维推广到 d 维：多个频率，像钟表的指针

真实的 head_dim 是 64、128 这种。RoPE 的做法是：**把 d 维向量两两分成 d/2 组，每一组是一个独立的 2 维平面，各转各的角度。**

关键在于：**每一组用不同的旋转频率 θ_i**：

```
θ_i = base^(−2i/d) = 1 / base^(2i/d)        i = 0, 1, ..., d/2−1     (base 通常取 10000)
```

- `i=0`：θ 最大 → 转得**最快**（高频）。
- `i=d/2−1`：θ 最小 → 转得**最慢**（低频）。

> **类比：钟表的秒针、分针、时针。**
> - 秒针（高频）转得快 → 能分辨"相邻几秒"这种**近距离**差别，但转太快，隔几分钟就绕圈重合了（分不清远）。
> - 时针（低频）转得慢 → 分辨不了几秒的差别，但能区分"几小时"这种**远距离**。
>
> 单独一根针会"绕圈混淆"（转超过 360° 就和某个近位置撞脸），**但多根不同速度的针组合起来**，就像钟表用三根针唯一确定一个时刻一样——d/2 个不同频率组合，才能让模型既分得清"隔 1 个词"也分得清"隔 1000 个词"。这就是为什么必须多频率，而不是所有维度用同一个 θ。

**一句话小结**：RoPE = 把向量切成 d/2 个 2 维平面 → 第 i 个平面按位置转 `m·θ_i` → 点积自动只依赖相对距离。RMSNorm 让数值稳，RoPE 让位置准，这俩就是今天把 nanoGPT "Llama 化" 的两块地基。

---

## 第三部分：底层视角 —— 复数为什么能"一行搞定旋转"

Llama 官方实现用了 `torch.view_as_complex`，很多人看不懂。这一节戳破它：**复数乘法，本质就是 2 维旋转。** 理解这个，你就理解了官方代码。

### 3.1 复数乘法 = 旋转（这就是全部秘密）

把一对分量 `(x₁, x₂)` 看成一个复数 `z = x₁ + i·x₂`。旋转角度 θ 对应乘以 `e^{iθ} = cosθ + i·sinθ`：

```
z · e^{iθ} = (x₁ + i·x₂)(cosθ + i·sinθ)
           = (x₁cosθ − x₂sinθ)  +  i·(x₁sinθ + x₂cosθ)
              └──── 实部 ────┘        └──── 虚部 ────┘
```

**把实部、虚部拆开看** —— 这不就是 2.2 里的 `R(θ)·(x₁,x₂)` 吗？一模一样！所以：**"乘以 e^{imθ}" 就等于 "用 R(mθ) 旋转"**。复数只是把 2×2 矩阵乘法压缩成了一次标量乘法，代码因此变得极短。

### 3.2 用最朴素的循环实现一遍（戳破"魔法"）

在信任 `view_as_complex` 之前，先用**纯循环**把 RoPE 写出来，你会发现它平平无奇——就是 2.2 那个旋转，对每一对分量做一遍：

```python
# rope_naive.py —— 用最笨的循环实现 RoPE，目的是"看清底层在干嘛"
# 运行环境：Python 3.10+ / numpy；纯 CPU 即可
import numpy as np

def rope_naive(x: np.ndarray, pos: int, base: float = 10000.0) -> np.ndarray:
    """把向量 x 按位置 pos 做 RoPE 旋转。x: [d]，d 必须是偶数。
    这段代码没有任何魔法，就是把 d 个分量两两一组、各转各的角度。"""
    d = x.shape[0]
    out = np.empty_like(x)
    for i in range(d // 2):                       # 遍历 d/2 个 2 维平面
        theta_i = base ** (-2.0 * i / d)          # 第 i 组的频率（越往后越慢）
        angle = pos * theta_i                     # 位置 pos 在这组要转的角度 m·θ_i
        cos, sin = np.cos(angle), np.sin(angle)
        x1, x2 = x[2 * i], x[2 * i + 1]           # 取出这一对分量
        out[2 * i]     = x1 * cos - x2 * sin      # 旋转矩阵第一行 → 实部
        out[2 * i + 1] = x1 * sin + x2 * cos      # 旋转矩阵第二行 → 虚部
    return out

# —— 亲手验证 2.2 的结论：点积只依赖 (m−n) ——
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 8
    q, k = rng.standard_normal(d), rng.standard_normal(d)

    def rope_dot(m, n):                            # 旋转后再点积
        return rope_naive(q, m) @ rope_naive(k, n)

    print("m=5,n=3 (差2):", round(rope_dot(5, 3), 6))
    print("m=12,n=10(差2):", round(rope_dot(12, 10), 6))   # 应与上一行几乎相等
    print("m=3,n=3 (差0):", round(rope_dot(3, 3), 6))
    print("未旋转 q·k     :", round(q @ k, 6))            # 应与"差0"那行相等
    # 你会看到：只要 m−n 相同，点积就相同；m−n=0 时等于未旋转的原始点积。
    # 这就是 2.2 推导的实验证据 —— 底层没有魔法，只有旋转。
```

> 跑一下这段，你会亲眼看到"差 2"的两行数值几乎相同、"差 0"的等于原始点积。**这比背公式牢一万倍。** 生产代码只是把这个循环用复数张量向量化了，逻辑完全一致。

---

## 第四部分：动手实现 —— `engine/rope.py`（生产级）

现在把朴素循环升级成向量化的生产实现。这版对照 Llama 官方思路，但每行都注释了"为什么"。

```python
# engine/rope.py
# ─────────────────────────────────────────────────────────────
# 运行环境：Python 3.10+ / torch 2.x；有无 GPU 均可（张量 .cuda() 即上 GPU）
# 作用：RoPE 的两个核心函数 —— 预计算旋转因子、把旋转应用到 Q/K。
# 约定张量布局：[B, S, H, D]  (Batch, Seq, num_Heads, head_Dim)  ← Llama 官方布局
# ─────────────────────────────────────────────────────────────
import torch


def precompute_freqs_cis(head_dim: int, max_seq_len: int,
                         base: float = 10000.0) -> torch.Tensor:
    """预计算所有 (位置, 频率) 组合的旋转因子 e^{i·m·θ_i}。

    为什么要"预计算"？旋转因子只跟"位置"和"维度"有关，跟具体输入无关，
    所以整段推理它都是常量。开机算一次、缓存起来，每步生成直接查表，
    避免在 attention 热路径里反复算 sin/cos —— 这是工业实现的标准优化。

    返回：复数张量 freqs_cis，形状 [max_seq_len, head_dim/2]，每个元素模长为 1。
    """
    assert head_dim % 2 == 0, "head_dim 必须是偶数：RoPE 把维度两两配对旋转"
    # θ_i = base^(-2i/d)，i = 0,2,4,...  → 得到 d/2 个频率，从快到慢
    exponent = torch.arange(0, head_dim, 2).float() / head_dim   # [d/2]，值为 0, 2/d, 4/d,...
    freqs = 1.0 / (base ** exponent)                             # [d/2]，即各组 θ_i

    t = torch.arange(max_seq_len).float()                        # [S] 所有位置 m
    # 外积：freqs[m, i] = m * θ_i，即"位置 m 在第 i 组要转的角度"
    freqs = torch.outer(t, freqs)                                # [S, d/2]

    # polar(模长=1, 角度=freqs) → e^{i·freqs} = cos + i·sin，一步造出复数旋转因子
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)       # complex64, [S, d/2]
    return freqs_cis


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """把 [S, d/2] 的 freqs_cis 变形成能和 [B, S, H, d/2] 广播的形状 [1, S, 1, d/2]。
    为什么单独抽出来：广播维度对不齐是 RoPE 最高频的 bug，集中在一处、写清楚。"""
    ndim = x.ndim                                                # 期望是 4: [B,S,H,d/2]
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f"freqs_cis {tuple(freqs_cis.shape)} 与 x 的 (S, d/2)=({x.shape[1]},{x.shape[-1]}) 不匹配"
    # 只在 S 维(第1维)和 d/2 维(最后一维)保留真实长度，其余(B、H)设为 1 让它广播
    shape = [dim if i == 1 or i == ndim - 1 else 1 for i, dim in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor,
               freqs_cis: torch.Tensor):
    """把 RoPE 旋转应用到 Q 和 K 上。xq/xk: [B, S, H, D]。

    只转 Q 和 K，不转 V —— 这是重点：位置信息只需影响"谁该注意谁"(由 QKᵀ 决定)，
    而 V 携带的是"内容本身"，不该被位置扰动。这也是 RoPE 比"加到嵌入上"更干净的地方。
    """
    # 1) 把最后一维 D 看成 D/2 个 (实, 虚) 对 → 转成复数张量 [B, S, H, D/2]
    #    .float() 是必须的：复数运算 + 角度精度对数值敏感，低精度会累积误差
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 2) 广播对齐后做复数乘法 = 旋转（见第三部分：乘 e^{imθ} 就是转 mθ）
    fc = _reshape_for_broadcast(freqs_cis, xq_c)                 # [1, S, 1, D/2]
    xq_rot = torch.view_as_real(xq_c * fc).flatten(-2)          # 转回实数并合并回 [B,S,H,D]
    xk_rot = torch.view_as_real(xk_c * fc).flatten(-2)

    # 3) 转回输入的原始 dtype（比如 bf16），保证和后续算子精度一致
    return xq_rot.type_as(xq), xk_rot.type_as(xk)
```

**几处"为什么这么写"的关键：**
- `precompute_freqs_cis` 缓存旋转因子：位置/维度是常量，**热路径不该反复算 sin/cos**。
- `apply_rope` 里 `.float()`：角度和复数乘法对精度敏感，**低精度算旋转会漂**（行业惯例：敏感运算升精度）。
- **只转 Q、K 不转 V**：位置只该影响"注意力权重"，不该污染"内容向量"。这是 RoPE 设计的克制之处。
- `_reshape_for_broadcast` 单独抽出并加断言：**广播维度对不齐是 RoPE 头号 bug**，把它围在一个带断言的函数里，错了立刻炸在正确的地方。

---

## 第五部分：性质测试 —— 比"对拍"更深的验证

`tests/test_rope.py`。**性质测试（Property-Based Test）**：不去逐个数值对答案，而是验证"这个实现必须满足的数学性质"。对 RoPE 来说，这比对拍更有洞察——因为你验证的是它"为什么对"，不只是"这次对"。

> **英文名词：Property-Based Testing（基于性质的测试）**
> 传统单测是"给定输入 X，断言输出等于我手算的 Y"。性质测试是"无论输入是什么，输出都必须满足某个不变式"。例如"排序后的列表长度不变、且非递减"就是排序的性质。RoPE 的两条铁性质：① 点积只依赖相对距离；② 位置 0 是恒等变换。

```python
# tests/test_rope.py
# 运行：pytest tests/test_rope.py -v
# 环境：Python 3.10+ / torch 2.x / pytest
import torch
from engine.rope import precompute_freqs_cis, apply_rope


def _rope_single(vec: torch.Tensor, pos: int, freqs_cis: torch.Tensor) -> torch.Tensor:
    """把 apply_rope 包成"对单个向量、指定位置"的便捷版，方便写性质测试。
    构造 [B=1, S=pos+1, H=1, D]，只在第 pos 个位置放 vec，旋转后取回来。"""
    D = vec.shape[0]
    x = torch.zeros(1, pos + 1, 1, D)
    x[0, pos, 0] = vec
    x_rot, _ = apply_rope(x, x, freqs_cis)     # 复用生产函数，保证测的就是真代码
    return x_rot[0, pos, 0]


def test_rope_relativity():
    """性质①：点积只依赖相对距离 (m−n)。
    只要 m−n 相同，dot(rope(q,m), rope(k,n)) 就必须不变 —— 这是 RoPE 的立身之本。"""
    torch.manual_seed(0)
    D = 64
    freqs_cis = precompute_freqs_cis(D, max_seq_len=128)
    q, k = torch.randn(D), torch.randn(D)

    def rope_dot(m, n):
        return (_rope_single(q, m, freqs_cis) * _rope_single(k, n, freqs_cis)).sum().item()

    base = rope_dot(5, 3)                       # m−n = 2
    for m, n in [(10, 8), (20, 18), (100, 98)]:  # 同样 m−n = 2，位置整体平移
        assert abs(rope_dot(m, n) - base) < 1e-4, \
            f"相对性被破坏：dot({m},{n})={rope_dot(m,n):.6f} != {base:.6f}"


def test_rope_zero_position_is_identity():
    """性质②：位置 0 的旋转是恒等变换。
    因为 e^{i·0·θ} = 1，位置 0 乘的旋转因子全是 1，向量原封不动。"""
    D = 64
    freqs_cis = precompute_freqs_cis(D, max_seq_len=16)
    v = torch.randn(D)
    v_rot = _rope_single(v, 0, freqs_cis)
    assert torch.allclose(v_rot, v, atol=1e-6), "位置0应恒等，但向量被改变了"


def test_rope_preserves_norm():
    """附赠性质③：旋转不改变向量长度（旋转是正交变换）。
    这是个便宜又强的 sanity check —— 如果模长变了，一定是实现错了。"""
    torch.manual_seed(1)
    D = 128
    freqs_cis = precompute_freqs_cis(D, max_seq_len=64)
    v = torch.randn(D)
    v_rot = _rope_single(v, 37, freqs_cis)
    assert torch.allclose(v.norm(), v_rot.norm(), atol=1e-5), "旋转不该改变模长"
```

> **为什么性质测试对 RoPE 特别合适**：RoPE 没有"标准答案表"可对拍，但它有**必须成立的数学不变式**。测不变式，等于测"它是不是真的 RoPE"，而不是"这次数值碰巧对"。这三条（相对性 / 位置0恒等 / 保模长）任何一条挂了，都能精确指向哪类实现错误。

---

## 第六部分：把 RMSNorm 和 RoPE 接进 `model.py`（Llama 化）

### 6.1 先说清 RMSNorm：它和 LayerNorm 差在哪，为什么能省

Day 1 已实现 `backend.rmsnorm`，今天要把 nanoGPT 的 `LayerNorm` 换掉。先理解为什么换：

> **英文名词：LayerNorm（层归一化） vs RMSNorm（均方根归一化）**
>
> LayerNorm：`y = (x − mean(x)) / sqrt(var(x) + ε) · γ + β`
> RMSNorm： `y = x / sqrt(mean(x²) + ε) · γ`
>
> 对比看差异：RMSNorm **砍掉了两样东西**——① 减均值（re-centering，重新居中）；② 偏置 β。只保留"按均方根缩放"（re-scaling，重新缩放）。

**为什么砍掉减均值也没事？** 研究（和大量工业实践）发现：归一化真正起作用的是 **re-scaling（把向量尺度拉到稳定范围）**，而 **re-centering（减均值）的贡献很小**。既然贡献小，砍掉它换来更快、更简单，非常划算。

> **类比**：LayerNorm 像"先把全班成绩减掉平均分（居中），再除以标准差（缩放）"；RMSNorm 说"其实居中那步对稳定训练帮助不大，我直接除以均方根缩放就行"。少算一遍均值、少一组偏置参数，在千亿次调用里省下的算力很可观。

- **省算力**：不用算均值、少一次遍历、少一组 β 参数。
- **数值更稳**：结构更简单，配合 pre-norm 效果好。
- **这就是 Llama 全系、Qwen、DeepSeek 都用 RMSNorm 的原因。**

> 补充一个架构惯例：**Pre-Norm（前置归一化）**——把 norm 放在"进入 attention/ffn 之前"（`x + attn(norm(x))`），而不是之后。nanoGPT 已经是 pre-norm 结构，Llama 也是。它让深层网络的梯度更稳、更好训。今天换 norm 类型时保持 pre-norm 位置不变即可。

### 6.2 改造 `model.py`：三处改动

把 nanoGPT 的 Transformer Block 从"绝对位置 + LayerNorm"改成"RoPE + RMSNorm"，一共三处：

```python
# engine/model.py （节选：展示三处 Llama 化改动，其余沿用 W6 nanoGPT）
# 环境：Python 3.10+ / torch 2.x
import torch
import torch.nn as nn
from engine.rope import precompute_freqs_cis, apply_rope


class Block(nn.Module):
    """改动①：LayerNorm → 走 backend.rmsnorm。从今天起，模型层只认 Backend 接口，
    不再自己 new 一个 nn.LayerNorm —— 这样 norm 的实现也能被后端替换（Day1 抽象的兑现）。"""
    def __init__(self, config, backend):
        super().__init__()
        self.backend = backend
        # RMSNorm 只有一个缩放参数 γ（没有 β），初始化为全 1
        self.attn_norm_w = nn.Parameter(torch.ones(config.n_embd))
        self.ffn_norm_w  = nn.Parameter(torch.ones(config.n_embd))
        self.attn = CausalSelfAttention(config)     # 内部接入 RoPE（见下）
        self.mlp  = MLP(config)

    def forward(self, x, freqs_cis):
        # 保持 pre-norm 结构：norm 在子层之前，残差连接不变
        x = x + self.attn(self.backend.rmsnorm(x, self.attn_norm_w), freqs_cis)
        x = x + self.mlp(self.backend.rmsnorm(x, self.ffn_norm_w))
        return x


class CausalSelfAttention(nn.Module):
    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # 投影出 Q/K/V
        # reshape 成 [B, T, H, D]（注意是 Llama 布局，H 在 D 前）
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        # 改动②：在这里接入 RoPE —— 只旋转 Q、K，V 原样不动
        q, k = apply_rope(q, k, freqs_cis[:T])   # 只取前 T 个位置的旋转因子

        # 转成 [B, H, T, D] 交给注意力算子（对齐 Day1 backend.attention 的布局）
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = self.backend.attention(q, k, v, causal=True)     # 复用 Day1 的接口
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GPT(nn.Module):
    def __init__(self, config, backend):
        super().__init__()
        self.backend = backend
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 改动③：删除绝对位置嵌入 self.wpe！位置信息全交给 RoPE。
        #   nanoGPT 原来有   self.wpe = nn.Embedding(block_size, n_embd)  ← 整行删掉
        self.h = nn.ModuleList([Block(config, backend) for _ in range(config.n_layer)])
        self.final_norm_w = nn.Parameter(torch.ones(config.n_embd))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 预计算旋转因子，注册成 buffer（随模型搬到 GPU，但不是可训练参数）
        head_dim = config.n_embd // config.n_head
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.block_size),
            persistent=False,     # 不存进 checkpoint：它是可复算的常量，存了浪费空间
        )

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)         # 改动③续：只有 token 嵌入，不再 + pos_emb
        for block in self.h:
            x = block(x, self.freqs_cis[:T])
        x = self.backend.rmsnorm(x, self.final_norm_w)
        return self.lm_head(x)
```

> **改造后务必回跑 Day 1 的基线快照**：换了 norm、加了 RoPE、去了绝对位置，生成结果**允许变化**（毕竟结构变了），但不该变成乱码或报维度错。有 Day 1 存的基线，你才有判断"变化是否合理"的尺子——这就是 Day 1 那一步的回报。

---

## 第七部分：工业锚点 —— RoPE 为什么对推理系统"特别友好"

这一节是今天完成标准的第二问，也是把 RoPE 和你暑假主线（推理引擎）连起来的关键。

### 7.1 先补一个背景：KV Cache 是什么，为什么需要它

> **英文名词：KV Cache（键值缓存）**
> 自回归生成（一次吐一个 token）时，每生成一个新 token，都要让它的 Query 和**前面所有 token 的 K、V**做注意力。如果每步都把前面所有 token 的 K、V 重算一遍，计算量随长度平方爆炸。KV Cache 的做法：**每个 token 的 K、V 只在它第一次出现时算一次，存起来（缓存），后面直接复用。**

- **Prefill（预填充）阶段**：把整段 prompt 一次性喂进去，算出所有 prompt token 的 K、V，存进 cache。
- **Decode（解码）阶段**：每步只算**新 token 一个**的 Q、K、V，新 K/V 追加进 cache，Q 和 **整个 cache** 做注意力。

**这是推理提速的头号功臣**，也是你 W5-6 要在引擎里实现的核心。而 RoPE 恰好和它天生契合——下面是关键。

### 7.2 核心问题：为什么"缓存的是旋转后的 K"？

**答案**：因为 RoPE 对一个 token 的旋转角度，**只由它自己的绝对位置决定，而一个历史 token 的位置永远不会变。**

想清楚这个链条：
1. 第 n 个 token 的 K，被旋转 `n·θ`。这个 n 是它的绝对位置，一旦它进了序列就**永久固定**。
2. 所以我们可以在**算出 K 的当下就旋转好**，把**旋转后的 K** 存进 cache。
3. 后面第 m 步来了个新 Query，旋转 `m·θ`，去和 cache 里那个"早就转好的 K_n"点积——由 2.2 的推导，结果自动只依赖 `(m−n)`，**正确的相对位置关系天然成立**。
4. 全程**没有任何历史 K 需要重算或重转**。

> **对比才显珍贵**：假设有一种位置编码，它的位置效应依赖"当前这一步 m 是多少"来动态地作用到每个历史 token 上，那每生成一个新 token，你就得把**缓存里所有历史 K 重新处理一遍**——KV Cache 的意义就没了。RoPE 的高明在于：**它把"相对"的效果，用"对每个 token 独立的、绝对的"旋转实现出来**（"absolute application, relative effect"）。正是这个"独立 + 绝对"，让每个 K 转一次就永久有效，完美适配缓存。

**一句话记住**：RoPE 缓存旋转后的 K，是因为"每个 token 的旋转只认自己的位置、和查询是谁无关"——所以转一次、存下来、永远能用。

### 7.3 还有一层友好：RoPE 和 FlashAttention 兼容

因为 RoPE 是"把位置折进 Q、K 里"，而不是"往注意力分数矩阵上加一个 [S,S] 的偏置矩阵"，所以它**不需要显式构造那张巨大的分数矩阵**。这让它能和 **FlashAttention**（你 W3 要学的、不落地完整分数矩阵的高效注意力 kernel）无缝配合。要是位置编码非得往分数矩阵上加东西，FlashAttention 的"不落地矩阵"优化就用不上了。**RoPE 的实现方式，恰好没给高效 kernel 添堵。**

### 7.4 长上下文外推（留一句话占位，知道门在哪）

RoPE 训练时最长比如 4K，能不能用到 32K？直接用会因为"没见过的大旋转角度"而效果崩。解决办法是**改 θ 的频率**：

> **NTK-aware Scaling / YaRN**：通过调整 RoPE 的 base（那个 10000）或各频率，把"没见过的远距离角度"映射回"训练时见过的角度范围"，从而在几乎不重训的情况下把上下文窗口拉长。这是长上下文（128K、1M token）背后的关键技巧之一。**今天只需知道这扇门在哪**，W7 之后要做长上下文时再深入。

---

## 第八部分：常见陷阱与调试技巧

| 陷阱 | 现象 | 根因 & 解法 |
|---|---|---|
| **配对约定搞混（头号大坑）** | 自己的 RoPE 结果和 HuggingFace 权重对不上、加载官方权重后输出乱码 | RoPE 有两种配对法：**Llama 官方**用相邻配对 `(0,1)(2,3)…`（`view_as_complex`）；**GPT-NeoX/HF** 用对半配对 `(0,d/2)(1,d/2+1)…`（`rotate_half`）。**两者数学等价但排列不同，混用必错**。用谁的权重就用谁的配对法。见下方补充。 |
| **head_dim 是奇数** | `view_as_complex` 或 reshape 报错 | RoPE 必须两两配对，head_dim 必须偶数。加断言提前拦。 |
| **广播维度对不齐** | shape 报错，或悄悄算错（没报错但结果不对） | freqs_cis `[S,d/2]` 要 reshape 成 `[1,S,1,d/2]` 才能和 `[B,S,H,d/2]` 广播。用带断言的 `_reshape_for_broadcast` 集中处理。 |
| **忘了截取 `freqs_cis[:T]`** | 序列比 max_seq_len 短时 shape 不匹配 | 每步只取当前长度对应的旋转因子。 |
| **把 V 也旋转了** | 效果变差、和参考实现对不上 | **只转 Q、K，不转 V**。V 是内容，不该被位置扰动。 |
| **低精度算旋转** | bf16 下数值漂移、长序列尤其明显 | `apply_rope` 内部 `.float()` 算完再转回。角度/复数运算对精度敏感。 |
| **RoPE 与 KV Cache 位置对不上** | decode 阶段位置编码错乱、输出退化 | cache 里存的是**已按各自绝对位置旋转好的 K**；新 token 的 Q 要用它**真实的绝对位置**（= 已缓存长度）去旋转，别用 0。 |

> **补充：两种配对约定的代码对照（务必分清）**
> ```python
> # 约定 A：Llama 官方 —— 相邻配对 (x0,x1),(x2,x3),...  → 用复数乘法
> xq_c = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))   # 本文用的就是这版
>
> # 约定 B：GPT-NeoX / HuggingFace —— 对半配对 (x0, x_{d/2}),...  → 用 rotate_half
> def rotate_half(x):
>     x1, x2 = x.chunk(2, dim=-1)      # 前一半、后一半
>     return torch.cat((-x2, x1), dim=-1)
> # q_embed = q * cos + rotate_half(q) * sin
> ```
> **两者对同一个 θ 序列的数学效果等价，但要求 Q/K 的分量排列不同。** 你自己从零写引擎、自己训权重，选哪个都行（本文选 Llama 版）；但**一旦要加载 HF 的开源权重，就必须用 B**，否则权重和旋转对不上，输出全是乱码。这是接开源模型时最常见、最难查的 bug 之一。

---

## 第九部分：副线 & 整理块（简要）

**副线块 1.5h —— `tech_notes/rope_from_scratch.md`**：按三段式提炼本文精华（原理直觉 → rope.py → KV Cache 锚点），开头一句话串起 W6：*"W6 我知道了 attention 天生看不见位置；今天我知道了现代解法为什么是'旋转'，以及它为什么对推理系统友好。"* —— 让你的笔记形成一条能自我串讲的知识链。

**整理块 1h —— nanoGPT 仓库 README 打磨**：第一屏不要写"如何安装"，要写**"这个项目证明了什么"**：
> *"从零手写 Transformer，实现了 KV Cache 与 prefill-decode 两阶段推理，理解并落地了现代 LLM 的核心组件（RMSNorm / RoPE / 因果注意力）。"*
> 招聘方/老师第一眼要看到的是**你的能力边界**，不是安装命令。安装说明放到后面。

---

## ✅ 今日完成标准自测（能脱稿讲清 = 真懂）

合上笔记，用自己的话/在纸上回答：

1. **在纸上推 2 维 RoPE 点积只含 (m−n)。** —— 写出 `R(mθ)q` 和 `R(nθ)k`，展开点积，用 `cos(a)cos(b)+sin(a)sin(b)=cos(a−b)` 收拢，得到 `(q₁k₁+q₂k₂)cos((m−n)θ)+(q₁k₂−q₂k₁)sin((m−n)θ)`。绝对位置消失。
2. **为什么现代 LLM 弃绝对 PE 改旋转？** —— 绝对 PE 要模型自己学相对关系、外推差、位置和内容焊死；RoPE 让相对位置成为旋转的几何必然，只作用于 Q/K 不污染 V，外推可通过改频率延展。
3. **为什么 K 缓存的是旋转后的值？** —— RoPE 对一个 token 的旋转只由它自己的绝对位置决定，历史 token 位置永不变，所以算出 K 时就转好、存下、永久可用；新 Query 用自己的位置旋转后与之点积，相对关系 (m−n) 自动成立，无需重算任何历史 K。
4. **多频率（钟表类比）解决了什么？** —— 单一频率会绕圈混淆远近；d/2 个不同频率组合，才能同时分辨"近几个词"和"远几百个词"。

**今日交付物清单：**
- [ ] `engine/rope.py`：`precompute_freqs_cis` + `apply_rope`（自己先写、再对照 Llama）
- [ ] `tests/test_rope.py`：相对性 + 位置0恒等（+ 保模长）性质测试全绿
- [ ] `engine/model.py`：LayerNorm→RMSNorm（走 backend）、删除 `wpe`、attention 接入 RoPE
- [ ] 回跑基线：Llama 化后能正常生成、不报维度错、不出乱码
- [ ] `tech_notes/rope_from_scratch.md`：三段式，串联 W6 attention 笔记
- [ ] nanoGPT README 第一屏改为"证明了什么"
- [ ] commit（建议：`feat: Llama-ify model with RMSNorm + RoPE (Day2)`）

---

> **一句话收尾**：绝对位置编码是"告诉模型每个人的座位号，让它自己算距离"；RoPE 是"让每个人按座位号转个角度，距离变成两人之间天然的夹角"。**把'要学的东西'变成'数学上必然成立的东西'——这就是好设计的味道。** 而它"每个 token 只认自己位置"的旋转方式，又恰好让推理系统的 KV Cache 免费获益。今天你不只学了一个位置编码，你学到了"为什么它同时赢在效果和工程"。
