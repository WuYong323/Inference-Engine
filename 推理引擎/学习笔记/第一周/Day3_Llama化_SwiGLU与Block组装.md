# Day 3 · Llama 化 ②：SwiGLU + 完整 Llama Block 组装

> **今天只回答一个问题**：FFN 里加一个"门"到底门住了什么？为什么值得多花一个矩阵的参数？
>
> 昨天（Day 2）你把 RMSNorm 和 RoPE 两块地基铺好了。今天把最后一个核心算子 SwiGLU 想透，然后**把所有零件组装成一台完整的 Llama 风格整机**——从此 miniLLM-serve 的模型层就是"现代 LLM 的骨架"，而不再是 nanoGPT。
>
> **本文沿用三段式**：原理 + 直觉类比 → 可运行代码（含底层视角）→ 工业锚点。**核心目标**：能脱稿画出 LlamaBlock 数据流图；能讲清 SwiGLU 三个矩阵各干什么、hidden dim 为什么是 `2/3×4d`。

---

## 0. 今日地图

```
┌─ 主线深块 3.5h ──────────────────────────────────────────────┐
│ 先懂(45min)：SwiGLU —— 门门住了什么 + 为什么 hidden=2/3×4d      │
│ 组装(2h)   ：LlamaBlock（RMSNorm→attn(RoPE)→残差→RMSNorm→      │
│              SwiGLU-FFN→残差）+ 整模型（emb→N×Block→norm→head） │
│ 差异表(30min)：我的实现 vs 真 Llama-3（GQA/tokenizer/scaling）  │
├─ 副线块 1.5h：全模型 sanity（shape/NaN/logits 量级/手算参数量） │
├─ 整理块 1h：week8_triton README（照 W8 Day7 §7.3 模板落地）     │
└──────────────────────────────────────────────────────────────┘
```

---

## 第一部分：问题背景 —— 先搞懂"普通 FFN"到底在干嘛

要理解 SwiGLU 加的"门"有什么用，得先看清没有门的普通 FFN 是什么、为什么 Transformer 离不开它。

### 1.1 FFN 是什么，Transformer 为什么需要它

> **英文名词：FFN（Feed-Forward Network，前馈网络）**
> Transformer 每一层里，跟在注意力后面的那个"小型两层全连接网络"。它对**每个 token 独立**做一次"升维 → 非线性 → 降维"：
> ```
> FFN(x) = W2 · act(W1 · x)      （原始 Transformer 用 ReLU 作为 act）
>   W1: [d → 4d]  升维（放大到 4 倍宽）
>   act: 非线性激活（ReLU）
>   W2: [4d → d]  降维（压回原宽度）
> ```

**为什么需要它？** 注意力干的是"**token 之间**互相看、交换信息"（混合信息）；FFN 干的是"**每个 token 自己**把混来的信息加工、提炼一遍"（处理信息）。**注意力负责"沟通"，FFN 负责"思考"**——两者一沟通一思考，交替堆叠，才是 Transformer 的一层。少了 FFN，模型就只会搬运信息、不会加工信息。

> **一个反常识的事实**：在 Llama 这类模型里，**FFN 的参数量通常比注意力还大**（升到 4 倍宽，两个大矩阵）。所以 FFN 是模型的"参数大户"，也是推理时算力和显存的大头之一——这正是它值得你花力气优化（fuse 成一个 kernel）的原因，也是它成为 Day 1 三大 Backend 算子之一的原因。

### 1.2 为什么要"先升维再降维"？—— key-value 记忆的视角（深度）

普通人会觉得"升到 4d 再压回 d"是浪费。但有个很有洞察的研究视角（Geva et al., *Transformer Feed-Forward Layers Are Key-Value Memories*）：

> **把 FFN 看成一张"记忆表"**：
> - `W1` 的每一行是一个 **key（模式探测器）**：`W1·x` 在算"输入 x 命中了哪些已知模式"（比如"这是不是在讲编程？""是不是疑问句？"）。4d 就是"我准备了 4d 个模式探测器"。
> - `act`（ReLU）：只保留真正命中的模式（没命中的清零）。
> - `W2` 的每一列是一个 **value（要写回的内容）**：命中哪些模式，就把对应的内容按强度加权写回残差流。

**类比**：FFN 像一次"联想记忆检索"。你看到"埃菲尔铁塔"（输入），大脑并行匹配一堆记忆键（W1：这是地标？在巴黎？很高？），命中的记忆把对应知识（W2：法国、旅游、铁）加权取出来。**升到 4d，就是"我一次能并行比对 4d 条记忆"**——维度越宽，记忆容量越大。这就是升维的意义：不是浪费，是给模型一块更大的"联想草稿纸"。

**记住这个视角**，因为 SwiGLU 的"门"正是在这张记忆表上做了一个升级——从"命中就全放"升级到"命中了还能调节放多少"。

---

## 第二部分：核心原理 —— SwiGLU：给 FFN 装一个"逐通道阀门"

### 2.1 SwiGLU 是什么：从"一条路"变"两条路相乘"

普通 FFN 只有一条路：`W2 · act(W1·x)`。SwiGLU 变成**两条路，逐元素相乘**：

```
SwiGLU-FFN(x) = W2 · ( SiLU(W1·x)  ⊙  W3·x )
                        └─ 内容路 ─┘   └门路┘
                              ⊙ = 逐元素相乘（element-wise product）
  W1: [d → h]  内容路（过 SiLU 激活）
  W3: [d → h]  门路（保持线性，当"阀门开度"）
  W2: [h → d]  把结果压回 d 维
```

> **英文名词：SwiGLU（Swish-Gated Linear Unit）**
> GLU（门控线性单元）家族的一员，由 Noam Shazeer 在 *GLU Variants Improve Transformer* 里提出，被 Llama/PaLM/Qwen 等采用。名字拆开：**Swish**（一种激活函数，= SiLU）+ **GLU**（门控线性单元）。核心动作：把 FFN 的一条前向路，拆成"内容"和"门"两条路逐元素相乘。

> **英文名词：⊙ 逐元素相乘（Element-wise / Hadamard Product）**
> 两个同形状张量，对应位置的数各自相乘，不是矩阵乘法。例如 `[a,b]⊙[c,d]=[ac,bd]`。它的意义：**第二个张量成了第一个张量"逐个通道的缩放系数"**——这正是"门"的数学本体。

### 2.2 SiLU 是什么（含底层实现）

> **英文名词：SiLU（Sigmoid Linear Unit）= Swish**
> 激活函数，公式极简：`SiLU(x) = x · σ(x)`，其中 `σ(x)=1/(1+e⁻ˣ)` 是 sigmoid。

它和你熟悉的 ReLU 差在哪、为什么现代模型爱用它？看底层实现最清楚：

```python
# silu_demo.py —— 看清 SiLU 到底是什么，以及它和 ReLU 的区别
# 运行环境：Python 3.10+ / numpy（画图需 matplotlib，可选）
import numpy as np

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def silu(x):    return x * sigmoid(x)        # SiLU = x·σ(x)，就这一行
def relu(x):    return np.maximum(0.0, x)

xs = np.array([-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4], dtype=float)
for x in xs:
    print(f"x={x:+.1f}   ReLU={relu(x):+.3f}   SiLU={silu(x):+.3f}")
# 你会观察到三个关键区别：
#  1) x 很负时两者都→0，但 SiLU 是"平滑滑到 0"，ReLU 是"一刀切到 0"
#  2) SiLU 在 x≈-1 附近会轻微探到负值（约 -0.28）再回升 —— 非单调！
#     这点负值让它能表达"轻微抑制"，比 ReLU 的"要么全砍要么放行"更细腻
#  3) x 很正时两者都≈x（线性放行）
```

> **为什么现代模型弃 ReLU 用 SiLU？**
> - **处处平滑可导**：ReLU 在 0 点有个尖角（导数突变），SiLU 是光滑曲线，梯度更稳、更好优化。
> - **允许"轻微负值"**：ReLU 把所有负输入一刀切成 0（信息全丢）；SiLU 让小负值平滑通过一点点，保留了更多信息。
> - **类比**：ReLU 是老式墙上开关（开/关两态）；SiLU 是可调旋钮，还能微微反向。旋钮比开关能表达的状态多，模型的表达力就更强。
>
> **工业细节**：在手写 CUDA/Triton kernel 里，SiLU 通常和前面的矩阵乘**融合（fuse）**在一起算——矩阵乘出结果的当下，寄存器里直接 `x*sigmoid(x)` 算完，不往显存写中间结果。这就是 Day 1 里 `fused_ffn` 那个"fused"要兑现的东西（W4 的活）。

### 2.3 核心问题：这个"门"到底门住了什么？

现在回答今天的主问题。普通 FFN 和 SwiGLU 的本质差别，在于一个词：**乘法**。

**普通 FFN（加法世界）**：`W2 · ReLU(W1·x)`。每个隐藏通道的值，是输入的**加权和**再过激活。整个网络本质是"把特征线性组合起来"。ReLU 只能做"这个通道要不要激活"的粗糙开关，且这个开关基本是内容自己决定自己。

**SwiGLU（乘法世界）**：`SiLU(W1·x) ⊙ (W3·x)`。关键在 `⊙`——**一条路的输出，去逐通道地"乘"另一条路的输出**。乘法带来两个加法给不了的能力：

1. **输入相关的逐通道调节（conditional / input-dependent gating）**：门路 `W3·x` 是随输入变化的。同一个内容通道，在上下文 A 里门开得大（乘以 1.5），在上下文 B 里门关得小（乘以 0.1）。**网络学会了"看情况放行"**，而不是像 ReLU 那样用一套固定标准。

2. **特征交互（feature interaction）**：`a ⊙ b` 能表达"特征 A **且** 特征 B 同时存在时才输出"这种**逻辑与**的关系（两个都大，积才大；一个接近 0，积就被掐灭）。纯加法网络要费很大劲、堆很多层才能近似出乘法关系。**一个乘号，直接把"条件组合"这种表达力送给了网络。**

> **回到 2.1 的类比升级**：普通 FFN 像所有货物过同一个安检门（一套固定标准，过或不过）；SwiGLU 像**每件货物配一个可调阀门**，而且**阀门开度由货物自己和当前情境共同决定**——"这类信息在这个上下文里全放，那类信息掐掉一半"。这种"逐通道、看情境"的精细控制，就是那个多花的矩阵 `W3` 买来的东西。

> **一句话本质**：门门住的是"**每个通道在当前上下文下该放行多少**"。它把 FFN 从"固定标准的信息处理器"升级成"能按情境自我调节的信息处理器"。多花一个矩阵，换来"乘法级"的表达力和条件计算能力——实测在同等参数量下，语言建模效果稳定更好，这就是 Llama 愿意付这个成本的原因。

> **诚实补一句**：`SiLU(W1·x) ⊙ W3·x` 里，把哪条叫"内容"、哪条叫"门"其实是**解释视角**（乘法是对称的，两条路互相调制）。习惯上把过了激活的 `W1` 路叫内容、线性的 `W3` 路叫门。你记住"两条路相乘 = 一条调制另一条"这个本质即可，别纠结命名。

---

## 第三部分：深度思考 —— 为什么 hidden dim 是 `2/3 × 4d`？

这是今天完成标准的必答题，也是一个**工业素养**的绝佳案例：改结构时，要控制参数量做公平对比。

### 3.1 一句话：为了"加了第三个矩阵后，总参数量不变"

**推导（自己算一遍，很短）**：

普通 FFN 有 **2 个**矩阵，hidden 用经典的 `4d`：
```
参数量 = W1[d×4d] + W2[4d×d] = 4d² + 4d² = 8d²
```

SwiGLU 有 **3 个**矩阵（W1、W3 升维，W2 降维），hidden 设为 `h`：
```
参数量 = W1[d×h] + W3[d×h] + W2[h×d] = 3dh
```

**要让两者参数量相等**（这样对比才公平——效果变好是因为结构好，不是因为偷偷加了参数）：
```
3dh = 8d²   ⟹   h = 8d/3 = (2/3)·(4d) ≈ 2.667d
```

所以 Llama 把 SwiGLU 的 hidden dim 设成 `(2/3)×4d`，而不是 `4d`。**多了一个矩阵，就把每个矩阵改窄一点，总账持平。**

> **为什么这件事是"素养"而不是"抠细节"？**
> 假设有人说"我把 FFN 换成 SwiGLU，效果涨了 2%！"——但他偷偷把 hidden 还留在 4d（3 个矩阵 × 4d = 12d²，比原来 8d² 多了 50% 参数）。那这 2% 到底是**结构更好**，还是**单纯参数变多**堆出来的？说不清。**控制变量（参数量对齐）后再比，才知道是不是结构本身的功劳。** 这是做架构改进、写实验报告、给张老师/师兄汇报时都要有的基本纪律——你 W8 立的"诚实对标"纪律，在这里换了个场景又出现了。

### 3.2 Llama 的实际公式：还多了一步"硬件对齐"

真实 Llama 代码不止 `2/3`，还会把 hidden **向上取整到某个数（如 256）的倍数**：

```python
# Llama 计算 FFN hidden dim 的真实逻辑（简化自官方 model.py）
# 环境：Python 3.10+
def llama_ffn_hidden_dim(dim: int, multiple_of: int = 256,
                         ffn_dim_multiplier: float | None = None) -> int:
    hidden = 4 * dim                       # ① 经典起点 4d
    hidden = int(2 * hidden / 3)           # ② 乘 2/3，给"第三个矩阵"腾出参数预算
    if ffn_dim_multiplier is not None:     # ③ 某些型号再微调（Llama-2 70B 等）
        hidden = int(ffn_dim_multiplier * hidden)
    # ④ 关键：向上取整到 multiple_of 的倍数（硬件对齐）
    hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
    return hidden

# 例：dim=4096 → 4*4096=16384 → *2/3≈10922 → 取整到256倍数 → 11008
print(llama_ffn_hidden_dim(4096))   # 11008，正是 Llama-7B 的 FFN hidden
```

> **为什么要取整到 256 的倍数？—— 硬件对齐（Hardware Alignment）**
> GPU 做矩阵乘用的是 **Tensor Core（张量核心）**，它一次吃固定大小的小块（tile，如 16×16、128×128）。如果矩阵维度是 128/256 的整数倍，就能被 tile 整齐切分、跑满算力；如果是 10922 这种"零头"维度，最后会剩一条算不满的边角，浪费算力还可能触发慢路径。
> **类比**：铺地砖，房间尺寸正好是砖的整数倍就不用切砖、铺得快又整齐；差一点点就得切一排砖，费工还难看。把 hidden 对齐到 256 倍数，就是"把房间尺寸修成砖的整数倍"。**这是"数学最优"向"硬件现实"让步的典型一例**——纯理论算出 10922，工程上用 11008，因为后者跑得快。这种"为硬件微调结构"的意识，正是 AI Infra 的核心思维。

---

## 第四部分：动手实现 —— SwiGLU FFN（走 Backend）

Day 1 的 `backend.fused_ffn(x, w_gate, w_up, w_down)` 已经实现了 SwiGLU 的数学（`down(silu(gate)*up)`）。今天在模型层把它用起来，权重用 `nn.Linear` 管理：

```python
# engine/model.py （节选：SwiGLU FFN 模块）
# 环境：Python 3.10+ / torch 2.x
import torch
import torch.nn as nn


class SwiGLUFFN(nn.Module):
    """Llama 风格 FFN。三个矩阵都无 bias（现代 LLM 惯例：bias 收益极小，省掉更简洁）。"""
    def __init__(self, dim: int, hidden_dim: int, backend):
        super().__init__()
        # 命名对齐 Llama 官方：gate_proj / up_proj / down_proj
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)   # W1：内容路(过SiLU)
        self.up_proj   = nn.Linear(dim, hidden_dim, bias=False)   # W3：门路(线性)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)   # W2：压回 d 维
        self.backend = backend

    def forward(self, x):
        # 走 Backend 接口：今天是 TorchBackend，将来一行配置换成 CUDA/Triton 融合版
        # 传 .weight（形状 [out,in]），与 Day1 fused_ffn 的约定一致
        return self.backend.fused_ffn(
            x, self.gate_proj.weight, self.up_proj.weight, self.down_proj.weight
        )
```

> **为什么无 bias**：现代 LLM（Llama/GPT-NeoX 之后）普遍去掉了线性层的 bias。原因：在有归一化（RMSNorm）的架构里 bias 的实际贡献很小，去掉它省参数、省一次加法、代码更干净。这也是你 Day 1 `fused_ffn` 签名里没有 bias 参数的原因——从算子层就贯彻了这个设计。

---

## 第五部分：组装完整 LlamaBlock 与整模型

### 5.1 LlamaBlock 数据流（今天必须能脱稿画出）

一个 Llama 层 = **两个 pre-norm 子层，各自带残差连接**：

```
输入 x ─┬─────────────────────────────────────────────┐
        │                                               │
        └─▶ RMSNorm ─▶ Attention(RoPE) ─────────────▶  (+) ─┐   ← 子层1：注意力 + 残差
                                                             │
        ┌────────────────────────────────────────────────┘
        │                                                
        x' ─┬────────────────────────────────────────┐
            │                                          │
            └─▶ RMSNorm ─▶ SwiGLU-FFN ────────────▶  (+) ─▶ 输出   ← 子层2：FFN + 残差
```

文字版数据流（**背下来**）：
```
x  → rmsnorm → attention(+RoPE) → 加回 x（残差）→  得到 h
h  → rmsnorm → SwiGLU-FFN        → 加回 h（残差）→  输出
```

> **英文名词：Residual Connection（残差连接）**
> 就是那个 `x + 子层(x)`——把子层的输出**加回**它的输入。为什么必须有它？深层网络里，梯度反向传播要穿过很多层，容易越传越小（梯度消失）。残差连接给梯度开了一条"高速公路"（`+x` 那条路），梯度可以几乎无损地直达浅层。
> **类比**：残差连接像文档的"修订模式"——每一层不是从头重写整篇，而是**在上一版基础上打补丁**（`x + Δ`）。改动小、可追溯，就算某层补丁很烂，原文（x）还在，不会把整篇毁掉。这就是能把网络堆到几十上百层还训得动的关键。

> **英文名词：Pre-Norm（前置归一化）**
> 归一化放在子层**之前**（`x + attn(norm(x))`），而不是之后（`norm(x + attn(x))`，即 post-norm）。为什么现代 LLM 全用 pre-norm？因为它让残差那条"高速公路"完全不被 norm 干扰（`+x` 是干净的原始信号），深层训练更稳。**易错点**：注意 norm 只作用在"进子层的那份拷贝"上，残差加的是**未经 norm 的原始 x**。写反了（残差加 norm 后的值）会破坏这条高速公路。

```python
# engine/model.py （节选：完整 LlamaBlock）
class LlamaBlock(nn.Module):
    def __init__(self, config, backend):
        super().__init__()
        self.backend = backend
        # 两个 RMSNorm 的可学习缩放（各一个，无 bias）
        self.attn_norm_w = nn.Parameter(torch.ones(config.dim))
        self.ffn_norm_w  = nn.Parameter(torch.ones(config.dim))
        self.attn = CausalSelfAttention(config, backend)      # 内部含 RoPE（Day2）
        hidden = llama_ffn_hidden_dim(config.dim, config.multiple_of)
        self.ffn = SwiGLUFFN(config.dim, hidden, backend)

    def forward(self, x, freqs_cis):
        # 子层1：pre-norm → attention(RoPE) → 残差。注意残差加的是原始 x！
        x = x + self.attn(self.backend.rmsnorm(x, self.attn_norm_w), freqs_cis)
        # 子层2：pre-norm → SwiGLU-FFN → 残差
        x = x + self.ffn(self.backend.rmsnorm(x, self.ffn_norm_w))
        return x
```

### 5.2 整模型组装 + 权重 tying

```python
# engine/model.py （节选：完整 LlamaModel）
from engine.rope import precompute_freqs_cis


class LlamaModel(nn.Module):
    def __init__(self, config, backend):
        super().__init__()
        self.backend = backend
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)   # token→向量
        self.layers = nn.ModuleList(
            [LlamaBlock(config, backend) for _ in range(config.n_layers)]
        )
        self.norm_w = nn.Parameter(torch.ones(config.dim))                  # 最后一层 RMSNorm
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False) # 向量→词表 logits

        # ── 权重 tying（权重绑定）：输入嵌入和输出投影共享同一张矩阵 ──
        self.lm_head.weight = self.tok_embeddings.weight

        # RoPE 旋转因子预计算并注册为 buffer（Day2）
        head_dim = config.dim // config.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, config.max_seq_len),
            persistent=False,
        )

    def forward(self, tokens):                      # tokens: [B, T] 整数 id
        B, T = tokens.shape
        x = self.tok_embeddings(tokens)             # [B,T,dim]，只有 token 嵌入，无绝对位置！
        freqs_cis = self.freqs_cis[:T]
        for layer in self.layers:
            x = layer(x, freqs_cis)                 # 逐层：沟通(attn) + 思考(ffn)
        x = self.backend.rmsnorm(x, self.norm_w)    # 最终归一化
        return self.lm_head(x)                      # [B,T,vocab_size] logits
```

> **英文名词：Weight Tying（权重绑定 / 权重共享）**
> 让**输入嵌入矩阵**（token id → 向量）和**输出投影矩阵**（向量 → 词表 logits）**用同一张权重**。
> - **为什么能共享**：这两件事互为逆操作——一个"把词编码成向量"，一个"把向量解码回词"。用同一张表，逻辑自洽。
> - **为什么要共享**：这张表尺寸是 `vocab_size × dim`，词表几万时它是**全模型最大的参数块之一**。共享直接省掉一整份，小模型上还常常提升效果（正则作用）。
> - **类比**：一本中英词典，正查（中→英）和反查（英→中）用**同一本书**，而不是印两本。省纸，且两个方向天然一致。
> - **易错点**：`self.lm_head.weight = self.tok_embeddings.weight` 必须是**同一个对象引用**（这样梯度会累加到同一张表），不能是 `.clone()`（那就变两张独立的了）。

---

## 第六部分：诚实的差异清单（我的实现 vs 真 Llama-3）

**知道自己简化了什么，和做出来同样重要。** 这延续你 W8 的诚实纪律，也是汇报时"我清楚边界在哪"的专业体现。

| 维度 | 我的实现 | 真 Llama-3 | 差异意味着什么 / 何时补 |
|---|---|---|---|
| **注意力** | MHA（每个 query 头配独立 K/V 头） | **GQA**（多个 query 头共享一组 K/V 头） | KV Cache 显存差数倍。**W5 学 KV Cache 分页时补**（见下方详解，它就是为省 cache 生的） |
| **Tokenizer/词表** | 沿用 nanoGPT（字符级或 GPT-2 BPE，5 万级词表） | SentencePiece/tiktoken，**12.8 万词表** | 词表大小影响 embedding 参数量和覆盖度。不影响架构学习，暂不动 |
| **长上下文 scaling** | 无（RoPE 原始频率） | RoPE scaling（长上下文外推） | 只在需要 >训练长度时才有影响。W7 之后再碰 |
| **权重 tying** | 绑定（省参数，nanoGPT 惯例） | Llama-3 **8B/70B 不绑定**（小模型如 3.2-1B 绑定） | 模型越大，独立 output 头收益越明显；小模型绑定更划算 |
| **精度/dtype** | 先 fp32/bf16 跑对 | bf16 训练、推理可量化 | Day7 之后引入量化后端时对齐 |

### GQA 详解（W5 的天然伏笔，今天先建立概念）

> **英文名词：GQA（Grouped-Query Attention，分组查询注意力）**
> 注意力头有三种玩法，区别在"K/V 头有几个"：
> - **MHA（Multi-Head Attention，多头注意力）**：n 个 query 头，配 n 个 K/V 头（一对一）。我现在用的就是这个。
> - **MQA（Multi-Query Attention）**：n 个 query 头，**只配 1 个 K/V 头**（全共享）。省 cache 最狠，但效果略降。
> - **GQA**：折中——n 个 query 头分成 g 组，**每组共享 1 个 K/V 头**。比如 32 个 query 头配 8 个 K/V 头，每 4 个 query 头共用一组 K/V。

> **为什么 GQA 是"为省 KV Cache 而生"？**
> 推理时 KV Cache 的显存 ∝ **K/V 头的数量**。MHA 有 32 组 K/V 要缓存；GQA 只缓存 8 组——**KV Cache 直接省 4 倍**。省下的显存可以：装更长的上下文、塞更大的 batch、或者省钱。而 query 头数量不变，模型表达力损失很小。
> **类比**：MHA 像 32 个学生每人各带一本参考书（32 本占空间）；GQA 像每 4 个学生共用 1 本参考书（8 本，省 4 倍书架），做题（query）还是各做各的，几乎不影响成绩。
> **为什么它是 W5 的伏笔**：你 W5 要做 KV Cache 分页（PagedAttention）去省显存——而 GQA 是从**模型结构层面**省 cache，PagedAttention 是从**显存管理层面**省 cache，两者正交、叠加使用。今天先埋下"KV Cache 是推理显存大头、值得从多个角度省"这个认知，W5 会串起来。

---

## 第七部分：sanity 测试 —— 随机权重跑通 + 手算参数量

组装完先别急着加载权重，用**随机权重**做一遍体检。**手算参数量这个动作，是把架构真正吃透的最快自检**——公式对不上，说明你哪块结构没算清。

```python
# tests/test_model_sanity.py
# 运行：pytest tests/test_model_sanity.py -v -s
# 环境：Python 3.10+ / torch 2.x / pytest
import torch
from engine.backend import TorchBackend
from engine.model import LlamaModel


class Cfg:                      # 一个小配置，CPU 上秒跑
    vocab_size = 50304
    dim = 512
    n_layers = 8
    n_heads = 8
    max_seq_len = 256
    multiple_of = 256


def build():
    torch.manual_seed(0)
    return LlamaModel(Cfg(), TorchBackend())


def test_forward_shape_and_finite():
    """体检①：形状对 + 无 NaN/Inf + logits 数量级合理。"""
    model = build().eval()
    B, T = 2, 16
    tokens = torch.randint(0, Cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits = model(tokens)

    # 形状：输出必须是 [B, T, vocab]
    assert logits.shape == (B, T, Cfg.vocab_size), f"形状错: {logits.shape}"
    # 有限性：随机初始化 + 正确结构，绝不该出 NaN/Inf（出了 = norm/残差/初始化有 bug）
    assert torch.isfinite(logits).all(), "出现 NaN/Inf！检查 RMSNorm 的 eps 和残差"
    # 数量级：随机权重下 logits 应是 O(1)~O(10)，几百上千说明缩放/初始化失控
    assert logits.abs().max() < 100, f"logits 数量级异常: {logits.abs().max():.1f}"
    print(f"[OK] logits shape={tuple(logits.shape)}, max|logit|={logits.abs().max():.2f}")


def count_params_by_formula(c: Cfg) -> int:
    """体检②：按架构公式手算参数量（权重 tying：embedding 只算一次）。
    把这个函数当'架构自测'——它逼你写清每一块到底有多少参数。"""
    from engine.model import llama_ffn_hidden_dim
    h = llama_ffn_hidden_dim(c.dim, c.multiple_of)

    emb   = c.vocab_size * c.dim                 # token 嵌入(与 lm_head 共享，只算一次)
    attn  = 4 * c.dim * c.dim                    # 每层注意力：Wq/Wk/Wv/Wo 四个 [d×d]
    ffn   = 3 * c.dim * h                        # 每层 SwiGLU：gate/up/down 三个矩阵
    norms = 2 * c.dim                            # 每层两个 RMSNorm 的缩放向量
    per_layer = attn + ffn + norms
    final_norm = c.dim                           # 最后一个 RMSNorm
    return emb + c.n_layers * per_layer + final_norm


def test_param_count_matches_formula():
    """实测参数量必须等于手算公式 —— 对不上就是架构理解有漏洞。"""
    model = build()
    actual = sum(p.numel() for p in model.parameters())
    # 权重 tying 下，sum(parameters()) 里 embedding 只出现一次，与公式一致
    formula = count_params_by_formula(Cfg())
    print(f"[参数量] 实测={actual:,}  手算={formula:,}")
    assert actual == formula, f"对不上！实测 {actual:,} vs 手算 {formula:,}"
```

> **参数量公式（记住这张账）**：
> ```
> 总参数 ≈ vocab×dim                          ← 词嵌入（tying 后只此一份）
>        + n_layers × ( 4d²  + 3dh  + 2d )    ← 每层：注意力 + FFN + 两个 norm
>        + dim                                 ← 最终 norm
> ```
> **易错点**：① 有没有 weight tying，决定 embedding 算一次还是两次（差一整个 `vocab×dim`，往往是最大的一块）；② GQA 会让注意力那项从 `4d²` 变小（K/V 矩阵变窄）；③ 别忘了 norm 的那些小向量。**手算和实测一分不差**，才说明你真把这台整机的每颗螺丝都数清楚了。

---

## 第八部分：常见陷阱与调试技巧

| 陷阱 | 现象 | 根因 & 解法 |
|---|---|---|
| **残差加错对象** | 训练不收敛/效果差，但不报错 | pre-norm 里残差必须加**原始 x**，不是 `norm(x)`。写成 `x=norm(x)+attn(norm(x))` 就毁了高速公路 |
| **hidden dim 直接用 4d** | 参数量比论文多 50%，"效果好"是假象 | SwiGLU 要用 `2/3×4d` 才和普通 FFN 参数对齐（第三部分）|
| **忘了取整 multiple_of** | 维度是零头数、matmul 走慢路径 | 向上取整到 256 倍数做硬件对齐 |
| **weight tying 用了 clone** | 参数量对不上、两张表各训各的 | 必须是同一对象引用 `lm_head.weight = tok_emb.weight` |
| **logits 爆炸出 NaN** | sanity 测试 isfinite 挂 | 查 RMSNorm 的 eps（太小易溢出）、残差是否写反、初始化是否失控 |
| **FFN 里 gate/up 用反** | 结果不对但形状没错，难查 | SiLU 只加在 gate 路(W1)，up 路(W3)保持线性。加反了数学就变了 |
| **误以为门是对称可随便叫** | 概念混乱 | 乘法对称但激活不对称：SiLU 在哪条路，哪条路就是"内容"。命名跟 Llama 官方走 |

---

## 第九部分：副线 & 整理块（简要）

**副线块 1.5h**：见第七部分——随机权重 forward、逐层核对 shape、查 NaN/Inf、验 logits 量级、**手算参数量对上**。把这次的层输出 shape 和参数量记进 `tech_notes/swiglu_and_llama_block.md`。

**副线产出 `tech_notes/swiglu_and_llama_block.md`**：三段式提炼——① 原理：门 = 逐通道条件阀门（乘法带来条件计算与特征交互）；② 代码：SwiGLUFFN + LlamaBlock 数据流图；③ 工业锚点：`2/3×4d` 参数对齐 + 256 硬件对齐 + GQA 是 W5 伏笔。开头一句串起本周：*"Day2 让位置对了(RoPE)、数值稳了(RMSNorm)，Day3 让每个 token 的'思考'更聪明(SwiGLU)，并把三块拼成一台完整的 Llama 整机。"*

**整理块 1h**：week8_triton README，照你 W8 Day7 笔记 §7.3 的模板落地（30 分钟能完），剩余时间清理仓库目录结构。

---

## ✅ 今日完成标准自测（能脱稿 = 真懂）

1. **脱稿画出 LlamaBlock 数据流图。**
   `x → rmsnorm → attn(RoPE) → +x → rmsnorm → SwiGLU-FFN → +（上一步结果）→ 输出`。两个 pre-norm 子层，残差各自加**未归一化的输入**。

2. **SwiGLU 三个矩阵各干什么？**
   `W1(gate_proj)`：内容路，过 SiLU；`W3(up_proj)`：门路，线性，当逐通道阀门；`W2(down_proj)`：把 `SiLU(W1x)⊙(W3x)` 压回 d 维。核心是那个 `⊙`——乘法带来"输入相关的逐通道调节"和"特征交互"。

3. **hidden dim 为什么是 `2/3×4d`？**
   普通 FFN 2 个矩阵、hidden=4d、参数 8d²；SwiGLU 3 个矩阵、参数 3dh；令 3dh=8d² 解得 h=2/3·4d。目的是**加了第三个矩阵后总参数量不变**，才能公平对比出"是结构好，不是参数多"。真实 Llama 还会向上取整到 256 倍数做硬件对齐。

4. **门门住了什么？** 门住"每个通道在当前上下文下该放行多少"——把 FFN 从固定标准升级成按情境自我调节，用一个乘号换来条件计算和特征交互的表达力。

**今日交付物清单：**
- [ ] `engine/model.py`：`SwiGLUFFN` + `LlamaBlock` + `LlamaModel`，全部走 Backend 接口
- [ ] `llama_ffn_hidden_dim`：`2/3×4d` + `multiple_of` 对齐
- [ ] 权重 tying（同对象引用）
- [ ] `tests/test_model_sanity.py`：shape/finite/logits 量级 + **手算参数量对上**，全绿
- [ ] 差异清单表（GQA/tokenizer/scaling/tying）写进笔记
- [ ] `tech_notes/swiglu_and_llama_block.md`（三段式）
- [ ] week8_triton README 落地
- [ ] commit（建议：`feat: assemble full Llama block with SwiGLU (Day3)`）

---

> **一句话收尾**：普通 FFN 是"所有信息过同一道固定安检"；SwiGLU 是"给每个通道配一个看情境开合的阀门"——一个乘号，把"条件计算"送给了网络。而 `2/3×4d` 那半步，是工程师的诚实：**改结构可以，但别偷偷加参数来虚报战功。** 今天你把 RMSNorm、RoPE、SwiGLU 三块地基拼成了一台完整的 Llama 整机——从明天起，你的引擎跑的不再是 nanoGPT，而是现代 LLM 的真骨架。
