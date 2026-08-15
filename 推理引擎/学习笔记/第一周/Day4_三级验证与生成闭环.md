# Day 4 · 闭环日：三级验证方法论 + generate() + 朴素 KV Cache

> **一句话主题**：怎么*科学地*证明"我手写的新架构（Llama 化的 nanoGPT）是对的"？
> 答案不是"能跑不报错"，而是一套**从便宜到贵、从粗 bug 到细 bug 的验证漏斗**。
> 这套方法论本身，比今天写的任何一行代码都值钱 —— 它是你以后每次改模型都会本能用到的反射。

---

## 0. 先想清楚：为什么"能跑起来"根本不算验证对了

你现在手上有一个刚 Llama 化的模型（RMSNorm + RoPE + SwiGLU 换掉了 nanoGPT 原来的 LayerNorm + 绝对位置编码 + GELU-MLP）。

一个新手最容易犯的错觉是：

> "我 `forward()` 跑通了，输出 shape 对，没报错 —— 那应该对了吧？"

**这是幻觉。** 随机初始化的模型 `forward` 一定能跑出一个形状正确的张量，但里面全是噪声。哪怕你的残差连接**接反了**、RoPE 用**错了维度**、norm 放**错了位置**，`forward` 照样"跑通"、照样输出 `[batch, seq, vocab]` 的 logits，照样能采样出 token —— 只不过生成的是**乱码**。

所以问题的核心变成了：

> **随机权重 → 乱码，这是正常的。我怎么区分"乱码是因为没训练"还是"乱码是因为实现有 bug"？**

唯一的办法：**让模型学会一点东西**。一个实现正确的模型，喂给它数据就一定能学会；一个实现错误的模型，要么根本学不会，要么学得明显更差。**"能不能学会"就是照妖镜。**

这就引出了今天的主线 —— 用三个层层递进的验证台阶，把这面照妖镜拆成三块，每块专抓一类 bug。

---

## 1. 三级验证：一张总览图（先建立全局观）

先给你一张地图，后面每一节都是在填这张图里的一格。**记住这个顺序，顺序本身就是知识点。**

| 台阶 | 名字 | 干什么 | 专抓哪类 bug | 成本 | 通过标准 |
|------|------|--------|-------------|------|---------|
| **一级** | 过拟合单 batch | 拿 32 条样本往死里训 | **梯度流断裂**（残差接反、norm 位置错、RoPE 维度错、某层没接进计算图） | 最便宜（~30s 就能看出苗头） | loss → ~0 |
| **二级** | 小训练对照 | tiny 配置跑 tinyshakespeare，对照 W6 nanoGPT | **隐性 bug**（能训但训不好，最阴险） | 中等（H100 上 ~1h） | loss 曲线 ≈ 或略优于 nanoGPT |
| **三级** | 生成质量 | 实现 generate()，看真实文本 | **推理路径专属 bug**（KV Cache 错、位置 offset 错、采样错） | 便宜（~几分钟） | 生成通顺的莎士比亚风格文本 |

**为什么顺序绝对不能反？** —— 这是今天完成标准里明确要你能讲清的点，我先埋在这，第 5 节专门展开：

> 核心逻辑：**每一级都假设前一级已经通过。** 如果梯度流是断的（一级没过），你去跑二级的完整训练纯属烧钱看一条永远不下降的 loss 曲线；如果模型压根训不好（二级没过），你去看三级的生成质量，看到的乱码分不清是"没训好"还是"generate 写错了"。**低级台阶用最小成本，把高级台阶的干扰变量提前排除掉。**

---

## 2. 一级验证 —— 过拟合单 batch（overfit a single batch）

### 2.1 是什么

> **过拟合（overfitting）**：模型不去学数据背后的*规律*，而是死记硬背了训练样本本身。在正常训练里过拟合是要极力避免的坏事 —— 但在**验证阶段，我们主动追求它**。

具体做法：从数据集里**只**拿一小撮样本（比如 32 条 token 序列），把它们当成*全部*数据，反复喂给模型训练几百步，看 loss 能不能掉到接近 0。

### 2.2 为什么这招能抓 bug（核心原理）

打个比方：**你要测一台复印机能不能用，最快的办法不是让它印一本 500 页的书，而是让它印一张纸,看印出来的那张和原件一不一样。**

一个**正确实现**的神经网络，其表达能力（参数量）远远超过"记住 32 条样本"所需 —— 它*一定*有能力把这 32 条死记硬背下来，loss 必然能压到近 0。

反过来，如果 loss **卡住下不去**（比如卡在 5.x 一直不动），几乎可以 100% 断定：**梯度没能正确地流回某些参数**。常见元凶：

- **残差连接接反 / 漏接**：`x = x + attn(norm(x))` 写成了 `x = attn(norm(x))`，梯度高速公路断了。
- **norm 位置错**：Pre-Norm 写成了 Post-Norm，或者 norm 加在了错误的地方。
- **RoPE 用错维度**：旋转位置编码作用在了错误的 head_dim 切分上，Q/K 语义被搅乱。
- **某个子模块没接进计算图**：算了但没用上，或用了 `.detach()` 把梯度掐断了。

**这是工业界调新模型时的第一反射动作，比瞪着代码逐行看高效十倍。** 因为它不需要你猜 bug 在哪，它直接告诉你"梯度流有问题"这个大方向。

### 2.3 可运行代码（贴近你的 W6 工程）

```python
# 环境: PyTorch >= 2.1, CUDA (H100 或任意 GPU 均可, 单 batch 极小 CPU 也行)
# 依赖: torch。假设你已有 GPTLlama 模型类和 nanoGPT 风格的 config。
import torch

def overfit_single_batch(model, device="cuda", steps=300, log_every=50):
    """一级验证: 拿一个固定 batch 往死里训, 看 loss 能否 -> ~0。
    通过 = 梯度流是通的; 卡住 = 计算图/梯度有 bug。"""
    model.train().to(device)

    # 关键: 只造一个 batch, 之后每一步都用这同一份数据 —— 这才叫"过拟合单 batch"
    B, T = 4, 64                                  # 32 条样本量级, 小到模型必能背下来
    x = torch.randint(0, model.config.vocab_size, (B, T), device=device)
    y = torch.randint(0, model.config.vocab_size, (B, T), device=device)
    # 注: 这里用随机 token 就够验梯度流了; 想更真实可从 tinyshakespeare 取真实片段

    # 学习率故意调大: 我们不怕过拟合, 只想尽快看到 loss 崩塌
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(steps):
        logits = model(x)                         # [B, T, vocab_size]
        # 交叉熵: 把 [B,T,vocab] 摊平成 [B*T, vocab], target 摊成 [B*T]
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        optimizer.zero_grad(set_to_none=True)     # set_to_none 比置零省一次写显存, 工业惯例
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    final = loss.item()
    # 判定: vocab 若为 ~65 (char 级), 随机基线 loss ≈ ln(65) ≈ 4.17
    # 训 300 步还在 4 附近 -> 梯度流几乎肯定断了
    assert final < 0.1, f"❌ 一级验证失败! loss 卡在 {final:.3f}, 去查残差/norm/RoPE"
    print(f"✅ 一级验证通过, final loss = {final:.4f}")
```

> **调试技巧**：如果 loss 卡住，别急着改。先加一行 `print(model.some_layer.weight.grad.abs().mean())` 看某层梯度是不是 `0` 或 `None`。**梯度是 `None`** → 这层根本没进计算图；**梯度是 `0`** → 进了图但被掐断（常见于 detach 或 ReLU 死区）。这一步能把范围从"整个模型"缩小到"某一层"。

---

## 3. 二级验证 —— 小训练对照（controlled small-scale training）

### 3.1 是什么，以及它抓的是最阴险的 bug

一级验证只能证明"梯度流是通的"。但有一类 bug **梯度流完全正常，模型也能学，就是学得比它应该的差** —— 这类 bug 一级验证抓不到，因为 loss 照样能降到 0（32 条样本太好背了，什么烂实现都能背下来）。

举个真实例子：**RoPE 的 base（θ 频率基数）写错了**，或者 **RMSNorm 的 eps 量级不对**，或者 **注意力缩放因子少除了一个 `sqrt(head_dim)`**。这些 bug 不会让梯度断掉，模型照样训，但收敛会明显变慢或最终 loss 偏高。这就是所谓 **"能训但训不好"** —— 最难抓，因为没有任何报错，一切看起来都正常。

**怎么抓？—— 找一个可信的参照物做对照实验（controlled experiment）。**

> **对照实验**：科学方法的基石。你想知道变量 X（这里是"Llama 化改造"）的影响，就固定其他所有条件，只对比"有 X"和"没 X"两组的结果差异。

你的参照物是现成的：**W6 亲手写的 nanoGPT**。它是经过验证的、正确的基线。做法：

1. 用**同规模的 tiny 配置**（4 层，和你 W6 nanoGPT 同参数量级）；
2. 在**同一份数据**（tinyshakespeare）上训；
3. 用**同样的超参**（lr、batch、步数）；
4. 把两条 loss 曲线画在**同一张图**上对比。

### 3.2 预期结果与判读

- **Llama 化版本 loss ≈ nanoGPT 或略优** → ✅ 通过。（RMSNorm/RoPE/SwiGLU 在小模型上通常带来微小但真实的提升，这也是 Llama 用它们的原因。）
- **两条曲线几乎重合** → 大概率也 OK，说明改造没引入退化。
- **Llama 版明显更差**（比如 nanoGPT 收敛到 1.5，你的卡在 2.5） → ❌ 有隐性 bug。**这时候回去查那些"不影响梯度流但影响质量"的地方**：RoPE base、注意力 scale、norm 的 eps、SwiGLU 的中间维度算错没。

> **为什么这一步值得花 1 小时在 H100 上跑？** 因为隐性 bug 会一路潜伏到你做推理优化时才爆发 —— 到那时你根本分不清"是我 kernel 优化写错了"还是"模型本来就有病"。**在源头把模型钉死为正确，是后面所有优化工作的地基。**

### 3.3 对照代码骨架

```python
# 环境: PyTorch >= 2.1 + CUDA(H100), 需要 tinyshakespeare 的 train.bin/val.bin
# 依赖: torch, numpy, matplotlib(可选, 画曲线)
import torch, numpy as np, matplotlib.pyplot as plt

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        losses[k] = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1))
    model.train()
    return losses.mean().item()

def train_and_log(model, data, name, device="cuda",
                  block_size=256, batch_size=32, max_iters=2000, eval_every=100):
    """训练并返回 loss 历史, 用于两个模型画在同一张图上对照。"""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    history = []
    model.train().to(device)
    for it in range(max_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        loss = torch.nn.functional.cross_entropy(
            model(x).view(-1, model.config.vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % eval_every == 0:
            val = estimate_loss(model, data, block_size, batch_size, device)
            history.append((it, val))
            print(f"[{name}] iter {it:4d} | val loss {val:.4f}")
    return history

# 用法: 同一份 data、同一套超参, 只换 model
# h_nano  = train_and_log(nanogpt_model,  data, "nanoGPT")
# h_llama = train_and_log(llama_model,    data, "Llama化")
# 把两条 history 画在一张图 -> 一眼看出有没有隐性 bug
```

---

## 4. 三级验证 —— 生成质量（实现 generate() + 朴素 KV Cache）

前两级验证的都是**训练路径**（一次性喂进整个序列，并行算所有位置）。但推理/生成走的是**另一条代码路径**：一个 token 一个 token 地往外吐。这条路径有它**专属的 bug**（KV Cache 写错、位置 offset 错、采样错），前两级一个都抓不到。所以必须有第三级。

而且 —— **KV Cache 是你整个暑假推理优化的主战场**，今天把它的朴素版彻底吃透，是后面 PagedAttention、连续批处理这些高级货的地基。

### 4.1 先理解痛点：为什么自回归生成天生很慢

> **自回归（autoregressive）生成**：模型每次只预测**下一个** token，然后把这个新 token 拼回输入，再预测下下个，循环往复。就像一个人写字，写一个字要先把前面写过的全部读一遍。

问题就出在"把前面全部读一遍"。假设已经生成了 100 个 token，要生成第 101 个：

- **朴素做法（无 Cache）**：把这 100 个 token 全塞进模型，走一遍完整的 attention。但注意 —— attention 里前 100 个 token 的 **Key 和 Value 其实和上一步算出来的一模一样**！你把它们**重算了一遍**，纯属浪费。
- 生成第 102 个时，又把 101 个重算一遍……

**这就是 O(N²) 的重复计算灾难。** 生成一篇 1000 词的文章，越到后面每一步越慢。

### 4.2 KV Cache 是什么：一句话与一个类比

> **KV Cache（键值缓存）**：把每一步算出来的 Key 和 Value **存起来**，下一步只算**新 token 那一个**的 K/V，然后和缓存里的老 K/V 拼在一起用。用**空间换时间**，把每步的计算从"重算全部"降到"只算一个"。

**类比**：你在做一道需要用到前面所有中间结果的连环计算题。

- **没有 Cache**：每算新一步，都把前面所有中间结果*从头推导一遍*。
- **有 Cache**：你拿张草稿纸（就是 Cache），把每步的中间结果记下来，下一步直接查草稿纸，只算新增的那部分。

**为什么只缓存 K 和 V，不缓存 Q？** —— 这是最容易被问懵的点。看 attention 的本质：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

生成第 101 个 token 时，我们只关心**这一个新位置**作为"query（提问者）"去看前面所有位置。所以：

- **Q（Query，查询）**：只需要**当前这个新 token 的** Q。老 token 的 Q 这一步用不上，不用缓存。
- **K（Key，键）/ V（Value，值）**：新 token 的 Q 要去和**所有**（含全部历史）token 的 K 算注意力、再对所有 V 加权。所以历史的 K/V **每一步都要用到** → 必须缓存。

一句话记牢：**Q 是"当前提问者"，只要最新一个；K/V 是"被查阅的全部历史"，要全留着。**

### 4.3 两个阶段：Prefill 与 Decode（工业界的标准划分）

带 Cache 的生成天然分成两个阶段，这个划分是所有推理引擎（vLLM、TensorRT-LLM）的通用语言，务必记牢：

> **Prefill（预填充）阶段**：处理用户输入的**整段 prompt**。因为 prompt 的所有 token 一开始就都知道，可以**一次性并行**跑完，顺便把它们的 K/V 全部算好、填进 Cache。这一步是**计算密集（compute-bound）**的 —— 一大堆矩阵乘法把 GPU 喂得饱饱的。

> **Decode（解码）阶段**：进入一个一个吐 token 的循环。每一步**只**把上一步新生成的**那一个** token 送进模型，算它的 Q/K/V，K/V 追加进 Cache，用它的 Q 和整个 Cache 做注意力，得到下一个 token。这一步是**访存密集（memory-bound）**的 —— 计算量极小（就一个 token），但每步都要把整个模型权重 + 整个 KV Cache 从显存搬一遍，瓶颈在带宽。

> **一个价值千金的认知**：Prefill 是 compute-bound、Decode 是 memory-bound —— 这是你暑假**所有推理优化的分水岭**。优化 Prefill 靠堆算力/更好的 kernel；优化 Decode 靠减少访存（KV Cache 压缩、量化、PagedAttention）。**你以后测的 tok/s，绝大部分时间花在 Decode 上。** 今天先把它跑通，后面每个优化都是在啃这块骨头。

### 4.4 朴素 KV Cache（`torch.cat` 版）—— 你 W6 亲手写的那版

"朴素"体现在：用 `torch.cat` 每步把新 K/V 拼到旧 Cache 后面。它简单、直观、正确，**但有性能缺陷**（4.6 节讲），正好是你后面要优化的对象。

先看**改造后的注意力层**如何吃 Cache：

```python
# 环境: PyTorch >= 2.1。这是 Llama 风格单头/多头注意力的 forward, 支持传入并更新 KV Cache。
import torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, freqs_cis, kv_cache=None):
        B, T, C = x.shape                          # 训练时 T=整段; decode 时 T=1
        q, k, v = self.qkv(x).split(C, dim=2)
        # 拆成多头: [B, T, n_head, head_dim] 再转成 [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE: 用"绝对位置"给 q,k 注入位置信息。decode 时 freqs_cis 必须是"当前位置"那一段!
        q, k = apply_rope(q, k, freqs_cis)

        if kv_cache is not None:
            past_k, past_v = kv_cache               # 取出历史 K/V
            if past_k is not None:
                # 朴素版核心: torch.cat 把新 K/V 拼到历史后面 (沿序列维 dim=2)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)                       # 更新后的完整 Cache 返回出去
        else:
            new_cache = None

        # decode 阶段 T=1 且 Cache 已含全部历史, 此时不需要因果 mask;
        # prefill/训练阶段 T>1, is_causal=True 保证不看未来。
        is_causal = (kv_cache is None) or (past_k is None)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache
```

> **注意那个 `is_causal` 的分支** —— 这是 decode 路径专属的坑，也是三级验证专抓的 bug 之一。Decode 时当前 token（T=1）本来就该看到 Cache 里的全部历史，如果你无脑套 `is_causal=True`，SDPA 会按"下三角"去 mask 一个 1×N 的注意力矩阵，逻辑就错了。这种 bug **训练时永远不会暴露**（训练走的是 `kv_cache is None` 那条路），只有生成时才现形。

### 4.5 头号大坑：RoPE 的位置 offset（decode 专属 bug）

这个坑值得单独拎出来，因为它是"三级验证不可替代"的最佳证据。

> **RoPE（Rotary Position Embedding，旋转位置编码）**：不给 token 加一个位置向量，而是根据 token 的**绝对位置**把它的 Q/K 向量"旋转"一个角度。位置越靠后转得越多，注意力算内积时这些角度差就编码了相对距离。

关键在于 RoPE 依赖**每个 token 的绝对位置**。训练/prefill 时你一次性喂 `[0, 1, 2, ..., T-1]` 整段位置，天经地义。**但 decode 时每步只送 1 个 token，它的位置不是 0，而是"当前 Cache 里已经有多少个 token"！**

- 生成第一个新 token（Cache 已有 100 个 prompt token）→ 它的位置是 **100**，不是 0。
- 下一个 → 位置 **101**。

如果你偷懒每次都用位置 0 去做 RoPE，会发生什么？**训练 loss 完全正常，一级二级验证全过，但生成出来是乱码或不断重复** —— 因为模型以为每个新 token 都站在序列开头。**这就是为什么必须有三级验证：这个 bug 只在自回归生成时才现形。**

正确做法：维护一个 `pos` 计数器，每步用 `freqs_cis[pos : pos + T]` 这一段切片。

### 4.6 完整的 generate()（prefill + decode 两阶段）

```python
# 环境: PyTorch >= 2.1 + CUDA。model 需支持 forward(idx, kv_caches, pos) 返回 (logits, new_caches)
import torch, torch.nn.functional as F

@torch.no_grad()                                   # 生成不需要梯度, 关掉省显存省时间
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, device="cuda"):
    """idx: [B, T_prompt] 起始 prompt 的 token id。返回 [B, T_prompt+max_new_tokens]。"""
    model.eval()
    B, T_prompt = idx.shape
    # 每一层一个 (k, v) Cache, 初始为 None
    kv_caches = [(None, None) for _ in range(model.config.n_layer)]

    # ---------- 阶段一: Prefill ----------
    # 一次性把整段 prompt 喂进去, 位置从 0 开始, 顺便填满所有层的 Cache
    logits, kv_caches = model(idx, kv_caches=kv_caches, pos=0)
    pos = T_prompt                                 # 关键: 下一个 token 的绝对位置 = prompt 长度
    next_logits = logits[:, -1, :]                 # 只取最后一个位置的输出来预测下一个 token

    generated = idx
    # ---------- 阶段二: Decode 循环 ----------
    for _ in range(max_new_tokens):
        # 1) 用当前 logits 采样出下一个 token
        next_token = sample(next_logits, temperature, top_k)   # [B, 1], 见 4.8 节
        generated = torch.cat([generated, next_token], dim=1)

        # 2) 只把这 1 个新 token 送进模型; pos 告诉 RoPE 它的真实绝对位置
        logits, kv_caches = model(next_token, kv_caches=kv_caches, pos=pos)
        next_logits = logits[:, -1, :]
        pos += 1                                    # 位置递增, 绝不能忘 (见 4.5 的坑)

    return generated
```

> **两阶段在代码里的体现**：Prefill 是循环外那**一次** `model(idx, ...)`（T=T_prompt，并行）；Decode 是循环内每次 `model(next_token, ...)`（T=1，串行）。`pos` 这个变量就是把 4.5 那个坑填平的关键 —— 它保证每个新 token 的 RoPE 都用对了绝对位置。

### 4.7 为什么叫"朴素"：`torch.cat` 的性能缺陷（你后面要优化的靶子）

朴素版正确、好懂，但每一步 `torch.cat([past_k, k], dim=2)` 都干了一件昂贵的事：

> `torch.cat` **不是**原地追加，它会**分配一块全新的、更大的连续显存**，然后把旧 Cache 整个复制过去 + 新的拼上。

于是每一步 decode 都在**重新拷贝整个历史 Cache**。生成到第 N 步，累计拷贝量是 `1 + 2 + 3 + ... + N ≈ N²/2` —— **又一个 O(N²)！** 只不过这次浪费在**显存搬运（memory traffic）**上，不是计算上。对 memory-bound 的 decode 阶段来说，这刀正好砍在命门上。

**工业界的解法（预告 W1+ 的优化方向）**：**预分配静态 Cache（static / pre-allocated KV cache）**。一开始就按 `max_seq_len` 开好一整块显存 `[B, n_head, max_len, head_dim]`，每步用**索引赋值**把新 K/V *写入*对应位置（`cache_k[:, :, pos] = new_k`），不再重新分配、不再拷贝。这是从 O(N²) 访存降到 O(N) 的关键一步，也是 `torch.compile` 能否 capture 成 CUDA Graph 的前提（动态 shape 会打断 graph）。

> 记住这个对比：**朴素 cat = 每步换一张更大的草稿纸并抄一遍；静态 cache = 一开始就用大本子，每步只写新的一行。** 今天先用朴素版跑通、拿到基线，后面再换静态版看 tok/s 提升多少 —— 这就是优化工作的意义。

### 4.8 采样：温度（temperature）与 top-k

模型每步输出的是 vocab 上的一串 **logits（未归一化的打分）**。怎么从中挑出下一个 token？直接取最大（贪心）会让生成呆板重复。工业界标配是**温度 + top-k**。

> **温度（temperature）**：一个除在 logits 上的数，控制"随机性"。`logits / T` 后再 softmax。**T 越小 → 分布越尖 → 越确定（趋近贪心）；T 越大 → 分布越平 → 越随机越有创意。** 类比：温度就像"脑洞开合度"，T=0.7 稳一点，T=1.2 放飞一点。

> **top-k**：softmax 前只保留分数最高的 k 个候选，其余全部置为 `-inf`（概率归零）。作用是**掐掉长尾里的垃圾 token**，防止偶尔采到一个莫名其妙的词把整句带跑偏。

```python
# 环境: PyTorch >= 2.1。logits: [B, vocab_size]。返回下一个 token id: [B, 1]
import torch, torch.nn.functional as F

def sample(logits, temperature=1.0, top_k=None):
    if temperature == 0.0:                          # T=0 约定为贪心解码 (取 argmax)
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature                   # 先按温度缩放, 调节尖锐程度
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)                # 取每行 top-k 的值
        # 把低于"第 k 大值"的所有 logits 打成 -inf, softmax 后概率=0
        logits[logits < v[:, [-1]]] = float("-inf")

    probs = F.softmax(logits, dim=-1)               # -> 概率分布
    return torch.multinomial(probs, num_samples=1)  # 按概率随机抽 1 个, 保留多样性

# 常用档位: 稳健复现用 temperature=0.8, top_k=200;
#           要多样性调高 T; 做确定性对比实验用 temperature=0.0(贪心)
```

> **三级验证怎么用采样？** 生成时**先用 `temperature=0.0`（贪心）**看输出是否通顺 —— 贪心是确定的，排除了采样随机性的干扰，最容易判断"是不是 generate 逻辑本身有 bug"。确认逻辑对了，再开温度看多样性。**先排除变量，再看效果**，和一级验证同一个思路。

---

## 5. tok/s 基线 —— "这个数字是暑假所有优化的分母"

### 5.1 为什么这一笔基线数据这么重要

> **吞吐量（throughput）**，这里具体指 **tokens/second（tok/s，每秒生成多少 token）**：衡量推理速度的最核心指标。

优化的本质是**对比**。你说"我把推理速度提升了 20%"，提升 20% 是相对**什么**？就是相对今天这第一笔基线。从 W1 起你每换一个后端（`torch.compile`、静态 Cache、FlashAttention、CUDA Graph……），都要把新的 tok/s 和它比。**没有基线，所有"提升"都是空话。** 所以今天这个数字，是你整个暑假优化叙事的分母（denominator）。

### 5.2 头号大坑：CUDA 是异步的，`time.time()` 直接量会量到假数据

这是 profiling 新手 100% 会踩的坑，也是你接手 H100 nsys profiling 前必须先内化的认知：

> **CUDA 异步执行（asynchronous execution）**：当你在 Python 里写 `y = model(x)`，CPU 只是把 GPU kernel **丢进一个队列**就立刻返回了，**并不等 GPU 真正算完**。GPU 在后台慢慢跑。

后果：如果你这样计时——

```python
t0 = time.time()
out = generate(...)      # CPU 瞬间返回, GPU 还在后台狂算
t1 = time.time()         # 你量到的只是"把任务扔进队列"的时间, 快得离谱且完全是假的!
```

—— 量出来的 tok/s 可能虚高几十倍，纯属自欺欺人。

**正解**：计时前后各插一个 `torch.cuda.synchronize()`，它会**阻塞 CPU 直到 GPU 把队列里的活全干完**，这样两个时间戳之间才是真实的 GPU 执行时间。

### 5.3 正确的基线测量代码

```python
# 环境: PyTorch >= 2.1 + CUDA(H100)。演示如何"正确"测 decode 吞吐。
import torch, time

@torch.no_grad()
def benchmark_tokens_per_sec(model, prompt_idx, max_new_tokens=256,
                             warmup=2, runs=5, device="cuda"):
    model.eval()

    # 1) 预热(warmup): 头几次跑包含 cuDNN 算法选择、CUDA 上下文初始化、
    #    (若用了 torch.compile)编译等一次性开销, 必须丢弃, 否则严重拉低 tok/s。
    for _ in range(warmup):
        generate(model, prompt_idx, max_new_tokens, device=device)
    torch.cuda.synchronize()                        # 等预热真正结束

    # 2) 正式计时: 前后都 synchronize, 中间是纯 GPU 执行时间
    timings = []
    for _ in range(runs):
        torch.cuda.synchronize()                    # 起点前先清空队列
        t0 = time.perf_counter()                    # perf_counter 比 time.time 精度更高
        generate(model, prompt_idx, max_new_tokens, device=device)
        torch.cuda.synchronize()                    # ★ 关键: 等 GPU 真正算完再记终点
        timings.append(time.perf_counter() - t0)

    avg = sum(timings) / len(timings)
    tok_per_s = max_new_tokens / avg                # 只算"新生成"的 token, 不含 prompt
    print(f"平均耗时 {avg*1000:.1f} ms | 吞吐 {tok_per_s:.1f} tok/s")
    return tok_per_s
```

> **基线要记录的不止一个数字，还要记下"测量条件"**：prompt 长度、生成长度、batch size、精度（fp32/bf16）、GPU 型号、有没有 KV Cache。因为 tok/s 对这些极度敏感 —— 换了条件的对比是耍流氓。工业界把这些叫 **benchmark 的 "运行配置指纹"**，你的 `bench_utils` 应该把它们一起存进结果文件。

> **decode vs prefill 分开测更专业**：Prefill(处理 prompt) 和 Decode(逐 token) 的 tok/s 差一个数量级。真正做优化时你会分别报告 "prefill tok/s" 和 "decode tok/s"，因为它们瓶颈不同(compute-bound vs memory-bound, 见 4.3)。今天先测端到端的总 tok/s 建个基线就够，心里知道这个区分即可。

---

## 6. 【完成标准核心】为什么三级验证的顺序绝对不能反

今天的自测标准明确要求：**能讲清三级验证各自抓什么 bug、为什么顺序不能反。** 这一节把它彻底讲透，这是今天最该带走的思想。

先把"各抓什么"钉死：

| 台阶 | 走哪条代码路径 | 独家能抓、别人抓不到的 bug |
|------|--------------|--------------------------|
| 一级 过拟合 | 训练路径 | **梯度流断裂**：残差断、norm 位置错、层没进图 |
| 二级 小训对照 | 训练路径 | **隐性质量 bug**：RoPE base、attn scale、eps —— 能训但训不好 |
| 三级 生成质量 | 推理路径 | **推理专属 bug**：KV Cache 拼错、RoPE 位置 offset 错、采样错 |

**顺序不能反的三条硬逻辑：**

1. **成本递增，先便宜后贵。** 一级 30 秒看苗头，二级要在 H100 烧 1 小时。你没理由先花一小时训练，回头发现是个残差接反、30 秒就能查出来的低级错误。**便宜的过滤器放前面，这是漏斗设计的第一原则。**

2. **每一级都建立在前一级的结论之上，是"排除干扰变量"的链条。** 二级要对比 loss 曲线，可如果梯度流本来就断了(一级没过)，你对比的是两条永远不降的废线，得不出任何结论。三级要看生成质量，可如果模型压根没训好(二级没过)，你看到乱码时**根本无法归因**——到底是"没训好"还是"generate 写错了"?**前一级不通过，后一级的结果就无法解释。**

3. **抓 bug 的粒度从粗到细，匹配你的排查精力。** 一级给你"梯度有问题"的大方向；二级给你"某个数值细节不对"的中方向；三级给你"推理路径某处不对"的窄范围。**从大到小逐层收缩包围圈**，而不是一上来就在最细的生成结果里大海捞针。

> 反过来想就懂了：**如果先做三级看到生成乱码，你面对的是"训练+推理"两条路径上所有可能 bug 的叠加态，无从下手。** 三级验证的本质，是**用前面的台阶把变量一个个锁死，让每一级都只面对一类新增的、可归因的问题。** 这就是科学方法在工程里的样子。

---

## 7. 【副线产出】写成可复用 SOP：`model_correctness_methodology.md`

把上面这套东西沉淀成一份**标准作业流程（SOP，Standard Operating Procedure）**，以后每次上新模型直接照着走。核心骨架：

```markdown
# 模型正确性验证 SOP (验"整个模型")
## Step 1 · 过拟合单 batch  [抓: 梯度流]
- [ ] 32 条样本, lr 调大, 训 300 步
- [ ] loss -> ~0 才算过; 卡住则打印各层 .grad 定位断点
## Step 2 · 小训练对照  [抓: 隐性质量 bug]
- [ ] tiny 配置 + tinyshakespeare, 超参对齐可信基线(nanoGPT)
- [ ] 两条 loss 曲线同图对比, 新模型应 ≈ 或略优
## Step 3 · 生成质量  [抓: 推理路径 bug]
- [ ] generate() 走 prefill+decode, KV Cache + RoPE pos 正确
- [ ] 先贪心(T=0)验逻辑, 再开温度验多样性
- [ ] 顺便记录第一笔 tok/s 基线(带运行配置指纹)
## 铁律: 顺序不可逆 —— 便宜先行, 前级为后级锁死变量
```

> **和 W8"三尺子"的关系(串联起完整正确性观)**：W8 的三尺子是验**单个 kernel** 的(数值对齐、性能、边界)；今天这三级是验**整个模型**的。两者合起来才是完整的正确性验证观 —— **微观(kernel 对不对) + 宏观(整模型对不对) 双层防线**。你以后写 CUDA kernel 用三尺子保证这块砖是好的，用三级验证保证这些砖砌成的楼是正的。在 SOP 里加一句交叉引用 `[[三尺子 kernel 验证]]`,把两份笔记链起来。

---

## 8. 【整理块】三个早期项目 README 统一

micrograd / numpy 网络 / MNIST CNN，每个 20 分钟，**不精雕**，统一成两段式即可：

- **"证明了什么"**：一句话说清这个项目在你的学习链条里补上了哪块认知。
  - *micrograd*：手写反向传播,证明我理解自动微分的本质(链式法则 + 计算图)。
  - *numpy 网络*：不靠框架、纯 numpy 实现前向+反向,证明我能脱离 PyTorch 造轮子。
  - *MNIST CNN*：第一个能跑通的真实任务,证明我掌握了完整训练闭环(数据→训练→评估)。
- **运行命令**：`python train.py` 之类,一行让别人(和三个月后的自己)能立刻跑起来。

> 目的不是文档漂亮,是让这三个项目在你简历/作品集里**各自站住一个清晰的"我证明了 X"的定位**,和今天学的"验证方法论"精神一致:**每件事都要能说清它证明了什么。**

---

## 9. 今日收尾 · 里程碑自测

**产出清单**（W0 最大的里程碑）：
- [x] 引擎端到端闭环：**训练 → 生成 → 有 tok/s 基线**
- [x] `model_correctness_methodology.md` 验证方法论 SOP
- [x] 三个早期项目 README 统一

**能过关的自测三问**（合上笔记问自己）：
1. 三级验证各抓什么 bug？ → 梯度流 / 隐性质量 / 推理路径。
2. 为什么顺序不能反？ → 成本递增 + 前级为后级锁死变量 + 粒度由粗到细。
3. 为什么 KV Cache 只存 K/V 不存 Q，decode 时 RoPE 的 pos 为什么不能是 0？ → Q 只要当前提问者，K/V 是全部历史；pos 必须是真实绝对位置，否则训练全对但生成乱码。

> **一句话总结今天**：你今天真正学会的不是"怎么写 generate"，而是**"怎么证明一个模型是对的"这套可复用的工程方法论** —— 以及顺手拿到了那个会跟随你整个暑假的 tok/s 分母。

