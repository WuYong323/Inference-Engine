# Day 1 · 引擎奠基：仓库骨架 + Backend 抽象 + 起点确认

> **今天只回答一个问题**：为什么所有真实推理框架（vLLM、TensorRT-LLM、llama.cpp……）都要把「**算子的调用**」和「**算子的实现**」拆成两层？
>
> 今天的产出不是"写了很多代码"，而是"亲手把这个分离做出来，并且能不看笔记讲清它解决了什么问题"。
>
> **本文风格沿用你的三段式**：每个核心点都走「原理 + 直觉类比 → 可运行代码（含底层视角）→ 工业锚点」。

---

## 0. 今日地图（先看全局，再逐块深入）

```
┌─ 主线深块 3.5h ──────────────────────────────────────────┐
│ 1) 建 miniLLM-serve 仓库骨架，第一天就有 commit            │
│ 2) 实现 Backend 基类 + TorchBackend（rmsnorm/attention/ffn）│
│ 3) 复制 W6 的 nanoGPT model.py，原样跑通一次生成（起点基线） │
│ 4) 写 ARCHITECTURE.md v0（五层架构 + 设计理由 + 8 周路线）   │
├─ 副线块 1.5h ────────────────────────────────────────────┤
│ 5) 给 TorchBackend.rmsnorm 写第一个单元测试（allclose 通过） │
├─ 整理块 1h ──────────────────────────────────────────────┤
│ 6) GitHub profile README 骨架（能力成长线四段小标题）        │
└──────────────────────────────────────────────────────────┘
```

这份笔记的重心放在 **1)、2)** —— 这是今天唯一的"新认知"，也是整个暑假引擎能"边学边进化"的命门。其余是把认知落地成工程习惯。

---

## 第一部分：核心原理 —— 「调用」与「实现」的分离

### 1.1 先搞清楚一个词：什么是"算子"（Operator / Op / Kernel）

**算子（Operator，简称 op）**：神经网络里的一次"计算动作"。你可以把它理解成一台专用机器，喂进去几个张量（Tensor），吐出来一个张量。

- `matmul(A, B)` 是一个算子（做矩阵乘法）
- `softmax(x)` 是一个算子
- `rmsnorm(x, w)` 是一个算子（做归一化）

> **英文名词对照**
> - **Operator（算子）**：数学上"把输入映射到输出的一个操作"。在深度学习框架里特指"框架能识别、能调度的一次计算单元"。
> - **Kernel（核函数）**：算子在某种硬件上的**具体实现代码**。同一个 `matmul` 算子，在 CPU 上是一段 C++ 代码（一个 kernel），在 GPU 上是一段 CUDA 代码（另一个 kernel）。**"算子"是抽象的动作，"kernel"是落地的代码**——这个区分是今天所有内容的地基。
> - **Tensor（张量）**：多维数组。标量是 0 维，向量 1 维，矩阵 2 维，再往上就统称张量。你 W6 里 `x.shape = [B, T, C]` 那个 `x` 就是三维张量。

**一句话**：算子 = 要做什么（矩阵乘），kernel = 具体怎么在这块硬件上做（这段 CUDA 代码）。**同一个算子可以有很多份 kernel 实现**——这正是今天分离的意义所在。

---

### 1.2 什么叫「调用」和「实现」的分离？—— 一个装修的类比

假设你家要装吊灯。有两种活法：

**活法 A（耦合）**：你在客厅墙上直接把电线焊死到某一款灯上。以后想换灯？得砸墙、重新布线。

**活法 B（分离）**：墙上先装一个**标准灯座接口**。任何符合这个接口的灯都能拧上去。想换更亮的灯？拧下来换一个，墙不用动。

在推理引擎里：

- **上层的模型结构**（Transformer 有几层、每层先 norm 再 attention 再 ffn）= 墙上的布线，它是**稳定的**，一旦模型定下来就很少变。
- **底层的算子实现**（rmsnorm 到底用 PyTorch 写、还是 Triton 写、还是手写 CUDA）= 那盏灯，它是**要反复替换的**——因为你整个暑假的核心任务，就是把这几个算子一个一个换成更快的手写版本。

**「分离」= 在这两者之间放一个标准接口（Backend）**。模型层永远只对接口说话（"给我做个 rmsnorm"），至于这次是哪份 kernel 干的活，模型层不关心、也不需要知道。

```
       模型层（稳定）                     后端实现（可替换）
  ┌──────────────────┐            ┌─────────────────────────┐
  │  for each layer:  │            │  TorchBackend  (Day 1)   │
  │    x = norm(x)    │            │  TritonBackend (W3-4)    │
  │    x = attn(x)    │──调用──▶  │  CudaBackend   (W2起)    │
  │    x = ffn(x)     │  接口      │  QuantBackend  (W7)      │
  └──────────────────┘            └─────────────────────────┘
        永远只认接口                 换实现 = 换一行配置，模型层不动
```

> **这就是"依赖倒置"（Dependency Inversion）**：不是高层代码去依赖某个具体的底层实现，而是**高层和底层都依赖同一个抽象接口**。高层说"我需要一个能做 rmsnorm 的东西"，而不是"我需要 TorchBackend 的那个 rmsnorm 函数"。这样底层怎么换，高层都无感。

---

### 1.3 为什么"分离"这件事，值得第一天就做对？

三个层层递进的理由：

**理由一：让"学一次新 kernel"立刻变成"引擎升级一次"。**
你暑假的学习节奏是：W2 学会用 CUDA 写 RMSNorm → W3 学会 FlashAttention → W7 学会量化。如果没有 Backend 抽象，每学会一个都要回去动模型层的代码，改错一次就可能把跑通的东西搞崩。有了抽象，"接入新 kernel" = 新写一个 Backend 子类 + 改一行配置，模型层一个字不用动。

**理由二：随时能"一键切回 baseline"做对标。**
这是你 W8 学的"三尺子验误差"纪律的工程化。你手写了一个 CUDA rmsnorm，怎么证明它是对的、是快的？答案：让 `CudaBackend` 和 `TorchBackend` 跑同一份输入，比数值（对不对）、比时间（快多少）。**没有分离，你连"和谁比"都拿不出来。** 分离让"正确的慢版本"永远在手边当裁判。

**理由三：这是"系统设计能力"的直接简历证据。**
简历上写"我写了个快 kernel"，是**点**的能力；写"我设计了一个可切换 kernel 后端的推理引擎，支持 PyTorch/Triton/CUDA 无缝替换并统一对标"，是**面**的能力，高一个层次。真实框架都这么做——你做的不是玩具，是工业界的标准架构缩影。

---

### 1.4 底层视角：其实 PyTorch 自己就是这么干的（Dispatcher 机制）

这一小节是给你"看穿底层"用的——**"调用/实现分离"不是我们发明的，PyTorch 内核天天在做**，只不过它做在 C++ 层，我们今天做在 Python 层。理解了这个，你就理解了整条技术栈是自相似的。

当你在 Python 里写 `torch.rms_norm(x, w)`，底下发生了什么？PyTorch 并不是"一个函数从头算到尾"，而是走了一套 **Dispatcher（分发器）**：

> **Dispatcher（分发器）**：PyTorch 内部的"总机接线员"。你喊一声"我要 rms_norm"，它先看你的张量在哪（CPU 还是 CUDA）、是什么数据类型（float32 还是 float16），然后**把电话转接到对应的那份 kernel 实现**。同一个算子名，背后挂着一张"设备 → 实现"的表。

用 C++ 伪代码看它的骨架（这是 PyTorch 注册自定义算子的真实写法，简化版）：

```cpp
// ===== 底层：PyTorch 用 TORCH_LIBRARY 宏声明"算子" =====
// 这一步只声明"有这么个动作、签名长这样"，不含任何实现
// 相当于我们 Python 里的 Backend 基类：只定义接口
TORCH_LIBRARY(myops, m) {
  m.def("rmsnorm(Tensor x, Tensor w, float eps) -> Tensor");
}

// ===== 为 CPU 注册一份实现（一个 kernel）=====
TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("rmsnorm", rmsnorm_cpu_kernel);   // CPU 上跑这段 C++
}

// ===== 为 CUDA 注册另一份实现（另一个 kernel）=====
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("rmsnorm", rmsnorm_cuda_kernel);  // GPU 上跑这段 CUDA
}
```

看清楚了吗：**`m.def(...)` 是"调用侧"（定义了叫什么、签名是什么），`m.impl(...)` 是"实现侧"（这块硬件上具体怎么算）。** 同一个 `rmsnorm` 挂了两份实现，Dispatcher 运行时根据张量在 CPU 还是 CUDA 自动选一份。

**这跟我们今天要写的 Backend 是一模一样的思想，只是层级不同：**

| | 抽象/接口侧 | 具体实现侧 | 谁来选择 |
|---|---|---|---|
| PyTorch C++ 底层 | `m.def("rmsnorm...")` | `m.impl(CPU/CUDA, ...)` | Dispatcher 按 device 自动选 |
| **我们的 Python 层** | `Backend.rmsnorm`（基类接口） | `TorchBackend / CudaBackend` | 我们按配置手动选 |

所以你今天做的，本质是"在业务层复刻了一遍 PyTorch 内核的设计哲学"。这不是巧合——**能被反复替换、又不牵动上层的东西，都长这个样子。**

---

## 第二部分：深度思考 —— 为什么接口切在 `rmsnorm / attention / fused_ffn` 这三个粒度？

这是今天完成标准里的"必答题"。别急着写代码，先把这个想透。

### 2.1 粒度问题：切太细 vs 切太粗，都是坑

抽象接口切在哪一层，是个"金发姑娘问题"（Goldilocks problem，不能太热也不能太冷，要刚刚好）：

**切太细（每个加法、每个 element-wise 乘法都抽象成一个 Backend 方法）：**
```python
# 反面教材：粒度过细
backend.add(a, b)
backend.mul(x, w)
backend.sqrt(v)
backend.mean(x)
# ... 一个 rmsnorm 被拆成七八次接口调用
```
问题在哪？
- **抽象税（Abstraction Tax）**：每过一次接口都有开销（Python 函数调用、可能的调度判断）。把 rmsnorm 拆成 8 个小算子分别过接口，等于交 8 次过路费，却没换来任何"可替换性"的价值。
- **失去融合机会**：你手写 CUDA/Triton 的核心收益恰恰来自"把 add+mul+sqrt+mean 揉进一个 kernel"（算子融合 fusion，少读写几趟显存）。如果接口逼着你一个动作一个 kernel，**你想融都融不了**——接口把你手脚绑住了。

> **英文名词：算子融合（Operator Fusion / Kernel Fusion）**
> 把本来要分开算、每步都得往显存里存一趟中间结果的多个小算子，合并成一个大 kernel 一口气算完。
> **类比**：做菜时"切菜→下锅→翻炒→装盘"，融合就是别切完菜先装进冰箱（写显存）再拿出来（读显存）下锅，而是切完直接下锅。省掉的"存取冰箱"就是省掉的显存读写——而显存带宽正是推理的头号瓶颈。所以接口粒度必须**粗到"一个可融合的完整动作"**，不能把可融合的步骤拆散。

**切太粗（整个 `forward` 抽象成一个 Backend 方法）：**
```python
# 另一个反面教材：粒度过粗
backend.forward(model, input)   # 整个前向就这一个接口
```
问题在哪？
- **无法逐个替换**。你这个暑假的推进方式是"这周先把 rmsnorm 换成 CUDA 版，attention 先不动，跑通对标了再换 attention"。如果整个 forward 是一坨，你想只换其中一个算子？做不到，只能整个重写，一崩全崩。
- **无法逐个对标**。你没法说"我的 CUDA rmsnorm 比 torch 快 3.2×"，因为时间全糊在一个大 forward 里，拆不出来。

### 2.2 刚刚好：这三个算子恰好是"三个战场"

`rmsnorm / attention / fused_ffn` 这个粒度之所以"刚刚好"，因为它同时满足三个条件：

1. **每一个都是"一个完整的、可独立替换的计算块"**——粗到能内部融合，细到能单独换。
2. **每一个都是 Transformer 里的计算大头**——值得你花力气去手写优化（80/20 法则，优化就该优化在耗时占比高的地方）。
3. **这三个恰好就是你暑假要逐个手写的三个战场**——接口粒度和你的学习路线完全对齐：

| 算子 | 它是什么 | 暑假哪周变成手写战场 |
|---|---|---|
| `rmsnorm` | 归一化（把每个向量缩放到合适的尺度，稳定训练/推理数值） | W2：第一个 CUDA kernel 练手（相对简单，适合入门） |
| `attention` | 注意力（让每个 token 看见并加权聚合其他 token 的信息） | W3-4：FlashAttention，暑假技术难度顶峰 |
| `fused_ffn` | 前馈网络（对每个 token 独立做一次"升维→非线性→降维"） | W4：fused QKV / fused FFN，融合技巧集大成 |

> 换句话说：**接口粒度不是随便切的，是照着"哪些东西我接下来要亲手换"来切的。** 一个东西如果你永远不会替换它的实现，就不该为它单独开一个 Backend 接口（那是过度设计）；一个东西如果你要反复替换、反复对标，它就必须是一个独立接口。这是判断"该不该抽象"的黄金标准——**为"预期会变化的部分"留接缝，不为"不变的部分"付抽象税。**

### 2.3 顺带认识这三个算子（W6/W8 已见过，这里补齐"为什么是它们"）

- **RMSNorm（Root Mean Square Normalization，均方根归一化）**
  为什么现代 LLM（Llama/Qwen）用它而不是你 W6 nanoGPT 里的 LayerNorm？因为它**砍掉了减均值和加偏置两步**，只保留"按均方根缩放"，更省算力、数值也够稳。公式后面代码里给。它是三个算子里最简单的，所以是你 W2 的 CUDA 练手首选。

- **Attention（注意力机制）**
  Transformer 的心脏。每个 token 生成一个 Query（我想找什么）、每个 token 有一个 Key（我是什么）和 Value（我携带的信息）。用 Query 和所有 Key 算相似度、softmax 成权重、再对 Value 加权求和。**它是推理里最耗时、最耗显存的部分**（因为要算 token 两两之间的关系，是平方级），所以 FlashAttention 才是暑假的重头戏。

- **Fused FFN（融合前馈网络）**
  Llama 用的是 **SwiGLU** 结构：`down( SiLU(gate(x)) * up(x) )`。三个矩阵乘 + 一个门控相乘。"fused"意味着这几步该融进尽量少的 kernel。它是"融合"这个技巧最典型的练习场。

---

## 第三部分：动手实现 —— Backend 基类 + TorchBackend

### 3.1 `engine/backend.py`（今天的核心代码）

```python
# engine/backend.py
# ─────────────────────────────────────────────────────────────
# 运行环境：Python 3.10+ / torch 2.x
#   - 有无 GPU 都能跑：CPU 上直接运行；GPU 上把张量 .cuda() 即可
#   - 无第三方依赖（只用 torch），Day 1 保持最小依赖
# 作用：定义"算子后端"的统一接口，让上层模型只认接口、不认实现。
# ─────────────────────────────────────────────────────────────
import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Backend(ABC):
    """算子后端基类：上层只认这三个方法，不关心底层是谁实现的。

    为什么用 abc.ABC + @abstractmethod，而不是像计划草稿里那样
    只写 `raise NotImplementedError`？
      - raise NotImplementedError：只有当你"真的调用"到那个没实现的
        方法时才报错，可能跑了半天才在深处炸出来（发现得太晚）。
      - ABC + abstractmethod：只要子类漏实现任何一个抽象方法，
        你在 `SomeBackend()` **实例化那一刻**就会报 TypeError。
        错误提前到"创建对象"这一步，暴露得越早越好。
    这就是"fail fast（尽早失败）"原则的工程化。
    """

    @abstractmethod
    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
        """RMSNorm 归一化。x: [..., d], weight: [d]。"""
        ...

    @abstractmethod
    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  *, causal: bool = True) -> torch.Tensor:
        """缩放点积注意力。q/k/v: [B, H, S, D]。

        注意：计划草稿里的签名是 attention(q, k, v, kv_cache)。
        但 KV Cache 是 W5-6 的系统层内容，Day 1 还没有 cache 的概念。
        接口本就是要演进的——今天先落地"能算对因果注意力"的最小版本，
        等 W5 再把 kv_cache 参数加进来。留个 TODO 标记这里会长东西。
        """
        ...

    @abstractmethod
    def fused_ffn(self, x: torch.Tensor,
                  w_gate: torch.Tensor, w_up: torch.Tensor,
                  w_down: torch.Tensor) -> torch.Tensor:
        """SwiGLU 前馈网络（对应草稿里的 w1/w2/w3）。
        w_gate=w1, w_up=w2, w_down=w3；每个 weight 形如 [out, in]。"""
        ...


class TorchBackend(Backend):
    """纯 PyTorch 实现：Day 1 的目标是"先能跑、结果对"，不追求快。

    这份实现有个特殊身份——它是【永远正确的 baseline（基准参照）】。
    以后你写的 TritonBackend / CudaBackend，正确性和速度都拿它当尺子量。
    所以这里每一行都要"宁可慢、务必对"，不要为了快在这里耍花招。
    """

    def rmsnorm(self, x, weight, eps=1e-6):
        # RMSNorm 数学：  y = x / sqrt(mean(x^2) + eps) * weight
        # 直觉：把向量按它自己的"均方根长度"缩放到统一尺度，再乘可学习的 weight。
        #
        # 关键工程细节：先转 float32 再算，最后转回原 dtype。
        #   为什么？推理常用 float16/bfloat16，x^2 在低精度下极易累加出误差，
        #   甚至上溢/下溢。Llama 官方实现就是在 float32 里做归一化的核心运算。
        #   这是"数值稳定性"的行业惯例——低精度存储，高精度算敏感步骤。
        in_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)      # 每个向量各算各的均方
        x_normed = x * torch.rsqrt(variance + eps)          # rsqrt = 1/sqrt，比先 sqrt 再除更快更稳
        return (x_normed.to(in_dtype)) * weight             # 缩放回原精度后再乘 weight

    def attention(self, q, k, v, *, causal=True):
        # q,k,v: [B, H, S, D]  (Batch, num_Heads, Seq_len, head_Dim)
        # 标准缩放点积注意力：softmax(QK^T / sqrt(D)) @ V
        d = q.size(-1)
        scale = 1.0 / math.sqrt(d)                          # 除以 sqrt(D) 防止点积过大导致 softmax 梯度消失
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B,H,S,S] 每个 token 对每个 token 的相似度

        if causal:
            # 因果掩码（causal mask）：生成任务里，第 i 个 token 只能看它自己和之前的，
            # 不能偷看未来。把"未来"位置的分数设成 -inf，softmax 后权重就是 0。
            S = q.size(-2)
            # diagonal=1 → 上三角(不含主对角线)为 True，正好是"未来"的位置
            future = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
            )
            scores = scores.masked_fill(future, float("-inf"))

        attn = torch.softmax(scores, dim=-1)                # 沿"被看的 token"维度归一化成权重
        return torch.matmul(attn, v)                        # 用权重对 V 加权求和 → [B,H,S,D]

    def fused_ffn(self, x, w_gate, w_up, w_down):
        # Llama 的 SwiGLU：down( SiLU(gate(x)) * up(x) )
        #   gate/up 把 x 从 d 维升到隐藏维 h，逐元素相乘做"门控"，down 再降回 d 维。
        # weight 形如 [out, in]，所以用 x @ w.T（等价于 nn.Linear 的 no-bias 前向）。
        #
        # "fused"在 TorchBackend 里其实没真融合——纯 PyTorch 会拆成好几个 kernel。
        #   这没关系：接口叫 fused_ffn 是为将来 CUDA/Triton 版本"真融合"占好位置。
        #   接口名描述"意图"，实现可以暂时达不到，等你 W4 来兑现。
        gate = F.silu(torch.matmul(x, w_gate.t()))          # SiLU(x·W_gate)：门控信号
        up = torch.matmul(x, w_up.t())                      # x·W_up：被门控的值
        return torch.matmul(gate * up, w_down.t())          # 逐元素门控后降维


# ── 冒烟测试：直接 `python engine/backend.py` 就能验证形状/能跑 ──
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    be = TorchBackend()

    # rmsnorm
    x = torch.randn(2, 8, 64, device=dev)
    w = torch.ones(64, device=dev)
    assert be.rmsnorm(x, w).shape == x.shape

    # attention
    q = k = v = torch.randn(2, 4, 16, 32, device=dev)   # [B,H,S,D]
    assert be.attention(q, k, v).shape == (2, 4, 16, 32)

    # fused_ffn
    d, h = 64, 172
    x2 = torch.randn(2, 8, d, device=dev)
    wg, wu = torch.randn(h, d, device=dev), torch.randn(h, d, device=dev)
    wd = torch.randn(d, h, device=dev)
    assert be.fused_ffn(x2, wg, wu, wd).shape == x2.shape

    print(f"[OK] TorchBackend 三个算子形状全部通过（device={dev}）")
```

**为什么这份代码值得逐行读，而不是复制了事：**
- `ABC + @abstractmethod` 而非 `raise NotImplementedError`：**fail fast**，漏实现在实例化就炸。
- rmsnorm 里 `float()` 再算：**数值稳定性行业惯例**，低精度存储、高精度算敏感步骤。
- attention 里 `masked_fill(-inf)`：**因果掩码**的标准实现，理解它 W3 学 FlashAttention 时"online softmax 怎么处理 mask"才不会懵。
- `fused_ffn` 名叫 fused 但实现没融合：**接口名描述意图、实现可后补**，这是抽象层的常态。

---

### 3.2 底层视角：真实框架里这一层长什么样？

你今天写的 `Backend` 抽象，在工业框架里都有对应物。看一眼，建立"我做的东西是真的"的信心：

**vLLM —— `_custom_ops` 层**
vLLM 里有一个模块专门收口所有底层算子调用，上层模型（如 `LlamaForCausalLM`）永远调它，不直接碰 CUDA：
```python
# vLLM 的真实用法（示意）：模型层这样调 rms_norm
from vllm import _custom_ops as ops
# ops.rms_norm 内部：若编译了 CUDA 扩展就走手写 kernel，否则回退 PyTorch。
# 上层模型代码完全不知道这次是 CUDA 还是 PyTorch 干的 —— 和你的 Backend 一模一样。
ops.rms_norm(out, x, weight, epsilon)
```
关键点：**上层调用点稳定（永远是 `ops.rms_norm`），底层实现可以有 CUDA 版和 PyTorch 回退版两条路。** 这就是你的 `TorchBackend` vs `CudaBackend`。

**TensorRT-LLM —— Plugin（插件）机制**
TensorRT 把网络编译成一张计算图，图里的某些节点可以用**插件**替换成自定义高性能实现。比如 `gptAttentionPlugin` 就是把整个注意力块换成一个高度优化的手写实现，而网络其余结构不变。
> **类比**：网络图是一条流水线，plugin 就是"把流水线上某个工位的机器换成更快的型号"，其他工位不动。这跟你"只把 attention 换成 CUDA 版、rmsnorm 先不动"是同一件事。

**结论**：从 PyTorch Dispatcher（C++）→ vLLM `_custom_ops`（Python）→ TensorRT-LLM plugin（图层）→ 你的 `Backend`（业务层），**同一个"调用/实现分离"的思想在每一层反复出现**。你不是在学一个孤立技巧，是在学一个贯穿整条推理技术栈的设计范式。

---

## 第四部分：测试基础设施（副线块 —— 今天就位，天天用）

### 4.1 为什么"第一天就建测试"，而不是"以后需要了再说"？

因为你的工作方式是**"反复替换实现 + 反复对标"**。测试不是负担，是你替换 kernel 时的**安全网**——没有它，你换完 CUDA rmsnorm 根本不敢确认"到底改对没有"。**先有网，才敢在高空换零件。** 这直接服务你 W8 立下的"三尺子验误差"纪律。

### 4.2 `engine/bench_utils.py`：三铁律计时 + 三尺子误差

这个文件从你 `week8_triton/benchmark_utils.py` 搬来并收口。核心是两组方法论：

```python
# engine/bench_utils.py
# 运行环境：Python 3.10+ / torch 2.x；GPU 计时需 CUDA，CPU 上自动降级
import time
import torch


# ══════════════ 三尺子：验"对不对" ══════════════
def measure_error(actual: torch.Tensor, ref: torch.Tensor) -> dict:
    """用三把尺子量"你的实现"和"参照实现"的差距。
    为什么要三把尺子而不是一把？因为单一指标会骗你：
      - 只看绝对误差：数值本身很大时，0.1 的误差可能无所谓，也可能是灾难，看不出。
      - 只看相对误差：数值接近 0 时，分母极小会把误差放大成天文数字，误报。
      - 只看 allclose：它只给 True/False，出了问题你不知道差多少、差在哪。
    三把一起看，才能既不漏报、又不误报、还能定位。
    """
    actual, ref = actual.float(), ref.float()
    abs_err = (actual - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-8)              # +eps 防止除以 0
    return {
        "max_abs": abs_err.max().item(),               # 尺1：最大绝对误差
        "max_rel": rel_err.max().item(),               # 尺2：最大相对误差
        "allclose": torch.allclose(actual, ref,        # 尺3：综合判定
                                   rtol=1e-3, atol=1e-5),
    }


# ══════════════ 三铁律：测"快不快" ══════════════
def benchmark(fn, *args, warmup: int = 10, iters: int = 100, **kwargs) -> float:
    """返回单次调用的中位数耗时（毫秒）。三条铁律缺一不可：

    铁律1 —— 预热（warmup）：先空跑几次再计时。
        GPU 有时钟爬升、cuDNN 有 autotune、首次调用有一次性开销，
        不预热测到的是"冷启动"时间，不是稳态性能。
    铁律2 —— 同步（synchronize）：GPU kernel 是异步下发的！
        你 Python 里调完函数它可能还没算完就返回了。不 synchronize，
        你测到的是"下发命令的时间"，不是"算完的时间"——能差几个数量级。
    铁律3 —— 多次取中位数：单次计时受系统抖动干扰大。
        取中位数（不是平均数）能抗住偶发的长尾毛刺。
    """
    use_cuda = torch.cuda.is_available() and any(
        isinstance(a, torch.Tensor) and a.is_cuda for a in args
    )

    def sync():
        if use_cuda:
            torch.cuda.synchronize()                   # 铁律2：等 GPU 真正算完

    for _ in range(warmup):                            # 铁律1：预热
        fn(*args, **kwargs)
    sync()

    times = []
    for _ in range(iters):                             # 铁律3：多次
        sync()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        sync()
        times.append((time.perf_counter() - t0) * 1e3) # 转毫秒

    times.sort()
    return times[len(times) // 2]                      # 取中位数，抗抖动
```

> **注意 `rtol`/`atol`（相对/绝对容差）怎么定**：`torch.allclose(a, b, rtol, atol)` 的判据是 `|a-b| ≤ atol + rtol*|b|`。
> - float32 参照：`rtol=1e-3, atol=1e-5` 是合理起点。
> - float16/bf16 参照：容差要**放宽**（低精度本身就有更大误差），常用 `rtol=1e-2` 量级。
> - **易错点**：拿 float16 结果去和 float32 参照用严格容差比，几乎必然 `allclose=False`——这不是你 kernel 错了，是你尺子用错了精度。

### 4.3 `tests/test_backend.py`：第一个单元测试

```python
# tests/test_backend.py
# 运行：pytest tests/test_backend.py -v      （需 pip install pytest）
# 作用：把 TorchBackend.rmsnorm 和 W8 的 Triton 参考实现对齐，allclose 通过。
import torch
import pytest

from engine.backend import TorchBackend


def rmsnorm_reference(x, weight, eps=1e-6):
    """W8 Day3 你写过的 Triton RMSNorm 的 PyTorch 参考实现。
    这里作为"金标准（golden reference）"——被测实现要和它对齐。"""
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)).type_as(weight) * weight


@pytest.mark.parametrize("shape", [(2, 8, 64), (1, 128, 512), (4, 1, 4096)])
def test_rmsnorm_matches_reference(shape):
    """数值 sanity：TorchBackend.rmsnorm 必须和参考实现一致。
    参数化多个 shape，覆盖不同 batch/seq/dim 组合，防止只在某个特殊尺寸下对。"""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(*shape, device=dev)
    w = torch.randn(shape[-1], device=dev)

    out = TorchBackend().rmsnorm(x, w)
    ref = rmsnorm_reference(x, w)

    # 用 allclose 判正确性；失败时打印最大误差，方便定位是"差一点"还是"全错"
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-6), \
        f"max_abs={ (out-ref).abs().max().item() }"


def test_rmsnorm_shape_preserved():
    """形状 sanity：归一化不该改变张量形状。"""
    be = TorchBackend()
    x = torch.randn(3, 10, 128)
    assert be.rmsnorm(x, torch.ones(128)).shape == x.shape
```

**测试要覆盖三种情形（贴合你笔记要求）：**
- **正常使用**：`test_rmsnorm_matches_reference`（多 shape 参数化）。
- **形状 sanity**：`test_rmsnorm_shape_preserved`（防止手滑把维度算错）。
- **常见错误 / 调试**：断言失败时打印 `max_abs`，让你一眼看出是"精度差一点"（容差问题）还是"逻辑全错"（差了好几个数量级）。这是把 4.2 的"三尺子"用在实战。

---

## 第五部分：起点确认 —— 跑通 nanoGPT 基线

### 5.1 为什么"改之前先确认它是好的"？—— 回归基线思想

把 W6 训好的 nanoGPT `model.py` 原样搬进 `engine/model.py`，**用你已有的权重，原样跑通一次文本生成**。这一步不写新逻辑，但极其重要。

> **回归基线（Regression Baseline）**：改造前先记录下"当前系统确实是好的"这个事实（能跑、输出正常）。以后任何一步改造出了问题，你都能靠"和基线对比"立刻判断：是**我这次改坏的**，还是**它本来就有问题**。

**类比**：你要给一辆能开的车换发动机。动手前先开一圈确认"现在车是好的"。换完发动机车打不着？因为你有"换之前能开"这个事实，你确定问题在发动机上，而不是"这车本来就是坏的、白折腾半天"。**没有这一圈，出了问题你会在错误的地方排查到天亮。**

这也是软件工程里 **regression test（回归测试）**一词的由来——"回归"指的就是"退回到比之前更差的状态"，基线就是判断有没有退步的那把尺子。

```python
# 快速跑通基线（示意，具体接口按你 W6 的 model.py 为准）
# 运行环境：与 W6 训练时一致的 torch 环境 + 你训好的 ckpt
import torch
from engine.model import GPT, GPTConfig   # 从 W6 原样搬来

ckpt = torch.load("ckpt.pt", map_location="cuda")
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.eval().cuda()

# 关键：记下这次生成的输出，作为"改造前的基线快照"存档。
# Day2 起你会把 LayerNorm 换 RMSNorm、加 RoPE、模型层改走 Backend 接口——
# 每次改完都回来跑同一个 prompt，对比输出有没有异常漂移。
ctx = torch.zeros((1, 1), dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=100)
print("BASELINE OUTPUT:", out[0].tolist()[:20], "...")
```

> **小提示**：把这次的输出（或它的 hash / 前若干 token）记进 `ARCHITECTURE.md` 或一个 `baseline.txt`。Day 2 改完 RMSNorm 后，生成结果**允许有变化**（毕竟换了归一化），但不该变成乱码——有基线快照，你才有"变化是否合理"的判断依据。

---

## 第六部分：`ARCHITECTURE.md` v0（把认知固化成文档）

今天写文档的意义：**逼自己用文字讲清设计，是检验"真懂还是自以为懂"的照妖镜。** 写不出来的地方，就是你没想透的地方。v0 要包含三块：

```markdown
# miniLLM-serve 架构文档 v0

## 1. 五层架构（对标真实推理框架的简化版）
（照暑假计划 §1.2 画那张五层图：模型层 → Kernel后端 → 显存/Cache → 调度层 → API层）
- 今天完成：② Kernel 后端抽象、① 模型层基线
- 后续长出：③ KV Cache（W5）、④ continuous batching（W6）、⑤ API（收口）

## 2. 为什么要 Backend 抽象（用我自己的话，不抄）
- 解决的问题：让"换一个更快的 kernel 实现"= 改一行配置，而不是动整个模型层。
- 接口为什么切在 rmsnorm / attention / fused_ffn 三个粒度：
    - 太细（每个 add 都抽象）→ 抽象税 + 丧失融合机会；
    - 太粗（整个 forward）→ 无法逐个替换、无法逐个对标；
    - 这三个恰是计算大头，也恰是我暑假要逐个手写的三个战场。
- 和工业界的对应：vLLM 的 _custom_ops、TensorRT-LLM 的 plugin、
    甚至 PyTorch 自己的 Dispatcher，都是同一个"调用/实现分离"思想。

## 3. 八周进化路线表
| 阶段 | 引擎能力里程碑 | 后端状态 |
|------|--------------|---------|
| W0 发射周 | 模型跑通单请求生成；Backend 抽象就位 | 纯 PyTorch |
| W3-4 | 核心算子换手写；FlashAttention 上线 | Triton + CUDA |
| W5-6 | Paged KV Cache + continuous batching | 系统层成型 |
| W7-8 | 真模型/AMK 上对标出可证明的优化；收口成作品 | 全栈可对标 |
```

---

## 第七部分：整理块 —— GitHub Profile README 骨架（低负荷任务，放最后）

先搭骨架、后填肉。按能力成长线的四段写小标题 + 每段一句话：

```markdown
# 你好，我是 XXX 👋
> 双非大一 · 方向：AI Infra / 大模型推理优化

### 🧱 能力成长线（从手搓到系统）
- **手写神经网络** — 从 0 实现 MLP/反向传播，理解梯度怎么流。
- **Transformer** — 手写 nanoGPT，训练 + 生成全流程跑通。
- **Kernel 优化** — Triton 手写 RMSNorm/Attention 等 kernel，端到端对标。
- **推理系统（建设中 🚧）** — miniLLM-serve：可切换 kernel 后端的 mini 推理引擎。
```

> 为什么整理块放最后、还是"最先砍"的：它是**低认知负荷**任务，不需要连续深度思考，天然适合晚上疲劳时段。深水活（Backend）放精力最好的时段，浅水活（README）放疲劳时段——按精力曲线分配任务，本身就是科学规划。累了就先砍它，欠账滚到 Day 7 统一补。

---

## 第八部分：常见陷阱与调试技巧（踩过的坑，提前排雷）

| 陷阱 | 现象 | 根因 & 解法 |
|---|---|---|
| **GPU 计时不 synchronize** | 手写 kernel 测出来"比 PyTorch 快 100×" | kernel 异步下发，你只测了"下发时间"。**必须 `torch.cuda.synchronize()` 后再停表**（见 4.2 铁律2）。 |
| **allclose 精度用错** | float16 实现和 float32 参照永远 `False` | 不是 kernel 错，是尺子错。低精度参照要放宽 `rtol/atol`（见 4.2）。 |
| **RMSNorm 没转 float32** | 训练/推理数值不稳、偶发 NaN | 低精度算 `x^2` 累加误差大。**敏感运算转 float32**（见 3.1 rmsnorm 注释）。 |
| **weight 转置方向搞反** | fused_ffn / attention 形状对不上或结果全错 | `nn.Linear` 权重是 `[out,in]`，前向是 `x @ w.T`。搬权重时确认方向。 |
| **不预热就计时** | 第一次测的时间明显偏大且不稳定 | GPU 时钟爬升 + autotune 一次性开销。**先 warmup 再测**（见 4.2 铁律1）。 |
| **过早抽象（over-engineering）** | 给一个永远不换实现的东西也开 Backend 接口 | 只为"预期会变化的部分"留接缝。不变的东西直接写死，别付抽象税。 |
| **忘了先跑基线就改造** | Day2 改完发现输出是乱码，不知是改坏的还是本来坏 | **改造前先跑通并存档基线**（见第五部分）。 |

---

## ✅ 今日完成标准自测（能脱稿讲清 = 真懂）

请合上笔记，用自己的话回答：

1. **后端抽象解决什么问题？** —— 让"换一个更快的 kernel 实现"变成改一行配置，模型层不动；并且能一键切回 baseline 做正确性/性能对标。
2. **接口为什么切在 rmsnorm / attention / fused_ffn 这三个粒度？** —— 太细付抽象税还丧失融合机会；太粗无法逐个替换和对标；这三个恰是计算大头，也恰是我暑假要逐个手写的三个战场。为"会变化的部分"留接缝，不为"不变的部分"付抽象税。
3. **这个思想在工业界哪里出现过？** —— PyTorch 的 Dispatcher（C++）、vLLM 的 `_custom_ops`、TensorRT-LLM 的 plugin，全是同一个"调用/实现分离"。

**今日交付物清单：**
- [ ] `miniLLM-serve` 仓库骨架建好并 push（第一天就有 commit 记录）
- [ ] `engine/backend.py`：`Backend` 基类 + `TorchBackend`（三个算子）
- [ ] `engine/bench_utils.py`：三铁律计时 + 三尺子误差
- [ ] `engine/model.py`：nanoGPT 原样搬入，**基线跑通并存档输出**
- [ ] `tests/test_backend.py`：rmsnorm 对齐参考实现，`pytest` 通过
- [ ] `ARCHITECTURE.md` v0：五层架构 + 设计理由 + 8 周路线表
- [ ] GitHub profile README 骨架（四段能力成长线）
- [ ] 首次 commit（建议信息：`feat: scaffold miniLLM-serve + Backend abstraction (Day1)`）

---

> **一句话收尾**：今天你没写什么"炫技"的代码，但你亲手立起了整个引擎的"脊椎"——一个让后续 7 周所有优化都能"插进来、比得了、不崩塌"的接口。**好的系统设计，价值不在今天，在未来每一次替换都省下的时间。**
