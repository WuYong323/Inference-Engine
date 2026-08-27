# 阶段一 · W2 Day 4 学习笔记 —— decode 战场：补 KV Cache 的债 + 第一次融合

> **对应规划**：`阶段一W2_CUDA进阶_手写算子接进引擎_逐日详细规划.md` → W2 Day 4（8/21 周五）
> **今日目标**：回答"为什么 decode 阶段的 RMSNorm 是一个完全不同的问题"；补上引擎缺失的 KV Cache（债务）；完成第一次算子融合 `fused_add_rmsnorm`。
> **今日定位**：还债日 + 划界日。今天的三张数据表决定你后面四周把力气花在哪。
> **前置**：Day 1–3 的单算子功力（桥/occupancy/四板斧）；`engine/model.py` 的 `generate()` 目前每生成一个 token 重算整个前缀——**引擎里没有 decode 阶段，今天补**。

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `csrc/fused_add_rmsnorm.cu` | 融合 kernel（残差加法 + RMSNorm） | 复制进 `推理引擎/csrc/`，接入见文件头注释 |
| `engine/kv_cache.py` | 朴素 KV Cache 类 | 复制进 `推理引擎/engine/` |
| `engine/model_snippets.py` | `CausalSelfAttention.forward` + `generate()` 改造 | 合并进 `推理引擎/engine/model.py` |
| `bench/profile_rmsnorm.py` | Amdahl 检查：torch.profiler 占比 + 天花板 | 放 `推理引擎/bench/` |
| `bench/bench_kvcache.py` | 三级验证：等价性 / 复杂度曲线 / TTFT·TPOT·tok/s | 放 `推理引擎/bench/` |
| `research/findings_day4_template.md` | 今日数据记录模板 | 复制进 `推理引擎/research/` |

---

## 0. 今天的问题与全景图

### 0.1 今天的两个问题

1. **为什么 decode 阶段的 RMSNorm 是一个完全不同的问题？**——它牵扯到 GPU 并行度的本质，也是你论文选题的核心论据。
2. **引擎为什么没有 decode 阶段？**——因为你的 `generate()` 是这样的：

```python
idx_cond = idx[:, -self.block_size:]
logits, _ = self(idx_cond)          # ← 每生成一个 token，把整个前缀重算一遍
```

每一步都在做 prefill，decode 从未存在过。这个债今天必须还：没有 KV Cache 就没有 decode 实验、tok/s 基线是错的、W5 的 PagedAttention 无从谈起。**债务越晚还越贵。**

### 0.2 今天的完整闭环

```
① 算账：prefill/decode 对照表（§1，W2 最重要的一张表）
② 划界：Amdahl 检查 → RMSNorm 端到端占比 → 优化天花板（§2）
③ 动手：fused_add_rmsnorm 融合（§3）+ KV Cache 补债（§4）
④ 验证：三级验证（等价性/复杂度/性能基线）→ 真 tok/s（§4.4）
⑤ 研究：实测 launch 开销 + proposal 边界声明（§5）
```

### 0.3 时间盒导航

| 时间块 | 内容 | 章节 |
|---|---|---|
| 【学】1h | decode 并行度崩塌：把账算清楚 | §1 |
| 【学】1h | Amdahl 检查：先测占比 | §2 |
| 【学】1.5h | 第一次融合 fused_add_rmsnorm | §3 |
| 【造】2.5h | KV Cache 补债 + 三级验证 | §4 |
| 【研】1.5h | nsys 实测 + proposal 边界 | §5 |

---

## 1. decode 的并行度崩塌：把账算清楚（1h）

### 1.1 先定义两个阶段（今天的一切从这里出发）

LLM 是**自回归（autoregressive）**的：一次只能预测下一个 token，预测出的 token 拼进输入，再预测下一个。这个循环里有两个截然不同的阶段：

- **prefill（预填充）**：用户把整个 prompt 一次喂进来。模型**同时**处理 prompt 的所有 token（比如 2048 个），并行度 = batch × prompt 长度。这是"批量处理"。
- **decode（解码）**：之后每生成一个新 token，模型只处理**这一个** token（但注意力要回看全部历史）。并行度 = batch × 1。这是"逐个处理"。

> **类比**：prefill 是流水线批量生产——一单 2048 件一起上线，机器满载；decode 是定制单件——每次只做 1 件，机器大部分时间在等活。**问题不在"单件难做"，而在"机器闲着"。**

### 1.2 那张表：W2 最重要的一张表（也是你选题的核心论据）

| | prefill (B=8, T=2048) | decode (B=1, T=1) | decode (B=32, T=1) |
|---|---|---|---|
| RMSNorm 行数 | 16384 | **1** | 32 |
| grid 大小 | 16384 blocks | **1 block** | 32 blocks |
| 5060 用上几个 SM | 26 / 26 | **1 / 26** | 26 / 26（但每 SM 1 个 block） |
| H100 用上几个 SM | 132 / 132 | **1 / 132** | 32 / 132 |
| 访存量（bf16） | 128 MB | 16 KB | 512 KB |
| 理论时间 @3350GB/s | 38 µs | **4.9 ns** | 153 ns |
| **kernel launch 开销** | 3–5 µs（占 10%） | **3–5 µs（是有效计算的 ~1000 倍）** | 3–5 µs（~25 倍） |

先亲手验算三个格子，确认这张表不是"抄来的"：

```
访存量 = rows × H × 2 字节（读+写，bf16）
  prefill: 16384 × 4096 × 2 = 128 MB   → 128e6 / 3350e9 = 38.2 µs  ✓
  decode:      1 × 4096 × 2 = 16 KB    →  16e3 / 3350e9 = 4.8 ns   ✓（表里写 4.9）
  B=32:       32 × 4096 × 2 = 512 KB   → 512e3 / 3350e9 = 153 ns   ✓
```

**launch 开销（kernel launch overhead）是什么**：每次 `cudaLaunchKernel`（Day 1 §1.4 讲过它只是"排队"）在 CPU 侧都要花 3–5 µs 做参数打包、校验、向 GPU 提交。**这个成本和 kernel 里要算多久毫无关系**——哪怕 kernel 只干 5 纳秒的活，发射它也要 3–5 µs。

### 1.3 这张表的三个读法（每个都是一条结论）

1. **单请求 decode 时，99.2% 的硬件在闲置**（H100：132 个 SM 只用了 1 个），且**launch 开销是有效计算的 ~1000 倍**——你把 RMSNorm kernel 优化到无穷快，端到端一点变化都没有。**单算子优化在这里有硬天花板：0。**
2. **这就是巨核（megakernel）存在的全部理由**：把 decode 路径上几十个小算子融进**一个** kernel，只付一次 launch。这是 AMK（师兄的论文线）的领域。
3. **你的方向是另一半**：既然单算子优化在单请求 decode 有天花板，**"配置选择"的价值就转移到 prefill 和 batch decode**——那里 grid 够大、带宽是真瓶颈（Day 2/3 的 %peak 世界）、配置选对选错可以差 2–3 倍。**你的方法要在那个区间证明价值。**

> **⚠️ 这条推论必须在今天想清楚，否则你会做错方向**：不要试图在 B=1 的 decode 上优化单个 RMSNorm——那是巨核的地盘（师兄的）。**你的战场是 prefill 和 batch decode 的配置选择。** 今天的数据帮你把边界划清。

---

## 2. Amdahl 检查：先测占比，再决定优化什么（1h）

### 2.1 阿姆达尔定律（Amdahl's Law）：优化收益的上限公式

**阿姆达尔定律**：如果某个环节占端到端时间的比例是 p，把它加速 s 倍，整体加速比是：

```
Speedup = 1 / ( (1 - p) + p / s )
```

两个极端帮你建立直觉：p=100% 时 speedup = s（优化全部时间）；p→0 时 speedup→1（优化空气）。**算一笔今天的账**：假设 RMSNorm 占 5%，加速 2 倍：

```
Speedup = 1 / (0.95 + 0.05/2) = 1 / 0.975 ≈ 1.026  → 端到端只快 2.6%
```

> **类比**：把机场摆渡车的速度翻倍——摆渡只占登机总时间的 5%，全程时间几乎不变。**优化一个占比 5% 的环节，上限就写在阿姆达尔定律里，不靠感觉。**

### 2.2 用 torch.profiler 把占比测出来（配套 `bench/profile_rmsnorm.py`）

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    model.generate(ctx, max_new_tokens=64)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
```

读表的两个关键概念：

- **`cuda_time_total` vs `self_cuda_time_total`（自耗时）**：前者**包含**子算子的时间（总耗时），后者只算**自己**、刨除子算子（自耗时）。占比计算要用 **self 时间**——否则父子算子重复计数，占比会超过 100%。新版本 torch 里这两个字段也叫 `device_time_total` / `self_device_time_total`（配套脚本里做了兼容）。
- **`record_shapes=True`**：记录每个算子的张量形状——同一个 `aten::mm` 在 prefill（大矩阵）和 decode（小矩阵）里是两个世界，形状能帮你分辨。
- profiler 本身有开销（每次调用加几 µs），但**相对占比不受影响**——我们要的是比例不是绝对值。

配套脚本最后直接输出你要的三个数：

```python
rms = sum(e.self_cuda_time_total for e in events if "rmsnorm" in e.key)
total = sum(e.self_cuda_time_total for e in events)
share = rms / total
print(f"RMSNorm 自耗时占比: {share*100:.1f}%")
print(f"把它优化到 0 的端到端天花板: {1/(1-share):.2f}x")
print(f"把它加速 2 倍的实际收益: {1/((1-share)+share/2):.3f}x")
```

### 2.3 心理准备：这是今天最重要的工业教训

**RMSNorm 大概率占 3–8%。** 这意味着你把它加速 2 倍，端到端只快 1.5–4%——**很可能淹没在测量噪声里**。

这不是失败。这是**"优化前必须先测占比（measure before optimize）"**这条工业铁律的第一次系统级演练：

- 你 W1 Day4 已经撞过一次（"教科书说该快 30 倍、实测只快 1.02 倍"）——那是在**单算子**层面：算术强度太低的 kernel，计算优化被内存时间盖住；
- 今天是它在**系统**层面的放大版：单算子占比太低的算子，kernel 优化被其他 95% 的时间盖住。

**把占比数字写进笔记**——它会防止你在后面七周里把力气花在错的地方。同时注意两把尺子的分工：Day 2 的 **%peak** 是"这个 kernel 离硬件极限多远"（单算子视角）；今天的 **Amdahl** 是"这个 kernel 离端到端瓶颈多远"（系统视角）。**两把尺子都测，才叫懂性能。**

---

## 3. 第一次融合：`fused_add_rmsnorm`（1.5h）

### 3.1 为什么要融合：把访存账摊开

真实推理里，RMSNorm 从不单独出现——transformer 的每个残差块都是这个结构：

```python
residual = residual + attn_out        # kernel 1: 读 2 份（residual, attn_out）、写 1 份（residual）
h = rmsnorm(residual)                 # kernel 2: 读 1 份（residual）、写 1 份（h）
```

拆开的访存账：**5 次**（读 2 + 写 1 + 读 1 + 写 1），2 次 launch。

融合成一个 kernel：`h = rmsnorm(residual + attn_out)`，同时把新残差写回 `residual`（**残差流必须保留**——下一个残差块还要用）：

```
融合：读 2 份（residual, attn_out）、写 2 份（residual, h）= 4 次访存
→ 省 20% 访存，launch 从 2 次降到 1 次
```

> **类比**：去两趟超市（第一趟买菜、第二趟买油）改成一趟买齐——路上来回的时间（launch）和超市门口的搬运量（访存）都省了。

### 3.2 kernel 完整代码（配套 `csrc/fused_add_rmsnorm.cu`）

```cuda
template <int BLOCK>
__global__ void fused_add_rmsnorm_kernel(float* __restrict__ x,          // residual：in-place 写回
                                         const float* __restrict__ a,    // attn_out：只读
                                         const float* __restrict__ w,
                                         float* __restrict__ y,          // 输出 h
                                         int H, float eps) {
    const int row = blockIdx.x;
    float*       xr = x + (size_t)row * H;
    const float* ar = a + (size_t)row * H;
    float*       yr = y + (size_t)row * H;

    // 第一遍：残差和的平方归约（和 Day1 rmsnorm 同一副骨架，多一个加数）
    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += BLOCK) {
        const float s = xr[i] + ar[i];
        acc = fmaf(s, s, acc);
    }
    acc = block_reduce_sum<BLOCK>(acc);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(acc / H + eps);
    __syncthreads();
    const float inv = inv_rms;

    // 第二遍：写回残差流 + 输出 h
    for (int i = threadIdx.x; i < H; i += BLOCK) {
        const float s = xr[i] + ar[i];
        xr[i] = s;                              // ★ in-place 写回残差流（下一层要用新 residual）
        yr[i] = s * inv * w[i];
    }
}
```

逐行要点：

- **与 Day 1 rmsnorm 的关系**：同一副骨架（grid-stride + 两级归约 + shared 广播），只在两处加了残差项——第一遍的 `xr[i] + ar[i]`、第二遍多一次写回 `xr[i] = s`。**"融合"在 kernel 层面就是这么朴素：把两个 kernel 的两遍循环，重排成一遍里的两遍。**
- **`x` 是非 const 指针**：kernel 会写回 x——x 就是 Python 侧的 `residual` 张量，**in-place（就地）修改**。这是工业标准语义（vLLM 的 `fused_add_rms_norm` 就是 input 就地更新 + 输出新张量）。
- **为什么"读 2 写 2"而不是"读 2 写 1"**：`residual` 的新值必须写回（残差流是持久状态），`h` 是新的输出——一个都不能省。
- **两个 in-place 纪律**（工业红线）：① in-place 修改绕过了 PyTorch 的版本计数/自动求导追踪——推理引擎没问题，训练/autograd 场景是危险的，代码里要注释声明；② 融合 kernel 发射在 torch 当前流上（Day 1 纪律），后续读 `residual` 的算子自动排在它后面，顺序天然正确。

wrapper 层（完整版在配套文件里）：校验两输入同形状同 dtype、发射、`C10_CUDA_KERNEL_LAUNCH_CHECK`——与 Day 1 同一套夹具模板。接入方式：`kernels.py` 的 sources 加本文件、`bindings.cpp` 加一行 `m.def("fused_add_rmsnorm", ...)`。

### 3.3 融合的边界纪律（从今天起要有意识）

**这是标准的算子级融合**（vLLM 里就叫 `fused_add_rms_norm`，见 [vllm/csrc/layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu)），**不是巨核**。做这个不会撞车。但记住这条线：

> **融合的深度一旦超过"相邻 2–3 个算子"，就进入师兄（AMK 巨核）的地盘了，停手。**

**工业对照一句话**：融合收益的极致是 FlashAttention——把整个 attention 从"反复读写 O(n²) 中间矩阵"变成"分块在片上算完"，访存量下降一个数量级。今天这个 -20% 的融合是同一思想的入门练习。

---

## 4. 【造】KV Cache 补债（2.5h，今天的硬任务）

### 4.1 KV Cache 是什么：先讲清"为什么可以缓存"

**问题**：自回归生成第 t 个 token 时，第 1..t-1 个 token 的 hidden state 已经算过了，但你的 `generate()` 每步把整个前缀重算一遍——生成 n 个 token 要跑 O(n²) 次前向。

**KV Cache（键值缓存）**：把每层算过的 **K 和 V 存下来**。生成新 token 时，只算这一个 token 的 Q/K/V，然后 attention 用 `[缓存的 K/V; 新 token 的 K/V]`——历史计算永不重复。

**为什么缓存 K、V 而不是 Q？** 这是因果注意力（causal attention）的直接推论：**因果注意力里，token i 只能看到位置 ≤ i 的信息**——所以：

- **Q 只被使用一次**：token i 的 Q 只参与"token i 看历史"这一次计算，用完即弃 → 缓存它是浪费；
- **K/V 被反复使用**：token i 的 K/V 会被之后**所有** token（i+1, i+2, …）的注意力反复查询 → 缓存收益随序列长度增长。

> **类比**：图书馆里，每个读者的"借阅申请单"（Q）用一次就作废，但**书**（K/V）会被无数后来者反复借——当然是把书留在架上，而不是每来一个读者都重新印一遍全馆藏书。

**显存账（一个算例，理解工业压力的来源）**：Llama-8B 量级（32 层 × 32 头 × head_dim 128，bf16）：

```
每 token 每层：K+V = 2 × 32 头 × 128 维 × 2 B = 16 KB
每 token 全模型：16 KB × 32 层 = 512 KB
2048 token 序列：≈ 1 GB；batch=32 并发：≈ 32 GB —— 显存直接吃穿
```

这就是 W5 PagedAttention（[vLLM 论文](https://arxiv.org/abs/2309.06180)）存在的理由——cache 的碎片化浪费高达 60–80%，需要分页管理。今天先做**朴素版**（正确性优先），W5 再升级。

### 4.2 ★ 今天最经典的 bug：KV Cache 的 RoPE 位置偏移

**RoPE（旋转位置编码，Rotary Position Embedding）**：一种把"位置"编码进 Q/K 向量的方法——按**绝对位置** m 的角度，把向量在二维子空间里旋转；两个位置的相对角度只依赖 m−n，注意力自然获得相对距离信息。实现上通常是一个预计算的旋转表 `freqs_cis`，按位置索引：

```python
q, k = apply_rope(q, k, freqs_cis[start_pos:start_pos + T])   # ★ 按绝对位置切片
```

**bug 长什么样**：decode 时每步只处理 1 个新 token，它的**绝对位置是 `start_pos`**（比如第 100 个）。如果你写成 `freqs_cis[:T]`（即位置 0..T-1），模型会以为**每个新 token 都是句子的第一个词**。

**症状**：开启 cache 后，生成质量立刻崩坏（重复、乱码、与无 cache 输出完全不同）；而"有 cache 和无 cache 输出逐个 token 相同"恰恰是三级验证的第一步——**这个 bug 会在验证第一步就被抓出来，这就是为什么要做三级验证**。

**为什么缓存"旋转后的 K"**：RoPE 的旋转只依赖 token 自己的绝对位置——K 被旋转过一次后，就永远有效，可以直接缓存。如果缓存旋转前的 K，每步都要重新旋转整个历史（O(n) 每次），缓存就失去意义。这正是 W0 Day2 笔记里"RoPE 对推理友好"的那一点，今天兑现。

### 4.3 代码改造（配套 `engine/model_snippets.py` 与 `engine/kv_cache.py`）

```python
# CausalSelfAttention.forward 的关键改动
def forward(self, x, freqs_cis, kv_cache=None, start_pos=0):
    B, T, C = x.shape
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T, self.n_head, C // self.n_head)
    k = k.view(B, T, self.n_head, C // self.n_head)
    v = v.view(B, T, self.n_head, C // self.n_head)

    # ★ RoPE 用绝对位置 start_pos..start_pos+T，不是 0..T（§4.2 的 bug）
    q, k = apply_rope(q, k, freqs_cis[start_pos:start_pos + T])

    q, k, v = (t.transpose(1, 2) for t in (q, k, v))    # (B, heads, T, head_dim)

    if kv_cache is not None:
        k, v = kv_cache.update(self.layer_idx, k, v, start_pos)   # 写槽位 + 取全部历史
        # ★ 缓存的是「旋转后」的 K

    # ★ decode 时 T=1：唯一 query 可见全部历史 → 不需要 causal mask
    y = self.backend.attention(q, k, v, causal=(T > 1))
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.c_proj(y)
    return y
```

配套的朴素 `KVCache` 类（预分配一块 `(2, layers, B, heads, max_seq, head_dim)` 大张量，按位置写入——写槽位是 O(1)）：

```python
def update(self, layer_idx, k, v, start_pos):
    T = k.size(2)
    self.buf[0, layer_idx, :, :, start_pos:start_pos + T, :] = k   # 旋转后的 K
    self.buf[1, layer_idx, :, :, start_pos:start_pos + T, :] = v
    return (self.buf[0, layer_idx, :, :, :start_pos + T, :],      # 视图切片，零拷贝
            self.buf[1, layer_idx, :, :, :start_pos + T, :])
```

`generate()` 的改造要点（完整版在配套文件）：prefill（第一步）喂整个 prompt、`start_pos=0`；之后每步只喂 `idx[:, -1:]`、`start_pos += 1`；贪心解码用 `argmax`（temperature=0 时唯一随机源被关闭——这是三级验证能"逐 token 相同"的前提）。

### 4.4 三级验证（延续你的 SOP，一级都不能省）

**① 等价性**：同一个 prompt、固定 seed、`temperature=0`（贪心），**有 cache 和无 cache 的输出 token 序列必须逐个相同**。这是最强的正确性证据——RoPE 位置错、mask 错、槽位写错，任何一个都会在这里现形。补充技巧：若极小概率因浮点路径不同导致某个 token 分歧，先把两边的 logits 用 `allclose` 对比定位分歧层。

**② 复杂度**：生成 512 token，记录每步耗时，画两条曲线（配套 `bench/bench_kvcache.py`）：

- **无 cache**：每步重算整个前缀 → 第 t 步耗时 O(t²)（前缀越长，注意力的平方项越重）→ 曲线**凸着往上冲**；
- **有 cache**：每步只算 1 个 token 的前向 + 对 t 个历史 K/V 的注意力 → 第 t 步 O(t)（注意力随历史线性增长）→ 曲线**基本平坦、缓坡**。
- 诚实说明：有 cache 的曲线不是绝对平——注意力成本随历史线性增长，只是相对无 cache 的二次增长，肉眼接近平坦。**两曲线一对比，"债"有多贵一目了然。**

**③ 性能**：三个指标 + 落盘（`logs/baseline_kvcache.json`）：

- **TTFT（Time To First Token，首 token 延迟）**：从请求到第一个 token 的时间 = prefill 时间 + 第一次采样；
- **TPOT（Time Per Output Token，每 token 时间）**：之后每个 token 的平均耗时 = decode 单步时间；
- **tok/s**：1 / TPOT。
- **这是暑假后半程所有优化的分母**——以后每个优化都要和这个 JSON 里的数比。

### 4.5 工业对照

vLLM 的 PagedAttention（W5）、连续批处理（W6）全部建立在 KV Cache 之上——今天这个朴素版是它们的地基。注意 vLLM 的 `fused_add_rms_norm` 正是今天 §3 那个 kernel 的名字：**你今天的两个动作，恰好是 vLLM 两条主线的微缩版**。

---

## 5. 【研】补全实测数字 + proposal 边界（1.5h）

### 5.1 用 nsys 量真实的 launch 开销（不要用估计值）

规划里的 3–5 µs 是估计值，论文里不能出现估计值。**Nsight Systems（nsys）**是系统级 profiler（ncu 是 kernel 级，它看"一个 kernel 内部"；nsys 看"kernel 之间的时间线"）——launch 开销正好在它的视野里：

```powershell
nsys profile -o day4_decode -- python bench/bench_kvcache.py
nsys stats -r cuda_gpu_kern_sum day4_decode.nsys-rep
# launch 开销测法：在 Nsight Systems GUI 的 timeline 里，
# 量两个相邻 kernel 之间的 gap —— 那就是 CPU 排队/发射的成本。
#（Windows 本机 GUI 可用；或按规划 Day 5 攒到 H100 上统一做）
```

把实测数字填进 §1.2 那张表的"launch 开销"行——**表里的每一个估计值换成实测值后，这张表就升级成论文的 Table 1。**

### 5.2 更新 proposal_v0.md：写下你的边界

```markdown
## 方法的适用区间（Day 4 更新）
我的方法（单算子 launch 配置的解析预测）适用区间是 **prefill 和 batch decode**：
那里 grid 足够大、带宽是真实瓶颈、配置选择决定 2–3 倍的性能差异。
**单请求 decode 不在此列**：launch 开销是有效计算的 ~1000 倍、99% 硬件闲置，
单算子优化天花板为 0 —— 那是巨核（多算子融合）的地盘，归 AMK 线管。
两者正交：巨核内部也要选配置，我的模型未来可以喂给它。
```

**为什么这么写**：**一个方法知道自己的边界在哪，比声称自己万能可信得多**——审稿人最吃这套。这四句话同时完成了：选题防御（不撞车）、定位互补（可合作）、实验聚焦（prefill/batch decode 才是你的数据主场）。

---

## 6. 常见错误与调试速查表（Day 4 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| 开 cache 后输出崩坏（重复/乱码） | **RoPE 位置偏移**：decode 用了 0..T 而不是 start_pos.. | `freqs_cis[start_pos:start_pos+T]`；等价性验证第一步就会抓到 |
| 有 cache 和无 cache 输出不同 | RoPE 错 / causal mask 错 / 槽位写错 | 二分定位：先对 logits allclose，逐层比对 |
| decode 时输出看不到历史 | `causal=(T>1)` 写成了恒 True | T=1 时必须关 mask |
| 每步耗时"平坦"但整体没变快 | cache 建了但没走（逻辑 bug：if 分支恒 False） | 打日志确认 kv_cache is not None；看复杂度曲线 |
| profiler 占比加起来超过 100% | 用了 cuda_time_total（含子算子）重复计数 | 占比用 self 时间（`self_cuda_time_total`/`self_device_time_total`） |
| 计时测出 0.01ms 的假数据 | 没同步就计时（Day 1 的异步坑） | 每步前后 `torch.cuda.synchronize()` |
| fused kernel 后下一层读到的 residual 是旧值 | 忘写回 `xr[i] = s` / 没在 torch 当前流发射 | 检查 kernel 第二遍与 stream 参数 |
| fused kernel 输入形状报错 | x 与 a 形状不一致 | wrapper 里 `TORCH_CHECK(x.sizes() == a.sizes())` |
| nsys 打不开/无权限 | Windows 管理员/开发者模式 | 管理员运行；或攒到 H100（Day 5） |
| 贪心对比偶尔一个 token 不同 | 浮点路径差异放大到 argmax | 先对 logits allclose；确认温度=0、seed 固定 |

---

## 7. 完成标准自测（三道题，先默写再对答案）

1. **"KV Cache 的 RoPE 位置偏移"这个 bug 的症状和成因？**
   *答案要点*：成因——decode 每步的新 token 绝对位置是 start_pos，RoPE 旋转角度必须按绝对位置取（`freqs_cis[start_pos:start_pos+T]`）；写成 `freqs_cis[:T]` 会让模型以为每个新 token 都是位置 0（句首）。症状——开 cache 后输出质量立即崩坏（重复/乱码），与无 cache 输出逐 token 分歧；等价性验证（贪心 + 固定 seed 逐 token 对比）第一步就能抓到。缓存"旋转后的 K"是因为旋转只依赖 K 自己的绝对位置，旋转一次永续有效。
2. **RMSNorm 在你引擎端到端里的占比？"优化到 0"的天花板是多少？**
   *答案要点*：用 torch.profiler（CUDA activity）测 self 时间占比，预期 3–8%；天花板 = 1/(1−p)（阿姆达尔定律），p=5% 时上限 1.053×——加速 2 倍也只有 2.6%。教训：优化前必须测占比；单算子 %peak（Day 2 的尺子）与系统 Amdahl（今天的尺子）是两把尺子，都要量。
3. **为什么单请求 decode 不是你的战场？**
   *答案要点*：grid=1 → 1/26（5060）或 1/132（H100）的 SM 利用率；launch 开销 3–5 µs 是有效计算（~5 ns）的约 1000 倍——单算子优化天花板为 0。那是巨核（多算子融合省 launch）的地盘 = AMK 师兄的线。你的方法价值在 prefill 和 batch decode（grid 大、带宽是真瓶颈、配置选错差 2–3 倍）。边界写进 proposal：知道自己不做什么，比声称万能可信。

---

## 8. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] 引擎有真 KV Cache：`engine/kv_cache.py` + `model.py` 改造，**三级验证通过**
- [ ] 真 tok/s 基线：`logs/baseline_kvcache.json`（TTFT / TPOT / tok/s）
- [ ] `fused_add_rmsnorm` kernel + 访存量账（5→4 次，-20%，launch 2→1）
- [ ] prefill/decode 对照表（launch 开销换成 nsys 实测值）
- [ ] `proposal_v0.md` 写入适用区间边界

**明日预告（Day 5）**：上 H100——前四天所有实验打包跑一遍（四方对标：eager / torch.compile / Triton / 你的 CUDA），产出跨架构对照。**上机前把脚本写完、本机测试跑通——集群时间比本机金贵。**

---

## 附 A：术语速查表（Day 4）

| 名词 | 一句话解释 |
|---|---|
| prefill（预填充） | 一次性并行处理整个 prompt 的阶段，并行度 = batch × prompt 长度 |
| decode（解码） | 每步只生成 1 个 token 的自回归阶段，并行度 = batch × 1 |
| 自回归（autoregressive） | 逐个 token 生成、生成的 token 拼回输入的循环 |
| KV Cache（键值缓存） | 缓存每层的 K/V，新 token 只算自己的 QKV，历史计算永不重复 |
| 因果注意力（causal attention） | token i 只能看到位置 ≤ i；它是"缓存 K/V 而非 Q"的数学根源 |
| RoPE（旋转位置编码） | 按绝对位置旋转 Q/K 向量编码位置信息；相对角度只依赖位置差 |
| 位置偏移 bug | decode 用 0..T 而非 start_pos.. 取旋转表 → 模型以为每个新 token 都是句首 |
| TTFT（首 token 延迟） | 请求到第一个 token 的时间 = prefill + 首次采样 |
| TPOT（每 token 时间） | decode 单步平均耗时；tok/s = 1/TPOT |
| Amdahl's Law（阿姆达尔定律） | Speedup = 1/((1−p)+p/s)：占比 p 的环节加速 s 倍的上限 |
| torch.profiler | PyTorch 官方 profiler；CUDA activity + key_averages().table() 读占比 |
| self 时间 / 总时间 | 刨除子算子 / 含子算子——占比必须用 self 时间，否则重复计数 |
| nsys（Nsight Systems） | 系统级 profiler：看 kernel 之间时间线 → launch 开销的观测窗 |
| megakernel（巨核） | 把几十个小算子融进一个 kernel、只付一次 launch——decode 的解法，AMK 的地盘 |
| 算子融合（operator fusion） | 把相邻算子的访存/launch 合并；今天做的是 2 算子级（-20% 访存） |
| in-place（就地修改） | kernel 直接改写输入张量（残差流写回）；绕过 autograd，推理 OK 训练危险 |
| 残差流（residual stream） | 逐层累加更新的主状态张量；融合时必须写回 |
| launch 开销 | CPU 侧排队/发射 kernel 的固定成本（3–5 µs），与 kernel 计算量无关 |
| 贪心解码（greedy） | 每步取 argmax（temperature=0）——等价性验证的确定性前提 |
| PagedAttention | vLLM 的分页 KV Cache 管理（W5 主角），解决 cache 碎片化浪费 |

## 附 B：参考与延伸

- vLLM 的 `fused_add_rms_norm`（今天的融合 kernel 的工业同名实现）：https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
- vLLM / PagedAttention 论文（KV Cache 分页管理，W5 精读对象）：https://arxiv.org/abs/2309.06180
- RoPE 原论文（RoFormer）：Su et al., *"RoFormer: Enhanced Transformer with Rotary Position Embedding"* —— https://arxiv.org/abs/2104.09864
- torch.profiler 文档：https://pytorch.org/docs/stable/profiler.html
- Nsight Systems 文档：https://docs.nvidia.com/nsight-systems/
- 阿姆达尔定律：https://en.wikipedia.org/wiki/Amdahl%27s_law
- HuggingFace 的 KV Cache 概念文档（工业通识的写法）：https://huggingface.co/docs/transformers/kv_cache
