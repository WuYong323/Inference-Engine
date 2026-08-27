# 阶段二 · W3 Day 4 学习笔记 —— 接进引擎 + 本机对标官方 SDPA

> **对应规划**：`阶段二W3_FlashAttention_逐日详细规划.md`（v1.1）→ W3 Day 4（8/30 周六）
> **今日目标**：回答"你的 Flash 和官方 SDPA 差多少？差在哪？**以及：为什么 decode 不用 Flash？**"——v1 接进引擎（形状分支）+ 本机对标 + 三层差距清单。
> **今日定位**：本周的"学→造"闭环日。三层差距清单是本周最有学术价值的产出之一。
> **前置**：Day 3 的 v1（三方对齐）、W2 Day4 的端到端三级验证 SOP、W2 Day2 的 occupancy 账方法。

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `bench/flash_branch.py` | 形状分支的纯函数（可独立测试） | 复制进 `推理引擎/bench/` |
| `engine/backend_flash_snippet.py` | `CudaBackend.attention` 的 Flash 路径 | 合并进 `推理引擎/engine/backend.py` |
| `tests/test_flash_branch.py` | 分支逻辑测试 + 引擎级对照（层②进仓库生效） | 复制进 `推理引擎/tests/` |
| `bench/bench_sdpa.py` | v0/v1/eager/SDPA 对标（三尺子 + 下游 top-1） | 复制进 `推理引擎/bench/` |
| `bench/softmax_cross_arch.py` | 研线：softmax 跨架构表（复用 W2 cross_arch.py） | 复制进 `推理引擎/bench/` |

---

## 0. 今天的问题与全景图

### 0.1 两个问题

1. **差多少、差在哪**：你的 v1 是"FA2 骨架的教科书版"（Day 3 完成）。官方 SDPA 大概率还快 2–5 倍——**这个差距必须被拆解成分层清单**，而不是一句"官方厉害"的挫败感。
2. **为什么 decode 不用 Flash**：W2 Day4 的 launch 账今天在引擎里兑现成一行形状分支——**方法适用区间思维的第四次落地**（前三次：Day5 caveat、Day7 proposal、W3 校准三）。

### 0.2 时间盒导航

| 时间块 | 内容 | 章节 |
|---|---|---|
| 学 3.5h | occupancy 三方对齐 + ncu 四问 + 差距清单方法论 | §1 – §3 |
| 造 2.5h | 形状分支接引擎 + 三级验证 + 本机对标 | §4 – §6 |
| 研 1.5h | softmax 跨架构表 | §7 |
| 沉淀 0.5h | 本笔记 | — |

---

## 1. 三方对齐：给 v1 算 occupancy 账（今天第一件事）

### 1.1 手算（延续 W2 的三限制方法）

```
5060（BLOCK=128，shared 32KB/块）：
  ① 线程槽 1536/128 = 12  ② 寄存器 65536/(R×128)：R=100 → 5
  ③ shared 100KB/32KB = 3  ④ block 槽 32
  → min = 3 blocks/SM = 384 线程 = 25% occupancy
H100：
  ③ shared 228/32 = 7 → min(12, 5, 7, 32) = 5 → 640/2048 = 31.25%
```

**读法**：v1 的瓶颈在 **shared**（每块 32KB 只能住 3 块）——比 v0 的 12.5% 翻倍，但离"喂饱"还远。**这个账今天要用 API 和 ncu 两次验证**（三方对齐纪律），并在差距清单里指向答案：官方用 cp.async 双缓冲不是为了少 shared，是为了**让加载不占时间**（§2.2）。

### 1.2 ncu 四问（W2 Day5 方法论复用，本机即可起步）

```bash
# ① 我在 Roofline 的哪里？
ncu --metrics dram__bytes.sum.per_second,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    -k regex:flash_attn python bench/bench_sdpa.py
# ② achieved occupancy vs 手算 25%
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active -k regex:flash_attn ...
# ③ 合并度：Q 按行读的 sector 效率（float 理想 4.00/请求）
ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio -k regex:flash_attn ...
# ④ 有没有寄存器溢出（acc[64] 的租金现形）
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum -k regex:flash_attn ...
```

---

## 2. ★ 差距清单的三层结构（今天最有学术价值的产出）

### 2.1 三层总图

```
第 0 层  你的 v1 = FA2 骨架的教科书版（分块 + online softmax + warp 归约）
         —— 本周目标已在此达成，这是"理解"层
第 1 层  工程技巧层（与官方 FA2 的差距，可逐项追）：
         ① cp.async 双缓冲（Ampere+ 即可用，你的 5060 就能练）★ 可追项第 1
         ② 更细分块 / 寄存器调优（split-D 分担 acc[64]）
         ③ tensor core（需 bf16 —— W7 量化周呼应）
第 2 层  架构特性层（与 FA3/FA4 的差距，记录方向，不当周追）：
         FA3 = Hopper 的 wgmma + TMA + warp specialization（大二上 H100 路线）
         FA4 = Blackwell 流水线大改、达矩阵乘法级速度（前沿瞭望）
```

**差距清单从"模糊的挫败感"变成"分层的可追路线图"**——每一层都有明确的"下一步是什么、什么时候做"。这就是你 W2 Day5"诚实标注对手优势来源"素养的升级形态。

### 2.2 可追项第 1 详解：cp.async 双缓冲是什么

**cp.async（异步拷贝指令）**：sm_80（Ampere）引入的指令——把 global → shared 的拷贝变成**异步**的：不经过寄存器、不等完成，CPU/GPU 先干别的。配合双缓冲（两块 shared 轮流使用）：

```
无 cp.async： [加载K块——等待] [计算] [加载K块——等待] [计算]    ← 加载与计算串行
cp.async：    [预算加载块1] [加载块2 进行中 ‖ 计算块1] [加载块3 ‖ 计算块2]
              ← 加载被计算"盖住"，K/V 搬移接近零暴露
```

**为什么它是可追项第 1**：① 与架构无关（Ampere 及以后全支持，5060/sm_120 就能练）；② 效果可预期（加载占 v1 时间的大头——§1.1 的 shared 瓶颈的另一面就是"每块都要等加载"）；③ 工业证据充分（lmdeploy 的 sm80 attention 主循环就是 cp.async 流水线的教科书）。**它是"不改算法、只改搬运方式"的典型工程技巧——正适合 W4 动手。**

### 2.3 架构特性层一句话定位（不展开）

[FA3](https://arxiv.org/abs/2407.08608)（Hopper）与 [FA4](https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet)（Blackwell）用上了**架构专属**特性（wgmma 张量核指令、TMA 张量内存加速器、warp specialization 生产者-消费者分工）——这些特性无法在 5060 上练，但**知道它们的存在决定了你的学习路线**：大二上上 H100 时，差距清单第 2 层就是路线图。

---

## 3. 对手分析：官方 SDPA 为什么快（对标前的功课）

**SDPA（scaled_dot_product_attention）** 是 PyTorch 的统一注意力入口，内部按"形状/dtype/硬件"自动**派发（dispatch）**到多个后端：flash（FA 系）/ mem_efficient（xFormers 系）/ cudnn / math（朴素版）。先看清它用了谁：

```python
import torch
print("flash 可用:", torch.backends.cuda.can_use_flash_attention(
    torch.empty(1, 1, 1, 1, dtype=torch.float16, device="cuda"), None))
print("mem_efficient 可用:", torch.backends.cuda.can_use_mem_efficient_attention(
    torch.empty(1, 1, 1, 1, dtype=torch.float16, device="cuda"), None))
# 也可用上下文管理器强制指定后端做 A/B（bench_sdpa.py 里有）
```

**官方快在哪（差距清单第 1 层的完整版）**：① cp.async/寄存器双缓冲；② warp 级软件流水线（warp specialization 的雏形）；③ 更细的分块与寄存器分派（split-D、BLOCK_M 更大摊薄固定开销）；④ fp16/bf16 下用 tensor core（fp32 反而常回落到 mem_efficient/math 后端——**今天 fp32 对标里"官方"未必是它的最强形态，这也是要写进对标表的事实**）。

---

## 4. 【造】接引擎：形状分支（校准三的落地）

### 4.1 分支条件设计（配套 `bench/flash_branch.py`）

```python
def should_use_flash(q, k, v, flash_min_t: int = 16) -> bool:
    """Flash 路径的准入条件。每一条都对应 wrapper 里的 TORCH_CHECK（防御清单同步）：
    ① 设备/精度/连续：fp32 CUDA 连续——v1 的能力边界；
    ② head_dim == 64：v1 今天的实例化（D=128 的 split-D 是大二上）；
    ③ T_q >= flash_min_t：★ 形状分支——decode 的 launch 账（W2 Day4：T=1 时
       发射是有效计算的 ~1000 倍），小形状换内核不如不换。
    阈值 16 是起点，今天对标数据出来后用 decode 区间的实测定标。"""
    return (q.is_cuda and q.dtype == torch.float32
            and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
            and q.size(-1) == 64
            and q.size(2) >= flash_min_t)
```

### 4.2 `CudaBackend.attention`（配套 `engine/backend_flash_snippet.py`）

```python
def attention(self, q, k, v, causal=True):
    from .kernels import flash_attn_v1
    if should_use_flash(q, k, v, self.flash_min_t):
        return flash_attn_v1(q, k, v, causal=causal)   # kv_heads 默认 -1 = MHA（引擎现状）
    return super().attention(q, k, v, causal=causal)   # ← 回退：TorchBackend 的 SDPA 路径
```

**两个设计点**：① **分层回退**（Day 1 思想的第四次复用）：条件不满足 → torch 路径，永不报错；② **KV Cache 拼接后的 T_kv 直接喂 kernel**——`causal` 里 Tk ≥ Tq 天然成立（Day 2 的 offset 掩码式已经在等它）。

---

## 5. 端到端三级验证（复用 W2 Day4 SOP，一级不省）

1. **等价性**：同一 prompt、贪心解码（temperature=0），**Flash 路径与纯 torch 的生成 token 逐 token 相同**——复用 `bench/bench_kvcache.py` 的等价性部分，把 backend 换成 `CudaBackend`（Flash 已接）；
2. **性能**：tok/s 对比存 `logs/baseline_flash.json`（与 W2 的 `baseline_kvcache.json` 并排——**暑假所有优化的分母，今天多了一个刻度**）；
3. **占比**：torch.profiler 看 attention 的 self 时间占比（Amdahl 判"值不值"——引擎里 attention 占比比 RMSNorm 大，这是 Flash 比 rmsnorm"值钱"的第一次证据）。

**三尺子的第三把换成系统级**：allclose（松 1e-4）/ cosine（≥0.9999）/ **下游 logits top-1 一致**（注意力输出过一层固定随机投影后取 argmax——kernel 级的微小误差只有"过一层模型"才算数，这是 W2 三尺子在引擎场景的升级）。

---

## 6. 本机对标：v0/v1/eager/SDPA（配套 `bench/bench_sdpa.py`）

**先写预测再跑**：

| 战场 | 我预测 v1/SDPA | 理由（写满） | 实测 | 对了吗 |
|---|---|---|---|---|
| prefill (2,4,1024,64) | | | | |
| prefill (1,4,2048,64) | | | | |

**导师的推理链（供对照，先自己写）**：fp32 下官方 flash 后端可能不启用（常回落到 mem_efficient/math）——所以差距可能比想象小，但也因此**今天的对标对象是"PyTorch 能拿出的最好 fp32 实现"**；预测 v1/SDPA 在 0.3–0.8 之间，v1/v0 延续 Day 3 的 1.5–3 倍。**记录 SDPA 实际派发到哪个后端**（脚本里打印），这个事实本身就是对标表的一列。

```python
# bench_sdpa.py 核心结构
impls = {
    "v0":    lambda: flash_attn(q, k, v, causal=True),
    "v1":    lambda: flash_attn_v1(q, k, v, causal=True),
    "eager": lambda: attn_ref(q, k, v, causal=True),
    "sdpa":  lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
}
# 三尺子：allclose(1e-4) / cosine / 下游 logits top-1（固定随机投影 proj，argmax 一致性）
# 附：torch.backends.cuda.sdp_kernel 上下文强制 flash 后端的可选 A/B（可用时）
```

---

## 7. 【研】softmax 跨架构表（1.5h，论文线第 4 步）

配套 `bench/softmax_cross_arch.py`——一行调用复用 W2 的 `cross_arch.py`：

```bash
python bench/softmax_cross_arch.py   # 自动找 research/data/sweep_softmax_*.json（两卡）
```

**产出即论文线第二算子的第一张跨架构图**：5060 vs H100 的 softmax 最优配置 + 移植损失。与 Day 6 的 rmsnorm 判决并排，就是"算子数 1→2"的初步证据。**移植损失 ≥20% = 选题成立一半的第二个数据点；~2% = 今天就知道该收敛范围。**

---

## 8. 常见错误与调试速查表（Day 4 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| 引擎里 Flash 没生效（走了 torch） | 分支条件不满足（dtype/D/阈值） | 打日志看 `should_use_flash` 各条件；对照 wrapper 的 TORCH_CHECK |
| 生成 token 与纯 torch 不一致 | 分支两端 causal 语义不同 | 层① 等价性验证抓；检查 `causal=(T>1)` 与 kernel 的 offset 掩码 |
| SDPA 报"no kernel available" | fp32 + 该后端不支持 | 记录实际派发后端；可 force mem_efficient（脚本里有） |
| ncu 的 achieved occupancy 远低于手算 25% | 波次尾效（grid 小时） | 用大 grid 战场重测；对照 §1.1 三限制找瓶颈项 |
| 差距比预想小/大 | fp32 下官方后端回落 | 把"实际后端"写进对标表（诚实记录的一部分） |
| tok/s 提升不明显 | attention 占比本身有限（Amdahl） | 占比数据说话；诚实记录（Day 4 素养的第三次演练） |
| 阈值 16 拍脑袋不安心 | 阈值应该数据定标 | 今天对标后：把 decode 区间的实测填进 flash_branch.py 注释 |

---

## 9. 完成标准自测（先默写再对答案）

**规划题 1**：差距清单三层各对应什么？可追项第 1 是什么？
*答案要点*：第 0 层 = 我的 v1（FA2 骨架教科书版，本周已达成）；第 1 层 = 工程技巧层（cp.async 双缓冲★可追项第 1、split-D/寄存器调优、tensor core）；第 2 层 = 架构特性层（FA3 的 wgmma/TMA/warp specialization、FA4 流水线——记录方向，大二上 H100 路线）。cp.async = 异步拷贝 + 双缓冲，加载被计算盖住，Ampere+ 即可练。

**规划题 2**：为什么 decode 分支不切 Flash？
*答案要点*：decode T=1 → grid 极小、launch 3–5µs 是有效计算 ~1000 倍（W2 Day4 表）→ Flash 天花板为 0。形状分支（T_q ≥ flash_min_t）+ 分层回退（不满足 → torch SDPA 路径）。这是"方法适用区间"思维的第四次落地。

**附加题**：为什么 v1 的 occupancy 只有 25% 而官方能跑满？（shared 32KB/块限 3 块 + acc[64] 寄存器租金——官方用 cp.async 双缓冲和 split-D 逐项解掉，正好是差距清单第 1 层。）

---

## 10. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] v1 occupancy 三方对齐记录（手算 25%/31% / API / ncu）
- [ ] `CudaBackend.attention` Flash 路径 + 形状分支 + 分层回退
- [ ] 端到端三级验证通过 + `logs/baseline_flash.json`
- [ ] 本机对标表（v0/v1/eager/SDPA × 两战场 × %peak + 三尺子 + 实际后端列）+ 预测对照
- [ ] **三层差距清单**（§2 结构，写进 findings）
- [ ] softmax 跨架构表（两卡，移植损失列）

**明日预告（Day 5）**：H100 正式对标 + 反向数学（D 修正项）+ torch 反向母本。今天本机的对标表是明天 H100 表的预演；三层差距清单的第 1 层（cp.async）会在 W4 动手兑现。

---

## 附 A：术语速查表（Day 4）

| 名词 | 一句话解释 |
|---|---|
| SDPA（缩放点积注意力统一入口） | `F.scaled_dot_product_attention`：按形状/dtype/硬件派发到多个后端 |
| dispatch（后端派发） | SDPA 自动选 flash / mem_efficient / cudnn / math 的机制 |
| 形状分支（shape branch） | `T_q ≥ flash_min_t` 才走 Flash——decode 不切 Flash 的落地 |
| cp.async（异步拷贝指令） | sm_80+ 的 global→shared 异步拷贝，不经寄存器、不等完成 |
| 双缓冲（double buffering） | 两块 shared 轮流：加载下一块与计算当前块重叠 |
| split-D | 把 head_dim 拆给多 warp 分担 acc 寄存器（差距清单第 1 层②） |
| 下游 logits top-1 | 三尺子第三把的系统级版：误差过一层投影才算数 |
| 实际后端（记录） | SDPA 到底用了谁——诚实对标表的一列 |
| 移植损失 | 用 A 卡最优配置跑 B 卡的性能损失（论文线判决数据） |
| 阈值定标 | flash_min_t 的初值 16 用今天 decode 区间实测校准 |

## 附 B：参考与延伸

- PyTorch SDPA 教程（后端派发的官方说明）：https://pytorch.org/tutorials/intermediate_source/scaled_dot_product_attention_tutorial.html
- sdp_kernel 上下文管理器（强制指定后端）：https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.sdp_kernel
- lmdeploy 的 sm80 attention 主循环（cp.async 流水线的工业教科书）：https://github.com/InternLM/lmdeploy
- FA3（架构特性层出处）：https://arxiv.org/abs/2407.08608
- FA4（Blackwell 前沿）：https://lambda.ai/blog/flashattention-4-gives-the-nvidia-blackwell-platform-its-most-optimized-attention-kernel-yet
- W2 Day4 笔记（三级验证 SOP / decode launch 账）、W2 Day5 笔记（ncu 四问 / 诚实 caveat）
