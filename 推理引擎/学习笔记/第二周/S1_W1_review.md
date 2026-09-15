# S1_W1_review.md —— 阶段一 W1 复盘：最有效动作 / 最大阻塞 / W2 承接点

> 2026-08-17 晚 · 对照 `阶段一W1_CUDA编程模型_逐日详细规划.md` 的完成标准逐条自检
> 本文只回答三件事：**哪个动作最值钱、哪里卡得最久、W2 第一天从哪直接开火（不冷启动）。**

---

## 1. 本周产出盘点（先摆事实）

| 类别 | 产出 | 状态 |
|---|---|---|
| CUDA kernel | `01_vector_add.cu`（含 grid-stride 版） | ✅ 与 PyTorch allclose |
| | `02_memory_access.cu`（coalesced vs strided） | ✅ 实测带宽差 ~10× |
| | `03_block_reduce.cu`（shared + 首次 syncthreads） | ✅ 与 `.sum()` 对齐，racecheck 过 |
| | `04_tree_reduction.cu`（7 个归约 kernel + 4 组实验） | ✅ 含 shuffle 两级归约 |
| | `05_tiled_matmul.cu`（朴素→tiling→寄存器分块） | ✅ 与 torch.matmul 对齐 |
| 笔记 | Day1 线程模型 / Day2 内存层级 / Day3 shared+同步 / Day4 归约 / Day5-6 tiling+差距分析 | ✅ 五份 |
| 引擎 | `test_generation.py` 安全网 + `benchmark_baseline.json` tok/s 基线 + `bench_harness.py` 对标脚手架 | ✅ |
| AMK | 用 warp/访存眼光重读 ncu，记下 warp 活跃度、DRAM 吞吐、memory/compute-bound 判定的带指标假设 | ✅（假设待 W4 验证） |
| 元笔记 | `week_summer1_view.md`（Triton 隐藏物总表 + 三杠杆 + 课题连线） | ✅ 本文 |

**完成标准自检（7 条）**：五个 kernel 数值对齐 ✅；能脱稿写归约+matmul 骨架 ✅（见 §4）；三杠杆能讲清 ✅；W0 三问闭环 ✅；引擎脊柱没退化 ✅；AMK 带指标假设 ✅；元笔记+复盘 ✅。

---

## 2. 最有效的动作（按 ROI 排序）

### 🥇 第一名：Day4 那次"优化无效"的诚实实验

**做了什么**：把树形归约用到"厚归约"（全数组求和，每线程上千元素）上，教科书说该快 30 倍，实测只快 1.02 倍。

**为什么最值钱**：这是本周唯一一次"理论预测和实测打架"的经历。逼我把 W7 的 Roofline 从"知道有这张图"变成"先判 bound 再决定优化什么"的肌肉记忆——**厚归约是 memory-bound，瓶颈在 HBM 带宽不在归约算法，优化归约当然没用**。

> **一句话**：优化前先问瓶颈在哪。这条纪律比"会写树形归约"本身值钱十倍，而且会贯穿整个暑假（W4 改 AMK 时就是靠它选靶子）。

### 🥈 第二名：Day5-6 把 tiling 的算术强度亲手推导两遍（朴素 0.25 → tiled TILE/4 → 寄存器分块 → 32）

**为什么值钱**：纸上推出 `AI = TILE/4` 那一刻，"FlashAttention 为什么分块"从"听说过"变成了"我算出来过"。而且推出 `BM·BN/(2(BM+BN))` 后发现 **BK 被约掉了**——BK 不影响算术强度只影响 shared 用量和流水线深度——这种"很多人搞错的点"亲手推一遍就再也不会错。

### 🥉 第三名：Day1 的 divergence 实测 + Day4 的 stride 方向对照

**为什么值钱**："stride 从大到小"本来只是要背的规则，但 Day1 先亲眼看过分支让 warp 闲置的 ncu 数据，Day4 再发现"stride 大到小恰好让活跃线程保持连续整 warp"——**两个概念自己咬合上了**。这种"知识点之间自己连线"的时刻，是元笔记存在的意义。

---

## 3. 最大阻塞（诚实记录，W2 别重踩）

| 阻塞 | 实际耗时 | 根因 | W2 对策 |
|---|---|---|---|
| **ncu 指标名看不懂**：`smsp__average_warps_issue_stalled_short_scoreboard` 这种名字第一次见直接懵 | ~1h | 指标命名有规律但没总结过 | 建一张"ncu 指标速查表"（按瓶颈类型分类：stall 类 / throughput 类 / hit_rate 类），W2 遇到新指标先归类再查文档 |
| **shared memory 声明大小的心智负担**：静态 vs 动态声明、和 occupancy 的关系，来回查了三次 | ~40min | 没把"shared 用量 ↔ 每 SM 能放几个 block"的换算公式写成一张卡 | 在笔记里钉死换算：`blocks/SM = min(寄存器限制, shared限制, 线程数限制)`，每次配 block 先算这三项 |
| **05 笔记的 §8/§9 没写完**（结尾还是 `<!--PART3-->`） | — | Day6 时间被暑训挤占 | 已在元笔记 §4.3 补上"tiling→FlashAttention"和"matmul→引擎 FFN/prefill-decode 两侧"的连线，知识链闭合；§8/§9 正文欠账滚到 W2 缓冲日补 |

> **没成为阻塞的隐患（侥幸）**：`__syncthreads()` 漏写的 race condition 本周没咬到我——因为 Day3 就养成了"先写 shared 后读 shared 中间必有栅栏 + 改完必跑 racecheck"的习惯。**这条习惯 W2 写 RMSNorm 时继续保命。**

---

## 4. W2 承接点（第一天不冷启动，直接从这里开火）

W2 主线 = **用原生 CUDA 重写 RMSNorm，接进引擎 `CudaBackend`，四方对标（torch / triton / 我的 CUDA / torch.compile）**。

### 4.1 直接起点：骨架已经存在，改三处就行

Day3 笔记 §7.3 已经写好了 RMSNorm 的完整骨架（当时作为"今天的知识怎么用"的预告）。W2 Day1 的活就三步：

```
① 把 04_tree_reduction.cu 里的两级归约（warp shuffle + 跨 warp shared）
   搬进 rmsnorm_kernel，归约目标从 sum 换成 sum of squares（local += v*v）
② 归约后加一次广播：thread0 算 scale = rsqrtf(mean + eps)，
   ★ 第二个 __syncthreads() 之后再让全 block 读（Day3 §7.3 的孪生模式）
③ 逐元素缩放写回：y[i] = x[i] * scale * w[i]（回到 element-wise，无需同步）
```

### 4.2 写之前就定好的验证与对标方案（不临场想）

- **正确性**：对照 `TorchBackend.rmsnorm`（W0 留的永远正确的 baseline），三尺子：allclose(atol=1e-5) → cosine → max abs err。**fp32 累加器**（即使输入 bf16——Day3 §1.3 的教训）。
- **性能**：用 `bench_harness.py`（Day2 搭好的）跑四方对标，预热 10 次 + 计时 50 次 + `cudaEvent`。
- **预期与心态**：我的 CUDA 版大概率与 Triton 打平或略慢——**这是预期产出**，分析"差在哪"就是"理解 Triton 编译器价值"的实证，不为刷赢熬夜。
- **先判 bound**：RMSNorm 算术强度 ≈ 1（读 2 次写 1 次），**memory-bound**，所以优化方向是减访存（如把 `xrow` 缓存进 shared 省第二次 global 读），不是减计算。**判错方向 = 白干一周**（Day4 的教训）。

### 4.3 引擎侧确认

`engine/backend.py` 的 `rmsnorm(x, weight, eps)` 接口签名已稳定（W0 Day1 定的粒度），W2 只需新增 `CudaBackend(Backend)` 实现这个接口，一行配置切换。**后端抽象的价值第一次兑现**——上层模型代码一行不动。

---

## 5. 下周砍什么（提前立好，不临场纠结）

按 W1 弹性规则执行顺序：

1. **先砍**【研】AMK 块：W2 的 AMK 任务（用 CUDA 眼光定位一个可改算子）若没时间，压缩到"只列候选清单"，不动手。
2. **再砍** 05 笔记 §8/§9 补写：知识链已在元笔记闭合，正文补写优先级最低。
3. **保底红线**：CUDA RMSNorm 主线深块 + 四方对标数据。**宁可某天真被暑训吃光，也要保住"RMSNorm 接进引擎并对标"这一件事**——它是 W2 的唯一里程碑。

---

## 6. GitHub commit 记录（本周 ≥3 次，实际 5+）

| commit | 内容 |
|---|---|
| `day1: cuda thread model + vector_add (grid-stride)` | kernel + 笔记 |
| `day2: memory hierarchy + coalesced vs strided 实测` | kernel + 笔记 + bench_harness |
| `day3: block reduce + 第一次 __syncthreads__` | kernel + 笔记 |
| `day4: tree reduction 7 版本 + shuffle + bank conflict` | kernel + 笔记 |
| `day5-6: tiled matmul + cuBLAS 差距分析` | kernel + 笔记 |
| `day7: W1 元笔记 + 复盘收口` | week_summer1_view.md + 本文 |

---

> **收尾一句话**：W1 是全程最陡的坡，扛下来了。下周做的事本质上只有一件——**把这周的树形归约，变成引擎里一个真的在跑、能对标、能讲清快慢原因的算子**。学的尽头是用，W2 开火。
