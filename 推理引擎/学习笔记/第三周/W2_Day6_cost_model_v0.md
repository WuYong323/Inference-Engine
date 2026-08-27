# 阶段一 · W2 Day 6 学习笔记 —— cost model v0：性能能被算出来吗

> **对应规划**：`阶段一W2_CUDA进阶_手写算子接进引擎_逐日详细规划.md` → W2 Day 6（8/23 周日）
> **今日目标**：给我一个配置，**不跑**就算出它大概多快——把前五天的数据变成"方法"。这一天做出来，你的选题就从"想法"变成"有雏形"。
> **今日定位**：论文判决日。top-1 命中率决定方法类走不走得通，判决标准**今天提前定死**。
> **前置**：Day 2 的 occupancy/Little's Law、Day 3 的四板斧实测、Day 4 的 launch 开销、Day 5 的跨架构表——今天把它们全部装进一个函数。

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `research/cost_model.py` | 三项解析模型（predict_ms + 两卡 ARCH 表） | 放 `推理引擎/research/` |
| `research/calibrate.py` | 常数标定：launch_us / wave_latency / η（每卡 5 分钟） | 放 `推理引擎/research/` |
| `research/validate_model.py` | 验证：Spearman / top-1 命中率 / top-1 损失 + 散点图 + 判决 | 放 `推理引擎/research/` |
| `engine/backend_auto_snippet.py` | CudaBackend 的 `auto` 模式（cost model 自动选配置） | 合并进 `推理引擎/engine/backend.py` |
| `bench/bench_endtoend.py` | 端到端 tok/s 对比 + 诚实记录 | 放 `推理引擎/bench/` |
| `research/findings_day6_template.md` | 今日判决记录模板 | 复制进 `推理引擎/research/` |

---

## 0. 今天的问题与全景图

### 0.1 一个问题：预测 vs 试跑

Day 2 把工程师分成了"试跑派"和"计算派"。前五天你已经把"计算派"的零件攒齐了：occupancy 三限制、Little's Law、带宽效率 η、波次、launch 开销。今天把它们**焊成一个函数**：

```python
predict_ms(rows, H, block, variant, arch) -> 预测耗时
```

> **类比**：试跑派像"把每条路都走一遍再选"；今天的你是"画一张地图，标出每条路的预计耗时，走之前就选好"。地图不准没关系——**能解释为什么哪条路最快**，就是论文。

### 0.2 今天的完整闭环

```
前五天数据（sweep × 两卡）
        │
① 建模：predict_ms = max(t_mem, t_lat) + t_launch        §1
② 标定：launch_us / wave_latency / regs / η（每卡 5 分钟）  §2
③ 验证：Spearman + top-1 命中率 + top-1 损失 + 散点图      §3
④ 判决：提前定死的标准 → 方法类 / 缩小搜索空间 / 转测量类    §3.3
⑤ 落地：CudaBackend(auto) + 端到端 tok/s（诚实记录）      §4
```

### 0.3 时间盒导航（学研合并 5h + 造 2.5h）

| 时间块 | 内容 | 章节 |
|---|---|---|
| 1.5h | 模型的三项：把前五天知识装进一个函数 | §1 |
| 1h | 标定常数（每卡 5 分钟 × 2） | §2 |
| 2.5h | 验证三指标 + 判决 | §3 |
| 2.5h | auto 模式 + 端到端闭环 | §4 |

---

## 1. 模型的三项：把前五天的知识装进一个函数（1.5h）

### 1.0 建模哲学（先想清楚"模型是用来干嘛的"）

> **模型的价值不在于精确，在于它能解释"为什么最优配置会随形状跳变"——因为主导项换了。这句话就是你论文 introduction 的第一段。**

Day 3 的扫描图已经显示"最优块随形状变"。为什么变？因为三个物理瓶颈（带宽、并行度、发射）**各自主导一个区间**。模型把这三个瓶颈写成三项——**最优配置的跳变，就是三项中最大值的那一项在换班**。能解释跳变的模型，比拟合误差更小的模型值钱一百倍。

### 1.1 项一 t_mem：访存下界（带宽主导区）

```python
bytes_dram = 2 * rows * H * dtype_bytes        # 读 x + 写 y：所有变体的 DRAM 下界
eta = arch["eta_vec4"] if "vec4" in variant else arch["eta_scalar"]
t_mem = bytes_dram / (arch["hbm_gbs"] * 1e9 * eta) * 1e3      # ms
```

- **为什么所有变体的下界都是 2 份**：reread 的第二遍读命中 L1（Day 3 板斧四的结论），stash 的第二遍走寄存器——**它们省的是 L1 以上的开销，DRAM 层面都要读一次 x、写一次 y**。规划代码里的 `reread = 1.0 if variant == "stash"` 是**死代码**（赋值后没用）——v0 版删掉它，但你必须理解删它的理由，否则就是"抄代码"。
- **η（带宽效率）是什么**：实测带宽永远够不到理论峰值，这个"够得到的比例"就是 η。为什么 vec4 的 η 更高（0.90 vs 0.72）：① sector 利用率高（Day 5 §3 的合并度：128-bit 请求零浪费）；② 在途字节 ×4（Day 2 的 Little's Law：同样的线程数搬更多字节）。**η 的两个值不是拍脑袋——直接从 Day 3 扫描的实测 %peak 里读**（vec4 版峰值 ≈ 90%，标量版 ≈ 72%）。
- dtype_bytes：今天 fp32=4；bf16=2 是后续天的开关，模型里留好了参数位。

### 1.2 项二 t_lat：并行度不足暴露的延迟（波数主导区）

```python
regs = math.ceil(arch["regs_thread"].get(variant, 24) / 8) * 8   # ★ Day2 的粒度取整内置进模型
blocks_per_sm = min(arch["max_thr_sm"] // block,                  # 限制① warp 槽位
                    arch["regs_sm"] // (regs * block),            # 限制② 寄存器
                    arch["smem_sm"] // max(block // 32 * 4, 1))   # 限制③ shared
concurrent = arch["sm"] * blocks_per_sm                           # 全卡能同时住的 block 数
waves = math.ceil(rows / max(concurrent, 1))                      # 波数（Day 2 §1.4 的尾波）
t_lat = waves * arch["wave_latency_ms"]
```

- 三项取 min = **Day 2 §1.2 的 occupancy 三限制**，原样搬进模型；`ceil(regs/8)*8` = Day 2 的寄存器分配粒度（"手算和 API 对不上的第一大原因"），**模型里忘了它，stash 的预测就会系统性偏乐观**。
- `wave_latency_ms`（单波延迟）的物理含义：**一波 block 从发射到做完的时间**——当并行度不足时，总时间 ≈ 波数 × 单波时间。它的值靠标定（§2.3），不靠猜。
- **注意模型的取舍**：Day 2 的 occupancy 计算里"regs_thread"对 stash 来说随 EPT 变化（40–56），v0 用 per-variant 的固定值近似——**近似要写在注释里**，这是"知道模型缺什么"的第一步（判决标准里的"找缺项"就从这里开始）。

### 1.3 项三 t_launch：发射开销（小形状主导区）

```python
t_launch = arch["launch_us"] / 1e3      # ms；Day 4 的教训：发射与计算无关，恒付一次
```

Day 4 那张表已经算过：单请求 decode 时发射（3–5 µs）是有效计算（~5 ns）的 1000 倍。**launch 不依赖 rows、H、block——它是个常数**，所以小形状区间里"选什么配置都一样慢"，模型如实反映。

### 1.4 ★ 为什么是 `max(t_mem, t_lat) + t_launch` 而不是全加起来

```python
return max(t_mem, t_lat) + t_launch
```

这是今天最值得想一分钟的一行：

- **t_mem 和 t_lat 描述的是同一段时间的两种瓶颈**：带宽饱和时，内存延迟被大量 warp 隐藏（t_lat 不成立）；并行度不足时，带宽空转（t_mem 不成立）。真实时间取**两者中较大的那个**——这是 Day 2 roofline 思想的延续：时间是"两个下界的包络"。
- **t_launch 独立相加**：发射发生在 GPU 计算开始**之前**，不和计算重叠。
- **标注为 v0 近似**：真实世界里"部分重叠"存在（max 偏乐观），Day 7/W3 再修。**模型要进化，第一步是知道它哪里不精确。**

### 1.5 三区间主导表（背下来，它就是论文 Figure 2 的骨架）

| 区间 | 谁主导 | 例子 | 对应 Day |
|---|---|---|---|
| rows 很大 | `t_mem` | prefill：带宽是真瓶颈 | Day 2/3 的 %peak 世界 |
| rows 中等 | `t_lat` | batch decode：SM 没喂满，波数决定时间 | Day 2 的波次尾效 |
| rows 很小 | `t_launch` | 单请求 decode：发射比计算贵 1000 倍 | Day 4 的对照表 |

**完整实现见 `research/cost_model.py`**（含两卡 ARCH 表 + 上面对规划代码的三处修正：死代码删除、粒度取整、per-variant regs）。

---

## 2. 标定常数：每台卡 5 分钟（1h）

模型里每个"拍脑袋的数字"都要换成实测值——**模型的可信度 = 标定的严谨度**。配套 `research/calibrate.py`：

### 2.1 regs_thread：从 `-Xptxas -v` 读（不用跑）

Day 1 开的 `-Xptxas -v` 输出里，每个变体一行 `Used N registers`——三个数填进 `ARCH[card]["regs_thread"]`。**每张卡各读一次**（同一份 nvcc，但 H100 的 SASS 是 sm_90 编的，寄存器数可能不同）。

### 2.2 launch_us：空 kernel 实测

```python
def calibrate_launch_us(n: int = 2000, device: int = 0) -> float:
    """空 kernel 背靠背发射 N 次，elapsed/N = 每次 launch 的真实成本。
    为什么"背靠背除 N"而不是"单次计时"：单次测的是往返延迟，
    流水线里真正吃掉吞吐的是背靠背速率——模型要的是后者。"""
    src = r'''
    __global__ void empty_kernel() {}
    void launch_empty(int n) {
        for (int i = 0; i < n; ++i) empty_kernel<<<1, 32>>>();
    }
    '''
    mod = load_inline(name="launch_probe", cuda_sources=src, functions=["launch_empty"])
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); mod.launch_empty(n); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / n        # µs / launch，预期 2–5
```

### 2.3 wave_latency_ms：反解（不是直接测）

选一个 **waves=1 且 latency 主导**的已知点反解：`(32, 4096) + block=256`——5060 上 concurrent = 26×6 = 156 > 32 → **waves = 1**；此时 t_mem ≈ 4 µs 而实测 ~10–30 µs → **latency 主导**。于是：

```
wave_latency_ms = 该点实测 ms − launch_ms
```

`calibrate.py` 里已实现（用 bench_op 取 median 实测，再减 §2.2 的 launch）。**反解是建模的日常动作**：很多量没有直接仪器，但可以在模型方程里倒推——像物理实验里"测不出的常数用已知数据反解"。

### 2.4 η：从 Day 3 的实测 %peak 直接读

不用跑：`eta_scalar` = 你扫描表里标量版的最好 %peak（≈0.72），`eta_vec4` = vec4 版的最好 %peak（≈0.90）。**η 是"实测带宽/理论带宽"，扫描表里现成。**

### 2.5 标定表模板（填进 findings）

| 常数 | 5060 | H100 | 来源 |
|---|---|---|---|
| regs_thread（reread/vec4/stash） | / / | / / | `-Xptxas -v` |
| launch_us | | | 空 kernel 实测 |
| wave_latency_ms | | | (32,4096) 反解 |
| eta_scalar / eta_vec4 | 0.72 / 0.90 | / | Day 3 扫描 %peak |

---

## 3. 验证：三个指标 + 提前定死的判决（2.5h，今天的关键）

配套 `research/validate_model.py`：读 Day 2–5 的两卡 sweep JSON，对**每一个数据点**算 `predict_ms`，然后打三个指标。

### 3.1 三个指标逐个讲

**① Spearman 秩相关（rank correlation）——排序对不对**

只比较**排名**而非数值：把所有点按"预测时间"排一次序、按"实测时间"排一次序，看两个排序的一致程度（范围 −1~1）。**为什么不用 Pearson**：预测与实测是**非线性**关系（max 包络是三段的拼接），且实测有长尾噪声——Pearson 会被这些带偏，Spearman 只问"我认为 A 比 B 快，实测也这么认为吗"。**对配置选择来说，排序对就够了**——argmin 只依赖排序。

**② top-1 命中率——我预测的最优，实际是不是最优**

每个形状独立看：模型在该形状的 12 个配置（4 block × 3 variant）里选 argmin，和实测 argmin 比，命中 = 相同。命中率 = 命中形状数 / 总形状数。

**③ top-1 性能损失——就算没猜中，差多少**

模型选错了时，用"模型选的配置的实测时间"对比"oracle 最优的实测时间"：

```
top-1 损失 = (实测[模型选] − 实测[oracle]) / 实测[oracle]
```

命中时损失 = 0。**命中率回答"对不对"，损失回答"错了多疼"**——两个都要报。

### 3.2 散点图：x=实测 ms，y=预测 ms

对角线 = 完美预测。好模型的散点应该是**贴着对角线的三团簇**——三大区间各自成团（大 shapes 团在右上、小 shapes 团在左下）。**团没分开 = 标定常数串了**；整体偏离对角线 = 有系统性偏差项（那是 §3.3 "找缺项"的线索）。

### 3.3 ★ 判决标准：现在就定死，别到时候自我安慰

| 结果 | 含义 | 行动 |
|---|---|---|
| top-1 命中率 > 60% **或** top-1 损失 < 5% | **方法类有戏** | 大二上按方法类推进 |
| 命中率 30–60% | **改成"缩小搜索空间"** | 从 84 个配置降到 top-3，autotune 快 28 倍，仍是贡献 |
| 命中率 < 30% | 模型缺关键项 | 找出缺什么（**这本身是发现**），或转测量类 |

**为什么提前定死**：判决标准是"跑实验之前"的承诺。事后定标准的人，会在看到 58% 时把线画到 55%、看到 25% 时重新定义"命中"——**那是自我安慰，不是科学**。这跟你在 ACM 上设"退出阈值"是同一个纪律。

---

## 4. 【造】端到端闭环：auto 模式 + 诚实记录（2.5h）

### 4.1 CudaBackend(auto)：引擎开始"自己选配置"

配套 `engine/backend_auto_snippet.py`：

```python
class CudaBackend(TorchBackend):
    def __init__(self, block: int = 256, variant: str = "reread", auto: bool = False):
        self.block, self.variant, self.auto = block, variant, auto

    def rmsnorm(self, x, weight, eps=1e-6):
        from .kernels import rmsnorm as cuda_rmsnorm
        if not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous():
            return super().rmsnorm(x, weight, eps)
        block, variant = self._pick(x)              # auto：cost model 选配置；否则用手设值
        # 回退链照旧（Day 3）：vec4 条件不满足 → reread；stash H%block≠0 → reread
        return cuda_rmsnorm(x, weight, eps, block, variant)

    def _pick(self, x):
        if not self.auto:
            return self.block, self.variant
        rows, H = x.numel() // x.size(-1), x.size(-1)
        arch = ARCH[detect_card()]                  # 按设备名查 ARCH 表
        configs = [(b, v) for b in (128, 256, 512, 1024) for v in ("reread", "vec4", "stash")]
        return min(configs, key=lambda cfg: predict_ms(rows, H, *cfg, arch))
```

**这是"方法"的第一次落地**：引擎不再硬编码 block=256，而是每次前向按形状现场算最优——**零试跑选配置的完整闭环**。`_pick` 的耗时是微秒级纯 Python 运算，相比 3–5 µs 的 launch 可忽略（可加 `@functools.lru_cache` 按 (rows,H) 缓存，工业惯例）。

### 4.2 端到端跑生成：TorchBackend vs CudaBackend(auto)

配套 `bench/bench_endtoend.py`：同一个模型换 backend（Day 1 的"一行配置切后端"红利），各跑一轮生成，比 tok/s。

### 4.3 ★ 诚实记录（今天最重要的工程素养）

**预期：端到端提升大概率只有个位数百分比**——Day 4 的 Amdahl 检查已经预告了：RMSNorm 占端到端 3–8%，就算你的 kernel 比 torch 快 2 倍，端到端也只有 1.5–4%。

**记录它，并解释为什么——这比虚报一个大数字有价值得多。** 工业界的真实剧本：优化报告的第一段永远是"这个优化在端到端里值 X%，因为它的占比是 Y%"——**用 Amdahl 解释小数字的人被信任；只报大数字的人被怀疑**。这和你 Day 5 的"分母不同"caveat 是同一素养的工程侧。

### 4.4 工业对照：三种选配置路线的定位差

| 路线 | 代表 | 成本 | 你的定位 |
|---|---|---|---|
| 试跑派 | [Triton autotune](https://triton-lang.org/main/python-api/generated/triton.autotune.html) | 每配置真跑 + 编译，随维度爆炸 | 你的对照组（Day 3 记过 84 点的耗时） |
| 学习派 | [TVM AutoScheduler](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html) | 用 ML 代价模型 + 少量实测搜索 | 需要大量训练数据与工程 |
| **解析派** | **你的 cost model** | **几个硬件常数，0 试跑** | **卖点：零搜索成本 + 跨架构只用重标定** |

三者不互斥：解析模型选 top-3 + autotune 在 top-3 里精挑 = 搜索空间降 28 倍——这正是"命中率 30–60% 档"的退路。

---

## 5. 常见错误与调试速查表（Day 6 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| 预测和实测差一个数量级 | 单位换算错（GB/s×1e9×η、ms/µs） | 手算一个已知点核对量纲 |
| stash 预测系统性偏乐观 | regs 没按变体设 / 忘粒度取整 → occupancy 掉档没进模型 | `ceil(regs/8)*8` + per-variant regs（§1.2） |
| waves=1 的点预测差 | wave_latency 标定形状选错（waves≠1 或 memory 主导） | 标定用 (32,4096)+block=256（concurrent>32） |
| Spearman 高但 top-1 低 | 排序大体对、临界点附近错 | 看散点图临界区；那是"模型缺项"的精确位置 |
| 散点图三团簇没分开 | 三区间标定常数串了 | 重跑 §2 标定，先对 t_mem 团（最容易对） |
| η 超过 1 | 分母用了理论带宽 | 必须实测（Day 2 的 copy kernel 校准） |
| 端到端提升为 0 或负 | 正常（Amdahl）；或没走 auto 路径 | 打日志确认 `_pick` 被调用；诚实记录并解释 |
| 58% 命中率想"算及格" | 自我安慰 | 判决标准已提前定死（§3.3），不许改 |
| auto 模式选出的配置跑不了 | stash 的 H%block≠0 没回退 | 回退链照 Day 3：条件不满足 → reread |

---

## 6. 完成标准自测（两道题，先默写再对答案）

1. **模型三项各在什么区间主导？**
   *答案要点*：rows 很大 → `t_mem`（带宽下界：2×rows×H×bytes / (带宽×η)，prefill 区）；rows 中等 → `t_lat`（波数×单波延迟，occupancy 三限制决定 concurrent，batch decode 区）；rows 很小 → `t_launch`（3–5 µs 常数，单请求 decode 区）。组合方式：max(t_mem, t_lat) + t_launch——前两项是同一段时间的两种瓶颈取包络，launch 不重叠独立相加。**最优配置随形状跳变 = 主导项换班**，这句话是论文 intro 第一段。
2. **你的 top-1 命中率是多少？按判决标准该走哪条路？**
   *答案要点*：报三个数：Spearman（排序对不对）、top-1 命中率（argmin 猜中率）、top-1 损失（猜错了多疼）。判决（提前定死）：>60% 或损失 <5% → 方法类有戏；30–60% → 改"缩小搜索空间"（top-3 + autotune，快 28 倍仍是贡献）；<30% → 找缺项（本身是发现）或转测量类。**标准是跑实验前的承诺，不许事后改。**

---

## 7. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] `research/cost_model.py`（三项模型 + 两卡 ARCH 标定表）
- [ ] `research/calibrate.py` 跑完两卡标定（launch/wave_latency/η 记录在案）
- [ ] `research/validate_model.py` 输出三指标 + 预测 vs 实测散点图
- [ ] **判决结论**：方法类走不走得通（按提前定死的标准）
- [ ] `CudaBackend(auto)` + 端到端 tok/s（TorchBackend vs auto，诚实记录）

**明日预告（Day 7）**：不学新东西——W2 元笔记 + **选题立项**。写 `proposal_v1.md`（按论文结构：Problem / Observation / Approach / Preliminary results / Plan / **Threats to validity**——主动写下"这不就是 autotune 吗"的回应）；核实 CCF 目录与 2027 deadline（用你自己的眼睛）；**带着 proposal 和数据给张老师发消息问三件事**（重叠？立得住？一作？）——"带着 proposal 问和空手问，是完全不同的两件事"。今天判决的命中率，就是明天 proposal 里 Preliminary results 那一节的数字。

---

## 附 A：术语速查表（Day 6）

| 名词 | 一句话解释 |
|---|---|
| cost model（代价模型） | 不运行就预测算子耗时的解析模型；今天的三项式是 v0 |
| t_mem / t_lat / t_launch | 访存下界 / 波数×单波延迟 / 发射常数——三个区间各自主导 |
| η（带宽效率） | 实测带宽占理论峰值的比例；vec4 更高（sector 利用率 + 在途字节） |
| 包络（max） | 时间 = max(两个下界) + 不重叠项——roofline 思想的建模形态 |
| 标定（calibration） | 用实测把模型常数定下来；反解是"测不出的量从已知点倒推" |
| 反解（back-solve） | wave_latency = 已知点实测 − launch（选 waves=1 且 latency 主导的点） |
| Spearman 秩相关 | 只比排序不比数值的相关性——配置选择只依赖排序 |
| top-1 命中率 | 模型 argmin 与实测 argmin 相同形状的比例 |
| top-1 性能损失 | 选错时相对 oracle 的实测损失；"错了多疼" |
| oracle（神谕） | 穷举实测得到的最优——模型的对照基准 |
| 判决标准（pre-committed） | 跑实验前定死的路线标准——防止事后自我安慰 |
| auto 模式 | 引擎按形状现场调用 cost model 选配置（零试跑闭环） |
| autotune（自动调优） | 试跑派的工业形态：每配置真跑+编译选最优 |
| lru_cache | 按 (rows,H) 缓存选配置结果——微秒级开销的工程惯例 |
| 端到端占比 | 单算子时间/全流程时间——Amdahl 的分母（Day 4 的尺子） |

## 附 B：参考与延伸

- Triton autotune 官方文档（试跑派对照组的官方形态）：https://triton-lang.org/main/python-api/generated/triton.autotune.html
- TVM AutoScheduler（学习派代价模型）：https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html
- 哈佛 MLSys 教材《ML Systems Infrastructure Modeling》建模章节（解析建模的系统化写法）：https://harvard-edge.github.io/cs249r_book/
- 近期 GEMM 性能景观分析（"From Roofline to Ruggedness"，2026）：https://arxiv.org/pdf/2605.29752
- scipy Spearman 文档（脚本里手写实现，可用它交叉验证）：https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
