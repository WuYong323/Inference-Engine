# 阶段二 · W3 Day 2 学习笔记 —— 前向 v0：分块 attention（先对，不求快）

> **对应规划**：`阶段二W3_FlashAttention_逐日详细规划.md`（v1.1）→ W3 Day 2（8/28 周四）
> **今日目标**：回答"T×T 的 S 矩阵是怎么做到算完就扔的"——把 Day 1 的 torch 母本逐行翻译成 CUDA（前向 v0：一个 block 一行 Q，block 级归约）。
> **今日定位**：先对后快。v0 的性能可以难看，但必须**三方可证**（母本 / torch sim / CUDA），并且"难看多少"要能用 W2 的账算出来。
> **前置**：Day 1 的母本（单因子形式）、reduce.cuh（含 Day 1 合并的 max 归约）、W2 的 occupancy 三限制与 48KB shared 认知。

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `csrc/flash_attn.cu` | 前向 v0 kernel + wrapper | 复制进 `推理引擎/csrc/`，接入见文件头注释 |
| `tests/test_flash_attn.py` | 三层测试：CPU sim / GPU kernel / GQA | 复制进 `推理引擎/tests/` |
| `bench/sim_v0.py` | v0 的 torch 语义模拟（与 kernel 逐行同构） | 复制进 `推理引擎/bench/` |
| `csrc/softmax_online.cu` | 研线：softmax v1（online 一遍版，2 扫 vs v0 的 3 扫） | 复制进 `推理引擎/csrc/` |
| `bench/sweep_softmax.py` | 研线：softmax 两变体全形状扫描（--gpu-tag） | 复制进 `推理引擎/bench/` |

---

## 0. 今天的问题与全景图

### 0.1 一个问题的精确表述

Day 1 的母本 `attn_fa_style_ref` 是 torch 里的一串 `einsum` + 循环。今天要把它**逐行翻译**成 CUDA。翻译的难点不在数学（Day 1 已推完），在**降维**：

```
母本的维度                       CUDA 的载体
B·H 个头            →   （并入 grid，与行一起摊平）
Tq 个查询行          →   grid = B·H·Tq 个 block，每 block 一行
Tk 个键（分块）      →   block 内的 for 循环（外层 K/V 块）
每块 BLOCK_N 列      →   每线程一列（BLOCK_N 个线程）
每行一次 max/sum 归约 →   block_reduce_max / block_reduce_sum（W2 资产）
```

**"S 算完就扔"的落地形态**：S 的每一块只活在一个 block 的一次迭代里——算出来的值直接进 P，P 进 acc，**S 从头到尾不碰 global memory**。今天你写的每一行都在兑现这句话。

### 0.2 时间盒导航

| 时间块 | 内容 | 章节 |
|---|---|---|
| 学 3.5h | 设计决策（三个"算出来的"选择）+ kernel 逐段精读 | §2 – §3 |
| 造 2.5h | sim 先行验证 + 接入 + 三层测试 | §4 – §5 |
| 研 1.5h | softmax v1（online 版）+ 5060 扫描启动 | §6 |
| 沉淀 0.5h | 本笔记 | — |

---

## 1. 翻译前的两个"降维"决策（先想清楚再动手）

### 1.1 grid 摊平：把 (B, H, Tq) 三个维度压成一条线

```cuda
const int row = blockIdx.x % Tq;    // 本 block 负责的 Q 行
const int bh  = blockIdx.x / Tq;    // B·H 索引（哪个 batch、哪个头）
// grid = B * H * Tq 个 block；K/V 的偏移用 bh 和绝对列号（含 Tk），Q/O 用 bh 和 row
```

**为什么一行一个 block**：softmax 的归约是**行内**的——block 内的归约正好服务一行，行与行零依赖。这与 W2 Day1 的 rmsnorm 排布是同一个决策（"归约在 block 内做"的硬件根源），今天在 attention 上复用。

### 1.2 线程分工：每线程一列 S（澄清规划里的一处混入）

规划写"lane 负责列 t+32k"——那是 **v1（每 warp 一行）的跨列模式**，不小心混进了 v0 的描述。**v0 的分工更简单：BLOCK_N 个线程，每线程负责本 K 块的一列**（t = threadIdx.x，S_t = q_r · k_t）。t+32k 的跨列模式明天 v1 再见。

---

## 2. 设计决策：三个"算出来的"选择（今天第一个学习点）

### 2.1 为什么 BLOCK_N = 64 而不是规划的 128（shared 预算账，含一处修正）

规划的账："128×64 fp32 = 32KB"——**只算了 K_j，漏了 V_j**。实际 shared 需求是 **K_j + V_j 两份**：

```
BLOCK_N=128, D=64: (128×64×4) × 2 = 64 KB   ← 超过 48KB 默认块上限！
BLOCK_N= 64, D=64: ( 64×64×4) × 2 = 32 KB   ✓ 安全落在默认 48KB 内
```

**48KB 这个数从哪来**：CUDA 编程指南规定每 block 默认最多用 48KB shared；要突破需 `cudaFuncSetAttribute` 设置 **opt-in carveout（可选配比）**（5060 上最多 99KB）。v0 的定位是"先对"，不值得为它开 opt-in——**BLOCK_N=64 是算出来的，不是拍脑袋**（这正是规划 Day2 那句话的兑现：账算错了就得改设计，而不是硬跑）。BLOCK_N=128 + opt-in 留作 Day 4 的性能实验。

### 2.2 occupancy 账（把 v0 的"慢"提前量化）

```
5060, BLOCK_N=64, D=64: shared/块 = 32KB → 100KB/32KB = 3 blocks/SM（48KB 默认下同时受 3 块限制?）
   实际：3 blocks × 64 threads = 192 threads/SM = 12.5% occupancy（1536 上限）
H100: 228KB/32KB → 7 blocks/SM = 448 threads = 21.9%
```

**v0 天生 occupancy 低**——先把它写进笔记：v0 的性能天花板就是 ~12%（5060）。明天 v1 的每一处改进（warp 级归约省 shared、行分块提高线程数）都对照这个账来算收益。**"先对"不等于"不量"：难看的数字写下来，明天才知道进步了多少。**

### 2.3 单因子形式照搬（Day 1 坑⑤的兑现）

母本已用单因子形式（先定 m_new 再算 P）。kernel 里照搬——**全掩块天然免疫 NaN**：

```cuda
const float m_j = block_reduce_max<BLOCK_N>(s);   // 本块行 max（thread0 正确）
if (threadIdx.x == 0) { sm_m = fmaxf(m, m_j); sm_a = __expf(m - sm_m); m = sm_m; }
__syncthreads();
const float m_new = sm_m, alpha = sm_a;
const float p = (threadIdx.x < n) ? __expf(s - m_new) : 0.f;  // 全掩/越界 = exp(-inf−有限) = 0
```

---

## 3. kernel 逐段精读（配套 `csrc/flash_attn.cu`）

### 3.0 全貌：一个 block 的生命周期

```
grid 里第 bh·Tq+row 个 block：
  状态：m（thread0 私有）、l（thread0 私有）、acc[BLOCK_D]（每线程一份，列分片）
  for 每个 K/V 块 j：
    ① 加载 K_j、V_j → shared（合并访问）
    ② 每线程算 S 的一列（点积 + scale + causal 掩码）
    ③ 归约 max → thread0 更新 m/alpha → 广播（屏障）
    ④ 算 P → 归约 sum → thread0 更新 l
    ⑤ 每线程 acc = acc·alpha + p·V_j[t]（列分片累加）
  （块间屏障：下一块加载前，全员用完 shared）
  ⑥ 最终：两级归约 Σ_t acc_t → 除以 l → 写回 O 行
```

### 3.1 ① 加载 K/V 块到 shared（合并访问）

```cuda
for (int i = threadIdx.x; i < n * BLOCK_D; i += BLOCK_N) {
    const int c = i / BLOCK_D, d = i % BLOCK_D;         // 摊平索引 → (列, 维)
    const size_t off = (size_t)bh * Tk * BLOCK_D + (size_t)(col0 + c) * BLOCK_D + d;
    k_s[c][d] = k[off];
    v_s[c][d] = v[off];
}
__syncthreads();                                        // 全员写完才读
```

- **合并访问（coalesced）**：连续线程搬连续地址（i 连续 → off 连续）→ 一个 warp 的请求落在连续 128B，sector 零浪费（Day 5 §3 的合并度知识第一次在实战 kernel 里用）。
- **shared 里按 [列][维] 存**：后面点积 `k_s[t][d]` 是**行连续**读取（每线程读自己那行的 64 个连续 float）→ 跨线程的 shared 访问 stride=64×4B → 轻微 bank conflict，v0 接受（v1 会改布局，Day 4 记入差距清单）。

### 3.2 ② 每线程算 S 的一列 + causal 掩码

```cuda
float s = -INFINITY;
if (threadIdx.x < n) {                                   // 尾块：越界列不参与
    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < BLOCK_D; ++d) dot += qr[d] * k_s[threadIdx.x][d];
    s = dot * scale;
    if (causal && (col0 + threadIdx.x) > (row + (Tk - Tq))) s = -INFINITY;   // ★ 通用 offset 掩码
}
```

- **因果掩码的 offset 式**（Day 1 §4.1 的 CUDA 版）：`col > row + (Tk − Tq)`——decode 形状（Tk>Tq）下依然正确；Tk=Tq 时退化为标准 `col > row`。
- 掩掉的列 = −inf → 进 max 归约不污染（max 单位元 −inf），进 P = 0。**S 算完立即掩掉，不碰 exp 的 inf 传播**（规划原文）。

### 3.3 ③④ 归约与状态更新的契约序列（今天最容易写错的时序）

```cuda
const float m_j = block_reduce_max<BLOCK_N>(s);          // ★ 契约：只有 thread 0 的值正确
if (threadIdx.x == 0) { sm_m = fmaxf(m, m_j); sm_a = __expf(m - sm_m); m = sm_m; }
__syncthreads();                                         // 广播 m_new 与 alpha
const float m_new = sm_m, alpha = sm_a;
const float p = (threadIdx.x < n) ? __expf(s - m_new) : 0.f;
const float l_j = block_reduce_sum<BLOCK_N>(p);          // ★ 同样只有 thread 0 正确
if (threadIdx.x == 0) l = l * alpha + l_j;               // l 只在 thread 0 维护
```

**这段是 W2 reduce.cuh 契约的完整应用**：归约结果只在 thread 0 正确 → 状态（m、l）只在 thread 0 更新 → 需要全体用的量（m_new、alpha）走"写 shared → 屏障 → 读"两步。**跳过任何一个屏障，其他 63 个线程就会拿着垃圾 m_new 算 P——静默错。**

### 3.4 ⑤ acc 的列分片累加

```cuda
#pragma unroll
for (int d = 0; d < BLOCK_D; ++d)
    acc[d] = acc[d] * alpha + p * v_s[threadIdx.x][d];
```

- **列分片**：线程 t 只负责"列 t 的贡献" P_t · V_j[t, :]——acc 是 D 维向量，**每个线程都有一份，但内容不同**（各自只装了自己那列的分量）。这和第 2.2 节的 occupancy 账呼应：acc[D] 每线程 D 个寄存器，D=64 时 ~64+ 寄存器/线程——**v0 的寄存器压力也写进明天的优化清单**。
- 注意 `acc = acc·alpha + ...` 的顺序：**先缩旧、再加新**——alpha 作用于旧累积（Day 1 公式的直接翻译）。
- **★ 越界线程必须跳过 v_s 的读**（sim 先行抓到的真实 bug）：尾块时线程 t ≥ n 的 p=0，但 `p * v_s[t]` 会读到 **shared 未初始化的行**——垃圾若是 inf/NaN，`0×inf = NaN` 污染整行。正确写法：`acc[d] *= alpha; if (t < n) acc[d] += p * v_s[t][d];`

### 3.5 ⑥ 最终两级归约与写回

```cuda
for (int d = 0; d < BLOCK_D; ++d) acc[d] = warp_reduce_sum(acc[d]);   // 第一级：warp 内
__shared__ float o_s[BLOCK_N / 32][BLOCK_D];
const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
if (lane == 0) for (int d = 0; d < BLOCK_D; ++d) o_s[wid][d] = acc[d];
if (threadIdx.x == 0) sm_l = l;                        // l 广播给写回者
__syncthreads();
if (wid == 0)
    for (int d = 0; d < BLOCK_D; ++d) {
        float vv = (threadIdx.x < BLOCK_N / 32) ? o_s[threadIdx.x][d] : 0.f;
        vv = warp_reduce_sum(vv);                      // 第二级：warp0 内
        if (lane == 0) o[qo_off + d] = vv / sm_l;      // O = Σacc / l
    }
```

**为什么最终是"列分片求和"**：O[d] = Σ_列 P_列·V[列,d] = Σ_线程 acc_线程[d]。这就是 W2 两级归约的第三个用途（rmsnorm 的统计量、softmax 的 m/l、今天的 acc 向量）——**同一件工具，第三次出现在不同的位置**。

---

## 4. 正确性三步走：sim 先行（今天先绿 CPU，再绿 GPU）

### 4.1 为什么今天先写 torch sim（配套 `bench/sim_v0.py`）

CUDA kernel 的编译+调试链路（nvcc→JIT→GPU）比纯 torch 慢一个数量级。**今天的三步走把第一步放在 CPU 上**：`sim_v0` 与 kernel **逐行同构**（同单因子形式、同列分片 acc、同 offset 掩码），先证明"设计是对的"，再让 CUDA 版本去对齐它——调试从"找 bug"变成"找两版差异"（W2 方法论）。

```python
# sim_v0 的核心循环（与 kernel §3 一一对应）
for jb in range(0, Tk, block_n):
    s = torch.full((block_n,), float("-inf"))
    s[:n] = (qr @ kj.T) * scale                 # ← kernel 的 ②
    if causal: s[:n] = torch.where(cols > row + offset, -inf_tensor, s[:n])
    m_j = s.max().item()                        # ← kernel 的 ③
    m_new = max(m, m_j); alpha = math.exp(m - m_new)
    p = torch.exp(s - m_new)                    # ← 单因子：全掩/越界 = 0
    l = l * alpha + p.sum().item()
    acc = acc * alpha + p[:, None] * vj         # ← 外积 = kernel 的 ⑤ 列分片
    m = m_new
o[b, h, row] = acc.sum(0) / l                   # ← kernel 的 ⑥
```

### 4.2 三层测试（配套 `tests/test_flash_attn.py`）

| 层 | 什么 | 何时绿 |
|---|---|---|
| ① sim vs 母本 | `sim_v0 == attn_fa_style_ref`（CPU，含 decode 形状/非对齐/D=128/GQA） | **今天，写 kernel 之前** |
| ② kernel vs sim | CUDA `flash_attn` vs `sim_v0`（GPU） | 今天，kernel 编译通过后 |
| ③ kernel vs 母本 | 端到端三重确认 | 今天收尾 |

判据：allclose(atol=1e-4, rtol=1e-4) + cosine ≥0.9999；fp32 主线；累加顺序差异 ~1e-6 正常。

### 4.3 D=128 的诚实标注

v0 实例化 `<BLOCK_D=128, BLOCK_N=32>`：shared 32KB 安全，但 **acc[128] 每线程 128 个寄存器必然溢出到 local memory**（性能差、结果对）。这是"先对"的边界：**D=128 今天只保证正确性，性能问题记入 v1 的动机清单**（Day 4 的 ncu 四问里 `local_op` 指标会现形）。

---

## 5. 【造】接入引擎前的"独立验证区"（2.5h）

1. `bindings.cpp` 加：`m.def("flash_attn", &flash_attn_cuda, "FlashAttention fwd v0 (block-per-row)", py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = true);`
2. `kernels.py` 的 sources 加 `"flash_attn.cu"`（改完重跑即重编）；
3. wrapper 的防御清单（照抄 Day 1 夹具层）：形状 4D、dtype fp32、contiguous、`causal 时 Tk ≥ Tq`、D ∈ {64, 128}；
4. **今天 kernel 不进引擎**（规划纪律）：在 `tests/test_flash_attn.py` 里独立验干净，Day 4 再接 `CudaBackend.attention`。

---

## 6. 【研】softmax v1（online 版）+ 扫描启动（1.5h）

### 6.1 v1 与 v0 的差异：3 扫 → 2 扫（配套 `csrc/softmax_online.cu`）

```
v0 两遍版：扫1 max 归约 → 扫2 exp+sum 归约 → 扫3 写回（重读 x）    = 3 扫
v1 online：扫1 在线合并 (m,l)（单因子更新）→ 扫2 写回               = 2 扫
```

关键在**每线程分片的合并**：每线程扫完自己的元素后有局部 (m_t, l_t)，先 `block_reduce_max` 得全局 m*，再 `l = l·exp(m_t−m*)` 换算后 `block_reduce_sum` 合并——**这是 Day 1 单因子公式在"归约合并"场景的第二次应用**（第一次在 attention 的块间，这次在软 max 算子的线程间）。

### 6.2 扫描启动（配套 `bench/sweep_softmax.py`）

复用 W2 sweep 框架：rows×H 网格（沿用 rmsnorm 的 7 形状，softmax 与 rmsnorm 同形状可对比）+ 两变体 + `--gpu-tag`（**不覆盖他卡数据**）。今天本机 5060 跑完，H100 随 Day 3 打包。bytes_moved 同 rmsnorm：`2·rows·H·4`。

---

## 7. GQA 可选加分（有余力）

**GQA（分组查询注意力）**：Q 头数 > K/V 头数，每 `G = Q_heads/KV_heads` 个 Q 头共享一组 K/V。**kernel 结构不变**，只需两处索引映射：

```cuda
const int kv_h = h / G;                    // Q 头 h → K/V 头 kv_h
// K/V 偏移用 kv_h；Q/O 偏移用 h。其余一字不改。
```

配套测试里加了 (Q_heads=8, KV_heads=2) 的 sim 用例。**这是 W0 差异清单"没做 GQA"伏笔的兑现**——Llama-3 全系标配，你的 kernel 从今天起就是工业形状的。

---

## 8. 常见错误与调试速查表（Day 2 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| kernel 结果与 sim 差 ~1e-2 级 | 归约后没走"写 shared→屏障→读" | 检查 m_new/alpha/l 的三处广播时序（§3.3） |
| 只差 decode 形状 | causal 的 offset=Tk−Tq 漏了 | §3.2 的通用掩码式 + 测试②decode 用例锁死 |
| 尾块（Tk 非整除）错 | 越界列没挡：s 初值/点积/写 shared 三处 | `if (threadIdx.x < n)` 三件套 + (63,65) 用例 |
| 尾块出现 NaN（其余对） | ★ 越界线程读 shared 未初始化行：0×inf=NaN | acc 更新先缩、再 `if (t<n)` 才加（sim 先行抓到的坑） |
| 全掩块 NaN | 用了论文双因子形式 | 单因子（Day1 坑⑤）；m 初值 −inf |
| BLOCK_N=128 编译/运行报错 | shared 超 48KB 默认块上限 | v0 用 64；128 需 opt-in（Day4 再开） |
| D=128 结果对但奇慢 | acc[128] 溢出到 local memory | 预期行为（§4.3），记入 v1 动机清单 |
| 编译报 reduce.cuh 缺 max | Day1 的 reduce_max 没合并进仓库版 | 先合并 `reduce_max.cuh` 内容再编 |
| sim 与母本差 1e-6 以上 | 别慌：两者都是 torch，应是 1e-7 内 | 差 1e-5 就查 alpha 更新顺序（先缩后加） |

---

## 9. 完成标准自测（先默写再对答案）

**规划题**：能说清 v0 里"哪些数据在 shared、哪些在寄存器、S 为什么没落地"。
*答案要点*：shared = K_j/V_j 块（每块迭代重载，32KB/块）+ 归约用的小槽位 + 广播槽 sm；寄存器 = acc[D]（每线程 D 个，列分片）、m/l（thread0）、s/p（每线程标量）；S 只在寄存器里活一轮（算完→进 max/exp→扔掉），**永不写 global**——HBM 只进出 Q/K/V/O，这正是"算完就扔"的落地形态。

**附加题**：
1. BLOCK_N 为什么是 64？（K+V 双份 32KB ≤ 48KB 默认块上限；128 需 opt-in）——账算出来，不拍脑袋。
2. 最终 acc 为什么需要两级归约？（O[d] = Σ_列 P·V[列,d] = Σ_线程 acc_t[d]——列分片求和，warp 内 shuffle + shared 跨 warp）。
3. 哪三处广播时序错了会静默算错？（m_new、alpha、l 的"写→屏障→读"）。

---

## 10. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] `csrc/flash_attn.cu`（v0 kernel + wrapper，防御清单齐全）
- [ ] `bench/sim_v0.py` 与母本全绿（含 decode/非对齐/D=128/GQA）
- [ ] `tests/test_flash_attn.py` 三层测试全绿（GPU 上 kernel vs sim vs 母本）
- [ ] `csrc/softmax_online.cu`（v1，2 扫）+ `bench/sweep_softmax.py` 5060 扫描数据
- [ ] shared 预算账 + occupancy 账写进笔记（v0 的 12.5% 天花板在案）

**明日预告（Day 3）**：v1（warp 版）——每 warp 一行、归约全部 warp 内 shuffle（block 内零同步）、BLOCK_M=4 行/block。今天的三笔账（occupancy 12.5%、acc 寄存器压力、D=128 溢出）就是明天每一步改进的对照基准。

---

## 附 A：术语速查表（Day 2）

| 名词 | 一句话解释 |
|---|---|
| 降维翻译 | 母本的张量维度 → CUDA 载体（grid/block/thread/循环）的映射 |
| 摊平（flatten） | (B,H,Tq) 三索引压成一个 grid 索引，再用 % 和 / 解回来 |
| 列分片 acc | 每线程只累加"自己那列"对输出的贡献，最后求和归并 |
| 合并访问（coalesced） | 连续线程访问连续地址——加载 shared 时的 sector 效率 |
| 48KB 默认块上限 | CUDA 每 block 默认最多 48KB shared；opt-in carveout 可放宽 |
| 契约广播序列 | 归约结果只在 thread0 正确 → 写 shared → 屏障 → 全员读 |
| 单因子形式 | 先定 m_new 再算 P：全掩块免疫（Day1 坑⑤） |
| 两级归约 | warp 内 shuffle + 跨 warp shared——本周第三次使用（rmsnorm/softmax/acc） |
| local memory 溢出 | 寄存器不够 → 溢出到 DRAM 备份；D=128 的 acc[128] 必然触发 |
| GQA（分组查询注意力） | 多 Q 头共享 K/V 头；kernel 只改两处索引映射 |
| sim 先行 | 与 kernel 逐行同构的 torch 模拟，先在 CPU 上验证设计 |

## 附 B：参考与延伸

- CUDA C++ Programming Guide（48KB 默认 shared 与 opt-in carveout 的权威出处）：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- FlashAttention 论文（v0 结构 = 其 Algorithm 1 的 block 级实现）：https://arxiv.org/abs/2205.14135
- FlashAttention-2（CTA 结构：每 warp 一行——明日 v1 的目标形态）：https://arxiv.org/abs/2307.08691
- Triton 教程 FA 章节（与 v0 同构的编译器版本，对照"它替你写了什么"）：https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- Day 1 笔记（母本 / 单因子形式 / offset 掩码 / reduce_max）
- W2 Day2 笔记（occupancy 三限制 / shared 预算的算账方法）
