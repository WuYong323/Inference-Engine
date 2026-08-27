# 阶段二 · W3 Day 3 学习笔记 —— 前向 v1：online softmax 融合 + warp 级结构（本周最难的一天）

> **对应规划**：`阶段二W3_FlashAttention_逐日详细规划.md`（v1.1）→ W3 Day 3（8/29 周五）
> **今日目标**：回答"FA2 论文的'每 warp 一行'结构，比昨天的'每 block 一行'强在哪、贵在哪？"——把 v0 升级为 warp 级结构（BLOCK_M=4 行/block，归约全部 warp 内 shuffle，零 shared、零屏障）。
> **今日定位**：本周最深的一天。**半成品里程碑已定死（21:00 前未三方对齐 → 交付"v0+online 融合"版，warp 级顺延 Day4）。先对后快，对 > 快。**
> **前置**：Day 2 的 v0（对照物与翻译母本）、reduce.cuh 的 warp_reduce_max/sum、W2 的寄存器/occupancy 账。

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `csrc/flash_attn.cu` | **v0 + v1 双 kernel**（v0 保留作对照）+ 两个 wrapper（v1 带 GQA 参数） | 替换 `推理引擎/csrc/flash_attn.cu` |
| `bench/sim_v1.py` | v1 的 lane 语义模拟（与 kernel 逐行同构，先绿 CPU） | 复制进 `推理引擎/bench/` |
| `tests/test_flash_attn_v1.py` | v1 三层测试（sim/GPU/母本） | 复制进 `推理引擎/tests/` |
| `bench/bench_attn_v0v1.py` | v0 vs v1 vs 参考的首笔 %peak（**预测先行**） | 复制进 `推理引擎/bench/` |
| `bench/ref_attn.py` + `bench/sim_v0.py` | Day1/2 资产副本（独立自测用；仓库中同文件） | — |

---

## 0. 今天的问题与全景图

### 0.1 一个问题：强在哪、贵在哪

昨天 v0 的结构是"一个 block 一行"——每行要**两次 block 级归约**（max、sum），每次归约 = shared 写 + `__syncthreads` 屏障 + shared 读。今天换成 FA2 的"**每 warp 一行**"：归约在 warp 内用 shuffle 完成，广播用**一条 shuffle** 完成——**整个内层循环零 shared、零屏障**。

> **类比**：v0 是"全车间（block）一起算一行，每次要对账都要开全体大会（屏障）"；v1 是"每个班组（warp）独立负责一行，班组内部用对讲机（shuffle）沟通"——**大会开一次的成本是全体停工，对讲机喊一句只要一个周期**。这就是"强在哪"的精确答案。

"贵在哪"也今天回答：v1 每线程要扛 **acc[64] 的寄存器**（Day 2 已经埋的伏笔）——**这是一笔要用 occupancy 账算清楚的租金**（§4）。

### 0.2 时间盒导航

| 时间块 | 内容 | 章节 |
|---|---|---|
| 学 3.5h | v1 结构总图 + kernel 逐段精读 + 两个坑 | §1 – §3 |
| 学（并入） | 寄存器/occupancy 账 + 收益预测 | §4 |
| 造 2.5h | sim_v1 先行 + 并入 + 三方对齐 + 首笔 %peak | §5 – §6 |
| 研 1.5h | H100 打包清单（合并上机） | §7 |
| 沉淀 0.5h | 本笔记 | — |

---

## 1. v0 → v1 的结构升级总图（先建框架再进细节）

| | v0（Day 2） | v1（今天） |
|---|---|---|
| 每 block 负责 | 1 行 Q | **4 行 Q（BLOCK_M=4）** |
| 行 ↔ 执行单元 | block 全部 64 线程 | **每 warp（32 线程）一行** |
| S 的分布 | 每线程 1 列 | **每 lane 2 列（t 与 t+32——跨列模式）** |
| max/sum 归约 | block 级（shared + 屏障） | **warp 级（shuffle，零屏障）** |
| m/alpha 广播 | shared 写 + 屏障 + 读 | **一条 `__shfl_sync`** |
| 状态 m/l | 只有 thread 0 正确 | **广播后全 lane 正确** |
| shared | k_s + v_s（32KB） | k_s + v_s（32KB） |
| 寄存器 | acc[64]（同） | acc[64]（同——"贵"就在这） |

**S 的跨列分布（"t+32k"模式的正式亮相）**：BLOCK_N=64 列摊给 32 个 lane，lane t 负责列 t 和 t+32（k=0,1）。规划 Day 2 里那句"lane 负责列 t+32k"描述的正是今天 v1 的结构（昨天 v0 是每线程一列）。

---

## 2. kernel 逐段精读（配套 `csrc/flash_attn.cu`，v1 部分）

### 2.0 摊平与行守卫（新出现的两个边界问题）

```cuda
const int lane = threadIdx.x & 31;
const int wid  = threadIdx.x >> 5;
const int row  = blockIdx.x * BLOCK_M + wid;       // 本 warp 负责的 Q 行
const bool valid = row < Tq;                       // ★ 尾巴守卫：Tq 不整除 4 时，空转 warp 存在
const int h = bh % H, b = bh / H;
const int kv_h = h / kv_group;                     // ★ GQA：Q 头 → K/V 头（Day2 伏笔，今天正式进 kernel）
```

- **空转 warp（idle warp）**：Tq=63 时最后一个 block 的第 4 个 warp 的 row=63 ≥ Tq → 无效。**不能 early-return**（shared 加载是 128 线程协同 + block 级屏障，少一个线程就死锁）——空转 warp 必须**参加加载、不参与计算、不写回**，它的 NaN 垃圾被隔离在自己的寄存器里（§3 坑③）。
- **GQA 进 kernel**：K/V 的偏移用 `kv_h`，Q/O 用 `h`——Day 2 笔记说的"两处索引映射"，今天落地。

### 2.1 shared 加载（与 v0 相同，但 stride 变为 BLOCK_M×32）

```cuda
for (int i = threadIdx.x; i < n * BLOCK_D; i += BLOCK_M * 32) { ... }   // 128 线程协同
__syncthreads();
```

### 2.2 每 lane 两个 S（跨列模式 + 双重守卫）

```cuda
float s0 = -INFINITY, s1 = -INFINITY;
if (valid && lane < n) {                       // 守卫 1：行有效；守卫 2：列在块内
    float d0 = 0.f;
    #pragma unroll
    for (int d = 0; d < BLOCK_D; ++d) d0 += qr[d] * k_s[lane][d];
    s0 = d0 * scale;
    if (causal && (col0 + lane) > (row + (Tk - Tq))) s0 = -INFINITY;   // 通用 offset 掩码
}
if (valid && lane + 32 < n) {                  // 第二列（t+32）同理
    ...
}
```

### 2.3 ★ warp 归约 + shfl 广播（v1 的灵魂，今天的核心段落）

```cuda
float m_j = warp_reduce_max(fmaxf(s0, s1));    // lane 先合并自己两列，再 warp 归约
m_j = __shfl_sync(0xffffffffu, m_j, 0);        // ★ 广播：一条 shuffle，全员拿到正确 m_j
const float m_new = fmaxf(m, m_j);             // 单因子形式（Day1 坑⑤）
const float alpha = __expf(fminf(m - m_new, 0.0f));   // 见 §3 坑③的 NaN 防护
const float p0 = __expf(s0 - m_new), p1 = __expf(s1 - m_new);
float l_j = warp_reduce_sum(p0 + p1);
l_j = __shfl_sync(0xffffffffu, l_j, 0);        // 同样一条 shuffle 广播
l = l * alpha + l_j;
m = m_new;
```

**对照 v0 看"强在哪"**：

```
v0 每块每行：block_reduce_max（内部 1 次屏障+shared）→ shared 写 → 屏障 → 读
            block_reduce_sum（又 1 次屏障+shared）→ shared 写 → 屏障 → 读
            = 4 次 __syncthreads + 4 次 shared 往返
v1 每块每行：warp_reduce_max（5 条 shuffle）→ 1 条 shfl 广播
            warp_reduce_sum（5 条 shuffle）→ 1 条 shfl 广播
            = 12 条 shuffle，零屏障、零 shared
```

**shuffle 为什么比 shared 快**：shuffle 是寄存器直连（warp 内互访问寄存器），一个周期级；shared 要经过 SM 的片上存储，还要屏障对齐。**FA2 能跑满带宽的结构性原因，就在这 12 条 shuffle 里。**

### 2.4 acc 更新（两个列的贡献 + Day2 的越界坑）

```cuda
#pragma unroll
for (int d = 0; d < BLOCK_D; ++d) {
    acc[d] *= alpha;                           // 先缩旧（全体 lane）
    if (lane < n)      acc[d] += p0 * v_s[lane][d];       // 列 t 的贡献
    if (lane + 32 < n) acc[d] += p1 * v_s[lane + 32][d];  // 列 t+32 的贡献
}
__syncthreads();                               // 下一块加载前用完 shared（唯一的块级屏障）
```

**Day 2 抓到的坑在这里再次生效**：越界列（`lane ≥ n`）必须跳过 `v_s` 的读——shared 未初始化行可能是 inf，`0×inf=NaN`。v1 的守卫写进列条件里。

### 2.5 最终写回（warp 内归约 + lane0 单点写）

```cuda
if (valid) {                                   // 空转 warp 不写（它的 NaN 不落地）
    for (int d = 0; d < BLOCK_D; ++d) acc[d] = warp_reduce_sum(acc[d]);
    if (lane == 0)
        for (int d = 0; d < BLOCK_D; ++d) o[qo_off + d] = acc[d] / l;
}
```

注意：**不需要任何 shared**——跨 warp 不需要归约（每 warp 的 row 不同，各行独立写自己的 O 行）。v0 的"两级归约"在 v1 里退化成"warp 内单级"——**这是结构升级的红利**。

---

## 3. 两个坑 + 一个我加的坑（今天会真踩）

**坑①（规划点名）`warp_reduce_max` 的初值**：causal 掩掉的列是 −inf，归约单位元必须是 `-INFINITY` 而不是 0（含全负/全掩的行会把 0 当最大值——Day 1 的 reduce_max.cuh 已经埋好）。

**坑②（规划点名）别把 `exp(0)=1` 写成条件分支**：

```cuda
// ❌ 错误示范：alpha = (m == m_new) ? 1.0f : __expf(m - m_new);
//    warp 内 32 个线程各自判断，m==m_new 与否按 lane 不同 → 发散（divergence），
//    两路都串行执行一遍，把"省下的屏障"又还了回去。
// ✅ 正确：alpha = __expf(fminf(m - m_new, 0.0f));
//    m−m_new ≤ 0 恒成立，exp(0)=1 由硬件直接给出——分支为零，warp 永不发散。
```

**坑③（v1 新增）空转 warp 与全掩块的 NaN 隔离**：空转 warp 里 m=m_j=−inf → `m−m_new = NaN` → 若直接 `__expf(m−m_new)` 得到 NaN。NaN 被隔离在空转 warp 的寄存器里（不写回、不参与他人的归约），**无害但丑陋**。`fminf(NaN, 0) = 0`（CUDA 的 fminf 返回非 NaN 操作数）→ `__expf(0) = 1`——**一条无分支的工业级 NaN 防护**。全掩块（m_j=−inf、m 有限）同样被覆盖：m−m_new = 0 → alpha = 1 ✓。

---

## 4. 收益账与租金账（先算清楚再跑）

### 4.1 收益账（v1 为什么快）

1. **每块每行：4 次屏障 + 4 次 shared 往返 → 12 条 shuffle**（§2.3 对照表）；
2. **occupancy**：BLOCK=128（4 行/块），shared 32KB/块 → 3 块/SM（5060）→ 384 线程/SM = 25%；H100 7 块 = 896 线程 = 43.75%——**比 v0 的 12.5% 翻倍以上**，且 v0 那 12.5% 是 64 线程的块，v1 是 128 线程的块，每 SM 活跃 warp 数从 4 涨到 12（5060）。

### 4.2 租金账（v1 贵在哪——对规划的一处诚实修正）

规划的预期"regs 30-40 → 100% 档"**漏算了 acc[64]**：每线程 64 个寄存器只用来存 acc，加上 m/l/s/p/索引 ≈ **90–120 regs**。按 W2 阈值表（BLOCK=128）：

```
regs=100 → 65536/(100×128) = 5.1 → 5 blocks/SM = 640 线程 = 41.7% occupancy
regs=128 → 4 blocks = 33.3%
```

**v1 的实际 occupancy 预计 33–42%，不是 100%**——最终以 `-Xptxas -v` 实测为准（今天验证清单第 3 条）。这个租金正是 FA2 后续演化的方向：**split-D（把 D 维拆给多个 warp 分担 acc）**和 **BLOCK_M 增大摊薄固定开销**——记入 Day 4 差距清单的"可追项"，本周不动。

### 4.3 先写预测再跑（跑之前填，错也值钱）

| 战场 | 我预测 v1/v0 | 理由（写满） | 实测 | 对了吗 |
|---|---|---|---|---|
| prefill (2,4,1024,64) | | | | |
| decode (2,4,1,1024) | | | | |

**导师的推理链（供对照，先自己写）**：prefill 上 v1 应快 **1.5–3 倍**（occupancy 12.5%→40% 是主因，屏障消失是次因）；decode（T=1、grid 极小）两者都被 launch 主导，**差距会小得多甚至打平**——这正是 Day 4"decode 不用 Flash"的又一次预告。

---

## 5. 验证：sim_v1 先行（今天先绿 CPU）

配套 `bench/sim_v1.py`——与 kernel 的 **lane 语义逐行同构**（每 lane 两列、warp 归约、广播、全体 lane 状态一致）：

```python
# 核心循环（与 kernel §2.3/2.4 一一对应）
s = torch.full((2, nlanes), float("-inf"))          # 2 列 × 32 lane
for kk in range(2):
    col = torch.arange(nlanes) + kk * 32            # lane t → 列 t+kk*32
    ok = col < n
    if ok.any():
        s[kk, ok] = (qr @ kj[col[ok]].T) * scale
        if causal: s[kk, ok] = torch.where(col[ok] + jb > row + offset, neg_inf, s[kk, ok])
m_j = s.max().item(); m_new = max(m, m_j); alpha = math.exp(m - m_new)
p = torch.exp(s - m_new)
l = l * alpha + p.sum().item()
acc = acc * alpha                                # 全体 lane 先缩
for kk in range(2):
    col = torch.arange(nlanes) + kk * 32
    ok = col < n
    acc[ok] += p[kk, ok, None] * vj_pad[col[ok]]  # 越界 lane 不读 v
```

**验证三步**：① sim_v1 vs 母本（block 大小故意不同，顺带验证分块无关性）；② sim_v1 vs sim_v0（两个不同结构的语义必须一致）；③ CUDA v1 vs sim_v1（GPU，进仓库后生效）。

---

## 6. 【造】并入 + 首笔 %peak（2.5h）

1. `flash_attn.cu` 换成双 kernel 版（v0 保留作对照——**对照物永远不下架**）；bindings 加 `flash_attn_v1`（带 `kv_heads` 参数，默认 = H 即 MHA）；kernels.py 的 sources 不变（同一文件）。
2. wrapper 防御清单：同 v0 + `TORCH_CHECK(D == 64, "v1 今天只支持 head_dim 64")`（D=128 的 split-D 是大二上路线，**诚实写进报错信息**）+ `H % kv_heads == 0`。
3. 三方对齐跑通后，跑 `bench/bench_attn_v0v1.py` 记首笔 %peak——**先填预测表（§4.3）再跑**。

---

## 7. 【研】H100 打包清单（1.5h，合并上机：一次登录干两件事）

```bash
# ① softmax 两变体全形状扫描（Day2 的脚本，--gpu-tag 绝不覆盖 5060 数据）
python bench/sweep_softmax.py --gpu-tag h100

# ② attention v0/v1 首笔 H100 数据（本机先跑通，上机只执行）
python bench/bench_attn_v0v1.py --gpu-tag h100

# ③ 顺手为 Day4 预跑 ncu 四问（occupancy / dram / sector / spill）——集群时间金贵，能多带就多带
ncu --section Occupancy -k regex:flash_attn python bench/bench_attn_v0v1.py
```

**打包纪律（W2 Day5 延续）**：脚本本机 dry-run 过、输出路径写死、一次登录全清、回来 diff 数据文件。

---

## 8. 常见错误与调试速查表（Day 3 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| v1 与 sim 差 ~1e-2 | shfl 广播漏了（m_j/l_j 没广播就全员用） | §2.3：两次 `__shfl_sync(..., 0)` 缺一不可 |
| 只有 Tq 不整除 4 的形状错 | 空转 warp 写了 O / 没参加 shared 加载 | valid 守卫 + 绝不 early-return（屏障会死锁） |
| 全掩块/空转 warp 出 NaN 且会传染 | alpha 的 exp(NaN) | `fminf(m−m_new, 0.0f)` 无分支防护（坑③） |
| alpha 用条件分支后变慢 | warp 发散 | 坑②：exp(0)=1 交给硬件 |
| 全负数行 max 算成 0 | warp_reduce_max 初值错 | 单位元 -INFINITY（坑①，Day1 已埋） |
| 尾块 NaN（其余对） | 越界 lane 读 shared 未初始化行 | `if (lane < n)` / `if (lane+32 < n)` 双守卫 |
| occupancy 实测 ~40% 不是 100% | acc[64] 寄存器租金（预期内） | §4.2 的账；split-D 记入 Day4 可追项 |
| D=128 报错 | v1 只实例化 D=64 | 诚实报错信息；D=128 走 v0 或大二上 split-D |
| 21:00 还没三方对齐 | 硬扛 | ★ 半成品里程碑：交付 v0+online 融合版，warp 级顺延 Day4 |

---

## 9. 完成标准自测（先默写再对答案）

**规划题**：能脱稿画 v1 的"一个 warp 一行"数据流图，标出每步归约发生在哪。
*答案要点*：block(128 线程) = 4 warps × 4 行；每 warp：加载 shared（块级协同+屏障）→ 每 lane 算 2 列 S（t, t+32）→ `warp_reduce_max`（5 shuffle，warp 内）→ `__shfl_sync` 广播 m_j（1 shuffle）→ 单因子更新（全员一致）→ `warp_reduce_sum`（5 shuffle）+ 广播 l_j（1 shuffle）→ acc 列分片更新（寄存器）→ 循环末 1 次块级屏障；最终 64 次 `warp_reduce_sum` + lane0 写 O 行。**全部归约 12 条 shuffle/块/行，零 shared 零屏障（除加载屏障）。**

**附加题**：
1. "强在哪"一句话？（4 屏障+4 shared 往返 → 12 shuffle，寄存器直连）
2. "贵在哪"一句话 + 数字？（acc[64] → ~90-120 regs → occupancy 33–42%，规划 30-40 的账漏了 acc）
3. 空转 warp 为什么不能 early-return？（shared 加载 + 屏障是 block 级协同，少线程死锁）

---

## 10. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] v1 warp 版 kernel（或半成品版）三方对齐：sim_v1 / sim_v0 / 母本 / CUDA
- [ ] `-Xptxas -v` 寄存器实测 + occupancy 账（预测 90-120 regs，实测为准）
- [ ] v0/v1/参考首笔 %peak（预测表先填后跑）
- [ ] H100 打包完成（softmax 扫描 + attention 首笔 + ncu 预跑）
- [ ] 半成品里程碑未触发（或如实触发并记录）

**明日预告（Day 4）**：接进引擎 + 对标官方 SDPA + **三层差距清单**（cp.async 双缓冲列为可追项第 1）。今天的两笔账（occupancy 40%、acc 寄存器租金）就是明天差距清单的"我方数据"。

---

## 附 A：术语速查表（Day 3）

| 名词 | 一句话解释 |
|---|---|
| BLOCK_M | 每 block 的 Q 行数（v1 = 4，即 4 warps 各管一行） |
| 跨列模式（t+32k） | 每 lane 负责列 t, t+32, ...——S 的列按 32 跨步分发 |
| warp 级归约 | shuffle 归约：寄存器直连，5 条指令收敛，零屏障 |
| shfl 广播 | `__shfl_sync(mask, v, 0)`：从 lane0 取数广播——1 条指令替代 shared+屏障 |
| 空转 warp（idle warp） | Tq 不整除 BLOCK_M 时的无效行 warp：参加加载、不算不写 |
| 发散（divergence） | warp 内分支两路串行执行——坑②的根源 |
| fminf NaN 防护 | `fminf(NaN, 0)=0`（返回非 NaN 操作数）→ exp(0)=1，无分支 |
| split-D | 把 head_dim 拆给多个 warp 分担 acc 寄存器（FA2 后续演化，大二上） |
| 半成品里程碑 | 21:00 未对齐 → 交付 v0+online 版，warp 级顺延 Day4（对>快） |

## 附 B：参考与延伸

- FlashAttention-2 论文（每 warp 一行的 CTA 结构出处）：https://arxiv.org/abs/2307.08691
- Dao-AILab flash-attention 仓库（warp 级实现的工业源码）：https://github.com/Dao-AILab/flash-attention
- tiny-flash-attention（极简 FA 实现，读源码的结构参照）：https://github.com/66RING/tiny-flash-attention
- CUDA 编程指南（shuffle 指令与 warp 同步语义）：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Day 2 笔记（v0 对照物、跨列模式预告、越界读坑）
