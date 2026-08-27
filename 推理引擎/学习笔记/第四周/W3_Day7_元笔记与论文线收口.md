# 阶段二 · W3 Day 7 学习笔记 —— 元笔记 + 论文线收口 + W4 桥接

> **对应规划**：`阶段二W3_FlashAttention_逐日详细规划.md`（v1.1）→ W3 Day 7（9/2 周二）
> **今日目标**：不学新东西——把本周沉淀成资产（元笔记），把论文线第二算子正式入账（findings + proposal 更新），桥接 W4。
> **今日定位**：W3 收官日。本周所有产出（含可能的降级路径产物）今天统一归档。
> **前置**：Day 1–6 全部产出（母本、v0/v1、反向 CUDA、对标、cost model softmax）。

## 本文件夹内容（笔记 + 配套模板）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `tech_notes/week_summer3_view.md` | W3 元笔记（预填五块骨架，闭卷默写后填数） | 放 `推理引擎/tech_notes/` |
| `research/findings_w3_template.md` | 本周数据记录模板（对标/差距清单/第二算子/反向） | 复制改名 `findings_w3.md` 使用 |
| `research/proposal_v1_update.md` | proposal 更新块（Preliminary 1→2 算子 + Plan + Threats 增补） | 合并进 `research/proposal_v1.md` |
| `tools/week3_checklist.py` | 本周完成标准自动盘点 | 放 `推理引擎/tools/` |

---

## 0. 今天的问题与全景图

### 0.1 三个任务，一个问题

1. **元笔记（3h）**：把一周 2000 行笔记压缩成一张可检索的地图（W2 Day7 的方法论第二次使用）；
2. **论文线收口（2h）**：softmax 第二算子的跨架构表 + cost model 命中率——**复现 or 不复现，都要写结论**（判决标准已提前定死，今天只读数）；
3. **桥接（1.5h）**：欠账清零 + W4 预热 + GitHub 里程碑。

为什么"不学新东西"的一天是收官关键：**知识不沉淀，遗忘曲线接管；数据不入账，实验等于白做。**

### 0.2 时间盒导航

| 时间块 | 内容 | 章节 |
|---|---|---|
| 3h | W3 元笔记（五块内容） | §1 |
| 2h | findings_w3 + proposal 更新 | §2 – §3 |
| 1.5h | 欠账 + W4 桥接 + GitHub | §4 |
| 收尾 | 周自检 + 复盘三行 | §5 |

---

## 1. 【主线 3h】W3 元笔记：五块内容（配套 `tech_notes/week_summer3_view.md`）

**方法论延续 W2 Day7**：元笔记 = 货位图（压缩 / 串联 / 检索）；**闭卷默写再对答案**（提取练习）。今天多一块——五块各回答什么问题：

**① FA 总表**——回答"这周学会了哪些**带数字**的知识"：

| 概念 | 一句话 | 关键数字（默写！） |
|---|---|---|
| online softmax 三状态 | m/l/acc + 单因子更新 | rescale 因子 = exp(差值)；全掩块免疫 |
| tiling 访存账 | S 永不落 HBM | O(T²) → O(T)：T=2048 省 ~33 倍（反向账） |
| v0 vs v1 结构 | block 级归约 → warp 级 shuffle | 4 屏障+4 shared → 12 shuffle；occupancy 12.5% → 25% |
| 反向 D 修正 | rowsum(dO⊙O)，行梯度零和 | ΣdS = 0（平移不变性） |
| 三件套三次出场 | reduce/列分片/两级归约 | rmsnorm → softmax → attention（前向+反向） |

**② 对标表 + 三层差距清单**——回答"和工业界差多远、怎么追"：本机对标数字 + **三层结构**（教科书版 = 本周已达成 / 工程技巧层 = cp.async 可追项第 1 / 架构特性层 = FA3/FA4 记方向）。差距清单是"诚实标注对手优势来源"的第三次升级。

**③ "FA 的战场边界"**——回答"本周最反直觉的结论"：prefill/decode 形状分支——**单算子优化适用区间思维的第三次落地**（前两次：W2 Day5 caveat、W2 Day7 proposal 边界）。一句话版本：`T_q ≥ 16 才走 Flash，decode 的天花板是 0`。

**④ 串联表**（知识换乘图，搬自规划 §6）：FA tiling↔W1 05_tiled_matmul；online softmax↔W1 Day4 §7.3；warp 归约↔W2 reduce.cuh；访存账↔W7 融合两笔账；softmax 第二算子↔W2 Day6 cost model；反向重算↔W6 attention 反向。

**⑤ ★ 工业趋势三行（v1.1 新增，各一句话 + 一个"接哪条线"）**：

```
FlexAttention（PyTorch 组合式 attention DSL）：mask_mod/score_mod 拼注意力变体，
  不再每种变体手写一个 kernel——官方已与 FA4 合流（"既快又灵活"）。
  → 你的定位：会写底层才看得懂 DSL 的边界——这是读 vLLM 源码的地基
MLA（DeepSeek 多头潜在注意力）：KV 压缩到低秩潜在空间，显存/吞吐双赢，已成推理主流方向。
  → 接 W5 线（KV Cache 管理的进化形态）
GQA（分组查询注意力）：本周 kernel 已顺手支持（kv_h = h/group）——W0 差异清单伏笔兑现。
  → Llama-3 全系标配：你的 attention 从今天起就是工业形状
```

**手写能力与上层抽象的关系一句话**：FA4 的流水线、FlexAttention 的 DSL、MLA 的压缩——**全部建立在本周那个 200 行骨架之上；地基不是终点，是通行证**（W3 规划 §7 的收尾语，今天正式写进元笔记）。

---

## 2. 【研 2h】findings_w3 + proposal 更新

### 2.1 findings_w3（配套模板，逐栏填写）

三块核心数据：**对标表**（v0/v1/eager/SDPA × 两战场 × %peak + 三尺子 + 实际后端）、**softmax 第二算子**（跨架构表 + cost model 三指标）、**反向**（三方对齐 or 降级记录）。

**★ 第二算子的判决写法（复现 or 不复现，都要写结论）**：

```
复现（同档命中率）→ "方法在第二个算子上泛化：Preliminary 升级为 2 个算子"
不复现          → "模型缺 softmax 特定项（两遍归约的 η / 归约密度）——模型进化清单第 1 条"
```

**两条路都是论文素材**——这正是 Day 6 判决标准"提前定死"的价值：今天不存在"结果好不好"，只存在"读数往哪归档"。

### 2.2 proposal_v1 更新（配套 `research/proposal_v1_update.md`）

| 节 | 更新内容 |
|---|---|
| **Preliminary results** | 算子数 **1 → 2**（rmsnorm + softmax）：各自 Spearman/top-1/损失三数字 + 复现结论 |
| **Plan** | 下一步算子：**layernorm → top-k**（vocab=128K 的长归约，Day5 风险表的备选战场） |
| **Threats to validity** | 增补一条："模型依赖每算子标定 η（5 分钟/算子/卡）——量化进搜索成本对比，autotune 每配置真跑+编译，我们仍是 O(1)" |

**文档长大论的第二次兑现**：proposal 从 Day7 的骨架开始，今天填进第二个算子的数据——不重写，只长大。

---

## 3. 【弹性 1.5h】欠账 + W4 桥接 + GitHub

### 3.1 欠账清零（配套 `tools/week3_checklist.py` 自动盘点）

```powershell
python tools/week3_checklist.py            # 文件级盘点（学/造/研三组 ✅/❌）
python tools/week3_checklist.py --run-tests # 顺带跑全量 pytest
```

**没有全绿的项今天补齐**——欠账跨周利滚利（W2 的 KV Cache 债已经演示过一遍）。

### 3.2 W4 预热：融合两笔账重读 + 两个预告

- **重读**：W7"融合两笔账"笔记（省访存 + 省 launch）——W4 的 fused QKV / fused FFN 就是这两笔账的下一批兑现对象；
- **两个预告**（各留一句话，别展开）：**fused QKV** = 三次小 GEMM 合成一次大 GEMM（vLLM 已有融合 q/k/v 线性组的工业实现）；**fused FFN/SwiGLU** = 门控线性层的算子融合——**融合深度纪律照旧：相邻 2–3 个算子，超过即入师兄地盘，停手**。

### 3.3 GitHub ≥5 次有意义 commit

kernel 里程碑各一：v0 / v1 / 引擎接入 / 反向两 pass / softmax 算子——**每个 commit 一个可陈述的改动**（W2 Day7 的定义不变）。

---

## 4. 本周完成标准自检（9/2 晚，全部勾选才算过）

**【学】**
- [ ] 能脱稿推 online softmax 的 m/l/acc 三行更新公式，并说清 rescale 因子的来历
- [ ] 能脱稿画 v1"每 warp 一行"的数据流图，标出每次归约发生在哪、用什么指令
- [ ] 能推 D_i 修正项，说清"重算 vs 存储"的字节账
- [ ] 能解释为什么 decode 不用 Flash（launch 账）

**【造】**
- [ ] 前向 v0/v1 standalone 三层验证全绿（含 D=128 用例）；v1 接进引擎，端到端等价性 + tok/s 基线入库
- [ ] H100 对标表：v0/v1/eager/SDPA × 两战场 × %peak + 三尺子 + **三层差距清单**
- [ ] 反向 CUDA 与 autograd 三方对齐（或按降级路径交付 torch 参考版+推导笔记）
- [ ] `tests/` 全绿；GitHub ≥5 次有意义 commit

**【研】**
- [ ] softmax 算子两变体 + 跨架构扫描（两卡数据不覆盖）+ 跨架构表
- [ ] cost model 支持 softmax（第二算子），Spearman/top-1 验证有结论（复现 or 不复现都入账）
- [ ] proposal_v1 的 Preliminary results 更新为"2 个算子"

---

## 5. 常见错误与调试速查表（Day 7 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| 元笔记变成抄笔记 | 提取练习没发生 | 闭卷默写五块，再对答案（§1 纪律） |
| findings 只写"复现了/没复现"不写数字 | 判决标准要求三指标 | Spearman/top-1/损失三个数齐全 |
| proposal 更新当成重写 | 文档长大论 | 只改 Preliminary/Plan/Threats 三节（§2.2 表） |
| 差距清单写"官方厉害"没分层 | 三层结构没进元笔记 | 对照 §1 块②：教科书版/工程技巧层/架构特性层 |
| checklist 报路径错 | 没从仓库根目录跑 | `python tools/week3_checklist.py`（根目录） |
| "差一条就明天吧" | 欠账跨周利滚利 | 弹性 1.5h 就是用来补的，别留到 W4 |

---

## 6. 完成标准自测（先默写再对答案）

**规划题（周自检 §5 已覆盖，此处加收官三题）**：

1. **元笔记五块各回答什么问题？**
   *答案要点*：① FA 总表——带数字的知识清单；② 对标表+三层差距清单——与工业界差多远、怎么追；③ FA 战场边界——本周最反直觉的结论（适用区间思维第三次落地）；④ 串联表——新知识与旧笔记的换乘图；⑤ 工业趋势三行——FlexAttention/MLA/GQA 各接哪条线。
2. **"复现 or 不复现"为什么两条路都写结论？**
   *答案要点*：复现 = 方法泛化（Preliminary 升级 2 算子）；不复现 = 模型缺 softmax 特定项，是"模型进化清单"的第 1 条——判决标准提前定死，今天只读数归档，不存在"结果好不好"。
3. **W4 的融合纪律是什么？**
   *答案要点*：相邻 2–3 个算子（fused QKV / fused FFN = 标准算子级融合）；超过即入师兄（AMK 巨核）地盘，停手——W3 Day4 的边界纪律延续。

---

## 7. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] `tech_notes/week_summer3_view.md`（五块内容，闭卷默写后核对完成）
- [ ] `research/findings_w3.md`（对标表 + 三层差距清单 + 第二算子判决 + 反向记录）
- [ ] `research/proposal_v1.md` 更新（Preliminary 2 算子 + Plan 下一步 + Threats 增补）
- [ ] 欠账清零（checklist 全绿）+ GitHub ≥5 次有意义 commit
- [ ] 周复盘三行（最有效动作 / 最大阻塞 / W4 砍什么）

**明日预告（W4 Day 1）**：算子融合深水——fused QKV（三次小 GEMM 合成一次）+ fused FFN/SwiGLU。本周的 cp.async（差距清单可追项第 1）在 W4 的融合 kernel 里动手兑现。**W3 的"论文吞噬机"能力（精读→手推→参考→翻译→对标→归因）是 W4 全部工作的默认流程。**

---

## 附 A：术语速查表（Day 7）

| 名词 | 一句话解释 |
|---|---|
| 元笔记五块 | FA 总表 / 对标+差距清单 / 战场边界 / 串联表 / 工业趋势三行 |
| 文档长大论 | proposal 不重写、只长大——今天填第二算子的数据 |
| 复现判决 | 第二算子命中率 vs 第一算子：同档=泛化，更差=模型进化清单第 1 条 |
| FlexAttention | PyTorch 组合式 attention DSL（mask_mod/score_mod），已与 FA4 合流 |
| MLA（多头潜在注意力） | DeepSeek 的 KV 低秩压缩——KV Cache 管理的进化形态，接 W5 |
| fused QKV | 三次小 GEMM 合成一次大 GEMM（W4 主菜，vLLM 有同款工业实现） |
| 融合深度纪律 | 相邻 2–3 个算子；超过即入师兄地盘，停手 |
| 欠账利滚利 | 跨周欠的债拖慢后续每一周——今天清零 |
| 周复盘三行 | 最有效动作 / 最大阻塞 / 下周砍什么 |

## 附 B：参考与延伸

- PyTorch 官方博客：FlexAttention + FlashAttention-4（既快又灵活，趋势的权威出处）：https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- PyTorch 2.9 FlexAttention 优化实践（Intel GPU 上的工程化进展）：https://pytorch.org/blog/pytorch-2-9-flexattention-optimization-practice-on-intel-gpus/
- vLLM PR：融合 q/k/v 与 gate/up 线性组（fused QKV 的工业证据）：https://github.com/vllm-project/vllm/pull/45284
- W2 Day7 笔记（元笔记方法论 / 文档长大论 / GitHub 纪律）
- W7 融合两笔账笔记（W4 预热的重读对象）
