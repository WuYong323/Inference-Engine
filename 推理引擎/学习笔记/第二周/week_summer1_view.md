# 阶段一 · W1 Day 7 —— W1 元笔记：Triton 帮我隐藏了什么 + 三个杠杆 + 复盘

> **今天的一句话**：W0 你搭好了发射台，W1 你亲手钻进了 Triton 的黑箱。七天之后，"Triton 一行代码底下是什么"从一个哲学问题，变成了你能拿代码指着说的具体事实。**这一周的价值不在于你写了五个 kernel，而在于你终于看清了"写 kernel"这件事的全貌——以及，看清了工业级优化师们（cuBLAS、vLLM、FlashAttention、AMK）每天在做什么。**
>
> **本笔记的三份产出**：
> 1. 本文（元笔记）：一张总表 + 三个杠杆 + 一条因果链，串起 Day1–Day6 的全部单点。
> 2. `S1_W1_review.md`：本周最有效动作 / 最大阻塞 / W2 承接点（见文件末尾）。
> 3. 五个 CUDA kernel 的 GitHub commit（已在 Day1–Day6 陆续提交，今天收口）。

---

## §1 本周元笔记是什么——以及为什么不是"周记"

延续 W4 Day7 的传统，先立住元笔记的定义：**元笔记是"把一周的单点串成网"，不是"记录本周做了什么"**。

- **单词本 vs 词族图**：普通笔记是"今天学了 warp，明天学了 bank conflict"；元笔记是"warp、bank conflict、树形归约、tiling、launch 开销——它们共同指向**三个性能杠杆**，而这三个杠杆就是所有 GPU 优化的通用语言"。
- **为什么必须写**：你下周（W2）就要用 CUDA 重写 RMSNorm。如果你脑子里只有"我写过树形归约"，你会卡在"怎么把它变成 RMSNorm"；如果你脑子里有"RMSNorm = 树形归约求平方和 + 广播缩放"，你直接就能写。**元笔记就是把"知识"变成"可调用代码"的那一步。**

---

## §2 CUDA 编程模型总表：Triton 帮我隐藏了什么（★ 本周核心）

这是本周最该沉淀下来的东西。你 W8 天天写 `tl.sum`、`tl.load`、`tl.dot`，现在你知道它们底下是什么了。

### §2.1 总表：从 CUDA 概念到 Triton 抽象

| CUDA 概念（我这周亲手管的） | Triton 里的样子 | Triton 替我省了什么 | 代价（失去的控制权） | 对应性能杠杆 |
|---|---|---|---|---|
| **线程层级**<br>grid → block → thread → warp | 一个 program 对应整个 block<br>（`num_warps` 控制 warp 数） | 索引拆分（`blockIdx*blockDim+threadIdx`）、warp 内分支的手动避免 | 无法做 warp specialization（不同 warp 干不同活） | **占用率 + 延迟隐藏** |
| **内存层级**<br>global / shared / register | `tl.load` / `tl.store` 自动管理<br>（通常只操作 global，shared 由编译器决定） | 手动搬数据、手动管 shared 生命周期 | 无法手工 prefetch、double buffering | **访存合并 + 数据复用** |
| **同步机制**<br>`__syncthreads()` | `tl.sum` 等自动插入<br>编译器保证正确性 | 手动放栅栏、手动防 race condition | 无法优化同步位置（如去掉不必要的同步） | **协作正确性** |
| **归约操作**<br>树形归约 / warp shuffle | `tl.sum(x, axis=0)` 一行 | 手写折半循环、shuffle 指令、bank 布局优化 | 归约算法不可定制（如要分段归约） | **归约效率** |
| **矩阵乘 tiling**<br>shared memory 分块 + 双缓冲 | `tl.dot` 自动分块、自动用 Tensor Core | 手动搬 tile、手动算索引、手写双缓冲 | Tile 大小不可精细控制（受 BLOCK 限制） | **数据复用（算术强度）** |
| **边界处理**<br>`if (idx < n)` / predicate | `mask=offs<n` 自动处理 | 手写边界判断、手动处理边角块 | 无法针对边界做特殊优化 | **代码简洁性 vs 性能** |

> **一句话总结**：Triton 是"编译器替你写 CUDA"，它把**正确性负担**（同步、边界、索引）和**常见优化**（合并访问、归约、tiling）自动化了。但它不能违反硬件规律，也不能做**跨算子融合**和**warp specialization**——后者正是 AMK 巨核要解决的。

### §2.2 三个杠杆：GPU 优化的通用语言

这一周你学的所有概念，最终都指向**三个可拧的旋钮**。任何 kernel 慢，先问这三个问题：

| 杠杆 | 是什么 | 怎么拧 | 本周证据 |
|---|---|---|---|
| **① 占用率 / 延迟隐藏** | GPU 同时放多少个 warp 来"填"延迟空隙 | 调 block 大小、减少寄存器/spill、增加并行任务 | Day1：divergence_heavy vs light（分支让 warp 闲置）<br>Day5：matmul 的 occupancy 25% 但吞吐高（ILP 路线） |
| **② 访存合并（带宽利用率）** | 一次访存事务搬多少有效字节 | 连续访问、对齐、向量化（float4）、布局转换（SoA） | Day2：coalesced vs strided，带宽差 10 倍<br>Day5：matmul 中 global 访问的合并优化 |
| **③ 数据复用（算术强度）** | 搬一次数据，算多少次 | tiling（shared memory 缓存）、寄存器分块（register tiling） | Day5-6：tiled matmul 把 AI 从 0.25 提到 8-32<br>Day4：树形归约在 RMSNorm 场景的收益 |

> **这三个杠杆的优先级**：先判 bound（计算还是访存）→ 如果是访存，先拧 ②（合并）→ 再拧 ③（复用）→ 最后看 ①（占用率）。**顺序错了就是白干**（Day4 的"厚归约优化无效"就是顺序错的教训：先优化归约算法，但没先确认瓶颈是不是归约）。

### §2.3 一条因果链：从"写 kernel"到"写巨核"

这是本周所有知识的地图，也是你对接课题主线 3（巨核/kernel）的桥梁：

```
写 CUDA kernel 的完整心智模型（W1 建立）
│
├─ 线程层：grid/block/thread/warp 是什么关系？
│   └─ Day1：SIMT、divergence、occupancy 是什么
│
├─ 内存层：数据在哪、怎么搬？
│   └─ Day2：global 慢、shared 快、register 最快
│       └─ Day2 实测：合并访问差 10 倍带宽
│
├─ 协作层：线程之间怎么安全地共享数据？
│   └─ Day3：shared memory + __syncthreads() 的栅栏语义
│       └─ Day4：树形归约（无 divergence 的协作）
│           └─ Day4：bank conflict（shared 的隐藏规则）
│
└─ 计算层：怎么让计算密度（AI）足够高？
    └─ Day5-6：tiling（分块）→ 复用数据 → 提高算术强度
        └─ Day6：寄存器分块（register tiling）→ 进一步复用
            └─ Day6：Tensor Core（换更高的计算屋顶）

到巨核（课题主线 3）的跃迁：
├─ 把多个 kernel 融成一个（省 launch 开销、省中间张量访存）
├─ 跨 SM 同步（grid-level barrier，比 block 级贵 3 个数量级）
└─ Warp specialization（不同 warp 干不同活，Triton 做不到）
```

---

## §3 本周代码资产清单：五个 kernel 的完整图谱

这是本周你亲手写的、可复用的代码。每个都对应一个真实场景，不是玩具。

| 文件 | 核心概念 | 工业对应 | 已验证的正确性 |
|---|---|---|---|
| `01_vector_add.cu` | 线程模型、索引、边界判断 | 所有 element-wise 算子的骨架 | 与 PyTorch `allclose` 对齐 |
| `02_memory_access.cu` | 合并访问、strided vs coalesced | 数据布局优化（SoA vs AoS）、KV Cache 布局 | 带宽差 10 倍实测 |
| `03_block_reduce.cu` | shared memory、`__syncthreads()`、规约求和 | RMSNorm / LayerNorm 的归约核心 | 与 PyTorch `.sum()` 对齐 |
| `04_tree_reduction.cu` | 树形归约、warp shuffle、bank conflict | 高效归约（vLLM 的 `blockReduceSum` 原型） | 7 种实现对比、ncu 验证 |
| `05_tiled_matmul.cu` | tiling、寄存器分块、Tensor Core 入口 | cuBLAS / CUTLASS 的简化版、FlashAttention 的 GEMM 部分 | 与 cuBLAS 对标、差距分析 |

> **W2 的直接起点**：`04_tree_reduction.cu` 里的 `block_reduce_sum` 就是 W2 CUDA RMSNorm 的心脏。你今天要做的不是"再写一个 kernel"，而是**把它改写成 RMSNorm**（见 §5 的承接点）。

---

## §4 对标工业：你的代码和真实世界差多远

这是本周最诚实、也最有价值的一部分。你写的 kernel 大概率打不过 cuBLAS，但**知道差在哪，就是最大的收获**。

### §4.1 对标 cuBLAS：tiled matmul 的四层差距

你的 `05_tiled_matmul.cu` 和 cuBLAS 的差距，不是一个 bug，而是**四层有名字的优化**：

| 层级 | 你做了什么 | cuBLAS 多做了什么 | 为什么有效 | 难度 |
|---|---|---|---|---|
| **L0** | 朴素 global 访问 | shared memory tiling | 复用数据，AI 0.25 → 8 | ★★ |
| **L1** | 1 线程 1 输出 | 二维寄存器分块（8×8） | 发射槽占比 33% → 80%，AI → 32 | ★★★ |
| **L2** | 简单搬运 | double buffering / `cp.async` | 搬运和计算重叠，GPU 不空转 | ★★★★ |
| **L3** | FP32 CUDA core | Tensor Core（WMMA / MMA） | 计算屋顶从 60 TFLOP/s 提到 1000+ TFLOP/s | ★★★★ |

> **关键洞察**：L0 和 L1 拿走了 80% 的收益（30-50 倍提升），L2 和 L3 是剩下的 20%（再提 1.5-2 倍）。**这就是"优化的二八定律"**——你知道自己站在曲线的哪一段，比闷头追 cuBLAS 重要。

### §4.2 对标 vLLM：你的 block_reduce 和 vLLM 的 blockReduceSum

你写的 `04_tree_reduction.cu` 里的 `block_reduce_sum`，和 vLLM 的 `blockReduceSum` 在结构上几乎一样：

```cpp
// vLLM 的实际代码（简化）
template<typename T, int numLanes = WARP_SIZE>
__inline__ __device__ T blockReduceSum(T val) {
    static __shared__ T shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE, wid = threadIdx.x / WARP_SIZE;
    val = warpReduceSum(val);              // ← 你 Day4 学的 shuffle
    if (lane == 0) shared[wid] = val;
    __syncthreads();                        // ← 你 Day3 学的栅栏
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : (T)0.0f;
    val = warpReduceSum(val);
    return val;
}
```

**你的代码和 vLLM 的差距**：
- vLLM 用 `warpReduceSum`（shuffle），你用的是 shared memory 树形——**shuffle 更快，但你的版本更通用**（适合理解原理）。
- vLLM 处理了边界（`threadIdx.x < blockDim.x / WARP_SIZE`），你的版本假设 blockDim 是 32 的倍数。
- vLLM 是模板函数，你的版本是硬编码 256 线程——**模板化是工业代码的标配**。

> **这个对标的价值**：你写的不是"玩具代码"，而是**生产级代码的原型**。vLLM 的 `blockReduceSum` 被调用在 `rms_norm_kernel`、`layernorm_kernel`、`softmax` 里——全是推理引擎的热点。

### §4.3 对标 FlashAttention：你的 tiling 和 FlashAttention 的 tiling

你 Day5-6 写的 tiled matmul，和 FlashAttention 的 tiling 是**同一个思想的不同应用**：

| 维度 | 你的 tiled matmul | FlashAttention |
|---|---|---|
| **分块对象** | A 和 B 矩阵 | Q、K、V 矩阵 |
| **分块目的** | 减少 global 访存（提高 AI） | 减少 HBM 访存（避免存 T×T 矩阵） |
| **复用次数** | TILE 次（shared memory 复用） | 每个 K/V 块被所有 Q 复用 |
| **核心技巧** | 二维寄存器分块 | online softmax（增量更新 max 和 sum） |
| **瓶颈** | shared memory 带宽、发射槽 | HBM 带宽、跨 SM 同步 |

> **直通 W3**：FlashAttention 的"分块"就是 tiling，"online softmax"就是把两次归约合成一次（Day4 的归约知识 + 增量更新）。你 W3 要做的就是把这个思想从 matmul 迁移到 attention。

---

## §5 对接课题主线 3：巨核/kernel 的地基

这是本周学习的最终目的——为小米课题主线 3（巨核）打地基。

### §5.1 巨核是什么，为什么需要它

**巨核（megakernel）= 把多个算子融合成一个 kernel**。为什么需要它？

- **省 launch 开销**：每个 kernel launch 有 3-5 µs 固定开销（Day2 实测）。一个 decoder layer 有 10+ 个 kernel，光 launch 就 30-50 µs。
- **省中间张量访存**：算子 A 的输出是算子 B 的输入，如果 A 和 B 是分开的 kernel，中间结果要落 HBM 再读回来。融合后，中间结果直接在 register/shared 里传递。
- **省同步开销**：多个 kernel 之间的同步是 GPU 级别的（要等所有 block 完成），融合后变成 block 内同步（`__syncthreads()`）。

### §5.2 巨核的代价：跨 SM 同步

巨核的代价是**跨 SM 同步（grid-level barrier）**：

| 同步层级 | 机制 | 成本 | 作用域 |
|---|---|---|---|
| warp 内 | `__syncwarp()` / shuffle | ~几个周期 | 32 线程 |
| block 内 | `__syncthreads()` = `bar.sync` | ~几十周期 | ≤1024 线程，同一 SM |
| **grid 级** | `cg::this_grid().sync()` | **~微秒级**（要走 global memory 原子计数 + L2 往返） | 全部 SM |

**差了三个数量级**。H100 有 132 个 SM，grid 级 barrier 要等**最慢的那个 SM**——SM 越多，"最慢的那个"的期望值越差（统计上的必然）。

### §5.3 巨核的实现技巧：warp specialization

巨核里不同 warp 要干不同活（比如 warp 0-3 搬数据、warp 4-7 算矩阵乘），这叫 **warp specialization**。Triton 做不到这一点（它假设 block 内所有 warp 做同样的事），所以巨核必须手写 CUDA。

**你本周学的知识如何支撑巨核**：
- **Day1 的 warp 模型**：让你理解 warp specialization 的可行性。
- **Day3 的 `__syncthreads()`**：让你理解 block 内同步，这是 warp specialization 的协作基础。
- **Day4 的归约**：让你理解巨核里的归约操作（如 softmax 的求和）。
- **Day5-6 的 tiling**：让你理解巨核里的矩阵乘怎么分块。

---

## §6 本周的三个杠杆：怎么拧，拧到什么程度

这是本周最实用的部分——遇到任何 kernel 慢，先问这三个问题。

### §6.1 杠杆 ①：占用率 / 延迟隐藏

**是什么**：GPU 同时放多少个 warp 来"填"延迟空隙。延迟（latency）是单个操作的等待时间，吞吐（throughput）是单位时间完成的操作数。**GPU 的强项是吞吐，弱项是延迟**。

**怎么拧**：
- **提高 occupancy**：增加 block 数量、减少寄存器用量、减少 shared memory 用量。
- **增加 ILP（指令级并行）**：让一个 warp 内有更多独立指令（Day6 的 64 条独立 FFMA）。

**拧到什么程度**：
- **计算密集型 kernel**（如 matmul）：ILP 路线，occupancy 可以低（25% 也正常）。
- **访存密集型 kernel**（如 copy、reduce）：TLP 路线，occupancy 要高（>50%）。

**本周证据**：
- Day1：divergence_heavy vs light（分支让 warp 闲置，occupancy 下降）。
- Day6：matmul 的 occupancy 25% 但吞吐高（ILP 路线）。

### §6.2 杠杆 ②：访存合并（带宽利用率）

**是什么**：一次访存事务搬多少有效字节。GPU 访存是以 **sector（32 字节）** 为单位的，如果一个 warp 的 32 个线程访问连续的 32 个 float（128 字节），硬件能合并成 4 个 sector 事务；如果访问分散的 32 个地址，就变成 32 个独立事务。

**怎么拧**：
- **连续访问**：让 warp 内线程访问连续地址。
- **对齐**：让数据结构按 128 字节对齐。
- **向量化**：用 `float4` 一次读 16 字节。

**拧到什么程度**：
- **目标**：让 DRAM 吞吐接近峰值（H100 约 3 TB/s）。
- **判断标准**：`ncu` 的 `dram__throughput.avg.pct_of_peak_sustained_elapsed`。

**本周证据**：
- Day2：coalesced vs strided，带宽差 10 倍。
- Day5：matmul 中 global 访问的合并优化。

### §6.3 杠杆 ③：数据复用（算术强度）

**是什么**：搬一次数据，算多少次。算术强度（AI）= FLOP / Byte，是 Roofline 模型的横轴。

**怎么拧**：
- **tiling（分块）**：把数据搬进 shared memory，复用 TILE 次。
- **寄存器分块**：让一个线程算多个输出，复用寄存器里的数据。

**拧到什么程度**：
- **目标**：让 AI 超过 ridge point（H100 约 18 FLOP/Byte）。
- **判断标准**：`ncu` 的 Roofline 图，看 kernel 落在哪个区域。

**本周证据**：
- Day5-6：tiled matmul 把 AI 从 0.25 提到 8-32。
- Day4：树形归约在 RMSNorm 场景的收益。

---

## §7 本周知识串联表：从 W0 到 W1 到 W2

| 本周内容 | 关联已有笔记 | 关系 |
|---|---|---|
| **warp / SIMT / divergence** | W0 Day5 三个钩子问题 | W0 埋的第一个问题，W1 Day1 亲手实测回答 |
| **内存层级 + coalesced** | W7 Roofline / W0 memory-bound | W7/W0 从宏观知道"访存是瓶颈"，W1 下沉到微观机制 |
| **`cudaMemcpy` 开销** | W0 KV Cache 常驻显存 | W0 知道"要缓存"，W1 知道"为什么不能来回搬 CPU" |
| **树形归约** | W8 `tl.sum` | W8 会用它，W1 知道它底下是折半 + 同步 + shuffle |
| **tiling + 算术强度** | W7 Roofline / W0 Day6 FlashAttention 分块 | tiling 是 FlashAttention 分块的原型，W3 直接放大它 |
| **matmul 差距地图** | W8 "我 vs 官方"差距地图纪律 | 延续诚实纪律，把 kernel 级差距落到 4 个有名字的优化 |
| **CUDA RMSNorm 接口预留** | W0 Backend 抽象 | W0 建的可切换后端，W2 第一次插进真正的 CUDA 实现 |

---

## §8 自测题：能不看资料回答，才算真懂

1. **线程模型**：warp 是什么？为什么 warp 内分支会让性能腰斩？Triton 的一个 program 对应多少 warp？
2. **内存层级**：register / shared / global 的延迟量级是多少？为什么"减访存比减计算重要"？
3. **同步**：`__syncthreads()` 解决什么问题？漏掉它会发生什么？
4. **归约**：树形归约为什么比串行快？stride 为什么要从大到小？`tl.sum` 底下是什么？
5. **tiling**：朴素 matmul 的算术强度是多少？tiling 怎么提升算术强度？两个 `__syncthreads()` 各自的职责是什么？
6. **对标 cuBLAS**：你的 tiled matmul 和 cuBLAS 差在哪四层？每层为什么有效？
7. **对接巨核**：巨核为什么要做？跨 SM 同步为什么贵？warp specialization 是什么？

---

## §9 完成标准 checklist（对照产出逐条打勾）

- [ ] 能**不看资料**写出「树形归约」和「tiled matmul」两个 kernel 的骨架
- [ ] 能讲清本周三个杠杆：占用率/延迟隐藏、访存合并、数据复用——各是什么、怎么拧
- [ ] 能回答 W0 Day5 留的全部三个钩子问题（warp、memcpy、divergence），且每个都有亲手写的代码/实测数据支撑
- [ ] 五个 CUDA kernel 全部和 PyTorch/cuBLAS 数值对齐（divergence / coalesce / block_reduce / tree_reduce / tiled_matmul）
- [ ] 引擎脊柱没退化：test_generation 通过、tok/s 基线已记录、bench 脚手架就位、RMSNorm 接口为 W2 备好
- [ ] AMK：用 CUDA 眼光重读了 ncu，对"AMK 是 memory/compute-bound、跨 SM 同步的证据"有了带指标的假设
- [ ] `week_summer1_view.md` + `S1_W1_review.md`
- [ ] GitHub ≥3 次有意义 commit

---

## §10 一句话总结

> **W1 结束，你不再是"会用 Triton 写 kernel"，而是"理解 GPU 为什么快、慢在哪、怎么用三个杠杆去救"。这个理解，是你后面 7 周（FlashAttention、vLLM、量化、巨核）每一块的通用地基，也是你简历上"懂底层"最硬的那块证据。**

---

*生成于 2026-08-17 · 衔接 W0 发射周（引擎骨架 + CUDA 链路 + Day5 三个钩子）· 按 8 周全日制 + H100 全程可用 + 暑训固定半天设计*
