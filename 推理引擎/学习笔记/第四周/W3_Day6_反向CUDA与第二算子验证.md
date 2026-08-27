# 阶段二 · W3 Day 6 学习笔记 —— 反向 CUDA 实现（两 pass 重算）

> **对应规划**：`阶段二W3_FlashAttention_逐日详细规划.md`（v1.1）→ W3 Day 6（9/1 周一）
> **今日目标**：回答"把昨天的母本翻译成 CUDA，哪些归约能复用前向、哪些是新东西？"——两 pass 重算的反向 kernel + 与 autograd 三方对齐。
> **今日定位**：本周最后一个 kernel。**性能不追（校准二）**：跑通即胜利；降级触发器照旧（卡壳 1.5h → torch 参考版 + 推导笔记交付）。
> **前置**：Day 5 的 torch 反向母本（数学已被 autograd 证实）、前向 v0/v1 的骨架与 reduce.cuh 契约。

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `csrc/flash_attn_bwd.cu` | 反向两 kernel（pass1 重算+存状态 / pass2 重算+累加）+ wrapper | 复制进 `推理引擎/csrc/` |
| `tests/test_flash_attn_bwd.py` | 反向 CUDA 三方测试（kernel vs 母本 vs autograd） | 复制进 `推理引擎/tests/` |
| `bench/validate_cost_model_softmax.py` | 研线：cost model softmax 验证（Spearman/top-1/判决） | 复制进 `推理引擎/bench/` |
| `bench/ref_attn.py` + `bench/ref_attn_backward.py` | Day1/5 母本副本（独立自测用） | — |

---

## 0. 今天的问题与全景图

### 0.1 翻译总图：哪些复用、哪些新（今天的主线答案）

| Day5 母本的结构 | CUDA 的载体 | 复用 or 新 |
|---|---|---|
| pass1 重算前向（单因子公式） | kernel B1：**前向 v0 骨架原样复用** + 五处存储 | 复用 |
| 存 O / m_global / l_global / m̃_每块 | B1 的 global 写（thread0） | 复用写回套路 |
| `D_i = rowsum(dO⊙O)` | B1 收尾：warp0 lane0 串行累加（先对） | 新（但结构同最终归约） |
| pass2 逐块重算 S、P̃、P | kernel B2：与 B1 相同的块循环 | 复用 |
| `dp = dO·Vᵀ`（每行每元素） | 新：dO 行进 shared，每线程一个点积 | 新 |
| `dQ += scale·(ds@K)` | dq_acc[64] 列分片 + 两级归约（**前向 acc 的兄弟**） | 复用 |
| **`dK/dV += ...`（跨行累加！）** | **atomicAdd**——本周第一个"多行写同一位置"的并行结构 | **全新** |

### 0.2 两个简化决策（v0 反向的诚实取舍，写进笔记）

1. **O 不落 global**：FA2 里 forward 存 O、backward 用 O 算 D_i；我们的 pass1 **自己重算**了 forward，所以 O 在寄存器里算完**当场**折进 D_i，省掉一整块显存与一轮读写——重算哲学的第二次收益（第一次是省 S/P）。
2. **l̃（每块行和）用不上，直接不存**：重建真权重 P = p̃·exp(m̃−m_global)/l_global，只依赖 **m̃ + 全局 (m, l)**——Day 5 笔记写"存 m/l 每块"里的 l̃ 实际多余，v0 反向把它省了。**母本的每一行都要问"真的需要吗"，这是今天学到的第三个翻译习惯。**

---

## 1. kernel B1（pass1）：前向骨架 + 五处存储 + D_i（配套 `csrc/flash_attn_bwd.cu`）

```cuda
// 与前向 v0 逐段同构：一个 block 一行、BLOCK_N 线程、单因子更新、契约广播。
// 新增五处：
//   ① 每块结束：m̃、l̃（v0 简化：l̃ 省了，只存 m̃）→ mblk_store
//   ② 循环结束：m_global、l_global → m_store/l_store
//   ③ 收尾：O 的两级归约后，warp0 lane0 串行累加 D_i = Σ_d dO_r[d]·O_d → D_store
```

```cuda
// 收尾段（前向 v0 的最终归约 + D_i 的新尾巴）
#pragma unroll
for (int d = 0; d < BLOCK_D; ++d) o_acc[d] = warp_reduce_sum(o_acc[d]);
__shared__ float o_s[BLOCK_N / 32][BLOCK_D];
const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
if (lane == 0) {
    #pragma unroll
    for (int d = 0; d < BLOCK_D; ++d) o_s[wid][d] = o_acc[d];
}
if (threadIdx.x == 0) sm_l = l;
__syncthreads();

float D_i = 0.f;                                   // ★ 新尾巴：先对（串行），并行版是 pass2 的活
if (wid == 0) {
    for (int d = 0; d < BLOCK_D; ++d) {
        float vv = (threadIdx.x < BLOCK_N / 32) ? o_s[threadIdx.x][d] : 0.f;
        vv = warp_reduce_sum(vv);
        if (lane == 0) D_i += dO_r[d] * (vv / sm_l);    // O_d = vv/l；乘 dO 累加
    }
}
if (threadIdx.x == 0) {
    const size_t r_idx = (size_t)bh * Tq + row;
    D_store[r_idx] = D_i;
    m_store[r_idx] = m;
    l_store[r_idx] = l;
}
```

**D_i 为什么在 pass1 算**：D_i = rowsum(dO⊙O) 只需要 O 与 dO——两者在 pass1 结束时都在手边。在 pass2 算也行，但 pass2 的块循环里每行会反复需要它，**放 pass1 一次算好、pass2 读一次**，是"提前把行标量备齐"的调度习惯（与 Day5 母本的顺序一致）。

---

## 2. kernel B2（pass2）：重算 + 累加（今天的正主）

```cuda
// 块结构同 B1（一个 block 一行）。每个线程 = S 的一列（t = threadIdx.x）。
const float D_i = D_store[r_idx];     // 行标量：进寄存器
const float m_g = m_store[r_idx];
const float l_g = l_store[r_idx];

__shared__ float k_s[BLOCK_N][BLOCK_D], v_s[BLOCK_N][BLOCK_D];
__shared__ float dO_s[BLOCK_D];       // ★ 新：上游梯度行进 shared（每线程点积都要全行）

dO_s[threadIdx.x] = dO_r[threadIdx.x];   // 合并加载
__syncthreads();

float dq_acc[BLOCK_D];                // ★ 前向 acc 的兄弟：列分片累加 dq
#pragma unroll
for (int d = 0; d < BLOCK_D; ++d) dq_acc[d] = 0.f;

for (int jb = 0; jb < nblocks; ++jb) {
    // 加载 k_s/v_s（与 B1 相同）...
    __syncthreads();

    const float m_j = mblk_store[r_idx * nblocks + jb];   // 本块行 max（全体一致读）

    float p = 0.f, ds = 0.f;
    if (threadIdx.x < n) {                          // 越界列三件套（Day2 的坑）
        // 重算 S（与 B1 相同的点积 + offset 掩码）
        float s = ...;
        if (causal && (col0 + threadIdx.x) > (row + (Tk - Tq))) s = -INFINITY;

        // ★ 重建真权重 P（Day5 §1.4 的全局换算 + 全掩块 NaN 防护）
        const float m_safe = (m_j == -INFINITY) ? 0.0f : m_j;  // 行均匀分支 → 无发散
        const float p_tilde = __expf(s - m_safe);              // 全掩行 → exp(−inf−0)=0
        const float scale_p  = __expf(m_j - m_g) / l_g;        // 全掩行 → exp(−inf−m)/l=0
        p = p_tilde * scale_p;

        // dp_t = Σ_d dO_s[d]·v_s[t][d]（Day5 公式②）
        float dp = 0.f;
        #pragma unroll
        for (int d = 0; d < BLOCK_D; ++d) dp += dO_s[d] * v_s[threadIdx.x][d];
        ds = p * (dp - D_i);                       // ★ D 修正（Day5 公式③）

        // dq 列分片累加（Day2 越界读坑的三件套第三件：≥n 的线程不碰 shared）
        #pragma unroll
        for (int d = 0; d < BLOCK_D; ++d) dq_acc[d] += ds * k_s[threadIdx.x][d];

        // ★ dK/dV：跨行累加 → atomicAdd（本周的全新结构）
        const size_t kv_col_off = kv_base + (size_t)(col0 + threadIdx.x) * BLOCK_D;
        #pragma unroll
        for (int d = 0; d < BLOCK_D; ++d) {
            atomicAdd(&dk[kv_col_off + d], ds * qr[d] * scale);   // ★ scale 反传（Day5 抓的坑）
            atomicAdd(&dv[kv_col_off + d], p * dO_s[d]);
        }
    }
    __syncthreads();
}

// 收尾：dq_acc 两级归约 ×scale 写回（与 B1 的 O 归约同构，多一个 scale）
```

### 2.1 全新结构：atomicAdd 与它的诚实代价

**为什么 dK/dV 必须 atomicAdd**：dK 的一个元素会被**多行**累加（每行 Q 都对同一个 K 块贡献梯度）——B1/B2 是"一行一个 block"，多 block 同时写同一地址 = **数据竞争**。`atomicAdd`（原子加）保证每次加都完整生效。

**两个必须写进笔记的代价**：
1. **顺序不确定性（nondeterminism）**：多 block 的 atomicAdd 顺序由调度决定，逐次运行结果有 ~1e-6 的抖动——**这决定了测试容差必须是 1e-4 而不是 1e-6**（Day 1 的容差纪律今天有了新理由）；工业上（如 Thinking Machines 的工程博客所述）确定性需要专门的分块累加设计，那是大二上的活。
2. **性能**：每线程每块 64 次 atomicAdd，且全打向同一块地址区域——慢，但"先对"（校准二）。**v1 反向（分块累加 + 一次归并）记入大二上清单。**

### 2.2 m_safe 分支为什么不会发散

`m_safe = (m_j == -INFINITY) ? 0.0f : m_j` 里 m_j 对**同一行的所有线程相同**（全体读同一地址）→ 分支方向行内一致 → **warp 不发散**。这是"分支发散只发生在 warp 内判断不一致时"（Day 3 坑②）的镜像：**判断一致的"分支"其实免费**。

---

## 3. 三方对齐与验证（配套 `tests/test_flash_attn_bwd.py`）

```python
# 三方：CUDA 反向 vs Day5 母本 vs autograd
# 形状矩阵：prefill 对齐 / decode / 非对齐尾巴 × causal 开与关（D=64）
# 判据：allclose(atol=1e-4, rtol=1e-4) —— 容差含 atomicAdd 顺序抖动（§2.1）
# 跑法：进仓库后 python -m pytest tests/test_flash_attn_bwd.py -v
```

**wrapper 防御清单**：同前向（4D/fp32/连续/D=64）+ `dO.shape == q.shape`；输出三路 `torch::zeros_like`（**必须零初始化**——atomicAdd 在零上累加，忘了清零 = 随机垃圾）。

---

## 4. 【造】bindings + 可选 autograd Function 骨架（2.5h）

1. `bindings.cpp` 加：

```cpp
m.def("flash_attn_backward", &flash_attn_backward_cuda,
      "FlashAttention backward v0 (two-pass recompute)",
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("dO"), py::arg("causal") = true);
```

2. `kernels.py` 的 sources 加 `"flash_attn_bwd.cu"`；
3. 时间富余的可选加分：把前向+反向包进一个最小 `torch.autograd.Function`（`forward` 返回 `(O, 保存的状态)`，`backward` 调 `flash_attn_backward`）——**这是大二上训练路线的最小接口雏形**，[autograd.Function 文档](https://pytorch.org/docs/stable/notes/extending.html)就是模板。

---

## 5. 【研】cost model 第二算子验证（1.5h，论文线第 6 步）

配套 `bench/validate_cost_model_softmax.py`：对两卡 softmax 扫描数据算 **Spearman / top-1 命中率 / top-1 损失**，与 Day 6 判决标准对照。**今天的核心问题：第二算子是否复现第一算子（rmsnorm）的命中率？**

```
能复现（同档命中率） → 方法泛化，论文 Preliminary 从"1 个算子"升级为"2 个算子"
不复现            → 模型缺 softmax 的特定项（如两遍归约的 η）——这本身是发现，
                    写进 findings 的"模型进化"清单
```

两条路都是论文素材——**判决标准已提前定死，只需如实读数**。

---

## 6. 常见错误与调试速查表（Day 6 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| dk/dv 每次跑都差一点 | atomicAdd 顺序不确定性（预期内） | 容差 1e-4；确定性版本记大二上清单 |
| dk/dv 全错且无规律 | 输出忘了 `zeros_like`（atomicAdd 在垃圾上累加） | wrapper 三路零初始化 |
| dq 差一个 scale（≈8 倍） | scale 反传漏了（Day5 的坑在 CUDA 里复发） | ② 公式两处 ×scale（dq 收尾、dk atomic） |
| 全掩块行 NaN | m_safe 分支写反/漏了 | §2.2 的行均匀分支；对拍非对齐用例 |
| 越界列贡献 NaN | ≥n 的线程读了 k_s 未初始化行 | 三件套守卫（dq_acc 也要包在 if 里） |
| D_i 不对 → 全部梯度差一个行常数 | pass1 的 D_i 用了未归约的 O | 两级归约后、除 l 后再算 D_i |
| 编译报 reduce 缺函数 | Day1 的 max 归约没合并进 reduce.cuh | 合并 reduce_max.cuh 再编 |

---

## 7. 完成标准自测（先默写再对答案）

**规划题**：能说清 pass1/pass2 各自存什么、重算什么；D_i 在哪个 pass 算、为什么。
*答案要点*：pass1 = 重算前向（复用 v0 骨架），存 **m̃（每行每块）、m_global、l_global、D_i**（O 不落 global——算完当场折进 D_i；l̃ 用不上省了）；pass2 = 逐块重算 S、P̃ → 全局换算重建真权重 P → ds = P(dp−D_i) → dq 列分片 + dK/dV atomicAdd 跨行累加。**D_i 在 pass1 算**：只需 O 与 dO，pass1 结束时都在手边，一次算好、pass2 只读一次。

**附加题**：
1. 哪些归约复用前向？（max/sum 的 block 归约、acc 列分片、两级归约——三件套第三次出场）
2. 哪些是全新结构？（dO 行进 shared、atomicAdd 跨行累加、m_safe 行均匀分支）
3. atomicAdd 的两个诚实代价？（顺序不确定性→容差 1e-4；性能→v1 分块累加记大二上）

---

## 8. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] `csrc/flash_attn_bwd.cu`（B1+B2+wrapper，防御清单齐全）
- [ ] 反向 CUDA 三方对齐（vs Day5 母本 vs autograd，形状矩阵 × causal）
- [ ] cost model softmax 验证三指标 + 判决对照（第二算子复现 or 不复现都入账）
- [ ] （可选）autograd Function 骨架

**明日预告（Day 7）**：元笔记 + 论文线收口——FA 总表、三层差距清单、"FA 战场边界"（prefill/decode 分支 = 适用区间思维第三次落地）、工业趋势三行（FlexAttention/MLA/GQA）；findings_w3 + proposal 更新为"2 个算子"。**W3 收官，本周所有降级路径的产出都要在明天归档。**

---

## 附 A：术语速查表（Day 6）

| 名词 | 一句话解释 |
|---|---|
| 两 pass 重算 | pass1 重算前向存状态；pass2 再重算 S/P 并累加梯度 |
| D_i（行标量） | rowsum(dO⊙O)：pass1 一次算好，pass2 只读 |
| atomicAdd（原子加） | 多 block 写同一地址的累加原语——dK/dV 跨行累加的唯一正确途径 |
| 顺序不确定性（nondeterminism） | atomicAdd 次序由调度决定 → 每次结果 ~1e-6 抖动 → 容差 1e-4 |
| 行均匀分支 | m_j 对整行相同 → 分支方向一致 → 免费（无发散） |
| 全局换算 | p = p̃·exp(m̃−m_g)/l_g：重建真权重的三步（Day5 §1.4 的 CUDA 版） |
| scale 反传 | dQ/dK 公式里的 ×scale——Day5 抓过的坑在 CUDA 的复发点 |
| 列分片累加（dq） | dq_acc[64]：前向 acc 的兄弟结构，第三次出场 |
| autograd Function 骨架 | 前向+反向打包成可训练接口的最小雏形（大二上路线） |

## 附 B：参考与延伸

- dao-ailab flash-attention 的 Backward Pass 实现（工业结构）：https://deepwiki.com/dao-ailab/flash-attention/7.2-backward-pass-algorithm
- Thinking Machines：消除 LLM 推理中的不确定性（atomicAdd 顺序的工业讨论）：https://www.mbgsec.com/_weblog/2025-09-11-defeating-nondeterminism-in-llm-inference-thinking-machines-lab/
- torch.autograd.Function 文档（可选加分项的模板）：https://pytorch.org/docs/stable/notes/extending.html
- Day 5 笔记（反向母本与四公式）、Day 2 笔记（越界读三件套）、Day 3 笔记（行均匀分支的镜像）
