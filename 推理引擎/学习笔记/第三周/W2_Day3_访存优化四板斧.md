# 阶段一 · W2 Day 3 学习笔记 —— 访存优化四板斧：兑现 04 笔记留的 A/B 决策

> **对应规划**：`阶段一W2_CUDA进阶_手写算子接进引擎_逐日详细规划.md` → W2 Day 3（8/20 周四）
> **今日目标**：`__restrict__`、`#pragma unroll`、`float4`、寄存器暂存——这四板斧**各值多少钱**？用数据回答 W1 Day4 笔记交接单里的 A/B 决策（x 读两遍 vs 暂存寄存器）。
> **今日定位**：Day 2 的"算出来"今天变成"做出来再量出来"；84 点扫描数据同时服务【造】与【研】。
> **前置**：Day 2 的 occupancy 阈值表与 Little's Law 结论（162 KB 在途字节、BLOCK=1024 只有 66.7% occupancy）——今天每一步都从它们推导。

## 本文件夹内容说明（仿 Desktop\01 结构：笔记 + 代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记.md` | 本笔记 | — |
| `rmsnorm_variants.cu` | 三变体 kernel + wrapper + 分发表（完整可编译） | 复制进 `推理引擎/csrc/rmsnorm.cu` |
| `bindings.cpp` | 带 variant 参数的 pybind11 绑定 | 复制进 `推理引擎/csrc/bindings.cpp` |
| `restrict_probe.cu` | 板斧一/二 的 SASS 对照探针（独立编译） | 放 `推理引擎/tools/` |
| `sweep_rmsnorm.py` | 84 点扫描脚本（含耗时记录） | 放 `推理引擎/bench/` |
| `test_cuda_backend.py` | 48 用例 + 对齐回退专项测试 | 复制进 `推理引擎/tests/` |

---

## 0. 今天的问题与全景图

### 0.1 一个问题：四板斧各值多少钱？

昨天你在扫描图上看到了"最优 block 随形状变化"。但 kernel 本身的写法还有四个旋钮没动过：`__restrict__`、`#pragma unroll`、`float4`、寄存器暂存。每个技巧网上都有人吹"必用"，但**没有人告诉你它在你这个算子、你这张卡、这个形状下值多少**。今天自己量出来。

**衡量单位不是"快了多少倍"，而是 %peak 带宽的变化**（Day 2 定的尺子）——同一把尺子量四个技巧，结果才可互相比较、可跨卡迁移。

### 0.2 四板斧总览表（先建框架，今天逐格填数）

| 板斧 | 治什么瓶颈 | 机制一句话 | 预期收益 | 验证手段 | 实测（今天填） |
|---|---|---|---|---|---|
| ① `__restrict__` | 编译器不敢优化 | 承诺指针不重叠 → 允许读提前/缓存 | 0–10% | SASS 指令数对比 | |
| ② `#pragma unroll` | 循环控制开销、ILP 不足 | 展开循环体 → 省控制指令、多指令并行 | 小，常被 -O3 自动做 | 寄存器/指令对比 | |
| ③ `float4` | **在途字节不足**（Day 2 算出的临界） | 128-bit 访存 → 每线程在途 ×4 | **最大**（未贴峰的形状上） | 扫描对比 | |
| ④ 寄存器暂存 | 第二遍读 global | 用寄存器换掉第二次读 → 代价是 occupancy | H=4096 打平或输；H=8192 可能反转 | A/B 扫描 | |

### 0.3 与 Day 2 的因果链（背下来，今天所有预测从这里推）

1. Day 2 算过：5060 要 ~162 KB 在途字节；每线程 4 B → 40600 线程，全卡只有 39936——**差一点**。→ 板斧③ float4（16 B/线程）把这个缺口直接抹掉，所以它是四板斧里最值钱的。
2. Day 2 算过 occupancy 阈值：BLOCK=256 时 regs 43+ 掉到 5 块（83%）、52+ 掉到 4 块（66.7%）。→ 板斧②④ 都是"拿寄存器换东西"，**会不会掉档决定它们值不值钱**。
3. Day 2 的判据：%peak 已 >85% 的形状是 memory-bound，任何技巧都无效。→ 四个板斧的收益**只可能出现在还没贴峰的形状上**（decode 小形状），大形状上它们都是 0——这本身就是一个可预测、可验证的结论。

### 0.4 今日时间盒导航

| 时间块 | 内容 | 章节 |
|---|---|---|
| 【学】30+30 min | 板斧①② + SASS 对照 | §1 §2 |
| 【学】1h | 板斧③ float4 | §3 |
| 【学】1.5h | 板斧④ 暂存 vs 读两遍（正题） | §4 |
| 【造】2.5h | 三变体进引擎 + 48 用例全绿 | §5 |
| 【研】1.5h | 84 点扫描 + 搜索成本基线 + 交接单闭环 | §6 |

---

## 1. 板斧一：`__restrict__`（30 min）

### 1.1 是什么：给编译器的"不重叠"承诺

**别名（aliasing）**：两个指针指向同一块内存。C/C++ 里编译器默认**必须假设任意两个指针都可能别名**——因为 `f(x, x)` 是完全合法的调用。

这带来一个巨大的保守性约束：**编译器不敢把"读"提前到"写"之前**。看这个例子（rmsnorm 第二遍的形态）：

```cuda
// 无 restrict：编译器视角的"被迫保守"
for (int i = threadIdx.x; i < H; i += BLOCK)
    yr[i] = xr[i] * s * w[i];
// 编译器想优化成：先把 xr 的一串值读进寄存器缓存着，再逐个算、逐个写。
// 但它不敢：万一 yr 和 xr 是同一块内存呢？写 yr[i] 会改掉 xr[i+1] 还没读的值！
// 于是每次迭代的读都必须"老老实实"排队在写之后，寄存器缓存、指令重排全都受限。
```

`__restrict__` 就是你对编译器说的一句话：**"我保证这几个指针指向的内存互不重叠。"** 编译器拿到承诺后，可以把读提前、把值留在寄存器里、做向量化——很多优化只在无别名假设下合法。

### 1.2 怎么做：SASS 对照实验

配套文件 `restrict_probe.cu` 里有两组对照 kernel（两遍式 pattern + 单遍 pattern）。编译并看两样东西：

```powershell
# 在 build_env.bat 环境里
nvcc -arch=sm_120 -O3 -Xptxas -v -cubin restrict_probe.cu -o restrict_probe.cubin
cuobjdump -sass restrict_probe.cubin > restrict_probe.sass
```

- **`-Xptxas -v` 的输出**：对比 `pass_no_restrict` 与 `pass_restrict` 的 `Used N registers`。
- **`.sass` 文件**：对比两个函数的**指令条数**，重点看 LDG（全局读）的数量和位置——无 restrict 版本里，你经常能看到同一地址被重复读、或读指令被压到写指令之后。

> **预期收益 0–10%，但"0"也是今天的产出。** 如果两个版本 SASS 一模一样（nvcc 在简单 pattern 上有时能自己证明无别名），就把"在此 pattern 上 restrict 免费"写进笔记。**"知道某个技巧不值钱"和"知道某个技巧值钱"一样有价值**——这是负结果清单的第一条，也是你 cost model 的输入之一。

### 1.3 红线与工业惯例

- **restrict 是承诺不是魔法**：违反承诺（实际重叠却标了 restrict）= **未定义行为（undefined behavior）**，症状是"偶尔错、难复现"——和 Day 1 的流 bug 一样阴险。今天 x/y/w 三者确实不重叠，安全。
- **工业惯例**：cuBLAS、CUB、FlashAttention 的热点 kernel 里，**所有指针一律标 restrict**——它是"免费的性能保险"，标准写法里没有理由不用。

---

## 2. 板斧二：`#pragma unroll`（30 min）

### 2.1 是什么：把循环体抄 N 份

**循环展开（loop unrolling）**：把 `for (j=0; j<4; j++) ...` 写成 4 份顺序的循环体，去掉"j++、j<4 比较、跳转"这三件控制开销。更重要的收益是 **ILP（指令级并行，instruction-level parallelism）**：展开后的 4 条语句之间没有依赖，可以**同时发射**——一条指令的延迟被另外三条盖住。

### 2.2 为什么是 trade-off：拿寄存器换 ILP

展开后，编译器倾向于把 4 次迭代的中间结果**同时**留在寄存器里 → 寄存器用量上升 → 撞上 Day 2 的阈值表 → occupancy 掉档。**展开过头，反而变慢。**

对照实验也在 `restrict_probe.cu` 里（`reduce_unroll_off` vs `reduce_unroll_on`，后者 `#pragma unroll 4`）：编译后对比两个函数的寄存器数——这是"ILP 的代价"第一次在你眼前显形。

### 2.3 三个必须知道的现实

1. **nvcc 在 -O3 下已经会自己展开**已知次数的循环（尤其模板参数定的循环界）。`#pragma unroll` 的常见用途是**强制展开编译器犹豫的循环**或**指定部分展开因子**（`#pragma unroll 4` = 抄 4 份再循环）。
2. **板斧④里 `#pragma unroll` 不是优化项，是性能正确性的前提**：`buf[EPT]` 的下标必须是编译期常量才能映射到寄存器；不展开 → 整个 buf 溢出到 **local memory**（见 §4.3 的词条解释）→ 性能灾难。这个联动今天就能在 -Xptxas -v 的 spill 输出里看到。
3. 收益通常很小（个位数 %），**为 0 甚至为负（寄存器掉档）都不奇怪**——这正是"要实测不要迷信"的又一个样本。

---

## 3. 板斧三：`float4` 向量化（1h，最值钱的一板斧）

### 3.1 是什么：128-bit 访存指令

**float4** 是 CUDA 内置的向量类型：4 个 float 打包成一个 16 字节的整体。读一个 float4，编译器生成一条 **LDG.E.128**（128-bit 全局读指令）——一次从显存搬 16 字节到寄存器，而普通 float 读是 **LDG.E**（32-bit，一次 4 字节）。

**为什么这直接命中 Day 2 的结论**：Little's Law 说的"在途字节"按**线程 × 每线程在飞字节**算。一个 float 在飞 = 4 B；一个 float4 在飞 = 16 B。**同样的线程数，向量化让在途字节 ×4**：

```
5060: 需求 162 KB 在途
每线程 4 B  → 需 40600 线程  （全卡 39936，差 2%，喂不饱）
每线程 16 B → 需 10156 线程  （25% occupancy 就够，轻松喂饱）
```

这就是 [NVIDIA 官方博客 "CUDA Pro Tip: Increase Performance with Vectorized Memory Access"](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) 讲的同一个道理：**同样的访存指令数，搬 4 倍数据**。

### 3.2 ★ 前置条件：对齐，以及对齐为什么会坏

128-bit 访存要求**地址 16 字节对齐**（地址是 16 的倍数）。这是硬性要求，不是性能建议：

- **torch 的张量起始地址通常没问题**：PyTorch 缓存分配器分配的块是 512 字节对齐的。
- **坏就坏在视图上**：`x[:, 1:]` 这种切片让 storage_offset 偏移了 1 个 float（4 字节）→ 起始地址变成 4 mod 16 → **未对齐** → 128-bit 访问是未定义行为（错数或崩溃）。
- 同理，`H % 4 != 0` 时最后一组 float4 会读到行尾之外——**越界**。

所以 wrapper 里必须两道检查 + Python 侧回退（§3.4 的完整代码）：

```cpp
TORCH_CHECK(H % 4 == 0, "vec4 requires H % 4 == 0");
TORCH_CHECK((uintptr_t)xp % 16 == 0 && (uintptr_t)wp % 16 == 0 && (uintptr_t)yp % 16 == 0,
            "vec4 requires 16-byte aligned pointers");
```

### 3.3 完整 kernel（配套文件 `rmsnorm_variants.cu` 里的版本，关键行注释）

```cuda
template <int BLOCK>
__global__ void rmsnorm_vec4(const float4* __restrict__ x4,
                             const float4* __restrict__ w4,
                             float4* __restrict__ y4,
                             int H4, float eps) {   // H4 = H/4
    const int row = blockIdx.x;
    const float4* xr = x4 + (size_t)row * H4;
    float4*       yr = y4 + (size_t)row * H4;

    float acc = 0.f;
    for (int i = threadIdx.x; i < H4; i += BLOCK) {
        const float4 v = xr[i];                  // ★ 一条 LDG.E.128：4 个 float 一次到位
        acc = fmaf(v.x, v.x, acc);               // 拆开做 4 次融合乘加
        acc = fmaf(v.y, v.y, acc);
        acc = fmaf(v.z, v.z, acc);
        acc = fmaf(v.w, v.w, acc);
    }
    acc = block_reduce_sum<BLOCK>(acc);

    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(acc / (H4 * 4) + eps);   // 注意分母是 H4*4 = H
    __syncthreads();
    const float s = inv_rms;

    for (int i = threadIdx.x; i < H4; i += BLOCK) {
        const float4 v = xr[i], g = w4[i];       // w 也按 float4 读 → w 也要 16B 对齐！
        yr[i] = make_float4(v.x*s*g.x, v.y*s*g.y, v.z*s*g.z, v.w*s*g.w);
    }
}
```

### 3.4 三层回退链（复用 Day 1 的"无缝回退"思想）

```
vec4 条件不满足（H%4≠0 / 指针未对齐）→ 回退 reread（标量 CUDA 版，仍比 torch 快）
reread 条件不满足（非 CUDA / 非 fp32 / 非连续）→ 回退 torch 实现
```

C++ 层负责**报错**（最终防线，防未来代码误用）；Python 层负责**回退**（用户体验，永不报错）：

```python
# engine/backend.py —— CudaBackend 更新
class CudaBackend(TorchBackend):
    def __init__(self, block: int = 256, variant: str = "reread"):
        self.block = block
        self.variant = variant

    def rmsnorm(self, x, weight, eps=1e-6):
        from .kernels import rmsnorm as cuda_rmsnorm
        if not x.is_cuda or x.dtype != torch.float32 or not x.is_contiguous():
            return super().rmsnorm(x, weight, eps)          # 第三层：回退 torch
        v = self.variant
        # 第二层：vec4 的条件不满足 → 退回标量 CUDA 版（保住 CUDA 路径，不必退到 torch）
        if v == "vec4" and (x.shape[-1] % 4 != 0
                            or x.data_ptr() % 16 != 0
                            or weight.data_ptr() % 16 != 0):
            v = "reread"
        if v == "stash" and x.shape[-1] % self.block != 0:  # stash 要求 H 整除 BLOCK
            v = "reread"
        return cuda_rmsnorm(x, weight, eps, self.block, v)
```

### 3.5 收益预期与常见错误

- **收益最大的地方不是大形状**：大形状昨天已经 ~90% 贴峰（Day 2 判据：memory-bound 时任何访存技巧都是 0）。**vec4 的价值在没贴峰的形状上**——(32, 4096)/(1, 4096) 那种在途字节不足的场景，预期提升最大。**"优化技巧的收益取决于瓶颈类型"——这条今天用数据坐实，它就是 cost model 的一条规则。**
- 常见错误：忘查 w 的对齐（只查了 x）；H4 和 H 混用（rsqrt 的分母写成 H4）；`make_float4` 里把 g 的四个分量顺序写错。

---

## 4. 板斧四：寄存器暂存 vs 读两遍（1.5h，今天的正题）

### 4.1 交接单原文（W1 Day4 你留给自己的作业）

> "待实测的设计决策：x 读两遍（靠 cache）vs 第一遍暂存寄存器/shared。H=4096 时寄存器方案要 16 个 float/线程，可能压 occupancy —— A/B 实测决定。"

今天就是"A/B 实测决定"的那天。两个版本：

- **版本 A（reread）**：第一遍读 x 算 acc；第二遍**再读一次 x** 写 y。赌第二次读命中缓存。
- **版本 B（stash）**：第一遍读 x 时**把数据存进自己的寄存器**（buf[EPT]）；第二遍从寄存器取，不再碰 global。

### 4.2 版本 A 的机制：L1 缓存命中到底有多便宜

**L1 cache（一级缓存）** 是 SM 片内的高速小缓存，与 shared memory 共用同一块片上 SRAM。按 [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/c-c-best-practices-guide/) 的说明：**计算能力 6.0+ 的 GPU 上，全局内存读默认在 L1 和 L2 中缓存**——也就是说版本 A 的第一遍读**免费**把整行 x 带进了 L1。

第二遍读的代价对比：

```
L1 命中：~30 个周期（片上，短途）
显存（L1 miss）：~400–600 ns（几百个周期，长途）
```

一行 fp32 H=4096 = 16 KB；5060 每 SM 片上 SRAM（L1+shared 共用）约 128 KB 量级——**一行数据在 L1 里绰绰有余**。而且第二遍读的延迟还有 warp 切换兜底（Day 2 的延迟隐藏）。**所以"读两遍"的第二遍，实际代价接近寄存器。**这就是导师预测"reread 会赢或打平"的第一个论据。

### 4.3 版本 B 的代价账：寄存器换来的省，会掉 occupancy 档

stash 每线程要多占 **EPT = H/BLOCK 个数据寄存器**（H=4096、BLOCK=256 时 EPT=16）。对着 Day 2 的阈值表算账（BLOCK=256）：

```
regs/thread ≤ 42 → 6 blocks/SM → occupancy 100%
43 – 51         → 5 blocks    → 83.3%
52 – 64         → 4 blocks    → 66.7%
65 – 85         → 3 blocks    → 50%
```

Day 1 的 reread 基准约 24 regs；stash 加 16 个数据寄存器 ≈ 40——**恰好还压在 100% 档里**。所以精确化的预测是：**只要编译器没把展开和地址运算再推高 12 个寄存器（≥52），stash 就不掉档，reread 与 stash 打平**；推高了，stash 掉到 66.7%，reread 赢。**用 -Xptxas -v 看一眼就知道你站哪边——这就是"先算再跑"的实操形态。**

**local memory 的坑**（板斧② 联动）：`buf[EPT]` 只有在三个循环全部 `#pragma unroll`（j 是编译期常量）时才是寄存器数组；否则整个 buf 溢出到 **local memory**——名字叫"本地"，实际是 **DRAM 里给每个线程划的私有备份区**，延迟和全局内存一个量级。溢出发生时 -Xptxas -v 会打印 `spill stores`——看到它就知道 stash 完蛋了。

### 4.4 先写预测再跑（对赌表）

| 形状 | 我预测谁赢 | 推理链（必须写满） | 实测 | 对了吗 |
|---|---|---|---|---|
| (16384, 4096) | | | | |
| (32, 4096) | | | | |
| (1, 4096) | | | | |
| (16384, 8192) | | | | |
| … | | | | |

**导师的预测（供对赌，先自己写）**：

> **H=4096 系列：reread 赢或打平。** 理由：16 KB 一行在 L1 里，第二遍读≈寄存器；stash 多占 16 个数据寄存器，要么不掉档（打平）要么掉档（输）。**但 H=8192 可能反转**：一行 32 KB，6 个 block 的工作集 = 192 KB，超过片上 SRAM——L1 命中率开始崩（可用 ncu 的 `l1tex__t_sector_hit_rate.pct` 实测命中率验证这个机制）；同时 stash 的 EPT 变 32，寄存器压力翻倍、必掉档。**两边同时恶化，谁先死是实验问题。** 如果反转点存在，**它就是 cost model 里最漂亮的一项——因为它能被 L1 容量和寄存器阈值算出来**，这正是"能算不用试"的样板。

### 4.5 stash 的完整代码（配套文件里有全套）

```cuda
// EPT = H / BLOCK，必须编译期常量（buf 是寄存器数组，下标必须编译期可解）
template <int BLOCK, int EPT>
__global__ void rmsnorm_stash(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ y,
                              int H, float eps) {
    const int row = blockIdx.x;
    const float* xr = x + (size_t)row * H;
    float*       yr = y + (size_t)row * H;

    float buf[EPT];                                // 每线程 EPT 个数据寄存器
    #pragma unroll
    for (int j = 0; j < EPT; ++j) buf[j] = xr[threadIdx.x + j * BLOCK];  // 唯一一次读 global

    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < EPT; ++j) acc = fmaf(buf[j], buf[j], acc);       // 归约走寄存器

    acc = block_reduce_sum<BLOCK>(acc);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) inv_rms = rsqrtf(acc / H + eps);
    __syncthreads();
    const float s = inv_rms;

    #pragma unroll
    for (int j = 0; j < EPT; ++j)
        yr[threadIdx.x + j * BLOCK] = buf[j] * s * w[threadIdx.x + j * BLOCK];  // 不再读 x
}
```

**分发层的现实**：stash 是双模板参数 `(BLOCK, EPT)`，运行时才知道的值只能靠**显式实例化 + 分发表**。EPT 支持 {2,4,8,16,32,64} 共 12 个实例（覆盖扫描与测试的全部组合），在 wrapper 里用一个宏展开（见 `rmsnorm_variants.cu` 的 `STASH_DISPATCH`）。**工业现实**：实例表是显式列的、封闭的——防止组合爆炸；真实框架（如 CUTLASS）用代码生成器干同样的事。

### 4.6 跑完把结论写回 04 笔记的交接单（闭环模板）

```markdown
## 【交接单闭环 · W2 Day3 答复】
> W1 Day4 原问题：x 读两遍（靠 cache） vs 第一遍暂存寄存器/shared，H=4096 时谁快？
数据（5060, fp32，84 点扫描）：
- H=4096：reread __% vs stash __% → 结论：______（预测：打平或 reread 小胜）
- H=8192：reread __% vs stash __% → 反转点是否出现：______
- 机制确认：L1 命中率实测 __%（ncu l1tex）；stash 实际寄存器 __ 个（掉档了吗：__）
- 学到的规则：______（一句话，进 cost model）
```

### 4.7 工业视角：这个 A/B 测试为什么值得做

FlashAttention、vLLM 的 persistent kernel 里，"数据留在寄存器/片上，绝不回读 global"是核心手法——但它们的适用场景是**一个 block 长期驻留、反复用同一批数据**。对"一行只碰两次"的 RMSNorm，寄存器的租金（occupancy 掉档）可能比省下的第二次读还贵。**今天这个实验训练的是判断力：任何"优化"都有租金，先算租金再签合同。** 以后读那些大 kernel 的源码时，你就能看出它们为什么在那种形状下选择那种存法。

---

## 5. 【造】三变体接进引擎 + 测试（2.5h）

### 5.1 改动清单

1. `csrc/rmsnorm.cu` ← 换成 `rmsnorm_variants.cu`（三 kernel + wrapper + 分发表）
2. `csrc/bindings.cpp` ← 换成带 variant 参数的版本：

```cpp
#include <torch/extension.h>
torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor w, double eps,
                           int64_t block, const std::string& variant);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm", &rmsnorm_cuda, "RMSNorm (CUDA), variants: reread/vec4/stash",
          py::arg("x"), py::arg("w"), py::arg("eps") = 1e-6,
          py::arg("block") = 256, py::arg("variant") = "reread");
}
```

3. `engine/backend.py` ← CudaBackend 加 variant 与回退（§3.4 代码）
4. `engine/kernels.py` **不用动**——三个变体在同一个 .cu 里，`load()` 的 sources 不变，改完源码重跑即重编（这正是 Day 1 选 load() 的红利）。

### 5.2 测试：48 用例 + 一个专门测回退的用例

```python
@pytest.mark.parametrize("shape", [(1,1,4096), (8,2048,4096), (3,7,2048), (1,1,4095)])
@pytest.mark.parametrize("block", [128, 256, 512, 1024])
@pytest.mark.parametrize("variant", ["reread", "vec4", "stash"])
def test_rmsnorm_matches_torch(shape, block, variant):
    ...   # 4 × 4 × 3 = 48 个用例

def test_vec4_alignment_fallback():
    # 构造“连续但 16B 未对齐”的张量：storage_offset=1 的窄视图
    # —— 这是 x[:, 1:] 类切片的精确复现，专门测回退逻辑
    raw = torch.randn(4100, device="cuda")
    x = torch.as_strided(raw, (4096,), (1,), storage_offset=1)   # 连续 ✓，对齐 ✗
    w = torch.randn(4096, device="cuda")
    ref = TorchBackend().rmsnorm(x, w)
    out = CudaBackend(variant="vec4").rmsnorm(x, w)              # 应自动回退 reread
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
```

**两个设计点**：① `(1,1,4095)` 这个旧用例在 `vec4`（H%4≠0）和 `stash`（H%BLOCK≠0）下**免费测试了回退路径**——测试设计的红利；② 对齐专项用例用 `as_strided` 精确复现"连续但未对齐"，这是工业上最容易漏的场景（切片 + contiguous 的组合极难用肉眼发现）。

**增量验证纪律（规划原话）**：每加一个 variant 就跑一次全量测试，别攒着。攒着的 bug 会互相掩盖，定位成本翻倍。

---

## 6. 【研】84 点扫描与搜索成本基线（1.5h）

### 6.1 扫描：block × shape × variant = 4 × 7 × 3 = 84 点

配套 `sweep_rmsnorm.py` 是 Day 2 脚本的扩展版，两处新增：`VARIANTS` 维度、**全程计时**。跑完它会打印并落盘：

```python
out = dict(gpu=hw_spec()["name"], sweep_seconds=round(sweep_s, 2), data=rows_out)
```

### 6.2 为什么"84 个配置跑了 X 秒"是论文的动机

把这行写进 `research/findings.md`：

> **"穷举 84 个配置耗时 X 秒；而这只是单个算子、单个 dtype、单张卡。"**

算术：真实场景的形状空间（batch×seq×hidden×dtype×架构）轻松到几千个点；Triton autotune 对每个配置都要**真跑一遍（含编译）**；生产部署时这种搜索要么离线跑（贵）、要么省掉（次优）。**你的方法（几个硬件常数算出最优，0 试跑）的全部价值，就建立在这行数据上。** 今天的 X 秒是这条证据链的第一个刻度。

### 6.3 findings.md 更新模板

```markdown
# W2 Day3 findings —— 访存优化四板斧（RTX 5060, fp32）

## 四板斧收益表
| 板斧 | 治什么 | 实测收益（哪个形状上） | 为 0 的条件 |

## A/B 决策闭环
（§4.6 的交接单答复贴这里）

## 搜索成本基线
穷举 84 配置耗时 __ 秒（单算子、单 dtype、单卡）→ 乘上形状空间/算子数/架构数的外推：__

## 学到的规则（进 cost model）
1. …
2. …

## 下一步
- Day 5 上 H100：同一脚本重跑，产出跨架构四板斧对比
```

---

## 7. 常见错误与调试速查表（Day 3 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| vec4 崩溃或数值错 | H%4≠0 / 指针未对齐 | C++ 层 TORCH_CHECK 两道；Python 层回退 reread |
| vec4 没变快 | 该形状本就 memory-bound（%peak>85%） | 预期之内：vec4 只救"没贴峰"的形状（Day 2 判据） |
| stash 明显变慢 | buf 溢出到 local memory / 寄存器掉档 | 查 -Xptxas -v 的 spill 与 regs；确认三个循环都 unroll 了 |
| restrict 收益为负 | 编译器利用承诺做了次优决策（少见但存在） | 记录负结果，复测确认后如实写入收益表 |
| unroll 后变慢 | 寄存器掉档（ILP 的租金） | 减 unroll 因子（#pragma unroll 4 → 2）重测 |
| cuobjdump 打不开 .pyd | cuobjdump 要 cubin/o/exe | 先 nvcc -cubin，或对 build 目录里的 .o 用 |
| 84 点扫描个别点报错 | stash 的 H%BLOCK≠0 没被 Python 回退接住 | 查 backend.py 的回退分支；这正是测试 (1,1,4095) 防的事 |
| 测试 vec4+切片场景全绿但心里没底 | 没测"连续但未对齐" | 加 `as_strided(storage_offset=1)` 专项用例（§5.2） |

---

## 8. 完成标准自测（三道题，先默写再对答案）

1. **四板斧各治什么瓶颈、实测各值多少？**
   *答案要点*：restrict 治"编译器不敢优化"（别名），收益 0–10%、简单 pattern 上可能为 0；unroll 治循环控制开销与 ILP 不足，代价是寄存器，收益小且常被 -O3 自动做；float4 治"在途字节不足"（Day 2 的 162 KB 临界），在未贴峰形状上收益最大、贴峰形状上为 0；寄存器暂存治"第二遍读 global"，H=4096 与 reread 打平或输（L1 命中≈寄存器、且不掉档/掉档），H=8192 反转与否是今天的实验结论。
2. **"读两遍"为什么可能不慢？什么条件下会反转？**
   *答案要点*：cc 6.0+ 全局读默认进 L1/L2——第一遍读把整行（16 KB）带进 L1，第二遍是 ~30 周期的片上命中，且可被 warp 切换隐藏；stash 省下这次读的代价是 +EPT 个寄存器，可能压 occupancy 掉档（BLOCK=256：regs 43+ 掉 5 块、52+ 掉 4 块）。反转条件：H 大到每 SM 的 block 工作集超过片上 SRAM（H=8192 时 6×32 KB=192 KB）→ L1 命中崩，此时 stash 的"不再读"才值钱——但 stash 的 EPT=32 也同时掉档，谁赢是实验问题。
3. **float4 版的前置条件与 CHECK/回退？**
   *答案要点*：H%4==0 且 x/w/y 指针 16 字节对齐（torch 分配 512B 对齐→起始 OK；`x[:,1:]` 类切片、storage_offset 破坏对齐）。代码里：C++ TORCH_CHECK 两道（H%4、三个 uintptr_t%16）作最终防线；Python 侧 CudaBackend 检查后回退 reread（再不行才回退 torch）——三层回退链。

---

## 9. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] 三个 kernel 变体接进引擎，`variant` 参数可切换
- [ ] `tests/test_cuda_backend.py` 48 用例 + 对齐专项用例全绿
- [ ] 84 点扫描数据 + 总耗时记录（`research/data/sweep_rmsnorm_5060_variants.json`）
- [ ] **04 笔记交接单的闭环答复**（带数据）
- [ ] 四板斧收益表（findings.md）

**明日预告（Day 4）**：战场切换——decode。规划要点：引擎里没有 KV Cache（`generate()` 每步重算整个前缀），今天是还债日；prefill/decode 对照表里 launch 开销是有效计算的 ~1000 倍；第一次融合 `fused_add_rmsnorm`（省 20% 访存、launch 减半）。**关键认知**：单请求 decode 不是你的战场（那是巨核的地盘），你的方法是 prefill 和 batch decode 的配置选择——今天的扫描数据帮你划清这条边界。

---

## 附 A：术语速查表（Day 3）

| 名词 | 一句话解释 |
|---|---|
| `__restrict__` | 承诺指针互不重叠，解锁读提前/寄存器缓存/向量化等优化 |
| aliasing（别名） | 两个指针指向同一内存；编译器默认必须假设存在 |
| `#pragma unroll`（循环展开） | 把循环体抄 N 份：省控制指令、增 ILP、费寄存器 |
| ILP（指令级并行） | 多条无依赖指令同时发射，互相掩盖延迟 |
| float4 / 向量化访存 | 4 个 float 打包；一条 LDG.E.128 搬 16 B（普通读 LDG.E 只搬 4 B） |
| alignment（对齐） | 地址是访问宽度的整数倍；128-bit 访问要求 16 B 对齐 |
| EPT（elements per thread） | 每线程处理的元素数；stash 里 = 数据寄存器数 |
| L1 / L2 cache | SM 片内一级缓存（与 shared 共用 SRAM）/ 全卡共享二级缓存 |
| local memory（本地内存） | 寄存器溢出时的 DRAM 备份区——名字骗人，其实很慢 |
| register spilling（寄存器溢出） | 寄存器不够用，值被挤到 local memory |
| SASS / cuobjdump | GPU 真实机器码 / 反汇编工具（`cuobjdump -sass`） |
| UB（未定义行为） | 违反承诺（restrict/对齐）的后果：偶尔错、难复现 |
| fallback chain（分层回退） | vec4 → reread → torch：逐层降级但永不崩溃 |
| 反转点（crossover） | 两个方案优劣互换的形状点；能被 L1 容量算出来 |
| 搜索成本基线 | autotune 穷举配置的真跑耗时——你方法的对照组 |
| 增量验证 | 每加一个变体跑一次全量测试，不攒 bug |

## 附 B：参考与延伸

- NVIDIA 官方博客：向量化访存（CUDA Pro Tip: Increase Performance with Vectorized Memory Access）—— https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
- CUDA C++ Best Practices Guide（L1/L2 缓存行为、访存优化章节）—— https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CUDA C++ Programming Guide（Vector Types、__restrict__、#pragma unroll）—— https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PTX ISA 文档（LDG.E / LDG.E.128 等指令语义）—— https://docs.nvidia.com/cuda/parallel-thread-execution/
- cuobjdump 文档 —— https://docs.nvidia.com/cuda/cuda-binary-utilities/
