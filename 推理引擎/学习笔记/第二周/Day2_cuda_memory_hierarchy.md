# 阶段一 · W1 Day 2 —— 内存层级 + 全局内存合并访问

> **一句话主题**：Day 1 我们把"几千个线程怎么组织"搞清楚了；今天回答一个更值钱的问题——**这几千个线程去取数据的时候，到底发生了什么，为什么"连续取"和"跳着取"能差 5–10 倍。**
>
> **今天的隐藏主线**：把 W7 学的 Roofline、W0 学的"KV Cache 常驻显存"、以及你桌面上那份 AMK report 里 `4794.19 µs` 这个数字，**用同一套内存带宽的语言全部串起来**。读到 §8 你会亲手把那个 4794 算出来。
>
> 配套产出：`tech_notes/02_memory_access.cu`（五种访问模式实测）、`tech_notes/bench_harness.py`（引擎对标脚手架）。

---

## 0. 先给六个学习目标问题一个"电梯答案"

读完全文回来，你应该能合上书复述这六条。

1. **register / shared / L2 / global 的延迟差多少？**
   大约 **1 : 25 : 200 : 500**（周期数）。从 global memory（全局内存，就是显存 HBM）读一个数的时间，够你在寄存器里做**几百次**浮点运算。**这条比例是你今后所有优化决策的标尺**：任何"多算几次以少读一次"的交易，都是赚的（§2）。

2. **什么叫"合并访问"（coalesced access）？**
   一个 warp（32 个线程）在同一条 load 指令里发出的 32 个地址，如果落在**同一段连续的 128 字节**里，硬件的合并器能把它打包成 **4 次 32 字节的内存事务**；如果地址散落，就退化成最多 **32 次独立事务**。同样是取 128 字节有用数据，前者搬 128B、后者搬 1024B——**8 倍的带宽浪费**（§3）。

3. **为什么是 8 倍而不是 32 倍？**
   因为 DRAM 的最小搬运单位是 **32 字节的 sector（扇区）**，不是 4 字节。你只要 1 个 float，硬件也得整段 32B 搬上来，浪费 8 倍。**这个"8"是你预测跨步访问损失的心算公式**（§3.3）。

4. **为什么"减少访存"比"减少计算"重要？**
   H100 的算力 989 TFLOPS、带宽 3.35 TB/s，**脊点（ridge point）≈ 295 FLOP/Byte**——每从显存搬 1 个字节，你得配 295 次浮点运算才算"喂饱"它。而 LLM decode 阶段的算术强度只有 **1 FLOP/Byte**，差了 295 倍。**大模型推理的绝大多数算子，时间根本不花在算，而花在等数据**（§6.1）。

5. **`cudaMemcpy` 有多贵？数据留在 GPU 上能省多少？**（W0 Day5 钩子 ②）
   PCIe Gen5 实测约 50 GB/s，HBM 是 3350 GB/s，**差 60–70 倍**。所以推理引擎的铁律是"**数据一旦上了 GPU 就别下来**"：权重常驻、KV Cache 常驻、中间激活值算完直接喂下一个 kernel（§6.3）。

6. **今天这堂课在引擎和 AMK 上分别落在哪？**
   引擎：`bench_harness.py` 里那个 `pct_of_peak_bw` 指标——从今天起，**报告一个算子只说"多少 ms"是没有信息量的，必须说"占了带宽峰值的百分之几"**（§7）。AMK：那份 report 里的 `HBM-bandwidth roofline floor: 4794.19 µs`，就是"16.06 GB 权重 ÷ 3.35 TB/s"，**今天学完你能自己推出这个数，并因此判断出报告里的 719.8 µs 是不可能达成的预测值**（§8）。

---

## 1. 问题背景：为什么"访存"是 GPU 编程的真正主题

### 1.1 一个反直觉的事实

初学 CUDA 的人有个默认假设：**GPU 快 = 算得快**。所以优化 = 减少计算量。

这个假设在 2005 年是对的，今天基本是错的。看一组硬件演进数字：

| | 2010 (Fermi) | 2024 (H100 SXM) | 涨了多少倍 |
|---|---|---|---|
| 峰值算力（FP32/BF16） | ~1 TFLOPS | ~989 TFLOPS (BF16) | **~1000×** |
| 显存带宽 | ~150 GB/s | ~3350 GB/s | **~22×** |

**算力涨了 1000 倍，带宽只涨了 22 倍。** 这个剪刀差有个名字叫 **memory wall（内存墙）**。结果就是：现代 GPU 是一台"**算得飞快但严重吃不饱**"的机器，绝大多数 kernel 的时间都花在等数据到位。

> **一个类比**：你雇了 1 万个能一秒钟算一道题的天才（算力），但整个工厂只有一条**每秒送 20 份材料**的传送带（带宽）。此时你去优化"让天才算得更快"完全无意义——**瓶颈在传送带**。GPU 优化的 80% 工作，是在优化传送带怎么用。

### 1.2 那"用好传送带"具体是什么意思？

两件事，正好对应今天的两个知识点：

1. **少搬**：能不从显存读的就别读——用更快的层（register / shared memory）把数据留住、复用。→ §2 **内存层级**
2. **搬得整齐**：非搬不可的时候，让硬件能把零散请求打包成大块传输，别浪费带宽。→ §3 **合并访问**

这两件事，是你这个暑假**每一个 kernel 优化**的底层动作。W2 的 tiled matmul（把数据搬进 shared 复用）是第 1 件事的极致，W3 的 FlashAttention（不落盘中间矩阵）也是第 1 件事的极致——**FlashAttention 从来不是一个"更聪明的数学"，它是一个纯粹的访存优化**。

---

## 2. 核心原理一：内存层级——五级仓库和它们的价格表

### 2.1 先把类比立住

把一个 SM（Streaming Multiprocessor，流式多处理器，Day1 §2.2 的"车间"）想成一个工人的工位：

| 存储层 | 延迟（周期，量级） | 容量（H100） | 类比 | 谁能访问 |
|---|---|---|---|---|
| **register（寄存器）** | ~1（基本免费） | 256 KB/SM（64K×32bit） | **手里正攥着的那张纸** | 单个线程私有 |
| **shared memory（共享内存）** | ~20–30 | 最多 228 KB/SM | **工位旁边的公共工作台** | 同一个 block 内所有线程 |
| **L1 cache** | ~30–40 | 与 shared 共用 256 KB | 工作台下面的抽屉（自动整理） | 单 SM（硬件管） |
| **L2 cache** | ~200 | **50 MB**（全卡共享） | **楼层中转柜** | 全 GPU |
| **global memory / HBM（显存）** | ~400–800 | 80 GB | **楼下的中央仓库** | 全 GPU + CPU 可 memcpy |
| **host memory（主机内存，过 PCIe）** | ~10,000+ | 几百 GB | **城市另一头的异地仓库** | CPU |

> **必须内化的一句话**：**从 HBM 读一个数 ≈ 500 个周期 ≈ 在寄存器里做 500 次乘加。** 所以"多算几遍换少读一次"几乎永远是赚的——**这是 recompute（重计算）、算子融合、FlashAttention 全部成立的经济学基础**。

### 2.2 为什么快的那么小、大的那么慢？——不是厂商偷懒，是物理

学生常问："既然 shared memory 这么快，为什么不做 80GB 的 shared memory？"

- **register / shared / L1 是 SRAM**，做在 SM 里面，**片上（on-chip）**。一个 SRAM 单元要 6 个晶体管，又贵又占面积，信号跑几毫米就到——所以**小而快**。
- **HBM 是 DRAM**，一个单元只要 1 个晶体管 + 1 个电容，密度高、便宜，但要**封装在芯片外面**（H100 是通过硅中介层堆在旁边），信号要走出芯片、还要走 DRAM 自己的行选/列选时序——所以**大而慢**。

**层级存在的唯一理由**：没有一种介质能同时做到又大又快又便宜，所以用"金字塔"逼近——**热数据往上放，冷数据往下压**。这套思路和 CPU 的 L1/L2/L3、和操作系统的"内存-磁盘"、甚至和 vLLM 的"GPU KV Cache ↔ CPU swap"是**同一个设计模式的不同尺度实例**。

### 2.3 register 到底是"什么"——看底层代码

寄存器不是你 malloc 出来的，**它是编译器在编译期静态分配的**。写在 kernel 里的普通局部变量，默认就住在寄存器里：

```cpp
__global__ void demo(const float* in, float* out, int n) {
    int   idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx  -> 寄存器
    float acc = 0.f;                                     // acc  -> 寄存器
    for (int k = 0; k < 4; ++k) acc += in[idx * 4 + k];  // in[] -> global，每次都要跑仓库
    out[idx] = acc;
}
```

编译成 PTX（`nvcc -ptx demo.cu -o demo.ptx`）能直接看到这两类的区别：

```ptx
// %f1, %r1 这些带 % 的就是虚拟寄存器 —— 访问它们不产生任何访存指令
add.s32         %r5, %r3, %r4;          // idx = blockIdx*blockDim + threadIdx，纯寄存器运算
ld.global.f32   %f2, [%rd7];            // ★ 这一条才是"跑一趟仓库"，~500 周期
add.f32         %f3, %f3, %f2;          // 寄存器累加，~1 周期
st.global.f32   [%rd9], %f3;            // 写回仓库
```

**你要养成的读码习惯**：在 PTX/SASS 里数 `ld.global` / `st.global` 的条数——**那才是真正花钱的地方**，其余的 `add/mul/fma` 相比之下几乎免费。

### 2.4 寄存器不是无限的：spill（溢出）是新手最隐蔽的性能杀手

每个 SM 的寄存器是**固定 65536 个 32-bit**，被这个 SM 上所有活跃线程瓜分。一个线程最多用 255 个。**如果你的 kernel 里局部变量太多（比如手写一个大 tile 的矩阵乘，开了 64 个累加器），编译器塞不下，就会把放不下的变量"溢出"到 local memory。**

> **致命误解**：local memory（局部内存）听起来"很近"，实际上**它就在 HBM 里**！只是被 L1/L2 缓存着。名字里的 "local" 指的是"线程私有"这个**作用域**，不是"物理位置近"。**一次 spill = 一次可能高达 500 周期的访存。**

怎么发现？编译时加一个参数就行，**这是每次写完 kernel 必看的一行**：

```bash
nvcc -O3 -arch=sm_90 -Xptxas -v -c 02_memory_access.cu
# 输出示例：
# ptxas info : Used 32 registers, 0 bytes cumulative stack size, 380 bytes cmem[0]
#                   ↑ 用了多少寄存器          ↑ 0 就是没 spill，非 0 立刻警觉
# 出问题时会长这样：
# ptxas info : Used 255 registers, 96 bytes spill stores, 128 bytes spill loads  ← 灾难
```

**工业实践**：`__launch_bounds__(256, 4)` 可以告诉编译器"我希望每 SM 至少跑 4 个 block"，逼它把寄存器用量压到预算内（宁可少开变量，也别 spill）。这个参数在 W2 写 tiled matmul、W3 写 FlashAttention 时你一定会用到——**FlashAttention 的 tile size 选多大，本质上就是在"寄存器/shared 够不够"和"访存够不够省"之间解一个约束优化**。

### 2.5 shared memory：唯一由你手动管理的高速缓存

- L1/L2 是**硬件自动**管的（你无法控制什么进什么出）。
- shared memory 是**你手动**管的（`__shared__ float tile[32][32];` + 自己 `__syncthreads()`）。

> **类比**：L1 像一个自作主张的助理，你用过的东西他随手放桌上，下次可能还在、也可能被他收走了；shared memory 是**你自己规划的工作台**——放什么、什么时候撤、谁能动，全你说了算。**能手动控制**这件事，就是 tiled matmul、FlashAttention 能榨干性能的根本原因。

今天不深入，但先埋一个钩子：shared memory 被分成 **32 个 bank（存储体）**，**同一 warp 的 32 个线程如果同时访问同一个 bank 的不同地址，就要排队**，这叫 **bank conflict（存储体冲突）**。

> **注意这个结构上的对偶**：
> - global memory 的性能陷阱叫 **uncoalesced（不合并）**——32 个线程的地址太散。
> - shared memory 的性能陷阱叫 **bank conflict**——32 个线程的地址在 bank 维度上撞车。
>
> **它们是同一个道理在两层存储上的两种表现：warp 是 32 个线程一起动的，所以任何存储都必须为"32 个地址同时来"这件事优化。** 这是明天（Day3 shared memory）的主线。

### 2.6 延迟 vs 带宽：Little's Law 解释"为什么光合并还不够"

这里是今天最容易被忽略、但最能体现深度的一点。

- **延迟（latency）**：一次访存**要等多久**（~500 周期）。
- **带宽（bandwidth）**：**单位时间总共能搬多少**（3.35 TB/s）。

Day1 你学了 GPU 靠**大量 warp 轮换来隐藏延迟**。今天用一个公式把它量化——**Little's Law**（利特尔法则，排队论）：

```
必须"在途"的数据量 = 带宽 × 延迟
H100：3.35e12 B/s × 600e-9 s ≈ 2.0 MB
```

**意思是：要跑满 H100 的带宽，任意时刻必须有约 2 MB 的数据"在路上"。** 分摊到 132 个 SM，每个 SM 要有 ~15 KB 在途。

现在算笔账：一个 warp 发一条 `ld.global.f32`（合并的）= 128 B 在途。一个 SM 最多驻留 2048 线程 = 64 个 warp，**每个 warp 只有一条 load 在途的话，总共才 8 KB——只有需求的一半，带宽跑不满！**

这就直接解释了两件实战中的事：

1. **为什么要用 `float4` 向量化**：一个线程一次搬 16 B，一个 warp 一条指令就是 512 B 在途，64 个 warp = 32 KB > 15 KB，**轻松喂饱**。所以 `02_memory_access.cu` 里的 (D) 版本通常比 (A) 还快一截。
2. **为什么 occupancy（占用率，Day1 §4.4）低会掉带宽**：warp 少 = 在途请求少 = 带宽空转。**占用率和访存合并不是两个独立指标，它们通过 Little's Law 连在一起——ncu 报告里 occupancy 和 dram throughput 常常一起低，根因是同一个。**

> **这条是你以后读 ncu 报告的核心思维模型**：DRAM 吞吐上不去，只有两种病因——**要么搬得不整齐（合并度差），要么在途请求不够多（并行度/MLP 不足）**。今天的实验同时压这两个变量。

---

## 3. 核心原理二：合并访问——硬件到底做了什么

### 3.1 先把机制讲清楚

关键前提（Day1 §2.2）：**warp 里 32 个线程执行的是同一条指令**。所以当执行到 `out[idx] = in[idx]` 这一行时，硬件面对的是**同时到达的 32 个地址**。

内存子系统的处理流程（这就是"合并"的全过程）：

```
32 个线程各自算出一个地址
        ↓
【地址合并器 coalescer / L1TEX 单元】把 32 个地址按 32 字节 sector 归类
        ↓
去掉重复，得到「这次要向下游请求哪几个 sector」
        ↓
L1 → 未命中的往 L2 要 → L2 未命中的往 HBM 要（HBM 也是按 32B 为单位搬）
```

**核心概念——sector（扇区）**：内存系统的最小搬运单位是 **32 字节**，缓存行（cache line）是 **128 字节 = 4 个 sector**。**你哪怕只要 1 个 float（4 B），硬件也必须把包含它的整个 32 B sector 搬上来。**

> **类比**：仓库只按"整箱"发货，一箱 32 件。
> - **合并访问**：32 个工人要的货正好装满 4 个箱子 → 发 4 箱、128 件，**一件不浪费**。
> - **跨步访问**：32 个工人各要一件，但分散在 32 个不同箱子里 → 仓库得发 **32 箱、1024 件**，你只用了 128 件——**浪费 8 倍运力**。传送带的载重是固定的，浪费的运力就是浪费的时间。

### 3.2 三种典型模式的账，手算一遍

设 `float`（4 B），warp = 32 线程，一次 load 共需 128 B 有用数据：

| 模式 | 代码 | 触及 sector 数 | 实际搬运 | 有效率 |
|---|---|---|---|---|
| **完美合并** | `in[idx]`（且首地址 128B 对齐） | 4 | 128 B | **100%** |
| **错位 1 个 float** | `in[idx+1]` | 5 | 160 B | 80% |
| **跨步 32** | `in[idx*32]` | 32 | 1024 B | **12.5%** |
| **完全随机** | `in[perm[idx]]` | ≤32 | ≤1024 B | ≥12.5% |

**记住"12.5% = 1/8"这个下界**：单精度跨步访问最坏是 8 倍浪费（不是 32 倍！因为 sector 是 32B 不是 4B）。如果你的数据类型是 `half`（2 B），一个 sector 能装 16 个，最坏就是 **16 倍**浪费——**数据类型越小，不合并的惩罚越重**。这一条对 fp16/int8 推理特别重要。

### 3.3 用一段可运行的 C++ 代码"看见"合并器

光看表格没有体感。下面这段程序**在 CPU 上就能跑**（不需要 GPU），它模拟合并器的行为：给定一个"线程号 → 地址"的映射，算出一个 warp 要触及多少 sector。**这是我认为理解 coalescing 最快的方式——你亲手把硬件那步"归类去重"写出来。**

```cpp
// coalesce_sim.cpp —— 合并器模拟器（纯 CPU，g++ -std=c++17 coalesce_sim.cpp -o sim）
// 作用：不用 GPU 就能预测任意访问模式的访存放大倍数
#include <cstdio>
#include <set>
#include <functional>

constexpr int WARP = 32, SECTOR = 32, LINE = 128, ESIZE = 4;  // float

// 输入：一个 warp 的起始全局线程号 base，和"线程号 -> 元素下标"的映射
// 输出：这一次 load 触及的 sector 数 / cache line 数
void analyze(const char* name, int base, std::function<size_t(size_t)> map) {
    std::set<size_t> sectors, lines;
    for (int lane = 0; lane < WARP; ++lane) {
        size_t byte_addr = map(base + lane) * ESIZE;   // 元素下标 -> 字节地址
        sectors.insert(byte_addr / SECTOR);             // ★ 硬件就是这样"整除归类"的
        lines.insert(byte_addr / LINE);
    }
    size_t moved = sectors.size() * SECTOR, useful = WARP * ESIZE;
    printf("%-24s sectors=%2zu lines=%2zu  搬运 %4zu B / 有用 %3zu B  -> 放大 %.1fx\n",
           name, sectors.size(), lines.size(), moved, useful, (double)moved / useful);
}

int main() {
    const size_t n = 1 << 24, stride = 32, chunk = n / stride;
    analyze("coalesced",        0, [](size_t i){ return i; });
    analyze("offset +1",        0, [](size_t i){ return i + 1; });
    analyze("stride 2",         0, [](size_t i){ return i * 2; });
    analyze("stride 8",         0, [](size_t i){ return i * 8; });
    analyze("stride 32 (naive)",0, [](size_t i){ return i * 32; });
    analyze("transposed read",  0, [=](size_t i){ return (i % stride) * chunk + i / stride; });
    analyze("float4 (16B/thr)", 0, [](size_t i){ return i * 4; });  // 注：等价于每线程搬 16B
    return 0;
}
```

**预期输出**（可以先自己算一遍再运行对答案）：

```
coalesced                sectors= 4 lines= 1  搬运  128 B / 有用 128 B  -> 放大 1.0x
offset +1                sectors= 5 lines= 2  搬运  160 B / 有用 128 B  -> 放大 1.2x
stride 2                 sectors= 8 lines= 2  搬运  256 B / 有用 128 B  -> 放大 2.0x
stride 8                 sectors=32 lines= 8  搬运 1024 B / 有用 128 B  -> 放大 8.0x
stride 32 (naive)        sectors=32 lines=32  搬运 1024 B / 有用 128 B  -> 放大 8.0x
transposed read          sectors=32 lines=32  搬运 1024 B / 有用 128 B  -> 放大 8.0x
float4 (16B/thr)         sectors=16 lines= 4  搬运  512 B / 有用 512 B  -> 放大 1.0x
```

**从输出里读出三个非平凡的结论**：

1. **stride ≥ 8 之后放大倍数就饱和在 8× 了**，再大也不会更糟（sector 已经全散开）。但注意 `lines` 还在涨（8 → 32），**这会进一步恶化 L1/L2 的命中率和 TLB 压力**，所以实测里 stride=32 通常还是比 stride=8 慢——**模拟器给的是访存量下界，实测才有完整故事**。
2. **`offset +1` 只多 1 个 sector（1.2×）**——对齐问题在单精度上没那么可怕。但如果每个线程搬 `float4`，未对齐会让每个 128B 请求都跨行，惩罚显著上升。**这就是为什么向量化访问要求 16B 对齐**。
3. **float4 版本"放大 1.0×"但 sector 数是 16**——它没有浪费带宽，而是**用更少的指令搬了更多的数据**（§2.6 的 Little's Law）。这是"合并度"和"在途量"两个维度的分离：**合并度看放大倍数，在途量看每指令搬多少**。

### 3.4 工业界的两个直接推论：SoA 和 KV Cache 布局

**推论一：AoS vs SoA（结构体数组 vs 数组结构体）**

```cpp
// AoS (Array of Structures) —— CPU 上的自然写法，GPU 上的性能灾难
struct Particle { float x, y, z, w; };
Particle p[N];
__global__ void bad(Particle* p)  { p[idx].x *= 2.f; }
// warp 内相邻线程访问相隔 16 B 的地址 -> stride 4 -> 放大 4×

// SoA (Structure of Arrays) —— GPU 上的标准写法
struct Particles { float *x, *y, *z, *w; };
__global__ void good(Particles p) { p.x[idx] *= 2.f; }
// warp 内相邻线程访问相邻 float -> 完美合并
```

> **这不是玄学，是 §3.3 那段模拟器算出来的确定性结论。** 任何 GPU 项目在设计数据结构的第一天就要定 SoA——**改数据布局比改算法难十倍，因为它牵动全部代码。**

**推论二：KV Cache 的形状不是随便定的**（呼应 W6 你亲手写的 KV Cache）

KV Cache 的张量形状常见两种：`[B, H, S, D]` 和 `[B, S, H, D]`（B=batch, H=head 数, S=序列长, D=head_dim，Llama-3-8B 里 D=128）。

- decode 阶段，attention 要为**某个 head** 读取**所有历史 token** 的 K。
- `[B, H, S, D]`：固定 head 后，`S×D` 是一整片连续内存 → **完美合并**。
- `[B, S, H, D]`：固定 head 后，相邻 token 之间隔了 `H×D`；但**每个 token 的 D=128 个元素（fp16 = 256 B）本身是连续的** → 每次读 256 B 连续块，仍然是合并的，只是块小一点、跳跃多一点。

> **真正的结论不是"哪个布局绝对好"，而是：只要保证「最内层维度连续、且连续块 ≥ 128 B」，合并度就有保障。** 这也解释了 **PagedAttention** 为什么可行——它把 KV Cache 切成 16 个 token 一块的物理页，**块内连续**（合并度保住了），块间靠 block table 跳转（只多一次指针查找）。**"显存可以不连续、但块内必须连续"，这是 vLLM 设计里最关键的一条访存约束，而它的依据就是今天这节课。**

---

## 4. 实测：把理论压成数字

### 4.1 实验设计（对应 `02_memory_access.cu`）

五个 kernel，**计算完全一样（copy），只有地址算法不同**：

| # | 模式 | 设计意图 |
|---|---|---|
| A | coalesced | 基线，应逼近 HBM 峰值 |
| B | offset=1 | 隔离"对齐"这一个变量 |
| C | strided（转置式读） | 隔离"合并度"这一个变量 |
| D | coalesced + float4 | 验证 Little's Law：在途量的作用 |
| E | grid-stride loop | 工业标准写法，验证它不损失带宽 |

**这里有一个必须讲清楚的设计选择**——为什么 C 版本我写成了"转置式"的双射，而不是最直觉的 `in[idx * stride % n]`：

```cpp
// ❌ 直觉写法（很多教程这么写，包括我原本的计划稿）
int src = (idx * 32) % n;
// 陷阱：idx*32 mod n 只能取到 32 的倍数！实际只访问了 n/32 个不同元素。
// n = 1<<24 时，真正被读的数据只有 2 MB —— 而 H100 的 L2 有 50 MB，
// 于是整个数据集第二次迭代起全部命中 L2，你测到的是 L2 带宽，不是 HBM 带宽。
// 后果：跨步版本可能只慢 1.5 倍，甚至看起来"没变慢"，实验结论完全错误。

// ✅ 正确写法：双射，每个元素恰好读一次，读的总量和基线严格相等
size_t chunk = n / stride;
size_t src = (idx % stride) * chunk + idx / stride;
// warp 内 32 个 lane 的地址间隔 chunk*4 B（约 8 MB），保证 32 个独立 sector，
// 同时覆盖全部 n 个元素 —— 差异 100% 来自访问模式。
```

> **这是今天最重要的方法论收获，比结论本身更值钱**：**做性能对比实验，必须保证除了被测变量以外，"数据总量、缓存状态、计算量"三者严格相等。** 否则你测出来的是一个"看起来很有道理的假数字"。这条纪律，就是 W8 那个"三尺子"在性能侧的孪生兄弟。

### 4.2 跑起来

```bash
# 1) 编译（-lineinfo 让 ncu 能对回源码行；-Xptxas -v 顺便看寄存器用量）
nvcc -O3 -arch=sm_90 -lineinfo -Xptxas -v -o memacc 02_memory_access.cu

# 2) 先跑一遍，确认正确性全 OK、并拿到有效带宽表
./memacc

# 3) 用 compute-sanitizer 确认没有越界（Day1 养成的肌肉记忆）
compute-sanitizer ./memacc
```

**H100 上的预期数量级**（你的实测数字请填进下表，这就是今天的产出）：

| pattern | time (ms) | eff. BW (GB/s) | 占峰值 | slowdown |
|---|---|---|---|---|
| coalesced (baseline) | | ~2600–3000 | ~80–90% | 1.00× |
| offset=1 | | ~2400–2900 | | ~1.05× |
| **strided (scatter)** | | **~350–600** | **~10–18%** | **~5–8×** |
| coalesced + float4 | | ~2800–3100 | ~85–93% | ~0.95× |
| grid-stride loop | | ≈ baseline | | ~1.0× |

> **为什么合并版也只有峰值的 80–90%，而不是 100%？** 因为 3.35 TB/s 是理论峰值，实际要扣掉 DRAM 刷新、行切换开销、ECC。**"实测能到峰值 80–90% 就算跑满了"是业界共识的经验值**——你以后看到任何号称"达到 100% 带宽"的数字都该怀疑（多半是把 L2 命中算进去了）。

### 4.3 ncu：从"慢"到"为什么慢"

`./memacc` 只告诉你**慢**，`ncu` 告诉你**为什么**。**今天必须掌握的四个 metric（度量指标）**：

```bash
ncu --metrics \
 l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
 dram__bytes_read.sum,\
 dram__throughput.avg.pct_of_peak_sustained_elapsed,\
 gpu__time_duration.sum \
 ./memacc

# 想看完整分析（含 Memory Workload Analysis 的那张内存层级图）：
ncu --set full -o memacc_rep ./memacc     # 生成 .ncu-rep，用 GUI 打开
ncu --import memacc_rep.ncu-rep --page details | less   # 或命令行看
```

| metric | 含义 | 怎么读 |
|---|---|---|
| `l1tex__average_t_sectors_per_request_..._op_ld.ratio` | **每个 load 请求平均触及几个 sector** | **今天最核心的一个数**。float 的理想值 = **4**，跨步版会飙到 **32**。§3.3 模拟器算的就是它——你可以拿实测值验证你的手算 |
| `dram__bytes_read.sum` | 真的从 HBM 读了多少字节 | 跨步版应约为合并版的 **8 倍**（这就是"浪费"的直接证据） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 吞吐占峰值百分比 | 合并版 80%+，跨步版 10–20% |
| `gpu__time_duration.sum` | kernel 实测耗时 | 和你 event 计时的结果交叉验证 |

> **"每请求 sector 数"这个指标，是你以后一眼判断 kernel 有没有访存问题的第一探针。** 看到它 ≫ 4，不用往下看，先去改数据布局。
>
> **和 W7 三级 profiler 的分工重申一遍**：`torch.profiler` 看**框架层**（哪个算子慢）→ `nsys` 看**时间线**（kernel 之间有没有空隙、CPU 有没有卡住 GPU）→ `ncu` 看**单个 kernel 内部**（今天这一层：访存合不合并、带宽跑没跑满）。**今天是第一次真正用到第三级的分辨率。**

### 4.4 想看得更深：SASS 里数一数真实指令

```bash
cuobjdump -sass ./memacc | grep -E "LDG|STG"
```

你会看到类似：

```
/*0090*/  LDG.E R4, [R2.64] ;            // (A) 每线程 4 B：普通 32-bit global load
/*00a0*/  STG.E [R6.64], R4 ;
...
/*0110*/  LDG.E.128 R4, [R2.64] ;        // (D) float4：一条指令搬 16 B ★
/*0120*/  STG.E.128 [R8.64], R4 ;
```

**`LDG.E.128` 就是向量化访问在硬件指令层面的样子。** 当你以后写 kernel 想确认"编译器有没有帮我向量化"，就是 grep 这一条——**这比猜测和读文档可靠。**

---

## 5. 常见陷阱与调试技巧（血泪清单）

1. **【最经典】数据集比 L2 小，测的是缓存不是显存。**
   H100 的 L2 有 **50 MB**——网上大量教程用 64 MB 数组，在 A100/H100 上早已失真。
   *规矩*：**benchmark 缓冲区至少 4× L2**（脚本里默认 512 MB/buffer，并且会打印警告）。

2. **【最隐蔽】"跨步"实验其实缩小了数据集**（§4.1 那个 `% n`）。
   *自查*：跑之前先问自己"这个索引映射是不是双射？读到的元素总数变了没有？"

3. **忘了预热 / 忘了同步。**
   kernel 是异步发射的，不 `cudaEventSynchronize` / `cudaDeviceSynchronize` 就计时，量到的是发射时间（微秒级的假数据）。Day1 陷阱 ④ 的复现。

4. **编译器把 kernel 优化掉了。**
   如果你写的 kernel 结果没人用（比如只读不写），`-O3` 会把整个 load 消掉，然后你测出"0.001 ms 的惊人带宽"。
   *规矩*：**每个 benchmark kernel 的结果必须被写回 global memory**（我们的 copy kernel 天然满足），或者用 `volatile` / 假分支挡住。

5. **只测了读、把写的账也算进"有效带宽"。**
   copy 的有效带宽公式是 `2*bytes/t`（读一遍 + 写一遍）。**漏乘 2 会让你以为只有一半带宽**，进而误判成"kernel 有问题"。

6. **在 ncu 下测出来的时间比裸跑慢很多，然后慌了。**
   ncu 会 **replay（重放）** kernel 多次来采集不同计数器组，还会默认刷 cache（`--cache-control all`）。**ncu 的绝对时间不可信，它的相对指标才可信**；绝对时间用 `nsys` 或 event 计时。

7. **`cudaMalloc` 的对齐是安全的，但指针偏移不是。**
   `cudaMalloc` 返回至少 256 B 对齐；但 `(float4*)(d_in + 1)` 就是未对齐的 16B 访问 → **直接 `misaligned address` 崩溃**。做向量化时永远检查 `reinterpret_cast` 前的偏移量。

8. **GPU 处于低功耗态导致首轮数据偏慢。**
   *规矩*：预热 ≥ 5 次；正式测量取**中位数**（`bench_harness.py` 已实现）；严谨场景可 `nvidia-smi -lgc <freq>` 锁频。

---

## 6. 工业锚点：这堂课在真实推理系统里长什么样

### 6.1 为什么 LLM decode 必然是 memory-bound——算一遍就懂

**Roofline 的脊点（ridge point）**：算力 ÷ 带宽 = **989e12 ÷ 3.35e12 ≈ 295 FLOP/Byte**。
含义：**每从显存搬 1 字节，必须配 295 次浮点运算，才能把 H100 喂饱。**

现在算 decode 阶段（一次生成一个 token）矩阵乘的算术强度（arithmetic intensity）：

```
y = W x        W: [M, K] fp16（权重），x: [K] （单个 token 的激活）
计算量 = 2 × M × K   FLOP
访存量 = 2 × M × K   Byte      （fp16 权重每个 2 字节，且几乎没有复用）
算术强度 = 1 FLOP/Byte
```

**1 vs 295——差 295 倍。** 这意味着 decode 阶段 GPU 的算力**只用上了千分之三**，剩下的时间全在等权重从 HBM 搬过来。

**由此推出的三条推理系统铁律，你会在整个暑假反复遇到：**

| 现象 | 根因（全是今天这一条） | 对应技术 |
|---|---|---|
| decode 慢 | 每 token 要把**整个模型权重**读一遍 | **量化**（int8/int4 直接把访存量砍 2–4 倍 → 速度近似线性提升） |
| batch 越大越划算 | 权重读一次可以服务 B 个请求，算术强度 ≈ B | **continuous batching**（W5-6 你要做的） |
| 小算子多了很慢 | 每个 kernel 都要把数据读进来再写回去 | **算子融合 / CUDA Graph**（W4） |

> **一个直击本质的推论**：**量化之所以快，主要不是因为"int8 算得快"，而是因为"int8 搬得少"。** 很多人以为是算力问题，其实是带宽问题。**这句话你面试时说出来，档次立刻不一样。**

而 prefill 阶段（一次处理 T 个 token）算术强度 ≈ T，T 超过 ~300 就进入 compute-bound。**"prefill 是 compute-bound、decode 是 memory-bound"——这条你早就背过，今天你终于能自己把这两个数算出来了。**

### 6.2 `contiguous()` 不是免费的：PyTorch 里的合并访问

```python
x = kv_cache.permute(0, 2, 1, 3)     # 只改元数据 stride，0 开销，但 x 已经"不连续"
y = x.contiguous()                   # ★ 这一行会启动一个真实的 copy kernel：
                                     #   读一遍 + 写一遍整个张量
```

为什么框架要做这个看似浪费的 copy？**因为下游的 cuBLAS/FlashAttention kernel 需要合并访问的布局**——与其让每个 kernel 忍受 8 倍访存放大，不如**先花一次 2× 的搬运把布局理顺**。

> **这就是"数据布局转换（layout transform）"在工业界的基本经济学：一次性付 2 倍访存，换后续所有 kernel 的 8 倍收益。** 你在 profiler 时间线里看到的那些神秘的 `elementwise_kernel` / `copy_kernel`，很多就是它。**下次在 nsys 里看到它们，你要能立刻判断"这是布局转换，值不值得？能不能通过改上游布局消掉？"——这是从"会看 profiler"到"会优化"的分水岭。**

### 6.3 回答 W0 Day5 钩子 ②：`cudaMemcpy` 到底多贵

| 路径 | 带宽 | 相对 HBM |
|---|---|---|
| GPU 内部 HBM 读写 | ~3350 GB/s | 1× |
| GPU ↔ CPU，PCIe Gen5 x16（pinned memory） | ~50 GB/s | **慢 67×** |
| GPU ↔ CPU，pageable memory（未锁页） | ~10–20 GB/s | **慢 200×+** |

**两个直接的工程结论：**

1. **数据一旦上 GPU 就别下来。** 权重常驻显存、KV Cache 常驻显存、中间激活值在 GPU 上流水传递。**vLLM/TensorRT-LLM 追求"整个 forward 一次不回 CPU"，原因就是这张表。**
2. **非传不可时，用 pinned memory（锁页内存）+ 异步拷贝 + 多流重叠。**

```python
# 工业写法：pinned + non_blocking，让 H2D 拷贝和 GPU 计算重叠
buf = torch.empty(shape, pin_memory=True)          # 锁页：DMA 可以直接搬，不用先复制到内核缓冲
gpu_t = buf.to("cuda", non_blocking=True)          # 异步：CPU 立刻返回去干别的
# ❗ 陷阱：non_blocking=True 只有在 pinned memory 上才真异步；
#         普通内存上它会静默退化成同步拷贝——你以为重叠了，其实没有。
```

> 这条在你 W5-6 做 KV Cache offload / swap 时会**直接**用上：把冷 KV Cache 换到 CPU 内存，代价就是这 50 GB/s。**能不能 offload，取决于"换出去省的显存"值不值"换回来的 67 倍延迟"。**

### 6.4 通向 W2/W3：今天是它们的地基

- **W2 tiled matmul**：为什么要把 tile 搬进 shared memory？因为一个 tile 被复用 N 次——**把 N 次 global 访问变成 1 次 global + N 次 shared**，按 §2.1 的价格表，这是 500 周期 → 25 周期的交易。
- **W3 FlashAttention**：标准 attention 要把 `S×S` 的注意力矩阵写回 HBM 再读回来（S=4096 时是 32 MB 的来回搬运）。FlashAttention 把它**切块留在 shared/register 里算完**，中间矩阵**从不落 HBM**。**它的全部收益来自今天这张价格表——数学上它算的是同一个 softmax。**

---

## 7. 【造 1.5h】benchmark 脚手架：把今天的知识变成引擎的量尺

产出 `engine/bench_harness.py`（已写到 `tech_notes/bench_harness.py`，请拷进引擎仓库）。它相对 W8 版本的**三个升级**，每一个都来自今天：

1. **`flush_l2()`**：每次迭代前用一块 1.5× L2 大小的 scratch 张量把 L2 冲掉。
   *为什么*：不刷 L2，小张量算子第二次跑全命中缓存，你会测出"超过 HBM 峰值"的物理不可能数字。**线上真实场景的数据大多是冷的，所以刷 L2 测出来的才是可信数字。**
2. **`pct_of_peak_bw`**：报告"占带宽峰值百分之几"，而不只是 ms。
   *为什么*：ms 是相对量，换个卡就没意义；**占峰值百分比是绝对量，它直接告诉你"还有多少优化空间"**。看到 15% 就知道值得动手，看到 85% 就知道该收手了——**知道什么时候停，和知道怎么优化一样重要。**
3. **`bound` 自动判定**：用算术强度和脊点比较，自动打上 memory/compute 标签。
   *为什么*：优化方向完全相反——memory-bound 就去融合/量化/改布局，compute-bound 就去提计算效率（tensor core、tile 调优）。**判错方向 = 白干一周。**

```bash
python engine/bench_harness.py
# 预期：eager 版 %peak 明显低于 compile 版
#   —— eager 的 RMSNorm 要多次读写中间张量（pow/mean/rsqrt/mul 各一趟），
#      compile 融合成一个 kernel 后接近"读一遍 + 写一遍"的访存下界。
#   这是你在引擎层面第一次亲眼看到"算子融合 = 省访存"。
```

**这个脚手架从今天起是不动的地基**：W2 插 `CudaBackend`、W3 插 FlashAttention、W4 做四方对标，全都往里插新的 `bench_op` 调用即可。**尺子先立好，再造东西——这是工程纪律，不是洁癖。**

---

## 8. 【研 1h】AMK：重跑 `small` 的 nsys，并用今天的知识审一遍基线

### 8.1 操作步骤

```bash
module load CUDA/12.4
nsys profile -t cuda,nvtx,osrt --stats=true \
     -o amk_small_$(date +%m%d) --force-overwrite true \
     amk run small --gpu h100

# 出报告后看这三张表：
nsys stats --report cuda_gpu_kern_sum  amk_small_0724.nsys-rep   # 每个 kernel 的总时间占比
nsys stats --report cuda_gpu_mem_time_sum amk_small_0724.nsys-rep # H2D/D2H 传输时间 ★ 今天的知识点
nsys stats --report cuda_api_sum       amk_small_0724.nsys-rep   # API 侧开销（同步点在哪）
```

**今天的目标只有一个：确认这次跑出来的数字和 W7 report v1 的基线一致（±5% 以内）。** 基线漂了，后面所有"提升 X%"的结论都不成立。

**对齐检查清单**（逐条打勾，不一致就先查环境）：

- [ ] GPU 型号、驱动、CUDA 版本与 v1 相同
- [ ] kernel 总数、名字列表一致（AMK 的调度产物变了没有？`schedule id` 是否还是 `sch_2f8b213192`）
- [ ] top-3 kernel 的时间占比与 v1 相差 < 5%
- [ ] 端到端 µs/token 与 v1 相差 < 5%
- [ ] H2D/D2H 传输时间占比（今天新增的关注点）——**如果它不接近 0，说明有数据在反复过 PCIe，那本身就是一个可优化点**

### 8.2 用今天的知识审你手上那份 report——一个真实的发现

你桌面上的 `..._Llama-3.1-8B-Instruct.h100.report.md` 里有这几行：

```
- weights: 16060.52 MB
- value: 719.80 µs/token  (PREDICTED (analytic cost model))
- GPU status: gpu_mismatch
- HBM-bandwidth roofline floor: 4794.19 µs  (15.0% of bound, 666.0% HBM utilization)
```

**用今天学的东西把 `4794.19` 亲手算出来：**

```
decode 生成一个 token，至少要把全部权重从 HBM 读一遍（算术强度 ≈ 1，无复用）
    权重 = 16060.52 MB ≈ 16.06 GB
    H100 SXM HBM 带宽 = 3.35 TB/s
    下界 = 16.06e9 / 3.35e12 = 4.79e-3 s = 4794 µs   ✅ 与报告完全吻合
```

**于是一个重要判断浮出来了**：报告里的 `719.80 µs/token` 是**解析成本模型的预测值**，而它比物理下界还快 **6.66 倍**（报告自己也标了 `666.0% HBM utilization`——超过 100% 的带宽利用率在物理上不存在）。

> **结论（记进 research 笔记）**：**719.8 µs 这个数不能作为 baseline 引用**，它是成本模型没有把 HBM 权重读取计入的产物（报告里 `GPU status: gpu_mismatch` 也印证了 GPU 端到端路径尚未打通）。**你的 baseline 必须是 nsys 实测值。** 这也正是你在这个项目里**独特价值的具体形态**——组里其他人用的是预测数字，**你有 H100 和 nsys 权限，你出的是实测数字**。
>
> 顺带：`region breakdown` 里 attention 364 µs / mlp 252 µs（合计约 86%）给出了后续该往哪儿看的方向；correctness `FAIL, max abs err 9.2e-1` 但 top-1 agreement 1.0000——**这是"分布数值不准但采样结果没变"的典型状态，优化前必须先确认它是已知问题还是新引入的**，否则你后面所有加速都建在错误结果上（W8 三尺子纪律的直接应用）。

**今天要为 W4 记下的一句话**：AMK 在 H100 上是 memory-bound 还是 compute-bound？**如果实测 `dram__throughput` 接近峰值而计算利用率低 → memory-bound → W4 的优化方向是减少访存/加强融合；反之则是提计算效率。** 这个判定，直接决定你动手改的第一个算子往哪个方向改。

---

## 9. 串联表 + 自测题 + 完成标准

### 9.1 和已有笔记的串联

| 本笔记的位置 | 呼应 / 铺垫 |
|---|---|
| §2.6 Little's Law | 把 **Day1 §4.4 occupancy** 从定性讲成定量：占用率低 → 在途请求少 → 带宽跑不满 |
| §2.5 bank conflict | **Day3（shared memory）**的主线预告；与合并访问是同一原理的两层表现 |
| §3.4 KV Cache 布局 | 呼应 **W6 KV Cache**、铺垫 **W5-6 PagedAttention** |
| §4.3 ncu metric | **W7 三级 profiler** 的第三级第一次真正用满 |
| §6.1 算术强度 / 脊点 | **W7 Roofline** 的定量落地；解释"量化为什么快" |
| §6.3 PCIe vs HBM | 收口 **W0 Day5 钩子 ②**（memcpy 开销）；铺垫 W5-6 的 KV offload |
| §6.4 tiling / FlashAttention | **W2 shared memory + tiled matmul**、**W3 FlashAttention** 的经济学基础 |
| §7 bench harness | 复用 **W8 三铁律 + 三尺子**，为 W2 四方对标立尺子 |
| §8 AMK roofline floor | 用今天的公式复算出 report 里的 4794 µs，**判定预测值不可用作 baseline** |

### 9.2 自测六问（合上书能答才算过）

1. 说出 register / shared / L2 / HBM 的延迟量级比例，并解释"为什么重计算常常比重读取划算"。（§2.1）
2. 一个 warp 跨步访问 float，最坏放大几倍？**为什么是 8 而不是 32**？如果换成 fp16 呢？（§3.2、§3.3）
3. 手算：`in[idx*8]`，一个 warp 触及几个 sector、几个 cache line？（答案：32 个 sector、8 条 line）（§3.3）
4. 在 H100 上做带宽 benchmark，缓冲区至少要多大？为什么？（§5 陷阱 1）
5. 推导 decode 阶段的算术强度和 H100 脊点，并解释"量化为什么能加速"的真正原因。（§6.1）
6. AMK report 里 4794 µs 的 roofline floor 是怎么算出来的？为什么 719.8 µs 不能当 baseline？（§8.2）

### 9.3 完成标准 checklist

- [ ] `02_memory_access.cu` 编译通过、五个 kernel 全部 `correctness: OK`
- [ ] 拿到 coalesced vs strided 的**实测倍数**，填进 §4.2 的表（这是今天的核心产出数据）
- [ ] `ncu` 跑出 `l1tex__average_t_sectors_per_request...` 两个值（约 4 vs 约 32），**与 §3.3 手算/模拟结果对上**
- [ ] `coalesce_sim.cpp` 在 CPU 上跑通，能徒手预测任意 stride 的放大倍数
- [ ] `cuobjdump -sass` 里亲眼看到 `LDG.E` 与 `LDG.E.128` 的区别
- [ ] `-Xptxas -v` 确认 0 bytes spill（并知道 spill 意味着什么）
- [ ] `bench_harness.py` 跑通 torch 后端，输出含 `%peak` 和 memory/compute 判定
- [ ] AMK `small` 的 nsys 重跑完成，基线与 W7 report v1 对齐（或记录下不一致的原因）
- [ ] 能默写 `脊点 = 算力/带宽 ≈ 295 FLOP/Byte`，并用它判定任意算子的 bound 类型

---

### 附：今日一句话总结

**GPU 是一台算力过剩、带宽稀缺的机器，所以性能优化的主题不是"少算"而是"少搬 + 搬得整齐"：内存层级给出了「离 SM 越近越快」的价格表（1 : 25 : 200 : 500），warp 的 32 线程同步取数则要求地址必须挤在同一段 128 字节里，否则硬件按 32 字节 sector 整箱发货，白搬 8 倍；把这两条合起来，就能解释从 float4 向量化、SoA 布局、PagedAttention 块内连续，到量化加速、算子融合、FlashAttention 的全部动机——也能让你亲手算出 AMK report 里那个 4794 µs 的物理下界，并一眼看穿 719.8 µs 的预测值不可能成立。**
