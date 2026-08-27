# 阶段一 · W2 Day 2 学习笔记（v2 优化版）—— occupancy：把"调参"变成"算出来"

> **对应规划**：`阶段一W2_CUDA进阶_手写算子接进引擎_逐日详细规划.md` → W2 Day 2（8/19 周三）
> **今日目标**：block size 选 128 还是 1024，凭什么？从"试几个看哪个快"升级成"**先算出理论上限，再用实测验证我算得对不对**"。
> **今日定位**：本周枢纽日——今天的扫描数据 = 你论文的 Figure 1 雏形。
> **前置**：Day 1 已跑通 JIT 桥（`CudaBackend.rmsnorm` 可调、16 用例全绿、`-Xptxas -v` 已开）；W1 Day2 的 `Desktop/02/bench_harness.py` 已存在（今天搬进仓库并升级）。

## v2 变更说明（本次联网核实与升级）

| # | 升级点 | 内容 |
|---|---|---|
| 1 | **配套 7 个代码文件** | 见下表：手算/API/ncu 三方对齐、扫描、画图全部脚本化，仿 Desktop\01 结构 |
| 2 | **新增 `occupancy_worksheet.py` 手算工作表** | 把 §1.2 的手算变成可复现脚本：输入 regs → 输出四限制 + occupancy + 波数 |
| 3 | **寄存器分配粒度给出来源** | CUDA 文档口径：occupancy 计算中寄存器按 warp 每份 256 个分配（每线程 8 的整数倍）——手算必须 `ceil(regs/8)*8`（§1.2 已讲，v2 补出处） |
| 4 | **ncu 指标名核实法** | 6 个指标名以官方 Metrics Reference 为准，并给出本机核实命令 `ncu --query-metrics \| findstr occupancy` |
| 5 | **sm_120 参数的诚实处理** | 网上对 Blackwell 消费级 shared/SM 的说法不一（100/128 KB）——**笔记坚持"运行时查询为准"，这正是 Day 2 的方法论本身**：不背表，问设备 |
| 6 | **NVIDIA 官方 Occupancy Calculator** | 补上 Excel 版计算器这一工业工具（API 是它的程序化形态），三方对齐其实有四件套 |
| 7 | **速查表 +3 行** | `--query-metrics` 用法、波数手算与实测 %peak 对不上、H100 上重跑要换 arch |

## 本文件夹内容（笔记 + 配套代码）

| 文件 | 用途 | 对应仓库位置 |
|---|---|---|
| `学习笔记_优化版.md` | 本笔记（v2） | — |
| `tools/occupancy_check.cu` | 三方对齐探针：四限制 + API + occupancy | 放 `推理引擎/tools/`，nvcc 编译 |
| `tools/occupancy_worksheet.py` | Python 手算工作表（输入 regs → 四限制/occupancy/波数） | 放 `推理引擎/tools/` |
| `tools/littles_law.py` | Little's Law 计算器（跨卡对比） | 放 `推理引擎/tools/` |
| `engine/bench_harness_updates.py` | HW 表 + 精确匹配 + calibrate_peak_bw | 合并进 `推理引擎/engine/bench_harness.py` |
| `bench/sweep_rmsnorm.py` | 28 点全配置扫描 | 放 `推理引擎/bench/` |
| `research/plot_sweep.py` | 第一张图（x=block, y=%peak, 每形状一条线） | 放 `推理引擎/research/` |
| `research/findings_template.md` | findings.md 模板 | 复制改名为 `findings.md` 使用 |

---

## 0. 今天的问题与完整闭环

### 0.1 一个问题的两种回答方式

昨天你写 kernel 时，block size 是"拍了脑袋选 256"的。今天要回答：**选 128 还是 1024，凭什么？**

这个世界上有两种工程师：

- **试跑派**：跑一遍 128、跑一遍 256、跑一遍 512，选最快的。快是快，但换一个形状、换一张卡，结论作废，一切重来。
- **计算派**：先用硬件常数算出理论上限（occupancy、带宽需求），**预测**哪个配置该最快，再实测验证预测。预测错了，说明模型缺了一项——**缺的那一项就是新知识**。

今天要把你变成第二种。这是"会写 kernel"和"会做 infra 研究"的分界线。

### 0.2 今天的完整闭环（一张图）

```
硬件常数（SM 数、寄存器、shared、带宽、延迟）
        │
        ▼
手算 occupancy ──► API 验证 ──► ncu 实测      （§1 三方对齐）
        │
        ▼
Little's Law：预测哪些形状喂得饱带宽、哪些喂不饱   （§2）
        │
        ▼
全配置扫描 28 组数据（先写预测，再实测）          （§3）
        │
        ▼
画图 + findings.md：最优块随形状变吗？            （§5）
        │
        ▼
      论文 Figure 1 的雏形 + Day 3 优化方向的依据
```

### 0.3 今日时间盒导航

| 时间块 | 内容 | 对应章节 |
|---|---|---|
| 【学】1h | occupancy 三个天花板 + 三方对齐 | §1 |
| 【学】1h | ★ 为什么 occupancy 高 ≠ 快（Little's Law） | §2 |
| 【学】1.5h | 全配置扫描 + 预测先行 | §3 |
| 【造】2.5h | bench_harness 入库（HW 表）+ 三个仓库债 | §4 |
| 【研】1.5h | 第一张图 + findings.md | §5 |
| 收尾 | 错误速查 + 自测三题 | §6 – §7 |

---

## 1. occupancy：三个天花板（1h）

### 1.1 occupancy 是什么：先讲"为什么需要它"，再讲定义

**延迟隐藏（latency hiding）是 GPU 高性能的根基**。GPU 上没有"缓存未命中就切线程"的操作系统——当一个 warp（线程束，32 个线程的硬件调度单位）发出一条读显存指令后，数据要 **400–800 ns** 才回来（这是显存延迟，约等于 CPU 主存延迟的 4–8 倍，且没有乱序执行引擎兜底）。在这段等待里，这个 warp 只能干等。

**GPU 的对策**：一个 SM（流式多处理器）上**同时驻留很多个 warp**。每个时钟周期，SM 里的 **warp 调度器（warp scheduler）** 从"就绪"（数据已到、可以继续算）的 warp 里挑一个发指令。等内存的 warp 在排队，就绪的 warp 顶上——**等待被其他 warp 的工作填满，这就是延迟隐藏**。

> **类比**：SM 是厨房，执行单元是厨师，warp 是一桌桌客人点的菜。一桌菜下锅后要等出锅（等内存），如果厨房里只有一桌，厨师就只能站着等；但只要有足够多桌客人在（驻留 warp 多），这桌等着、炒那桌，厨师就永远不会闲。**occupancy 就是"厨房里同时坐了几桌"占"最多能坐几桌"的比例。**

**定义**（严格版）：

```
occupancy = 一个 SM 上实际驻留的活跃 warp 数 / 该 SM 支持的 warp 上限
5060: warp 上限 = 1536 threads / 32 = 48 warps
```

区分两种 occupancy，别混用：

- **理论 occupancy（theoretical）**：由资源（线程槽位/寄存器/shared）算出的**最多能驻留多少**。今天手算和 API 得到的就是它。
- **实际 occupancy（achieved）**：运行中**实际平均活跃多少**，ncu 实测得到。它常常低于理论值——原因见 §1.4。

### 1.2 三个天花板（其实是四个）

一个 SM 能同时驻留几个 block？**每项资源各算一个上限，取最小值**。5060（sm_120）的资源清单：

| 资源 | 5060 每 SM | H100 每 SM | 说明 |
|---|---|---|---|
| 线程槽位 | 1536 threads = 48 warps | 2048 = 64 warps | 硬件 warp 槽位总数 |
| 寄存器文件 | 65536 个 32-bit 寄存器 | 65536 | 所有架构近几代都是 64K |
| shared memory | 100 KB（单 block 默认 48 KB，opt-in 99 KB） | ~228 KB | 你的 kernel 用不了几 KB |
| block 槽位（隐藏第四条） | **32 个 block** | 32 | 硬件寄存器式的上限，容易漏 |

> ⚠️ **关于 sm_120 的 shared/SM**：网上规格表说法不一（100 KB / 128 KB）。**本笔记的处理原则：不背表，问设备**——`occupancy_check.cu` 里 `cudaDevAttrMaxSharedMemoryPerMultiprocessor` 运行时查询的结果才是权威。这本身就是 Day 2 方法论的一次演练。

```
限制① warp 槽位：  max_threads_per_SM / BLOCK
       5060: 1536 / 256 = 6 个 block     ← 注意 5060 只有 1536！H100 是 2048
限制② 寄存器：      regs_per_SM / (regs_per_thread × BLOCK)
       5060: 65536 / (32 × 256) = 8 个 block   （regs_per_thread 从 -Xptxas -v 读）
限制③ shared：      shared_per_SM / smem_per_block
       5060: 102400 / 36 ≈ 2844（你的 rmsnorm 只用 BLOCK/32×4 + 4 = 36 B，基本不限制）
限制④ block 槽位：  hw 上限 32 个 block/SM（BLOCK ≥ 48 时不影响）

→ 实际驻留 = min(6, 8, 2844, 32) = 6 个 block = 1536 threads = 48 warps
→ occupancy = 48 / 48 = 100%
```

**两个反直觉点，今天必须亲眼看到**：

1. **BLOCK=1024 反而只有 66.7% occupancy**：1536/1024 = 1 个 block/SM = 32 warps，剩下的 512 个线程槽位装不下第二个 1024 的 block，白白浪费。全配置扫描里 1024 常常不是最快的，根源就在这里（§3 预测时要想到它）。
2. **BLOCK 太小也满不了**：BLOCK=32 时 1536/32 = 48 个 block，但硬件 block 槽位只有 32 → 最多 32×32 = 1024 threads = 66.7%。所以"block 越小越灵活"是错的，两头都亏。

**寄存器那条的工业细节——分配粒度（granularity）**：寄存器文件不是按"个"分的，而是**按 warp 为单位、每份 256 个寄存器（即每线程 8 的整数倍）**分配的（CUDA 文档在 occupancy 计算一节明确此口径，Best Practices Guide 的"Registers"一节亦指出每线程寄存器数是 occupancy 计算的关键因子）。所以 `regs_per_thread` 会先向上取整到 8 的倍数：24 → 24（正好），25 → 32。手算时用 `ceil(regs/8)*8` 才和硬件行为一致。这是"手算和 API 对不上"的第一大原因——配套的 `tools/occupancy_worksheet.py` 已把这步内置。

**怎么拿到真实的 regs_per_thread**：Day 1 你在 `engine/kernels.py` 里开了 `-Xptxas -v`，重新 import 时（或看昨天的编译输出）会看到类似：

```
ptxas info    : Compiling entry function '_ZN16rmsnorm_rereadILi256EE...' for 'sm_120'
ptxas info    : Used 24 registers, 36 bytes smem, 388 bytes cmem[0]
```

- `Used 24 registers` → 手算限制② 用 24：65536/(24×256) = 10.7 → 10 个 block（仍 > 6，不构成瓶颈）。
- 注意**模板有 4 个实例**，输出里 4 组数字，**对着 BLOCK 找自己的那组**——找错实例是常见失误。

### 1.3 三方对齐：手算 = API = ncu（配套工具已脚本化）

手算容易错（粒度、第四条限制、动态 shared）。工业界的三方对齐法：**手算、API、ncu 三者一致才算真懂**。配套文件 `tools/occupancy_check.cu` 是独立查询工具（不依赖 torch，nvcc 直接编译）：

```cuda
// tools/occupancy_check.cu —— 运行环境：本机 build_env.bat 环境（nvcc + MSVC）
// 编译：nvcc -arch=sm_120 tools/occupancy_check.cu -o occupancy_check.exe
//      ★ -arch 必须写！occupancy 是按“编译目标架构”算的，
//        不写默认老架构（如 sm_52），查出来的数字和你这张卡完全无关。
// 上 H100 时改 -arch=sm_90 再编一次。
#include <cstdio>
#include <cuda_runtime.h>

// 与 rmsnorm_reread 同构的最小探针 kernel：grid-stride 归约 + 同样的 shared 用量
// （真实 rmsnorm 还多 4B 的 inv_rms，差别在本机的 100KB/SM 面前可忽略）
template <int BLOCK>
__global__ void probe_kernel(const float* __restrict__ x, float* __restrict__ y, int H) {
    __shared__ float smem[BLOCK / 32];        // 与 reduce.cuh 相同
    float acc = 0.f;
    for (int i = threadIdx.x; i < H; i += BLOCK) acc += x[i];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = acc;           // 真正用上 smem，防止被编译器优化掉
    __syncthreads();
    if (threadIdx.x == 0) y[blockIdx.x] = smem[0];
}

template <int BLOCK>
void check(const char* tag) {
    int thr_sm = 0, regs_sm = 0, sh_sm = 0, max_blk_sm = 0;
    cudaDeviceGetAttribute(&thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cudaDeviceGetAttribute(&regs_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&sh_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&max_blk_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0);

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)probe_kernel<BLOCK>);   // 真实寄存器/smem 用量

    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(                    // ★ 官方 API 答案
        &max_blocks, probe_kernel<BLOCK>, BLOCK, 0 /*动态 smem 字节*/);

    const int gran = ((attr.numRegs + 7) / 8) * 8;                    // 寄存器分配粒度：向上取整到 8
    printf("[%s] regs/thread=%d (granularity→%d)  smem/block=%zu B\n",
           tag, attr.numRegs, gran, attr.sharedSizeBytes);
    printf("  限制①线程=%d  ②寄存器=%d  ③shared=%d  ④block槽=%d   → API=%d 个 block/SM\n",
           thr_sm / BLOCK,
           regs_sm / (gran * BLOCK),
           (int)(attr.sharedSizeBytes ? sh_sm / (int)attr.sharedSizeBytes : 99999),
           max_blk_sm, max_blocks);
    printf("  occupancy = %.1f%%\n\n", 100.0 * max_blocks * BLOCK / thr_sm);
}

int main() {
    check<128>("BLOCK=128");
    check<256>("BLOCK=256");
    check<512>("BLOCK=512");
    check<1024>("BLOCK=1024");
    return 0;
}
```

**预期输出形态**（数字以你的机器实测为准，这里给推理示例）：BLOCK=256 → `限制①=6 ②=10 ③=2844 ④=32 → API=6，occupancy=100%`；BLOCK=1024 → `限制①=1 → API=1，occupancy=66.7%`。

**Python 侧的手算工作表**（`tools/occupancy_worksheet.py`，把手算做成可复现流程）：

```python
# 输入 -Xptxas -v 读到的 regs，脚本打印四限制 + occupancy + 波数
for b in (128, 256, 512, 1024):
    worksheet(regs_per_thread=24, block=b, rows=16384)
# → 与 occupancy_check.cu 的输出、ncu 的实测三方对齐
```

**ncu 第三方**（Nsight Compute，NVIDIA 的 kernel 级 profiler）：

```powershell
# 只看 Occupancy 一节，并过滤到 rmsnorm kernel：
ncu --section Occupancy -k regex:rmsnorm python -c "from engine.backend import CudaBackend; import torch; x=torch.randn(256,4096,device='cuda'); w=torch.randn(4096,device='cuda'); CudaBackend(256).rmsnorm(x,w)"

# ★ v2 新增：指标名不要背，用官方查询命令核实（Windows 用 findstr）：
ncu --query-metrics | findstr /i occupancy
ncu --query-metrics | findstr /i dram
```

关键的六个指标名（在 Occupancy 一节，以官方 Metrics Reference 为准）：

| ncu 指标 | 含义 | 对应 |
|---|---|---|
| `launch__occupancy_limit_warps` | 线程槽位允许的 block/SM | 限制① |
| `launch__occupancy_limit_registers` | 寄存器允许的 block/SM | 限制② |
| `launch__occupancy_limit_shared_mem` | shared 允许的 block/SM | 限制③ |
| `launch__occupancy_limit_blocks` | 硬件 block 槽位 | 限制④ |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **实际**活跃 warp 占比 | achieved occupancy |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率 % | 和你的 `pct_of_peak_bw` 互相印证 |

**第四方（顺带知道）**：NVIDIA 官方还提供 **Excel 版 CUDA Occupancy Calculator**（随 toolkit 文档发布，[官方文档](https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html)）——手算表的工业祖传形态，`cudaOccupancyMaxActiveBlocksPerMultiprocessor` 是它的程序化版本。**手算 = API = ncu = 官方计算器，四者一致才算彻底懂。**

**记录模板**（贴进今天的笔记/`research/findings.md`）：

```markdown
## occupancy 三方对齐（5060, sm_120）
| BLOCK | ptxas regs | 手算 min(①,②,③,④) | API | ncu achieved |
|---|---|---|---|---|
| 256 | | 6 | 6 | |
```

> **为什么三方对齐才算真懂**：手算代表"理解模型"，API 代表"权威实现"，ncu 代表"物理现实"。任何两方对不上，都指向一个具体的知识缺口（粒度取整？动态 smem？找错模板实例？）——这正是 W1 纪律的延续：**不靠"我觉得对"，靠三方互证。**

### 1.4 为什么 achieved 常常低于 theoretical（顺带理解"波次"）

即使资源算出来能住 6 个 block，运行中平均活跃 warp 也可能只有 80%。两个主要原因：

1. **波次量化（wave quantization，尾波效应）**：GPU 是一波一波地调度 block 的。总波数 = ceil(总 block 数 / (每 SM 驻留数 × SM 数))。如果总 block 数不是整波，**最后一波只有部分 SM 有活干**，那些空着的 SM 拉低平均 occupancy。例：5060 上 BLOCK=256 每 SM 住 6 个 → 一波 156 个 block；`(32, 4096)` 只有 32 个 block → 一波装不满，26 个 SM 里有 6 个拿 2 个 block、20 个拿 1 个，只有 32×8 = 256 个 warp 活跃，远低于 48×26 = 1248 的上限。**这就是 decode 小形状 occupancy 崩塌的机制。**（`occupancy_worksheet.py` 同时输出波数，把这段推理变成数字。）
2. **负载不均**：行与行的工作量一样（本 kernel 没有），但真实算子常有（稀疏、变长）。

---

## 2. ★ 为什么 occupancy 高不等于快（1h，本日核心）

### 2.1 先定义"快"的判据：%peak 带宽

昨天之前你可能只看 `ms`。今天起换尺子：**有效带宽（effective bandwidth）**。

```
有效带宽 = bytes_moved / time
%peak    = 有效带宽 / 峰值带宽 × 100%
```

`bytes_moved` 用**访存下界**（这个算子至少要让多少字节跨过 HBM），不是实测流量。rmsnorm 的访存下界 = 读 x（rows×H×4）+ 写 y（rows×H×4）= `2*rows*H*4` 字节。为什么权重 w 不算：w 只有 H×4 = 16 KB，第一遍读进 L2 后就常驻了（5060 L2 = 33.5 MB），逻辑上被读 rows 次、但 HBM 只付一次钱——**下界算的是 HBM 流量，不是逻辑访问量**。

这个定义正好对应你 W1 Day2 的 `bench_harness.py` 里 `BenchResult.pct_of_peak_bw`——那把尺子今天正式启用。

### 2.2 Little's Law：喂饱带宽的充要条件

**利特尔法则（Little's Law）**是排队论的经典公式：L = λ × W——系统里平均排队的顾客数 = 到达率 × 平均逗留时间。

**GPU 上的形态**：

```
在途字节数（in-flight bytes）= 带宽 × 延迟
```

> **水管类比**：把显存带宽想成一根水管的流量（GB/s），延迟是水从水龙头流到出口的时间（ns）。要让水管**满负荷**出水，管子里必须始终有"流量 × 时间"这么多水在路上。管子半空，出口就时断时续；管子里同时流动的水量够大，出口才能一直满载。

**5060 的数字**：

```
在途字节需求 = 325 GB/s × 500 ns ≈ 162 KB
每线程只发一个 float（4B）→ 需要 162 KB / 4 B ≈ 40,600 个并发线程
5060 全卡线程上限 = 26 SM × 1536 = 39,936 个线程   ← 恰好差一点点！
```

**结论（今天最重要的一句话）**：在 5060 上，如果每个线程同时只有一个 4 字节的读在飞，**即使 occupancy 100%（39936 线程全上），也差 2% 喂不饱带宽**。occupancy 解决的是"有没有足够多 warp 可切换"，但**每个 warp 手里只有一个 load 时，在途字节总量由线程数决定，而线程数有硬上限**。解法不是提高 occupancy（已满），而是**让每个线程同时发更多字节**——这就是 Day 3 向量化访存（float4 = 16 B/线程，需求瞬间降到 ~10K 线程）的理论依据。

**用代码把这条法则变成随身计算器**（配套 `tools/littles_law.py`；跨架构对照——这本身就是你论文的素材）：

```python
# tools/littles_law.py —— 纯 Python，无依赖，任何环境可跑
def threads_needed(bw_gbs: float, latency_ns: float, bytes_per_thread: int):
    """喂饱带宽最少需要多少个“同时有活”的线程。"""
    inflight = bw_gbs * 1e9 * latency_ns * 1e-9      # 在途字节 = 带宽 × 延迟
    return inflight / bytes_per_thread, inflight

cards = {                              # (HBM 实测带宽 GB/s, 显存延迟 ns)
    "RTX 5060 Laptop (sm_120)": (325, 500),
    "H100 SXM (sm_90)":        (3350, 700),   # HBM3 延迟略高于 GDDR，量级相当
}
for name, (bw, lat) in cards.items():
    for bpt in (4, 16):                              # 4B = 1 个 float；16B = float4
        need, inflight = threads_needed(bw, lat, bpt)
        print(f"{name:<28} {bpt:>2}B/线程 → 在途 {inflight/1e3:>7.0f} KB，需 {need:>8.0f} 线程")
# 5060:  4B → 162 KB，需 ~40600 线程（全卡 39936，差 2%）
#        16B → 162 KB，需 ~10156 线程（25% occupancy 就够）
# H100:  4B → 2.3 MB，需 ~586000 线程（全卡 270336，差 2.2 倍！16B 才勉强够）
```

**注意这个对照结论**：H100 每线程 4B 时缺口是 2.2 倍，比 5060 严重得多——**"多少向量化才够"在不同架构上答案不同**，这正是一个跨架构可量化、可预测、可发论文的现象（你的选题 §2 的"跨架构泛化"里，这一项是 cost model 的输入之一）。

### 2.3 反向情况：已经贴住带宽时，occupancy 一文不值

**当 %peak 已经 > 85%**（有效带宽贴近实测峰值），kernel 是**带宽受限（memory-bound）**的：瓶颈是 DRAM 的物理宽度，不是延迟隐藏。这时候：

- 调 occupancy、调 block size、加 shared——**全部无效**，因为延迟早已被完全隐藏，改这些只是把"已隐藏的等待"再隐藏一遍。
- 唯一能提速的是**减少字节数**（算子融合、低精度、压缩）或换算法。

**你 W1 Day4 的归约实验就是这个现象**："教科书说 tree reduction 该快 30 倍、实测只快 1.02 倍"——为什么？归约的算术强度（arithmetic intensity，FLOP/字节）极低，v1 朴素版早就贴住了带宽：总时间 = 字节数/带宽 + 计算时间，而"计算时间"这一项被内存时间完全盖住了。把计算时间从"30 个单位"降到"1 个单位"，总时间几乎不变。**今天要把它从"撞见过"升级成"能预测"**：动手之前，先用算术强度判断这个 kernel 的瓶颈在哪。

**三态判断（决策树，背下来）**：

```
算术强度 AI = FLOP / 访存字节
脊点 ridge = 峰值算力 / 实测带宽     （5060: 41e12 / 325e9 ≈ 126 FLOP/B；H100: 989e12/3350e9 ≈ 295）

AI << ridge  → memory-bound（带宽受限）：看 %peak 带宽，够 85% 就别折腾 occupancy
AI >> ridge  → compute-bound（算力受限）：看 TFLOPS%，优化指令与寄存器
并行度不足    → latency-bound（延迟受限）：小形状/decode，%peak 极低，occupancy 和大 block 才有用
```

rmsnorm 的 AI：每元素约 4 FLOP（第一遍 fma 算 2、第二遍两次乘算 2），每元素 8 字节 → AI ≈ 0.5 ≪ 126 → **恒为 memory-bound**（除了 grid=1 的 decode 情形，那里并行度崩塌，转成 latency-bound）。这个判断直接写进今天的预测表。

### 2.4 类比总结（以及类比的边界）

> occupancy 是"餐厅坐了多少人"，在途字节是"厨房同时在做多少道菜"。**餐厅坐满但每人只点一道菜，厨师照样闲。让每人点四道菜（float4），比硬塞更多客人有效。**
>
> 类比的边界：真实瓶颈里"厨师"（执行单元）不是关键——rmsnorm 这种 kernel 里厨师本来就闲（计算量极小），真正的瓶颈是**传菜窗口（DRAM 带宽）的宽度**。所以更贴切的第二层类比：水管宽度不够时，加客人（occupancy）没用，得让每个客人一次端更多盘子（向量化），或者干脆少端几趟（融合/低精度）。

---

## 3. 全配置扫描：把"选 block"变成实验（1.5h）

### 3.1 实验设计：形状矩阵的意图

```python
SHAPES = [                       # (rows, H)  —— 覆盖 prefill 到 decode 的整个跨度
    (16384, 4096),   # prefill  B=8,T=2048      ← 大并行度、贴带宽区
    ( 2048, 4096),   # prefill  B=1,T=2048      ← 中等并行度
    (  256, 4096),   # batch decode B=256
    (   32, 4096),   # batch decode B=32        ← 一波都装不满（156 block/波）
    (    1, 4096),   # ★ 单请求 decode —— grid=1，最极端的情况
    (16384, 2048), (16384, 8192),   # hidden 维度的敏感性
]
BLOCKS = [128, 256, 512, 1024]
```

每个形状为什么在列（对着 §2 的模型看）：

| 形状 | 波数（BLOCK=256，一波 156 block） | 预期瓶颈 | 为什么值得测 |
|---|---|---|---|
| (16384, 4096) | 105 波 | memory-bound | 基线：所有 block 应贴近峰值 |
| (2048, 4096) | 13.1 波 | memory-bound，尾波可见 | 尾波损失开始显现 |
| (256, 4096) | 1.6 波 | memory-bound + 明显尾波 | 第二波只装 64% |
| (32, 4096) | 0.2 波 | latency 成分上升 | SM 有闲、block 内并行度成关键 |
| **(1, 4096)** | **1 block，26 SM 只有 1 个在干活** | **latency/launch 主导** | decode 问题的极端样本，论文核心 |
| (16384, 2048/8192) | — | memory-bound | H 改变每 block 工作量，验证"block 大小最优值是否随 H 变" |

28 组（7 形状 × 4 block）里的**每一项都要先预测、再实测**（§3.4）。

### 3.2 测法纪律：对着真实的 bench_harness 讲

W1 Day2 的 `timeit()`（真实代码）已经内置了四条纪律，**每一条都对应一个真实会踩的坑**，今天逐条理解为什么：

1. **warmup（预热）**：首次调用包含 cuBLAS/cuDNN 算法选择、JIT 编译、显存分配——不预热测的是"编译时间"不是"算子时间"。
2. **synchronize（同步）**：CUDA 是异步的（Day 1 §1.4）——不同步的话 CPU 计时器早停了，测出 0.01 ms 的假数据。
3. **中位数而非均值**：GPU 会被 ECC 刷新、其他进程、时钟波动打断，均值被长尾拖偏。p10/p90 用来量化抖动——**如果 p90/p10 > 1.2，先别信这个数，查环境**。
4. **flush_l2（冲 L2）**：H100 有 50 MB L2，小张量第二次跑全在 L2 命中，测出的"带宽"能超过 HBM 峰值（物理上不可能，说明测的是缓存不是显存）。每次迭代前用 1.5 倍 L2 大小的 scratch 写零，测的才是"冷数据"——线上真实场景。（这正是 ncu 默认 `--cache-control=all` 在做的事。）

笔记本额外纪律（工业现场同样适用）：**插电 + 高性能电源模式 + 关掉浏览器/微信**。GPU 睿频和功耗墙会随温度漂移，这是笔记本测 benchmark 最大的噪声源。

### 3.3 完整可运行扫描脚本（配套 `bench/sweep_rmsnorm.py`）

```python
# bench/sweep_rmsnorm.py —— 运行环境：本机 build_env.bat 环境
# 运行：python bench/sweep_rmsnorm.py   （从仓库根目录）
import itertools, json, torch
from engine.backend import CudaBackend
from engine.bench_harness import bench_op, hw_spec   # Day2 已搬进仓库（§4.1）

SHAPES = [                       # (rows, H) —— 与 §3.1 的表一一对应
    (16384, 4096), (2048, 4096), (256, 4096), (32, 4096), (1, 4096),
    (16384, 2048), (16384, 8192),
]
BLOCKS = [128, 256, 512, 1024]

rows_out = []
for (rows, H), block in itertools.product(SHAPES, BLOCKS):
    torch.manual_seed(0)                     # 固定种子：不同配置间数据分布一致，结果可复现
    x = torch.randn(rows, H, device="cuda")
    w = torch.randn(H, device="cuda")
    be = CudaBackend(block=block)
    r = bench_op(f"cuda/b{block}",
                 lambda: be.rmsnorm(x, w),
                 bytes_moved=2 * rows * H * 4)   # 访存下界：读 x + 写 y（w 常驻 L2 不计）
    rows_out.append(dict(rows=rows, H=H, block=block,
                         ms=r.ms_median, gbs=r.hbm_gbs, pct=r.pct_of_peak_bw))
    print(f"(rows={rows:>6}, H={H:>4}, block={block:>4})  "
          f"{r.ms_median*1e3:>9.1f} us  {r.hbm_gbs:>7.1f} GB/s  {r.pct_of_peak_bw:>6.1f}%")

out = dict(gpu=hw_spec()["name"], data=rows_out)
json.dump(out, open("research/data/sweep_rmsnorm_5060.json", "w"), indent=2, ensure_ascii=False)
print("-> saved research/data/sweep_rmsnorm_5060.json")
```

要点：JSON 落盘带 GPU 名（**可追溯**：以后 H100 跑同一脚本，两个文件一对比就是跨架构图）；`bytes_moved` 用下界（§2.1 的理由）；seed 固定（**可复现**）。

### 3.4 ★ 预测先行：今天最重要的纪律（不能跳）

**跑之前，先把下表填完。** 这是从"试跑派"变"计算派"的关键一步——**预测错了比预测对了更有价值**：错的地方就是你模型缺失的项，而那正是论文的内容。

| 形状 | 我预测的最优 block | 理由（写清推理链，不是猜） | 实测 | 对了吗 |
|---|---|---|---|---|
| (16384, 4096) | | | | |
| (2048, 4096) | | | | |
| (256, 4096) | | | | |
| (32, 4096) | | | | |
| (1, 4096) | | | | |
| (16384, 2048) | | | | |
| (16384, 8192) | | | | |

> **写预测时用 §1–§2 的模型**：occupancy 表（1024 只有 66.7%！）、波次、Little's Law。写完之后再往下看——下面是导师的示例推理，**先自己写，再对照，差异就是收获**。

**示例推理**（供对照，不是标准答案）：

- (16384, 4096)：105 波，尾波损失可忽略；memory-bound；128/256/512 都是 100% occupancy，1024 只有 66.7%。预测：256 或 512 最优，1024 略差（差 5–10%），128 与 256 接近。**若实测 1024 不差**，说明 66.7% occupancy 已够隐藏延迟——这也是要学的结论。
- (2048, 4096)：13 波，尾波装 64%…（2048/156 = 13.1）；memory-bound，整体比 16384 略低；排序同上。
- (32, 4096)：只有 32 个 block，**block 内并行度比 occupancy 更重要**：1024 线程/block × 32 block = 32768 线程 vs 256×32 = 8192 线程。预测：**1024 最优**（延迟隐藏靠大 block），且整体 %peak 明显下降。这是"小形状最优块反转"的第一个样本。
- (1, 4096)：grid=1，26 个 SM 只用了 1 个；%peak 预计**个位数**；block 越大越好（1024 最优）；时间被 launch 开销（3–5 µs）和单 SM 带宽主导。**若这条线异常低/噪声大，那不是 bug，是 decode 问题的本相**。
- (16384, 2048) vs (16384, 8192)：H 减半 → 每 block 工作量减半、波数翻倍；H 翻倍 → 反之。若最优 block 在这两条线之间移动，说明"最优配置依赖 H"——**这正是你选题要的效应**。

### 3.5 结果解读四步套路（拿到数据后照做）

1. **最优 block 随形状变吗？** 是 → 选题成立一半；否 → 回去想为什么模型预测错。
2. **(1, 4096) 的 %peak 有多低？** 把它量化（个位数 = decode 短归约问题的严重度，直接写进 proposal 的"问题"段）。
3. **最优点附近曲线陡还是平？** 平 = 选错代价小；陡 = 自动选参的方法更有价值。
4. **逐条对照预测表，把"错了的预测"列成清单**——每条错 = cost model 缺一项（今天的错项清单就是明天论文 method 的草稿）。

---

## 4. 【造】bench_harness 入库 + 三个仓库债（2.5h）

### 4.1 HW 表升级：为什么 5060 用实测 325 而不是理论 384

把 `Desktop/02/bench_harness.py` 搬进 `engine/bench_harness.py`，并把 HW 表换成两台卡的真实数据（完整代码在配套 `engine/bench_harness_updates.py`）：

```python
HW = {
    "NVIDIA H100 80GB HBM3":        dict(hbm_gbs=3350.0, bf16_tflops=989.0, sm=132, max_thr_sm=2048),
    "NVIDIA H100 PCIe":             dict(hbm_gbs=2000.0, bf16_tflops=756.0, sm=114, max_thr_sm=2048),
    "NVIDIA GeForce RTX 5060 Laptop GPU":
        dict(hbm_gbs=325.0,      # ★ 实测值（理论 384，实测 85%）—— 用实测做分母才诚实
             bf16_tflops=41.0,   # 待你自己用 GEMM 实测填准（规划里标了"待实测"）
             sm=26, max_thr_sm=1536),
}
```

**为什么用实测 325 做分母**：理论峰值（规格书上的 384 GB/s）永远达不到——DRAM 刷新、行激活/预充电、读写切换、ECC、功耗墙都在吃带宽。如果你拿 384 当分母，你的 kernel 永远显示"只有 85%"，你会误以为还有 15% 优化空间——**其实已经到顶了**。工业界通行做法：先用纯 copy kernel 实测一张卡的**可达峰值（achievable peak）**，拿它当 100%。这样"100% = 和 memcpy 一样好"，诚实且可操作。CPU 界的 STREAM benchmark 是同一传统的祖师爷。

**校准代码**（配套文件里，以后每台新卡先跑一次）：

```python
def calibrate_peak_bw(n: int = 1 << 28, iters: int = 20, device: int = 0) -> float:
    """1 GiB 的 D2D copy，读+写各 1 GiB。返回 GB/s —— 作为 %peak 的分母。
    为什么用 copy：它没有别的瓶颈，只有访存，是这台卡“现实可达”的带宽上限。"""
    x = torch.empty(n, dtype=torch.float32, device=device)   # 1 GiB ≈ 2^28 × 4B
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)                              # 预热：热起内存页与时钟
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)   # CUDA Event：GPU 侧计时
    s.record()
    for _ in range(iters):
        y.copy_(x)
    e.record(); torch.cuda.synchronize()
    return (2 * n * 4) / (s.elapsed_time(e) / 1e3 / iters) / 1e9   # 5060 预期 ≈ 325
```

顺带注意一个真实细节：原 `hw_spec()` 用 `k.split()[1] in name` 做模糊匹配——对 "GeForce" 关键字能撞上，但这种启发式很脆弱（同厂商多卡、名字变体都会翻车）。**工业建议：改成精确匹配 `k == name`，匹配不到就报错提示补 HW 表**——宁可显式失败，不要静默拿错尺子。

### 4.2 三个仓库债（30 min，别拖）

```powershell
# ① 空格文件名 → 永远 import 不了，改名
git mv "engine/online softmax.py" engine/online_softmax.py   # 未纳入 git 用 Rename-Item
#    原因：import online softmax 是语法错误——模块名必须是合法标识符。
#    Python 惯例（PEP 8）：模块名全小写 + 下划线（snake_case）。
#    改名后 grep 所有引用处："online softmax" / "online_softmax"，一并更新。

# ② tests/regression_test.py 里失效的 import
#    from 第一周.engine.model import ...   ← “第一周”是旧目录布局，已不存在
#    改为：from engine.model import ...（从仓库根运行 python -m pytest，sys.path 里有根目录）
#    排查方法：python -m pytest tests/ -q  用失败列表兜出所有失效 import，一次清干净。

# ③ 语料归位：训练语料属于 data/，不属于 engine/
Remove-Item engine/斗破苍穹.txt, engine/遮天.txt   # 先确认 data/ 已有副本再删！
```

这三件事背后的统一原则：**仓库里每样东西都该在它该在的地方，坏的东西会拖慢每次迭代**——空格文件 import 不了、失效路径让测试红着、语料混在引擎里让"引擎代码"和"数据资产"无法分开版本管理。

### 4.3 benchmark 三纪律（工业规范，贴在工作区）

1. **可复现**：固定种子、记录环境（GPU 名/驱动/CUDA/torch 版本/电源模式）。换个人跑得出一样的数。
2. **可对比**：同一把尺子（同一 HW 表、同一 bytes_moved 定义）才能跨配置、跨卡、跨后端对比。尺子改了，历史数据全部作废重测。
3. **可追溯**：数据落 JSON（不是截图）、文件名带 GPU、配 `gpu` 字段。三个月后你回来看得懂这数是怎么来的。

---

## 5. 【研】第一张图 + findings.md（1.5h）

### 5.1 画图（配套 `research/plot_sweep.py`，完整可运行）

```python
import json
import matplotlib.pyplot as plt

data = json.load(open("research/data/sweep_rmsnorm_5060.json", encoding="utf-8"))["data"]

by_shape = {}
for r in data:
    by_shape.setdefault((r["rows"], r["H"]), {})[r["block"]] = r["pct"]

plt.figure(figsize=(9, 5.5))
for (rows, H), pts in by_shape.items():
    xs, ys = zip(*sorted(pts.items()))
    plt.plot(xs, ys, marker="o", label=f"rows={rows}, H={H}")
    b_best = max(pts, key=pts.get)               # 标注每条线的最优点
    plt.annotate(f"{pts[b_best]:.0f}%", (b_best, pts[b_best]),
                 textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
plt.xlabel("block size"); plt.ylabel("% of measured peak BW (325 GB/s)")
plt.xticks([128, 256, 512, 1024]); plt.ylim(0, 105)
plt.grid(alpha=0.3); plt.legend(fontsize=8)
plt.title("RMSNorm config sweep on RTX 5060 (fp32)")
plt.tight_layout(); plt.savefig("research/figs/sweep_rmsnorm_5060.png", dpi=150)
plt.show()
```

### 5.2 看三件事（对着图回答，每题都写进 findings.md）

1. **不同形状的最优 block 是否不同？** → 若 (16384,·) 最优在 256/512 而 (32,4096)/(1,4096) 最优在 1024，**"最优配置随形状变化"被证实——选题成立了一半**。这正是"必须按形状选配置、而试跑法在形状空间爆炸"的证据。
2. **(1, 4096) 那条线的 %peak 有多低？** → 把它变成 proposal 里的一句话："单请求 decode 时 RMSNorm 仅达峰值的 X%，且 launch 开销占 Y%。"量化过的痛点才叫问题。
3. **最优点附近的曲线是陡还是平？** → 平 = 选错代价小（autotune 需求弱）；陡 = 选错代价大（你的自动选参方法价值高）。两种结果都有论文写法，但**先如实记录，再谈立场**。

### 5.3 findings.md（配套 `research/findings_template.md` 可直接改名使用）

```markdown
# W2 Day2 findings —— RMSNorm 配置扫描（RTX 5060, fp32）

## 三个问题
1. 最优 block 随形状变化吗？
   证据：（图/表）   结论：
2. (1, 4096) 的 %peak？意味着什么？
3. 最优点附近曲线陡/平？对自动选参价值的含义？

## 与预测的差异（论文 method 的素材）
| 形状 | 预测 | 实测 | 我模型缺了哪一项 |

## occupancy 三方对齐
（§1.3 的表格贴这里）

## 下一步
- Day 3：float4 向量化 + persistent kernel，重跑本扫描对比
- H100：同一脚本直接复用（HW 表已含 H100），产出跨架构第二张图
```

### 5.4 为什么今天是枢纽日

这张图一个动作服务三条线：**【造】线**得到 block 选择的依据（引擎从此按形状配 block）；**【研】线**得到 Figure 1 雏形；**【学】线**得到 occupancy/Little's Law 的实证。**"一个实验同时服务三条线"就是本周乃至整个暑假的工作方式。**

---

## 6. 常见错误与调试速查表（Day 2 版）

| 症状 | 根因 | 处理 |
|---|---|---|
| API 结果和手算差很远 | 编译 arch 不对（默认老架构！） | `nvcc -arch=sm_120`（H100 用 sm_90）；occupancy 按**编译目标**算 |
| 手算寄存器限制对不上 | 忘了粒度取整 | 用 `ceil(regs/8)*8`，再 × BLOCK（`occupancy_worksheet.py` 已内置） |
| ptxas -v 里找不到自己的数字 | 模板有 4 个实例，输出 4 组 | 对着 BLOCK 找对应那组 |
| 手算忘了第四条限制 | block 槽位 32/SM | BLOCK ≤ 48 时必查这一条 |
| ncu 指标名记不住 | 版本间指标名会变 | 别背：`ncu --query-metrics \| findstr /i occupancy` 现场查 |
| 手算 100% 但实测 %peak 与波数对不上 | 波次尾效 / 负载不均 / 内存等待 | 用 worksheet 的波数先算理论尾波；再看 `sm__warps_active` 与 DRAM 指标 |
| %peak 超过 100% | bytes_moved 算错（w 算了 rows 次）/ L2 没冲 | 下界只算 HBM 流量；确认 flush_l2 生效 |
| 笔记本数字漂移、每次不同 | 睿频/温度墙/后台进程 | 插电+高性能模式+关后台；看 p90/p10 比值 > 1.2 先别信 |
| (1, 4096) 每次测得都不一样 | launch 开销与时钟抖动主导 | 加大 iters、信 median、接受宽误差带（这是本相不是 bug） |
| ncu 报权限/打不开 | Windows 需管理员或开发者模式 | 管理员运行；或攒到 H100 上 profiler |
| 校准 copy 只有 200 GB/s | 后台 CUDA 进程抢带宽 / 没插电 | `nvidia-smi` 查占用；重跑校准 |
| hw_spec 匹配不到 5060 | 模糊匹配逻辑脆弱 | 改精确匹配 `k == name`（§4.1） |
| H100 上 occupancy_check 数字不对 | 二进制还是 sm_120 编的 | 上 H100 前用 `-arch=sm_90` 重编 |

---

## 7. 完成标准自测（三道题，先默写再对答案）

1. **手算：你的 rmsnorm 在 5060 上 BLOCK=256 时每 SM 驻留几个 block？和 API 对上吗？**
   *答案要点*：限制① 1536/256 = 6；限制② 用 ptxas 实际 regs（如 24，粒度后仍 24）→ 65536/(24×256) = 10.7 → 10；限制③ 36 B → 2844；限制④ 32。min = **6 个 block** = 1536 threads = 48 warps = **100% occupancy**。API（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`，编译 `-arch=sm_120`）应返回 6；ncu 的 `launch__occupancy_limit_warps` 显示 6 为瓶颈项。
2. **为什么 occupancy 100% 了还可能喂不饱带宽？5060 需要多少在途字节？**
   *答案要点*：Little's Law——喂饱带宽需在途字节 = 带宽 × 延迟 = 325 GB/s × 500 ns ≈ **162 KB**。occupancy 决定"有多少 warp 可切换"，但每线程只发 1 个 float（4B）时，全卡 26×1536 = 39936 线程最多在途 39936×4 ≈ 156 KB < 162 KB——**100% occupancy 也差一点**。解法：每线程同时发更多字节（float4/展开 → Day 3）。H100 上缺口更大（需 ~586K 线程，全卡 270K，差 2.2 倍）。
3. **指着扫描图说：哪个形状最反常？为什么？**
   *答案要点*：(1, 4096)：%peak 个位数——grid=1，26 个 SM 只有 1 个干活，延迟与 launch 开销主导；(32, 4096)：一波装不满、最优 block 可能翻转到 1024（block 内并行度成为主要矛盾）；大形状 (16384, ·) 里 1024 因 66.7% occupancy 往往略输 256/512。"最优块随形状变"若成立 = 选题成立一半。

---

## 8. 今日产出清单 & 明日预告

**产出**（全部完成才算过关）：

- [ ] `research/data/sweep_rmsnorm_5060.json` + `research/figs/sweep_rmsnorm_5060.png` + `research/findings.md`
- [ ] `engine/bench_harness.py` 入库（HW 表含两卡实测数据 + `calibrate_peak_bw`）
- [ ] occupancy 三方对齐记录（手算 / API / ncu / 官方计算器）
- [ ] 三个仓库债清完（空格文件改名、regression_test 修复、语料移出 engine/）
- [ ] 预测表在跑扫描**之前**填完

**明日预告（Day 3）**：今天算出的"每线程要多发字节"明天兑现——float4 向量化访存 + persistent kernel（一个 block 常驻处理多行，消灭 decode 场景的 launch 开销）。今天的扫描基线数据，就是明天优化的对照系。

---

## 附 A：术语速查表（Day 2）

| 名词 | 一句话解释 |
|---|---|
| occupancy（占用率） | 活跃 warp 数 / SM 支持的最大 warp 数；分理论值（资源算）与实际值（ncu 测） |
| theoretical / achieved occupancy | 理论上限（手算/API）/ 运行中实测（ncu），后者常因波次低于前者 |
| warp scheduler（线程束调度器） | SM 里每周期从就绪 warp 中挑一个发指令的硬件单元 |
| latency hiding（延迟隐藏） | 用别的 warp 的工作填满内存等待期，让执行单元不空转 |
| Little's Law（利特尔法则） | L = λW；GPU 形态：在途字节 = 带宽 × 延迟 |
| in-flight bytes（在途字节） | 已发出、尚未完成的访存总量；喂饱带宽的下界 |
| MLP（内存级并行，memory-level parallelism） | 每个线程同时有多少个未完成的访存 |
| roofline model（屋顶线模型） | 用"带宽屋顶 + 算力屋顶"给 kernel 性能划上界 |
| effective bandwidth（有效带宽） | bytes_moved / time；衡量"离硬件极限多远" |
| arithmetic intensity（算术强度） | FLOP / 访存字节；与脊点比较判定瓶颈类型 |
| ridge point（脊点） | 峰值算力 ÷ 峰值带宽；AI 与它比较出 memory/compute-bound |
| memory-bound / compute-bound / latency-bound | 带宽受限 / 算力受限 / 延迟受限（并行度不足） |
| wave quantization（波次量化/尾波效应） | 总 block 数不是整波时，最后一波只有部分 SM 有活 → achieved occupancy 下降 |
| L2 flush（二级缓存冲刷） | 每次迭代前写满大缓冲把 L2 冲干净，测"冷数据"而非缓存命中 |
| CUDA Event | GPU 侧时间戳，测异步执行时间必须用它（CPU 计时器会撒谎） |
| median / p10 / p90 | 中位数 / 10% 分位 / 90% 分位：抗长尾的统计量 |
| ptxas | NV 的 GPU 汇编器；`-Xptxas -v` 打印寄存器/shared 用量 |
| ncu（Nsight Compute） | NVIDIA kernel 级 profiler；`--section Occupancy` 看占用率 |
| `--query-metrics` | ncu 列出全部可用指标名的命令——指标名不要背，现场查 |
| register granularity（寄存器分配粒度） | 寄存器按 warp 每份 256 个分配 → regs/thread 取整到 8 的倍数 |
| CUDA Occupancy Calculator | NVIDIA 官方 Excel 版占用率计算器；API 是它的程序化形态 |
| measured peak（实测峰值/可达峰值） | copy kernel 实测带宽，作为 %peak 的诚实分母 |
| SASS | GPU 真实机器码；ptxas 的产物 |

## 附 B：参考与延伸

- CUDA C++ Programming Guide（occupancy calculator 一节 + 各代 compute capability 资源表）：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- occupancy API 官方文档（`cudaOccupancyMaxActiveBlocksPerMultiprocessor`）：https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
- CUDA Occupancy Calculator（官方 Excel 版）文档：https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html
- Nsight Compute 文档（Occupancy 一节与 Metrics Reference，指标名以它为准）：https://docs.nvidia.com/nsight-compute/
- Roofline 模型原论文：Williams, Waterman, Patterson, *"Roofline: An Insightful Visual Performance Model for Multicore Architectures"*, CACM 2009 —— https://dl.acm.org/doi/10.1145/1498765.1498785
- 延迟隐藏的经典定量分析：V. Volkov, *"Understanding Latency Hiding on GPUs"*, UC Berkeley Tech Report —— https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html
- CPU 界同传统的带宽校准基准：STREAM benchmark —— https://www.cs.virginia.edu/stream/
- 本机快速核对硬件资源（Python 侧）：`python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, p.multi_processor_count, p.L2_cache_size)"`
