# 阶段一 · W1 Day 3 —— shared memory 入门 + block 内规约求和（第一次亲手写 `__syncthreads()`）

> **一句话主题**：Day 1 解决"几千个线程怎么组织"，Day 2 解决"它们怎么高效取数据"——但这两天里，**线程之间从来没有说过一句话**。今天开始它们要协作：把 n 个数加成 1 个数。这件小学一年级的事，是 GPU 编程里第一道真正需要**通信 + 同步**的题。
>
> **今天的隐藏主线**：`__syncthreads()` 看起来只是一行"等一下"，但它背后连着一整条链——**为什么 block 内能同步 → 因为 block 不跨 SM（Day1 的映射事实）→ 为什么 block 之间不能同步 → 因为它们可能压根没被同时调度 → 所以巨核（megakernel）必须自己造跨 SM 同步 → 所以 AMK 在 H100 上卡在那里**。今天这行 `__syncthreads()`，是你理解自己那个科研课题瓶颈的第一块砖。
>
> **配套产出**：`cuda/03_block_reduce.cu`（5 个 kernel，含 2 个故意写错的反面教材 + 确定性对照实验）。
>
> **缓冲日说明**：计划里今天是缓冲日（学 2h，ACM 暑训优先）。所以这篇笔记按 **"§0–§4 是今天的保底、§5–§7 是读一遍留印象、Day4 会回来啃"** 的密度写。§8 给了 30 分钟极简路径，暑训真的重的话就走那条。

---

## 0. 先给六个学习目标问题一个"电梯答案"

读完全文回来，你应该能合上书复述这六条。

1. **为什么"求和"比"向量加"难一个数量级？**
   向量加是 **element-wise（逐元素）**：`c[i] = a[i] + b[i]`，第 i 个线程只碰第 i 份数据，**线程之间零交互**。求和是 **reduction（规约）**：n 个输入塌缩成 1 个输出，**多个线程必须往同一个地方写**——一旦出现"多写一"，就必然要面对两件事：**数据竞争**（谁先写谁后写）和**同步**（我读的时候你写完了吗）。GPU 编程的难度台阶，就在这里（§1）。

2. **`__shared__` 到底是什么东西？**
   是 **SM 芯片内部的一块 SRAM**，容量小（H100 每 SM 最多 228 KB）、速度快（~25 周期，比 global 快 20 倍）、**由你手动装填、block 私有、block 结束即消失**。它不是缓存——缓存是硬件自动决定放什么，shared memory 是**你决定放什么**。类比：L1/L2 缓存是"超市自动补货的货架"，shared memory 是"你自己带进车间的工具箱"（§2）。

3. **为什么求和必须 `__syncthreads()`？**
   因为 **一个 block 里的 8 个 warp 是异步推进的**。你以为 256 个线程"一起"执行 `s[tid] = sum;`，实际上 warp 0 可能已经跑到下一行了，warp 7 还在等调度器给它发指令。thread 0 如果不等就去读 `s[200]`，读到的是**上一个 block 遗留的垃圾或未初始化内存**。`__syncthreads()` 就是那句"全班都到齐了再往下走"（§3.1）。

4. **`__syncthreads()` 最容易犯的致命错误是什么？**
   **把它写进只有部分线程会进入的分支里**（包括用 `if (idx >= n) return;` 做边界检查）。barrier 要等 block 内**所有**线程到达，少一个就永远等不齐——官方文档原话是"很可能挂死或产生意外副作用"。**规约里的边界处理不是 `return`，是补单位元 0**（§3.4、§4.2）。

5. **规约的结果怎么验证"对不对"？**
   **不能用 `==`**。float 加法不满足结合律，换个加法顺序结果就变。今天你会亲眼看到：CPU 上用 float 串行加 1.34 亿个 `1.0f`，结果卡死在 **16777216 = 2²⁴** 再也加不动；而 GPU 的树形规约给出精确的 134217728。**树形规约不只是更快，它的误差界从 O(n·ε) 降到 O(log n·ε)——又快又准**（§5）。

6. **今天这堂课在引擎和 AMK 上分别落在哪？**
   引擎：**W2 你要手写的 CUDA RMSNorm，核心就是今天的 block reduce + 一次广播**（§7.3 给了完整骨架）；再往后 softmax、FlashAttention 的 online softmax，全是规约的变体。AMK：**`__syncthreads()` 是硬件免费提供的 block 内 barrier，而巨核需要的是 block 之间的 barrier——CUDA 没有免费提供，只能软件自旋实现，贵 2–3 个数量级。这就是 AMK 在 H100 上被跨 SM 同步拖累的物理原因**（§7.6）。

---

## 1. 问题背景：为什么"求和"是 GPU 上第一道真正的难题

### 1.1 从"互不相干"到"必须通信"

回头看你前两天写的 kernel，它们有一个共同的、你可能没意识到的奢侈条件：

```cuda
c[idx] = a[idx] + b[idx];   // Day1 vector_add
out[idx] = in[idx];         // Day2 copy
```

**每个线程只碰自己那一格，写的地址互不重叠。** 所以几千个线程可以完全乱序、任意快慢地执行，结果都一样。这类计算叫 **element-wise（逐元素运算）**，是并行编程里最舒服的一类，学名叫 **embarrassingly parallel（尴尬并行 / 天然并行）**——"尴尬"的意思是：并行得太容易了，容易到不好意思说自己在做并行编程。

现在把题目改成一个字：**求和**。

```
输入：in[0..n-1]
输出：一个数 sum = in[0] + in[1] + ... + in[n-1]
```

> **Reduction（规约）**：把一个大集合按某个**满足结合律的二元运算**塌缩成一个（或少量）结果的过程。求和是最典型的规约，同类的还有求最大值（max）、最小值、乘积、逻辑与/或、计数。
>
> **类比**：element-wise 是"全班每人做自己的卷子"，规约是"全班算总分"。做卷子可以各干各的；算总分必须有人把纸条收上来、汇总——**收纸条这个动作，就是通信；等所有人交完再汇总，就是同步。**

n 个输入、1 个输出，意味着**至少有一个地方要被多个线程写**。GPU 编程的所有难点（竞争、同步、确定性）都从这一句话里长出来。

### 1.2 第一反射写法为什么是错的——看它编译成了什么

几乎每个人的第一反应都是这样：

```cuda
// ❌ 错误示范：看起来天经地义，实际上错得很彻底
__global__ void reduce_wrong(const float* in, float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[0] += in[idx];   // 每个线程都往 out[0] 上加自己那份
}
```

跑一遍，你会发现结果**远小于**真值，而且**每次跑都不一样**。为什么？因为 `out[0] += x` 这一行**不是一条指令**。看它编译出来的 PTX（`nvcc -ptx -arch=sm_90 xxx.cu -o -`，节选）：

```ptx
ld.global.f32   %f2, [%rd5];       // ① 读：把 out[0] 的当前值读进寄存器
add.f32         %f3, %f2, %f1;     // ② 算：加上自己那份
st.global.f32   [%rd5], %f3;       // ③ 写：写回 out[0]
```

**读—改—写（read-modify-write）三条独立指令。** 两个线程同时执行时可以这样交错：

```
时刻    线程 A                        线程 B                     out[0] 真实值
 t0     读 out[0] → 拿到 10                                          10
 t1                                   读 out[0] → 也拿到 10          10
 t2     算 10 + 3 = 13                                               10
 t3                                   算 10 + 5 = 15                 10
 t4     写回 13                                                      13
 t5                                   写回 15                        15   ← A 的 3 被吃掉了
```

正确答案应该是 18，实际得到 15。**线程 A 的贡献被线程 B 的写覆盖掉了**，这叫 **lost update（更新丢失）**。

> **Race condition（数据竞争 / 竞态条件）**：两个以上的执行单元在**没有同步保护**的情况下访问同一块内存，且至少有一个是写操作，导致结果依赖于它们的相对执行时序。
>
> **它最恶心的性质是"薛定谔的 bug"**：小规模数据可能碰巧对，大规模就错；调试模式对，release 就错；今天对，明天换个驱动就错。**你不能靠"跑一遍看结果对不对"来排除竞争**——这是今天最该刻进骨头的一句话，§4.6 会给你正确的排查工具。

### 1.3 修正方案一：`atomicAdd` —— 对了，但代价是什么

CUDA 提供了原子操作，把"读-改-写"三条指令打包成硬件层面不可分割的一步：

```cuda
atomicAdd(out, in[idx]);   // 硬件保证：这个读-改-写序列不会被别的线程插进来
```

> **Atomic operation（原子操作）**：不可被中断、不可被观察到"中间状态"的操作。名字来自希腊语"不可分割"。硬件实现上，GPU 的 `atomicAdd` 是在 **L2 cache** 里完成的——请求发到 L2，L2 的原子单元排队串行执行读-改-写，然后返回。
>
> **类比**：普通的 `+=` 像三个人共用一个记账本，各自"看一眼余额、心算、写回去"，必然打架。`atomicAdd` 像给账本上了一把锁，一次只准一个人打开、改完锁上——**正确了，但所有人排成一队**。

`atomicAdd` 版本**结果是对的**（不会丢更新），但两个问题致命：

**问题一：并行度归零。** 1.34 亿个线程排队改**同一个地址**，硬件层面完全串行化。你花大价钱买的 132 个 SM，在这一刻退化成一条单车道。

**问题二（更隐蔽）：精度崩塌。** 串行累加意味着 `acc` 会长成一个很大的数，然后不断地"大数 + 小数"。float 只有 24 位有效尾数，当 `acc` 涨到 2²⁴ = 16777216 时：

```
16777216.0f + 1.0f = 16777216.0f      ← 舍入回原值，加了等于没加
```

**这不是 bug，是 IEEE 754 的正确行为**：16777216 和它的下一个可表示 float 之间间隔是 2.0，1.0 落在中间、按"就近偶数舍入"回到原值。所以你今天跑代码会看到 CPU 上 float 串行求和的结果**永远停在 16777216**，无论后面还有多少个 1 要加。

> 这个数字请记住：**float 的"加不动"阈值是 2²⁴ ≈ 1677 万**。在 LLM 里它随时会咬人——一个 4096 维的 hidden state 求平方和还好，但 batch×seq×hidden 上的全局统计量、大 batch 的 loss 累加、长序列 attention 分数求和，都可能踩到。**工业界的标准做法是：累加器用 fp32 甚至 fp64，即使数据本身是 fp16/bf16。** 你在 PyTorch 里见过的 `x.sum(dtype=torch.float32)`、`torch.mean` 内部的 `acc_type<T>`，都是为了这个。

### 1.4 正确解法的形状：分层规约

既然"全员挤一个地址"必然堵死，自然的解法是**分层汇总**——这也是所有真实系统的做法：

```
1.34 亿个数
   │  ① 每个线程私有累加（寄存器，零冲突）        ← 段一
   ▼
27 万个线程各持一个局部和
   │  ② block 内 256 个线程汇总（shared memory + __syncthreads）  ← 段二/三 ★ 今天的主题
   ▼
1056 个 block 各持一个局部和
   │  ③ block 之间汇总（第二趟 kernel，或每 block 一次 atomicAdd）
   ▼
1 个数
```

> **类比**：全校统计总分。不是让 5000 个学生排队去校长办公室报数（atomic 版本），而是**组长收本组 → 班长收本班 → 年级主任收本年级 → 校长**。每一层内部并行，层与层之间串行。**这个"分层规约"的结构，会从今天一路放大到 W6 的多卡 all-reduce——只是层数更多而已**（§7.5）。

而"班长收本班"这一步——**同一个 block 内的 256 个线程怎么把数据交给彼此**——就需要今天的两个新工具：**shared memory**（放纸条的桌子）和 **`__syncthreads()`**（"都交完了吗"）。

---

## 2. 核心原理一：shared memory —— 唯一由你手动管理的高速内存

### 2.1 它在物理上是什么、在哪

Day 2 那张价格表再抄一遍，今天要用第二行：

| 层级 | 延迟（周期，量级） | 容量 | 作用域 | 谁来管 |
|---|---|---|---|---|
| register（寄存器） | ~1 | 每 SM 64K 个 32-bit | 单线程私有 | 编译器 |
| **shared memory** | **~25** | **每 SM 最多 228 KB（H100）** | **单 block 共享** | **★ 你自己** |
| L1 / L2 cache | ~30 / ~200 | L2 50 MB（H100） | 全 GPU | 硬件自动 |
| global memory (HBM) | ~500 | 80 GB | 全 GPU + host | 你（分配）+ 硬件（缓存） |

**物理位置**：shared memory 是**刻在 SM 内部的 SRAM**。在 H100（compute capability 9.0）上，每个 SM 有一块 **256 KB 的统一数据缓存（unified data cache）**，硬件把它切成两半用——一部分当 L1 cache，一部分当 shared memory，**最多可以把 228 KB 划给 shared**。

这解释了 Day 1 学的那条映射事实的**物理来源**：

> **为什么"同 block 的线程能共享内存、能同步"？因为一个 block 整体住进一个 SM，它们摸的是同一块物理 SRAM。**
> **为什么"不同 block 不能直接通信"？因为它们可能住在 132 个 SM 里的任意两个，物理上根本不是同一块 SRAM。**

这不是"CUDA 规定不让"，是**芯片布线上就没有那条路**。理解这一点，你才不会问"能不能想办法让两个 block 共享一下"——除了 H100 新增的 cluster（§7.7），答案是物理上不能。

### 2.2 它不是缓存——这是最关键的区别

新手最容易混淆的一点：shared memory 和 L1 cache 用的是**同一块物理 SRAM**，速度也差不多，那要 shared memory 干嘛？

| | L1 cache | shared memory |
|---|---|---|
| 放什么 | **硬件**根据访问历史自动决定 | **你**用一行 `s[tid] = ...` 显式决定 |
| 能不能确定命中 | 不能，可能被别的数据挤掉 | 能，你放进去它就在那儿 |
| 能不能做线程间通信 | **不能**（你无法"往缓存里写给别人看"） | **能，这才是它的杀手锏** |
| 编程模型 | 透明 | 显式（要自己搬进搬出） |

**类比**：L1 cache 是超市自动补货的货架——你不知道明天上什么货，但常买的东西大概率有。shared memory 是**你自己带进车间的工具箱**——里面放什么完全由你决定，而且**整个班组都能从这个箱子里拿东西**。

> **一句话记住：shared memory 的价值有两个，一是"确定性的快"（scratchpad，暂存区），二是"线程间通信的信箱"。今天的规约用的是第二个价值；W2 的 tiled matmul 用的是第一个价值。**

### 2.3 怎么用：静态声明、动态声明，和它们编译成什么

**静态声明**（大小编译期已知，最常用）：

```cuda
__global__ void k() {
    __shared__ float s[256];      // 每个 block 独立拥有一份 256 float = 1 KB
    s[threadIdx.x] = threadIdx.x; // 每个线程写自己那格
}
```

**动态声明**（大小运行时才定，比如按 `hidden_dim` 开）：

```cuda
__global__ void k(int m) {
    extern __shared__ float s[];  // 注意：无长度、extern
    s[threadIdx.x] = 0.0f;
}
// 启动时用第三个参数传【字节数】
k<<<grid, block, m * sizeof(float)>>>(m);
//                ^^^^^^^^^^^^^^^^^ ★ 忘记 * sizeof(float) 是经典 bug：
//                                    分配少了 4 倍，越界写会静默破坏别的数据
```

**看底层**：`nvcc -ptx -arch=sm_90` 一下，会看到 shared memory 在 PTX 里是**一个独立的地址空间**：

```ptx
// 静态 shared 变量被放进 .shared 地址空间，编译期就分配好偏移
.shared .align 4 .b8 _ZZ1kvE1s[1024];

    st.shared.f32   [%rd3], %f1;    // 写 shared：注意是 st.shared，不是 st.global
    bar.sync        0;              // ← __syncthreads() 的真身
    ld.shared.f32   %f2, [%rd4];    // 读 shared
```

再往下一层看真实机器码（`cuobjdump -sass ./blockreduce | less`）：

```
STS   [R5], R7          // Store To Shared —— 专用指令，不走 global 的 LDG/STG 通道
BAR.SYNC 0x0            // 硬件 barrier 指令
LDS   R9, [R6]          // Load From Shared
```

> **这里有个值得记住的细节**：`LDS`/`STS` 和 `LDG`/`STG` 是**不同的指令、走不同的硬件通路**。这意味着 shared memory 访问**不消耗 HBM 带宽**（Day 2 那个"3.35 TB/s 的传送带"）。所以"把数据搬进 shared 复用 N 次"这笔交易，赚的不只是延迟，还有**带宽配额**——这是 W2 tiled matmul 和 W3 FlashAttention 全部收益的来源。

### 2.4 作用域与生命周期：block 私有，block 结束即销毁

```
grid  ┌── block 0 ──┐  ┌── block 1 ──┐  ┌── block 2 ──┐
      │ __shared__  │  │ __shared__  │  │ __shared__  │
      │ s[256] ①    │  │ s[256] ②    │  │ s[256] ③    │   ← 三份完全独立的物理内存
      └─────────────┘  └─────────────┘  └─────────────┘
      block 结束 → ① 被回收，下一个 block 复用同一块 SRAM（★ 所以里面是"上一家的垃圾"）
```

三条结论：

1. **每个 block 有自己独立的一份**，同名不同物。block 0 改 `s[5]` 和 block 1 的 `s[5]` 毫无关系。
2. **不会自动清零**。新 block 拿到的 shared memory 里是**上一个 block 留下的残留数据**。所以"忘了初始化就读"的 bug 不会得到 0，而是得到看似随机的旧值——**和 §1.2 的竞争一样，是"有时候对"的那类 bug**。
3. **kernel 返回后内容全部丢失**，不能用它在两次 kernel launch 之间传递数据。要跨 kernel 传，只能落 global memory。

### 2.5 容量：它和 occupancy 是一对冤家（工业调参的关键权衡）

H100（sm_90）的硬性数字，记住这几个：

| 项目 | 数值 |
|---|---|
| 每 SM 的统一 L1/shared 数据缓存 | 256 KB |
| 每 SM 最多可配置为 shared 的量 | 228 KB |
| **单个 block 能拿到的上限（需 opt-in）** | **227 KB** |
| **静态 `__shared__` 的固定上限（不 opt-in）** | **48 KB** |
| 每 SM 最大常驻线程数 / block 数 | 2048 / 32 |

超过 48 KB 必须显式申请（这是工业代码里非常常见的一段样板）：

```cpp
// 想给一个 block 用 100 KB 的动态 shared memory，必须先 opt-in，否则 launch 直接失败
CUDA_CHECK(cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,   // 只对【动态】 shared 生效
    100 * 1024));
my_kernel<<<grid, block, 100 * 1024>>>(...);
```

**关键权衡**：每个 SM 的 shared memory 总量是固定的，**一个 block 用得越多，SM 上能同时住的 block 就越少**：

```
SM 可用 shared = 228 KB
每 block 用 48 KB  → 228/48 = 4 个 block 常驻   → occupancy 中等
每 block 用 96 KB  → 228/96 = 2 个 block 常驻   → occupancy 减半
每 block 用  1 KB  → 不受 shared 限制（今天的规约就是这种）
```

> **Occupancy（占用率）**：SM 上实际常驻的 warp 数 ÷ 硬件支持的最大 warp 数。Day 1 §4.4 讲过它的意义——**常驻 warp 越多，一个 warp 卡在等内存时，调度器越有别的 warp 可切换，越能把访存延迟藏起来**（Day 2 §2.6 的 Little's Law）。
>
> **所以 shared memory 的用量是一个"两头拉扯"的旋钮**：用多了能减少 global 访问（省带宽），但会压低 occupancy（藏不住延迟）。**这个权衡在 W2 tiled matmul 上会第一次真正咬到你**——tile 开多大，就是在调这个旋钮。

查询工具（工业代码里常配 autotune 用）：

```cpp
int max_blocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, my_kernel, /*blockSize=*/256,
                                              /*dynamicSMemBytes=*/1024);
printf("每个 SM 能常驻 %d 个 block\n", max_blocks);
```

编译时也能顺手看：`nvcc -Xptxas -v` 会打印每个 kernel 用了多少 shared 和寄存器——**Day 2 你用它看 spill，今天开始也用它看 shared 用量**。

### 2.6 顺带一提：bank（Day 4 的正主，今天只留个印象）

shared memory 在硬件上被切成 **32 个 bank（存储体）**，每个 bank 每周期能服务一个 4 字节的访问。一个 warp 的 32 个线程如果**恰好落在 32 个不同的 bank** 上，就能一个周期全部完成；如果两个线程落在**同一个 bank 的不同地址**上，硬件只能排队 → **bank conflict（存储体冲突）**，慢一倍。

> **这和 Day 2 的合并访问是同一个道理的两层表现**：global memory 要求"地址挤在同一段 128 B 里"，shared memory 要求"地址分散到 32 个不同 bank 里"——**一个要求聚，一个要求散，因为一个受制于 DRAM 的整箱发货，一个受制于 SRAM 的多端口并发**。
>
> 今天代码里用的是**顺序寻址（sequential addressing）**版树形规约 `s[tid] += s[tid + off]`，活跃线程的 tid 是连续的 → 天然无冲突。教科书里常见的**交错寻址（interleaved addressing）**`s[2*off*tid] += s[2*off*tid + off]` 步长是 2/4/8...，会严重冲突。**Day 4 会把这两个版本都写出来实测。**

---

## 3. 核心原理二：`__syncthreads()` —— 今天真正的新东西

### 3.1 为什么必须有它：warp 之间是异步的

这是今天最重要的一段，请慢读。

你写 kernel 时脑子里的画面往往是："256 个线程**齐步走**，一起执行第 1 行，再一起执行第 2 行……"。**这个画面是错的，而且错得很致命。**

Day 1 学过：硬件真正调度的单位是 **warp（32 线程）**，一个 256 线程的 block = **8 个 warp**。SM 上有 4 个 warp 调度器，每个周期各挑一个"就绪"的 warp 发射指令。**8 个 warp 之间的推进速度完全不保证同步**——warp 3 可能因为它读的数据还在路上而挂起，warp 0 早就往下跑了十几条指令。

```
真实时序（示意，横轴是时间）：

warp0  ██ 算完 sum ██ 写 s[0..31] ██ ← thread0 在这里，已经想读 s[200] 了
warp1  ░░░░░ 等内存 ░░░░░ ██ 算 ██ 写 s[32..63]
warp2  ░░░░░░░ 等内存 ░░░░░░░ ██ 算 ██ 写 s[64..95]
...
warp7  ░░░░░░░░░░░ 还在等第一批数据 ░░░░░░░░░░░   ← s[224..255] 里还是上个 block 的垃圾
```

如果 thread 0（在 warp 0 里）不等就开始 `for (i=0..255) acc += s[i]`，它读到的 `s[200]` 是**未定义的残留值**。结果就是那种最讨厌的 bug：**小数据量偶尔对（warp 少、恰好都跑完了），大数据量必错；本地对、上服务器错。**

`__syncthreads()` 就是插在中间的那道闸门：

```cuda
s[tid] = sum;        // 每个线程往共享桌面上放自己的纸条
__syncthreads();     // ★ 闸门：所有 256 个线程都放完了，才准继续
if (tid == 0) { ... 读全部 256 张纸条 ... }
```

> **类比**：小组做接力题，每人把自己算的中间结果写在共用白板上，组长负责汇总。`__syncthreads()` 就是组长喊的那句 **"都写完了吗？写完的举手——好，8 个人都举了，我开始抄"**。没有这句话，组长可能在第 3 个人还没写完时就开始抄白板。

### 3.2 它在硬件里到底做了什么

`__syncthreads()` 不是一个函数调用（没有栈、没有跳转），它编译成**一条指令**：

```ptx
bar.sync 0;        // PTX
```
```
BAR.SYNC 0x0       // SASS（Volta+ 上有时显示为 BAR.SYNC.DEFER_BLOCKING）
```

硬件实现（理解到这个程度就够了）：

1. 每个 SM 有一组**硬件 barrier 资源**（PTX 层面每个 block 可以用 16 个具名 barrier，编号 0–15；`__syncthreads()` 固定用 0 号）。
2. block 被调度上 SM 时，硬件知道它有几个 warp（比如 8 个）。
3. 一个 warp 执行到 `BAR.SYNC 0` 时，**barrier 计数器 +1，然后这个 warp 被标记为"阻塞"，调度器不再给它发指令**。
4. 当计数器达到 8（block 的全部 warp 都到了），硬件**一次性把 8 个 warp 全部标记回"就绪"**，计数器清零。

**两个重要推论：**

- **它很便宜。** 没有软件轮询、没有内存往返，就是 SM 内部一个计数器。**空转成本大约几十个周期**——和一次 shared memory 访问同量级，比一次 global 访问（~500 周期）还便宜。
- **但"等最慢的那个"不便宜。** barrier 的真实代价不是指令本身，而是**让快的 warp 空等慢的 warp**。如果 8 个 warp 里有一个因为访存慢了 500 周期，其余 7 个就白等 500 周期。**这是"barrier 越少越好"的真正原因**——不是指令贵，是它把负载不均衡的代价显性化了。

### 3.3 它同时是一道"编译器屏障"

除了让线程等齐，`__syncthreads()` 还有一个容易被忽略的语义：**它保证 barrier 之前的所有共享内存/全局内存写操作，对 barrier 之后的同 block 线程可见**。

这句话有两层含义：

1. **对编译器**：不许把 barrier 前的 store 挪到 barrier 后面（也不许把后面的 load 提到前面）。没有这条，编译器完全可以把 `s[tid] = sum;` 这个"看起来没人用"的写操作延后甚至优化掉。
2. **对硬件**：保证写已经真正落到 shared memory / L1，不是还躺在写缓冲里。

> **对比一个常见误解**：有人以为加个 `volatile` 就能代替同步。**不能。** `volatile` 只解决"编译器别把它缓存在寄存器里"，**不解决"另一个 warp 到底跑到哪一行了"**。可见性和时序是两回事——`__syncthreads()` 两个都管，`volatile` 只管半个。

### 3.4 三条铁律（违反任何一条都是未定义行为）

**铁律一：block 内所有线程必须一致地到达同一个 barrier。**

```cuda
// ❌ divergent barrier —— 官方文档原话：「很可能挂死或产生意外副作用」
if (threadIdx.x < 128) {
    s[threadIdx.x] = x;
    __syncthreads();       // 另外 128 个线程永远不会到达 → 计数器永远凑不齐
}

// ❌ 同样错：早退（early return）本质上是同一个问题
if (idx >= n) return;      // 尾块里越界的线程直接退出
s[threadIdx.x] = in[idx];
__syncthreads();           // 语义已经不完整了

// ✅ 正确：barrier 放在所有线程都会执行的位置，用【单位元】处理边界
float v = (idx < n) ? in[idx] : 0.0f;   // 越界的线程贡献 0（加法单位元）
s[threadIdx.x] = v;
__syncthreads();                         // 256 个线程一个不少地到达
```

> **为什么"单位元"这个词值得记住**：规约的边界处理有一个统一范式——**用运算的单位元填充（padding with identity element）**。求和填 `0`，求积填 `1`，求 max 填 `-INFINITY`，求 min 填 `+INFINITY`，逻辑与填 `true`。
> 这在 attention 里你已经见过它的兄弟：**causal mask 给被遮住的位置填 `-inf`，正是因为后面要做 max 和 exp-sum 规约，`-inf` 是 max 的单位元、也让 `exp(-inf)=0` 成为 sum 的单位元。** 你 W6 写 mask 时可能只当它是"数值技巧"，今天你知道了它的真名。

**铁律二：`__syncthreads()` 只管 block 内，管不了 block 之间。**
它对别的 block 完全无效。跨 block 的可见性要用 `__threadfence()`（内存栅栏，只保证顺序不保证等待），跨 block 的"等待"CUDA 根本没有免费提供（§7.6）。

**铁律三：不能假设"warp 内 32 个线程天然同步"。**
见下一节——这是最贵的一条历史遗留坑。

### 3.5 Volta 之后的大坑：warp 内隐式同步不再成立

你在网上（尤其是 2017 年前的教程、包括 NVIDIA 官方那份著名的 `reduction.pdf`）会看到这种"优化"：

```cuda
// ⚠️ 这是 Pascal 时代的写法，Volta（2017）之后【不再安全】
if (tid < 32) {
    volatile float* vs = s;      // 用 volatile 骗编译器别优化
    vs[tid] += vs[tid + 32];     // 最后 6 轮省掉 __syncthreads()
    vs[tid] += vs[tid + 16];     // 理由：一个 warp 内 32 线程"锁步执行"，天然同步
    vs[tid] += vs[tid +  8];
    ...
}
```

它的前提是 **warp-synchronous programming（warp 同步编程）**——假定 warp 里 32 个线程永远处在同一条指令上。**Volta 架构引入了 independent thread scheduling（独立线程调度）后，这个前提被打破了**：每个线程有了自己的程序计数器，warp 内的线程可以分头执行、并且**不保证自动重新汇合**。上面这段代码在 Volta+ 上可能给出错误结果。

**正确的现代写法**用显式的 warp 级原语：

```cuda
// ✅ 现代写法：__shfl_down_sync 直接在寄存器之间交换，连 shared memory 都不用
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    //                       ^^^^^^^^^^ mask：显式声明哪些线程参与（0xffffffff = 全部 32 个）
    return v;   // 结果在 lane 0 手里
}
```

> **`__shfl_down_sync`（束内洗牌）**：让 warp 内的线程**直接读取另一个线程的寄存器**，不经过任何内存。`__shfl_down_sync(mask, v, off)` 表示"把 lane `i+off` 的 `v` 值取过来"。这是 GPU 上最快的线程间通信方式——**连 25 周期的 shared memory 都省了**。
>
> **名字里的 `_sync` 后缀是 CUDA 9 之后新增的**，正是为了应对 Volta：它强制 mask 里的线程在此处汇合。**看到没有 `_sync` 后缀的 `__shfl_down`（老 API，已废弃），就知道那段代码是 2017 年之前写的。**

**Day 4 你会把这个函数写出来**：工业级 block reduce 的标准结构是**两级**——先 `__shfl_down_sync` 在每个 warp 内规约（0 次 shared 访问），再把 8 个 warp 的结果放进一个只有 8 个元素的 shared 数组，最后由第一个 warp 再规约一次。**shared memory 的访问量从 256 次降到 8 次。**

### 3.6 `__syncthreads()` 的三个"带返回值"的变体（顺手认识一下）

```cuda
int  __syncthreads_count(int predicate);  // 同步 + 返回「有多少个线程的 predicate 非 0」
int  __syncthreads_and(int predicate);    // 同步 + 返回「是否所有线程都非 0」
int  __syncthreads_or(int predicate);     // 同步 + 返回「是否存在线程非 0」
```

它们把"同步"和"一次简单规约"合成一条指令。**实际用途**：稀疏计算里统计本 block 有多少个非零元素、迭代算法里判断"全 block 是否已收敛"——`if (__syncthreads_and(converged)) break;` 一行搞定，比自己写 shared 规约快也短。

### 3.7 成本小结

| 操作 | 量级 | 说明 |
|---|---|---|
| `__syncthreads()` 指令本身 | 几十周期 | SM 内硬件计数器，很便宜 |
| 因 barrier 造成的等待 | **取决于最慢的 warp** | 真正的成本在这里；负载越不均衡越贵 |
| `__shfl_down_sync`（warp 内） | ~几周期 | 寄存器直连，最快 |
| shared memory 读写 | ~25 周期 | 无 bank conflict 时 |
| 跨 block 的软件 barrier | **几千~上万周期** | 见 §7.6，比 `__syncthreads()` 贵 2–3 个数量级 |

---

## 4. 动手：`03_block_reduce.cu` 逐段拆解

### 4.1 三段式骨架（今后所有规约 kernel 都长这样）

```cuda
__global__ void reduce_shared_serial(const float* __restrict__ in,
                                     float* __restrict__ partial, size_t n) {
    __shared__ float s[BLOCK];
    const unsigned tid   = threadIdx.x;
    const size_t   gsize = (size_t)gridDim.x * blockDim.x;

    // ── 段一：grid-stride 私有累加（在【寄存器】里）───────────────────
    float sum = 0.0f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + tid; i < n; i += gsize)
        sum += in[i];

    // ── 段二：把私有结果暴露到 shared，让同 block 的伙伴能看见 ────────
    s[tid] = sum;

    // ── 段三：同步之后合并 ──────────────────────────────────────────
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < BLOCK; ++i) acc += s[i];
        partial[blockIdx.x] = acc;
    }
}
```

**四个"为什么这么写"，每一条都是今天的知识点：**

**① 为什么段一要先在寄存器里攒？**
按 Day 2 的价格表：register ~1 周期，shared ~25 周期。1.34 亿个元素，如果每个都往 shared 里写一次再规约，光 shared 访问就是 1.34 亿次。先在寄存器里把自己负责的 ~500 个元素加完，**shared 只碰 1 次**。
**这是所有高性能规约的第一条原则：把绝大部分工作压在最便宜的层级，昂贵的层级只碰一次。**

**② 为什么用 grid-stride loop 而不是"一个线程一个元素"？**
Day 2 已经见过这个写法。这里多一层理由：**它让 grid 大小和数据规模解耦**。grid 固定为 `SM数 × 8 = 1056` 个 block，`n` 再大也不用改配置，而且第二趟只需要规约 1056 个数（如果一个线程一个元素，第二趟要规约 52 万个数）。
**顺带：grid-stride 的访问是完全合并的**——每一步里相邻线程读相邻地址，符合 Day 2 的规矩。

**③ 为什么 `if (tid == 0)` 这个分支不需要担心 barrier？**
因为 barrier 在它**前面**，所有线程都到达了。分支里没有 `__syncthreads()`，所以不违反铁律一。

**④ 为什么这个版本"故意写得笨"？**
`tid == 0` 串行加 256 个数，意味着 **255 个线程在干等，并行度 1/256**。今天故意先写这个版本，是为了 **Day 4 换成树形规约时，你能量出一个具体的加速比**——没有 baseline 的优化是耍流氓（W8 的老规矩）。

### 4.2 边界处理：规约里不能用 `return`

这是今天最贵的一条实践规则，值得单独拎出来：

```cuda
// ❌ element-wise kernel 的习惯做法，搬到规约里就是灾难
size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= n) return;         // Day1/Day2 里这么写完全没问题
s[threadIdx.x] = in[idx];
__syncthreads();              // 尾块里退出的线程不会到达 → 铁律一被违反

// ✅ 规约的正确姿势：用单位元让所有线程"活到最后"
float sum = 0.0f;             // 加法单位元
for (size_t i = ...; i < n; i += gsize) sum += in[i];   // 没活干的线程自然保持 0
s[threadIdx.x] = sum;
__syncthreads();              // 所有线程一个不少
```

> 注意本文件用 grid-stride loop 后，边界问题**自动消失**了——循环条件 `i < n` 本身就是边界检查，没有元素可处理的线程 `sum` 保持 0，照样走完全程。**好的循环结构能让一整类 bug 不可能发生，这比"记得加检查"可靠得多**——这也是工业代码偏爱 grid-stride 的隐藏理由之一。

### 4.3 跑起来 + 预期输出

```bash
nvcc -O3 -arch=sm_90 -lineinfo -Xptxas -v -o blockreduce block_reduce.cu
./blockreduce
```

`-Xptxas -v` 会先打印一行（今天开始也关注 smem）：

```
ptxas info : Used 24 registers, 1024 bytes smem, 380 bytes cmem[0]
//                                ^^^^^^^^^^^ 我们的 __shared__ float s[256] = 1 KB
```

程序输出的关键部分（H100 上的量级，你的实测数字填进笔记）：

```
[CPU 参考] double 串行 = 134217728.000000
[CPU 参考] float  串行 = 16777216.000000   (相对误差 8.750e-01)
           ↑ 注意它卡在 16777216 = 2^24 附近了 ...

kernel                         time(ms)       eff.BW      %peak            result      rel.err
atomic per element (BAD)        152.400      3 GB/s        0.1%      16777216.0000    8.750e-01
shared + serial merge             0.290   1852 GB/s       55.3%     134217728.0000    0.000e+00
NO __syncthreads (BAD)            0.288   1864 GB/s       55.6%     134217728.0000    0.000e+00  ← ★ 看这里
tree + 2nd pass (det.)            0.174   3085 GB/s       92.1%     134217728.0000    0.000e+00
tree + block atomic               0.172   3122 GB/s       93.2%     134217728.0000    0.000e+00
```

**三个必须自己盯住的观察点：**

1. **`atomic per element` 慢了近千倍，而且结果只有真值的 1/8**——§1.3 讲的两个问题，一次性全看到。
2. **★ 漏掉 `__syncthreads()` 的版本"跑对了"。** 这不是运气好，这是**数据竞争的常态**：多数情况下 warp 之间的推进差距没大到出错，但它随时可能在别的输入规模、别的 GPU、别的驱动版本上翻脸。**这一行输出是今天最值钱的一课——"跑对了"不等于"是对的"。** 下一节给你正确的判定方法。
3. **serial merge 只跑到 ~55% 带宽峰值，tree 能到 ~92%。** 差距来自"255 个线程干等"。这个数字请记下来，Day 4 优化完再对比。（`%peak` 是 Day 2 立的规矩：报性能只说 ms 没有信息量。）

### 4.4 用 `compute-sanitizer --tool racecheck` 抓住那个"跑对了"的 bug

```bash
compute-sanitizer --tool racecheck ./blockreduce 20
#                              ^^^^^^^^^ 专门检测 shared memory 的数据竞争
#                                        （用小规模 n=1<<20 跑，它比正常执行慢几十倍）
```

对漏掉同步的那个 kernel，它会报出类似：

```
========= ERROR: Race reported between Write access at reduce_shared_nosync+0x150
=========     and Read access at reduce_shared_nosync+0x1a0 [8160 hazards]
=========     Address 0x7f... in shared memory
```

**这就是"确定性地证明代码有竞争"的方法**——它不依赖运气，是靠分析访问序列判定的。把注释掉的 `__syncthreads()` 加回去再跑，hazards 归零。

配套的另外两个工具（今天一并养成肌肉记忆）：

```bash
compute-sanitizer --tool synccheck ./blockreduce   # 抓 divergent barrier（铁律一的违反）
compute-sanitizer --tool memcheck  ./blockreduce   # 抓越界，含 shared memory 越界
```

> **工业习惯**：`compute-sanitizer` 的这三个 tool 应该进 CI。真实团队里，kernel 合入前跑一遍 racecheck 是标配——因为竞争 bug 一旦漏到线上，表现是"模型偶尔输出乱码"，排查成本是 kernel 开发的十倍。**你现在就把这个习惯建立起来，比会写 kernel 更能体现工程成熟度。**

### 4.5 `ncu` 该看哪几个指标

```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__inst_executed_op_shared_ld.sum,\
smsp__inst_executed_op_shared_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./blockreduce 24
```

| 指标 | 含义 | 今天该看出什么 |
|---|---|---|
| `dram__throughput...pct_of_peak` | HBM 带宽利用率 | 规约是纯 memory-bound，好的实现该逼近 90%+ |
| `smsp__inst_executed_op_shared_ld/st` | shared 读/写指令数 | serial 版比 tree 版多得多；Day4 换 shuffle 后会再降一个量级 |
| `l1tex__data_bank_conflicts...` | bank 冲突次数 | **顺序寻址版应该接近 0**——Day4 换成交错寻址版对比，这个数会跳起来 |

---

## 5. 深水区：规约的正确性——float 加法不满足结合律

### 5.1 亲眼看到的那个 2²⁴

程序输出里最刺眼的一行：

```
[CPU 参考] double 串行 = 134217728.000000
[CPU 参考] float  串行 =  16777216.000000
```

同样是把 1.34 亿个 `1.0f` 加起来，**float 串行版停在了 2²⁴，剩下 1.17 亿个 1 全部被舍入吃掉了**。

原因在 §1.3 讲过：float 的尾数只有 24 位（含隐含位），当累加器 `acc ≥ 2²⁴` 时，`acc` 和它的下一个可表示数之间的间隔已经 ≥ 2.0，加 1.0 落在中点、按 IEEE 754 的"就近偶数舍入"回到原值。

**而 GPU 的树形规约给出了精确值 134217728。** 为什么？因为树形规约**每一步都在加两个量级相近的数**：1+1=2、2+2=4、4+4=8……全程精确，一位都没丢。

### 5.2 树形规约不只是更快，还更准（这是个反直觉的结论）

浮点求和的误差界（`ε` 是机器精度，float 的 `ε = 2⁻²⁴ ≈ 5.96e-8`）：

| 求和方式 | 最坏误差界 | 直觉 |
|---|---|---|
| 串行累加 | **O(n · ε · Σ\|xᵢ\|)** | 每加一次都可能引入一次舍入，误差线性累积 |
| 树形 / 成对求和（pairwise） | **O(log₂n · ε · Σ\|xᵢ\|)** | 树高只有 log₂n 层，误差只累积 log₂n 次 |
| Kahan 补偿求和 | O(ε · Σ\|xᵢ\|)（与 n 无关） | 额外用一个变量追踪被丢掉的低位，代价是 4 倍加法 |

n = 1.34 亿时，`n = 1.34e8` vs `log₂n = 27`——**误差界差了 500 万倍**。

> **这个结论值得单独记住：在 GPU 上做规约，"为了并行而采用的树形结构"顺带把数值精度也改善了。这是少见的"性能和精度同向"的情况。**
>
> 反过来也解释了一个你迟早会遇到的现象：**同一个求和，GPU 结果和 CPU 结果不一样，而且往往是 GPU 更准。** 遇到这种情况不要默认"GPU 错了"——**先用 double 算个真值当裁判**。这正是 `03_block_reduce.cu` 里 `ref_double` 的作用。

### 5.3 三尺子在规约上怎么用（W8 纪律的延续）

你 W8 立的规矩是"三尺子验误差"。放到规约上，具体化成三条：

1. **绝对值不能用 `==` 判**，用相对误差 `|got - ref| / |ref| < 1e-6`（fp32 规约的经验阈值）。
2. **参考值必须用更高精度算**（double 串行，或者 CPU 上的 Kahan 求和）。**用 float CPU 串行当参考值是错误的做法**——它自己就是精度最差的那个。
3. **确定性单独测**：同一个 kernel 连跑 5 次，结果应当逐位相同；不同则说明实现里有非确定性来源（见下节）。代码里的 `[determinism]` 那一段就是干这个的。

### 5.4 工业锚点：确定性 vs 性能，一个真实的取舍

代码里两个树形版本，性能几乎一样，但有本质区别：

| | `tree + 2nd pass` | `tree + block atomic` |
|---|---|---|
| 怎么合并 block 结果 | 第二趟 kernel，**固定顺序** | 每 block 一次 `atomicAdd`，**顺序由硬件调度决定** |
| 可复现性 | ✅ 每次逐位相同 | ❌ 每次可能差最后几位 |
| 代价 | 多一次 kernel launch（~3–5 µs） | 无 |

跑一下代码里的 `[determinism]` 实验（用随机数据 `./blockreduce 27 rand` 更容易看出抖动），你会看到 atomic 版本的最后几位在跳。

**这在工业界是一个天天要做的取舍**：

```python
# PyTorch 里对应的开关
torch.use_deterministic_algorithms(True)
# 它会把内部用 atomic 的算子（index_add_、scatter_add_、部分 pooling 的反向等）
# 换成慢但确定的实现；换不了的直接抛错，逼你显式面对这个问题。
# cuBLAS 还需要额外设环境变量（因为 workspace 复用会影响 split-k 的归约顺序）：
#   CUBLAS_WORKSPACE_CONFIG=:4096:8
```

> **为什么这件事值钱**：训练时"同样的种子、同样的数据，两次跑出来的 loss 曲线不一样"，绝大多数情况就是 atomic 规约顺序导致的。它会让**调试**变得极其痛苦——你改了一行代码，loss 变了 0.001，你根本无法判断是改动的效果还是浮点噪声。
> **推理侧同理**：你 W7 的 AMK report 里 `correctness: FAIL, max abs err 9.2e-1 但 top-1 agreement 1.0000`，就是"数值有差但采样结果没变"的典型状态。**优化之前必须先把 baseline 的确定性和误差水平摸清楚**，否则后面所有加速都建在流沙上。

---

## 6. 常见陷阱清单（血泪版，按踩坑概率排序）

| # | 陷阱 | 症状 | 正解 |
|---|---|---|---|
| 1 | 漏 `__syncthreads()` | **有时对有时错**，换规模/换卡就错 | `racecheck` 确定性判定；别信"跑对了" |
| 2 | barrier 写进 divergent 分支 / early return | 挂死，或结果诡异 | 补单位元让所有线程走完；`synccheck` 检测 |
| 3 | **规约里只加了一个 barrier** | 树形规约结果偶尔偏小 | 每一轮循环内都要同步——这一轮读的是上一轮写的 |
| 4 | 忘了 shared 不会清零 | 结果里混入上个 block 的残留 | 所有 shared 元素都必须先被写过再被读 |
| 5 | 动态 shared 第三个参数忘乘 `sizeof` | 越界，破坏相邻数据 | `<<<g, b, n * sizeof(float)>>>`；`memcheck` 能抓 |
| 6 | 静态 `__shared__` 超 48 KB | launch 直接失败 | 改动态 shared + `cudaFuncSetAttribute` opt-in |
| 7 | BLOCK 不是 2 的幂却写树形折半 | 静默漏加一部分元素 | `BLOCK` 用 2 的幂；或补单位元 pad 到 2 的幂 |
| 8 | 用 float CPU 串行结果当参考真值 | "GPU 算错了"的假警报 | 参考值用 double 或 Kahan |
| 9 | 沿用老教程的 `volatile` warp 展开 | Volta+ 上偶发错误 | 用 `__shfl_down_sync` / `__syncwarp` |
| 10 | 在 barrier 后忘了再同步就复用 shared | WAR 竞争（读还没完就被覆盖） | 复用同一块 shared 前再加一个 `__syncthreads()` |

**第 10 条展开一下**（它比看上去隐蔽）：

```cuda
// 第一阶段用 s[] 做规约
s[tid] = a[tid]; __syncthreads();
... 规约 ...                       // 有些线程还在读 s[]
// ❌ 第二阶段想复用 s[]
s[tid] = b[tid];                   // 快的 warp 已经开始覆盖，慢的 warp 还在读上一阶段的值
// ✅ 中间必须再插一道
__syncthreads();
s[tid] = b[tid]; __syncthreads();
```
这叫 **WAR hazard（write-after-read，写后读冲突）**。规约循环里"每轮都要 sync"，本质上防的就是它。

---

## 7. 工业锚点：今天这块砖砌在哪

### 7.1 收口 W8 的钩子：Triton 的 `tl.sum` 底下就是今天这些东西

你 W8 写 Triton kernel 时，一行 `tl.sum(x, axis=0)` 就拿到了块内求和。今天你知道它替你做了什么：

| 你在 Triton 里写的 | 编译器替你生成的 CUDA 级操作 |
|---|---|
| `tl.sum(x, axis=0)` | ① warp 内 `__shfl_down_sync` 规约（寄存器级）<br>② 跨 warp 的 shared memory 暂存<br>③ 若干次 `bar.sync`<br>④ 自动选择 bank-conflict-free 的布局 |
| 块大小 `BLOCK` | 自动映射到 `num_warps × 32` 个线程 |
| 边界 `mask=` | 自动补单位元（sum 补 0、max 补 -inf）——**§3.4 讲的那件事，它替你做了** |

> **所以"Triton 帮我省了什么"的第三个答案出来了**（Day1 答的是索引拆分，Day2 答的是访存布局）：**它省掉了同步的正确性负担**。上面陷阱清单里的第 1、2、3、7、9 条，写 Triton 时根本不会遇到——因为它们发生在编译器生成的那一层。
> **代价是**：你也失去了对同步位置的控制。当你想做"warp specialization（warp 专业化分工，Hopper 上的主流优化）"这类精细调度时，Triton 的抽象会挡路——**这就是为什么真正的高性能 kernel（cuBLAS、FlashAttention-3、AMK 的巨核）还是要下到 CUDA/PTX。**

### 7.2 别造轮子：工业代码里的 block reduce 长什么样

**你手写是为了懂；上线用现成的。** 两个业界标准答案：

**① CUB（CUDA 官方模板库，已随 CUDA Toolkit 分发）**

```cpp
#include <cub/cub.cuh>

__global__ void reduce_with_cub(const float* in, float* out, size_t n) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp;   // CUB 自己要一块 shared

    float v = /* ... 每线程的私有累加 ... */;
    float block_sum = BlockReduce(temp).Sum(v);          // 一行搞定，内部自动选最优算法
    //   ★ 注意：返回值只在 threadIdx.x == 0 上有效，其余线程拿到的是垃圾
    if (threadIdx.x == 0) atomicAdd(out, block_sum);
}
```
CUB 内部会根据架构自动选 warp shuffle / raking 等不同策略，**几乎总是比手写快**。

**② PyTorch 自己的实现**（`aten/src/ATen/native/cuda/block_reduce.cuh`）
结构就是 §3.5 说的两级法：`WarpReduceSum` 用 `__shfl_down_sync`，然后把每个 warp 的结果写进一个 `shared[C10_WARP_SIZE]` 的小数组，最后由第一个 warp 再规约一次。**你 Day 4 要手写的就是这个结构**——写完之后回去读这个头文件，你会发现自己能逐行看懂。这是很好的"从会写到会读工业代码"的检验点。

### 7.3 ★ 最直接的落地：W2 你要手写的 CUDA RMSNorm，核心就是今天 + 一次广播

计划里 W2 的重头戏是"用原生 CUDA 重写 RMSNorm 接进引擎"。它的数学是：

```
y[i] = x[i] / sqrt( mean(x²) + eps ) * w[i]
```

中间那个 `mean(x²)` 就是**一次规约**。完整骨架（这段可以直接当 W2 的起点）：

```cuda
// 一个 block 负责一行（一个 token 的 hidden vector），N = hidden_dim（如 4096）
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ y,
                               int N, float eps) {
    const int row = blockIdx.x;               // 第几个 token
    const int tid = threadIdx.x;
    const float* xrow = x + (size_t)row * N;
    float*       yrow = y + (size_t)row * N;

    // ── 第一步：规约求平方和（今天学的东西，一字不差）─────────────
    float local = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {   // grid-stride 的行内版本，天然合并访问
        float v = xrow[i];
        local += v * v;                            // 累加器是 fp32，即使输入是 bf16（§1.3 的教训）
    }
    __shared__ float s[256];
    s[tid] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();                            // 每轮都要
    }

    // ── 第二步：广播 —— ★ 这里需要今天的【第二个 __syncthreads】────
    __shared__ float s_scale;
    if (tid == 0) s_scale = rsqrtf(s[0] / N + eps);  // 只有 thread0 算出这个标量
    __syncthreads();                                  // ← 没有它，其他线程可能读到旧值！
    const float scale = s_scale;                      // 现在全 block 都能安全读

    // ── 第三步：逐元素缩放（回到 element-wise，无需同步）────────────
    for (int i = tid; i < N; i += blockDim.x)
        yrow[i] = xrow[i] * scale * w[i];
}
```

**三个值得注意的设计点：**

1. **两个 `__syncthreads()` 的角色完全不同**：第一个是"等大家把纸条交齐"（规约前），第二个是"等 thread0 把结果写好"（**广播前**）。**规约 + 广播是一对孪生模式，今后每次见到"算一个全局标量再用它"的结构，都是这两步。**
2. **`x` 被读了两次**（求平方和一次、缩放一次）。第二次多半命中 L2，但如果 `N` 不大（≤ 几千），**把 `xrow` 也缓存进 shared memory 能省掉第二次 global 读**——这就是 W2 你要做的优化实验，也是"shared memory 的第一种价值（暂存区）"的首次落地。
3. **这个 kernel 完美示范了 Day 2 的结论**：它是纯 memory-bound 的（每个元素读 2 次写 1 次，算术强度约 1），所以优化方向是**减少访存**，不是减少计算。**判错方向 = 白干一周。**

### 7.4 通向 W3：softmax 的两次规约，和 FlashAttention 为什么要合成一次

标准 softmax 需要**两次规约**：

```
① m = max(x)              ← 规约 1（为了数值稳定，减去最大值防 exp 溢出）
② l = Σ exp(xᵢ - m)       ← 规约 2
③ yᵢ = exp(xᵢ - m) / l    ← 广播 + element-wise
```

两次规约 = 两遍读数据 + 至少两组 barrier。当这个"数据"是 attention 的 `S×S` 分数矩阵时（S=4096 时 32 MB），**两遍读就是 64 MB 的 HBM 流量**。

**FlashAttention 的 online softmax 干的事，就是把这两次规约合并成"边走边修正"的一次**：每来一个新块，用新的 running max 去**重缩放**已经累积的结果。你 W8 已经手写过它的 numpy 版本——**今天补上的是它在 CUDA 层面的意义：少一次规约 = 少一遍数据 + 少一组 barrier + 中间矩阵永不落 HBM。**

> 换句话说：**规约的次数，在 GPU 上是一个需要精打细算的资源。** 今天你第一次真切感到"一次规约要付出什么"，W3 学 online softmax 时才会真懂它为什么是个大发明。

### 7.5 同一个思想的放大：从 block 到多机

今天的"分层规约"结构会一路长大，**每一层的原理完全一样，只是通信介质不同**：

| 层级 | 通信介质 | 同步原语 | 延迟量级 |
|---|---|---|---|
| warp 内（32 线程） | 寄存器 | `__shfl_down_sync` | ~几周期 |
| **block 内（今天）** | **shared memory** | **`__syncthreads()`** | **~几十周期** |
| grid 内（block 之间） | global memory (HBM) | 二次 launch / `grid.sync()` / 软件自旋 | ~µs |
| GPU 之间（单机 8 卡） | NVLink | NCCL all-reduce | ~几十 µs |
| 节点之间 | InfiniBand | NCCL / MPI | ~百 µs |

> **NCCL 的 ring all-reduce 本质上就是今天这个分层规约的最外层。** 它之所以是"ring（环形）"而不是"树形"，是因为在网络层带宽比延迟更稀缺——**同一个问题，在不同的成本结构下会长出不同的最优解，这是系统设计里最迷人的部分。** 你 W6 学 continuous batching、大二学 tensor parallelism 时会反复见到这个模式。

### 7.6 ★ 为什么 block 之间不能同步 —— 直通你的 AMK 巨核课题

今天这行 `__syncthreads()` 之所以能"免费"，是因为 **block 不跨 SM**（Day 1 的映射事实）：8 个 warp 都在同一个 SM 上，硬件用一个计数器就搞定。

那能不能有个 `__syncgrid()`，让所有 block 等齐？**CUDA 默认没有提供，而且原因是根本性的：**

> **GPU 对 block 之间不做前进保证（no forward progress guarantee）。** grid 里有 10000 个 block，但 GPU 一次只能容纳（比如）2000 个常驻。剩下 8000 个要等前面的跑完腾位置。**如果 block 0 停下来等 block 9999，而 block 9999 因为没有空位根本还没被调度——死锁，而且是必然死锁。**

所以工业上只有四条路：

| 方案 | 怎么做 | 代价 |
|---|---|---|
| ① 拆成两次 kernel launch | kernel 边界天然是全局屏障 | 每次 launch ~3–10 µs + 中间结果必须落 HBM |
| ② Cooperative Groups | `cg::this_grid().sync()` + `cudaLaunchCooperativeKernel` | **要求 grid 里所有 block 同时常驻**，grid 大小被 occupancy 卡死 |
| ③ 手写软件全局屏障 | persistent kernel + global 计数器 + `atomicAdd` + `__threadfence()` + 自旋 | 走 L2/HBM，**比 `__syncthreads()` 贵 2–3 个数量级** |
| ④ （H100 新增）cluster 同步 | `cluster.sync()`，见 §7.7 | 只能同步同一个 cluster 内 ≤8 个 block |

**这直接就是 AMK（AutoMegaKernel）的核心矛盾**：

- 巨核（megakernel）的动机是**消灭方案 ①**——把整个模型融进一个 kernel，省掉几百次 launch 开销和中间张量的 HBM 往返（你 W7 Day4 算过这两笔账）。
- 但代价是：**层与层之间的依赖，原本由"kernel 边界"这个免费的全局屏障保证，现在必须自己实现**，只能走方案 ②/③。
- **而 ②/③ 在 H100 上尤其难受**：H100 有 132 个 SM，参与同步的 block 越多、跨的物理距离越远，自旋屏障的代价越大。

> **今天该记进 research 笔记的一句话**：AMK 论文点名的 future work"跨 SM 同步实测"，本质上就是**测方案 ③ 的软件全局屏障在 H100 上的真实代价**，以及它相对省下来的 launch 开销是否划算。
>
> **而这正好是一个你现在就能设计的实验**（W2 的【研】线可以直接用）：
> ```
> 写一个微基准：persistent kernel + 软件全局屏障，
> 扫 block 数 = 32 / 66 / 132 / 264 / 528，测单次全局屏障的 µs；
> 对照组：一次空 kernel launch 的开销（cudaEvent 测）。
> 交点在哪里 = "巨核什么时候开始不划算"的定量答案。
> ```
> **这个实验组里可能没人做过，你有 H100，一天就能出数据。** 它不需要读懂 AMK 全部代码，但结论直接命中论文的 future work——**这就是"独特价值"的具体形态**。

### 7.7 H100 的新东西：thread block cluster —— shared memory 的作用域第一次被扩大

Hopper（sm_90）加了一层新的层级，值得知道它存在（**对你的 AMK 课题尤其相关**）：

> **Thread Block Cluster（线程块簇）**：把最多 8 个 block 绑成一个 cluster，硬件保证它们**同时被调度到同一个 GPC（图形处理簇，物理上相邻的一组 SM）上**。cluster 内的 block 可以：
> - **互相访问对方的 shared memory**（DSMEM，distributed shared memory 分布式共享内存）
> - **用 `cluster.sync()` 做硬件级同步**——比软件自旋屏障快得多

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// 声明这个 kernel 以 2 个 block 为一簇启动（编译期指定）
__global__ void __cluster_dims__(2, 1, 1) my_kernel(...) {
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float s[256];

    s[threadIdx.x] = ...;
    cluster.sync();                                   // 硬件级的跨 block 同步（限 cluster 内）

    // 读隔壁 block 的 shared memory —— 在 Hopper 之前这是不可能的
    float* peer = cluster.map_shared_rank(s, cluster.block_rank() ^ 1);
    float v = peer[threadIdx.x];
    cluster.sync();                                   // 用完再同步一次，防 WAR
}
```

**为什么这个特性存在**：正是因为 §7.6 的痛点——大 kernel（巨核、大 tile 的 matmul）越来越需要跨 block 协作，NVIDIA 就在硬件上开了一条**有限范围**的口子。它不能同步全部 132 个 SM，但能同步物理相邻的一小撮。

> **对你的价值**：这是"H100 是 AMK 的反主场"这个判断可以被证伪或深化的地方。**AMK 的软件全局屏障如果能部分改用 cluster 机制，代价可能大幅下降。** 这不是今天要做的事，但**知道这扇门在哪**——当你 W2 重读 AMK 生成的 kernel 时，可以专门看一眼它有没有用 cluster。这是一个非常"内行"的观察角度。

---

## 8. 缓冲日执行建议（按 ACM 暑训的实际占用三选一）

**路径 A · 30 分钟极简保底**（暑训吃掉一整天时）
1. 把 `03_block_reduce.cu` 里的 `reduce_shared_serial`（kernel 1）**不看原文自己敲一遍**——三段式骨架必须能默写。
2. 编译跑通，确认结果对。
3. 读本笔记 §3.1（warp 异步）+ §3.4（三条铁律）。
> **保底线：能说清"为什么求和需要同步"和"为什么边界不能 return"。** 这两条是计划里今天的唯一硬指标。

**路径 B · 2 小时完整版**（正常缓冲日）
1. 路径 A 全部。
2. 把 `reduce_shared_nosync` 跑起来，**亲眼看到它"跑对了"**，然后用 `compute-sanitizer --tool racecheck` 抓出它的竞争（§4.4）。**这一步的体验价值最高——它会永久改变你对"测试通过"的信任度。**
3. 跑完整程序，把 §4.3 那张表的**实测数字填进笔记**（serial 版的 %peak 是 Day4 优化的分母）。
4. 读 §5（浮点结合律）+ §7.3（RMSNorm 骨架）。

**路径 C · 有余力**（暑训轻）
5. 读 §7.6，**把那个跨 SM 同步微基准的实验设计写进 `research_notes/amk_questions.md`**——它可以直接当 W2【研】线的动作项。
6. 提前把 `warp_reduce_sum`（§3.5 的 `__shfl_down_sync` 版）写出来，Day 4 直接用。

**【造】线（有时间才做，10 分钟）**：把今天的 `%peak` 数字记进 `bench_harness.py` 的结果表里，保持 Day2 立的"报性能必带 %peak"的纪律。

---

## 9. 串联表 + 自测题 + 完成标准

### 9.1 和已有笔记的串联

| 本笔记的位置 | 呼应 / 铺垫 |
|---|---|
| §1.1 规约 vs element-wise | 把 **Day1 vector_add / Day2 copy** 的"线程零交互"这个隐含前提第一次打破 |
| §2.1 shared 的物理位置 | 给 **Day1 §2.3「block 不跨 SM」** 一个物理解释：因为它们摸同一块 SRAM |
| §2.1 价格表 | 直接复用 **Day2 §2.1** 的 1 : 25 : 200 : 500 |
| §2.3 `LDS/STS` 不走 HBM | 解释 **Day2 §6.4** 说的"tiling 为什么省带宽"——省的是带宽配额，不只是延迟 |
| §2.5 shared vs occupancy | 承接 **Day1 §4.4 occupancy**、**Day2 §2.6 Little's Law**；铺垫 **W2 tiled matmul 的 tile 尺寸调参** |
| §2.6 bank conflict | **Day2 §9.1 预告过的那一条**，今天点到、**Day4 实测** |
| §3.4 单位元填充 | 解释 **W6 attention causal mask 为什么填 `-inf`** ——它是 max 规约的单位元 |
| §3.5 Volta 独立线程调度 | 为 **Day4 的 `__shfl_down_sync` 优化**扫清历史坑 |
| §5 浮点结合律 | **W8 三尺子**在规约场景的具体化；解释 GPU/CPU 结果不一致的常见原因 |
| §5.4 确定性 | 呼应 **W7 AMK report 的 `correctness: FAIL / top-1 1.0000`**：优化前先摸清 baseline 的误差水平 |
| §7.1 `tl.sum` 的真身 | 收口 **W8 Triton 钩子**：Triton 省掉的第三样东西是"同步的正确性负担" |
| §7.3 RMSNorm 骨架 | **W2 手写 CUDA RMSNorm** 的直接起点（规约 + 广播两段式） |
| §7.4 两次规约 | **W3 FlashAttention online softmax** 的动机铺垫 |
| §7.6 跨 SM 同步 | ★ **AMK 巨核课题的核心矛盾**；给出一个可立刻执行的微基准实验设计 |
| §7.7 cluster / DSMEM | H100 专属；**W2 重读 AMK kernel 时的观察角度** |

### 9.2 自测六问（合上书能答才算过）

1. 为什么 `out[0] += in[idx]` 是错的？把它编译出的三条 PTX 指令写出来，并画一个丢更新的时序。（§1.2）
2. shared memory 和 L1 cache 用同一块 SRAM，那 shared memory 存在的意义是什么？说出两个。（§2.2）
3. 为什么 `if (idx >= n) return;` 在 vector_add 里是对的，在规约里是灾难？规约的边界该怎么处理？（§3.4、§4.2）
4. 树形规约每轮循环里都要 `__syncthreads()`，为什么不能只在循环外同步一次？（§4.1、§6 第 3/10 条）
5. 为什么 CPU 上 float 串行加 1.34 亿个 1.0 会卡在 16777216？为什么 GPU 树形规约反而精确？两者的误差界各是多少？（§1.3、§5.2）
6. 为什么 CUDA 不提供 `__syncgrid()`？巨核想要跨 SM 同步有哪几条路，各自代价是什么？（§7.6）

### 9.3 完成标准 checklist（缓冲日缩减版，前 4 条是硬指标）

- [ ] `03_block_reduce.cu` 编译通过，`reduce_shared_serial` 结果与 double 参考值一致
- [ ] **能不看笔记默写出三段式骨架**（私有累加 → 写 shared → sync → 合并）
- [ ] 能讲清"为什么求和需要同步"（warp 之间异步，不是线程之间乱序）
- [ ] 能讲清"规约的边界处理是补单位元，不是 return"，并说出 3 个运算的单位元
- [ ] 亲眼看到漏 `__syncthreads()` 的版本"跑对了"，再用 `racecheck` 抓出它的 hazards
- [ ] 记下 serial 版的 `%peak` 实测值（Day4 优化的分母）
- [ ] `-Xptxas -v` 里确认 smem 用量 = 1024 bytes，理解它怎么影响 occupancy
- [ ] （有余力）读完 §7.6，把跨 SM 同步微基准写进 `research_notes/amk_questions.md`

---

### 附：今日一句话总结

**求和是 GPU 上第一件必须让线程说话的事，而线程一说话就带出两个新问题——数据放哪（shared memory：SM 内的 SRAM，快 20 倍、block 私有、由你手动装填，是"信箱"而不是"缓存"）、什么时候能读（`__syncthreads()`：一条 `bar.sync` 硬件指令，几十周期，代价不在指令本身而在"等最慢的 warp"，且必须被 block 内所有线程一致到达，所以规约的边界处理是补单位元而不是 return）；把这两样组合成"私有累加 → 写 shared → 同步 → 合并"的三段式，就得到了从 RMSNorm、softmax 到 FlashAttention、再到 NCCL all-reduce 的同一个骨架——而这个骨架的天花板，恰恰是 `__syncthreads()` 只在 block 内免费：一旦要跨 SM 同步，就得付软件自旋屏障几千周期的代价，这正是 AMK 巨核在 H100 上的核心矛盾，也是你手上那张 H100 能测出真数据的那道题。**

---

*笔记生成：阶段一 W1 Day 3（缓冲日）· 配套代码 `03_block_reduce.cu` · 上承 Day1 线程模型 / Day2 内存层级，下接 Day4 树形归约 + bank conflict*
