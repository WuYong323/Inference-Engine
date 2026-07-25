# Day 5 · CUDA 链路就位 + 第一个原生 kernel + AMK 保温

> **一句话主题**：同一个 vector-add，Triton 的"一个 program 管一块"和 CUDA 的"一个 thread 管一个元素"，差异背后到底是什么？
> **今天不是学 CUDA**（那是 W1 的正课），今天只做两件事：① 把 H100 上"能编译能跑 CUDA"这条**链路**焊通；② 亲手写一个最小 kernel，给 W1 埋下**对照的钩子**——带着问题进 W1，比冷启动快一倍。

---

## 0. 先摆正今天的定位：为什么"链路就位"本身就是一个里程碑

新手常忽略一件事：**写出 kernel 从来不是第一道坎，让 kernel 能编译、能上 GPU 跑、能验证结果，这一整条链路才是。** 在 H100 这种共享集群上，这条链路有五个环节，任何一环卡住，你连"Hello World"都跑不出来：

> **登录 → module load（加载 CUDA 环境）→ nvcc（编译）→ 运行 → 验证结果**

今天的**完成标准**就一句话：**这五环里任何一环，你都不需要现查资料。** 这不是记命令，是把"我在 H100 上有一条随时能用的 CUDA 通路"变成你的肌肉记忆。以后你每天在这条路上跑几十遍，它必须像开灯一样自然。

所以今天的心态是：**不求写出多牛的 kernel，只求把路修通、把对照的问题问对。**

---

## 1. 先建心智模型：GPU 编程 = 指挥一大群工人同时干活

在碰任何代码前，先把 GPU 的世界观装进脑子，否则后面每个 API 都像天书。

> **CPU vs GPU 的本质区别**：CPU 是**几个博士生**——每个都很强，适合干复杂的、有先后顺序的活。GPU 是**几千个小学生**——每个只会做简单算术，但人多，适合"同一道简单题、换不同数字，几千道一起做"。

vector-add（向量加法）`c[i] = a[i] + b[i]`，就是最典型的"几千道简单题"：每个 `i` 之间毫无依赖，天生适合几千个"小学生"一人算一个。**这就是为什么 vector-add 是所有 GPU 教程的第一课**——它把"数据并行"这件事扒到最干净。

### 1.1 CUDA 的三级组织：Thread → Block → Grid

几千个工人不能一盘散沙，CUDA 用**三级层次**把他们编好队。这是理解一切的地基：

> **Thread（线程）**：最小的干活单位，**一个 thread 干一个元素**。在 vector-add 里，thread #37 就负责算 `c[37] = a[37] + b[37]`。它是那个"小学生"。

> **Block（线程块）**：一组 thread 打包成一个 block（比如每 256 个 thread 一个 block）。**同一个 block 里的 thread 能协作**——共享一块高速内存（shared memory）、能相互同步。类比：一个"班级"，同班同学坐一个教室、能互相借草稿纸。block 是被整体调度到 GPU 一个物理核心（SM）上执行的单位。

> **Grid（网格）**：所有 block 合起来就是一个 grid，对应一次 kernel 启动的"全部工人"。类比：整个"年级"。

> **kernel（核函数）**：你写的、要让几千个 thread 每人跑一遍的那段函数。注意——**你只写一份 kernel 代码，但它会被成千上万个 thread 各自执行一次**，每个 thread 靠自己的编号去认领不同的数据。这个模型叫 **SIMT（Single Instruction, Multiple Threads，单指令多线程）**：同一条指令，一大群线程拿不同数据同时跑。

### 1.2 灵魂公式：全局索引 `blockIdx.x * blockDim.x + threadIdx.x`

既然几千个 thread 跑的是**同一份代码**，每个 thread 怎么知道"我该算哪个元素"？靠它的身份证——三个内置变量：

- `threadIdx.x`：我在**自己 block 内**排第几（班级里的学号，0 ~ blockDim.x-1）。
- `blockIdx.x`：我所在的 block 在 grid 里排第几（第几个班）。
- `blockDim.x`：每个 block 有多少 thread（每班多少人）。

把"第几班 × 每班人数 + 班内学号"一算，就是我在全年级的唯一编号：

```
全局索引 idx = blockIdx.x * blockDim.x + threadIdx.x
```

**类比**：全校学生排唯一学号 = `班级号 × 每班人数 + 班内座位号`。比如每班 256 人，第 3 班（blockIdx=3）第 5 座（threadIdx=5）→ 全局第 `3*256+5 = 773` 号，他就负责 `c[773]`。

**这一行是 CUDA 编程的第一个"啊哈"时刻**：几千个线程跑同一段代码，靠这个公式把自己映射到不同数据上，实现了"分工"。

---

## 2. 主线第一步：H100 工具链五环确认

### 2.1 `module load CUDA/12.4` —— 为什么要这一步

> **module（环境模块系统）**：HPC/超算集群上管理软件版本的工具。一台共享集群上常常**同时装了 CUDA 11.8、12.1、12.4 好几个版本**（不同用户、不同项目依赖不同版本）。默认什么都不给你选，`module load` 就是告诉系统"这次会话我要用 12.4 这套"，它会把对应的 `nvcc`、库路径、环境变量一次性配好。

**为什么不能省？** 不 load，你敲 `nvcc` 要么"command not found"，要么用到一个版本不对的编译器，编出来的东西和 H100（Hopper 架构，计算能力 sm_90）对不上。**这是链路第二环，也是新手最容易忽略、最容易踩的一环。**

```bash
# 环境: H100 集群登录节点, 已通过 ssh 登录 (链路第一环)
module load CUDA/12.4        # 加载指定版本 CUDA 工具链 (链路第二环)
module list                  # 确认已加载, 会列出当前 session 生效的所有 module
```

### 2.2 `nvcc --version` —— 确认编译器就位

> **nvcc（NVIDIA CUDA Compiler，英伟达 CUDA 编译器）**：CUDA 的专用编译器。它做一件很妙的事——**把一份 `.cu` 文件里的"主机代码（CPU 跑的）"和"设备代码（GPU 跑的）"拆开**：CPU 那部分丢给系统的 g++ 编译，GPU 那部分（kernel）自己编译成 GPU 能认的机器码（PTX/SASS），最后缝合成一个可执行文件。

```bash
nvcc --version               # 应打印 "release 12.4"; 打印不出来 = module 没 load 成功
```

看到 `release 12.4` 字样，链路第三环通了。

### 2.3 编译一个最小 `.cu` 跑通 —— 焊死链路

先别急着写完整 kernel，用一个"能编译、能跑、能打印"的最小程序，把第三、四环（编译、运行）验证一遍：

```cpp
// 文件: hello.cu   编译: nvcc hello.cu -o hello   运行: ./hello
#include <cstdio>

__global__ void hello_kernel() {
    // __global__ = 这函数在 GPU 上跑, 由 CPU 发起调用 (见第 3 节详解)
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    hello_kernel<<<2, 4>>>();     // 启动配置: 2 个 block, 每 block 4 个 thread, 共 8 个线程
    cudaDeviceSynchronize();      // ★ 必须! CPU 等 GPU 打印完再退出, 否则啥都看不到
    return 0;
}
```

> **`cudaDeviceSynchronize()` 这行是链路调试的头号高频坑**：kernel 启动 `<<<...>>>` 是**异步**的——CPU 发完命令立刻往下跑，不等 GPU。如果 main 直接 return，进程结束了 GPU 可能还没打印。加上同步，CPU 才会"等 GPU 干完活"。（这和你 Day4 测 tok/s 时的 `torch.cuda.synchronize()` 是同一个道理——CUDA 世界里 CPU 和 GPU 是两条异步的时间线。）

能看到 8 行 Hello 打印出来，**五环链路全部焊通**。剩下的就是把 kernel 换成真正的 vector-add。

### 2.4 插一个必须懂的概念：三个函数限定符

刚才那个 `__global__` 是什么？CUDA 给函数加了三种"身份标签"，决定这函数**在哪跑、谁能调**：

| 限定符 | 在哪跑 | 谁来调 | 用途 |
|--------|--------|--------|------|
| `__global__` | GPU | **CPU** 发起（`<<<>>>`） | kernel 入口，CPU 和 GPU 的桥 |
| `__device__` | GPU | GPU（被 kernel 调） | kernel 内部的辅助函数 |
| `__host__` | CPU | CPU | 普通 C++ 函数，默认就是它 |

> 记住这个区别的关键：`__global__` 是**唯一能被 CPU"隔空发起"到 GPU 上的**，它是两个世界的入口门。`__device__` 是 GPU 内部自己人调用的小工具。为什么要分？因为 CPU 和 GPU 是两套完全不同的指令集和内存，编译器必须知道每段代码给谁编。

---

## 3. 主线核心：`cuda_playground/vector_add.cu` 逐段拆解

一个完整的 CUDA 程序有清晰的**五步套路**，几乎所有 kernel 程序都是这个骨架。先看 kernel 本体，再看外面的"数据搬运"。

### 3.1 kernel 本体：全局索引 + 边界检查

```cpp
// vector_add.cu 的核心 kernel
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // 灵魂公式: 每个线程用自己的编号认领唯一一个元素 (见 1.2)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ★ 边界检查: 为什么必不可少? 见下方深挖
    if (idx < n) {
        c[idx] = a[idx] + b[idx];   // 每个线程只干这一件极简单的事
    }
}
```

> **为什么必须有 `if (idx < n)`？—— 这是 CUDA 新手第一个真正的坑。**
>
> block 的大小通常固定（如 256），但你的数据长度 `n` 不一定是 256 的整数倍。假设 `n = 1000`，每 block 256 个线程，你需要 `ceil(1000/256) = 4` 个 block，一共启动了 `4 × 256 = 1024` 个线程。**多出来的 1000~1023 号线程算什么？** 如果不拦住，它们会去读写 `a[1000]`、`c[1023]` 这些**根本不存在的内存**——轻则结果错，重则程序崩（非法内存访问）或悄悄污染别的数据。
>
> `if (idx < n)` 就是让这些"多余的工人"直接躺平不干活。**宁可多派人、让富余的人闲着，也不能让人越界乱摸**——这是 GPU 编程里"整块整块派工"必然产生的边角料处理。**记住这个"多派+拦截"的模式，它是后面 Triton 的 mask 要解决的同一个问题**（第 5 节对照）。

### 3.2 host 端：数据搬运的五步套路

kernel 只管算，但数据从哪来？**CPU 内存（host memory）和 GPU 显存（device memory）是两块物理上分开的内存，GPU 的 kernel 只能碰显存里的数据。** 所以 CPU 必须先把数据"搬"到显存，算完再"搬"回来。这就是下面五步：

```cpp
// vector_add.cu 的 main —— 依赖: CUDA 12.4。编译: nvcc vector_add.cu -o vadd
#include <cstdio>
#include <cstdlib>

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(float);

    // 步骤① 在 CPU(host) 上准备数据。h_ 前缀 = host, 是工业界约定俗成的命名
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = 2 * i; }

    // 步骤② 在 GPU(device) 上开显存。d_ 前缀 = device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);         // 注意传 &d_a: 让 CUDA 把显存地址写回这个指针
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 步骤③ 把输入从 CPU 搬到 GPU (HostToDevice)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 步骤④ 配置启动参数并发射 kernel
    int blockSize = 256;                            // 每 block 256 线程, 常用值(32的倍数)
    int gridSize  = (n + blockSize - 1) / blockSize;// ★ 向上取整, 见下方深挖
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // 步骤⑤ 把结果从 GPU 搬回 CPU (DeviceToHost) 才能读
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy 自带同步语义, 会等 kernel 跑完再拷, 所以这里不必额外 synchronize

    printf("c[999] = %.1f (期望 2997)\n", h_c[999]);  // 999 + 2*999 = 2997

    // 收尾: 显存和内存都要各自释放, 否则泄漏
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

> **深挖那行向上取整 `(n + blockSize - 1) / blockSize`**：
> 整数除法是**向下取整**的，`1000 / 256 = 3`——只开 3 个 block（768 线程）就少算了 232 个元素！加上 `blockSize - 1` 这个"凑整技巧"，`(1000+255)/256 = 4`，正好够。**这是"必须覆盖所有数据"和"block 大小固定"之间的必然妥协**：宁可多开一个不满的 block（配合 3.1 的 `if` 拦截边角料），也不能少开。这个 `(n + b - 1) / b` 的向上取整写法，你会在 CUDA 代码里见到一万次，直接背下来。

### 3.3 工业级必备：错误检查宏（别让 GPU 悄悄失败）

上面的代码有个隐患——**CUDA 的错误默认是"沉默"的**。`cudaMalloc` 失败、kernel 越界，函数照样返回，你却毫不知情，最后拿到一堆莫名其妙的错值还找不到原因。工业代码里**每个 CUDA 调用都要查返回码**，用一个宏统一处理：

```cpp
#include <cstdio>
// 工业界标配: 包住每个 cuda 调用, 出错立刻打印文件+行号并退出
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA err %s:%d: %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(1);                                                \
    }                                                           \
} while (0)

// 用法: 把调用包起来
CUDA_CHECK(cudaMalloc(&d_a, bytes));
CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

// kernel 启动不返回错误码, 要专门查两类错误:
vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
CUDA_CHECK(cudaGetLastError());       // 查"启动配置"错误 (如 block 太大)
CUDA_CHECK(cudaDeviceSynchronize());  // 查"执行期"错误 (如越界访问)
```

> **为什么 kernel 要查两次？** kernel 是异步的：`cudaGetLastError()` 抓的是"发射那一刻"的错（比如 block 配了 2048 个线程超上限）；`cudaDeviceSynchronize()` 等它真跑完，才能抓到"跑的过程中"的错（比如某线程越界）。这俩抓的是**不同时间点**的错误，缺一不可。这也是你以后在 H100 上调 kernel 的第一救命工具。

### 3.4 验证：和 PyTorch `allclose` 对齐（呼应 Day4 的正确性观）

写完 kernel，怎么证明它对？**拿一个可信基线做数值对照**——和 Day4 学的正确性方法论一脉相承。PyTorch 的向量加就是那把"标准尺"：

```python
# 环境: PyTorch, 假设已把 CUDA 算出的 c 存成 out.npy (或用 ctypes/pybind 直接对接)
import torch, numpy as np
n = 1000
a = torch.arange(n, dtype=torch.float32)
b = 2 * torch.arange(n, dtype=torch.float32)
ref = a + b                                    # 可信基线
cuda_out = torch.from_numpy(np.load("out.npy"))
# allclose: 允许浮点微小误差的"约等于"。rtol/atol 是相对/绝对容差
assert torch.allclose(cuda_out, ref, rtol=1e-5, atol=1e-6), "❌ 和 PyTorch 不一致!"
print("✅ CUDA vector_add 与 PyTorch 数值对齐")
```

> **为什么用 `allclose` 不用 `==`？** 浮点运算有舍入误差，GPU 和 CPU 的计算顺序/精度可能有极微小差异，直接 `==` 几乎必然失败。`allclose` 允许一个极小容差，是所有数值 kernel 验证的标准做法。这正是 W8"三尺子"里的**第一把尺：数值对齐**——今天你已经在用了。

---

## 4. 今天的灵魂：`cuda_vs_triton_first_look.md` 对照笔记

这才是今天真正的产出。把 W8 Day1 的 Triton vector_add 和今天的 CUDA 版**并排贴**，标出三个差异。**每个差异留一个问题当钩子**——这些问题就是 W1 Day1 的学习起点。

先并排看两段（Triton 版凭记忆还原，以你 W8 Day1 的实际代码为准）：

```python
# ===== Triton 版 vector_add (W8 Day1) =====
import triton, triton.language as tl
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                       # 我是第几个 program (不是第几个线程!)
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 我这个 program 负责的一整块下标
    mask = offs < n                              # 边界: 用 mask 数组, 不是 if
    a = tl.load(a_ptr + offs, mask=mask)         # 一次性 load 一整块
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)     # 一次性 store 一整块
# 启动: add_kernel[(triton.cdiv(n, BLOCK),)](a, b, c, n, BLOCK=256)
# 内存: a,b,c 直接是 PyTorch 的 GPU 张量, 没有任何 malloc/memcpy!
```

```cpp
// ===== CUDA 版 vector_add (今天) =====
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 我是第几个线程
    if (idx < n)                                       // 边界: 用 if
        c[idx] = a[idx] + b[idx];                      // 我只处理 1 个元素
}
// 外面还要 cudaMalloc + cudaMemcpy 来回搬数据 (见 3.2)
```

### 差异一：线程粒度 —— program-per-block vs thread-per-element

> **这是最本质的差异。** CUDA 里**你站在单个线程的视角**写代码——"我这一个线程，处理 `idx` 这一个元素"。Triton 里**你站在一整块（program/block）的视角**写代码——"我这个 program，处理 `offs` 这一整块 256 个元素"，块内怎么拆给具体线程、怎么向量化，**Triton 编译器替你安排了**。

**类比**：CUDA 是你给每个工人单独下指令"你搬第 37 块砖"；Triton 是你给一个工头下指令"你带队把 37~292 号砖搬了"，具体怎么分派给队里的人，工头（编译器）自己定。

> ❓**留给 W1 的问题 1**：CUDA 里"32 个线程一组"叫 **warp（线程束）**，是硬件真正的调度单位。那么——Triton 的一个 program 底层到底对应多少个 warp？我在 CUDA 里手动管的 thread，和 Triton 帮我隐藏的东西，边界到底在哪？

### 差异二：内存管理 —— 手动 malloc/memcpy vs 全自动

> CUDA 要你**亲手**在显存 `cudaMalloc`、把数据 `cudaMemcpy` 搬来搬去、用完 `cudaFree`。Triton 直接吃 PyTorch 的 GPU 张量指针，**一行内存管理代码都不用写**——因为张量本来就已经在显存里，PyTorch 的分配器管着。

**为什么 Triton 能省掉？** 因为它生在 PyTorch 生态里，默认你的数据已是 GPU 张量。而 CUDA 是底层，它假设你从 CPU 数据起步，必须显式表达"跨越 CPU/GPU 这道内存鸿沟"。**省事的代价是黑盒，可控的代价是繁琐**——这是所有"高层框架 vs 底层"的永恒权衡。

> ❓**留给 W1 的问题 2**：`cudaMemcpy` 这次数据搬运在整个耗时里占多大？（Day4 已知 CPU/GPU 是异步的）如果数据一直留在 GPU 上、多个 kernel 接力算，是不是就能省掉来回搬的开销？这和推理引擎里"KV Cache 常驻显存"是不是同一个思路？

### 差异三：边界处理 —— if 分支 vs mask 掩码

> 同一个"数据长度不是块大小整数倍"的问题（3.1 讲的边角料），两者解法形态不同：CUDA 用 `if (idx < n)` 让越界线程**跳过**；Triton 用一个布尔 `mask` 数组，告诉 `load/store`**哪些位置真读写、哪些当作没有**。

**本质是同一件事**：都是"多派了工人，拦住越界的那部分"。区别只是——CUDA 是**标量视角的分支**（每个线程各自判断），Triton 是**向量视角的掩码**（一整块一起带个 mask）。

> ❓**留给 W1 的问题 3**：`if (idx < n)` 里，同一个 warp 的 32 个线程如果有的进 if、有的不进，会发生什么？（这引出 **warp divergence，线程束分化**——一个影响 CUDA 性能的关键概念）mask 在硬件层面是不是也躲不开这个问题？

> **把这三个问题抄进 `cuda_vs_triton_first_look.md` 结尾**。带着具体问题进 W1，你的学习是"找答案"（主动、高效）；没有问题进 W1，是"从头灌"（被动、慢一倍）。**这就是今天埋钩子的全部意义。**

---

## 5. 富余任务：改成逐元素乘加 `y = a*x + b`，体会两种"改"的本质区别

如果链路顺利有富余，做这个小改动，它能让你**分清两个从此天天打交道的概念**：

```cpp
// 只改 kernel 内的这一行计算逻辑:
__global__ void saxpy(const float* x, const float* a, const float* b, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] = a[idx] * x[idx] + b[idx];   // 从 加法 变成 乘加 —— 改的是"算什么"
}
// 启动配置 <<<gridSize, blockSize>>> 一个字都不用动 —— "派多少工人、怎么编队" 没变
```

> **这里藏着一个关键区分**：
> - **改 kernel 逻辑**（`a+b` → `a*x+b`）= 改"每个工人**干什么活**"。这关乎**算法/正确性**。
> - **改启动配置**（`<<<grid, block>>>`）= 改"派**多少工人、怎么编队**"。这关乎**性能/并行度**，和算什么无关。
>
> 这俩是**正交的两件事**。以后你做 kernel 优化，绝大多数时间在调后者（block 大小、grid 划分、每线程处理几个元素）而**完全不碰前者**——因为你要在"算出同样正确的结果"前提下压榨速度。今天亲手感受一次"逻辑变了但配置没变"，这个直觉以后价值千金。
>
> 顺带一提：`y = a*x + b` 这种"乘加"在 GPU 上有专门的硬件指令 **FMA（Fused Multiply-Add，融合乘加）**，一条指令同时做乘和加、还只舍入一次，又快又准。你写 `a*x+b`，编译器通常自动帮你合成一条 FMA——这是 GPU 数值计算的基本功，记下这个名字，W1 深挖。

---

## 6. 【副线·研】AMK 保温：列出 3 个 W1-W2 要深挖的问题

今天对 AMK **不深挖，只保温**——重读你的 `Llama-3.1-8B-Instruct.h100.report.md` 和 ncu 摘要，产出一张**问题清单**，明天联络师兄时直接用。目标是让 AMK 这条线在你脑子里"不凉"，且下次一上手就有明确抓手。

建议照这三个方向各锁一个问题（示例，按你 report 实际数据填）：

1. **跨 SM 同步的证据**：AMK 的"巨核"要跨多个 SM 协作，理论上有同步开销。**在我的 ncu 数据里，这个开销体现在哪几个指标上？**（候选：`sm__throughput`、L2 命中率、achieved occupancy、某个 barrier/stall 相关计数器）——先定位"看哪个数"。
2. **热点 region 归因**：**整份 report 里哪个 region（哪段计算）耗时占比最大？** 它是访存瓶颈还是计算瓶颈（回忆 Day4 的 memory-bound vs compute-bound）？这决定 W1-W2 优先啃哪块。
3. **和论文对标**：AMK 论文声称达到 **0.60–0.72× cuBLAS** 的性能。**我 H100 上实测的比值对得上吗？** 对不上（更低或更高）分别说明什么——是硬件差异、配置差异，还是我 profiling 的姿势有问题？

> **为什么此刻只列问题不动手？** 因为 AMK 是你的正式科研线，W0 的主线是"引擎闭环 + CUDA 入门"。现在深挖会分散精力，但完全不碰又会"凉掉、下次重新热身"。**列一张精准的问题清单，是成本最低的"保温"**——既锁住了思考的成果，又给下周的自己和师兄的对话留了明确接口。这就是科研节奏管理。

---

## 7. 【整理块】`portfolio_overview.md` 初稿

把 Day1–4 攒的各 README 串成**一页总览**。目的不是罗列，是让人（HR/导师/三个月后的你）**30 秒看懂你这条学习线的完整叙事**。建议结构：

```markdown
# Portfolio 总览 —— 从"会调库"到"懂底层"的一条线
| 项目 | 证明了什么 | 关键词 | 跑法 |
|------|-----------|--------|------|
| micrograd     | 懂自动微分本质      | 反向传播/计算图     | python ... |
| numpy 网络     | 能脱离框架造轮子     | 纯手写前向反向      | python ... |
| MNIST CNN     | 掌握完整训练闭环     | 数据→训练→评估     | python ... |
| Llama化 nanoGPT| 会改架构+验证正确性  | RMSNorm/RoPE/三级验证| python ... |
| CUDA vector_add| CUDA 链路+底层入门  | kernel/内存/warp   | nvcc ...   |
> 一句话主线: 每个项目都能说清"我证明了什么", 串起来是一条从上层到底层、
> 从会用到会造的清晰成长轨迹。
```

> 和 Day4 整理块同一个精神：**每件事都要能说清它证明了什么。** portfolio 不是作品堆砌，是一条**有方向的叙事线**——今天你正好新增了"往底层扎"的 CUDA 一环。

---

## 8. 今日收尾 · 里程碑自测

**产出清单**：
- [x] `vector_add.cu` 在 H100 上编译、运行、和 PyTorch `allclose` 对齐
- [x] `cuda_vs_triton_first_look.md`：三差异 + 三个带进 W1 的问题
- [x] AMK 三问清单（明天联络师兄直接用）
- [x] `portfolio_overview.md` 初稿

**完成标准（今天的硬指标）**：**链路五环——登录 / module load / nvcc / 运行 / 验证——任何一环都不需要现查资料。** 合上笔记，能从头默背一遍这条路，才算过关。

**能过关的自测四问**：
1. `blockIdx.x * blockDim.x + threadIdx.x` 每一项是什么？为什么这样就能让几千个线程各认领一个元素？
2. 为什么必须有 `if (idx < n)`？`gridSize` 为什么要向上取整？这两件事是同一个问题的两面吗？（是——都在处理"整块派工"的边角料）
3. Triton 的 program 视角和 CUDA 的 thread 视角，本质差在哪？内存管理为什么一个全自动一个全手动？
4. "改 kernel 逻辑"和"改启动配置"分别对应什么？为什么说它们正交？

> **一句话总结今天**：你没在"学 CUDA"，你在**修一条以后天天要走的路，并在路口立好三块写着问题的路牌**。W1 你不是从零出发，是顺着这三块路牌去找答案——这就是"带着问题进正课"的复利。

