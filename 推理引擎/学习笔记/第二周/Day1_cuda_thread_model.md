# 阶段一 · W1 Day 1 —— CUDA 线程模型 + 第一个真 kernel(对照 Triton)

> **一句话主题**:W0 Day5 我们把"能编译能跑 CUDA"这条路焊通了;今天正式开上这条路,把**线程模型**这一个概念挖到地基——五级层级到底是哪五级、灵魂公式为什么是那个样子、`if (idx < n)` 在硬件里到底发生了什么。
>
> **今天的隐藏主线:收口 W0 Day5 埋下的三个钩子。** 那天你并排贴了 Triton 和 CUDA 两个 vector-add,留了三个问题带进 W1。今天这篇笔记会把它们**逐个答掉**。读到最后你会发现:这三个问题的答案,其实就是"线程模型"这一个东西的三个侧面。

---

## 0. 先给四个学习目标问题一个"电梯答案"

在展开之前,把今天必须能脱口而出的四个答案放这里。读完全文后回来,你应该能合上书自己复述。

1. **CUDA 的五级层级是哪五级?为什么这么分?**
   软件抽象三层:**Grid(网格)→ Block(线程块)→ Thread(线程)**;硬件实体两层:**SM(流式多处理器)→ Warp(线程束,固定 32 线程)**。关键映射是 **Block 整体住进一个 SM、Block 内部被切成若干 Warp**。分层的原因只有一个词——**可扩展性(scalability)**:Block 之间互不依赖,所以同一份程序不用改,就能跑在 8 个 SM 的小卡上、也能跑在 132 个 SM 的 H100 上(见 §2.4)。

2. **`blockIdx.x * blockDim.x + threadIdx.x` 为什么是这个公式?**
   它是把"二维编号(第几块、块内第几个)"压成"一维全局编号"的映射。本质是**按块进位**:`blockDim.x` 是进位基数(每块多少人),`blockIdx.x` 是高位(第几块),`threadIdx.x` 是个位(块内第几)。全校排学号 = `班级号 × 每班人数 + 班内座号`,一模一样的结构(见 §3)。

3. **为什么必须 `if (idx < n)`?它在硬件里贵不贵?**
   因为 grid 是按 block **整块**派工的,`n` 几乎不可能被 `blockDim` 整除,最后一块必然多派几个线程,`if` 就是拦住这些"多出来的工人"别去摸不存在的显存。至于贵不贵——**几乎免费**,因为整一个 warp(32 线程)通常"同进同退",真正发生分化的最多只有最后那个不满的 warp;而且这种短 `if` 编译器多半用**谓词执行(predication)**实现,连分支都不算(见 §4,这里回答钩子③)。

4. **CUDA 一线程一元素、Triton 一 program 一块,粒度差异的本质是什么?**
   不是两个对立的东西,是**同一套机制的两个抽象层**:Triton 的一个 program ≈ CUDA 的一个 block,你写的"块级向量加"最终仍被 Triton 编译器翻译成 CUDA 那样的逐线程索引拆分(拆成 `num_warps × 32` 个线程)——只不过这套拆分它替你做、做得更系统更不易错。**你在 CUDA 里手写的那套索引,就是 Triton 藏起来不让你看的东西**(见 §5,这里回答钩子①)。

下面把每个答案背后的"为什么"讲透。

---

## 1. 问题背景:为什么 GPU 编程需要"层级"这套组织

### 1.1 先回收一个旧直觉(一句话带过)

W0 Day5 已经立过这个类比,这里只钉一遍:**CPU 是几个博士生**(核少但每个都强,擅长复杂串行逻辑),**GPU 是几千个小学生**(每个只会简单算数,但人多,适合"同一道题换不同数字、几万道一起做")。vector-add `c[i]=a[i]+b[i]` 每个位置互不依赖,是 GPU 的主场。

### 1.2 今天真正的新问题:几千个人,怎么"组织"?

W0 我们直接用了三级,但没追问**为什么需要组织**。想一个生活场景:你要指挥 5000 个工人搬砖,有两种管法——

- **管法一:一盘散沙。** 5000 人全堆在工地门口,你拿大喇叭喊"谁去搬第 1 块?谁去搬第 2 块?"——光点名就累死了,而且没人能互相搭手。
- **管法二:编成层级。** 把 5000 人分成若干"班"(block),每班 256 人;每个班整体派到一个"工位"(SM)上;班内再按 32 人一个"小队"(warp)齐步走。你只需要发"派多少个班"这一条命令。

CUDA 选的就是管法二。**层级的存在,是为了让"派几千份活"这件事变成"派几个班"这一条命令,同时让同班的人能协作(共享内存、能同步)。** 这就是全部动机——后面五个名词,都是这棵"为了好组织"的树上结的果子。

---

## 2. 核心原理一:五级层级,从软件到硬件

这是今天的地基,请慢读。很多人学了很久 CUDA,脑子里 Grid/Block/Thread 和 SM/Warp 还是两盆搅在一起的浆糊——因为没人告诉他们**这五个词分属两个世界**:前三个是**你写的软件抽象**,后两个是**GPU 的物理实体**。理解线程模型,就是理解这两个世界怎么对上。

### 2.1 软件三层:Grid → Block → Thread(你写的代码管这层)

> **Thread(线程)**:最小的干活单位。在 vector-add 里,一个 thread 负责算一个 `c[i]`。它是那个"小学生/工人"。你写 kernel 时,**写的就是单个 thread 的视角**——"作为第 i 个线程,我该干嘛"。

> **Block(线程块)**:把一组 thread(比如 256 个)打包成一个 block。**同一个 block 里的线程能协作**:共享一块高速的 shared memory(共享内存)、能用 `__syncthreads()` 互相同步。类比"一个班":同班同学坐一个教室、能互相借草稿纸(共享内存)、能听班长喊"都停一下"(同步)。**block 是被整体派到 GPU 一个物理核心上的最小单位。**

> **Grid(网格)**:一次 kernel 启动所派出的**所有** block 合起来是一个 grid。类比"整个年级"。你在 CPU 端写 `<<<gridSize, blockSize>>>`,定义的就是 grid 的形状——发几个班、每班几人。

### 2.2 硬件两层:SM → Warp(GPU 的物理实体,你管不到但必须懂)

> **SM(Streaming Multiprocessor,流式多处理器)**:GPU 里真正干活的"车间/工位"。一块 H100 有 **132 个 SM**。每个 SM 内部有 128 个 FP32 计算单元(可以理解成 128 个"算术工位")、一块共享内存、一堆寄存器、4 个调度器。你 launch 的所有 block,会被 GPU 硬件调度器**分批塞进这 132 个车间**滚动执行,跑完一个换下一个。

> **Warp(线程束)**:GPU 硬件**真正一次性调度的最小单位,固定 32 个线程**。一个 block 住进 SM 后,会被切成 `blockDim / 32` 个 warp。调度器每次不是挑一个线程来跑,而是**挑一整个 warp(32 线程)齐步走**。
>
> **类比**:warp 是一个"必须齐步走的 32 人小队"。教官(调度器)喊口令是以小队为单位的,绝不会单独喊小队里的某一个人。小队里 32 个人永远做同一个动作。

> **SIMT(Single Instruction, Multiple Threads,单指令多线程)**:这就是 warp 的执行方式——**一条指令,同时驱动 32 个线程,每个线程操作自己那份数据**。类比广播体操:喇叭里喊"第八节伸展运动"(单指令),操场上几千人同时做同一动作(多线程),但每人动的是自己的胳膊(各自的数据)。CUDA 的"你写一份代码、几千线程各跑一遍"之所以成立,底层就是 SIMT。

### 2.3 两个世界的映射(今天最重要的一张对应表)

| 软件抽象(你写) | 硬件实体(GPU) | 映射关系 |
|---|---|---|
| Grid | 整块 GPU | 一个 grid 铺开,占满能用的所有 SM |
| **Block** | **SM** | **一个 block 整体住进一个 SM,绝不跨 SM;一个 SM 可同时住多个 block** |
| Thread | SM 里的计算单元 | 一个 thread 最终落到 SM 内一个算术工位上执行 |
| (代码里没有) | **Warp** | block 住进 SM 后被切成 `blockDim/32` 个 warp,**warp 才是硬件真正一次调度的单位** |

三条必须刻进骨头的映射事实:

1. **Block 不跨 SM。** 一个 block 的所有线程都在同一个 SM 上出生、执行、结束。这就是为什么"同 block 内能共享内存、能同步"——因为它们物理上就挤在同一个车间里,摸得到同一块共享内存。反过来,**不同 block 之间默认完全无法直接通信/同步**(它们可能在不同 SM、甚至不同时间才被调度)。
2. **Warp 是调度的"量子"。** 你以为 GPU 一个一个挑线程跑,错——它一挑就是 32 个(一个 warp)。这个"32 人小队齐步走"的事实,是后面 warp 分化、合并访存、occupancy 一整串性能概念的根。
3. **一个 SM 能同时住好几个 block。** 具体能住几个,取决于每个 block 吃多少资源(寄存器、共享内存、线程数)——这就是 §4.4 occupancy 的话题。

### 2.4 为什么要这样设计?一个词:可扩展性

这是"为什么被设计成这样"的答案,也是整个 CUDA 编程模型最精妙的一笔。

**关键设计:Block 之间相互独立、执行顺序完全不定。** GPU 从不保证"block 0 先于 block 1 跑",也不保证它们同时在跑——硬件调度器看哪个 SM 有空就把下一个 block 塞过去。

这个"不保证顺序"看似是限制,实则是**故意的天才设计**,它换来的是:

```
同一份 vector_add 程序:
  在 8 个 SM 的小卡上  → 硬件把 4 个 block 分两批塞 8 个 SM,跑得慢点
  在 132 个 SM 的 H100 → 硬件把 4 个 block 同时撒进去,跑得快
  → 你的代码一个字都不用改!
```

> **类比**:你印了 977 份独立试卷(block),考场(SM)有多少座位就同时开考多少份,座位多就多开几份、座位少就分批。**试卷之间没有依赖、不用按顺序交卷**,所以这套考试系统不管考场大小都能正常运转,考场越大只是越快而已。如果试卷之间有依赖(比如"第 5 份要用第 3 份的答案"),那考场再大也得排队,扩展性就死了。

**这条设计哲学直接影响你以后写 kernel 的红线:永远不要假设 block 的执行顺序,永远不要指望 block 之间直接同步。** 需要全局协作?要么拆成多次 kernel 启动(kernel 之间天然有顺序),要么用专门的全局同步机制——AMK 的"巨核"之所以难、之所以有同步开销,根子就在于它想打破"block 独立"这条铁律去跨 SM 协作(这是你研线的核心矛盾,先记住它)。

---

## 3. 核心原理二:灵魂公式,亲手算一遍

### 3.1 三个内置变量:每个线程的"身份证"

既然几千个线程跑的是**同一份** kernel 代码,每个线程怎么知道"我该认领哪个元素"?靠 CUDA 给的三个只读内置变量(在 kernel 里直接可用):

| 变量 | 含义 | 类比(全校排学号) |
|---|---|---|
| `threadIdx.x` | 我在**自己 block 内**排第几 | 班内座号(0 ~ blockDim-1) |
| `blockIdx.x` | 我所在 block 在 grid 里排第几 | 第几班 |
| `blockDim.x` | 每个 block 有多少 thread | 每班人数(固定) |
| `gridDim.x` | grid 里一共多少 block | 全校几个班 |

(`.x` 是因为这些其实是三维的 `x/y/z`,处理图像、矩阵时用得到;一维数组只用 `.x`,见 §3.4。)

### 3.2 公式推导:按块进位

把"第几班 × 每班人数 + 班内座号"翻译成代码,就是:

```
全局索引 idx = blockIdx.x * blockDim.x + threadIdx.x
              └─ 第几班 ─┘ └ 每班人数 ┘ └ 班内座号 ┘
```

**为什么乘的是 `blockDim.x` 而不是别的?** 因为它是"进位基数"。就像十进制里"百位 × 100 + 十位 × 10 + 个位",这里的"进制"是"每块多少人"。`blockIdx` 是高位,要乘以"每块多少人"才能折算成全局的第几个人;`threadIdx` 是个位,直接加。

### 3.3 亲手算:用纸和笔推一遍(今天必须亲手做)

计划里要求"亲手算",这不是走过场——算过一遍,这个公式才会长在手上。拿具体数字:

**设定:`n = 1000`,`blockDim.x = 256`,所以 `gridDim.x = ceil(1000/256) = 4`(共 4 个 block、1024 个线程)。**

| 这个线程(blockIdx, threadIdx) | 计算 | 全局 idx | 它负责 |
|---|---|---|---|
| block 0, thread 0 | `0*256 + 0` | **0** | `c[0] = a[0]+b[0]` |
| block 0, thread 255 | `0*256 + 255` | **255** | `c[255]` |
| block 1, thread 0 | `1*256 + 0` | **256** | `c[256]` |
| block 2, thread 100 | `2*256 + 100` | **612** | `c[612]` |
| block 3, thread 255 | `3*256 + 255` | **1023** | ❌ 越界!`n` 只到 999,被 `if` 拦下 |

看最后一行:`block 3` 覆盖 idx 768~1023,但合法下标只到 999,**768~999 这 232 个线程是真干活的,1000~1023 这 24 个是多派来凑满一个 block 的**——它们就是下一节要拦的"边角料"。

> 想"看见"这个分工,`cuda/01_vector_add.cu` 里有个 `vector_add_verbose` 版本,会让前 8 个全局线程各自打印自己的 `(blockIdx, threadIdx) → idx`。跑一遍,你会亲眼看到"第 1 班第 0 座接管了第 256 号元素"这种事。

### 3.4 为什么是三维?(.x / .y / .z)

今天只用一个 `.x`,但你得知道 CUDA 为什么给三维。因为**真实数据常常是二维/三维的**:一张图像是 `(宽, 高)`、一个 batch 的矩阵是 `(行, 列)`。如果数据是二维的,用 `threadIdx.x` 管列、`threadIdx.y` 管行,索引天然对得上,不用手工把二维下标折算成一维。

```cpp
// 处理一张 H×W 图像时,二维索引长这样(知道有这回事即可,W1 后面会练):
int col = blockIdx.x * blockDim.x + threadIdx.x;  // 哪一列
int row = blockIdx.y * blockDim.y + threadIdx.y;  // 哪一行
if (row < H && col < W) img[row * W + col] = ...; // row*W+col 把二维压回一维显存地址
```

> **本质没变**:不管几维,最后都要折算成一个一维显存地址(显存是平的)。多维索引只是让"写代码的人"少做一次换算。今天先把一维吃透。

---

## 4. 核心原理三:`if (idx < n)` + warp 分化(收口钩子③)

### 4.1 越界是"必然",不是"可能"

§3.3 已经算死了:`n=1000`、`block=256` 必然启动 1024 个线程、必然多出 24 个越界线程。这不是偶发的边界情况,是**每次派工都会产生的结构性边角料**。`if (idx < n)` 就是发给这 24 个多派线程的命令:"你们躺平,别动。"

为什么宁可多派、再拦?因为 block 是**整块**派出去的(SM 一次接一整块),没法"只派 232 个线程的零头"。所以策略只能是:**向上取整派够整块,再用 `if` 把零头拦下。** 这和 gridSize 向上取整 `(n+block-1)/block` 是**同一个问题的两面**——一个负责"派够",一个负责"拦多余的"。

### 4.2 深挖:这个 `if` 到底花多少钱?——引出 warp 分化

钩子③问的是:同一个 warp 的 32 个线程,如果有的进 `if`、有的不进,会发生什么?这正是 CUDA 性能里的关键概念。

> **warp divergence(线程束分化)**:warp 是齐步走的(§2.2 SIMT),32 个线程本该同做一条指令。但如果一个 `if/else` 让 warp 里一部分线程走 if 分支、另一部分走 else 分支,齐步走就破裂了——硬件只能**把两条路先后各走一遍**:走 if 路径时,走 else 的那些线程被"按停"(屏蔽);走 else 路径时,走 if 的那些被按停。**两条路串行执行,这一段的耗时 ≈ 两者之和。**
>
> **类比**:32 人小队齐步走到一个岔路口,16 人要往左、16 人要往右。但小队规定不许散开,于是全队先一起走左路(想去右路的 16 人原地踏步当摆设),回来再一起走右路(想去左路的 16 人当摆设)。**一趟岔路,走了两遍。** 这就是分化的代价。

听着吓人,但回到我们的 `if (idx < n)`,算一笔账你就安心了:

- 我们的 warp 里,`idx` 是**连续**的(一个 warp 恰好是 32 个连续 idx)。
- 对于**所有完整的 warp**:要么 32 个 idx 全 `< n`(全进 if,无分化),要么全 `≥ n`(全不进,无分化)。**整队同进同退,不分化。**
- 唯一可能分化的,是 **idx 跨越 n 边界的那一个 warp**(本例里覆盖 992~1023 的那个,其中 992~999 进、1000~1023 不进)。
- **一千多个线程里,最多只有这 1 个 warp 发生分化。** 其余 warp 干干净净。

**结论:边界检查的分化代价可以忽略,因为它至多影响一个 warp。** 真正要警惕的是"同一个 warp 内、相邻线程因为数据不同而走不同路"的写法(比如 `if (data[i] % 2 == 0) 走A else 走B`——奇偶交错,每个 warp 都必然分化),那才是性能杀手。

### 4.3 更狠的一层:这种短 `if` 甚至不算分支——谓词执行

再往下挖一层(这也呼应你 W8 Day1 §3.3 见过的 PTX)。对于 `if (idx < n) { 就一行 }` 这种**极短**的条件,编译器常常**根本不生成跳转**,而是用谓词执行:

> **predication(谓词执行)**:不给代码分岔,而是让每条指令带一个"是否生效"的开关位(谓词寄存器)。指令照常发出,但只有开关为真的线程,其内存读写才真正生效;开关为假的线程"空转"、结果被丢弃。因为 warp 本来就要齐步走,与其跳转,不如"都执行、用开关决定谁算数"。

编译出来的中间指令(PTX)大致是:

```
setp.lt.s32   %p1, %r_idx, %r_n;     // %p1 = (idx < n),算出开关
@%p1 ld.global.f32  %f1, [%r_a];     // @%p1:仅当 idx<n 才真的读 a
@%p1 ld.global.f32  %f2, [%r_b];
add.f32       %f3, %f1, %f2;         // 加法无害,大家都算
@%p1 st.global.f32  [%r_c], %f3;     // 仅当 idx<n 才真的写 c
```

**所以对你这个一行 `if` 而言,真相是:它大概率被谓词化了,连"分化串行"都不会发生,代价约等于零。** 这就是钩子③的完整答案——"warp 分化会不会发生、有多贵",取决于分支长短和 warp 内线程是否同向;而我们的边界检查,恰好落在"几乎零开销"的那一格。

> **这也顺带回答了"Triton 的 mask 在硬件层面躲不躲得开分化"**:躲不开,也不需要躲。Triton 的 `mask` 编译出来正是这套 `@%p` 谓词化的 load/store(W8 Day1 §3.3 已经见过)。**CUDA 的 `if` 和 Triton 的 `mask`,在硬件上是同一个谓词机制,只是一个写成分支、一个写成掩码。** 两者殊途同归。

### 4.4 顺手立住 occupancy(占用率),给 W1 后面留接口

> **occupancy(占用率)**:一个 SM 上"实际同时活跃的 warp 数"占"该 SM 理论最多能容纳的 warp 数"的比例。H100 每个 SM 最多容纳 64 个 warp;如果你只跑起 16 个,occupancy = 16/64 = 25%。**它衡量你把车间塞得多满。**

为什么它重要:GPU 靠"人多"来**掩盖访存延迟**——一个 warp 去读显存要等几百个周期,SM 立刻切到另一个 ready 的 warp 去算,不让工位闲着。活跃 warp 越多越有得可切。而**一个 SM 能住几个 block,取决于每个 block 吃多少寄存器/共享内存/线程数**——block 越大吃越多、SM 住得越少、occupancy 可能越低。

> **今天只立概念、不调参**:vector-add 是 **memory-bound(访存受限)** 的,瓶颈在搬数据的带宽,不在算得快不快。对这种算子,occupancy 够藏住访存延迟就够了,再高没用。真正拿 occupancy 说事,是后面 compute-bound 算子和 W7 Day2 ncu 报告的事。你 W8 Day1 §4.3 已经见过这套,这里是同一个机制在 CUDA 侧的对应。

---

## 5. 灵魂对照:CUDA thread-per-element vs Triton program-per-block(收口钩子①)

这是今天最有"啊哈"价值的一节。钩子①问:Triton 的一个 program 底层对应多少 warp?CUDA 手动管的 thread,和 Triton 隐藏的东西,边界到底在哪?

### 5.1 并排看两段(一个加法,两种视角)

```cpp
// ===== CUDA 版:你站在【单个线程】视角 =====
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 我是第几个【线程】
    if (idx < n)                                       // 我自己判断该不该干活(标量 if)
        c[idx] = a[idx] + b[idx];                      // 我只处理【1 个】元素
}
// 启动: vector_add<<<4, 256>>>(...)  → 1024 个线程,一个线程一个元素
```

```python
# ===== Triton 版(W8 Day1):你站在【单个 program】视角 =====
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                    # 我是第几个【program】
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # 我这个 program 负责的一整块下标(向量)
    mask = offs < n                           # 一整块一起带个掩码(向量 mask)
    a = tl.load(a_ptr + offs, mask=mask)      # 一次 load 一整块
    tl.store(c_ptr + offs, a + b, mask=mask)  # 一次 store 一整块
# 启动: add_kernel[(cdiv(n, BLOCK),)](...)  → 几个 program,一个 program 一整块
```

### 5.2 一张表钉死粒度差异

| | CUDA | Triton |
|---|---|---|
| 你的视角 | 一个 **thread(工人)** | 一个 **program(班组长)** |
| 一个单位处理多少数据 | **1 个**元素 | **一整块 BLOCK 个**元素 |
| "我是第几个"怎么问 | `threadIdx/blockIdx` 自己拼 | `tl.program_id(0)` 直接给 |
| 块内线程怎么分工 | **你手写**(那套索引) | **编译器负责**,你看不见 |
| 越界保护 | 标量 `if (idx<n)` | 向量 `mask`(底层同为谓词) |
| 代码风格 | 标量、逐线程 | 向量、逐块(像 NumPy) |
| 内存管理 | 手动 `cudaMalloc/Memcpy` | 直接吃 PyTorch GPU 张量 |
| 类比 | 指挥**每一个工人** | 指挥**每一个班组长** |

### 5.3 底层追问:一个 program 到底编译成几个 warp?(钩子①的正解)

"编译器负责"听着像魔法,拆开它你才真正信任 Triton。Triton 编译一个 program 时有个隐藏旋钮 **`num_warps`(用几个 warp 执行一个 program)**。设 `BLOCK=1024`、`num_warps=4`:

```
一个 program 要处理 1024 个元素
num_warps = 4  →  4 × 32 = 128 个线程 来执行这【一个】program
每个线程分到  1024 / 128 = 8 个元素

于是编译器把你写的 "c = a + b"(1024 长向量加)翻译成大致这样的逐线程代码:
    // 每个线程(共 128 个)干这段:
    for (int k = 0; k < 8; k++) {
        int idx = 我的起点 + k * 128;   // 编译器算好的步长
        c[idx] = a[idx] + b[idx];
    }
```

**看明白了吗——Triton 的"块级向量加",最终还是落回 CUDA 那样的"逐线程标量加"。** 它做的事本质上就是 §3 里你手写的那套索引拆分(块→线程、步长怎么排、要不要向量化访存),只是编译器做得更系统、更不易错。

**所以钩子①的答案是:**

> **一个 Triton program ≈ 一个 CUDA block,它被编译成 `num_warps` 个 warp(默认常是 4)。** CUDA 和 Triton 不是对立的两个东西,是**同一套线程模型的两个抽象层**:program/block 在上(两个框架里都归你管),thread 在下——**区别只在于"块内拆线程"这一步,CUDA 让你手写,Triton 替你生成。** 你在 CUDA 里逐行写的 `blockIdx*blockDim+threadIdx`,就是 Triton 藏在 `tl.arange` 背后不让你看的那部分。

### 5.4 边界到底在哪?一句话说清"谁管到哪一层"

```
你(无论 CUDA 还是 Triton)都管:  grid 发多少、每块多大、算什么东西
CUDA 额外让你手写:             块内 → 线程 的索引拆分、共享内存、warp 级细节
Triton 帮你接管:               块内 → 线程 的拆分(它自动生成上面那套循环)
硬件(谁都管不到):              block→SM 的调度、warp 的齐步走、谓词执行
```

学 CUDA 的价值正在于此:**当你以后看 vLLM / SGLang / FlashAttention 里那些 Triton kernel,或者读 AMK 的调度逻辑时,你脑子里有"块内到底发生了什么"的完整图景**——因为你亲手写过那一层。只会 Triton 的人,那一层是黑盒。

### 5.5 顺带收口钩子②:数据搬运 vs 常驻显存

钩子②问 `cudaMemcpy` 占多大、数据常驻 GPU 是不是 KV Cache 的思路。现在你有了线程模型,可以答得很准:

- `cudaMemcpy`(CPU↔GPU 搬数据)走的是 PCIe/NVLink 总线,带宽远低于 GPU 片内 HBM 带宽,**往往是整个流程的大头**(尤其数据大、算得少时)。这就是为什么 Triton 版"一行内存代码都不用写"那么香——它假设数据已经是 GPU 张量,根本不过总线。
- **推理引擎的整条设计主线就是"让数据尽量留在显存里"**:权重常驻、激活值常驻、**KV Cache 常驻**(你 W6 学的)——都是为了躲开"算一步就把中间结果搬回 CPU 再搬回来"的灾难。你 Day1-4 搭的引擎里,RMSNorm/RoPE/attention 一路在 GPU 上接力,从不过 CPU,正是这个思路。
- **一句话:`cudaMemcpy` 的存在提醒你 CPU 和 GPU 之间有道昂贵的鸿沟;好的 CUDA/推理程序,会想方设法让数据跨过这道沟的次数越少越好。**

---

## 6. 完整可运行代码:`cuda/01_vector_add.cu`(深化版)

下面是可直接编译运行的完整文件。它在 W0 Day5 的基础上做了三处**深化**——每一处都是工业界的真实写法,不是玩具:

1. **加了一个 `verbose` 教学 kernel**,让前几个线程打印自己的身份证,帮你"看见"分工(§3.3)。
2. **加了 grid-stride loop 版本**——这是推理引擎/官方库里**真实在用的**写法,和"一线程一元素"并列,让你理解两者的取舍(§6.1)。
3. **用 `cudaEvent` 计时并算有效带宽(GB/s)**,接上你 W7 roofline、W8 `gbps` 的那套带宽直觉。

> **依赖/环境**:NVIDIA GPU;CUDA 12.x(集群上先 `module load CUDA/12.4`);编译器 nvcc。
> **编译**:`nvcc -O2 -arch=sm_90 01_vector_add.cu -o vadd`(H100 是 sm_90;本地 40 系用 sm_89、30 系 sm_86,不写 `-arch` 让 nvcc 自选也行)。
> **运行**:`./vadd` 。**调试**:`compute-sanitizer ./vadd`(一键定位越界/非法访存,§8)。

```cpp
// ============================================================================
// 阶段一 W1 Day 1 — 01_vector_add.cu   CUDA 线程模型 · 第一个真 kernel(深化版)
// 环境: NVIDIA GPU + CUDA 12.x; 集群先 `module load CUDA/12.4`
// 编译: nvcc -O2 -arch=sm_90 01_vector_add.cu -o vadd    (H100=sm_90)
// 运行: ./vadd        调试: compute-sanitizer ./vadd
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---- 工业标配: CUDA 错误检查宏(W0 Day5 §3.3 已学,直接复用)----
// 为什么必须:CUDA 的错误默认"沉默",不查返回码,越界了你还以为跑对了。
#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(err));            \
        exit(1);                                                         \
    }                                                                    \
} while (0)

// ---- 版本 A: 一线程一元素(今天正课的灵魂公式)----
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // 灵魂公式: 把(第几块, 块内第几)压成全局第几个 —— 见 §3
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查: 拦住整块派工多出来的边角料线程 —— 见 §4
    // 为什么几乎免费: 短 if 被谓词化,且至多只有跨边界的 1 个 warp 分化
    if (idx < n)
        c[idx] = a[idx] + b[idx];            // 一个线程只干这一件简单事
}

// ---- 教学版: 让前 8 个线程打印身份证, "看见"分工(仅调试用)----
__global__ void vector_add_verbose(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 8)   // 只放前 8 个全局线程打印,否则 n 很大时会刷屏
        printf("  block=%2d thread=%3d -> 全局 idx=%d, 负责 c[%d]=a[%d]+b[%d]\n",
               blockIdx.x, threadIdx.x, idx, idx, idx, idx);
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

// ---- 版本 B: grid-stride loop(工业深化版,库里真实这么写)----
// 思想: 不追求"一线程一元素",而是开固定数量的线程,每个线程【跨步】处理多个元素。
// stride = 整个 grid 的线程总数,保证每轮相邻线程仍访问相邻地址(访存友好)。
__global__ void vector_add_gridstride(const float* a, const float* b, float* c, int n) {
    int stride = gridDim.x * blockDim.x;                 // 整个 grid 一共有多少线程
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;  // 起点还是那套公式
         i < n;
         i += stride)                                    // 每轮往前跨"整个 grid"那么远
        c[i] = a[i] + b[i];
}

// ---- CPU 参考实现(可信基线,用来验证 GPU 算得对不对)----
void cpu_reference(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

int main() {
    // ============ 第 0 步: 先用小 n 跑 verbose 版, "看见"线程分工 ============
    {
        int n_small = 1000;                       // 故意不是 256 的整数倍,好演示边界
        size_t bts = n_small * sizeof(float);
        float *ha = (float*)malloc(bts), *hb = (float*)malloc(bts), *hc = (float*)malloc(bts);
        for (int i = 0; i < n_small; i++) { ha[i] = (float)i; hb[i] = 2.0f * i; }
        float *da, *db, *dc;
        CUDA_CHECK(cudaMalloc(&da, bts)); CUDA_CHECK(cudaMalloc(&db, bts)); CUDA_CHECK(cudaMalloc(&dc, bts));
        CUDA_CHECK(cudaMemcpy(da, ha, bts, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, hb, bts, cudaMemcpyHostToDevice));

        int block = 256;
        int grid  = (n_small + block - 1) / block;   // 向上取整: 4 块, 共 1024 线程
        printf("=== 教学演示: n=%d, block=%d, grid=%d(共 %d 线程, 多派 %d 个)===\n",
               n_small, block, grid, grid*block, grid*block - n_small);
        vector_add_verbose<<<grid, block>>>(da, db, dc, n_small);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());       // 等 GPU 打印完再往下走
        printf("  ...(只打印前 8 个线程)...\n");

        cudaFree(da); cudaFree(db); cudaFree(dc);
        free(ha); free(hb); free(hc);
    }

    // ============ 下面是正式跑大 n、验证正确性、计时 ============
    int n = 1 << 24;                              // 约 1677 万元素,够看出带宽
    size_t bytes = n * sizeof(float);

    // 五步套路①: host 准备数据(h_ 前缀 = host,工业约定)
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);         // CPU 参考结果
    for (int i = 0; i < n; i++) { h_a[i] = 1.0f * i; h_b[i] = 2.0f * i; }

    // 五步套路②: device 开显存(d_ 前缀 = device)
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 五步套路③: 输入 CPU→GPU(H2D)
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 五步套路④: 配启动参数并发射(两个版本各发一次)
    int blockSize = 256;                                   // 常用值, 32(warp)的倍数
    int gridSize  = (n + blockSize - 1) / blockSize;       // 向上取整, 覆盖所有元素

    // 计时工具: cudaEvent(GPU 侧的秒表, 呼应 W7 profiler 思路)
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    int iters = 50;
    float ms;

    // ---- 跑版本 A: 一线程一元素 ----
    for (int i = 0; i < 5; i++)                            // warmup: 预热, 量到的才是计算时间
        vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
        vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));                  // 等 GPU 真跑完再读表
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float ms_A = ms / iters;

    // ---- 跑版本 B: grid-stride(开固定数量的块, 不再随 n 膨胀)----
    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    int gridB = numSMs * 8;                                // 经验值: 每个 SM 派 8 块, 把机器填满足矣
    for (int i = 0; i < 5; i++)
        vector_add_gridstride<<<gridB, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
        vector_add_gridstride<<<gridB, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    float ms_B = ms / iters;

    // 五步套路⑤: 结果 GPU→CPU(D2H), 才能读出来验证
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    cpu_reference(h_a, h_b, h_ref, n);

    // ---- 正确性: 和 CPU 基线比, 允许浮点微小误差(同 PyTorch allclose 精神)----
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double e = fabs((double)h_c[i] - (double)h_ref[i]);
        if (e > max_err) max_err = e;
    }
    printf("\n=== 正确性 ===\n");
    printf("c[%d] = %.1f (期望 %.1f)\n", n-1, h_c[n-1], 3.0*(n-1));
    printf("最大逐元素误差 = %.3e  %s\n", max_err, max_err < 1e-4 ? "✅ 通过" : "❌ 超差");

    // ---- 性能: 算有效带宽(读 a + 读 b + 写 c = 3 次 n 个 float 的搬运)----
    double moved = 3.0 * n * sizeof(float);                // 总搬运字节数
    printf("\n=== 性能(memory-bound, 看带宽而非算力)===\n");
    printf("%-22s %10s %16s\n", "版本", "耗时(ms)", "有效带宽(GB/s)");
    printf("A 一线程一元素  %14.4f %16.1f\n", ms_A, moved / (ms_A*1e-3) / 1e9);
    printf("B grid-stride   %14.4f %16.1f\n", ms_B, moved / (ms_B*1e-3) / 1e9);
    printf("(H100 HBM 约 3.35 TB/s; 本卡跑到几百~上千 GB/s 即正常, 越接近峰值越好)\n");

    // 收尾: 显存/内存各自释放, 否则泄漏
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c); free(h_ref);
    return 0;
}
```

### 6.1 为什么要多写一个 grid-stride 版?(这是"深化"的灵魂)

计划让你写"深化版",最大的深化就是这个 grid-stride loop。它和"一线程一元素"的区别,值得你停下来想清楚:

| | 版本 A:一线程一元素 | 版本 B:grid-stride loop |
|---|---|---|
| 开多少线程 | 随 n 膨胀(n=1 亿就开 1 亿线程) | **固定**(比如 SM 数 × 8 块) |
| 每个线程干几件事 | 1 件 | 多件(循环) |
| 数据规模和 launch 规模 | 耦合(n 变,grid 就得变) | **解耦**(grid 固定,循环自适应) |
| 好处 | 直观,教学友好 | 线程复用摊薄启动开销、每线程多元素有指令级并行、grid 大小可独立调优 |

**为什么工业界偏爱 grid-stride?** 因为真实场景里 `n` 可能巨大(几亿),你不可能真开几亿个线程让调度器慢慢消化——不如开一个"刚好把机器填满"的固定 grid(比如 `SM数 × 每SM住几块`),让每个线程**循环着多搬几趟**。这样既控制了 launch 规模,又能独立调节 grid 去压 occupancy,还能让一个线程连续处理多个元素来摊薄循环开销。**这是 NVIDIA 官方博客钦点的标准模式**,你以后翻 cuBLAS/cuDNN 之外的很多手写 kernel 都会见到它。

> **一个反直觉点**:grid-stride 里 `i += stride` 跨的是"整个 grid 的线程总数",这样**每一轮**相邻线程访问的仍是相邻地址(`i` 和 `i+1` 在线程间是连续的),访存依然是合并(coalesced)的——这就是为什么 stride 取 `gridDim*blockDim` 而不是取 `blockDim`。这个细节 W1 后面讲"合并访存"时会回来深挖,今天先记住"stride=整个 grid 的线程数"是为了保住访存友好性。

### 6.2 跑完怎么看结果(预期与解读)

- **教学演示段**:你会看到 `block=0 thread=0 -> idx=0`、`block=1 thread=0 -> idx=256` 这样逐行打印,且提示"多派 24 个"——亲手验证了 §3.3 的推算。
- **正确性**:最大误差应为 `0.000e+00`(整数当 float 加,无舍入差)。如果不是 0 或报错,先跑 `compute-sanitizer ./vadd` 查越界。
- **性能**:A、B 两版带宽通常**接近**(都贴着 HBM 带宽跑),因为 vector-add 是纯 memory-bound,怎么写都快——这本身就是要学的感觉:**对这种算子,优化空间在"别犯错",不在"写出花"。** 真正的优化战场是后面 compute-bound 的 matmul/attention。

---

## 7. 工业锚点:这套线程模型在真实项目里长什么样

### 7.1 你引擎里的每个 kernel,都是今天这个骨架

你 Day1-4 搭的 miniLLM-serve 里,RMSNorm、RoPE、SwiGLU、attention,如果用 CUDA 原生写,**每一个 kernel 开头都是同一行**:`int idx = blockIdx.x * blockDim.x + threadIdx.x;` 后面接 `if (idx < ...)`。今天你写的 vector-add 骨架,和 FlashAttention 的某个 elementwise 部分骨架**结构上完全一样**——区别只在中间"算什么"。**这就是为什么 vector-add 是 GPU 的第一课:它把"分工 + 边界"这套所有 kernel 共有的外壳,扒到最干净给你看。**

### 7.2 启动配置 `<<<grid, block>>>` 是推理性能的一大调参旋钮

W0 Day5 你学过"改 kernel 逻辑 ≠ 改启动配置,两者正交"。现在你能更具体地理解:推理引擎里大量性能调优,就是在**不改"算什么"的前提下,调"派多少块、每块多大、一线程处理几个元素"**。block 取 256 不是巧合——它是 32(warp)的倍数、又不至于大到浪费资源,是经验甜点。这套"调 block/grid/grid-stride"的手感,是你后面做 kernel 优化的日常。

### 7.3 "block 独立"这条铁律,正是 AMK 巨核要打破的东西

§2.4 说过:CUDA 的可扩展性来自"block 互不依赖、顺序不定"。而你研线的 AMK(AutoMegaKernel)做的是"巨核"——让计算**跨多个 SM 协作**,这本质上是**在打破 block 独立这条铁律**,所以必须自己处理跨 SM 同步,也因此产生同步开销。**你今天对"为什么 block 默认独立"的理解,就是你下周分析 AMK 那份 ncu report 里同步开销的理论基础**——report 里那些 stall/barrier 计数器,讲的就是"打破独立要付多少钱"。这是学和研两条线今天的交汇点。

---

## 8. 常见陷阱与调试技巧

1. **忘了 `if (idx < n)`,或写反方向** → `illegal memory access` 或偶发错值。
   *自查*:只要 `n` 不是 block 整数倍(几乎永远不是),最后一块就有边角料,就必须 `if`。

2. **gridSize 忘了向上取整**,写成 `n / blockSize` → 少派一个块,漏算尾部元素,结果尾部是垃圾值。
   *记住*:`(n + blockSize - 1) / blockSize`,背下来,你会写一万次。

3. **block 超过上限(1024 线程/块)** → kernel 启动直接失败。
   *自查*:`cudaGetLastError()` 抓的正是这种"启动配置错"。block 取 128/256/512 这几个 32 的倍数里最稳妥。

4. **以为 kernel 同步,忘了 `cudaDeviceSynchronize()`** → 打印看不到、计时量到的是发射空档不是真算的时间。
   *记住*:kernel 启动是异步的,CPU 发完就走;要看结果/计时,必须先同步。

5. **不查 CUDA 错误码** → 错了也不知道,拿到一堆错值还找不到原因。
   *习惯*:`CUDA_CHECK` 包住每个 cuda 调用;kernel 后连查两次(`cudaGetLastError` 抓启动错 + `cudaDeviceSynchronize` 抓执行错)。

6. **首选调试器:`compute-sanitizer`。**
   ```bash
   compute-sanitizer ./vadd          # 一键扫描越界/非法访存/未初始化读, 精确到哪个线程
   compute-sanitizer --tool memcheck ./vadd     # 专查内存错误
   compute-sanitizer --tool racecheck ./vadd    # 查共享内存竞争(W1 后面用)
   ```
   比肉眼找越界快一百倍。**这是你在 H100 上调 kernel 的第一救命工具**,和 `CUDA_CHECK` 是左右护法。

7. **kernel 里打印用 `printf`,不是 Python 思维** → CUDA kernel 里可以直接 `printf`(记得配合 `cudaDeviceSynchronize` 才会flush出来),但**只在少量线程里打**(像我们的 `if (idx<8)`),否则刷屏。

---

## 9. 【研 · Track B】H100 环境自检(今天只保温,不深挖)

承接 W0 Day5 的链路,确认 H100 上的 AMK 工具链没凉(久没碰,uv 环境可能要重装):

```bash
module load CUDA/12.4                                        # 链路第二环
nvcc --version                                               # 应打印 release 12.4
nvidia-smi                                                   # 能看到 H100、驱动正常
python -c "import torch; print(torch.cuda.get_device_name(0))"   # 应打印 H100
amk compile small --gpu h100                                 # 跑通小算例,确认 AMK 环境没坏
```

目标只有一个:**确认"H100 上能编译能跑 CUDA + AMK 工具链没坏"**,为下周在真实硬件上做 nsys/ncu profiling 铺路。今天不深入 AMK,别跑偏——§7.3 已经把"今天的线程模型"和"AMK 的跨 SM 同步"这条线接上了,那就是你下周的抓手。

> 另一条线【造 1.5h】(补完整 Llama decoder、确认 `generate()` 出通顺文本)是引擎代码任务,不属于本笔记的"知识点整理"范围——但它的每个算子未来要 CUDA 化时,用的全是今天的线程模型骨架(见 §7.1)。

---

## 10. 串联表 + 自测题 + 完成标准

### 10.1 和已有笔记的串联(知识沉淀链)

| 本笔记的位置 | 呼应/铺垫 |
|---|---|
| 承接 **W0 Day5**(链路 + first-look) | 把那天埋的三个钩子(warp 边界 / memcpy 开销 / warp 分化)在 §5、§5.5、§4 全部收口 |
| 对照 **W8 Day1**(Triton vector-add) | §5 是同一件事的两个抽象层;mask↔if、program↔block 一一对应 |
| 呼应 **W8 Day1 §3.3 / §4.3** | 谓词执行(§4.3)、occupancy(§4.4)在 CUDA 侧的对应,机制是同一个 |
| 呼应 **W7 Day2**(ncu/occupancy) | §4.4 的占用率概念,是读 ncu report 的前置 |
| 呼应 **W6**(KV Cache) | §5.5"数据常驻显存"正是 KV Cache 的设计动机 |
| 铺垫 **W1 后续**(shared memory、合并访存、gemm) | §2 block 内协作、§6.1 coalescing 伏笔,都是后面几天的主角 |
| 铺垫 **AMK 研线** | §2.4 / §7.3"block 独立"铁律 = 分析巨核跨 SM 同步开销的理论基础 |

### 10.2 自测四问(合上书能答才算过;答案位置已标)

1. **五级层级是哪五级?软件层和硬件层各管什么?Block 和 SM 是什么映射关系?**(答案:§2.3)
2. **`blockIdx.x * blockDim.x + threadIdx.x` 每一项是什么?`blockDim` 为什么在乘号里?亲手算一遍 `n=1000, block=256` 下 block 2 thread 100 的 idx。**(答案:§3.2、§3.3)
3. **`if (idx < n)` 为什么必须有?为什么说它"几乎免费"?(提示:从 warp 分化和谓词执行两个角度答)**(答案:§4.1、§4.2、§4.3)
4. **一个 Triton program 底层对应几个 warp?CUDA 手写的那套索引,和 Triton 藏起来的东西,边界划在哪一层?**(答案:§5.3、§5.4)

### 10.3 完成标准 checklist(对照产出逐条打勾)

- [ ] **`cuda/01_vector_add.cu` 编译、运行通过**,正确性验证最大误差为 0(或 < 1e-4)。
- [ ] **教学演示段亲眼看到** `block/thread → idx` 的打印,和"多派 24 个"的提示。
- [ ] **能默写灵魂公式**,并徒手算出任意 `(blockIdx, threadIdx)` 的全局 idx。
- [ ] **能口述五级层级**,说清"Block 住进 SM、Warp 一次调度 32 线程"这两条映射。
- [ ] **能讲清 `if (idx<n)` 为什么几乎免费**(谓词化 + 至多一个 warp 分化)。
- [ ] **能一句话说清 CUDA 与 Triton 的粒度差异本质**(program≈block,差在"块内拆线程谁来做")。
- [ ] **跑了 `compute-sanitizer ./vadd`**,确认无越界,把它加进自己的调试肌肉记忆。

> 全部打勾,今天的线程模型就焊进地基了。往后你每天写的每个 CUDA kernel,开头都是今天这行灵魂公式——区别只在中间"算什么"。

---

### 附:今日一句话总结

**GPU 用"Grid→Block→Thread"的软件抽象对上"SM→Warp"的硬件实体,靠 Block 独立换来任意规模的可扩展性;每个线程靠 `blockIdx.x*blockDim.x+threadIdx.x` 认领一个元素,用一句几乎免费的 `if (idx<n)` 拦住整块派工的边角料;而 Triton 的 program 不过是把"块内拆线程"这一步从你手里交给编译器——你手写 CUDA 时写的,正是它藏起来的那一层。**
