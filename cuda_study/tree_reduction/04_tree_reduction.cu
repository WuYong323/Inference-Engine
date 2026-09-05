// ============================================================================
// 04_tree_reduction.cu   树形归约 + bank conflict + warp shuffle
//
// 四个问题，每个问题对应一组实验：
//   Q1 把 256 个数加成 1 个，从「1 个线程串 256 步」升级到「8 步树形」，到底快多少?
//      > 实验 A（厚归约，每线程处理很多元素）+ 实验 B（薄归约，每线程只处理 1 个元素）
//      > 两个实验的结论完全不同，优化的收益取决于占多大比例
//   Q2 教科书上的「交错寻址」为什么慢？bank conflict 到底能不能被量出来?
//      > 实验 C（bank_probe：把 bank conflict 从访存噪声里隔离出来）
//   Q3 warp 内的最后 5 轮，能不能连 shared memory 都不用？
//      > kernel (E)(F)：__shfl_down_sync 两级归约（工业标准结构）
//   Q4 学的东西怎么变成 CUDA RMSNorm ?
//      > 实验 D（rmsnorm 三个版本，行归约形状）
//
// 本文件的 kernel 清单（(B)(C) 是【反面教材】，故意保留）：
//   (A) reduce_serial_merge     Day3 的基线：shared + thread0 串行合并
//   (B) reduce_interleave_div   NVIDIA reduction.pdf 的 reduce#1：交错寻址 + warp 分化
//   (C) reduce_interleave_conf  reduce#2：修掉分化，但踩满 bank conflict
//   (D) reduce_tree_seq         reduce#3：顺序寻址树形归约 ← 今天的正主
//   (E) reduce_tree_unroll      顺序寻址 + 最后一个 warp 用 shuffle 收尾（模板全展开）
//   (F) reduce_warp_2stage      纯 warp shuffle 两级归约 ← 工业里最常见的写法
//   (G) reduce_cub              CUB 官方实现：你手写是为了懂，上线用这个
//
// 环境要求：H100 (sm_90) + CUDA 12.4；显存 ≥ 2 GB（默认输入 n = 1<<27 = 512 MB）
// 编译：
//   nvcc -O3 -arch=sm_90 -lineinfo -Xptxas -v -o treereduce 04_tree_reduction.cu
//     -lineinfo  : ncu / compute-sanitizer 能把结果对回源码行（Day2 起的习惯）
//     -Xptxas -v : 打印每个 kernel 的寄存器用量 + shared memory 用量
// 运行：
//   ./treereduce              # 默认 n=1<<27，全 1.0f 数据
//   ./treereduce 24 rand      # n=1<<24，随机数据（看相对误差而不是整数值）
//
// ★ 今天必须跑的三条 profiling 命令（比看时间更能建立认知）：
//   # 1) 看 bank conflict：对比 (C) 和 (D)，这个计数器会差几个数量级
//   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
//                l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./treereduce 24
//   # 2) 看 warp 分化：对比 (B) 和 (D)，「每条指令平均有几个活跃线程」（满分 32）
//   ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio ./treereduce 24
//   # 3) 看 shared memory 访问指令数：对比 (D) 和 (F)，shuffle 版应当断崖式下降
//   ncu --metrics smsp__inst_executed_op_shared_ld.sum,\
//                smsp__inst_executed_op_shared_st.sum ./treereduce 24
//   # 附：正确性护栏（Day3 养成的肌肉记忆，改归约代码后必跑）
//   compute-sanitizer --tool racecheck  ./treereduce 20
//   compute-sanitizer --tool synccheck  ./treereduce 20
// ============================================================================