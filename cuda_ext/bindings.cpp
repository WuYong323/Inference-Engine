#include <torch/extension.h>        // PyTorch C++ 扩展的“万能头文件”

// ============================================================
// 所有 kernel 的统一注册入口。
// 新增 kernel 时:
//   1. 在下方声明它的 wrapper(实现在各自的 .cpp 里)
//   2. 在 TORCH_LIBRARY 里加 schema,在 TORCH_LIBRARY_IMPL 里加 impl
//   3. 在 cuda_ext.py 的 sources 里加上新的 .cpp / .cu
//   4. 在 Python 侧给它写一个 register_fake(形状推导)
// ============================================================


//这行并不是实现，是告诉编译器：存在一个叫 vector_add 的函数，它接收两个张量，返回一个张量。
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
// torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps);


// ------------------------------------------------------------
// 1) 声明算子 schema。注册进 torch 的算子表,调用方式 torch.ops.kernels.vector_add
//    schema 字符串是给 dispatcher 看的类型签名,Tensor/int/float/bool 等
// ------------------------------------------------------------
TORCH_LIBRARY(kernels, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
    // m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
}

// ------------------------------------------------------------
// 2) 给 CUDA 这个后端绑定具体实现。
//    以后要支持 CPU,再写一个 TORCH_LIBRARY_IMPL(kernels, CPU, m),
//    调用方代码不用改,dispatcher 按张量所在设备自动选。
// ------------------------------------------------------------
TORCH_LIBRARY_IMPL(kernels, CUDA, m) {
    m.impl("vector_add", &vector_add);
    // m.impl("rmsnorm", &rmsnorm);
}

// ------------------------------------------------------------
// 3) 空的 pybind 模块。上面两个宏是靠静态初始化注册的,
//    而静态初始化要等 .pyd 被加载才会跑,所以仍然需要一个 PyInit_kernels
//    入口让 `import kernels` 能成功 —— 里面不导出任何函数。
// ------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
