#include <torch/extension.h>        // PyTorch C++ 扩展的“万能头文件”

// ============================================================
// 所有 kernel 的统一注册入口。
// 新增 kernel 时:
//   1. 在下方声明它的 wrapper(实现在各自的 .cpp 里)
//   2. 在 PYBIND11_MODULE 里加一行 m.def
//   3. 在 test_cuda_ext.py 的 sources 里加上新的 .cpp / .cu
// ============================================================


//这行并不是实现，是告诉编译器：存在一个叫 vector_add 的函数，它接收两个张量，返回一个张量。
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
// torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps);


// 核心绑定宏
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "vector add (CUDA)");
    // m.def("rmsnorm", &rmsnorm, "rmsnorm (CUDA)");
}
