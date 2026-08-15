#include <torch/extension.h>

// ============================================================
// 所有 kernel 的统一注册入口。
// 新增 kernel 时:
//   1. 在下方声明它的 wrapper(实现在各自的 .cpp 里)
//   2. 在 PYBIND11_MODULE 里加一行 m.def
//   3. 在 test_cuda_ext.py 的 sources 里加上新的 .cpp / .cu
// ============================================================

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);
// torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "vector add (CUDA)");
    // m.def("rmsnorm", &rmsnorm, "rmsnorm (CUDA)");
}
