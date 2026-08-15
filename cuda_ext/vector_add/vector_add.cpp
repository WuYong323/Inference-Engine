#include <torch/extension.h>

// launcher implemented in vector_add_kernel.cu
void launch_vector_add(const float* a, const float* b, float* c, int n);

// C++ wrapper: check inputs, call kernel launcher
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = torch::empty_like(a_c);

    launch_vector_add(
        a_c.data_ptr<float>(),
        b_c.data_ptr<float>(),
        c.data_ptr<float>(),
        static_cast<int>(a_c.numel()));

    return c;
}

// PYBIND11_MODULE 统一放在 bindings.cpp,这里只实现 wrapper
