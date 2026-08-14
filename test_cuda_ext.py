"""编译并测试自定义 CUDA 扩展"""
import os
import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")  # RTX 5060 = sm_120

ext = load(
    name="vector_add_ext",
    sources=["cuda_ext/vector_add.cpp", "cuda_ext/vector_add_kernel.cu"],
    extra_cuda_cflags=["-O3", "-Xcompiler", "/utf-8"],
    verbose=True,
)

a = torch.randn(1_000_000, device="cuda")
b = torch.randn(1_000_000, device="cuda")
c = ext.vector_add(a, b)

torch.cuda.synchronize()
assert torch.allclose(c, a + b), "结果不匹配!"
print("CUDA extension compiled, GPU kernel result correct")
