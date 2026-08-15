"""编译并测试自定义 CUDA 扩展"""
import os
import torch
from torch.utils.cpp_extension import load

# 环境准备（指定 GPU 架构）
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")  # RTX 5060 = sm_120

#即时编译（JIT）加载扩展
"""
torch.utils.cpp_extension.load 是 PyTorch 提供的工具：
它会在后台自动调用 nvcc（CUDA编译器）和 cl.exe（MSVC编译器），
将 C++ 和 .cu 文件编译成一个 Python 可以直接导入的动态链接库（.pyd / .so）
"""

"""
编译选项：
    -O3：开启最高等级的优化。
    -Xcompiler /utf-8：将参数传递给 MSVC 编译器，强制使用 UTF-8 编码，避免 Windows 下中文路径或注释导致的编译报错。
verbose=True：编译时会在控制台输出详细的编译命令和日志，便于排查链接或语法错误
"""
ext = load(
    name="_kernels",  # 所有 kernel 编译进这一个扩展（私有二进制，外面用 kernels.py 包一层）
    sources=[
        "cuda_ext/bindings.cpp",        # 连接 C++ 和 Python 的胶水代码
        "cuda_ext/vector_add/vector_add.cpp",      # 调用 CUDA 内核的 C++ 封装函数
        "cuda_ext/vector_add/vector_add_kernel.cu",    # 实际的 GPU 内核代码（.cu）
        # 新 kernel 的 .cpp / .cu 加在这里
    ],
    extra_cflags=["/utf-8"],
    extra_cuda_cflags=["-O3", "-Xcompiler", "/utf-8"],
    verbose=True,
)

print("CUDA extension compiled:",ext.__file__)
