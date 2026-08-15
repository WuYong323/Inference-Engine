"""自定义 CUDA kernel 的 Python 接口层。

用法就是你想要的那样:
    import kernels
    c = kernels.vector_add(a, b)          # 可反向传播，torch.compile 不断图

这一层做三件事:
    1. 加载编译好的二进制 _kernels.pyd,触发 C++ 里 TORCH_LIBRARY 的静态注册
    2. 给每个算子注册 fake(形状推导),torch.compile 才能 trace 它
    3. 给每个算子注册反向,才能用在神经网络里

二进制由 cuda_ext.py 编译产生(build_kernel.bat),本文件只负责加载,
运行时不需要 MSVC / nvcc。
"""
import os
import torch
from torch.utils.cpp_extension import LIB_EXT, _get_build_directory

# ---------------------------------------------------------------
# 1) 加载二进制
#    load_library 走的是 LoadLibrary/dlopen,不经过 Python import 机制,
#    所以不用改 sys.path,也不用把 .pyd 复制到项目里。
# ---------------------------------------------------------------
_LIB = os.path.join(_get_build_directory("_kernels", False), f"_kernels{LIB_EXT}")
if not os.path.exists(_LIB):
    raise ImportError(f"找不到编译产物:\n  {_LIB}\n先编译一次: build_kernel.bat")
torch.ops.load_library(_LIB)


# ---------------------------------------------------------------
# 2) vector_add
# ---------------------------------------------------------------
@torch.library.register_fake("kernels::vector_add")
def _vector_add_fake(a, b):
    """只推形状和 dtype,不做真实计算 —— torch.compile 在 FakeTensor 下调这个。
    没有它,编译时会因为 kernel 跑不了 meta 张量而失败。"""
    torch._check(a.shape == b.shape)
    return torch.empty_like(a)


def _vector_add_backward(ctx, grad):
    """c = a + b  =>  dL/da = dL/db = grad。
    没有它,backward() 不报错但 grad 是 None(静默错)。"""
    return grad, grad


torch.library.register_autograd("kernels::vector_add", _vector_add_backward)

vector_add = torch.ops.kernels.vector_add


# 新增 kernel 时,照上面 vector_add 的三段(fake / backward / 导出)复制一份。

__all__ = ["vector_add"]
