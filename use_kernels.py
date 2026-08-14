"""日常使用示例:直接导入已编译好的 CUDA 扩展,无需编译环境"""
import torch
import vector_add_ext  # 就是项目目录下的 vector_add_ext.pyd

a = torch.randn(5, device="cuda")
b = torch.randn(5, device="cuda")
c = vector_add_ext.vector_add(a, b)

print("a     :", a)
print("b     :", b)
print("a + b :", c)
print("match :", torch.allclose(c, a + b))
