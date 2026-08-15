import torch
import kernels

def vector_add():
    a = torch.randn(1000, device='cuda')
    b = torch.randn(1000, device='cuda')
    c = kernels.vector_add(a,b)
    print(c)


if __name__=="__main__":
    vector_add()