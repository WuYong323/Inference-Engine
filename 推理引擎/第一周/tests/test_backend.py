import torch
import pytest

from 第一周.engine.backend import TorchBackend



def rmsnorm_reference(x,weight,eps=1e-6):
    #baseline
    x=x.float()
    var=x.pow(2).mean(-1,keepdim=True)
    return (x*torch.rsqrt(var+eps)).type_as(weight)*weight


@pytest.mark.parametrize("shape",[(2,8,64),(1,128,512),(4,1,4096)])
def test_rmsnorm_matches_reference(shape):
    torch.manual_seed(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    x=torch.randn(*shape,device=device)
    w=torch.randn(shape[-1],device=device)

    out=TorchBackend().rmsnorm(x,w)
    ref=rmsnorm_reference(x,w)

    # 用 allclose 判正确性；失败时打印最大误差，方便定位是"差一点"还是"全错"
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-6), f"max_abs={(out - ref).abs().max().item()}"









































