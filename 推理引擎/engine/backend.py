# ─────────────────────────────────────────────────────────────
# 运行环境：Python 3.10+ / torch 2.x
#   - 有无 GPU 都能跑：CPU 上直接运行；GPU 上把张量 .cuda() 即可
#   - 无第三方依赖（只用 torch），Day 1 保持最小依赖
# 作用：定义"算子后端"的统一接口，让上层模型只认接口、不认实现。
# ─────────────────────────────────────────────────────────────

import math
from abc import ABC,abstractmethod

import torch
import torch.nn.functional as F


class Backend(ABC):
    """
    算子后端基类：上层只认这三个方法，不关心底层是谁实现的。
    """

    @abstractmethod
    def rmsnorm(self,x:torch.Tensor,weight:torch.Tensor,eps:float=1e-6)->torch.Tensor:
        ...

    @abstractmethod
    def attention(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,*,causal:bool=True)->torch.Tensor:
        ...

    @abstractmethod
    def fused_ffn(self,x:torch.Tensor,w_gate:torch.Tensor,w_up:torch.Tensor,w_down:torch.Tensor)->torch.Tensor:
        """SwiGLU 前馈网络（对应草稿里的 w1/w2/w3）。
            w_gate=w1, w_up=w2, w_down=w3；每个 weight 形如 [out, in]。"""
        ...


class TorchBackend(Backend):
    """
    纯 PyTorch 实现
    baseline（基准参照）。
    """

    def rmsnorm(self,x,weight,eps=1e-6):
        # RMSNorm 数学：  y = x / sqrt(mean(x^2) + eps) * weight
        # 关键工程细节：先转 float32 再算，最后转回原 dtype。
        in_dtype=x.dtype
        x=x.float()
        variance=x.pow(2).mean(dim=-1,keepdim=True)
        x_norm=x*torch.rsqrt(variance+eps)
        return (x_norm.to(in_dtype))*weight

    def attention(self,q,k,v,*,causal=True):
        # q,k,v: [B, H, S, D]  (Batch, num_Heads, Seq_len, head_Dim)
        # 标准缩放点积注意力：softmax(QK^T / sqrt(D)) @ V
        d=q.size(-1)
        scale=1.0/math.sqrt(d)
        scores=torch.matmul(q,k.transpose(-2,-1))*scale

        if causal:
            # 因果掩码（causal mask）：生成任务里，第 i 个 token 只能看它自己和之前的，
            # 不能偷看未来。把"未来"位置的分数设成 -inf，softmax 后权重就是 0。
            S=q.size(-2)
            future=torch.triu(torch.ones(S,S,dtype=torch.bool,device=q.device),diagonal=1)
            scores=scores.masked_fill(future,float("-inf"))

        attn=torch.softmax(scores,dim=-1)

        return torch.matmul(attn,v)

    def fused_ffn(self,x,w_gate,w_up,w_down):
        # Llama 的 SwiGLU：down( SiLU(gate(x)) * up(x) )
        # gate/up 把 x 从 d 维升到隐藏维 h，逐元素相乘做"门控"，down 再降回 d 维。
        # weight 形如 [out, in]，所以用 x @ w.T（等价于 nn.Linear 的 no-bias 前向）。
        # "fused"在 TorchBackend 里其实没真融合——纯 PyTorch 会拆成好几个 kernel。
        gate=F.silu(x@w_gate.T)
        up=x@w_up.T
        return (gate*up)@w_down.T


if __name__=="__main__":
    torch.manual_seed(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    be=TorchBackend()

    # rmsnorm
    x=torch.randn(2,8,64,device=device)
    w=torch.ones(64,device=device)
    assert  be.rmsnorm(x,w).shape==x.shape

    # attention
    q = k = v = torch.randn(2, 4, 16, 32, device=device)  # [B,H,S,D]
    assert be.attention(q, k, v).shape == (2, 4, 16, 32)

    # fused_ffn
    d,h=64,172
    x2=torch.randn(2,8,d,device=device)
    wg,wu=torch.randn(h,d,device=device),torch.randn(h,d,device=device)
    wd=torch.randn(d,h,device=device)
    assert be.fused_ffn(x2, wg, wu, wd).shape == x2.shape

    print(f"TorchBackend 三个算子形状全部通过（device={device}）")





















