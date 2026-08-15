import torch
from 推理引擎.engine.backend import TorchBackend
from 推理引擎.engine.model import LlamaModel


class config:
    vocab_size:int=50257
    batch_size:int=16
    block_size:int=1024
    n_embd:int=768
    n_head:int=12
    n_layer:int=12
    dropout:float=0.1
    multiple_of:int=256
    max_seq_len:int=1000


def build():
    torch.manual_seed(0)
    return LlamaModel(config(),TorchBackend())


def test_forward_shape_and_finite():
    """体检①：形状对 + 无 NaN/Inf + logits 数量级合理。"""
    model=build().eval()
    B,T=2,16
    tokens=torch.randint(0,config.vocab_size,(B,T))
    with torch.no_grad():
        logits,_=model(tokens)

    assert logits.shape == (B, T, config.vocab_size), f"形状错: {logits.shape}"

    assert torch.isfinite(logits).all(), "出现 NaN/Inf！检查 RMSNorm 的 eps 和残差"

    assert logits.abs().max() < 100, f"logits 数量级异常: {logits.abs().max():.1f}"
    print(f"[OK] logits shape={tuple(logits.shape)}, max|logit|={logits.abs().max():.2f}")


def count_params_by_formula(c:config)-> int:
    """体检②：按架构公式手算参数量（权重 tying：embedding 只算一次）。
        把这个函数当'架构自测'——它逼你写清每一块到底有多少参数。"""
    from 推理引擎.engine.model import llama_ffn_hidden_dim
    h=llama_ffn_hidden_dim(c.n_embd,c.multiple_of)

    emb=c.vocab_size*c.n_embd
    attn=4*c.n_embd*c.n_embd
    ffn=3*c.n_embd*h
    norms=2*c.n_embd
    per_layer=attn+ffn+norms
    final_norm=c.n_embd
    return emb+c.n_layer*per_layer+final_norm


def test_param_count_matches_formula():
    """实测参数量必须等于手算公式 —— 对不上就是架构理解有漏洞。"""
    model = build()
    actual = sum(p.numel() for p in model.parameters())
    # 权重 tying 下，sum(parameters()) 里 embedding 只出现一次，与公式一致
    formula = count_params_by_formula(config())
    print(f"[参数量] 实测={actual:,}  手算={formula:,}")
    assert actual == formula, f"对不上！实测 {actual:,} vs 手算 {formula:,}"










