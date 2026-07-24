import torch
from 推理引擎.第一周.engine.rope import precompute_freqs_cis,apply_rope


def _rope_single(vec:torch.Tensor,pos:int,freqs_cis:torch.Tensor)->torch.Tensor:
    D=vec.shape[0]
    x=torch.zeros(1,pos+1,1,D)
    x[0,pos,0]=vec
    x_rot,_=apply_rope(x,x,freqs_cis)
    return x_rot[0,pos,0]


def test_rope_relativity():
    """性质①：点积只依赖相对距离 (m−n)。
    只要 m−n 相同，dot(rope(q,m), rope(k,n)) 就必须不变 —— 这是 RoPE 的立身之本。"""
    torch.manual_seed(0)
    D=64
    freqs_cis=precompute_freqs_cis(D,max_seq_len=128)
    q,k=torch.randn(D),torch.randn(D)

    def rope_dot(m,n):
        return (_rope_single(q,m,freqs_cis)*_rope_single(k,n,freqs_cis)).sum().item()

    base = rope_dot(5, 3)
    for m, n in [(10, 8), (20, 18), (100, 98)]:
        assert abs(rope_dot(m, n) - base) < 1e-4, f"相对性被破坏：dot({m},{n})={rope_dot(m, n):.6f} != {base:.6f}"


def test_rope_zero_position_is_identity():
    """性质②：位置 0 的旋转是恒等变换。
    因为 e^{i·0·θ} = 1，位置 0 乘的旋转因子全是 1，向量原封不动。"""
    D = 64
    freqs_cis = precompute_freqs_cis(D, max_seq_len=16)
    v = torch.randn(D)
    v_rot = _rope_single(v, 0, freqs_cis)
    assert torch.allclose(v_rot, v, atol=1e-6), "位置0应恒等，但向量被改变了"



def test_rope_preserves_norm():
    """附赠性质③：旋转不改变向量长度（旋转是正交变换）。
    这是个便宜又强的 sanity check —— 如果模长变了，一定是实现错了。"""
    torch.manual_seed(1)
    D = 128
    freqs_cis = precompute_freqs_cis(D, max_seq_len=64)
    v = torch.randn(D)
    v_rot = _rope_single(v, 37, freqs_cis)
    assert torch.allclose(v.norm(), v_rot.norm(), atol=1e-5), "旋转不该改变模长"









































