# engine/rope.py
# ─────────────────────────────────────────────────────────────
# 运行环境：Python 3.10+ / torch 2.x；有无 GPU 均可（张量 .cuda() 即上 GPU）
# 作用：RoPE 的两个核心函数 —— 预计算旋转因子、把旋转应用到 Q/K。
# 约定张量布局：[B, S, H, D]  (Batch, Seq, num_Heads, head_Dim)  ← Llama 官方布局
# ─────────────────────────────────────────────────────────────

import torch

def precompute_freqs_cis(head_dim:int,max_seq_len:int,base:float=10000.0)->torch.Tensor:
    """预计算所有 (位置, 频率) 组合的旋转因子 e^{i·m·θ_i}。

    为什么要"预计算"？旋转因子只跟"位置"和"维度"有关，跟具体输入无关，
    所以整段推理它都是常量。开机算一次、缓存起来，每步生成直接查表，
    避免在 attention 热路径里反复算 sin/cos —— 这是工业实现的标准优化。

    返回：复数张量 freqs_cis，形状 [max_seq_len, head_dim/2]，每个元素模长为 1。
    """
    assert head_dim&2==0, "head_dim 必须是偶数：RoPE 把维度两两配对旋转"
    exponent=torch.arange(0,head_dim,2).float()/head_dim
    freqs=1.0/(base**exponent)
    t=torch.arange(max_seq_len).float()
    # 外积：freqs[m, i] = m * θ_i，即"位置 m 在第 i 组要转的角度"
    freqs=torch.outer(t,freqs)                                # [S, d/2]
    # polar(模长=1, 角度=freqs) → e^{i·freqs} = cos + i·sin，一步造出复数旋转因子
    freqs_cis=torch.polar(torch.ones_like(freqs),freqs)       # complex64, [S, d/2]
    return freqs_cis


def _reshape_for_broadcast(freqs_cis:torch.Tensor,x:torch.Tensor)->torch.Tensor:
    """把 [S, d/2] 的 freqs_cis 变形成能和 [B, S, H, d/2] 广播的形状 [1, S, 1, d/2]。
    为什么单独抽出来：广播维度对不齐是 RoPE 最高频的 bug，集中在一处、写清楚。"""
    ndim=x.ndim
    assert freqs_cis.shape==(x.shape[1],x.shape[-1]), f"freqs_cis {tuple(freqs_cis.shape)} 与 x 的 (S, d/2)=({x.shape[1]},{x.shape[-1]}) 不匹配"
    # 只在 S 维(第1维)和 d/2 维(最后一维)保留真实长度，其余(B、H)设为 1 让它广播
    shape=[dim if i==1 or i==ndim-1 else 1 for i,dim in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rope(xq:torch.Tensor,xk:torch.Tensor,freqs_cis:torch.Tensor):
    """把 RoPE 旋转应用到 Q 和 K 上。xq/xk: [B, S, H, D]。

    只转 Q 和 K，不转 V —— 这是重点：位置信息只需影响"谁该注意谁"(由 QKᵀ 决定)，
    而 V 携带的是"内容本身"，不该被位置扰动。这也是 RoPE 比"加到嵌入上"更干净的地方。
    """
    # 1) 把最后一维 D 看成 D/2 个 (实, 虚) 对 → 转成复数张量 [B, S, H, D/2]
    #    .float() 是必须的：复数运算 + 角度精度对数值敏感，低精度会累积误差
    xq_c=torch.view_as_complex(xq.float().reshape(*xq.shape[:-1],-1,2))
    xk_c=torch.view_as_complex(xk.float().reshape(*xk.shape[:-1],-1,2))
    
































