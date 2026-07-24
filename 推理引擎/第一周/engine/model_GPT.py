from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from rope import precompute_freqs_cis,apply_rope
from backend import TorchBackend


def get_batch(block_size,batch_size,device):
    ix=torch.randint(len(data)-1-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device),y.to(device)



class CausalSelfAttention(nn.Module):
    def __init__(self,n_embd,dropout,n_head,backend):
        super().__init__()
        self.c_attn=nn.Linear(n_embd,3*n_embd,bias=False)
        self.c_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.n_embd=n_embd
        assert n_embd % n_head == 0
        self.n_head=n_head
        self.backend=backend


    def forward(self,x,freqs_cis):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        # reshape 成 [B, T, H, D]（注意是 Llama 布局，H 在 D 前）
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head)

        # 改动②：在这里接入 RoPE —— 只旋转 Q、K，V 原样不动
        q, k = apply_rope(q, k, freqs_cis[:T])  # 只取前 T 个位置的旋转因子

        # 转成 [B, H, T, D] 交给注意力算子（对齐 Day1 backend.attention 的布局）
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        y=self.backend.attention(q,k,v,causal=True)
        y=y.transpose(1,2).contiguous().view(B,T,C)
        return self.dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self,n_embd,dropout):
        super().__init__()
        self.c_fc=nn.Linear(n_embd,4*n_embd,bias=False)
        self.gelu=nn.GELU()
        self.c_proj=nn.Linear(4*n_embd,n_embd,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """改动①：LayerNorm → 走 backend.rmsnorm。从今天起，模型层只认 Backend 接口，
    不再自己 new 一个 nn.LayerNorm —— 这样 norm 的实现也能被后端替换（Day1 抽象的兑现）。"""
    def __init__(self,n_embd,dropout,n_head,backend):
        super().__init__()
        self.backend=backend
        # RMSNorm 只有一个缩放参数 γ（没有 β），初始化为全 1
        self.attn_norm_w=nn.Parameter(torch.ones(n_embd))
        self.ffn_norm_w=nn.Parameter(torch.ones(n_embd))
        self.attn=CausalSelfAttention(n_embd,dropout,n_head,backend)     # 内部接入 RoPE（见下）
        self.mlp=MLP(n_embd,dropout)
        self.mlp._is_residual_proj=True     #初始化

    def forward(self,x,freqs_cis):
        # 保持 pre-norm 结构：norm 在子层之前，残差连接不变
        x=x+self.attn(self.backend.rmsnorm(x,self.attn_norm_w),freqs_cis)
        x=x+self.mlp(self.backend.rmsnorm(x,self.ffn_norm_w))
        return x


class GPT(nn.Module):
    def __init__(self,config,backend):
        super().__init__()
        self.backend=backend
        self.wte=nn.Embedding(config.vocab_size,config.n_embd)
        # 改动③：删除绝对位置嵌入 self.wpe！位置信息全交给 RoPE。
        self.h=nn.ModuleList([Block(config.n_embd,config.dropout,config.n_head,self.backend) for _ in range(config.n_layer)])
        self.final_norm_w=nn.Parameter(torch.ones(config.n_embd))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.wte.weight=self.lm_head.weight

        # 预计算旋转因子，注册成 buffer（随模型搬到 GPU，但不是可训练参数）
        head_dim = config.n_embd // config.n_head
        # 不存进 checkpoint：它是可复算的常量，存了浪费空间
        self.register_buffer("freqs_cis",precompute_freqs_cis(head_dim, config.block_size),persistent=False)
        self.n_layer=config.n_layer
        self.vocab_size=config.vocab_size
        self.block_size=config.block_size
        self.apply(self._init_weights)  # 先从最外层的 GPT 开始，调用 _init_weights( ),遍历所有的子模块

    def _init_weights(self,module):
        std=0.02
        if isinstance(module,nn.Linear):
            if hasattr(module,"_is_residual_proj"):
                std*=(2*self.n_layer)**-0.5
            nn.init.normal_(module.weight,mean=0.0,std=std)         #将一个张量（Tensor）原地（in-place）填充为服从正态（高斯）分布的随机数。
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=std)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        x=self.wte(idx)
        for block in self.h:
            x=block(x,self.freqs_cis[:T])
        x=self.backend.rmsnorm(x,self.final_norm_w)
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,self.vocab_size),targets.view(-1))
        return logits,loss

    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-self.block_size:]
            logits,_=self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)
        return idx


@dataclass
class GPTConfig:
    vocab_size:int=50257
    batch_size:int=16
    block_size:int=1024
    n_embd:int=768
    n_head:int=12
    n_layer:int=12
    dropout:float=0.1


if __name__=="__main__":
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("遮天.txt","r",encoding="gbk") as f:
        words=f.read()
    chars = sorted(list(set(words)))
    vocab_size = len(chars)
    stoi = {s: i for i, s in enumerate(chars)}
    itos = {i: s for s, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join(itos[i] for i in l)
    data = torch.tensor(encode(words), dtype=torch.long)

    config={
        'vocab_size':vocab_size,
        'batch_size':16,
        'block_size':128,
        'n_embd':256,
        'n_head':4,
        'n_layer':4,
        'dropout':0.0,
    }

    if Path("checkpoint.pt").exists():
        print("权重文件存在")
        ckpt=torch.load("checkpoint.pt",map_location="cuda")
        model = GPT(GPTConfig(**ckpt['model_args']),TorchBackend()).to(device)
        model.load_state_dict(ckpt["model"])

    else:
        print("未找到权重文件，从头训练")
        model = GPT(GPTConfig(**config),TorchBackend()).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW + 3e-4:LLM 默认起手式
        for step in range(10000):
            xb,yb=get_batch(config['block_size'], config['batch_size'], device)
            _,loss=model(xb,yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 200 == 0:
                print(f"step {step:4d} | loss {loss.item():.4f}")

        # 保存参数
        save_dict={
            'model':model.state_dict(),
            'model_args':{
                'batch_size':config['batch_size'],
                'vocab_size':vocab_size,
                'block_size':config['block_size'],
                'n_layer':config['n_layer'],
                'n_head':config['n_head'],
                'n_embd':config['n_embd'],
                'dropout':config['dropout']
            }
        }

        torch.save(save_dict,"checkpoint.pt")
        print("模型已保存为 checkpoint.pt")

    #输出
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--------------- 生成样例 -------------------")
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))















































