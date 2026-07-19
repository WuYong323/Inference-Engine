from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path



def get_batch(block_size,batch_size,device):
    ix=torch.randint(len(data)-1-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device),y.to(device)



class CausalSelfAttention(nn.Module):
    def __init__(self,n_embd,dropout,n_head):
        super().__init__()
        self.c_attn=nn.Linear(n_embd,3*n_embd,bias=False)
        self.c_proj=nn.Linear(n_embd,n_embd,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.n_embd=n_embd
        self.n_head=n_head

    def forward(self,x):
        B,T,C=x.shape
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
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
    def __init__(self,n_embd,dropout,n_head):
        super().__init__()
        self.ln_1=nn.RMSNorm(n_embd)
        self.attn=CausalSelfAttention(n_embd,dropout,n_head)
        self.ln_2=nn.RMSNorm(n_embd)
        self.mlp=MLP(n_embd,dropout)
        self.mlp._is_residual_proj=True     #初始化

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config.n_embd,config.dropout,config.n_head) for _ in range(config.n_layer)]),
            ln_f=nn.RMSNorm(config.n_embd)
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight=self.lm_head.weight
        self.apply(self._init_weights)      #先从最外层的 GPT 开始，调用 _init_weights( ),遍历所有的子模块
        self.n_layer=config.n_layer
        self.vocab_size=config.vocab_size
        self.block_size=config.block_size

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
        tok_emb=self.transformer.wte(idx)
        pos_emb=self.transformer.wpe(torch.arange(T,device=idx.devicd))
        x=tok_emb+pos_emb
        for block in self.transformer.h:
            x=block(x)
        x=self.transformer.ln_f(x)
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
    block_size:int=1024
    n_embd:int=768
    n_head:int=12
    n_layer:int=12
    dropout:float=0.1


if __name__=="__main__":
    torch.manual_seed(0)

    config={
        'batch_size':16,
        'block_size':128,
        'n_embd':256,
        'n_head':4,
        'n_layer':4,
        'dropout':0.0,
    }

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

    if Path("checkpoint.pt").exists():
        print("权重文件存在")
        ckpt=torch.load("checkpoint.pt",map_location="cuda")
        model = GPT(GPTConfig(**ckpt['model_args'])).to(device)
        model.load_state_dict(ckpt["model"])

    else:
        print("未找到权重文件，从头训练")
        model = GPT(GPTConfig(**config)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)  # AdamW + 3e-4:LLM 默认起手式
        for step in range(2000):
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















































