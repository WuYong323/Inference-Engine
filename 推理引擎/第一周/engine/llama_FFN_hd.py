# Llama 计算 FFN hidden dim 的真实逻辑（官方 model.py）

def llama_ffn_hidden_dim(dim:int,multiple_of:int=256,ffn_dim_multiplier:float|None=None)->int:
    hidden=4*dim
    hidden=int(2*hidden/3)
    if ffn_dim_multiplier is not None:
        hidden=int(ffn_dim_multiplier*hidden)
    hidden=multiple_of*((hidden+multiple_of-1)//multiple_of)
    return hidden

print(llama_ffn_hidden_dim(4096))