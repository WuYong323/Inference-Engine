import torch
from 第一周.engine.model_GPT import GPT, GPTConfig


ckpt=torch.load("ckpt.pt", map_location="cuda")
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.eval().cuda()

ctx = torch.zeros((1, 1), dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=100)
print("BASELINE OUTPUT:", out[0].tolist()[:20], "...")





































