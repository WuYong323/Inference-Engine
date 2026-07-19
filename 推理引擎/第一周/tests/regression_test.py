import torch
from 第一周.engine.model_GPT import GPT, GPTConfig


ckpt=torch.load("checkpoint.pt", map_location="cuda")
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.eval().cuda()

ctx = torch.zeros((1, 1), dtype=torch.long, device="cuda")
with torch.no_grad():
    out = model.generate(ctx, max_new_tokens=100)
print("BASELINE OUTPUT:", out[0].tolist()[:20], "...")

"""
root@autodl-container-a3b611b352-0bc552d6:~/autodl-fs/engine# python regression_test.py
BASELINE OUTPUT: [0, 44, 44, 549, 366, 888, 532, 644, 3422, 4298, 104, 1090, 1798, 4294, 3912, 112, 4295, 0, 44, 44] ...
"""



































