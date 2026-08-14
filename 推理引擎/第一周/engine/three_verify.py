import torch

def overfit_single_batch(model,device="cuda",steps=30,log_every=50):
    """一级验证: 拿一个固定 batch 往死里训, 看 loss 能否 -> ~0。
        通过 = 梯度流是通的; 卡住 = 计算图/梯度有 bug。"""
    model.train().to(device)

    # 关键: 只造一个 batch, 之后每一步都用这同一份数据 —— 这才叫"过拟合单 batch"
    B,T=4,64
    x=torch.randint(0,model.config.vocab_size,(B,T),device=device)
    y=torch.randint(0,model.config.vocab_size,(B,T),device=device)

    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)

    for step in range(steps):
        logits,loss=model(x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step%log_every==0:
            print(f"step  {step:4d}  |  loss  {loss.item():.4f}")

    final=loss.item()
    assert final < 0.1, f"一级验证失败 loss 卡在 {final:.3f}, 查残差/norm/RoPE"
    print(f"一级验证通过, final loss = {final:.4f}")