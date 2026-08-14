import torch,time


@torch.no_grad()
def benchmark_tokens_per_sec(model,prompt_idx,max_new_tokens=256,warmup=2,runs=5,device="cuda"):
    model.eval()

    # 1) 预热(warmup): 头几次跑包含 cuDNN 算法选择、CUDA 上下文初始化、
    #    (若用了 torch.compile)编译等一次性开销, 必须丢弃, 否则严重拉低 tok/s。
    for _ in range(warmup):
        model.generate(prompt_idx,max_new_tokens)
    torch.cuda.synchronize()

    timings=[]
    for _ in range(runs):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        model.generate(prompt_idx,max_new_tokens)
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-t0)

    avg=sum(timings)/len(timings)
    tok_per_s=max_new_tokens/avg
    print(f"平均耗时 {avg * 1000:.1f} ms | 吞吐 {tok_per_s:.1f} tok/s")
    return tok_per_s



