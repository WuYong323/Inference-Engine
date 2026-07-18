import time
import torch

# ══════════════ 三尺子：验"对不对" ══════════════
def measure_error(actual:torch.Tensor,ref:torch.Tensor)->dict:
    actual,ref=actual.float(),ref.float()
    abs_err=(actual-ref).abs()
    rel_err=abs_err/(ref.abs()+1e-8)
    return {
        "max_abs": abs_err.max().item(),  # 尺1：最大绝对误差
        "max_rel": rel_err.max().item(),  # 尺2：最大相对误差
        "allclose": torch.allclose(actual, ref,  # 尺3：综合判定
                                   rtol=1e-3, atol=1e-5),
    }



# ══════════════ 测"快不快" ══════════════
def benchmark(fn,*args,warmup:int=10,iters:int=100,**kwargs)->float:
    """返回单次调用的中位数耗时（毫秒）
    铁律1 —— 预热（warmup）：先空跑几次再计时。
        GPU 有时钟爬升、cuDNN 有 autotune、首次调用有一次性开销，
        不预热测到的是"冷启动"时间，不是稳态性能。
    铁律2 —— 同步（synchronize）：GPU kernel 是异步下发的
        Python 里调完函数它可能还没算完就返回了。不 synchronize，
        测到的是"下发命令的时间"，不是"算完的时间"——能差几个数量级。
    铁律3 —— 多次取中位数：单次计时受系统抖动干扰大。
    """
    use_cuda=torch.cuda.is_available() and any(isinstance(a,torch.Tensor) and a.is_cuda for a in args)

    def sync():
        if use_cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        fn(*args,**kwargs)
    sync()

    times=[]
    for _ in range(iters):
        sync()
        t0=time.perf_counter()
        fn(*args,**kwargs)
        sync()
        times.append((time.perf_counter()-t0)*1e3)

    times.sort()
    return times[len(times)//2]











































