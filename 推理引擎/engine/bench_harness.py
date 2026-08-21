"""engine/bench_harness.py —— 推理引擎的对标脚手架

用途：W2 起「torch vs triton vs cuda」多后端对标的唯一量尺。
    今天先用 torch 后端跑通，W2 直接往里插新后端。

设计信条：
  三铁律计时 = 预热 / 同步 / 多次取中位数
  三尺子误差 = allclose(过不过) / cosine(方向对不对) / max_abs+max_rel(最坏差多少)
  新增的第四条：算出 achieved bandwidth 和 roofline 上界，
  只报 ms 是没有信息量的 —— 必须知道「离硬件极限还有多远」。
"""

from __future__ import annotations

import json                                         # 处理 JSON 格式数据的序列化（导出）与反序列化（加载）。
import statistics                                   # 提供基础的数学统计函数（均值、中位数、方差等）。
from dataclasses import dataclass,asdict,field      # 专门用来优雅地定义和管理数据类（主要存数据，而非逻辑）。
from typing import Callable,Any                     # 提供类型提示（Type Hints），给代码加上“说明书”和“边界约束”。

import torch
from mpmath import rand
from torch._C import device, dtype

# ---------------------------------------------------------------------------
# 硬件常数：对标必须有「尺子的刻度」。数值来自官方规格，跑在别的卡上要改。
# ---------------------------------------------------------------------------
HW={
    "NVIDIA H100 80G HBM3": dict(hbm_gbs=3350.0,bf16_tflops=989.0),   # SXM, 稠密算力
    "NVIDIA H100 PCIe":     dict(hbm_gbs=2000.0,bf16_tflops=756.0),
    "NVIDIA RTX 5060":      dict(hbm_gbs=384.0,  bf16_tflops=14.98),
}


def hw_spec(dev:int=0) -> dict:                   # dev：GPU 设备编号，默认 0，第一张卡
    name=torch.cuda.get_device_name(dev)
    for k,v in HW.items():
        # k.split() 将键按空格拆分成列表
        if k.split()[1] in name and ("PCIe" in name) == ("PCIe" in k):
            return dict(v,name=name)
    return dict(hbm_gbs=float("nan"),bf16_tflops=float("nan"),name=name)


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------
#核心作用：自动生成类里面重复的 __init__（构造函数）和 __repr__（打印方法）等样板代码。
@dataclass
class BenchResult:
    name:str                                    # 算子名字，如 "rmsnorm/eager"
    ms_median:float                             # 中位数耗时（毫秒）
    ms_p10:float                                # 10% 分位点，看波动下限
    ms_p90:float                                # 90% 分位点，看波动上限
    bytes_moved:int=0                           # 理论访存量（手动填入，非实测）
    flops:int=0                                 # 理论计算量
    hbm_gbs:float=0.0                           # 实测等效带宽（带宽 = 字节 / 秒）
    pct_of_peak_bw:float=0.0                    # 核心指标：占 HBM 峰值百分之几
    tflops:float=0.0                            # 实测等效算力
    bound:str=""                                # 瓶颈类型："memory" 或 "compute" —— 由算术强度判定
    # field() : 告诉 @dataclass：这个字段不按普通赋值处理，要特殊配置
    # default_factory=dict : 配置内容是：每次创建新对象时，调用 dict() 函数生成一个全新的空字典作为默认值。
    # 不能直接写成 extra: dict = {}。其在类定义时只创建一次，所有实例的 extra 都指向同一个字典对象。
    extra:dict=field(default_factory=dict)      # 杂项，如算术强度、脊点值


def timeit(fn:Callable[[],Any],warmup:int=20,iters:int=100,flush_l2:bool=True,device:int=0)->tuple[float,float,float]:
    # Callable[[], Any] 是 Python 的类型注解（Type Hint）
    # 表示这里要传入一个“可以像函数一样被调用”的东西，这个东西不接受任何参数，并且会返回任意类型的值。([[],Any])

    """
    返回 (中位数, p10, p90) 毫秒。
    warmup      —— 首次调用含 cuBLAS/cuDNN 算法选择、Triton JIT、显存分配，
                    不预热测出来的是「编译时间」不是「算子时间」。
    synchronize —— CUDA 是异步的：不同步的话 CPU 早就跑完了，测到 0.01ms 的假数据。
    中位数而非均值 —— GPU 会被 ECC 刷新 / 其他进程 / 时钟波动打断，均值被长尾拖偏。
    flush_l2    —— Day2 新增。H100 有 50MB L2，小张量第二次跑全在 L2 里命中，
                    测出来的带宽能超过 HBM 峰值（物理上不可能）。每次迭代前刷掉 L2，
                    测的才是「冷数据」，也就是线上真实场景的数字。
                    （这正是 ncu 默认 --cache-control=all 在做的事）
    """
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备；本机无 GPU")

    l2_bytes=torch.cuda.get_device_properties().L2_cache_size
    # 在显存里开辟一块 “垃圾缓冲区”，用来冲刷 L2 缓存的。
    # torch.empty 只分配显存空间，但不初始化内容（里面的值是随机的脏数据）。分配速度极快，几乎不耗时。
    # 冲刷缓冲区的作用：第一次跑，数据从 HBM 搬到 L2，耗时正常。第二次跑，数据已经在 L2 缓存里了，GPU 直接从 L2 读，耗时很少。所以需要冲刷缓冲区。
    scratch=torch.empty(
        int(l2_bytes*1.5),
        dtype=torch.int8,
        device=f"cuda:{device}"
    )

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times=[]
    for _ in range(iters):
        if flush_l2:
            scratch.zero_()
        # 当 GPU 执行到start.record()，如果 enable_timing=True，则会读取 GPU 内部的一个高精度硬件计数器（时钟周期），把当时的数值存到这个 Event 对象里。
        # 如果 enable_timing=False（默认值），这个 Event 就只是个“路标”或“栅栏”，仅仅用来做同步，不会浪费时间去读硬件时钟。
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return(statistics.median(times),
           times[int(0.1*len(times))],
           times[int(0.9*len(times))-1])


# *: 表示之后的参数必须以关键字形式传递
def bench_op(name:str,fn:Callable[[],Any],*,bytes_moved:int=0,flops:int=0,**kwargs) -> BenchResult:
    """测一个算子，并直接换算成「离硬件极限多远」。"""
    med,p10,p90=timeit(fn,**kwargs)
    spec=hw_spec()
    gbs=bytes_moved/(med/1e3)/1e9 if bytes_moved else 0.0
    tfs=flops/(med/1e3)/1e12 if flops else 0.0
    ridge =spec["bf16_tflops"] * 1e12/(spec["hbm_gbs"]*1e9)   # 脊点：FLOP/Byte
    ai=flops/bytes_moved if bytes_moved and flops else 0.0
    return BenchResult(
        name=name,ms_median=med,ms_p10=p10,ms_p90=p90,
        bytes_moved=bytes_moved, flops=flops,
        hbm_gbs=gbs, pct_of_peak_bw=100.0 * gbs / spec["hbm_gbs"] if gbs else 0.0,
        tflops=tfs,
        bound=("memory" if ai < ridge else "compute") if ai else "",
        extra=dict(arithmetic_intensity=ai, ridge_point=ridge, gpu=spec["name"]),
    )



# ---------------------------------------------------------------------------
# 正确性：三尺子
# ---------------------------------------------------------------------------
def compare(out:torch.Tensor,ref:torch.Tensor,name:str,atol:float=1e-3,rtol:float=1e-3) -> dict:
    """三尺子验误差。为什么要三把：
      allclose 只给 True/False，FAIL 了不知道差多少；
      cosine   对整体「方向」敏感 —— 掉到 0.99 以下通常是算法写错而非精度问题；
      max_abs/max_rel 定位最坏点 —— fp16 累加误差通常 max_rel 大但 cosine≈1。
    """
    out32,ref32=out.float(),ref.float()
    diff=(out32-ref32).abs()
    cos=torch.nn.functional.cosine_similarity(
        out32.flatten(),ref32.flatten(),dim=0).item()
    rel=(diff/ref32.abs().clamp_min(1e-6)).max().item()         # 计算最大相对误差
    rec=dict(name=name,allclose=bool(torch.allclose(out32,ref32,atol=atol,rtol=rtol)),
             cosine=cos,max_abs=diff.max().item(),max_rel=rel)
    print(f"[{name}] allclose={rec['allclose']} cosine={cos:.6f} "
          f"max_abs={rec['max_abs']:.2e} max_rel={rel:.2e}")
    return rec



def report(results:list[BenchResult],path:str|None=None)->None:
    base=results[0].ms_median
    print(f"\n{'op':<28}{'ms':>9}{'p10-p90':>16}{'GB/s':>9}{'%peak':>8}{'speedup':>9}")
    for r in results:
        print(f"{r.name:<28}{r.ms_median:>9.4f}"
              f"{f'{r.ms_p10:.3f}-{r.ms_p90:.3f}':>16}"
              f"{r.hbm_gbs:>9.0f}{r.pct_of_peak_bw:>7.1f}%{base / r.ms_median:>8.2f}x")
    if path:                                    # 落盘：里程碑要「可复现」
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        print(f"\n-> saved {path}")


# ---------------------------------------------------------------------------
# 自测：用 RMSNorm 跑通全流程
# ---------------------------------------------------------------------------
if __name__=="__main__":
    torch.manual_seed(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    B,T,D=8,2048,4098

    x=torch.randn(B,T,D,device=device,dtype=torch.bfloat16)
    w=torch.randn(D,device=device,dtype=torch.bfloat16)

    def rmsnorm_torch(x,w,eps=1e-6):
        h=x.float()
        h=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+eps)
        return (h*w.float()).to(x.dtype)

    ref=rmsnorm_torch(x,w)
    compare(rmsnorm_torch(x,w),ref,"torch-eager")    # 自比：确认 harness 本身没问题

    # RMSNorm 的访存下界：读 x + 写 y（权重 D 个元素可忽略）
    nbytes=2*x.numel()*x.element_size()
    res=[bench_op("rmsnorm/eager",lambda :rmsnorm_torch(x,w),bytes_moved=nbytes)]

    compiled=torch.compile(rmsnorm_torch)
    compiled(x,w)                                               # 触发编译，别把编译时间算进去
    compare(compiled(x,w),ref,"torch-compile")
    res.append(bench_op("rmsnorm/compile",lambda :compiled(x,w),bytes_moved=nbytes))

    report(res,"logs/w1d2_rmsnorm_baseline.json")
    # 预期：eager 因为多次读写中间量，%peak 明显低于 compile（融合后接近纯访存下界）。
    # 这个 %peak 差距，就是 Day2 学的「访存才是瓶颈」在引擎层面的第一次现形。













