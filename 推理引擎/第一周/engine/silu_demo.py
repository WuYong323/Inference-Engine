# silu_demo.py —— 看清 SiLU 到底是什么，以及它和 ReLU 的区别

import numpy as np


def sigmoid(x): return 1.0/(1.0+np.exp(-x))

def silu(x): return x*sigmoid(x)

def relu(x):return np.maximum(0.0,x)

xs=np.array([-4,-2,-1,-0.5,0,0.5,1,2,4],dtype=float)

for x in xs:
    print(f"x={x:+.1f}     ReLU={relu(x):+.3f}      SiLU={silu(x):+.3f}")


