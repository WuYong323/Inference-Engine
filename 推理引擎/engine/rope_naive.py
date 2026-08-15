import numpy as np


def rope_naive(x:np.ndarray,pos:int,base:float=10000.0)->np.ndarray:
     d=x.shape[0]
     out=np.empty_like(x)
     for i in range(d//2):
         theta_i=base**(-2.0*i/d)
         angle=pos*theta_i
         cos,sin=np.cos(angle),np.sin(angle)
         x1,x2=x[2*i],x[2*i+1]
         out[2*i]=x1*cos-x2*sin
         out[2*i+1]=x1*sin+x2*cos
     return out



if __name__=="__main__":
    rng=np.random.default_rng(0)
    d=8
    q,k=rng.standard_normal(d),rng.standard_normal(d)

    def rope_dot(m,n):
        return rope_naive(q,m)@rope_naive(k,n)

    print("m=5,n=3 (差2):", round(rope_dot(5, 3), 6))
    print("m=12,n=10(差2):", round(rope_dot(12, 10), 6))   # 应与上一行几乎相等
    print("m=3,n=3 (差0):", round(rope_dot(3, 3), 6))
    print("未旋转 q·k     :", round(q @ k, 6))  # 应与"差0"那行相等
    # 你会看到：只要 m−n 相同，点积就相同；m−n=0 时等于未旋转的原始点积。
    # 这就是 2.2 推导的实验证据 —— 底层没有魔法，只有旋转。



















