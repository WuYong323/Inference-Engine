import numpy as np


def online_softmax_attention(Q,K,V,block_size=64):
    N,d=Q.shape
    O=np.zeros((N,d))
    for i in range(N):
        m=-np.inf
        l=0.0
        acc=np.zeros(d)
        for j in range(0,N,block_size):
            k_blk=K[j:j+block_size]
            v_blk=V[j:j+block_size]
            s=Q[i]@k_blk.T
            m_new=max(m,s.max())
            correction=np.exp(m-m_new)
            p=np.exp(s-m_new)
            l=l*correction+p.sum()
            acc=acc*correction+p@v_blk
            m=m_new
        O[i]=acc/l
    return O


def standard_attention(Q,K,V):
    S=Q@K.T
    P=np.exp(S-S.max(axis=1,keepdims=True))
    P=P/P.sum(axis=1,keepdims=True)
    return P@V

if __name__=="__main__":
    np.random.seed(0)
    Q,K,V=np.random.randn(128,32),np.random.randn(128,32),np.random.randn(128,32)
    assert np.allclose(online_softmax_attention(Q,K,V),standard_attention(Q,K,V),atol=1e-6)
    print("online softmax 分块结果 与 标准 softmax 完全一致")




