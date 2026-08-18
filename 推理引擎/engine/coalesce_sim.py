# 作用：模拟 GPU 硬件如何将 32 个线程的地址归类去重

WARP=32
SECTOR=32       # 硬件最小传输单元
LINE=128        # Cache Line 大小
ESIZE=4         # float 占 4 字节

def analyze(name,map_func,base=0,bytes_per_thread=ESIZE):
    sectors=set()
    lines=set()

    # 模拟一个 Warp 里的 32 个线程（lane）
    for lane in range(WARP):
        thread_id=base+lane
        element_idx=map_func(thread_id)         # 线程号 -> 取第几个元素
        byte_addr=element_idx*ESIZE             # 转成字节地址
        sectors.add(byte_addr//SECTOR)
        lines.add(byte_addr//LINE)

    moved=len(sectors)*SECTOR
    useful=WARP*bytes_per_thread       # 32 个线程，每个取 4 字节 = 128B 有用数据

    print(f"{name:<24} sectors={len(sectors):2d} lines={len(lines):2d}  "
          f"搬运 {moved:4d} B / 有用 {useful:3d} B  -> 放大 {moved / useful:.1f}x")


if __name__=="__main__":
    n=1<<24
    stride=32
    chunk=n//stride

    # 以下每个 lambda 表达式的入参 i 就是全局线程号
    analyze("coalesced",lambda i:i)
    analyze("offset+1",lambda i:i+1)
    analyze("stride 2",lambda i:i*2)
    analyze("stride 8",lambda i:i*8)
    analyze("stride 32(naive)",lambda i:i*32)
    analyze("transposed read",lambda i:(i%stride)*chunk+i//stride)

    # float4：每线程搬 16B（等价于取下标 i*4，因为 4 个 float 连续排放）
    analyze("float4 (16B/thr)",lambda i:i*4,bytes_per_thread=16)


