# 演示 PagedAttention 的显存管理骨架
from numpy.ma.core import append


class BlockAllocator:
    """管理物理块的分配与回收 —— 对应操作系统的物理内存管理器"""
    def __init__(self,num_blocks,block_size=16):
        self.block_size=block_size
        self.free_blocks=list(range(num_blocks))  # 空闲物理块编号池

    def allocate(self):
        if not self.free_blocks:
            raise MemoryError("显存块用尽 —— 真实引擎此时会触发抢占/换出")
        return self.free_blocks.pop()               # 取一个空闲物理块

    def free(self,block_id):
        self.free_blocks.append(block_id)           # 请求结束, 归还物理块给池


class Sequence:
    """一个请求的 KV Cache 视图 —— 只维护 block table, 不要求物理连续"""
    def __init__(self,allocator):
        self.allocator=allocator
        self.block_table=[]       # 核心: 逻辑块索引 -> 物理块编号 的映射表
        self.length=0

    def append_token(self):
        # 当前最后一个块满了(或还没有块), 才申请新物理块 —— 这就是"按需分配"
        if self.length%self.allocator.block_size==0:
            phys=self.allocator.allocate()
            self.block_table.append(phys)
        self.length+=1

    def locate(self,token_idx):
        """把逻辑 token 位置 翻译成 (物理块, 块内偏移) —— 这就是页表查询"""
        logical_block=token_idx//self.allocator.block_size
        offset=token_idx%self.allocator.block_size
        phys_block=self.block_table[logical_block]  # 查表: 逻辑->物理
        return phys_block,offset


if __name__=="__main__":
    alloc=BlockAllocator(num_blocks=8,block_size=16)
    a,b=Sequence(alloc),Sequence(alloc)
    for _ in range(20): a.append_token()    # A 生成 20 个 token -> 2 个块
    for _ in range(20): b.append_token()    # B 也生成 20 个 -> 2 个块
    print("请求 A 的物理块:", a.block_table)  # 例如 [7, 6] —— 和 B 的块交错
    print("请求B 的物理块:", b.block_table)  # 例如 [5, 4]
    print("A 的第 17 个 token 在:", a.locate(17))  # (物理块, 偏移) —— 逻辑第2块的第1个



