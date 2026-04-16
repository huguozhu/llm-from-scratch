# ==============================================================================
# JIT 编译版注意力性能基准测试
# ==============================================================================
# 功能概述：
#   调用 bench_mark_atten.benchmark_attention(jit=True) 测试
#   torch.compile 编译优化后的注意力性能。
#   torch.compile 通过算子融合和内存优化通常能带来 1.5-3x 的加速。
# ==============================================================================
from .bench_mark_atten import benchmark_attention

if __name__ == "__main__":
    benchmark_attention(jit=True)