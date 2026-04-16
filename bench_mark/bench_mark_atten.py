# ==============================================================================
# 注意力机制性能基准测试
# ==============================================================================
# 功能概述：
#   测量自定义 ScaledDotProductAttention 在不同配置下的推理延迟。
#   遍历多种数据类型（float32, bfloat16）、模型维度（16~128）、
#   序列长度（256~4096）的组合，报告每次迭代的平均耗时。
#
# 测试配置：
#   - batch_size = 64
#   - warmup = 10 次迭代（不计时，预热 GPU）
#   - 计时 = 100 次迭代取平均
#   - 支持 torch.compile JIT 加速模式（jit=True）
# ==============================================================================
from llm.transformer import ScaledDotProductAttention
import torch
import timeit


def benchmark_attention(jit=False):
    for dt in [torch.float32, torch.bfloat16]:
        for d_model in [16, 32, 64, 128]:
            for seq_len in [256, 1024, 4096]:
                benchmark_attention_each(d_model, seq_len, dt, jit=jit)


def benchmark_attention_each(
    d_model: int, seq_len: int, dtype, batch_size=64, warmup_iters=10, steps=100, jit=False, device="cuda"
):
    atten = ScaledDotProductAttention().to(device)
    if jit:
        atten = torch.compile(atten)
    q = torch.randn((batch_size, seq_len, d_model), dtype=dtype, device="cuda")
    k = torch.randn((batch_size, seq_len, d_model), dtype=dtype, device="cuda")
    v = torch.randn((batch_size, seq_len, d_model), dtype=dtype, device="cuda")
    for _ in range(warmup_iters):
        _ = atten(q, k, v)
    torch.cuda.synchronize()

    start = timeit.default_timer()
    for _ in range(steps):
        with torch.no_grad():
            _ = atten(q, k, v)
    torch.cuda.synchronize()
    end = timeit.default_timer()

    print(
        f"dtype: {str(dtype):10}, d_model: {d_model:6}, seq_len: {seq_len:6}: "
        f"Time per iteration: {(end - start) * 1000 / steps:.3f} ms, total time: {(end - start) * 1000:.3f} ms"
    )


if __name__ == "__main__":
    benchmark_attention()