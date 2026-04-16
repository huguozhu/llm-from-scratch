# ==============================================================================
# Flash Attention 性能基准测试
# ==============================================================================
# 功能概述：
#   对比 Triton Flash Attention 与标准注意力实现的推理延迟。
#   使用 triton.testing.do_bench 进行精确的 GPU 计时。
#
# 测试矩阵：
#   - 数据类型：float32, bfloat16
#   - 模型维度：16, 32, 64, 128
#   - 序列长度：256, 1024, 4096
#   - 批次大小：1, 64
#
# 每种配置同时测试 Flash Attention 和标准注意力，便于直接对比加速比。
# ==============================================================================
from kernel import FlashAttention
import torch
from llm.transformer import ScaledDotProductAttention
import triton


def bench_mark_flash_attention():
    for dtype in [torch.float32, torch.bfloat16]:
        for d_model in [16, 32, 64, 128]:
            for seq_len in [256, 1024, 4096]:
                for batch_size in [1, 64]:
                    q = torch.randn(
                        (batch_size, seq_len, d_model), dtype=dtype, device="cuda"
                    )
                    k = torch.randn(
                        (batch_size, seq_len, d_model), dtype=dtype, device="cuda"
                    )
                    v = torch.randn(
                        (batch_size, seq_len, d_model), dtype=dtype, device="cuda"
                    )

                    def fn():
                        return bench_mark_flash_attention_each(q, k, v)

                    val = triton.testing.do_bench(fn)
                    print(
                        f"Flash_attention dtype: {str(dtype):10}, batch_size: {batch_size:6}, d_model: {d_model:6}, seq_len: {seq_len:6}: mean latency {val:.3f} ms"
                    )

                    def fn2():
                        return benchmark_attention_each(q, k, v)

                    val = triton.testing.do_bench(fn2)
                    print(
                        f"Original_attention dtype: {str(dtype):10}, batch_size: {batch_size:6}, d_model: {d_model:6}, seq_len: {seq_len:6}: mean latency {val:.3f} ms"
                    )


def bench_mark_flash_attention_each(
    q,
    k,
    v,
):
    FlashAttention.apply(q, k, v, True)


def benchmark_attention_each(q, k, v, device="cuda"):
    atten = ScaledDotProductAttention().to(device)
    atten(q, k, v)


if __name__ == "__main__":
    bench_mark_flash_attention()
