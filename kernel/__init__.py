# ==============================================================================
# kernel 包初始化模块
# ==============================================================================
# 功能概述：
#   自定义 GPU 算子（Kernel）包，提供两种 Flash Attention 实现：
#   - FlashAttentionMock  : 纯 PyTorch 实现的 Flash Attention（用于测试和对比验证）
#                           支持朴素（naive）和分块（tiled）两种前向模式
#   - FlashAttention      : 基于 Triton 的高性能 Flash Attention GPU 内核
#                           使用 Triton JIT 编译，直接操作 GPU 线程块
# ==============================================================================
from .flash_attention_mock import *
from .flash_attention_triton import *
