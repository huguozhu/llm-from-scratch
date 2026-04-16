# ==============================================================================
# llm 包初始化模块
# ==============================================================================
# 功能概述：
#   作为 llm 包的入口文件，负责将各子模块的公开接口统一导出，使得外部代码
#   可以直接通过 `from llm import Transformer, BpeTokenizer` 等方式使用。
#
# 导出的子模块：
#   - transformer : Transformer 模型及所有基础组件（Linear, RoPE, Attention 等）
#                   以及优化器（AdamW, SGDDecay）、损失函数、学习率调度器
#   - bpe_tokenizer: BPE 分词器，支持训练、编码、解码
#   - checkpoint   : 模型检查点的保存与加载
#   - training     : 训练主循环，含分布式训练支持
# ==============================================================================
from .transformer import *
from .bpe_tokenizer import *
from .checkpoint import *
from .training import *
