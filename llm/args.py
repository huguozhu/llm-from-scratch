# ==============================================================================
# 命令行参数解析模块
# ==============================================================================
# 功能概述：
#   定义并管理 Transformer 模型训练和推理所需的全部命令行参数。
#   通过 argparse 提供统一的参数入口，涵盖以下几大类：
#
#   1. 模型超参数：d_model(模型维度), num_heads(注意力头数), d_ff(FFN隐藏层维度),
#      vocab_size(词表大小), num_layers(Transformer层数), max_seq_len(最大序列长度)
#
#   2. 优化器超参数：lr(学习率), lr_min/lr_max(余弦退火学习率范围),
#      beta1/beta2(AdamW动量参数), weight_decay(权重衰减)
#
#   3. 训练超参数：batch_size(批次大小), iterations(训练迭代次数),
#      warmup_iters(学习率预热步数), cos_cycle_iters(余弦周期步数),
#      device(训练设备), 各种文件路径和日志间隔
#
#   4. 分布式训练：world_size(进程数), backend(通信后端, 默认 nccl)
#
#   5. 推理超参数：temperature(温度采样), top_p(核采样阈值)
#
# 使用方式：
#   parser = get_parser()
#   args = parser.parse_args()
# ==============================================================================
import argparse


def get_parser():
    """
    创建并返回命令行参数解析器。
    包含模型架构、优化器、训练流程、数据路径、推理等全部可配置参数。
    """
    parser = argparse.ArgumentParser(description="Train a Transformer model.")

    # ---------- 模型超参数 ----------
    # d_model: 模型的隐藏层维度，决定了 Embedding、Attention、FFN 各层的宽度
    parser.add_argument("--d_model", type=int, default=288, help="Model dimension")
    # num_heads: 多头注意力的头数，d_model 必须能被 num_heads 整除
    parser.add_argument(
        "--num_heads", type=int, default=6, help="Number of attention heads"
    )
    # d_ff: SwiGLU 前馈网络的中间隐藏层维度
    parser.add_argument("--d_ff", type=int, default=1024, help="Feed-forward dimension")
    # vocab_size: BPE 分词器的词表大小（含特殊 token）
    parser.add_argument("--vocab_size", type=int, default=2048, help="Vocabulary size")
    # num_layers: Transformer 解码器堆叠的层数
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    # max_seq_len: 模型支持的最大序列长度，影响 RoPE 位置编码和因果掩码的预计算
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )

    # ---------- 优化器超参数 ----------
    # lr: AdamW 优化器的基础学习率
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    # lr_min: 余弦退火调度器的最小学习率（退火终值）
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-5,
        help="Min learning rate in cos lr scheduler",
    )
    # lr_max: 余弦退火调度器的最大学习率（预热目标值）
    parser.add_argument(
        "--lr_max",
        type=float,
        default=1e-4,
        help="Max learning rate in cos lr scheduler",
    )
    # beta1, beta2: AdamW 的一阶和二阶动量衰减系数
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    # weight_decay: AdamW 解耦权重衰减系数（L2 正则化）
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="AdamW weight decay"
    )

    # Loss Hyperparameters
    # parser.add_argument(
    #     "--eot_weight", type=float, default=0.1, help="Weight for the end-of-text token in the loss function"
    # )

    # ---------- 训练超参数 ----------
    # batch_size: 全局批次大小，分布式训练时会按 world_size 均分为 mini-batch
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--world_size", type=int, default=1, help="World size")
    parser.add_argument(
        "--backend", type=str, default="nccl", help="Backend for distributed training"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument(
        "--iterations", type=int, default=20000, help="Number of training iterations"
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=5000, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--cos_cycle_iters",
        type=int,
        default=20000,
        help="Number of cos cycle iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to train on (e.g., 'cpu', 'cuda:0', 'mps')",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="data/training_data.npy",
        # required=True,
        help="Path to the training data file (numpy array)",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default="data/tokenizer",
        help="Path to the tokenizer checpoint path",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/validation_data.npy",
        # required=True,
        help="Path to the validation data file (numpy array)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--train_source_file",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="path to the train source file, used to generated token ids",
    )
    parser.add_argument(
        "--valid_source_file",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="path to the valid source file, used to generated token ids",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="log",
        help="Path to store log",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Interval for logging training loss",
    )
    parser.add_argument(
        "--val_interval", type=int, default=500, help="Interval for running validation"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Interval for saving checkpoints",
    )

    # ---------- 推理超参数 ----------
    # temperature: 温度系数，控制输出概率分布的平滑度。越小越确定性，越大越随机
    parser.add_argument("--temperature", type=float, default=0.8, help="temperature")
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Inferencing top_p kernel search"
    )
    return parser
