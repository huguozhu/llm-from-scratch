# ==============================================================================
# 训练模块
# ==============================================================================
# 功能概述：
#   实现 Transformer 语言模型的完整训练流程，支持单机单卡和多卡分布式训练。
#
# 训练流程：
#   1. 数据加载：使用 numpy mmap 模式加载预处理好的 token ID 数组（节省内存）
#   2. 模型初始化：创建 Transformer 模型，多卡时用自定义 DDP 包装
#   3. 优化器：单卡用 AdamW，多卡用 ShardedOptimizer（分片优化器，降低显存）
#   4. 训练循环：
#      a. 随机采样 batch：从 token 数组中随机截取连续片段
#      b. 前向传播：模型输出 logits
#      c. 计算损失：交叉熵损失 (CrossEntropyLoss)
#      d. 反向传播 + 梯度裁剪（max_norm=1.0）
#      e. 多卡模式下同步梯度 (finish_gradient_sync)
#      f. 优化器更新参数
#      g. 余弦退火学习率调度
#   5. 验证：每隔 val_interval 步在验证集上计算平均损失
#   6. 检查点：每隔 checkpoint_interval 步保存模型和优化器状态
#   7. 日志：TensorBoard 记录训练/验证损失和学习率曲线
#
# 分布式训练：
#   - 使用 torch.multiprocessing.spawn 启动多进程
#   - 通信后端：NCCL（GPU 间高速通信）
#   - DDP：自定义的数据并行（非 PyTorch 原生），每个进程处理 batch_size/world_size 的数据
#   - ShardedOptimizer：分片优化器，每个进程只维护一部分参数的优化器状态
#
# 数据采样策略：
#   get_batch() 从整个 token 数组中随机采样 batch_size 个起始位置，
#   每个样本是长度为 max_seq_len 的连续 token 片段。
#   输入 = tokens[i : i+seq_len]，目标 = tokens[i+1 : i+1+seq_len]（错位一个 token）
# ==============================================================================
import torch
import numpy as np
from llm.args import get_parser
from llm.checkpoint import save_checkpoint
from llm.transformer import (
    CrossEntropyLoss,
    Transformer,
    AdamW,
    gradient_clip,
    cos_lr_scheduler,
)
import os
from torch.utils.tensorboard.writer import SummaryWriter
from torch import distributed as dist
from torch import multiprocessing as mp
from parallel import DDP, ShardedOptimizer
import random


def set_random_seed(seed: int, rank: int):
    """
    设置全局随机种子以确保可重复性。
    每个进程使用 seed + rank 作为种子，保证不同进程有不同的随机序列，
    同时整体实验可通过设定相同的 global seed 完全复现。
    同时禁用 cuDNN 的非确定性优化（benchmark=False, deterministic=True）。
    """
    global_seed = seed
    seed_to_use = global_seed + rank

    torch.manual_seed(seed_to_use)
    torch.cuda.manual_seed_all(seed_to_use)
    np.random.seed(seed_to_use)
    random.seed(seed_to_use)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_batch(
    x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input and target sequences from the tokenized data.

    Args:
        x: A numpy array of token IDs.
        batch_size: The number of sequences in a batch.
        context_length: The length of each sequence.
        device: The PyTorch device to place the tensors on (e.g., 'cpu', 'cuda:0').

    Returns:
        A tuple containing the input and target sequences as PyTorch tensors.
    """
    # Generate random starting indices for the batches
    ix = torch.randint(0, len(x) - context_length, (batch_size,))

    # Create the input and target sequences
    input_seqs = torch.stack(
        [torch.from_numpy(x[i : i + context_length].astype(np.int64)) for i in ix]
    )
    target_seqs = torch.stack(
        [
            torch.from_numpy(x[i + 1 : i + 1 + context_length].astype(np.int64))
            for i in ix
        ]
    )

    # Move the tensors to the specified device
    return input_seqs.to(device), target_seqs.to(device)


def _setup_process_group(rank, world_size, backend):
    """
    初始化分布式训练进程组。
    设置 NCCL 通信后端，分配每个进程到对应的 GPU 设备。
    使用 localhost 单机多卡模式（MASTER_ADDR=localhost, MASTER_PORT=12390）。
    """
    os.environ["NCCL_DEBUG"] = "NONE"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _cleanup_process_group():
    """清理分布式进程组：先同步所有进程（barrier），再销毁进程组。"""
    dist.barrier()
    dist.destroy_process_group()


def _train(rank, world_size, backend, args):
    """
    单个进程的训练主函数（被 mp.spawn 调用）。

    参数：
        rank       : 当前进程的全局序号（0 ~ world_size-1）
        world_size : 总进程数
        backend    : 分布式通信后端（"nccl" 或 "gloo"）
        args       : 命令行参数对象

    训练循环核心步骤（每个 iteration）：
        1. 从训练数据中随机采样一个 mini-batch
        2. 前向传播得到 logits
        3. 计算交叉熵损失
        4. 反向传播计算梯度
        5. 全局梯度裁剪（max_norm=1.0）
        6. 多卡模式下同步梯度（AllReduce）
        7. 优化器更新参数
        8. 余弦退火调整学习率
    """
    device = _setup_process_group(rank, world_size, backend)
    set_random_seed(args.seed, rank)
    try:
        print(f"rank: {rank}, device: {device}")

        writer = SummaryWriter(log_dir=args.log_dir)

        mini_batch_size = args.batch_size // world_size

        # Data Loading
        if rank == 0:
            print("Loading data...")
        # train_data = np.load(args.train_data)
        # val_data = np.load(args.val_data)
        train_data = np.load(args.train_data, mmap_mode="r")
        val_data = np.load(args.val_data, mmap_mode="r")

        # Model Initialization
        if rank == 0:
            print("Initializing model...")
        model = Transformer(
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            device=device,
        ).to(device)

        if args.world_size > 1:
            model = DDP(model)

        # Optimizer Initialization
        if args.world_size > 1:
            optimizer = ShardedOptimizer(
                model.parameters(),
                AdamW,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )

        criterion = CrossEntropyLoss()
        # criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        # Training Loop
        if rank == 0:
            print("Starting training...")
        for i in range(args.iterations + 1):
            # Validation
            if i % args.val_interval == 0 and rank == 0:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for _ in range(100):  # 100 batches for validation
                        val_inputs, val_targets = get_batch(
                            val_data, mini_batch_size, args.max_seq_len, device
                        )
                        val_logits = model(val_inputs)
                        val_loss += criterion(val_logits, val_targets).item()
                val_loss /= 100
                print(f"Iteration {i}, Validation Loss: {val_loss:.4f}")
                model.train()
                writer.add_scalar("val_loss", val_loss, i)

            # Get a batch of training data
            inputs, targets = get_batch(
                train_data, mini_batch_size, args.max_seq_len, device
            )

            # Forward pass
            logits = model(inputs)
            loss = criterion(logits, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            gradient_clip(model.parameters(), max_norm=1.0)
            if args.world_size > 1:
                model.finish_gradient_sync()
            optimizer.step()

            # Learning rate decay
            lr = cos_lr_scheduler(
                it=i,
                warmup_iters=args.warmup_iters,
                cos_cycle_iters=args.cos_cycle_iters,
                lr_min=args.lr_min,
                lr_max=args.lr_max,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if rank == 0:
                writer.add_scalar("loss_train", loss.item(), i)
                writer.add_scalar("lr", lr, i)
                # Logging
                if i % args.log_interval == 0:
                    print(
                        f"Iteration {i}, Training Loss: {loss.item():.4f}, LR: {lr:.6f}"
                    )

                # Checkpointing
                if i % args.checkpoint_interval == 0 and i > 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        i,
                        os.path.join(args.checkpoint_path, f"chpt_{i}.pt"),
                    )
                    print(f"Saved checkpoint at iteration {i}")
    finally:
        _cleanup_process_group()


def train():
    """
    训练入口函数。
    解析命令行参数，验证 batch_size 能被 world_size 整除，
    创建必要的目录，然后使用 mp.spawn 启动多进程训练。
    """
    parser = get_parser()
    args = parser.parse_args()

    assert args.batch_size % args.world_size == 0, (
        "Batch size must be divisible by world size"
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    mp.spawn(
        _train,
        args=(args.world_size, args.backend, args),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    train()
