# ==============================================================================
# 模型检查点（Checkpoint）保存与加载模块
# ==============================================================================
# 功能概述：
#   提供训练过程中模型状态的持久化能力，支持断点续训（resume training）。
#   检查点文件中保存三项内容：
#     1. model.state_dict()     - 模型所有可学习参数的权重
#     2. optimizer.state_dict() - 优化器状态（含动量缓冲 m, v 以及步数 t 等）
#     3. iteration              - 当前训练迭代次数，用于恢复训练进度
#
#   文件格式：使用 torch.save/torch.load (基于 pickle) 序列化为 .pt 文件
#
# 包含函数：
#   - save_checkpoint : 将模型、优化器状态和迭代次数保存到文件
#   - load_checkpoint : 从文件恢复模型和优化器状态，返回已完成的迭代次数
# ==============================================================================
import torch
import os
import typing


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    保存训练检查点到指定路径。

    参数：
        model     : 要保存的 Transformer 模型
        optimizer : 对应的优化器（AdamW），保存其内部状态以支持续训
        iteration : 当前训练迭代步数
        out       : 输出文件路径或文件对象（通常为 .pt 文件）

    保存内容：
        - "model"     : 模型参数字典 (state_dict)
        - "optimizer" : 优化器状态字典（包含每个参数的一阶动量 m、二阶动量 v、步数 t）
        - "iteration" : 训练步数，用于恢复时确定学习率调度位置
    """
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    source: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """
    从检查点文件恢复模型和优化器状态。

    参数：
        source    : 检查点文件路径或文件对象
        model     : 需要加载权重的模型（必须与保存时结构一致）
        optimizer : 可选的优化器，若提供则同时恢复优化器状态

    返回：
        iteration : 检查点保存时的迭代步数，用于接续训练
    """
    obj = torch.load(source)
    model.load_state_dict(obj["model"])
    if optimizer and "optimizer" in obj:
        optimizer.load_state_dict(obj["optimizer"])
    iteration = obj["iteration"]
    return iteration
