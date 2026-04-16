# ==============================================================================
# 分片优化器（Sharded Optimizer）- ZeRO Stage 1 风格
# ==============================================================================
# 功能概述：
#   实现类似 DeepSpeed ZeRO-1 的优化器状态分片，将优化器的内存开销分摊到多个 GPU 上。
#
# 核心思想：
#   在标准 DDP 中，每个 GPU 都持有完整的优化器状态（m, v 动量等），
#   这在模型较大时非常浪费显存。ShardedOptimizer 的做法是：
#     1. 将所有参数按索引轮询分配给各进程：参数 i 归 rank = i % world_size 所有
#     2. 每个进程只为自己负责的参数创建优化器状态（m, v 等）
#     3. step() 时的流程：
#        a. AllReduce 同步所有参数的梯度（所有进程梯度一致）
#        b. 每个进程只用内部优化器更新自己负责的那部分参数
#        c. Broadcast 将更新后的参数广播给所有进程（保持参数一致）
#
# 显存节省：
#   - 标准 DDP：每个 GPU 的优化器状态 = 全部参数量 × 状态大小（AdamW 约 2x）
#   - ShardedOptimizer：每个 GPU 的优化器状态 = 全部参数量 / world_size × 状态大小
#   - 例如 4 卡训练，优化器显存占用降低约 75%
#
# 使用方式：
#   optimizer = ShardedOptimizer(model.parameters(), AdamW, lr=1e-4, ...)
#   optimizer.zero_grad()
#   loss.backward()
#   optimizer.step()  # 内部自动完成 AllReduce -> 局部更新 -> Broadcast
# ==============================================================================
import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Any, Type, Dict, Optional, Callable, Iterable


class ShardedOptimizer(Optimizer):
    """
    分片优化器：每个进程只为 1/world_size 的参数维护优化器状态。

    参数：
        params        : 模型参数迭代器
        optimizer_cls : 内部使用的优化器类（如 AdamW）
        **kwargs      : 传递给内部优化器的参数（lr, betas, weight_decay 等）
    """
    def __init__(
        self, params: Iterable[Dict], optimizer_cls: Type[Optimizer], **kwargs: Any
    ):
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self._optimizer_cls = optimizer_cls
        self._optimizer_kwargs = kwargs

        self.optimizer: Optimizer

        super().__init__(params, kwargs)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        添加参数组时，按 round-robin 分片：参数 i 分配给 rank = i % world_size。
        只有归当前 rank 所有的参数会传给内部优化器，其余参数不创建状态。
        """
        full_params = list(param_group["params"])

        # 轮询分片：只取属于当前 rank 的参数
        sharded_params = []
        for i, param in enumerate(full_params):
            if i % self.world_size == self.rank:
                sharded_params.append(param)

        sharded_param_group = {k: v for k, v in param_group.items() if k != "params"}
        sharded_param_group["params"] = sharded_params

        # 延迟创建内部优化器（首次调用时创建）
        if not hasattr(self, "optimizer"):
            self.optimizer = self._optimizer_cls(
                [sharded_param_group], **self._optimizer_kwargs
            )
        else:
            self.optimizer.add_param_group(sharded_param_group)

        # 同时在父类中保留完整参数组（用于 zero_grad 等操作）
        super().add_param_group(param_group)

    def _average_gradients(self) -> None:
        """AllReduce 同步所有进程的梯度，保证每个进程看到相同的平均梯度。"""
        if self.world_size == 1:
            return

        backend = dist.get_backend()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if backend == "nccl":
                        # NCCL 直接支持 AVG 操作
                        dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)
                    else:
                        # GLOO 先 SUM 再手动除
                        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                        p.grad.data /= self.world_size

    def _synchronize_parameters(self) -> None:
        """
        Broadcast 同步参数：每个参数由其所属 rank 广播给所有进程。
        参数 i 的 owner_rank = i % world_size。
        """
        if self.world_size == 1:
            return

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                owner_rank = i % self.world_size
                dist.broadcast(p.data, src=owner_rank)

    @torch.no_grad()
    def step(
        self, closure: Optional[Callable] = None, **kwargs: Any
    ) -> Optional[float]:
        """
        优化器更新步骤：
        1. AllReduce 同步梯度
        2. 内部优化器只更新当前 rank 负责的参数
        3. Broadcast 将更新后的参数广播给所有进程
        """
        self._average_gradients()
        loss = self.optimizer.step(closure, **kwargs)
        self._synchronize_parameters()
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        """清零所有参数的梯度（不仅是分片部分，而是全部参数）。"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        """返回内部优化器的状态字典（仅包含当前 rank 负责的参数状态）。"""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """加载内部优化器的状态字典。"""
        self.optimizer.load_state_dict(state_dict)
