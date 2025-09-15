import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Any, Type, Dict, Optional, Callable, Iterable


class ShardedOptimizer(Optimizer):
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
        full_params = list(param_group["params"])

        sharded_params = []
        for i, param in enumerate(full_params):
            if i % self.world_size == self.rank:
                sharded_params.append(param)

        sharded_param_group = {k: v for k, v in param_group.items() if k != "params"}
        sharded_param_group["params"] = sharded_params

        if not hasattr(self, "optimizer"):
            self.optimizer = self._optimizer_cls(
                [sharded_param_group], **self._optimizer_kwargs
            )
        else:
            self.optimizer.add_param_group(sharded_param_group)

        super().add_param_group(param_group)

    def _average_gradients(self) -> None:
        if self.world_size == 1:
            return

        backend = dist.get_backend()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if backend == "nccl":
                        dist.all_reduce(p.grad.data, op=dist.ReduceOp.AVG)
                    else:
                        dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                        p.grad.data /= self.world_size

    def _synchronize_parameters(self) -> None:
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
        self._average_gradients()

        loss = self.optimizer.step(closure, **kwargs)

        self._synchronize_parameters()

        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
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
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
