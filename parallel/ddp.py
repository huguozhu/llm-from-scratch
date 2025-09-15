import torch
from torch import distributed as dist

KB = 1024
MB = 1024 * KB


class DDP(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, bucket_size_mb: float = 128.0):
        super().__init__()
        self.module = model
        self.handles = []
        self.bucket_size = bucket_size_mb * MB
        self.grads_bucket = []
        self.size = 0
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    lambda _, param=param: self._sync_gradients(param)
                )

    def _sync_gradients(self, p: torch.Tensor):
        if p.grad is not None:
            self.size += p.grad.numel() * p.grad.element_size()
            self.grads_bucket.append(p.grad)
            if self.size >= self.bucket_size:
                self._sync_grads_in_buckets()

    def _sync_grads_in_buckets(self):
        if self.grads_bucket:
            flatten_grads = torch._utils._flatten_dense_tensors(self.grads_bucket)
            if dist.get_backend() == dist.Backend.GLOO:
                handle = dist.all_reduce(
                    flatten_grads, op=dist.ReduceOp.SUM, async_op=True
                )
            else:
                handle = dist.all_reduce(
                    flatten_grads, op=dist.ReduceOp.AVG, async_op=True
                )
            self.handles.append((handle, flatten_grads, list(self.grads_bucket)))
            self.size = 0
            self.grads_bucket.clear()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_sync(self):
        self._sync_grads_in_buckets()
        for handle, flatten_grads, grads_bucket in self.handles:
            handle.wait()
            unflatten_grads = torch._utils._unflatten_dense_tensors(
                flatten_grads, grads_bucket
            )
            for grad, syncd_grad in zip(grads_bucket, unflatten_grads):
                grad.copy_(syncd_grad)
        if dist.get_backend() == dist.Backend.GLOO:
            world_size = dist.get_world_size()
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad /= world_size
        self.handles.clear()


# naive ddp implementation
def ddp_on_after_backword(model: torch.nn.Module, opt: torch.optim.Optimizer):
    model.finish_gradient_sync()
