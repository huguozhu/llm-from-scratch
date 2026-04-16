# ==============================================================================
# 自定义分布式数据并行（DDP）模块
# ==============================================================================
# 功能概述：
#   从零实现的 DistributedDataParallel，核心功能是在多 GPU 训练时同步各进程的梯度。
#   与 PyTorch 原生 DDP 类似，但采用手动分桶（bucketing）+ 异步 AllReduce 策略。
#
# 工作原理：
#   1. 初始化时，将 rank=0 的模型参数广播（broadcast）到所有进程，确保起点一致
#   2. 为每个需要梯度的参数注册 post_accumulate_grad_hook 回调
#   3. 反向传播中，每当一个参数的梯度计算完成，回调自动触发：
#      a. 将梯度放入当前桶（bucket）
#      b. 当桶的大小达到 bucket_size（默认 128MB）时，启动异步 AllReduce
#   4. 训练步结束时调用 finish_gradient_sync()：
#      a. 将剩余未满的桶也发起 AllReduce
#      b. 等待所有异步操作完成（handle.wait()）
#      c. 将同步后的梯度写回各参数的 .grad 属性
#      d. GLOO 后端需额外除以 world_size（NCCL 直接用 AVG 操作）
#
# 分桶优化：
#   - 小梯度张量打包为大缓冲区后再做 AllReduce，减少通信次数
#   - 使用 _flatten_dense_tensors / _unflatten_dense_tensors 实现零拷贝打包
#   - 异步操作（async_op=True）使得通信与计算可以部分重叠
#
# 使用方式：
#   model = DDP(model)           # 包装模型
#   loss.backward()              # 反向传播（自动触发梯度同步）
#   model.finish_gradient_sync() # 等待所有梯度同步完成
#   optimizer.step()             # 更新参数
# ==============================================================================
import torch
from torch import distributed as dist

KB = 1024
MB = 1024 * KB


class DDP(torch.nn.Module):
    """
    自定义分布式数据并行封装。
    通过梯度分桶 + 异步 AllReduce 实现多 GPU 梯度同步。
    """
    def __init__(self, model: torch.nn.Module, bucket_size_mb: float = 128.0):
        """
        参数：
            model          : 要包装的模型
            bucket_size_mb : 梯度桶大小（MB），桶满时触发异步 AllReduce
        """
        super().__init__()
        self.module = model
        self.handles = []                          # 存储异步操作的 (handle, flat_grad, grad_list)
        self.bucket_size = bucket_size_mb * MB     # 桶大小（字节）
        self.grads_bucket = []                     # 当前桶中的梯度列表
        self.size = 0                              # 当前桶已累积的字节数

        # 将 rank=0 的参数广播到所有进程，确保初始权重一致
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                # 注册梯度累积完成后的回调，自动触发分桶同步
                param.register_post_accumulate_grad_hook(
                    lambda _, param=param: self._sync_gradients(param)
                )

    def _sync_gradients(self, p: torch.Tensor):
        """梯度回调：将梯度加入桶，桶满则启动异步 AllReduce。"""
        if p.grad is not None:
            self.size += p.grad.numel() * p.grad.element_size()
            self.grads_bucket.append(p.grad)
            if self.size >= self.bucket_size:
                self._sync_grads_in_buckets()

    def _sync_grads_in_buckets(self):
        """将当前桶内的梯度打包为连续内存，发起异步 AllReduce。"""
        if self.grads_bucket:
            # 将多个不连续的梯度张量打包为一个连续的大张量
            flatten_grads = torch._utils._flatten_dense_tensors(self.grads_bucket)
            if dist.get_backend() == dist.Backend.GLOO:
                # GLOO 不支持 AVG，先 SUM 后手动除 world_size
                handle = dist.all_reduce(
                    flatten_grads, op=dist.ReduceOp.SUM, async_op=True
                )
            else:
                # NCCL 支持直接求平均
                handle = dist.all_reduce(
                    flatten_grads, op=dist.ReduceOp.AVG, async_op=True
                )
            self.handles.append((handle, flatten_grads, list(self.grads_bucket)))
            self.size = 0
            self.grads_bucket.clear()

    def forward(self, *args, **kwargs):
        """前向传播直接委托给内部模型。"""
        return self.module(*args, **kwargs)

    def finish_gradient_sync(self):
        """
        完成所有梯度同步操作。在 optimizer.step() 之前调用。
        1. 将剩余不满的桶也发起 AllReduce
        2. 等待所有异步操作完成
        3. 将同步后的梯度解包并写回各参数的 .grad
        4. GLOO 后端额外除以 world_size
        """
        # 处理最后一个未满的桶
        self._sync_grads_in_buckets()
        # 等待所有异步 AllReduce 完成，并将结果写回
        for handle, flatten_grads, grads_bucket in self.handles:
            handle.wait()
            unflatten_grads = torch._utils._unflatten_dense_tensors(
                flatten_grads, grads_bucket
            )
            for grad, syncd_grad in zip(grads_bucket, unflatten_grads):
                grad.copy_(syncd_grad)
        # GLOO 后端需要手动求平均
        if dist.get_backend() == dist.Backend.GLOO:
            world_size = dist.get_world_size()
            for param in self.module.parameters():
                if param.grad is not None:
                    param.grad /= world_size
        self.handles.clear()


def ddp_on_after_backword(model: torch.nn.Module, opt: torch.optim.Optimizer):
    """简单的 DDP 辅助函数：反向传播后调用梯度同步。"""
    model.finish_gradient_sync()
