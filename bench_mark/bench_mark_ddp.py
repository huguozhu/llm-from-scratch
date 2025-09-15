import torch
from torch import multiprocessing as mp
from torch import distributed as dist
import os
import timeit

K = 1024
M = 1024 * 1024
G = 1024 * 1024 * 1024


def benchmark_all_reduce():
    os.environ["NCCL_DEBUG"] = "NONE"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # for device in ["cpu", "cuda"]:
    for device in ["cuda"]:
        # for backend in ["gloo", "nccl"]:
        for backend in ["nccl"]:
            if device == "cpu" and backend == "nccl":
                continue
            for dtype in [torch.float32, torch.bfloat16]:
                for num_process in [2, 4, 6]:
                    for data_size in [1 * K, 1 * M, 1 * G]:
                        start = timeit.default_timer()
                        try:
                            benchmark_all_reduce_each(
                                dtype, device, backend, num_process, data_size
                            )
                        except Exception as e:
                            print(
                                f"Error running {device}, {backend}, {num_process}, {data_size}: {e}"
                            )
                        end = timeit.default_timer()
                        print(
                            f"device: {str(device):10}, dtype: {str(dtype):10}, backend: {backend:6}, num_process: {num_process:6}, "
                            f"data_size: {data_size:8}: , total time: {(end - start) * 1000:.3f} ms"
                        )


def setup(backend, rank, world_size):
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{rank}" if backend == "nccl" else None),
    )


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def worker(rank, world_size, device, dtype, backend, data_size):
    if device == "cuda":
        torch.cuda.set_device(rank)

    setup(backend, rank, world_size)

    try:
        data = torch.randn(data_size, dtype=dtype).to(device)

        dist.barrier()

        dist.all_reduce(data, op=dist.ReduceOp.SUM)

        if device == "cuda":
            torch.cuda.synchronize()

    except Exception as e:
        print(f"Rank {rank} encountered error: {e}")
        raise
    finally:
        cleanup()


def benchmark_all_reduce_each(dtype, device, backend, num_process: int, data_size: int):
    world_size = num_process
    mp.spawn(
        fn=worker,
        args=(world_size, device, dtype, backend, data_size),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    benchmark_all_reduce()
