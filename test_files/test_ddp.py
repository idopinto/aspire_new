import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    print(f"Rank {rank}, pid={os.getpid()} - before init_process_group")
    dist.init_process_group(
        backend="nccl",  # or "gloo" for debugging
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=3600)
    )
    print(f"Rank {rank}, pid={os.getpid()} - after init_process_group")
    torch.cuda.set_device(rank)
    assert torch.cuda.current_device() == rank, f"Rank {rank} is not correctly assigned to GPU {rank}"

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    setup(rank, world_size)
    model = torch.nn.Linear(10, 10).to(rank)
    print(f"Rank {rank} reached before DDP initialization")

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    print(f"Rank {rank} completed DDP initialization")
    for _ in range(5):
        inputs = torch.randn(20, 10).to(rank)
        outputs = model(inputs)
        print(f"Rank {rank}, outputs: {outputs.mean().item()}")

    cleanup()

if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(run, args=(world_size,), nprocs=world_size, join=True)