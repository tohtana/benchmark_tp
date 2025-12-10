"""Distributed training initialization utilities."""

import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed(use_deepspeed: bool = True) -> Tuple[int, int, torch.device]:
    """
    Initialize torch.distributed for multi-GPU training.

    Automatically handles:
    - LOCAL_RANK environment variable from torchrun
    - Process group initialization with NCCL backend
    - DeepSpeed initialization when use_deepspeed=True
    - CUDA device assignment

    Args:
        use_deepspeed: Use DeepSpeed's init_distributed (required for AutoTP)

    Returns:
        Tuple of (rank, world_size, device)
    """
    # Get local rank from environment (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Initialize process group if not already done
    if world_size > 1 and not dist.is_initialized():
        if use_deepspeed:
            # DeepSpeed requires its own initialization for AutoTP
            import deepspeed
            deepspeed.init_distributed()
        else:
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all ranks.

    Args:
        tensor: Tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor
