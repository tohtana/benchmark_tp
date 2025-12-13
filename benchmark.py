#!/usr/bin/env python
"""
TP Benchmarking Tool

Benchmarks tensor parallelism implementations:
- DeepSpeed AutoTP
- FSDP + DTensor (2D mesh)

Usage:
    # AutoTP with 4 GPUs
    torchrun --nproc_per_node=4 benchmark.py --impl autotp --tp_size 4

    # FSDP + DTensor with 8 GPUs (2 DP x 4 TP)
    torchrun --nproc_per_node=8 benchmark.py --impl fsdp_dtensor --dp_size 2 --tp_size 4
"""

import os
import time
from contextlib import nullcontext

import torch

from benchmark.args import get_args
from benchmark.data import create_dataloader
from benchmark.distributed import init_distributed, is_main_process
from benchmark.metrics import MetricsCollector
from benchmark.models import get_model_builder
from benchmark.parallel import create_tp_strategy


def training_loop(
    strategy,
    dataloader,
    num_training_steps: int,
    warmup_steps: int,
    log_interval: int,
    metrics_collector: MetricsCollector,
    gradient_accumulation_steps: int = 1,
    rank: int = 0,
    profiler=None,
):
    """
    Unified training loop for both TP implementations.

    Args:
        strategy: TP strategy (AutoTP or FSDP+DTensor)
        dataloader: DataLoader with training data
        num_training_steps: Total number of training steps
        warmup_steps: Number of warmup steps to skip in metrics
        log_interval: Logging interval in steps
        metrics_collector: Metrics collector instance
        gradient_accumulation_steps: Gradient accumulation steps
        rank: Process rank
        profiler: Optional PyTorch profiler instance
    """
    data_iterator = iter(dataloader)
    device = strategy.device

    for step in range(num_training_steps):
        step_start = time.perf_counter()
        is_warmup = step < warmup_steps

        # Get batch
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            batch = next(data_iterator)

        # Move to device
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        # Zero gradients
        strategy.zero_grad()

        accumulated_loss = 0.0
        forward_time_total = 0.0
        backward_time_total = 0.0

        # Gradient accumulation loop
        for micro_step in range(gradient_accumulation_steps):
            # Forward pass timing
            torch.cuda.synchronize()
            forward_start = time.perf_counter()

            loss = strategy.forward(batch)

            # Scale loss for gradient accumulation
            scaled_loss = loss / gradient_accumulation_steps

            torch.cuda.synchronize()
            forward_time = time.perf_counter() - forward_start
            forward_time_total += forward_time

            # Backward pass timing
            torch.cuda.synchronize()
            backward_start = time.perf_counter()

            strategy.backward(scaled_loss)

            torch.cuda.synchronize()
            backward_time = time.perf_counter() - backward_start
            backward_time_total += backward_time

            accumulated_loss += loss.item()

        # Optimizer step timing
        torch.cuda.synchronize()
        optimizer_start = time.perf_counter()

        strategy.optimizer_step()

        torch.cuda.synchronize()
        optimizer_time = time.perf_counter() - optimizer_start

        step_time = time.perf_counter() - step_start

        avg_loss = accumulated_loss / gradient_accumulation_steps

        # Collect metrics (skip warmup)
        if not is_warmup:
            metrics_collector.record_step(
                step=step,
                loss=avg_loss,
                step_time_s=step_time,
                forward_time_s=forward_time_total,
                backward_time_s=backward_time_total,
                optimizer_time_s=optimizer_time,
            )

        # Logging
        if rank == 0 and (step + 1) % log_interval == 0:
            status = "warmup" if is_warmup else "training"
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_peak = torch.cuda.max_memory_allocated() / 1e9
            print(
                f"[{status}] Step {step+1}/{num_training_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Step: {step_time*1000:.1f}ms | "
                f"Fwd: {forward_time_total*1000:.1f}ms | "
                f"Bwd: {backward_time_total*1000:.1f}ms | "
                f"Opt: {optimizer_time*1000:.1f}ms | "
                f"Mem: {mem_allocated:.1f}GB (peak: {mem_peak:.1f}GB)"
            )

        # Step the profiler
        if profiler is not None:
            profiler.step()

    return metrics_collector


def main():
    args = get_args()

    # Initialize distributed (use DeepSpeed for AutoTP, standard NCCL for FSDP+DTensor)
    use_deepspeed = args.impl == "autotp"
    rank, world_size, device = init_distributed(use_deepspeed=use_deepspeed)

    if rank == 0:
        print(f"=" * 60)
        print("TP BENCHMARKING")
        print(f"=" * 60)
        print(f"Implementation: {args.impl}")
        print(f"Model: {args.model_name}")
        print(f"World size: {world_size}")
        print(f"TP size: {args.tp_size or 'auto'}")
        print(f"DP size: {args.dp_size}")
        print(f"Vocab parallel: {args.use_vocab_parallel}")
        print(f"Batch size: {args.batch_size}")
        print(f"Sequence length: {args.seq_length}")
        print(f"Training steps: {args.num_training_steps}")
        print(f"Warmup steps: {args.warmup_steps}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"Data type: {args.dtype}")
        print(f"Autocast: {args.autocast}")
        print(f"Profiling: {args.profile}")
        print(f"=" * 60)

    # Set random seed for model initialization
    # All ranks must use the SAME seed so TP sharding gets consistent weight slices
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Get dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Get model builder
    model_builder = get_model_builder(args.model_name)
    model_config = model_builder.get_config()

    if rank == 0:
        print(f"\nModel configuration:")
        print(f"  Hidden size: {model_config.hidden_size}")
        print(f"  Attention heads: {model_config.num_attention_heads}")
        print(f"  KV heads: {model_config.num_key_value_heads}")
        print(f"  Layers: {model_config.num_hidden_layers}")
        print(f"  Vocab size: {model_config.vocab_size}")
        print()

    # Create TP strategy
    strategy = create_tp_strategy(
        impl=args.impl,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        use_vocab_parallel=args.use_vocab_parallel,
    )

    # Setup strategy (creates model and optimizer)
    config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "batch_size": args.batch_size,
        "zero_stage": args.zero_stage,
        "max_grad_norm": args.max_grad_norm,
        "activation_checkpointing": args.activation_checkpointing,
        "num_layers": args.num_layers,
        "attn_impl": args.attn_impl,
        "autocast": args.autocast,
    }

    if rank == 0:
        print(f"Setting up {strategy.strategy_name}...")

    strategy.setup(model_builder, device, dtype, config)

    if rank == 0:
        print(f"\nStrategy: {strategy.strategy_name}")
        print(f"Effective TP size: {strategy.tp_size}")
        print(f"Effective DP size: {strategy.dp_size}")

    # Create dataloader
    # Use DP rank for data seed so TP ranks in same DP group see same data,
    # while different DP groups see different data
    dp_rank = getattr(strategy, 'dp_rank', 0) if hasattr(strategy, 'dp_rank') else 0
    data_seed = args.seed + dp_rank
    dataloader = create_dataloader(
        vocab_size=model_config.vocab_size,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        seed=data_seed,
    )

    # Create metrics collector
    metrics_collector = MetricsCollector(args.seq_length, args.batch_size)
    metrics_collector.set_config({
        "model_name": args.model_name,
        "impl": args.impl,
        "tp_size": strategy.tp_size,
        "dp_size": strategy.dp_size,
        "use_vocab_parallel": args.use_vocab_parallel,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "learning_rate": args.learning_rate,
        "dtype": args.dtype,
        "autocast": args.autocast,
        "num_training_steps": args.num_training_steps,
        "warmup_steps": args.warmup_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "activation_checkpointing": args.activation_checkpointing,
        "world_size": world_size,
    })

    # Reset peak memory before training
    torch.cuda.reset_peak_memory_stats()

    # Set up profiling
    profiler_context = nullcontext()
    if args.profile:
        profile_dir = args.profile_dir or os.path.join(args.output_dir, "profiles")
        profile_subdir = os.path.join(
            profile_dir,
            f"{args.impl}_tp{strategy.tp_size}_dp{strategy.dp_size}_rank{rank}",
        )
        os.makedirs(profile_subdir, exist_ok=True)

        if rank == 0:
            print(f"Profiling enabled, traces will be saved to: {profile_dir}")

        profiler_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=args.warmup_steps,
                active=args.profile_steps,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_subdir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    if rank == 0:
        print(f"\nStarting training loop...")
        print()

    # Run training loop
    with profiler_context as profiler:
        training_loop(
            strategy=strategy,
            dataloader=dataloader,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            log_interval=args.log_interval,
            metrics_collector=metrics_collector,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            rank=rank,
            profiler=profiler,
        )

    # Print and save results
    if rank == 0:
        metrics_collector.print_summary()

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(
            args.output_dir,
            f"results_{args.impl}_tp{strategy.tp_size}_dp{strategy.dp_size}.json",
        )
        metrics_collector.save(output_path)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
