"""CLI argument parsing for TP benchmarking."""

import argparse


def get_args():
    """Parse command line arguments for TP benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark tensor parallelism implementations (DeepSpeed AutoTP vs FSDP+DTensor)"
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name or path (HuggingFace model ID)",
    )

    # Parallelism configuration
    parser.add_argument(
        "--impl",
        type=str,
        choices=["autotp", "fsdp_dtensor"],
        default="autotp",
        help="TP implementation to benchmark",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=None,
        help="Tensor parallel degree (defaults to world_size for autotp, world_size/dp_size for fsdp_dtensor)",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Data parallel degree (only used for fsdp_dtensor)",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="DeepSpeed ZeRO stage (only used for autotp)",
    )

    # Training configuration
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-GPU micro batch size",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=2048,
        help="Sequence length for synthetic data",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=100,
        help="Number of training steps to run",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # Memory optimization
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Enable activation checkpointing (gradient checkpointing)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Training data type",
    )

    # Benchmarking configuration
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5,
        help="Number of warmup steps to skip in metrics",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Logging interval (in steps)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save benchmark results",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=0,
        help="Override number of layers (0 = use model default)",
    )
    parser.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation",
    )

    return parser.parse_args()
