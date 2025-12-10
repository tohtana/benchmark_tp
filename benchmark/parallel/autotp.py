"""DeepSpeed AutoTP tensor parallelism strategy."""

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from ..models.base import BaseModelBuilder
from .base import BaseTPStrategy


class AutoTPStrategy(BaseTPStrategy):
    """
    DeepSpeed AutoTP tensor parallelism strategy.

    Uses DeepSpeed's automatic tensor parallelism via:
    - set_autotp_mode() for instrumentation
    - deepspeed.tp_model_init() for automatic model sharding
    - DeepSpeed engine for training
    """

    def __init__(self, tp_size: Optional[int] = None, dp_size: int = 1):
        super().__init__(tp_size, dp_size)
        self.engine = None
        self.tp_group = None

    @property
    def strategy_name(self) -> str:
        return "DeepSpeed AutoTP"

    def setup(
        self,
        model_builder: BaseModelBuilder,
        device: torch.device,
        dtype: torch.dtype,
        config: Dict[str, Any],
    ) -> None:
        """
        Set up DeepSpeed AutoTP.

        Based on multimodal-training/python/text_autotp_example.py and
        multimodal-training/python/ray/text.py:_build_model_with_autotp()
        """
        import deepspeed
        from deepspeed.module_inject.layers import set_autotp_mode

        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Determine TP size
        if self.tp_size is None:
            self.tp_size = self.world_size

        # Validate configuration
        model_config = model_builder.get_config()
        if model_config.num_key_value_heads % self.tp_size != 0:
            raise ValueError(
                f"TP size {self.tp_size} must divide num_key_value_heads "
                f"{model_config.num_key_value_heads}"
            )

        if self.rank == 0:
            print(f"[AutoTP] Setting up with tp_size={self.tp_size}")

        # Enable AutoTP instrumentation BEFORE model creation
        set_autotp_mode(training=True)

        # Create model
        model = model_builder.create_model(
            device=device,
            dtype=dtype,
            num_layers=config.get("num_layers", 0),
            attn_impl=config.get("attn_impl", "sdpa"),
        )

        # Apply activation checkpointing if requested
        if config.get("activation_checkpointing", False):
            model.gradient_checkpointing_enable()
            if self.rank == 0:
                print("[AutoTP] Enabled activation checkpointing")

        # Apply TP sharding with deepspeed.tp_model_init()
        model = deepspeed.tp_model_init(model, tp_size=self.tp_size, dtype=dtype)

        # Store TP group reference
        self.tp_group = getattr(model, "tp_group", None)

        # Get all parameters
        params = list(model.parameters())

        # Build DeepSpeed config
        zero_stage = config.get("zero_stage", 1)
        batch_size = config.get("batch_size", 1)
        gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        ds_config = {
            "train_batch_size": batch_size * self.world_size * gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": config.get("max_grad_norm", 1.0),
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
            },
            "tensor_parallel": {
                "autotp_size": self.tp_size,
            },
            "zero_allow_untested_optimizer": True,
            "steps_per_print": 2000,
            "wall_clock_breakdown": False,
        }

        # Add precision config
        if dtype == torch.bfloat16:
            ds_config["bf16"] = {"enabled": True}
        elif dtype == torch.float16:
            ds_config["fp16"] = {"enabled": True, "initial_scale_power": 8}

        # Create optimizer
        learning_rate = config.get("learning_rate", 1e-5)
        weight_decay = config.get("weight_decay", 0.01)
        optimizer = torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay
        )

        # Initialize DeepSpeed engine
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config,
        )

        self.model = self.engine.module

        if self.rank == 0:
            num_params = sum(p.numel() for p in params)
            print(f"[AutoTP] Model initialized with {num_params:,} parameters")
            print(f"[AutoTP] DeepSpeed engine created with ZeRO-{zero_stage}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass with AutoTP model."""
        outputs = self.engine(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        return outputs.loss

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass through DeepSpeed engine."""
        self.engine.backward(loss)

    def optimizer_step(self) -> None:
        """Run optimizer step through DeepSpeed engine."""
        self.engine.step()

    def zero_grad(self) -> None:
        """Zero gradients - handled internally by DeepSpeed."""
        # DeepSpeed handles this internally, but we can explicitly call it
        pass
