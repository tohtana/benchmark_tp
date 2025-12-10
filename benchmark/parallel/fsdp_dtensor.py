"""FSDP + DTensor 2D mesh tensor parallelism strategy."""

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from ..models.base import BaseModelBuilder
from .base import BaseTPStrategy


class FSDPDTensorStrategy(BaseTPStrategy):
    """
    FSDP2 + DTensor 2D mesh tensor parallelism strategy.

    Uses a 2D device mesh where:
    - First dimension (dp) is for data parallelism via FSDP2 (fully_shard)
    - Second dimension (tp) is for tensor parallelism via DTensor

    For example, with 8 GPUs, dp_size=2, tp_size=4:
    - Device mesh shape: (2, 4)
    - TP groups (same row): {0,1,2,3}, {4,5,6,7}
    - DP groups (same column): {0,4}, {1,5}, {2,6}, {3,7}
    """

    def __init__(self, tp_size: Optional[int] = None, dp_size: int = 1):
        super().__init__(tp_size, dp_size)
        self.device_mesh = None
        self.tp_mesh = None
        self.dp_mesh = None
        self.tp_group = None

    @property
    def strategy_name(self) -> str:
        return "FSDP2 + DTensor (2D Mesh)"

    def setup(
        self,
        model_builder: BaseModelBuilder,
        device: torch.device,
        dtype: torch.dtype,
        config: Dict[str, Any],
    ) -> None:
        """
        Set up FSDP2 + DTensor with 2D device mesh.

        The mesh is organized as (dp, tp) where:
        - dp dimension: FSDP2 shards optimizer states and gradients
        - tp dimension: DTensor shards model weights for tensor parallelism
        """
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

        self.device = device
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Determine TP size
        if self.tp_size is None:
            self.tp_size = self.world_size // self.dp_size

        # Validate configuration
        if self.dp_size * self.tp_size != self.world_size:
            raise ValueError(
                f"dp_size ({self.dp_size}) * tp_size ({self.tp_size}) must equal "
                f"world_size ({self.world_size})"
            )

        model_config = model_builder.get_config()
        if model_config.num_key_value_heads % self.tp_size != 0:
            raise ValueError(
                f"TP size {self.tp_size} must divide num_key_value_heads "
                f"{model_config.num_key_value_heads}"
            )

        if self.rank == 0:
            print(
                f"[FSDP2+DTensor] Setting up 2D mesh: dp_size={self.dp_size}, tp_size={self.tp_size}"
            )

        # Create 2D device mesh: (dp, tp)
        self.device_mesh = init_device_mesh(
            "cuda", (self.dp_size, self.tp_size), mesh_dim_names=("dp", "tp")
        )
        self.tp_mesh = self.device_mesh["tp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_group = self.tp_mesh.get_group()

        if self.rank == 0:
            print(f"[FSDP2+DTensor] Device mesh created: {self.device_mesh}")

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
                print("[FSDP2+DTensor] Enabled activation checkpointing")

        # Step 1: Apply DTensor TP to transformer layers
        tp_mapping = model_builder.get_tp_mapping()
        layers = model_builder.get_transformer_layers(model)

        if self.rank == 0:
            print(f"[FSDP2+DTensor] Applying DTensor TP to {len(layers)} layers")

        for layer in layers:
            parallelize_module(layer, self.tp_mesh, tp_mapping)

        # Step 2: Apply FSDP2 (fully_shard) to each transformer layer
        # FSDP2 is composable and works well with DTensor
        mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)

        if self.dp_size > 1:
            if self.rank == 0:
                print(f"[FSDP2+DTensor] Applying FSDP2 to transformer layers")

            for layer in layers:
                fully_shard(layer, mesh=self.dp_mesh, mp_policy=mp_policy)

            # Apply to the whole model
            fully_shard(model, mesh=self.dp_mesh, mp_policy=mp_policy)
        else:
            if self.rank == 0:
                print("[FSDP2+DTensor] dp_size=1, skipping FSDP sharding (TP only)")

        self.model = model

        # Store TP group on model for compatibility
        self.model.tp_group = self.tp_group

        # Create optimizer
        # Note: Use foreach=False because DTensor doesn't support fused optimizer ops
        # when there's a mix of DTensor and regular Tensor parameters
        learning_rate = config.get("learning_rate", 1e-5)
        weight_decay = config.get("weight_decay", 0.01)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            foreach=False,  # Required for DTensor compatibility
        )

        if self.rank == 0:
            num_params = sum(p.numel() for p in model.parameters())
            print(f"[FSDP2+DTensor] Model initialized with {num_params:,} parameters")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run forward pass with FSDP2+DTensor model."""
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
        )
        return outputs.loss

    def backward(self, loss: torch.Tensor) -> None:
        """Run backward pass."""
        loss.backward()

    def optimizer_step(self) -> None:
        """Run optimizer step."""
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=True)
