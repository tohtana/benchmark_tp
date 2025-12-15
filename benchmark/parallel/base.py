"""Base tensor parallelism strategy abstraction."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..models.base import BaseModelBuilder


class BaseTPStrategy(ABC):
    """
    Abstract base class for tensor parallelism strategies.

    Provides a common interface for different TP implementations
    (DeepSpeed AutoTP, FSDP+DTensor, etc.).
    """

    def __init__(self, tp_size: Optional[int] = None, dp_size: int = 1):
        """
        Initialize the TP strategy.

        Args:
            tp_size: Tensor parallel degree (None = auto-detect)
            dp_size: Data parallel degree (default: 1)
        """
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.rank: int = 0
        self.world_size: int = 1
        self.device: Optional[torch.device] = None

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the human-readable name of this strategy."""
        pass

    @abstractmethod
    def setup(
        self,
        model_builder: BaseModelBuilder,
        device: torch.device,
        dtype: torch.dtype,
        config: Dict[str, Any],
    ) -> None:
        """
        Set up the parallelism strategy.

        Creates the model, applies parallelization, and initializes optimizer.

        Args:
            model_builder: Model builder to create the model
            device: Device to place the model on
            dtype: Data type for model parameters
            config: Configuration dictionary with training parameters
        """
        pass

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run forward pass and compute loss.

        Args:
            batch: Dictionary with input_ids, labels, attention_mask

        Returns:
            Loss tensor
        """
        pass

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """
        Run backward pass.

        Args:
            loss: Loss tensor from forward pass
        """
        pass

    def forward_backward(
        self, batch: Dict[str, torch.Tensor], loss_scale: float = 1.0
    ) -> tuple:
        """
        Run forward and backward passes together.

        This method allows strategies to wrap both forward and backward
        in a shared context (e.g., for loss_parallel in DTensor). The default
        implementation calls forward() and backward() separately.

        Override this method if the strategy requires both passes to be in the same context.

        Args:
            batch: Input batch dictionary
            loss_scale: Scale factor for the loss (e.g., 1/gradient_accumulation_steps)

        Returns:
            Tuple of (loss, forward_time, backward_time) where times are in seconds
        """
        import time

        torch.cuda.synchronize()
        forward_start = time.perf_counter()

        loss = self.forward(batch)

        torch.cuda.synchronize()
        forward_time = time.perf_counter() - forward_start

        scaled_loss = loss * loss_scale

        torch.cuda.synchronize()
        backward_start = time.perf_counter()

        self.backward(scaled_loss)

        torch.cuda.synchronize()
        backward_time = time.perf_counter() - backward_start

        return loss, forward_time, backward_time

    @abstractmethod
    def optimizer_step(self) -> None:
        """Run optimizer step."""
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero gradients."""
        pass

    def get_model_parameters(self) -> int:
        """
        Get total number of model parameters.

        Returns:
            Total parameter count
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_parameters(self) -> int:
        """
        Get number of trainable parameters.

        Returns:
            Trainable parameter count
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
