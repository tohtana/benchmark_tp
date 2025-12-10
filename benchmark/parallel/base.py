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
