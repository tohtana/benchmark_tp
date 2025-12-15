"""Base model builder abstraction for TP benchmarking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration dataclass for model specifications."""

    name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int


class BaseModelBuilder(ABC):
    """
    Abstract base class for model builders.

    Provides a common interface for creating models and getting
    their tensor parallelism configurations.
    """

    def __init__(self, model_name: str):
        """
        Initialize the model builder.

        Args:
            model_name: HuggingFace model name or local path
        """
        self.model_name = model_name
        self._config: Optional[ModelConfig] = None

    @abstractmethod
    def get_config(self) -> ModelConfig:
        """
        Get the model configuration.

        Returns:
            ModelConfig with model specifications
        """
        pass

    @abstractmethod
    def create_model(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_layers: int = 0,
        attn_impl: str = "sdpa",
    ) -> nn.Module:
        """
        Create the model with random initialization.

        Args:
            device: Device to place the model on
            dtype: Data type for model parameters
            num_layers: Override number of layers (0 = use model default)
            attn_impl: Attention implementation to use

        Returns:
            Initialized model
        """
        pass

    @abstractmethod
    def get_tp_mapping(self) -> Dict[str, Any]:
        """
        Get tensor parallel layer mapping for DTensor.

        Returns a mapping from layer names to their parallelization
        strategies (ColwiseParallel or RowwiseParallel).

        Returns:
            Dictionary mapping layer names to parallelization strategies
        """
        pass

    @abstractmethod
    def get_transformer_layers(self, model: nn.Module):
        """
        Get the list of transformer layers from the model.

        Args:
            model: The model to get layers from

        Returns:
            List or ModuleList of transformer layers
        """
        pass

    @abstractmethod
    def get_embedding_module(self, model: nn.Module) -> nn.Embedding:
        """
        Get the embedding module from the model.

        Args:
            model: The model to get embedding from

        Returns:
            The embedding module
        """
        pass

    def get_lm_head(self, model: nn.Module) -> nn.Linear:
        """
        Get the language model head from the model.

        Args:
            model: The model to get lm_head from

        Returns:
            The lm_head linear layer
        """
        if hasattr(model, "lm_head"):
            return model.lm_head
        raise AttributeError(f"Model {type(model)} does not have lm_head attribute")

    def get_vocab_parallel_mapping(self, loss_parallel: bool = True) -> Dict[str, Any]:
        """
        Get tensor parallel mapping for embedding and lm_head layers with vocab parallelism.

        When loss_parallel is enabled:
        - Embedding: RowwiseParallel - input tokens replicated, output embeddings sharded by hidden dim
        - lm_head: ColwiseParallel - output logits sharded by vocab dim for efficient loss computation

        When loss_parallel is disabled:
        - Embedding: RowwiseParallel - same as above
        - lm_head: ColwiseParallel - output logits replicated across TP ranks

        Args:
            loss_parallel: Whether to enable loss parallel (shard output logits by vocab dim)

        Returns:
            Dictionary mapping layer names to parallelization strategies
        """
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        return {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        }

    @abstractmethod
    def replace_embedding_module(self, model: nn.Module, new_embedding: nn.Module) -> None:
        """
        Replace the embedding module in the model with a new one.

        Args:
            model: The model whose embedding should be replaced
            new_embedding: The new embedding module (e.g., VocabParallelEmbedding)
        """
        pass
