"""Qwen model builder for TP benchmarking."""

from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .base import BaseModelBuilder, ModelConfig
from .registry import register_model


@contextmanager
def use_default_device(device):
    """Context manager to temporarily set default device."""
    prev_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(prev_device)


@register_model("qwen")
class QwenModelBuilder(BaseModelBuilder):
    """
    Model builder for Qwen models (Qwen2, Qwen2.5, Qwen3).

    Supports both Qwen2.5-VL and standard Qwen models.
    """

    def get_config(self) -> ModelConfig:
        """Get model configuration from HuggingFace."""
        if self._config is None:
            hf_config = AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            # Handle Qwen2.5-VL which has nested text_config
            if hasattr(hf_config, "text_config"):
                text_config = hf_config.text_config
            else:
                text_config = hf_config

            self._config = ModelConfig(
                name=self.model_name,
                hidden_size=text_config.hidden_size,
                num_attention_heads=text_config.num_attention_heads,
                num_key_value_heads=getattr(
                    text_config, "num_key_value_heads", text_config.num_attention_heads
                ),
                num_hidden_layers=text_config.num_hidden_layers,
                intermediate_size=text_config.intermediate_size,
                vocab_size=text_config.vocab_size,
            )
        return self._config

    def create_model(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_layers: int = 0,
        attn_impl: str = "sdpa",
    ) -> nn.Module:
        """
        Create Qwen model with random initialization.

        Args:
            device: Device to place the model on
            dtype: Data type for model parameters
            num_layers: Override number of layers (0 = use model default)
            attn_impl: Attention implementation to use

        Returns:
            Initialized Qwen model
        """
        hf_config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )

        # Handle Qwen2.5-VL which has nested text_config
        if hasattr(hf_config, "text_config"):
            config_to_modify = hf_config.text_config
        else:
            config_to_modify = hf_config

        # Override number of layers if specified
        if num_layers > 0:
            original_layers = config_to_modify.num_hidden_layers
            config_to_modify.num_hidden_layers = num_layers
            print(f"Overriding num_hidden_layers: {original_layers} -> {num_layers}")

        # Create model with random initialization
        with use_default_device(device):
            model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)

        # Move to target dtype
        model = model.to(dtype=dtype)

        return model

    def get_tp_mapping(self) -> Dict[str, Any]:
        """
        Get tensor parallel layer mapping for Qwen models.

        Based on QwenTextMixin._get_tensor_parallel_mapping() from
        multimodal-training/python/ray/text.py:1165-1177

        Returns:
            Dictionary mapping layer names to DTensor parallelization strategies
        """
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        return {
            # Attention projections
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            # MLP projections
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }

    def get_transformer_layers(self, model: nn.Module):
        """
        Get transformer layers from Qwen model.

        Args:
            model: The Qwen model

        Returns:
            ModuleList of transformer layers
        """
        # Standard Qwen/Qwen2/Qwen3 structure
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers

        raise AttributeError(
            f"Cannot find transformer layers in model {type(model)}. "
            f"Expected model.model.layers attribute."
        )

    def get_embedding_module(self, model: nn.Module) -> nn.Embedding:
        """
        Get embedding module from Qwen model.

        Args:
            model: The Qwen model

        Returns:
            The embedding module
        """
        # Standard Qwen structure uses embed_tokens
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens

        raise AttributeError(
            f"Cannot find embedding module in model {type(model)}. "
            f"Expected model.model.embed_tokens attribute."
        )
