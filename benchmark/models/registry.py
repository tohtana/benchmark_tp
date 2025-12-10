"""Model registry for extensible model support."""

from typing import Dict, Type

from .base import BaseModelBuilder


# Global model registry
_MODEL_REGISTRY: Dict[str, Type[BaseModelBuilder]] = {}


def register_model(name: str):
    """
    Decorator to register a model builder class.

    Usage:
        @register_model("qwen")
        class QwenModelBuilder(BaseModelBuilder):
            ...

    Args:
        name: The name to register the model under (case-insensitive matching)

    Returns:
        Decorator function
    """

    def decorator(cls: Type[BaseModelBuilder]) -> Type[BaseModelBuilder]:
        _MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_model_builder(model_name: str) -> BaseModelBuilder:
    """
    Get a model builder instance for the given model name.

    Tries exact match first, then prefix matching.

    Args:
        model_name: HuggingFace model name or local path

    Returns:
        Instantiated model builder

    Raises:
        ValueError: If no matching model builder is found
    """
    model_name_lower = model_name.lower()

    # Try exact match first
    if model_name_lower in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[model_name_lower](model_name)

    # Try prefix matching (e.g., "Qwen/Qwen3-32B" matches "qwen")
    for key, builder_cls in _MODEL_REGISTRY.items():
        if key in model_name_lower:
            return builder_cls(model_name)

    available = list(_MODEL_REGISTRY.keys())
    raise ValueError(
        f"No model builder found for '{model_name}'. "
        f"Available models: {available}. "
        f"Register new models using @register_model decorator."
    )


def list_registered_models() -> list:
    """
    List all registered model names.

    Returns:
        List of registered model names
    """
    return list(_MODEL_REGISTRY.keys())
