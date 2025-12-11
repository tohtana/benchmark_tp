"""Factory for creating tensor parallelism strategies."""

from typing import Optional

from .base import BaseTPStrategy
from .autotp import AutoTPStrategy
from .fsdp_dtensor import FSDPDTensorStrategy


def create_tp_strategy(
    impl: str,
    tp_size: Optional[int] = None,
    dp_size: int = 1,
    use_vocab_parallel: bool = True,
) -> BaseTPStrategy:
    """
    Factory function to create a TP strategy.

    Args:
        impl: Implementation name ("autotp" or "fsdp_dtensor")
        tp_size: Tensor parallel degree (None = auto-detect)
        dp_size: Data parallel degree (default: 1)
        use_vocab_parallel: Enable vocabulary-parallel embedding (AutoTP only)

    Returns:
        Configured TP strategy instance

    Raises:
        ValueError: If unknown implementation is requested
    """
    if impl == "autotp":
        return AutoTPStrategy(tp_size=tp_size, dp_size=dp_size, use_vocab_parallel=use_vocab_parallel)
    elif impl == "fsdp_dtensor":
        return FSDPDTensorStrategy(tp_size=tp_size, dp_size=dp_size)
    else:
        raise ValueError(
            f"Unknown TP implementation: '{impl}'. "
            f"Available implementations: 'autotp', 'fsdp_dtensor'"
        )


def list_implementations() -> list:
    """
    List available TP implementations.

    Returns:
        List of implementation names
    """
    return ["autotp", "fsdp_dtensor"]
