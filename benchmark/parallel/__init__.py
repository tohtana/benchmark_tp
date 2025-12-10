"""Parallelism strategies for TP benchmarking."""

from .base import BaseTPStrategy
from .factory import create_tp_strategy

__all__ = ["BaseTPStrategy", "create_tp_strategy"]
