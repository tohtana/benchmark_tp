"""Metrics collection for TP benchmarking."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch


@dataclass
class StepMetrics:
    """Metrics for a single training step."""

    step: int
    loss: float
    step_time_s: float
    forward_time_s: float
    backward_time_s: float
    optimizer_time_s: float
    tokens_per_sec: float
    samples_per_sec: float
    memory_allocated_gb: float
    memory_peak_gb: float


class MetricsCollector:
    """
    Collects and aggregates benchmark metrics.

    Tracks per-step metrics including timing breakdown, throughput,
    memory usage, and loss values.
    """

    def __init__(self, seq_length: int, batch_size: int):
        """
        Initialize metrics collector.

        Args:
            seq_length: Sequence length for throughput calculation
            batch_size: Batch size for throughput calculation
        """
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.steps: List[StepMetrics] = []
        self.config: Dict = {}

    def set_config(self, config: Dict) -> None:
        """
        Store benchmark configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def record_step(
        self,
        step: int,
        loss: float,
        step_time_s: float,
        forward_time_s: float,
        backward_time_s: float,
        optimizer_time_s: float,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
    ) -> None:
        """
        Record metrics for a single training step.

        Args:
            step: Step number
            loss: Loss value
            step_time_s: Total step time in seconds
            forward_time_s: Forward pass time in seconds
            backward_time_s: Backward pass time in seconds
            optimizer_time_s: Optimizer step time in seconds
            batch_size: Override batch size (optional)
            seq_length: Override sequence length (optional)
        """
        batch_size = batch_size or self.batch_size
        seq_length = seq_length or self.seq_length

        tokens = batch_size * seq_length
        tokens_per_sec = tokens / step_time_s if step_time_s > 0 else 0
        samples_per_sec = batch_size / step_time_s if step_time_s > 0 else 0

        metrics = StepMetrics(
            step=step,
            loss=loss,
            step_time_s=step_time_s,
            forward_time_s=forward_time_s,
            backward_time_s=backward_time_s,
            optimizer_time_s=optimizer_time_s,
            tokens_per_sec=tokens_per_sec,
            samples_per_sec=samples_per_sec,
            memory_allocated_gb=torch.cuda.memory_allocated() / 1e9,
            memory_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        self.steps.append(metrics)

    def get_summary(self) -> Dict:
        """
        Compute summary statistics from collected metrics.

        Returns:
            Dictionary with summary statistics
        """
        if not self.steps:
            return {}

        step_times = [s.step_time_s for s in self.steps]
        forward_times = [s.forward_time_s for s in self.steps]
        backward_times = [s.backward_time_s for s in self.steps]
        optimizer_times = [s.optimizer_time_s for s in self.steps]
        tokens_per_sec = [s.tokens_per_sec for s in self.steps]
        samples_per_sec = [s.samples_per_sec for s in self.steps]
        losses = [s.loss for s in self.steps]

        total_time = sum(step_times)

        return {
            "num_steps": len(self.steps),
            "avg_step_time_ms": sum(step_times) / len(step_times) * 1000,
            "std_step_time_ms": self._std(step_times) * 1000,
            "min_step_time_ms": min(step_times) * 1000,
            "max_step_time_ms": max(step_times) * 1000,
            "avg_tokens_per_sec": sum(tokens_per_sec) / len(tokens_per_sec),
            "avg_samples_per_sec": sum(samples_per_sec) / len(samples_per_sec),
            "avg_loss": sum(losses) / len(losses),
            "final_loss": losses[-1],
            "peak_memory_gb": max(s.memory_peak_gb for s in self.steps),
            "timing_breakdown": {
                "forward_pct": sum(forward_times) / total_time * 100 if total_time > 0 else 0,
                "backward_pct": sum(backward_times) / total_time * 100 if total_time > 0 else 0,
                "optimizer_pct": sum(optimizer_times) / total_time * 100 if total_time > 0 else 0,
                "other_pct": (
                    total_time - sum(forward_times) - sum(backward_times) - sum(optimizer_times)
                ) / total_time * 100 if total_time > 0 else 0,
            },
        }

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def save(self, path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "config": self.config,
            "steps": [asdict(s) for s in self.steps],
            "summary": self.get_summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.get_summary()
        if not summary:
            print("No metrics collected")
            return

        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Steps completed: {summary['num_steps']}")
        print(f"Average step time: {summary['avg_step_time_ms']:.2f} ms (std: {summary['std_step_time_ms']:.2f} ms)")
        print(f"Throughput: {summary['avg_tokens_per_sec']:.0f} tokens/sec")
        print(f"Throughput: {summary['avg_samples_per_sec']:.2f} samples/sec")
        print(f"Average loss: {summary['avg_loss']:.4f}")
        print(f"Final loss: {summary['final_loss']:.4f}")
        print(f"Peak memory: {summary['peak_memory_gb']:.2f} GB")
        print("\nTiming breakdown:")
        print(f"  Forward:   {summary['timing_breakdown']['forward_pct']:.1f}%")
        print(f"  Backward:  {summary['timing_breakdown']['backward_pct']:.1f}%")
        print(f"  Optimizer: {summary['timing_breakdown']['optimizer_pct']:.1f}%")
        print(f"  Other:     {summary['timing_breakdown']['other_pct']:.1f}%")
        print("=" * 60)
