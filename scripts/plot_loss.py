#!/usr/bin/env python3
"""
Plot loss curves from benchmark log files.

Usage:
    python scripts/plot_loss.py /tmp/autotp_wikitext.log /tmp/fsdp_dtensor_wikitext.log
    python scripts/plot_loss.py --output loss_comparison.png log1.log log2.log
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: str) -> Tuple[List[int], List[float]]:
    """Parse a benchmark log file and extract step numbers and loss values."""
    steps = []
    losses = []

    # Pattern to match log lines like:
    # [training] Step 6/1000 | Loss: 0.8990 | Step: 327.5ms | ...
    pattern = re.compile(r'\[(?:warmup|training)\] Step (\d+)/\d+ \| Loss: ([\d.]+)')

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)

    return steps, losses


def extract_impl_name(log_path: str) -> str:
    """Extract implementation name from log filename."""
    path = Path(log_path)
    name = path.stem.lower()

    if 'autotp' in name:
        return 'DeepSpeed AutoTP'
    elif 'fsdp' in name or 'dtensor' in name:
        return 'FSDP2 + DTensor'
    else:
        return path.stem


def plot_loss_comparison(log_files: List[str], output_path: str = None,
                         smoothing_window: int = 50, title: str = None):
    """Plot loss curves from multiple log files."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']

    all_data = []
    for i, log_file in enumerate(log_files):
        steps, losses = parse_log_file(log_file)
        impl_name = extract_impl_name(log_file)
        color = colors[i % len(colors)]
        all_data.append((impl_name, steps, losses, color))

    # Left plot: Raw loss values
    ax1 = axes[0]
    for impl_name, steps, losses, color in all_data:
        ax1.plot(steps, losses, alpha=0.7, linewidth=0.8, color=color, label=impl_name)

    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss (Raw)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Right plot: Smoothed loss with moving average
    ax2 = axes[1]
    for impl_name, steps, losses, color in all_data:
        if len(losses) >= smoothing_window:
            smoothed = np.convolve(losses, np.ones(smoothing_window)/smoothing_window, mode='valid')
            smoothed_steps = steps[smoothing_window-1:]
            ax2.plot(smoothed_steps, smoothed, linewidth=2, color=color, label=impl_name)
        else:
            ax2.plot(steps, losses, linewidth=2, color=color, label=impl_name)

    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title(f'Training Loss (Smoothed, window={smoothing_window})', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Print statistics
    print("\n" + "="*60)
    print("LOSS COMPARISON STATISTICS")
    print("="*60)
    for impl_name, steps, losses, _ in all_data:
        print(f"\n{impl_name}:")
        print(f"  Total steps: {len(steps)}")
        print(f"  Initial loss (step 1): {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Average loss: {np.mean(losses):.4f}")
        print(f"  Min loss: {np.min(losses):.4f}")
        print(f"  Max loss: {np.max(losses):.4f}")

        # Skip warmup (first 5 steps) for training stats
        if len(losses) > 5:
            training_losses = losses[5:]
            print(f"  Training avg loss (after warmup): {np.mean(training_losses):.4f}")

    print("\n" + "="*60)

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    else:
        output_path = "loss_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Plot loss curves from benchmark logs')
    parser.add_argument('log_files', nargs='+', help='Log files to plot')
    parser.add_argument('--output', '-o', default='loss_comparison.png',
                        help='Output plot filename (default: loss_comparison.png)')
    parser.add_argument('--smoothing', '-s', type=int, default=50,
                        help='Smoothing window size for moving average (default: 50)')
    parser.add_argument('--title', '-t', default='TP Implementation Loss Comparison',
                        help='Plot title')

    args = parser.parse_args()

    # Validate files exist
    for log_file in args.log_files:
        if not Path(log_file).exists():
            print(f"Error: Log file not found: {log_file}")
            return 1

    plot_loss_comparison(
        args.log_files,
        args.output,
        smoothing_window=args.smoothing,
        title=args.title
    )

    return 0


if __name__ == '__main__':
    exit(main())
