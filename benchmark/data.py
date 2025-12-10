"""Synthetic data generation for TP benchmarking."""

import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    """
    Synthetic dataset that generates random token sequences.

    Used for benchmarking to avoid I/O bottlenecks and ensure
    consistent data across runs.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_length: int,
        num_samples: int = 10000,
        seed: int = 42,
    ):
        """
        Initialize synthetic dataset.

        Args:
            vocab_size: Vocabulary size for random token generation
            seq_length: Length of each sequence
            num_samples: Number of samples in the dataset
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.seed = seed

        # Pre-generate all data for consistency
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_length),
            generator=generator,
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary with input_ids, labels, and attention_mask
        """
        input_ids = self.input_ids[idx]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # For causal LM, labels = input_ids
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
        }


def create_dataloader(
    vocab_size: int,
    seq_length: int,
    batch_size: int,
    num_samples: int = 10000,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader with synthetic data.

    Args:
        vocab_size: Vocabulary size for random token generation
        seq_length: Length of each sequence
        batch_size: Batch size
        num_samples: Number of samples in the dataset
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers

    Returns:
        DataLoader with synthetic data
    """
    dataset = SyntheticDataset(
        vocab_size=vocab_size,
        seq_length=seq_length,
        num_samples=num_samples,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for reproducibility in benchmarking
        drop_last=True,  # Drop last incomplete batch for consistent timing
        num_workers=num_workers,
        pin_memory=True,
    )
