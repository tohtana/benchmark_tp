"""Data loading utilities for TP benchmarking."""

from typing import Optional, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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


def get_tokenizer(model_name: str, trust_remote_code: bool = True) -> Any:
    """
    Load and configure the tokenizer for the given model.

    Args:
        model_name: Name of the model to load tokenizer for
        trust_remote_code: Whether to trust remote code

    Returns:
        Configured tokenizer
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback for models without eos_token
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(2)

    return tokenizer


def load_real_dataset(
    dataset_name: str,
    dataset_percentage: float,
    tokenizer: Any,
    seq_length: int,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True,
) -> DataLoader:
    """
    Load and prepare a real dataset with tokenization and data loader creation.

    Args:
        dataset_name: Name of the dataset to load
        dataset_percentage: Percentage of dataset to use (0.0-100.0)
        tokenizer: Tokenizer to use for text processing
        seq_length: Maximum sequence length for tokenization
        batch_size: Batch size for the data loader
        rank: Process rank for distributed sampling
        world_size: Total number of processes
        is_main_process: Whether this is the main process

    Returns:
        DataLoader with real dataset
    """
    from datasets import load_dataset, DownloadConfig
    from datasets.utils.logging import disable_progress_bar

    # Disable progress bar for non-main processes
    if not is_main_process:
        disable_progress_bar()

    # Convert percentage from 0-100 to 0-1.0
    fraction = dataset_percentage / 100.0

    if is_main_process:
        print(f"Loading dataset: {dataset_name} ({dataset_percentage:.1f}% of data)...")

    # Calculate split string based on percentage
    if fraction >= 1.0:
        split_str = "train"
    else:
        # Convert to integer percentage to avoid decimal points
        percentage_int = int(fraction * 100)
        split_str = f"train[:{percentage_int}%]"

    # Load the specified dataset
    if dataset_name == "wikitext":
        dataset = load_dataset(
            'wikitext', 'wikitext-103-raw-v1', split=split_str,
            download_config=DownloadConfig(disable_tqdm=True)
        )
        text_column = 'text'
    elif dataset_name == "openwebtext":
        # For openwebtext, use a smaller percentage since it's very large
        if fraction >= 1.0:
            split_str = "train[:1%]"  # Cap at 1% for full dataset requests
        dataset = load_dataset(
            'openwebtext', split=split_str,
            download_config=DownloadConfig(disable_tqdm=True)
        )
        text_column = 'text'
    elif dataset_name == "c4":
        # For C4, use even smaller percentage since it's massive
        if fraction >= 1.0:
            split_str = "train[:0.1%]"  # Cap at 0.1% for full dataset requests
        dataset = load_dataset(
            'c4', 'en', split=split_str,
            download_config=DownloadConfig(disable_tqdm=True)
        )
        text_column = 'text'
    elif dataset_name == "ag_news":
        dataset = load_dataset(
            'ag_news', split=split_str,
            download_config=DownloadConfig(disable_tqdm=True)
        )
        text_column = 'text'
    else:
        # Try to load as custom dataset
        try:
            dataset = load_dataset(
                dataset_name, split=split_str,
                download_config=DownloadConfig(disable_tqdm=True)
            )
            # Try common text column names
            if 'text' in dataset.column_names:
                text_column = 'text'
            elif 'content' in dataset.column_names:
                text_column = 'content'
            elif 'body' in dataset.column_names:
                text_column = 'body'
            else:
                text_column = dataset.column_names[0]
                if is_main_process:
                    print(f"Warning: Using column '{text_column}' as text column. "
                          f"Available columns: {dataset.column_names}")
        except Exception as e:
            if is_main_process:
                print(f"Error loading dataset '{dataset_name}': {e}")
                print("Falling back to wikitext dataset...")
            dataset = load_dataset(
                'wikitext', 'wikitext-103-raw-v1', split=split_str,
                download_config=DownloadConfig(disable_tqdm=True)
            )
            text_column = 'text'

    if is_main_process:
        print(f"Dataset loaded: {len(dataset)} examples using column '{text_column}'")

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding='max_length',
            max_length=seq_length,
            truncation=True
        )

    # Tokenize the dataset
    if is_main_process:
        print("Tokenizing dataset...")

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, num_proc=1, keep_in_memory=True
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    if is_main_process:
        print(f"Tokenization complete. Dataset ready with {len(tokenized_dataset)} examples.")

    # Create data loader with distributed sampler
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=world_size,
        rank=rank
    )

    # Wrapper to add labels for causal LM training
    class LabeledDataset(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            item = self.base_dataset[idx]
            return {
                "input_ids": item["input_ids"],
                "labels": item["input_ids"].clone(),
                "attention_mask": item["attention_mask"],
            }

    labeled_dataset = LabeledDataset(tokenized_dataset)

    data_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    return data_loader


def create_dataloader(
    vocab_size: int,
    seq_length: int,
    batch_size: int,
    num_samples: int = 10000,
    seed: int = 42,
    num_workers: int = 0,
    # Real dataset parameters
    dataset_name: Optional[str] = None,
    dataset_percentage: float = 10.0,
    model_name: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with either synthetic or real data.

    Args:
        vocab_size: Vocabulary size for random token generation (synthetic only)
        seq_length: Length of each sequence
        batch_size: Batch size
        num_samples: Number of samples in the dataset (synthetic only)
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers (synthetic only)
        dataset_name: Name of real dataset to use (if None, uses synthetic data)
        dataset_percentage: Percentage of real dataset to use (0-100)
        model_name: Model name for tokenizer (required for real dataset)
        rank: Process rank for distributed sampling
        world_size: Total number of processes
        is_main_process: Whether this is the main process

    Returns:
        DataLoader with data
    """
    if dataset_name is not None:
        # Use real dataset
        if model_name is None:
            raise ValueError("model_name is required when using a real dataset")

        tokenizer = get_tokenizer(model_name)
        return load_real_dataset(
            dataset_name=dataset_name,
            dataset_percentage=dataset_percentage,
            tokenizer=tokenizer,
            seq_length=seq_length,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            is_main_process=is_main_process,
        )
    else:
        # Use synthetic dataset
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
