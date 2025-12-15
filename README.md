# TP Benchmarking Tool

Benchmark and compare tensor parallelism implementations for large language models.

## Supported Implementations

1. **DeepSpeed AutoTP** - Uses DeepSpeed's automatic tensor parallelism via `deepspeed.tp_model_init()` with vocabulary-parallel embeddings
2. **FSDP2 + DTensor** - Uses PyTorch's 2D device mesh with FSDP2 (`fully_shard`) for data parallelism and DTensor for tensor parallelism

## Benchmark Comparison with TorchTitan

**Versions tested**: PyTorch 2.9.0, DeepSpeed 0.18.3, TorchTitan v0.2.0

### Commands

```bash
# DeepSpeed AutoTP
torchrun --nproc_per_node=8 benchmark.py \
    --impl autotp \
    --tp_size 8 \
    --model_name Qwen/Qwen3-32B \
    --batch_size 1 \
    --seq_length 2048 \
    --dtype bfloat16 \
    --autocast \
    --attn_impl flash_attention_2 \
    --activation_checkpointing \
    --num_training_steps 100 \
    --warmup_steps 5 \
    --log_interval 10

# FSDP + DTensor
torchrun --nproc_per_node=8 benchmark.py \
    --impl fsdp_dtensor \
    --tp_size 8 \
    --dp_size 1 \
    --model_name Qwen/Qwen3-32B \
    --batch_size 1 \
    --seq_length 2048 \
    --dtype bfloat16 \
    --autocast \
    --attn_impl flash_attention_2 \
    --activation_checkpointing \
    --num_training_steps 100 \
    --warmup_steps 5 \
    --log_interval 10

# TorchTitan (run from torchtitan directory)
# First, download the tokenizer:
python torchtitan/datasets/download_tokenizer.py --repo_id Qwen/Qwen3-32B --tokenizer_path /tmp/qwen3_assets/Qwen3-32B

# Then run the benchmark:
cd /path/to/torchtitan
export PYTORCH_ALLOC_CONF="expandable_segments:True"
torchrun --nproc_per_node=8 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --job.config_file /path/to/qwen3_32b_benchmark.toml
```

TorchTitan config (`qwen3_32b_benchmark.toml`):
```toml
[job]
dump_folder = "/tmp/torchtitan_outputs"

[metrics]
log_freq = 10

[model]
name = "qwen3"
flavor = "32B"
hf_assets_path = "/tmp/qwen3_assets/Qwen3-32B"

[optimizer]
name = "AdamW"
lr = 1e-5

[lr_scheduler]
warmup_steps = 5

[training]
local_batch_size = 1
seq_len = 2048
steps = 100
dataset = "c4"
dtype = "bfloat16"

[parallelism]
data_parallel_replicate_degree = 1
data_parallel_shard_degree = 1
tensor_parallel_degree = 8

[activation_checkpoint]
mode = "full"

[checkpoint]
enable = false
```

### Results (8xH100, Qwen3-32B, TP=8, BS=1, seq=2048, BF16+autocast)

PyTorch 2.9.0 Results (8xH100, Qwen3-32B, TP=8, BS=1, seq=2048, BF16+autocast)

| Implementation   | Throughput        | Avg Step Time | Peak Memory |
|------------------|-------------------|---------------|-------------|
| DeepSpeed AutoTP | 2,479 tokens/sec  | 829 ms        | 49.2 GB     |
| FSDP + DTensor   | 1,513 tokens/sec  | 1,354 ms      | 34.6 GB     |
| TorchTitan       | ~1,000 tokens/sec | ~2,028 ms     | 32.8 GB     |

## Installation

### Dependencies

```bash
pip install torch transformers deepspeed
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Quick Start

### Comparing Implementations (Matching Settings)

Both implementations support DP+TP 2D parallelism. Here's how to run them with equivalent settings:

```bash
# AutoTP: 8 GPUs with 2 DP x 4 TP
torchrun --nproc_per_node=8 benchmark.py \
    --impl autotp \
    --model_name Qwen/Qwen3-32B \
    --dp_size 2 \
    --tp_size 4 \
    --batch_size 1 \
    --seq_length 2048 \
    --num_training_steps 100

# FSDP+DTensor: 8 GPUs with 2 DP x 4 TP
torchrun --nproc_per_node=8 benchmark.py \
    --impl fsdp_dtensor \
    --model_name Qwen/Qwen3-32B \
    --dp_size 2 \
    --tp_size 4 \
    --batch_size 1 \
    --seq_length 2048 \
    --num_training_steps 100
```

Or use the convenience scripts:

```bash
# AutoTP
DP_SIZE=2 TP_SIZE=4 ./scripts/run_autotp.sh

# FSDP+DTensor
DP_SIZE=2 TP_SIZE=4 ./scripts/run_fsdp_dtensor.sh
```

### TP-Only Mode (No Data Parallelism)

```bash
# AutoTP: 4 GPUs, TP only
torchrun --nproc_per_node=4 benchmark.py \
    --impl autotp \
    --tp_size 4

# FSDP+DTensor: 4 GPUs, TP only
torchrun --nproc_per_node=4 benchmark.py \
    --impl fsdp_dtensor \
    --dp_size 1 \
    --tp_size 4
```

## Configuration Options

### Model Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-32B` | HuggingFace model name or local path |
| `--num_layers` | `0` | Override number of layers (0 = use model default) |
| `--attn_impl` | `sdpa` | Attention implementation (`sdpa`, `flash_attention_2`, `eager`) |

### Parallelism Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--impl` | `autotp` | TP implementation (`autotp` or `fsdp_dtensor`) |
| `--tp_size` | auto | Tensor parallel degree |
| `--dp_size` | `1` | Data parallel degree (both implementations) |
| `--zero_stage` | `1` | DeepSpeed ZeRO stage (AutoTP only, 0-2) |

Both implementations require `dp_size * tp_size == world_size`.

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | `1` | Per-GPU micro batch size |
| `--seq_length` | `2048` | Sequence length |
| `--num_training_steps` | `100` | Total training steps |
| `--learning_rate` | `1e-5` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation steps |
| `--max_grad_norm` | `1.0` | Maximum gradient norm |

### Memory Optimization

| Argument | Default | Description |
|----------|---------|-------------|
| `--dtype` | `bfloat16` | Training dtype (`bfloat16`, `float16`, `float32`) |
| `--autocast` | `false` | Enable torch.autocast for mixed precision (uses `--dtype`) |
| `--activation_checkpointing` | `false` | Enable activation checkpointing |

### Benchmarking Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--warmup_steps` | `5` | Steps to skip in metrics collection |
| `--log_interval` | `10` | Logging interval |
| `--output_dir` | `./results` | Output directory for results |
| `--seed` | `42` | Random seed |

## Output Format

Results are saved as JSON files in `{output_dir}/results_{impl}_tp{tp_size}_dp{dp_size}.json`:

```json
{
  "config": {
    "model_name": "Qwen/Qwen3-32B",
    "impl": "autotp",
    "tp_size": 4,
    "dp_size": 1,
    "batch_size": 1,
    "seq_length": 2048,
    "autocast": false,
    ...
  },
  "steps": [
    {
      "step": 5,
      "loss": 10.234,
      "step_time_s": 0.85,
      "forward_time_s": 0.32,
      "backward_time_s": 0.45,
      "optimizer_time_s": 0.08,
      "tokens_per_sec": 2409.4,
      "samples_per_sec": 1.18,
      "memory_allocated_gb": 42.5,
      "memory_peak_gb": 45.2
    },
    ...
  ],
  "summary": {
    "num_steps": 95,
    "avg_step_time_ms": 850.2,
    "std_step_time_ms": 12.3,
    "avg_tokens_per_sec": 2409.4,
    "avg_samples_per_sec": 1.18,
    "avg_loss": 8.123,
    "final_loss": 6.543,
    "peak_memory_gb": 45.2,
    "timing_breakdown": {
      "forward_pct": 37.6,
      "backward_pct": 52.9,
      "optimizer_pct": 9.4,
      "other_pct": 0.1
    }
  }
}
```

## Adding New Models

To add support for a new model architecture:

1. Create a new file in `benchmark/models/` (e.g., `llama.py`)

2. Implement the `BaseModelBuilder` interface:

```python
from benchmark.models.base import BaseModelBuilder, ModelConfig
from benchmark.models.registry import register_model

@register_model("llama")
class LlamaModelBuilder(BaseModelBuilder):
    def get_config(self) -> ModelConfig:
        """Return model configuration."""
        # Load HuggingFace config and extract dimensions
        ...

    def create_model(self, device, dtype, num_layers=0, attn_impl="sdpa"):
        """Create model with random initialization."""
        # Use AutoModelForCausalLM.from_config()
        ...

    def get_tp_mapping(self):
        """Return DTensor tensor parallel mapping."""
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
        return {
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }

    def get_transformer_layers(self, model):
        """Return the transformer layers module list."""
        return model.model.layers

    def get_embedding_module(self, model):
        """Return the embedding module."""
        return model.model.embed_tokens

    def replace_embedding_module(self, model, new_embedding):
        """Replace embedding with VocabParallelEmbedding."""
        model.model.embed_tokens = new_embedding
```

3. Import the new model in `benchmark/models/__init__.py`:

```python
from . import llama  # This registers the model via @register_model decorator
```

4. Run with your new model:

```bash
torchrun --nproc_per_node=4 benchmark.py --model_name meta-llama/Llama-3-8B ...
```

## Understanding the 2D Device Mesh

Both implementations support 2D parallelism (DP+TP). When using `dp_size > 1`, the GPUs are organized in a 2D mesh:

```
For 8 GPUs with dp_size=2, tp_size=4:

        TP dimension (tensor parallel)
        ----------------------->
   DP  | GPU0  GPU1  GPU2  GPU3 |  <- DP replica 0
   dim | GPU4  GPU5  GPU6  GPU7 |  <- DP replica 1
    |
    v

- TP groups (same row): {0,1,2,3}, {4,5,6,7}
  - Model weights are sharded across these groups
  - Each group processes the same data

- DP groups (same column): {0,4}, {1,5}, {2,6}, {3,7}
  - Optimizer states/gradients are reduced across these groups
  - Each group processes different data batches
```

### Implementation Differences

| Aspect | AutoTP | FSDP+DTensor |
|--------|--------|--------------|
| TP mechanism | `deepspeed.tp_model_init()` | DTensor `parallelize_module()` |
| DP mechanism | DeepSpeed engine with MPU | FSDP2 `fully_shard()` |
| Optimizer | DeepSpeed ZeRO (stage 0-2) | PyTorch AdamW |
| Memory optimization | ZeRO sharding | FSDP sharding |
| Vocab parallelism | VocabParallelEmbedding | DTensor loss_parallel |
| Autocast | DeepSpeed `torch_autocast` config | `torch.autocast()` context |

## Vocabulary Parallelism

Both implementations automatically partition vocabulary embeddings and lm_head for correct and efficient training.

### AutoTP (Megatron-style)

1. **Partitions embeddings** - Each TP rank stores `vocab_size / tp_size` embeddings
2. **Shards lm_head** - Output projection weights are partitioned to match
3. **Uses vocab-parallel loss** - Cross-entropy computed correctly over partitioned logits

```
Example: vocab_size=151936, tp_size=4
  Rank 0: vocab [0, 37984)      - 37,984 embeddings
  Rank 1: vocab [37984, 75968)  - 37,984 embeddings
  Rank 2: vocab [75968, 113952) - 37,984 embeddings
  Rank 3: vocab [113952, 151936) - 37,984 embeddings
```

The `VocabParallelEmbedding` uses all-reduce to combine embedding outputs from all ranks, and `vocab_parallel_causal_cross_entropy` computes loss correctly when logits are split across ranks.

### FSDP+DTensor (TorchTitan-style)

1. **Embedding**: `RowwiseParallel` - embeddings sharded by hidden dimension
2. **lm_head**: `ColwiseParallel` with `Shard(-1)` - logits sharded by vocabulary dimension
3. **Loss**: `loss_parallel()` context handles cross-entropy with sharded logits

This uses PyTorch's native DTensor infrastructure for automatic gradient handling.

## Architecture

```
benchmark_tp/
├── benchmark.py              # Main entry point
├── benchmark/
│   ├── args.py              # CLI argument parsing
│   ├── data.py              # Synthetic data generation
│   ├── distributed.py       # torch.distributed utilities
│   ├── metrics.py           # Metrics collection
│   ├── models/
│   │   ├── base.py          # BaseModelBuilder abstract class
│   │   ├── registry.py      # Model registry
│   │   └── qwen.py          # Qwen model implementation
│   └── parallel/
│       ├── base.py          # BaseTPStrategy abstract class
│       ├── factory.py       # Strategy factory
│       ├── autotp.py        # DeepSpeed AutoTP strategy
│       ├── fsdp_dtensor.py  # FSDP+DTensor strategy
│       └── vocab_parallel.py # VocabParallelEmbedding
└── scripts/
    ├── run_autotp.sh        # AutoTP convenience script
    └── run_fsdp_dtensor.sh  # FSDP+DTensor convenience script
```

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce `--batch_size`, enable `--activation_checkpointing`, or use smaller `--num_layers`

2. **TP size validation error**: Ensure `num_key_value_heads % tp_size == 0`

3. **FSDP+DTensor: dp_size * tp_size != world_size**: Ensure total GPUs equals `dp_size * tp_size`

4. **DeepSpeed initialization errors**: Check DeepSpeed version compatibility with your PyTorch version

### Debug Tips

```bash
# Reduce model size for testing
torchrun --nproc_per_node=2 benchmark.py \
    --impl autotp \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --num_layers 4 \
    --num_training_steps 10

# Enable verbose output
TORCH_DISTRIBUTED_DEBUG=INFO torchrun ...
```
