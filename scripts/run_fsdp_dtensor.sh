#!/bin/bash
# Run FSDP + DTensor benchmark
#
# Usage:
#   ./scripts/run_fsdp_dtensor.sh                          # Uses defaults (DP=1, TP=4)
#   DP_SIZE=2 TP_SIZE=4 ./scripts/run_fsdp_dtensor.sh      # Custom DP and TP sizes
#   ./scripts/run_fsdp_dtensor.sh --batch_size 2           # Pass additional args

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
DP_SIZE=${DP_SIZE:-1}
TP_SIZE=${TP_SIZE:-4}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}

# Calculate total GPU count
TOTAL_GPUS=$((DP_SIZE * TP_SIZE))

echo "Running FSDP + DTensor benchmark"
echo "  DP_SIZE: $DP_SIZE"
echo "  TP_SIZE: $TP_SIZE"
echo "  TOTAL GPUs: $TOTAL_GPUS"
echo "  MODEL: $MODEL_NAME"
echo ""

cd "$PROJECT_DIR"

torchrun --nproc_per_node=${TOTAL_GPUS} benchmark.py \
    --impl fsdp_dtensor \
    --model_name "$MODEL_NAME" \
    --dp_size ${DP_SIZE} \
    --tp_size ${TP_SIZE} \
    "$@"
