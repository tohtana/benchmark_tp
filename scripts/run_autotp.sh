#!/bin/bash
# Run DeepSpeed AutoTP benchmark
#
# Usage:
#   ./scripts/run_autotp.sh                    # Uses default TP_SIZE=4
#   TP_SIZE=8 ./scripts/run_autotp.sh          # Custom TP size
#   ./scripts/run_autotp.sh --batch_size 2     # Pass additional args

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
TP_SIZE=${TP_SIZE:-4}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}

echo "Running DeepSpeed AutoTP benchmark"
echo "  TP_SIZE: $TP_SIZE"
echo "  MODEL: $MODEL_NAME"
echo ""

cd "$PROJECT_DIR"

torchrun --nproc_per_node=${TP_SIZE} benchmark.py \
    --impl autotp \
    --model_name "$MODEL_NAME" \
    --tp_size ${TP_SIZE} \
    "$@"
