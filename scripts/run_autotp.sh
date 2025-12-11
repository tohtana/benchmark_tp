#!/bin/bash
# Run DeepSpeed AutoTP benchmark
#
# Usage:
#   ./scripts/run_autotp.sh                              # Uses default TP_SIZE=4, DP_SIZE=1
#   TP_SIZE=8 ./scripts/run_autotp.sh                    # Custom TP size (TP only)
#   DP_SIZE=2 TP_SIZE=4 ./scripts/run_autotp.sh          # 2D parallelism: 2 DP x 4 TP
#   ./scripts/run_autotp.sh --batch_size 2               # Pass additional args

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
DP_SIZE=${DP_SIZE:-1}
TP_SIZE=${TP_SIZE:-4}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-32B"}

# Calculate total GPUs needed
NPROC=$((DP_SIZE * TP_SIZE))

echo "Running DeepSpeed AutoTP benchmark"
echo "  DP_SIZE: $DP_SIZE"
echo "  TP_SIZE: $TP_SIZE"
echo "  Total GPUs: $NPROC"
echo "  MODEL: $MODEL_NAME"
echo ""

cd "$PROJECT_DIR"

torchrun --nproc_per_node=${NPROC} benchmark.py \
    --impl autotp \
    --model_name "$MODEL_NAME" \
    --dp_size ${DP_SIZE} \
    --tp_size ${TP_SIZE} \
    "$@"
