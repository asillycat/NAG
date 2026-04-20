#!/bin/bash
# Rank pool documents by NAG similarity to a single target set, select top r_f.
#
# Paper default: top_k=20 per layer, r_f=0.2 (20% of pool tokens).
# num_layers must match the backbone: qwen3-1.7b=28, llama3.2-3b=28, smollm3-3b=36.
#
# Usage: bash scripts/02_rank_single_target.sh
set -euo pipefail

TARGET_FEATURES=${TARGET_FEATURES:-./outputs/nag_features/qwen3-1.7b-base/target/*.jsonl}
TARGET_PAYLOAD=${TARGET_PAYLOAD:-./data/target/*.parquet}
POOL_FEATURES=${POOL_FEATURES:-./outputs/nag_features/qwen3-1.7b-base/pool/*.jsonl}
POOL_PAYLOAD=${POOL_PAYLOAD:-./data/pool/*.parquet}
OUTPUT_PATH=${OUTPUT_PATH:-./outputs/selected/hellaswag_single.parquet}
FEATURE_COL=${FEATURE_COL:-fwd_up_feature}
NUM_LAYERS=${NUM_LAYERS:-28}
TOP_K=${TOP_K:-20}
FRACTION=${FRACTION:-0.2}
TOKEN_COL=${TOKEN_COL:-token_num}
TARGET_FILTER=${TARGET_FILTER:-hellaswag_train}

python -m nag.ranking.rank \
    --target_features "${TARGET_FEATURES}" \
    --target_payload "${TARGET_PAYLOAD}" \
    --pool_features "${POOL_FEATURES}" \
    --pool_payload "${POOL_PAYLOAD}" \
    --output_path "${OUTPUT_PATH}" \
    --feature_col ${FEATURE_COL} \
    --num_layers ${NUM_LAYERS} \
    --top_k ${TOP_K} \
    --fraction ${FRACTION} \
    --token_col ${TOKEN_COL} \
    --target_filter ${TARGET_FILTER}
