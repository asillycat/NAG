#!/bin/bash
# End-to-end demo: extract NAG features on a tiny dataset and rank the pool
# against the target. First run downloads Qwen3-0.6B-Base (~1.2 GB) from the
# HuggingFace Hub.
#
# Expected runtime: ~1 minute on a single GPU, ~5 minutes on CPU.
#
# Usage:  bash examples/demo_extract_and_rank.sh
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_TYPE=qwen3-0.6b-base
MODEL_PATH=Qwen/Qwen3-0.6B-Base
NUM_LAYERS=28
TOP_K=10
BATCH_SIZE=2
MAX_LENGTH=128

DATA=examples/data
OUT=examples/outputs

echo "==> [1/4] generate tiny sample dataset"
python -m examples.make_sample_data

echo "==> [2/4] extract NAG features for the target set"
python -m nag.extraction.extract \
    --model_type ${MODEL_TYPE} --model_path ${MODEL_PATH} \
    --in_data_path ${DATA}/target \
    --out_data_path ${OUT}/features/target \
    --format web --max_length ${MAX_LENGTH} --batch_size ${BATCH_SIZE} \
    --top_k ${TOP_K} --neuron_types fwd_up

echo "==> [3/4] extract NAG features for the candidate pool"
python -m nag.extraction.extract \
    --model_type ${MODEL_TYPE} --model_path ${MODEL_PATH} \
    --in_data_path ${DATA}/pool \
    --out_data_path ${OUT}/features/pool \
    --format web --max_length ${MAX_LENGTH} --batch_size ${BATCH_SIZE} \
    --top_k ${TOP_K} --neuron_types fwd_up

echo "==> [4/4] rank the pool against the target (keep 50% by tokens)"
python -m nag.ranking.rank \
    --target_features "${OUT}/features/target/*.jsonl" \
    --target_payload  "${DATA}/target/*.parquet" \
    --pool_features   "${OUT}/features/pool/*.jsonl" \
    --pool_payload    "${DATA}/pool/*.parquet" \
    --output_path     "${OUT}/selected.parquet" \
    --feature_col fwd_up_feature \
    --num_layers ${NUM_LAYERS} --top_k ${TOP_K} \
    --fraction 0.5

echo
echo "==> selected rows (smallest nag_distance first):"
python -c "
import pandas as pd
df = pd.read_parquet('${OUT}/selected.parquet').sort_values('nag_distance')
pd.set_option('display.max_colwidth', 80)
print(df[['docid', 'dataset', 'nag_distance']].to_string(index=False))
"
