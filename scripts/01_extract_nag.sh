#!/bin/bash
# Extract NAG features from a pool of documents.
#
# Multi-GPU on a single node via `accelerate launch`. For multi-node, consult
# https://huggingface.co/docs/accelerate for the correct rendezvous flags.
#
# Usage: bash scripts/01_extract_nag.sh
set -euo pipefail

MODEL_TYPE=${MODEL_TYPE:-qwen3-1.7b-base}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B-Base}
IN_DATA=${IN_DATA:-./data/pool}
OUT_DATA=${OUT_DATA:-./outputs/nag_features/${MODEL_TYPE}}
FORMAT=${FORMAT:-web}
MAX_LENGTH=${MAX_LENGTH:-120}
BATCH_SIZE=${BATCH_SIZE:-8}
TOP_K=${TOP_K:-20}
NEURON_TYPES=${NEURON_TYPES:-fwd_up}
NUM_GPUS=${NUM_GPUS:-1}

accelerate launch \
    --num_processes ${NUM_GPUS} \
    -m nag.extraction.extract \
    --model_type ${MODEL_TYPE} \
    --model_path ${MODEL_PATH} \
    --in_data_path ${IN_DATA} \
    --out_data_path ${OUT_DATA} \
    --format ${FORMAT} \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --top_k ${TOP_K} \
    --neuron_types ${NEURON_TYPES}
