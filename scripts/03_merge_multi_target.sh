#!/bin/bash
# Build the multi-target mixture used in the paper (6 benchmarks, r_f/6 each).
#
# Prerequisites: 6 per-target selection parquet files produced by
# scripts/02_rank_single_target.sh with FRACTION=$(bc -l <<< 0.2/6).
#
# Usage: bash scripts/03_merge_multi_target.sh
set -euo pipefail

OUTPUT_PATH=${OUTPUT_PATH:-./outputs/selected/multi_target.parquet}

python -m nag.ranking.merge_multitarget \
    --input_paths \
        ./outputs/selected/arc_c.parquet \
        ./outputs/selected/hellaswag.parquet \
        ./outputs/selected/mmlu.parquet \
        ./outputs/selected/triviaqa.parquet \
        ./outputs/selected/xwinograd.parquet \
        ./outputs/selected/xstorycloze.parquet \
    --output_path "${OUTPUT_PATH}"
