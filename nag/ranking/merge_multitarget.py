# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Merge per-target NAG selections into a single multi-target corpus.

Implements the simple equal-budget mixture described in the paper: each target
independently selects r_f/M of the pool, and the resulting subsets are
concatenated (with possible duplicates) to form the multi-target training set.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def run(input_paths: list[str], output_path: str, deduplicate: bool) -> None:
    frames = []
    for p in input_paths:
        df = pd.read_parquet(p)
        print(f"read {p}: {len(df):,} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    if deduplicate:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["docid"], keep="first").reset_index(drop=True)
        print(f"deduplicated {before:,} -> {len(merged):,}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"wrote {output_path}: {len(merged):,} rows")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_paths", nargs="+", required=True,
                   help="parquet files produced by rank.py (one per target)")
    p.add_argument("--output_path", required=True)
    p.add_argument("--deduplicate", action="store_true",
                   help="deduplicate on docid (default keeps duplicates as in the paper)")
    args = p.parse_args()
    run(args.input_paths, args.output_path, args.deduplicate)


if __name__ == "__main__":
    main()
