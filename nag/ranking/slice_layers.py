# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Post-process NAG features to keep only a subset of layers.

Used to reproduce the "all layers vs final layer" study in the paper: after
extracting NAG features with all L layers, slice each record down to a chosen
layer subset and feed the sliced features into the standard ranker.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd


def _parse_layer_spec(spec: str, num_layers: int) -> List[int]:
    if spec == "last":
        return [num_layers - 1]
    if spec == "first":
        return [0]
    indices = sorted({int(x) for x in spec.split(",") if x.strip()})
    for i in indices:
        if not 0 <= i < num_layers:
            raise ValueError(f"layer index {i} out of range [0, {num_layers})")
    return indices


def slice_record(feature: dict, keep_layers: List[int], num_layers: int, top_k: int) -> dict:
    arr = np.asarray(feature["layer_topk_value_index"], dtype=np.int64)
    assert arr.shape[0] == num_layers * top_k, (
        f"expected vector of length {num_layers * top_k}, got {arr.shape[0]}"
    )
    arr = arr.reshape(num_layers, top_k)
    return {"layer_topk_value_index": arr[keep_layers].reshape(-1).tolist()}


def run(
    input_glob: str,
    output_path: str,
    feature_col: str,
    num_layers: int,
    top_k: int,
    layer_spec: str,
) -> None:
    keep = _parse_layer_spec(layer_spec, num_layers)
    print(f"keeping {len(keep)} of {num_layers} layers: {keep}")

    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"no files match {input_glob!r}")

    frames = []
    for f in files:
        if f.endswith(".parquet"):
            frames.append(pd.read_parquet(f))
        elif f.endswith((".json", ".jsonl")):
            frames.append(pd.read_json(f, lines=True))
        else:
            raise ValueError(f"unsupported file type: {f}")
    df = pd.concat(frames, ignore_index=True)

    df[feature_col] = df[feature_col].apply(
        lambda x: slice_record(x, keep, num_layers, top_k)
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"wrote {output_path}: {len(df):,} rows")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_glob", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--feature_col", default="fwd_up_feature")
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--layers", default="last",
                   help='"last" | "first" | comma-separated 0-indexed layer list')
    args = p.parse_args()
    run(
        input_glob=args.input_glob,
        output_path=args.output_path,
        feature_col=args.feature_col,
        num_layers=args.num_layers,
        top_k=args.top_k,
        layer_spec=args.layers,
    )


if __name__ == "__main__":
    main()
