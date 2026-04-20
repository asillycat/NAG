# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Single-target NAG ranking and selection (pandas backend).

Fits comfortably in memory for pools up to ~tens of millions of documents; for
web-scale corpora (e.g. the 150B-token RefinedWeb pool used in the paper) wrap
the same `build_target_profile` and `similarity_to_profile` calls in your own
distributed framework (PySpark, Ray, Dask).

Inputs
------
- `--target_features`: output of `nag.extraction.extract` for the target set
  (parquet/jsonl with `docid` and the feature column). A `dataset` column in
  the original payload is needed if you want to filter via `--target_filter`,
  in which case also pass `--target_payload`.
- `--pool_features`: output of `nag.extraction.extract` for the candidate pool.
- `--pool_payload`: the original pool parquet(s) with `docid`, `token_num`,
  and any columns you want to carry into the output (e.g. `doc`, `dataset`).
  Joined to `pool_features` on `docid`.

Output
------
Parquet shard containing the selected rows with an added `nag_distance` column.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

from nag.ranking.nag_similarity import (
    build_target_profile,
    distance_to_profile,
)


def _read_features(path_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"no files match {path_glob!r}")
    frames = []
    for f in files:
        if f.endswith(".parquet"):
            frames.append(pd.read_parquet(f))
        elif f.endswith((".json", ".jsonl")):
            frames.append(pd.read_json(f, lines=True))
        else:
            raise ValueError(f"unsupported file type: {f}")
    return pd.concat(frames, ignore_index=True)


def _join_features_and_payload(
    features_glob: str,
    payload_glob: Optional[str],
    feature_col: str,
) -> pd.DataFrame:
    feats = _read_features(features_glob)
    if payload_glob is None:
        return feats
    payload = _read_features(payload_glob)
    if "docid" not in feats.columns or "docid" not in payload.columns:
        raise KeyError("both features and payload must have a 'docid' column")
    feat_cols = ["docid", feature_col]
    payload_cols = [c for c in payload.columns if c != feature_col]
    return feats[feat_cols].merge(payload[payload_cols], on="docid", how="inner")


def _extract_feature_matrix(df: pd.DataFrame, feature_col: str) -> np.ndarray:
    def _pick(x):
        return x["layer_topk_value_index"] if isinstance(x, dict) else x
    rows = df[feature_col].apply(_pick).tolist()
    return np.asarray(rows, dtype=np.int64)


def _select_by_token_quota(
    df: pd.DataFrame,
    score_col: str,
    quota_fraction: float,
    token_col: str,
) -> pd.DataFrame:
    df_sorted = df.sort_values([score_col, "docid"], ascending=[True, True]).reset_index(drop=True)
    total_tokens = int(df_sorted[token_col].fillna(0).sum())
    if total_tokens <= 0:
        return df_sorted.iloc[:0]
    target = int(total_tokens * quota_fraction)
    cum = df_sorted[token_col].fillna(0).cumsum()
    mask = cum <= target
    if not mask.any():
        return df_sorted.iloc[:1]
    return df_sorted[mask]


def run(
    target_features: str,
    pool_features: str,
    pool_payload: str,
    output_path: str,
    feature_col: str,
    num_layers: int,
    top_k: int,
    fraction: float,
    token_col: str,
    target_payload: Optional[str] = None,
    target_filter: Optional[str] = None,
    target_filter_col: str = "dataset",
) -> None:
    target_df = _join_features_and_payload(target_features, target_payload, feature_col)
    if target_filter:
        if target_filter_col not in target_df.columns:
            raise KeyError(
                f"--target_filter_col={target_filter_col!r} not in target; "
                "pass --target_payload with a column of that name"
            )
        target_df = target_df[target_df[target_filter_col] == target_filter]
    if len(target_df) == 0:
        raise ValueError("empty target set after filtering")

    pool_df = _join_features_and_payload(pool_features, pool_payload, feature_col)
    if token_col not in pool_df.columns:
        raise KeyError(f"pool payload is missing token column {token_col!r}")

    target_features_mat = _extract_feature_matrix(target_df, feature_col)
    pool_features_mat = _extract_feature_matrix(pool_df, feature_col)

    print(f"target set: {len(target_df):,} rows")
    print(f"pool:       {len(pool_df):,} rows")

    profile = build_target_profile(target_features_mat, num_layers, top_k)
    pool_df = pool_df.copy()
    pool_df["nag_distance"] = distance_to_profile(
        pool_features_mat, profile, num_layers, top_k
    )

    selected = _select_by_token_quota(pool_df, "nag_distance", fraction, token_col)
    selected = selected.drop(columns=[feature_col], errors="ignore")

    print(f"selected:   {len(selected):,} rows "
          f"({int(selected[token_col].sum()):,} tokens, "
          f"dist range [{selected['nag_distance'].min():.4f}, "
          f"{selected['nag_distance'].max():.4f}])")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    selected.to_parquet(output_path, index=False)
    print(f"wrote {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_features", required=True,
                   help="glob to target NAG feature files (output of nag.extraction.extract)")
    p.add_argument("--pool_features", required=True,
                   help="glob to pool NAG feature files (output of nag.extraction.extract)")
    p.add_argument("--pool_payload", required=True,
                   help="glob to pool payload (original parquet with docid, token_num, doc, ...)")
    p.add_argument("--target_payload", default=None,
                   help="optional: payload parquet for the target set (needed for --target_filter)")
    p.add_argument("--output_path", required=True, help="parquet file to write")
    p.add_argument("--feature_col", default="fwd_up_feature",
                   help="column holding {layer_topk_value_index: [...]}")
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--fraction", type=float, default=0.2,
                   help="token-budget fraction r_f (paper default: 0.2)")
    p.add_argument("--token_col", default="token_num")
    p.add_argument("--target_filter", default=None,
                   help="optional: keep only rows where target_filter_col == this value")
    p.add_argument("--target_filter_col", default="dataset")
    args = p.parse_args()
    run(
        target_features=args.target_features,
        target_payload=args.target_payload,
        pool_features=args.pool_features,
        pool_payload=args.pool_payload,
        output_path=args.output_path,
        feature_col=args.feature_col,
        num_layers=args.num_layers,
        top_k=args.top_k,
        fraction=args.fraction,
        token_col=args.token_col,
        target_filter=args.target_filter,
        target_filter_col=args.target_filter_col,
    )


if __name__ == "__main__":
    main()
