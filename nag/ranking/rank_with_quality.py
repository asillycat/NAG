# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Rank pool documents by NAG distance combined with an external quality score.

Reproduces the combined-signal experiment in the paper (NAG + FineWeb-Edu), and
selects the top-r_f fraction by a min-max-normalized sum of the two signals.

Sign convention
---------------
Both signals are treated as "lower is better" by default. `nag_distance` is
naturally lower-is-better. If your quality column is higher-is-better (e.g. a
FineWeb-Edu educational probability), pass `--invert_quality` so it is
converted to `1 - qual_norm` before summation.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from nag.ranking.nag_similarity import (
    build_target_profile,
    distance_to_profile,
)
from nag.ranking.rank import (
    _extract_feature_matrix,
    _join_features_and_payload,
    _select_by_token_quota,
)


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


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
    quality_col: str,
    invert_quality: bool,
    target_payload: Optional[str],
    target_filter: Optional[str],
    target_filter_col: str,
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

    pool_df = _join_features_and_payload(pool_features, pool_payload, feature_col).copy()
    for col in (token_col, quality_col):
        if col not in pool_df.columns:
            raise KeyError(f"pool payload is missing column {col!r}")
    pool_df = pool_df[pool_df[quality_col].notna()].reset_index(drop=True)
    pool_df[quality_col] = pool_df[quality_col].astype(float)

    target_features_mat = _extract_feature_matrix(target_df, feature_col)
    pool_features_mat = _extract_feature_matrix(pool_df, feature_col)

    profile = build_target_profile(target_features_mat, num_layers, top_k)
    pool_df["nag_distance"] = distance_to_profile(
        pool_features_mat, profile, num_layers, top_k
    )

    dist_norm = _minmax(pool_df["nag_distance"])
    qual_norm = _minmax(pool_df[quality_col])
    if invert_quality:
        qual_norm = 1.0 - qual_norm
    pool_df["combined_score"] = dist_norm + qual_norm

    selected = _select_by_token_quota(pool_df, "combined_score", fraction, token_col)
    selected = selected.drop(columns=[feature_col], errors="ignore")

    print(f"selected: {len(selected):,} rows ({int(selected[token_col].sum()):,} tokens)")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    selected.to_parquet(output_path, index=False)
    print(f"wrote {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target_features", required=True)
    p.add_argument("--pool_features", required=True)
    p.add_argument("--pool_payload", required=True,
                   help="glob to pool payload (parquet with docid, token_num, quality_col, ...)")
    p.add_argument("--target_payload", default=None)
    p.add_argument("--output_path", required=True)
    p.add_argument("--feature_col", default="fwd_up_feature")
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--fraction", type=float, default=0.2)
    p.add_argument("--token_col", default="token_num")
    p.add_argument("--quality_col", required=True,
                   help="column holding the auxiliary quality score (e.g. finewebedu_score)")
    p.add_argument("--invert_quality", action="store_true",
                   help="set if higher quality_col value means higher-quality (e.g. FineWeb-Edu probability)")
    p.add_argument("--target_filter", default=None)
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
        quality_col=args.quality_col,
        invert_quality=args.invert_quality,
        target_filter=args.target_filter,
        target_filter_col=args.target_filter_col,
    )


if __name__ == "__main__":
    main()
