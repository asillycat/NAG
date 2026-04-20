# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""t-SNE visualization of NAG features, colored by source dataset.

Reproduces the clustering figure in Section 4.1.2 of the paper. Reads a parquet
file containing a feature column (e.g. `fwd_up_feature`) whose value is a dict
`{"layer_topk_value_index": [L * K ints]}`, plus a `dataset` label column.

Run:
    python -m nag.analysis.tsne \
        --parquet path/to/target_nag_features.parquet \
        --feature_col fwd_up_feature \
        --num_layers 28 --top_k 20 \
        --out tsne_dataset.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd


def _as_py(x):
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    if hasattr(x, "as_py"):
        try:
            return x.as_py()
        except Exception:
            pass
    return x


def _extract_nested(cell, nested_key: str):
    cell = _as_py(cell)
    if not isinstance(cell, dict):
        raise TypeError(f"feature cell is not dict-like: type={type(cell)}")
    if nested_key in cell:
        return _as_py(cell[nested_key])
    raise KeyError(
        f"nested key {nested_key!r} not found in {list(cell.keys())}"
    )


def _flatten(feat, num_layers: int, top_k: int) -> np.ndarray:
    feat = _as_py(feat)
    if feat is None:
        raise ValueError("feature is None")
    if isinstance(feat, list):
        feat = [_as_py(a) for a in feat]
    arr = np.asarray(feat, dtype=np.int32)
    if arr.ndim == 2:
        if arr.shape == (top_k, num_layers):
            arr = arr.T
        return arr.reshape(-1)
    if arr.ndim == 1:
        return arr.reshape(-1)
    raise ValueError(f"unexpected feature ndim={arr.ndim}")


def run(
    parquet: str,
    feature_col: str,
    label_col: str,
    num_layers: int,
    top_k: int,
    nested_key: str,
    target_datasets: Optional[List[str]],
    per_dataset: int,
    seed: int,
    perplexity: float,
    max_iter: int,
    use_pca: bool,
    pca_dim: int,
    standardize: bool,
    out: str,
) -> None:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not os.path.exists(parquet):
        print(f"[ERROR] parquet not found: {parquet}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet)
    for col in (feature_col, label_col):
        if col not in df.columns:
            print(f"[ERROR] missing column {col!r}. available: {list(df.columns)}",
                  file=sys.stderr)
            sys.exit(1)

    if target_datasets:
        df = df[df[label_col].isin(target_datasets)].copy()
        if df.empty:
            print(
                f"[ERROR] no rows after filtering to {target_datasets}.",
                file=sys.stderr,
            )
            sys.exit(1)

    per = max(1, int(per_dataset))

    def _sample(group):
        return group if len(group) <= per else group.sample(n=per, random_state=seed)

    df = df.groupby(label_col, group_keys=False).apply(_sample).reset_index(drop=True)
    print("[INFO] per-dataset counts:")
    print(df[label_col].value_counts().sort_index())

    feats, labels = [], []
    bad = 0
    for i, cell in enumerate(df[feature_col].values):
        try:
            nested = _extract_nested(cell, nested_key)
            feats.append(_flatten(nested, num_layers, top_k))
            labels.append(str(df[label_col].iloc[i]))
        except Exception:
            bad += 1

    if len(feats) < 10:
        print(f"[ERROR] too few valid rows ({len(feats)}), bad={bad}", file=sys.stderr)
        sys.exit(1)

    x = np.stack(feats, axis=0).astype(np.float32)
    labels = np.asarray(labels, dtype=str)
    print(f"[INFO] X shape: {x.shape}, bad rows skipped: {bad}")

    if standardize:
        x = StandardScaler().fit_transform(x)
    if use_pca:
        d = min(int(pca_dim), x.shape[1], x.shape[0] - 1)
        x = PCA(n_components=d, random_state=seed).fit_transform(x)
        print(f"[INFO] PCA -> {d} dims before t-SNE")

    plex = min(perplexity, max(5.0, (x.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=plex,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=max_iter,
        verbose=1,
    )
    x2d = tsne.fit_transform(x)

    order = target_datasets if target_datasets else sorted(set(labels))
    order = [d for d in order if d in set(labels)]
    palette = sns.color_palette("tab10", len(order))
    colors = {lab: palette[i] for i, lab in enumerate(order)}

    plt.figure(figsize=(9, 7))
    for lab in order:
        mask = labels == lab
        plt.scatter(
            x2d[mask, 0], x2d[mask, 1],
            s=18, alpha=0.75,
            label=f"{lab} (n={mask.sum()})",
            color=colors[lab],
        )
    plt.legend(markerscale=1.4, fontsize=10, frameon=True)
    plt.title(f"t-SNE of {feature_col}.{nested_key} (colored by {label_col})")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out, dpi=200)
    print(f"[INFO] saved {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, help="input parquet with features and labels")
    p.add_argument("--feature_col", default="fwd_up_feature")
    p.add_argument("--nested_key", default="layer_topk_value_index")
    p.add_argument("--label_col", default="dataset")
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--top_k", type=int, required=True)
    p.add_argument("--target_datasets", nargs="+", default=None,
                   help="optional whitelist of dataset names to plot")
    p.add_argument("--per_dataset", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--max_iter", type=int, default=3000)
    p.add_argument("--use_pca", action="store_true")
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--out", default="tsne_dataset.png")
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
