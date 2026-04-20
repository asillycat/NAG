# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Build and apply NAG-based neuron deactivation (paper Section 4.1.1).

Given a target set's NAG features, pick the most frequently activated neurons
per layer and zero the corresponding rows of `up_proj.weight` on a loaded
HuggingFace LLM. The patched model can then be evaluated with any standard
harness (e.g. lm-eval-harness) to reproduce Table "deactivation_sub".

Pipeline:
    1. Extract NAG features for the target set (`nag.extraction.extract`).
    2. `python -m nag.analysis.deactivation build \
            --target_features ... --out keys.json --num_layers 28 \
            --top_k_extraction 20`
    3. In your eval driver:
            from transformers import AutoModelForCausalLM
            from nag.analysis.deactivation import apply_deactivation, load_keys
            model = AutoModelForCausalLM.from_pretrained(...)
            apply_deactivation(model, load_keys("keys.json"))
            # pass `model` to lm-eval-harness
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch


def build_deactivation_keys(
    target_features: np.ndarray,
    num_layers: int,
    top_k_extraction: int,
    top_k_deactivate: int,
) -> Dict[str, List[int]]:
    """Pick the top-K most frequent neurons per layer across the target set.

    `target_features` has shape `(n, num_layers * top_k_extraction)` — the
    output of `nag.extraction.extract`. `top_k_deactivate` is usually equal to
    `top_k_extraction` (the paper's default).
    """
    arr = np.asarray(target_features).reshape(-1, num_layers, top_k_extraction)
    keys: Dict[str, List[int]] = {}
    for layer in range(num_layers):
        indices = arr[:, layer, :].reshape(-1)
        unique, counts = np.unique(indices, return_counts=True)
        order = np.argsort(-counts)
        picked = unique[order[:top_k_deactivate]]
        keys[str(layer)] = picked.astype(int).tolist()
    return keys


def load_keys(path: str) -> Dict[int, List[int]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): list(v) for k, v in raw.items()}


@torch.no_grad()
def apply_deactivation(
    model: torch.nn.Module,
    deactivation_keys: Dict[int, List[int]],
    mlp_attr: str = "mlp",
    up_proj_attr: str = "up_proj",
) -> int:
    """Zero rows of `up_proj.weight` in-place on `model`.

    Returns the total number of neurons deactivated. Compatible with any
    causal LM exposing `model.model.layers[i].mlp.up_proj` (Qwen3, Llama,
    SmolLM3 all follow this layout).
    """
    inner = model.model if hasattr(model, "model") else model
    layers = inner.layers
    total = 0
    for layer_idx, neuron_ids in deactivation_keys.items():
        layer = layers[int(layer_idx)]
        up_proj = getattr(getattr(layer, mlp_attr), up_proj_attr)
        idx = torch.as_tensor(neuron_ids, dtype=torch.long, device=up_proj.weight.device)
        up_proj.weight.index_fill_(0, idx, 0.0)
        total += int(idx.numel())
    return total


def _build_cli():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compute deactivation keys from target NAG features")
    b.add_argument("--target_features", required=True,
                   help="glob to target NAG feature files (output of nag.extraction.extract)")
    b.add_argument("--target_payload", default=None,
                   help="optional: payload parquet for the target set (needed for --target_filter)")
    b.add_argument("--out", required=True, help="output JSON path")
    b.add_argument("--feature_col", default="fwd_up_feature")
    b.add_argument("--num_layers", type=int, required=True)
    b.add_argument("--top_k_extraction", type=int, required=True,
                   help="K used during NAG extraction")
    b.add_argument("--top_k_deactivate", type=int, default=None,
                   help="neurons to zero per layer (default: same as top_k_extraction)")
    b.add_argument("--target_filter", default=None)
    b.add_argument("--target_filter_col", default="dataset")
    return p


def _cmd_build(args):
    from nag.ranking.rank import _extract_feature_matrix, _join_features_and_payload
    df = _join_features_and_payload(
        args.target_features, args.target_payload, args.feature_col
    )
    if args.target_filter:
        if args.target_filter_col not in df.columns:
            raise KeyError(
                f"--target_filter_col={args.target_filter_col!r} not in target; "
                "pass --target_payload with a column of that name"
            )
        df = df[df[args.target_filter_col] == args.target_filter]
    if len(df) == 0:
        raise ValueError("empty target set after filtering")
    feats = _extract_feature_matrix(df, args.feature_col)

    top_deact = args.top_k_deactivate or args.top_k_extraction
    keys = build_deactivation_keys(
        feats,
        num_layers=args.num_layers,
        top_k_extraction=args.top_k_extraction,
        top_k_deactivate=top_deact,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(keys, f, indent=2)
    total = sum(len(v) for v in keys.values())
    print(f"wrote {args.out}: {len(keys)} layers, {total} neurons to deactivate")


def main():
    parser = _build_cli()
    args = parser.parse_args()
    if args.cmd == "build":
        _cmd_build(args)
    else:
        parser.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
