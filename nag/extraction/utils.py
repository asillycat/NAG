# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""GPU-side helpers for NAG extraction.

`to_bln` reshapes a `{layer_idx: (B, N)}` mapping returned by the patched models
into a contiguous `(B, L, N)` tensor. `topk_indices` then returns the indices of
the K highest-impact neurons per (sample, layer), which form the NAG of each
input (see Section 2 of the paper).
"""
from typing import Dict

import torch


def to_bln(scores_by_layer: Dict[int, torch.Tensor]) -> torch.Tensor:
    items = sorted(
        ((int(k), torch.as_tensor(v)) for k, v in scores_by_layer.items()),
        key=lambda x: x[0],
    )
    stacked = torch.stack([t for _, t in items], dim=0)  # (L, B, N)
    return stacked.transpose(0, 1).contiguous()           # (B, L, N)


@torch.no_grad()
def topk_indices(x_bln: torch.Tensor, top_k: int) -> torch.Tensor:
    b, l, n = x_bln.shape
    k = min(top_k, n)
    _, idx = torch.topk(x_bln, k=k, dim=2, largest=True, sorted=True)
    return idx.reshape(b, l * k)
