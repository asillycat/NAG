# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Core numpy routines for NAG-based similarity and ranking.

Given a set of NAG feature arrays with shape `(n, L * K)` (L layers, K neurons
per layer, flattened in layer-major order), these helpers build a layer-wise
target profile and score each candidate against it. See Section 2.4 of the
paper for the exact definitions.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def reshape_as_layers(features: np.ndarray, num_layers: int, top_k: int) -> np.ndarray:
    features = np.asarray(features)
    assert features.ndim == 2 and features.shape[1] == num_layers * top_k, (
        f"expected (n, {num_layers * top_k}), got {features.shape}"
    )
    return features.reshape(features.shape[0], num_layers, top_k)


def build_target_profile(
    target_features: np.ndarray,
    num_layers: int,
    top_k: int,
    power: float = 1.0,
) -> List[Dict[int, float]]:
    """Aggregate per-layer neuron activation frequencies across a target set.

    Returns a list of length L; each element maps `neuron_index -> weight`, with
    weights normalized to sum to 1 per layer.
    """
    layered = reshape_as_layers(target_features, num_layers, top_k)  # (n, L, K)
    profile: List[Dict[int, float]] = []
    for layer in range(num_layers):
        indices = layered[:, layer, :].reshape(-1)
        unique, counts = np.unique(indices, return_counts=True)
        weights = counts.astype(np.float64)
        if power != 1.0:
            weights = np.power(weights, power)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        profile.append(dict(zip(unique.tolist(), weights.tolist())))
    return profile


def similarity_to_profile(
    features: np.ndarray,
    profile: List[Dict[int, float]],
    num_layers: int,
    top_k: int,
) -> np.ndarray:
    """Vectorized scoring of a batch of candidates against a target profile.

    Returns similarity scores in `[0, 1]` of shape `(n,)`.
    """
    n = features.shape[0]
    layered = features.reshape(n, num_layers, top_k).astype(np.int64)
    scores = np.zeros(n, dtype=np.float64)

    for layer in range(num_layers):
        layer_profile = profile[layer]
        if not layer_profile:
            continue
        max_idx = max(layer_profile)
        weight_arr = np.zeros(max_idx + 1, dtype=np.float64)
        for idx, w in layer_profile.items():
            weight_arr[int(idx)] = w

        idx_matrix = layered[:, layer, :]
        valid = (idx_matrix >= 0) & (idx_matrix <= max_idx)
        clipped = np.where(valid, idx_matrix, 0)
        scores += (weight_arr[clipped] * valid).sum(axis=1)

    return scores / num_layers


def distance_to_profile(
    features: np.ndarray,
    profile: List[Dict[int, float]],
    num_layers: int,
    top_k: int,
) -> np.ndarray:
    return 1.0 - similarity_to_profile(features, profile, num_layers, top_k)
