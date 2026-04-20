# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Minimal end-to-end demo of the NAG ranking algorithm on synthetic features.

Skips the actual NAG extraction (which requires a GPU and a backbone LLM) and
instead exercises `build_target_profile` / `distance_to_profile` on random
integer features. Useful as a smoke test after install.
"""
import numpy as np

from nag.ranking.nag_similarity import (
    build_target_profile,
    distance_to_profile,
    similarity_to_profile,
)


def main():
    rng = np.random.default_rng(0)
    num_layers = 8
    top_k = 5
    neurons_per_layer = 256

    target = rng.integers(0, neurons_per_layer, size=(100, num_layers * top_k))
    pool = rng.integers(0, neurons_per_layer, size=(1000, num_layers * top_k))

    profile = build_target_profile(target, num_layers=num_layers, top_k=top_k)
    sims = similarity_to_profile(pool, profile, num_layers=num_layers, top_k=top_k)
    dists = distance_to_profile(pool, profile, num_layers=num_layers, top_k=top_k)

    assert sims.shape == (1000,)
    assert np.allclose(sims + dists, 1.0)
    print(f"similarity min/mean/max: {sims.min():.4f} / {sims.mean():.4f} / {sims.max():.4f}")

    top_20pct = np.argsort(dists)[: int(0.2 * len(pool))]
    print(f"selected {len(top_20pct)} rows (top 20% by similarity)")
    print(f"selected rows mean distance: {dists[top_20pct].mean():.4f}")
    print(f"all rows mean distance:      {dists.mean():.4f}")


if __name__ == "__main__":
    main()
