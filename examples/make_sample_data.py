# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Generate a tiny target set and candidate pool for the end-to-end demo.

Produces two parquet shards under `examples/data/`:

- `target/target.parquet`: 5 science/arithmetic-reasoning prompts. The goal is
  to mimic an ARC-Challenge-style target set.
- `pool/pool.parquet`: 20 mixed documents spanning science, narrative, code,
  and news. A correctly-working NAG ranker should prefer the science-flavored
  rows when the target set above is used.

This file is intended for smoke testing only; real experiments should use the
targets and pool described in the paper.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


TARGET_DOCS = [
    "Which of the following is a renewable energy source? A wind turbine "
    "converts kinetic energy from moving air into electricity, unlike coal "
    "which releases stored carbon when burned.",
    "A train travels 60 miles in one hour at a constant speed. If it keeps "
    "moving at the same rate without stopping, how far will it have traveled "
    "after three hours?",
    "Plants convert sunlight into chemical energy through photosynthesis. "
    "Chlorophyll absorbs light, which drives the reaction that turns carbon "
    "dioxide and water into glucose and oxygen.",
    "A fair six-sided die is rolled twice independently. What is the "
    "probability that the sum of the two outcomes is exactly seven, expressed "
    "as a fraction?",
    "Water boils at 100 degrees Celsius at sea level because atmospheric "
    "pressure is one standard atmosphere; at higher altitudes the reduced "
    "pressure lowers the boiling point.",
]

POOL_DOCS = [
    # Science / reasoning (should score high vs the science target)
    ("An electric circuit connects a battery, a resistor, and an LED in series. "
     "Current flows from the positive terminal through the resistor, and the "
     "voltage drop across the resistor obeys Ohm's law.",
     "science"),
    ("A chemist mixes an acid with a base in a neutralization reaction. The "
     "products are water and a salt, and the temperature of the solution "
     "rises because the reaction releases energy as heat.",
     "science"),
    ("Consider a right triangle with legs of length three and four units. "
     "By the Pythagorean theorem the hypotenuse has length five, which can be "
     "verified by squaring and summing.",
     "science"),
    ("The greenhouse effect traps heat in Earth's atmosphere. Gases such as "
     "carbon dioxide and methane absorb outgoing infrared radiation and "
     "re-emit some of it back toward the surface.",
     "science"),
    ("A ball thrown upward at ten meters per second will rise, decelerate "
     "under gravity, momentarily stop, and then fall back. Ignoring air "
     "resistance, the time to reach the peak is about one second.",
     "science"),
    # Narrative (should score lower)
    ("She opened the old wooden door and stepped into the dusty library. "
     "Rows of leather-bound books stretched to the ceiling, and a single "
     "window cast a thin beam of light across the floor.",
     "narrative"),
    ("The fisherman cast his net into the bay at dawn. The water was calm, "
     "and the gulls wheeled overhead, waiting for scraps that would be "
     "tossed back into the waves.",
     "narrative"),
    ("For years he had dreamed of climbing the mountain, and now, standing "
     "at the base camp, the wind sharp against his face, he finally "
     "believed that the summit was within reach.",
     "narrative"),
    ("The old cat padded silently across the kitchen tiles, watching the "
     "shadows shift as the afternoon sun moved lower. She had spent eighteen "
     "summers in this house.",
     "narrative"),
    ("On the last night of the festival, lanterns drifted across the river "
     "in long red lines. Children ran between the stalls, holding sticks of "
     "grilled sweet corn.",
     "narrative"),
    # Code / technical
    ("def quicksort(arr):\n    if len(arr) < 2:\n        return arr\n    "
     "pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    "
     "right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left)"
     " + [pivot] + quicksort(right)",
     "code"),
    ("When implementing a REST API, return 2xx codes for success, 4xx for "
     "client errors, and 5xx for server errors. Document every endpoint with "
     "its request body, response schema, and auth requirements.",
     "code"),
    ("A B-tree keeps keys sorted and balances itself during inserts and "
     "deletes. Each node stores up to t-1 keys and can have t children, "
     "which bounds the tree height at O(log n).",
     "code"),
    ("Docker containers package an application with its dependencies into a "
     "single portable image. Unlike a virtual machine, a container shares "
     "the host kernel and starts in milliseconds.",
     "code"),
    ("In React, the useState hook returns a state value and a setter. "
     "Calling the setter schedules a re-render, and the component receives "
     "the updated value on the next render pass.",
     "code"),
    # News / factual
    ("Central bankers raised interest rates by a quarter point on Wednesday, "
     "citing stubborn inflation in services. Markets had priced in the move, "
     "and equity indices closed slightly higher.",
     "news"),
    ("A storm system moved across the coast overnight, bringing heavy rain "
     "and gusts up to eighty kilometers per hour. Local authorities closed "
     "two coastal roads as a precaution.",
     "news"),
    ("The tournament ended in a penalty shootout after ninety minutes of "
     "regulation and thirty of extra time produced no goals. The home side "
     "converted four of five attempts.",
     "news"),
    ("A regional airline announced new direct routes to three second-tier "
     "cities, aiming to capture leisure travelers. The first flights are "
     "scheduled for early next quarter.",
     "news"),
    ("Researchers reported that a long-running wildlife survey recorded a "
     "modest rebound in the wolf population, following two decades of "
     "protective measures and habitat restoration.",
     "news"),
]


def _token_count(text: str) -> int:
    return len(text.split())


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


def main():
    root = Path(__file__).resolve().parent / "data"

    target_df = pd.DataFrame({
        "docid": [f"target_{i:02d}" for i in range(len(TARGET_DOCS))],
        "doc": TARGET_DOCS,
        "token_num": [_token_count(d) for d in TARGET_DOCS],
        "dataset": ["science_target"] * len(TARGET_DOCS),
    })
    _write(target_df, root / "target" / "target.parquet")

    pool_df = pd.DataFrame({
        "docid": [f"pool_{i:02d}" for i in range(len(POOL_DOCS))],
        "doc": [d for d, _ in POOL_DOCS],
        "token_num": [_token_count(d) for d, _ in POOL_DOCS],
        "dataset": [tag for _, tag in POOL_DOCS],
    })
    _write(pool_df, root / "pool" / "pool.parquet")

    print(f"\ntarget tokens: {int(target_df['token_num'].sum())}")
    print(f"pool tokens:   {int(pool_df['token_num'].sum())}")
    print("pool by category:")
    print(pool_df["dataset"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
