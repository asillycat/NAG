# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Data collator for NAG extraction.

Reads a batch of `{docid, text}` records and returns left-padded input ids plus
the preserved docid list so extracted features can be joined back to the pool.
"""
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers


TEXT_FIELDS = {
    "web": "doc",
    "final": "content_split",
}


@dataclass
class DataCollatorForNAG:
    """Tokenize raw documents with left padding for causal LM feature extraction."""
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    format: str = "web"

    def _extract(self, example: Dict) -> Dict:
        if self.format == "web":
            text = example["doc"]
            docid = example["docid"]
        elif self.format == "final":
            text = example["content_split"]
            docid = example["meta"]["docid"]
        else:
            raise ValueError(
                f"Unknown format {self.format!r}. Supported: {list(TEXT_FIELDS)}"
            )
        return {"input": text, "docid": docid}

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        records = [self._extract(x) for x in instances]
        sources = [r["input"] for r in records]
        tokens = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            return_tensors="pt",
            truncation=True,
            padding=True,
            add_special_tokens=False,
        )
        return {
            "prompts": sources,
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "docid": [str(r["docid"]) for r in records],
        }
