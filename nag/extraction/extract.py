# Copyright 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Extract Neuron-Activated Graphs (NAGs) for a corpus of documents.

For each input the script records, per neuron type and per layer, the indices of
the top-K most impactful neurons. Neuron impact follows the definition in
Section 2.2 of the paper: for a projection W with column k,
    Imp(N_k | h_in) = | h_in^T W[:, k] | .

The top-K indices across all layers form the NAG of that input.

Input format
------------
Parquet files containing, per record:
    - `format=web`   : {"docid": str, "doc": str}
    - `format=final` : {"meta": {"docid": str}, "content_split": str}

Output format
-------------
One JSONL shard per input file. Each record:
    {
        "docid": str,
        "<neuron_type>_feature": {"layer_topk_value_index": [L * top_k ints]},
        ...
    }
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List

import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from nag.extraction.data_collator import DataCollatorForNAG
from nag.extraction.utils import to_bln, topk_indices
from nag.models import LlamaForCausalLM, Qwen3ForCausalLM, SmolLM3ForCausalLM


ALL_NEURON_TYPES = ["fwd_up", "fwd_down", "att_q", "att_k", "att_v", "att_o"]

QWEN3_VARIANTS = {
    "qwen3-0.6b", "qwen3-0.6b-base",
    "qwen3-1.7b", "qwen3-1.7b-base",
    "qwen3-4b", "qwen3-4b-base",
    "qwen3-8b", "qwen3-8b-base",
}
LLAMA_VARIANTS = {"llama3.2-3b", "llama3.2-3b-it"}
SMOLLM_VARIANTS = {"smollm3-3b", "smollm3-3b-base"}


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _load_model(model_type: str, model_path: str, attn_impl: str, device: torch.device):
    cfg = AutoConfig.from_pretrained(model_path)
    cfg.attn_implementation = attn_impl

    if device.type == "cuda" and os.environ.get("LOCAL_RANK") is not None:
        device_map = {"": int(os.environ["LOCAL_RANK"])}
    else:
        device_map = {"": device.type if device.index is None else str(device)}

    kwargs = dict(device_map=device_map, config=cfg, dtype=_pick_dtype(device))

    if model_type in QWEN3_VARIANTS:
        model = Qwen3ForCausalLM.from_pretrained(model_path, **kwargs)
    elif model_type in LLAMA_VARIANTS:
        model = LlamaForCausalLM.from_pretrained(model_path, **kwargs)
    elif model_type in SMOLLM_VARIANTS:
        model = SmolLM3ForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Supported: {sorted(QWEN3_VARIANTS | LLAMA_VARIANTS | SMOLLM_VARIANTS)}"
        )
    return model.eval()


def _safe_barrier(local_rank: int):
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier(device_ids=[local_rank])
        except TypeError:
            dist.barrier()


def _world_rank() -> tuple[int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return world, rank


@dataclass
class ExtractArgs:
    model_type: str = field(metadata={"help": "e.g. qwen3-1.7b-base, llama3.2-3b, smollm3-3b-base"})
    model_path: str = field(metadata={"help": "HuggingFace model dir or repo id"})
    in_data_path: str = field(metadata={"help": "directory containing parquet shards"})
    out_data_path: str = field(metadata={"help": "directory to write JSONL outputs"})
    format: str = field(default="web", metadata={"help": "web | final"})
    max_length: int = field(default=120, metadata={"help": "tokenizer truncation length"})
    batch_size: int = field(default=6)
    top_k: int = field(default=20, metadata={"help": "K neurons per layer"})
    neuron_types: str = field(
        default="fwd_up",
        metadata={"help": f"comma-separated subset of {ALL_NEURON_TYPES}"},
    )
    attn_implementation: str = field(default="eager")


class NAGExtractor:
    def __init__(self, args: ExtractArgs):
        self.args = args
        types = [t.strip() for t in args.neuron_types.split(",") if t.strip()]
        unknown = [t for t in types if t not in ALL_NEURON_TYPES]
        if unknown:
            raise ValueError(f"Unknown neuron types: {unknown}. Allowed: {ALL_NEURON_TYPES}")
        self.neuron_types = types

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, padding_side="left", use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = _pick_device()
        self.model = _load_model(
            args.model_type, args.model_path, args.attn_implementation, self.device
        )
        self.all_layers = list(range(self.model.config.num_hidden_layers))

        torch.set_grad_enabled(False)
        self.accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=14400))]
        )

        self.collator = DataCollatorForNAG(
            tokenizer=self.tokenizer,
            source_max_len=args.max_length,
            format=args.format,
        )

    def _forward(self, batch):
        _, _, sc_up, sc_down, sc_q, sc_k, sc_v, sc_o, _ = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            early_exit_layers=self.all_layers,
            use_cache=False,
        )
        return {
            "fwd_up": sc_up,
            "fwd_down": sc_down,
            "att_q": sc_q,
            "att_k": sc_k,
            "att_v": sc_v,
            "att_o": sc_o,
        }

    def _encode_batch(self, batch) -> List[dict]:
        docids = batch["docid"]
        inputs = {
            k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }

        with torch.no_grad():
            scores = self._forward(inputs)

        per_type = {}
        for nt in self.neuron_types:
            x_bln = to_bln(scores[nt]).to(torch.float32)
            per_type[nt] = topk_indices(x_bln, self.args.top_k).cpu().numpy().astype("int64")

        out = []
        for b, docid in enumerate(docids):
            rec = {"docid": docid}
            for nt in self.neuron_types:
                rec[f"{nt}_feature"] = {"layer_topk_value_index": per_type[nt][b].tolist()}
            out.append(rec)
        return out

    def _process_shard(self, in_file: str, out_file: str):
        df = pd.read_parquet(in_file, engine="pyarrow")
        if self.args.format == "web" and "meta" in df.columns:
            df = df.drop(columns=["meta"])
        dataset = Dataset.from_pandas(df)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=4,
            collate_fn=self.collator,
        )

        with open(out_file, "w") as fout:
            for batch in tqdm(loader, desc=os.path.basename(in_file)):
                for rec in self._encode_batch(batch):
                    fout.write(json.dumps(rec) + "\n")

    def run(self):
        world, rank = _world_rank()
        os.makedirs(self.args.out_data_path, exist_ok=True)
        in_files = sorted(glob.glob(os.path.join(self.args.in_data_path, "*.parquet")))
        shard_files = [f for i, f in enumerate(in_files) if i % world == rank]
        existing = {
            os.path.basename(p)
            for p in glob.glob(os.path.join(self.args.out_data_path, "*.json*"))
        }

        for in_file in shard_files:
            out_name = os.path.basename(in_file).replace(".parquet", ".jsonl")
            if out_name in existing:
                continue
            out_file = os.path.join(self.args.out_data_path, out_name)
            tmp_file = out_file + ".tmp"
            self._process_shard(in_file, tmp_file)
            os.replace(tmp_file, out_file)

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            _safe_barrier(local_rank)
        self.accelerator.wait_for_everyone()


def _parse_args() -> ExtractArgs:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_type", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--in_data_path", required=True)
    p.add_argument("--out_data_path", required=True)
    p.add_argument("--format", default="web", choices=["web", "final"])
    p.add_argument("--max_length", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--neuron_types", default="fwd_up",
                   help=f"comma-separated subset of {ALL_NEURON_TYPES}")
    p.add_argument("--attn_implementation", default="eager",
                   help="e.g. eager, sdpa, flash_attention_2")
    ns = p.parse_args()
    return ExtractArgs(**vars(ns))


def main():
    args = _parse_args()
    extractor = None
    try:
        extractor = NAGExtractor(args)
        extractor.run()
    finally:
        if extractor is not None and hasattr(extractor, "accelerator"):
            try:
                extractor.accelerator.free_memory()
                extractor.accelerator.end_training()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
