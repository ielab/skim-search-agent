#!/usr/bin/env python3
"""Merge a released LoRA adapter into its base model and save a plain checkpoint the library can
load without `peft` (RepLLaMA: `castorini/repllama-v1-7b-lora-passage` on `meta-llama/Llama-2-7b-hf`).

Run it in an environment that has peft (the library's does not):

    python scripts/merge_lora.py --adapter castorini/repllama-v1-7b-lora-passage \\
        --out /path/to/repllama-v1-7b-passage-merged --dtype float16

It does what the adapter's model card does (`PeftModel.from_pretrained(base).merge_and_unload()`),
saves the merged weights with the base model's tokenizer, and writes MERGED_FROM.json naming
the adapter and base it came from. Imports nothing from agent_search.
"""
import argparse
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()
    from peft import PeftConfig, PeftModel
    from transformers import AutoModel, AutoTokenizer
    cfg = PeftConfig.from_pretrained(args.adapter)
    base = AutoModel.from_pretrained(cfg.base_model_name_or_path, torch_dtype=getattr(torch, args.dtype))
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload().eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg.base_model_name_or_path).save_pretrained(args.out)
    with open(os.path.join(args.out, "MERGED_FROM.json"), "w") as fh:
        json.dump({"adapter": args.adapter, "base": cfg.base_model_name_or_path, "dtype": args.dtype}, fh, indent=1)
    print(f"merged {args.adapter} into {cfg.base_model_name_or_path} -> {args.out}")


if __name__ == "__main__":
    main()
