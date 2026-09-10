"""Train a dense retriever from trajectory triples (the ITER recipe) and plug it back in.

The training runs FlagEmbedding's decoder-only embedder trainer with ITER's patch (tier-weighted
InfoNCE: diversity, hard and weak negatives get different weights, plus a per-example weight from
how much the agent wrote after reading the positive). This module builds the exact command, checks
the environment, applies the patch, and writes the serving note the dense retriever reads, so the
checkpoint is used with the same instruction and pooling it was trained with.

Typical use (each step is also a console script):

    skimsearchagent-build-triples --runs runs/paper/agent/hotpotqa_structured --dataset hotpotqa_structured \\
        --out train_data/hotpotqa_i2.jsonl --query-style i2 --labeller oracle
    skimsearchagent-train-retriever template > train.yaml      # edit train_data / output_dir
    skimsearchagent-train-retriever check                        # FlagEmbedding present and patched?
    sbatch --export=ALL,TRAIN=train.yaml scripts/slurm/train_retriever.sbatch
    skimsearchagent run configs/paper/hotpotqa_structured_sieve.yaml retrieval.dense_model=models/my-retriever \\
        retrieval.dense_query_style=i2

Training needs its own environment (`pip install "skimsearchagent[train]"` installs FlagEmbedding
1.3.5, which wants an older transformers than the retrieval extra pins); see docs/TRAINING.md.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from agent_search.training.queries import DEFAULT_STYLE, INSTRUCTIONS, STYLES

PATCH_PATH = Path(__file__).with_name("patches") / "flagembedding-1.3.5-iter.patch"
SERVING_NOTE = "skimsearchagent_dense.json"
FLAGEMBEDDING_VERSION = "1.3.5"


@dataclass
class TrainConfig:
    """Every knob of the recipe. Defaults are ITER's paper settings for Qwen3-Embedding-0.6B."""
    train_data: str = "train_data/triples.jsonl"
    output_dir: str = "models/skimsearchagent-retriever"
    base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    query_style: str = DEFAULT_STYLE          # must match the triples' query_style
    query_instruction: Optional[str] = None   # null = INSTRUCTIONS[query_style]
    query_max_len: int = 8192                 # 512 for the plain style
    passage_max_len: int = 512
    train_group_size: int = 10
    per_device_batch_size: int = 32
    learning_rate: float = 1e-6
    epochs: int = 2
    warmup_ratio: float = 0.1
    temperature: float = 0.02
    pooling: str = "last_token"
    normalize: bool = True
    neg_w_div: float = 3.0
    neg_w_hard: float = 1.0
    neg_w_weak: float = 0.3
    div_neg_cap: int = 3
    hard_neg_cap: int = 3
    bf16: bool = True
    gradient_checkpointing: bool = True
    nproc_per_node: int = 1
    deepspeed: Optional[str] = None           # path to a DeepSpeed config for the 4B/8B recipe
    report_to: str = "none"                   # "wandb" if you have it set up
    run_name: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42
    extra_args: list[str] = field(default_factory=list)

    @property
    def instruction(self) -> str:
        return self.query_instruction or INSTRUCTIONS[self.query_style]


HELP = {
    "train_data": "triples jsonl from skimsearchagent-build-triples",
    "output_dir": "where the checkpoint is written (then use it as retrieval.dense_model)",
    "base_model": "the embedder to fine-tune (last-token pooling models: Qwen3-Embedding-*)",
    "query_style": "how queries were rendered in train_data (plain, mem, docs, i1..i7); served identically",
    "query_instruction": "null = the style's instruction from agent_search.training.queries.INSTRUCTIONS",
    "query_max_len": "tokens; 8192 for history-conditioned styles, 512 for plain",
    "passage_max_len": "document tokens (re-encode the corpus with the same length)",
    "train_group_size": "1 positive + negatives per query",
    "per_device_batch_size": "queries per GPU step",
    "learning_rate": "ITER: 1e-6", "epochs": "ITER: 2", "warmup_ratio": "ITER: 0.1",
    "temperature": "InfoNCE temperature", "pooling": "last_token | mean | cls",
    "normalize": "L2-normalise embeddings (cosine retrieval)",
    "neg_w_div": "weight of diversity negatives (already-read relevant docs)",
    "neg_w_hard": "weight of hard negatives (already-read irrelevant docs)",
    "neg_w_weak": "weight of weak negatives (returned, never read)",
    "div_neg_cap": "max diversity negatives per query", "hard_neg_cap": "max hard negatives per query",
    "bf16": "mixed precision", "gradient_checkpointing": "saves memory; keep on",
    "nproc_per_node": "GPUs for torchrun", "deepspeed": "DeepSpeed config path (ZeRO) or null",
    "report_to": "none | wandb", "run_name": "label for the trainer", "resume_from_checkpoint": "path or null",
    "seed": "training seed", "extra_args": "extra trainer flags, verbatim",
}


def template(preset: Optional[str] = None) -> str:
    cfg = TrainConfig()
    if preset == "plain":
        cfg.query_style, cfg.query_max_len = "plain", 512
    lines = ["# SkimSearchAgent retriever-training file (ITER recipe). One file = one training run.",
             "# Run: skimsearchagent-train-retriever run train.yaml   (or sbatch scripts/slurm/train_retriever.sbatch)"]
    for k, v in asdict(cfg).items():
        val = yaml.safe_dump(v, default_flow_style=True).strip().rstrip(".").strip() if v is not None else "null"
        lines.append(f"{k}: {val:<36} # {HELP.get(k, '')}")
    return "\n".join(lines) + "\n"


def load(path: str) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    unknown = [k for k in data if k not in TrainConfig.__dataclass_fields__]
    if unknown:
        raise ValueError(f"unknown training keys {unknown}; allowed: {list(TrainConfig.__dataclass_fields__)}")
    cfg = TrainConfig(**data)
    if cfg.query_style not in STYLES:
        raise ValueError(f"query_style {cfg.query_style!r}; choose from {STYLES}")
    return cfg


def build_command(cfg: TrainConfig, *, python: str = sys.executable) -> list[str]:
    """The exact trainer invocation (ITER's flags), as an argv list."""
    launcher = [python, "-m", "torch.distributed.run", "--nproc_per_node", str(cfg.nproc_per_node)]
    args = [
        "-m", "FlagEmbedding.finetune.embedder.decoder_only.base",
        "--model_name_or_path", cfg.base_model,
        "--train_data", cfg.train_data,
        "--output_dir", cfg.output_dir,
        "--run_name", cfg.run_name or Path(cfg.output_dir).name,
        "--query_max_len", str(cfg.query_max_len),
        "--passage_max_len", str(cfg.passage_max_len),
        "--pad_to_multiple_of", "8",
        "--query_instruction_for_retrieval", cfg.instruction,
        "--query_instruction_format", "Instruct: {}\nQuery: {}",
        "--train_group_size", str(cfg.train_group_size),
        "--per_device_train_batch_size", str(cfg.per_device_batch_size),
        "--learning_rate", str(cfg.learning_rate),
        "--num_train_epochs", str(cfg.epochs),
        "--warmup_ratio", str(cfg.warmup_ratio),
        "--save_strategy", "epoch", "--save_steps", "500", "--logging_steps", "1",
        "--overwrite_output_dir",
        "--dataloader_drop_last", "True",
        "--same_dataset_within_batch", "True",
        "--temperature", str(cfg.temperature),
        "--sentence_pooling_method", cfg.pooling,
        "--normalize_embeddings", "True" if cfg.normalize else "False",
        "--neg_w_div", str(cfg.neg_w_div), "--neg_w_hard", str(cfg.neg_w_hard),
        "--neg_w_weak", str(cfg.neg_w_weak),
        "--div_neg_cap", str(cfg.div_neg_cap), "--hard_neg_cap", str(cfg.hard_neg_cap),
        "--report_to", cfg.report_to, "--seed", str(cfg.seed),
    ]
    if cfg.bf16:
        args.append("--bf16")
    if cfg.gradient_checkpointing:
        args += ["--gradient_checkpointing", "--gradient_checkpointing_kwargs", '{"use_reentrant": false}']
    if cfg.deepspeed:
        args += ["--deepspeed", cfg.deepspeed]
    if cfg.resume_from_checkpoint:
        args += ["--resume_from_checkpoint", cfg.resume_from_checkpoint]
    args += list(cfg.extra_args)
    return launcher + args


def flagembedding_root() -> Optional[Path]:
    try:
        spec = importlib.util.find_spec("FlagEmbedding")
    except Exception:
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def _patch_markers() -> dict[str, str]:
    """For every file the patch touches, one added line that must be present afterwards."""
    markers: dict[str, str] = {}
    current = None
    for line in PATCH_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("+++ b/FlagEmbedding/"):
            current = line[len("+++ b/FlagEmbedding/"):]
        elif current and current not in markers and line.startswith("+") and not line.startswith("+++"):
            body = line[1:].strip()
            if len(body) >= 12:
                markers[current] = body
    return markers


def is_patched(root: Optional[Path] = None) -> bool:
    """True only when every file the patch touches carries its change (a partial apply after
    a FlagEmbedding point release would otherwise pass as patched)."""
    root = root or flagembedding_root()
    if root is None:
        return False
    for rel, marker in _patch_markers().items():
        f = root / rel
        if not f.exists() or marker not in f.read_text(errors="ignore"):
            return False
    return True


def check_environment() -> dict:
    root = flagembedding_root()
    info = {"flagembedding": str(root) if root else None, "patched": is_patched(root),
            "torch": None, "patch_file": str(PATCH_PATH), "patch_sha256": None}
    if PATCH_PATH.exists():
        info["patch_sha256"] = hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest()[:16]
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
    except Exception:
        pass
    return info


def apply_patch(root: Optional[Path] = None, *, dry_run: bool = False) -> str:
    """Apply ITER's changes to an installed FlagEmbedding 1.3.5 (idempotent: skips if patched)."""
    root = root or flagembedding_root()
    if root is None:
        raise RuntimeError("FlagEmbedding is not installed (pip install 'skimsearchagent[train]')")
    if is_patched(root):
        return "already patched"
    if shutil.which("patch") is None:
        raise RuntimeError("the `patch` command is required to apply the FlagEmbedding patch")
    cmd = ["patch", "-p2", "--forward", "-d", str(root)] + (["--dry-run"] if dry_run else [])
    out = subprocess.run(cmd, input=PATCH_PATH.read_text(), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"patch failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout.strip()


def write_serving_note(cfg: TrainConfig, output_dir: Optional[str] = None) -> str:
    """The note `agent_search.retrievers.dense` reads when `dense_model` is this directory: the
    instruction, pooling, normalisation and lengths the checkpoint was trained with."""
    out = Path(output_dir or cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    note = {"query_instruction": cfg.instruction, "query_style": cfg.query_style,
            "pooling": cfg.pooling, "normalize": cfg.normalize,
            "max_seq_length": cfg.passage_max_len, "query_max_len": cfg.query_max_len,
            "dtype": "bfloat16" if cfg.bf16 else "float32",     # served in the precision it was trained in
            "base_model": cfg.base_model, "recipe": "iter", "trainer": f"FlagEmbedding=={FLAGEMBEDDING_VERSION}+iter-patch"}
    path = out / SERVING_NOTE
    path.write_text(json.dumps(note, indent=2))
    return str(path)


def run(cfg: TrainConfig, *, dry_run: bool = False) -> int:
    cmd = build_command(cfg)
    print(">> " + " ".join(repr(a) if " " in a or "\n" in a else a for a in cmd), file=sys.stderr)
    if dry_run:
        return 0
    if not is_patched():
        raise RuntimeError("FlagEmbedding is not patched; run `skimsearchagent-train-retriever patch` first")
    write_serving_note(cfg)
    rc = subprocess.call(cmd)
    if rc == 0:
        write_serving_note(cfg)
    return rc


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="skimsearchagent-train-retriever",
                                 description="Train a dense retriever from trajectory triples (ITER recipe).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("template").add_argument("preset", nargs="?", default=None, choices=[None, "plain"])
    sub.add_parser("check")
    p = sub.add_parser("patch"); p.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("run"); r.add_argument("config"); r.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("serving-note"); s.add_argument("config"); s.add_argument("--output-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "template":
        print(template(a.preset), end=""); return 0
    if a.cmd == "check":
        print(json.dumps(check_environment(), indent=2)); return 0
    if a.cmd == "patch":
        print(apply_patch(dry_run=a.dry_run)); return 0
    if a.cmd == "serving-note":
        print(write_serving_note(load(a.config), a.output_dir)); return 0
    return run(load(a.config), dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
