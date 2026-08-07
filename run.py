#!/usr/bin/env python
"""One-command launcher: run a complete research-agent experiment with key=value args.

    python run.py dataset=fixture strategy=sieve_bm25
    python run.py dataset=hotpotqa_structured strategy=search_visit model=gpt-4o-mini
    python run.py dataset=browsecomp_plus_structured_full strategy=sieve \\
        model=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \\
        backend=api api_base=http://localhost:8000/v1     # model served via `vllm serve`

`strategy` accepts the friendly names below or any raw `--retriever` value; every other
key=value pair is forwarded to `evaluation.run_eval` as `--key value`. If `model` is given
the agent policy defaults to `llm`; without it, the dependency-free stub policy runs.
Defaults: runs_dir=runs/quick.

The tool-surface knobs in ENV_KNOBS below are environment variables rather than run_eval
flags (they are read once, at import, by the tool modules). They are accepted here as
ordinary key=value pairs and exported before the harness loads, so a sweep reads the same
as any other argument and lands in the run's config.json:

    python run.py dataset=fixture strategy=sieve_bm25 snippet_tokens=64
    python run.py dataset=fixture strategy=search_visit max_visit_tokens=12000
"""
from __future__ import annotations

import os
import sys

# key=value names accepted here and exported as UPPERCASE environment variables before the
# harness is imported. These are read at import time by agent_search/agent/tools/* and
# retrievers/*, so they cannot be passed as run_eval flags — but there is no reason the user
# should have to know that. run_eval records every one of them in config.json.
ENV_KNOBS = (
    "snippet_tokens",                               # query-biased snippet window
    "max_visit_tokens", "max_section_tokens",       # read budgets (whole doc / named section)
    "bm25_visit_topk", "bm25_fetch_topk", "dense_visit_topk", "dense_fetch_topk",
    "autoread_topk", "hybrid_fetch_topk",           # listing depths
    "bql_soft_fallback", "indri_dense",             # method switches
    "structured_backend", "bm25_backend",           # engine selection
    "agent_ctx_window", "agent_ctx_stop_frac",      # context-budget early stop
)

STRATEGIES = {
    # conventional baselines
    "search_visit": "agent_research_bm25",
    "search_visit_dense": "agent_research_dense",
    "search_visit_hybrid": "agent_research_hybrid",
    "autoread": "agent_research_bm25_autoread",
    "autoread_dense": "agent_research_dense_autoread",
    "dci": "agent_research_dci",
    "bounded_dci": "agent_research_bm25_dci",
    "search_fetch": "agent_research_bm25_fetch_snip",
    "search_fetch_dense": "agent_research_dense_fetch",
    "search_fetch_hybrid": "agent_research_hybrid_fetch_snip",
    # Sieve family
    "sieve": "agent_research_bql_dense_snip",
    "sieve_bm25": "agent_research_snip",
    "sieve_dense": "agent_research_bql_donly_snip",
    "sieve_nosnip": "agent_research_bql_dense_fetch",
    # structured-retrieval control
    "indri": "agent_research_indri_snip",
    # retrieval-only floors
    "bm25": "bm25_local",
    "bm25_lucene": "bm25_pyserini",
}


def main(argv: list[str]) -> int:
    kv = {}
    for a in argv:
        if a in ("-h", "--help"):
            print(__doc__)
            print("strategies:", ", ".join(sorted(STRATEGIES)))
            return 0
        if "=" not in a:
            print(f"error: expected key=value, got {a!r} (try: python run.py --help)")
            return 2
        k, v = a.split("=", 1)
        kv[k.strip()] = v.strip().strip("'\"")

    if "dataset" not in kv:
        print("error: dataset=... is required (e.g. dataset=fixture)")
        return 2
    strategy = kv.pop("strategy", "sieve")
    retriever = STRATEGIES.get(strategy, strategy)

    args = ["--dataset", kv.pop("dataset"), "--retriever", retriever]
    if "model" in kv:
        args += ["--model", kv.pop("model")]
        args += ["--policy", kv.pop("policy", "llm")]
    elif "policy" in kv:
        args += ["--policy", kv.pop("policy")]
    elif retriever.startswith("agent_"):
        args += ["--policy", "stub"]
    args += ["--runs-dir", kv.pop("runs_dir", "runs/quick")]

    # export env-style knobs BEFORE evaluation.run_eval is imported below: the tool modules
    # resolve them at import, so setting them afterwards would silently have no effect.
    for k in [k for k in kv if k in ENV_KNOBS]:
        os.environ[k.upper()] = kv.pop(k)
        print(f">> {k.upper()}={os.environ[k.upper()]}", file=sys.stderr)

    for k, v in kv.items():
        args += [f"--{k.replace('_', '-')}", v]

    print(">> python -m evaluation.run_eval " + " ".join(args), file=sys.stderr)
    from evaluation.run_eval import main as run_eval_main
    sys.argv = ["run_eval"] + args
    return run_eval_main() or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
