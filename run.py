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
"""
from __future__ import annotations

import sys

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
    for k, v in kv.items():
        args += [f"--{k.replace('_', '-')}", v]

    print(">> python -m evaluation.run_eval " + " ".join(args), file=sys.stderr)
    from evaluation.run_eval import main as run_eval_main
    sys.argv = ["run_eval"] + args
    return run_eval_main() or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
