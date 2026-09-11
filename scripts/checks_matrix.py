"""Write one experiment file per check of the verification matrix and print the sbatch
commands that run them, one GPU job each.

    python scripts/checks_matrix.py write      # configs/checks/<dataset>__<strategy>__<backbone>.yaml
    python scripts/checks_matrix.py submit [DATASET]   # sbatch one job per file (express queue), optionally one dataset
    python scripts/checks_matrix.py table      # judge and tabulate runs/checks

The matrix: every strategy on the InfoSeek-Eval sample with Tongyi; the structured
strategies on the structured BrowseComp-Plus corpus; the code strategies on the code fixture;
a subset of strategies with two more served backbones and one in-process vLLM backbone; the
retrieval-only floors; the ITER training path (a separate launcher). Every run uses seed 42.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "configs", "checks")
RUNS = "runs/checks"
ITER = "ielabgroup/ITER-Qwen3-Embedding-0.6B"
TONGYI = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
# name -> (model id, backend, context window, tensor parallelism). The paper backbones served
# here: Tongyi (both papers), gpt-oss-120b, Qwen3-30B-A3B-Thinking-2507 and Qwen-AgentWorld-35B-A3B
# (ITER); gpt-4o-mini (Sieve) runs from a node with API egress, not from this matrix.
BACKBONES = {
    "tongyi": (TONGYI, "api", 98304, 1),
    "qwen3_30b": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "api", 98304, 1),
    "qwen3_8b": ("Qwen/Qwen3-8B", "api", 32768, 1),
    "qwen3_30b_thinking": ("Qwen/Qwen3-30B-A3B-Thinking-2507", "api", 98304, 1),
    "agentworld_35b": ("Qwen/Qwen-AgentWorld-35B-A3B", "api", 98304, 2),
    "gpt_oss_120b": ("openai/gpt-oss-120b", "api", 98304, 2),
}
DOC_STRATEGIES = ["search_visit", "search_visit_dense", "search_visit_hybrid", "autoread", "autoread_dense",
                  "search_fetch", "search_fetch_dense", "search_fetch_hybrid", "sieve_bm25", "sieve", "sieve_dense",
                  "sieve_nosnip", "dci", "bounded_dci", "indri", "dedup_dense", "dedup_bm25",
                  "rag_bm25", "rag_dense", "rag_hybrid"]
CHECKS = (
    [("infoseek_eval_sample", s, "tongyi") for s in DOC_STRATEGIES]
    + [("browsecomp_plus_structured", s, "tongyi") for s in ("sieve_bm25", "sieve", "search_fetch", "indri")]
    + [("hotpotqa_structured", s, "tongyi") for s in ("sieve_bm25", "search_fetch", "indri", "search_visit")]
    + [("browsecomp_plus_chunks_sample", s, "tongyi") for s in ("search_visit", "dedup_dense")]
    + [("code_fixture", s, "tongyi") for s in ("codefix", "codefix_grep", "codefix_patch")]
    + [("infoseek_eval_sample", s, b) for b in ("qwen3_30b", "qwen3_8b") for s in ("search_visit", "sieve_bm25", "dedup_dense", "rag_bm25")]
    + [("infoseek_eval_sample", s, b) for b in ("qwen3_30b_thinking", "agentworld_35b", "gpt_oss_120b") for s in ("search_visit", "sieve_bm25", "dedup_dense")]
    # the in-process vLLM backend (model.backend: vllm) needs the vllm package in the run's own
    # environment; this cluster serves vLLM from a separate environment, so it is not in the matrix
)
DENSE = {"search_visit_dense", "search_visit_hybrid", "autoread_dense", "search_fetch_dense", "search_fetch_hybrid",
         "sieve", "sieve_dense", "dedup_dense", "rag_dense", "rag_hybrid"}


def _setk(t: str, section: str, key: str, value: str) -> str:
    pat = re.compile(rf"^(  {key}:)[^\n#]*?(\s*#.*)?$", re.M)
    m_sec = re.search(rf"^{section}:\n", t, re.M)
    if not m_sec:
        return t
    start = m_sec.end()
    m_end = re.search(r"^\S", t[start:], re.M)
    end = start + (m_end.start() if m_end else len(t) - start)
    block = t[start:end]
    if not pat.search(block):
        return t
    block = pat.sub(lambda m: f"{m.group(1)} {value}" + (" " + m.group(2).strip() if m.group(2) else ""), block, count=1)
    return t[:start] + block + t[end:]


def write() -> list[str]:
    os.makedirs(OUT, exist_ok=True)
    paths = []
    for dataset, strategy, backbone in CHECKS:
        model, backend, window, tp = BACKBONES[backbone]
        t = subprocess.run(["skimsearchagent", "template", "paper", strategy], capture_output=True, text=True, check=True).stdout
        name = f"check_{dataset}_{strategy}_{backbone}"
        t = re.sub(r"^name:[^\n#]*", f"name: {name}", t, count=1, flags=re.M)
        for sec, k, v in [("dataset", "name", dataset), ("model", "name", model), ("model", "backend", backend),
                          ("model", "api_base", "http://127.0.0.1:8000/v1"), ("model", "tp", str(tp)),
                          ("agent", "max_steps", "40"), ("agent", "ctx_tokens", str(int(window * 0.85))),
                          ("agent", "ctx_window", str(window)), ("evaluation", "workers", "2"),
                          ("output", "runs_dir", RUNS), ("retrieval", "structured_backend", "python"),
                          ("retrieval", "bm25_backend", "local")]:
            t = _setk(t, sec, k, v)
        if dataset in ("browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"):
            t = _setk(t, "dataset", "limit", "20")
        if strategy in DENSE:
            for k, v in [("dense_model", ITER), ("dense_dtype", "bfloat16"), ("dense_query_style", "i2"),
                         ("dense_query_instruction", "Given the main question, the current sub-query, and the sub-queries already tried"),
                         ("dense_pooling", "last_token")]:
                t = _setk(t, "retrieval", k, v)
        t = re.sub(r"^env: \{\}", "env:\n  HF_HUB_OFFLINE: '1'", t, flags=re.M)
        p = os.path.join(OUT, f"{dataset}__{strategy}__{backbone}.yaml")
        open(p, "w").write(t)
        r = subprocess.run(["skimsearchagent", "validate", p], capture_output=True, text=True)
        print(os.path.relpath(p, ROOT), "|", (r.stdout + r.stderr).strip().split("\n")[0][:100])
        paths.append(p)
    return paths


def submit(only: str | None = None) -> None:
    """Submit every check, or only those whose dataset, backbone or file stem equals `only`."""
    common = ("VLLM_PYTHON=/scratch3/wan458/DIVER/envs/bin/python,"
              "JAVA_HOME_OVERRIDE=/scratch3/wan458/conda-pkgs/openjdk-21.0.10-ha668962_0/lib/jvm")
    for i, (dataset, strategy, backbone) in enumerate(CHECKS):
        if only and only not in (dataset, backbone, f"{dataset}__{strategy}__{backbone}"):
            continue
        model, backend, window, tp = BACKBONES[backbone]
        port = 8100 + i          # jobs share nodes: each server needs its own port
        p = os.path.join(OUT, f"{dataset}__{strategy}__{backbone}.yaml")
        needs_dense = strategy in DENSE
        datasets = dataset if needs_dense else ""
        cmd = ["sbatch", "-A", "OD-236007", "--qos=express", "--time=02:30:00", f"--gres=gpu:{tp}",
               f"--job-name=chk-{strategy}-{backbone}",
               f"--export=ALL,{common},MODEL={model},TP={tp},PORT={port},MAX_MODEL_LEN={window},DATASETS={datasets},"
               f"EXPERIMENTS={p},RETRIEVER={ITER},OVERRIDES=",
               "scripts/slurm/iter_sample.sbatch"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(os.path.basename(p), "->", (out.stdout + out.stderr).strip())


def table() -> None:
    import glob
    import json
    rows = []
    for res in sorted(glob.glob(os.path.join(ROOT, RUNS, "*", "*", "*", "*", "results.json"))):
        d = os.path.dirname(res)
        parts = d.split(os.sep)
        kind, dataset, model, retriever = parts[-4], parts[-3], parts[-2], parts[-1]
        m = json.load(open(res))
        m = m.get("metrics", m)
        n = sum(1 for _ in open(os.path.join(d, "rows.jsonl")))
        judged = ""
        js = os.path.join(d, "judge_summary.json")
        if os.path.exists(js):
            j = json.load(open(js))
            judged = f"{j.get('n_correct', '?')}/{j.get('n_judged', '?')}"
        errors = sum(1 for l in open(os.path.join(d, "rows.jsonl")) if json.loads(l).get("error"))
        rows.append((dataset, retriever, model.split("/")[-1], n, errors, m.get("n_steps"), m.get("acc@5", m.get("hit@5")),
                     m.get("fix_file_ok"), judged))
    print("| dataset | retriever | backbone | rows | errors | steps | acc@5 | fix ok | judged |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join("" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "submit":
        submit(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        {"write": write, "table": table}[sys.argv[1]]()
