#!/usr/bin/env python3
"""Re-eval a finished run from its saved rows.jsonl — NO retrieval re-run.

rows.jsonl is the source of truth (gold_ids + the retrieved ranking + the stored
set/answer/cost scalars). This recomputes the @k metrics from the saved rankings
and re-aggregates everything into results.json. Use it after changing a metric
definition or --k, instead of paying for another eval pass.

  python scripts/recompute_metrics.py runs/agent/<config> [--k 1 3 5 10]
  python scripts/recompute_metrics.py runs --all          # every run under runs/
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.getcwd())
from evaluation import metrics as M

_PASSTHROUGH = ("set_size", "set_recall", "set_precision", "set_f1",
                "answer_em", "answer_f1", "llm_calls", "n_steps",
                "prompt_tokens", "completion_tokens")


def reeval_dir(run_dir: str, ks) -> bool:
    rows_path = os.path.join(run_dir, "rows.jsonl")
    if not os.path.isfile(rows_path):
        return False
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    scored = []
    for r in rows:
        if r.get("skipped"):
            continue
        gold = set(r.get("gold_ids", []))
        rank = [d["doc_id"] if isinstance(d, dict) else d for d in r.get("retrieved", [])]
        if not gold or not rank:
            continue
        m = {}
        for k in ks:                                   # recomputed from saved ranking
            m[f"recall@{k}"] = M.recall_at_k(rank, gold, k)
            m[f"hit@{k}"] = M.hit_at_k(rank, gold, k)
            m[f"acc@{k}"] = M.acc_at_k(rank, gold, k)
            m[f"precision@{k}"] = M.precision_at_k(rank, gold, k)
            m[f"f1@{k}"] = M.f1_at_k(rank, gold, k)
            m[f"map@{k}"] = M.average_precision_at_k(rank, gold, k)
            m[f"ndcg@{k}"] = M.ndcg_at_k(rank, gold, k)
        m["mrr@10"] = M.mrr_at_k(rank, gold, 10)
        # Pass through metrics that are stored per row, not recomputable from the
        # ranking alone (set metrics need the agent's full surfaced set; answer/cost
        # are episode facts). A row that never stored them simply stays absent here —
        # the aggregator drops keys missing from any row, so we never invent values.
        for key in _PASSTHROUGH:
            if key in r:
                m[key] = r[key]
        scored.append(m)
    if not scored:
        return False
    keys = [k for k in scored[0] if all(k in s for s in scored)]
    agg = {k: sum(s[k] for s in scored) / len(scored) for k in keys}
    out = {"n": len(scored), "n_skipped": len(rows) - len(scored),
           "reevaluated": True, "metrics": agg}
    json.dump(out, open(os.path.join(run_dir, "results.json"), "w"), indent=1)
    print(f">> {run_dir}: {len(scored)} instances re-evaluated")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a run dir, or runs/ with --all")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--all", action="store_true", help="recurse: every run under path")
    a = ap.parse_args()
    if a.all:
        dirs = sorted({os.path.dirname(p) for p in
                       glob.glob(os.path.join(a.path, "**", "rows.jsonl"), recursive=True)})
        n = sum(reeval_dir(d, a.k) for d in dirs)
        print(f">> re-evaluated {n} runs")
    else:
        if not reeval_dir(a.path, a.k):
            print("no scored rows.jsonl found", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
