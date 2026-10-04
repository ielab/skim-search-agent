#!/usr/bin/env python3
"""Compute a BrowseComp-Plus leaderboard row from one judged cell, with the official definitions.

    python scripts/leaderboard_row.py runs/baseline/colbert_tongyi/agent/browsecomp_plus/Tongyi-DeepResearch-30B-A3B/agent_research_iter_colbert

The cell must carry the Qwen3-32B verdicts (`judge_correct_officialbcp`, docs/BASELINES.md step 7).
Prints one JSON object:

    accuracy      percent of all questions the Qwen3-32B judge marks correct. An answer with no
                  verdict counts as wrong.
    recall        mean over questions of the share of evidence documents (gold and evidence,
                  data/browsecomp_plus/qrels/evidence.tsv) that any search result showed the agent.
    search_calls  mean number of `search` calls per question.
    tokens        mean tokens per question, each token counted once: the largest prompt the
                  episode held plus every token the model generated (`total_tokens_once`). It is
                  not the sum of the prompts over the turns.
    judged        how many answers have a Qwen3-32B verdict.

The leaderboard leaves out calibration error. The agent never states a confidence, so that
number would just be 100 minus accuracy.
"""
import argparse
import collections
import json
import os


def load_evidence(path):
    evidence = collections.defaultdict(set)
    with open(path) as f:
        next(f)
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                evidence[qid].add(docid)
    return evidence


def row_metrics(cell, evidence):
    rows = [json.loads(l) for l in open(os.path.join(cell, "rows.jsonl"))]
    n = len(rows)
    correct = judged = 0
    recall = searches = tokens = 0.0
    for r in rows:
        verdict = r.get("judge_correct_officialbcp")
        judged += verdict is not None
        correct += verdict is True
        rel = evidence.get(r["instance_id"].split("__")[-1], set())
        shown = {str(d) for d in (r.get("surfaced_docs") or [])}
        recall += len(shown & rel) / len(rel) if rel else 0.0
        searches += sum(1 for a in (r.get("actions") or []) if a == "search")
        tokens += r.get("total_tokens_once") or 0
    return {"questions": n, "judged": judged, "accuracy": round(100 * correct / n, 1),
            "recall": round(100 * recall / n, 1), "search_calls": round(searches / n, 1),
            "tokens": round(tokens / n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cell", help="the condition directory holding rows.jsonl")
    ap.add_argument("--evidence", default="data/browsecomp_plus/qrels/evidence.tsv",
                    help="written by scripts/stage_browsecomp_plus.py")
    args = ap.parse_args()
    print(json.dumps(row_metrics(args.cell, load_evidence(args.evidence))))


if __name__ == "__main__":
    main()
