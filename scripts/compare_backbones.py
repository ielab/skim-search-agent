"""Controlled backbone pair on browsecomp_plus_structured / agent_research (BQL):
gpt-4o-mini (SDK driver) vs Tongyi-DeepResearch-30B-A3B (loop driver), SAME instance_ids.

Fairness notes:
- Answers re-extracted with the fixed last-<answer> extractor on BOTH sides.
- Accuracy = EM + the BrowseComp LLM judge (its paper metric) on BOTH sides.
- Tokens = CUMULATIVE prompt+completion usage (real API usage, comparable across drivers);
  total_tokens_once is NOT comparable (the SDK adapter only stamps step-0 prompt_tokens).
"""
import json, sys, glob, os
from evaluation.doc_scoring import extract_answer_span
from evaluation.metrics import answer_em
from evaluation.llm_judge import make_judge, judge_answer_detail

BACKBONES = {   # tag -> rows.jsonl (first existing glob match wins)
    "gpt-4o-mini": "runs/_openai_pair/agent/browsecomp_plus_structured/gpt-4o-mini/agent_research/rows.jsonl",
    "gpt-5.4-mini": "runs/_openai_pair54/agent/browsecomp_plus_structured/gpt-5.4-mini/agent_research/rows.jsonl",
    "Tongyi-30B": "runs/agent/browsecomp_plus_structured/Tongyi*/agent_research/rows.jsonl",
}


def _raw(r):
    t = [s.get("raw_output", "") for s in (r.get("trajectory") or []) if isinstance(s, dict)]
    return t[-1] if t else ""


def load(path):
    return {json.loads(l)["instance_id"]: json.loads(l) for l in open(path) if l.strip()}


def main():
    from evaluation.doc_scoring import answer_in_evidence
    data = {}
    for tag, pat in BACKBONES.items():
        hits = glob.glob(pat)
        if hits and os.path.exists(hits[0]):
            data[tag] = load(hits[0])
    tags = list(data)
    ids = [i for i in data[tags[0]] if all(i in data[t] for t in tags[1:])]
    print(f"backbones: {tags}   paired instances: {len(ids)}\n")
    gen = make_judge("gpt-4o-mini")
    stats = {t: {"em": 0, "judge": 0, "surf": 0, "ptok": 0, "ctok": 0, "steps": 0} for t in tags}
    for iid in ids:
        any_row = data[tags[0]][iid]
        gold = any_row.get("gold_answer") or ""
        q = any_row.get("question") or ""
        line = [f"--- {iid}  gold={gold!r}"]
        for tag in tags:
            row = data[tag][iid]
            pred = extract_answer_span(_raw(row) or (row.get("final_answer") or ""))
            em = answer_em(pred, gold)
            j = judge_answer_detail(q, gold, pred, gen)
            jc = bool(j.get("judge_correct"))
            obs = [s.get("observation", "") for s in (row.get("trajectory") or []) if isinstance(s, dict)]
            surf = answer_in_evidence(gold, obs)
            s = stats[tag]
            s["em"] += em; s["judge"] += jc; s["surf"] += surf
            s["ptok"] += row.get("prompt_tokens") or 0
            s["ctok"] += row.get("completion_tokens") or 0
            s["steps"] += row.get("n_steps") or 0
            mark = "JUDGE-OK" if jc else ("EM-OK  " if em else "wrong  ")
            line.append(f"    {tag:13}: {mark} surf={'Y' if surf else 'n'} pred={pred[:52]!r}")
        print("\n".join(line))
    n = len(ids)
    print(f"\n{'':14} {'EM':>6} {'judge':>6} {'gold-surfaced':>14} {'mean cum-tok':>14} {'steps':>7}")
    for tag in tags:
        s = stats[tag]
        print(f"{tag:14} {s['em']/n:6.0%} {s['judge']/n:6.0%} {s['surf']}/{n:<12} "
              f"{(s['ptok']+s['ctok'])/n:14,.0f} {s['steps']/n:7.1f}")


if __name__ == "__main__":
    main()
