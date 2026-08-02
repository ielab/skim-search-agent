"""LLM-judge (BrowseComp-Plus protocol) over FINISHED browsecomp run dirs. Re-extracts the answer
with the fixed extractor first (so the malformed final_answer doesn't leak into the judge), then
grades each row with threaded gpt-4o-mini calls. Writes judge_correct into rows.jsonl + judge_summary.json."""
import json, os, sys, glob
from concurrent.futures import ThreadPoolExecutor
from evaluation.llm_judge import make_judge, judge_answer_detail
from evaluation.doc_scoring import extract_answer_span


def _raw(r):
    t = [s.get("raw_output", "") for s in (r.get("trajectory") or []) if isinstance(s, dict)]
    return t[-1] if t else ""


def judge_dir(rd, gen, workers=16):
    rows = [json.loads(l) for l in open(os.path.join(rd, "rows.jsonl")) if l.strip()]
    for r in rows:                                   # re-extract with the FIXED last-<answer> logic
        r["final_answer"] = extract_answer_span(_raw(r) or (r.get("final_answer") or ""))
    def work(r):
        if "gold_answer" not in r:
            return r
        r.update(judge_answer_detail(r.get("question", ""), r.get("gold_answer", ""),
                                     r.get("final_answer", ""), gen))
        return r
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, rows))
    open(os.path.join(rd, "rows.jsonl"), "w").write(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    graded = [r for r in rows if "judge_correct" in r]
    nc = sum(1 for r in graded if r.get("judge_correct"))
    summ = {"judge_model": "gpt-4o-mini", "judge_prompt": "BrowseComp Appendix F (verbatim)",
            "n_judged": len(graded), "n_correct": nc,
            "judge_accuracy": (nc / len(graded)) if graded else 0.0}
    json.dump(summ, open(os.path.join(rd, "judge_summary.json"), "w"), indent=2)
    return summ


if __name__ == "__main__":
    gen = make_judge("gpt-4o-mini")
    # every FINISHED browsecomp condition (results.json present)
    dirs = sorted({os.path.dirname(f) for f in glob.glob("runs/agent/browsecomp_plus_*/*/*/rows.jsonl")
                   if os.path.exists(os.path.join(os.path.dirname(f), "results.json"))})
    for rd in dirs:
        tag = "/".join(rd.split("/")[2:]).replace("Tongyi-DeepResearch-30B-A3B/", "")
        print(f">> judging {tag} ...", flush=True)
        s = judge_dir(rd, gen)
        print(f"   JUDGE accuracy {100*s['judge_accuracy']:.1f}%  ({s['n_correct']}/{s['n_judged']})", flush=True)
    print(">> done", flush=True)
