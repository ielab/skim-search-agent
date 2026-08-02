"""Generate docs/results.md from the live run dirs. Re-extracts answers with the FIXED last-<answer>
extractor (recovers the ~37% mis-extracted rows), computes canonical EM/F1 + MEAN tokens/steps,
folds in the browsecomp LLM-judge accuracy (judge_summary.json) where present, and marks completion.

`--with-recovery` (OPT-IN, default OFF): overlays each condition dir's `recovered_answers.jsonl`
(scripts/force_answer_backfill.py's offline forced-answer backfill — see that module's docstring)
onto rows whose `final_answer` is empty, via `scripts.force_answer_backfill.load_rows_with_recovery`.
Without the flag, `main()` is BYTE-IDENTICAL to before this option existed (same glob, same rows,
same extraction) — the flag only swaps the source of `rows` and adds a `recovered %` column."""
import json, os, glob, statistics, datetime

TARGETS = {"musique": 2409, "browsecomp_plus": 830, "hotpotqa": 7405, "2wiki": 12576}
ORDER = ["research", "research_bm25", "research_dci", "research_bm25_dci", "research_bm25_fetch"]


def _raw(r):
    t = [s.get("raw_output", "") for s in (r.get("trajectory") or []) if isinstance(s, dict)]
    return t[-1] if t else ""


def main(with_recovery=False):
    from evaluation.doc_scoring import extract_answer_span
    from evaluation.metrics import answer_em, answer_f1
    rows_by = {}
    for f in glob.glob("runs/agent/*/*/*/rows.jsonl"):
        p = f.split("/"); ds = p[2]; cond = p[4].replace("agent_", "")
        rd = os.path.dirname(f)
        if with_recovery:
            from scripts.force_answer_backfill import load_rows_with_recovery
            rows = load_rows_with_recovery(rd)
        else:
            rows = [json.loads(l) for l in open(f) if l.strip()]
        if not rows:
            continue
        # a row recovered by the offline backfill already carries its short answer span in
        # `final_answer` (no raw trajectory tail to re-extract from) — use it as-is; every other
        # row keeps the existing re-extraction-from-raw-trajectory behavior unchanged.
        preds, n_recovered = [], 0
        for r in rows:
            if with_recovery and r.get("recovered"):
                preds.append(r.get("final_answer") or "")
                n_recovered += 1
            else:
                preds.append(extract_answer_span(_raw(r) or (r.get("final_answer") or "")))
        golds = [r.get("gold_answer") or "" for r in rows]
        em = 100 * sum(answer_em(p, g) for p, g in zip(preds, golds)) / len(rows)
        f1 = 100 * sum(answer_f1(p, g) for p, g in zip(preds, golds)) / len(rows)
        toks = [r.get("total_tokens_once") for r in rows if r.get("total_tokens_once")]
        steps = [r.get("n_steps") for r in rows if r.get("n_steps")]
        base = ds.replace("_structured", "").replace("_flat", "")
        tgt = TARGETS.get(base, len(rows))
        done = os.path.exists(os.path.join(rd, "results.json"))
        judge = None
        js = os.path.join(rd, "judge_summary.json")
        if os.path.exists(js):
            try:
                judge = 100 * json.load(open(js)).get("judge_accuracy", 0)
            except Exception:
                judge = None
        rows_by.setdefault(ds, {})[cond] = dict(
            n=len(rows), tgt=tgt, done=done, em=em, f1=f1, judge=judge,
            medtok=int(statistics.mean(toks)) if toks else 0,
            medsteps=round(statistics.mean(steps), 1) if steps else 0,
            recovered_pct=(100 * n_recovered / len(rows)) if with_recovery else None)

    out = []
    out.append("# BoolAgent — deep-research (doc arm) results")
    out.append("")
    out.append("_Tongyi-DeepResearch-30B-A3B, full clean re-run. EM/F1 = SQuAD-style canonical QA "
               "metrics (re-extracted with the fixed last-`<answer>` logic). browsecomp also reports "
               "**judge** = LLM-as-judge accuracy (BrowseComp-Plus / Appendix-F protocol, gpt-4o-mini), "
               "its paper metric. tokens/steps = MEAN per question (mean, not median: budget-capped "
               "episodes make the median saturate at the cap and hide the distribution)._")
    out.append("")
    out.append("Method key: **research** = BQL field-tagged `search`→`fetch` (the method); "
               "**bm25** = retrieve-then-visit-whole-doc; **dci** = bash/read brute force; "
               "**bm25_dci** / **bm25_fetch** = shared BM25 retrieval, differing read strategy. "
               "`_flat` corpus runs only bm25 (no structure to exploit).")
    out.append("")
    for ds in sorted(rows_by):
        out.append(f"## {ds}")
        out.append("")
        hasj = any(v.get("judge") is not None for v in rows_by[ds].values())
        hdr = ("| method | n | % | EM | F1 |" + (" judge |" if hasj else "")
              + (" recovered |" if with_recovery else "") + " mean tok | mean steps | status |")
        sep = ("|---|---|---|---|---|" + ("---|" if hasj else "") + ("---|" if with_recovery else "")
              + "---|---|---|")
        out.append(hdr); out.append(sep)
        conds = sorted(rows_by[ds], key=lambda c: ORDER.index(c) if c in ORDER else 99)
        for c in conds:
            v = rows_by[ds][c]
            pct = min(100, 100 * v["n"] / v["tgt"])
            jcell = (f" {v['judge']:.1f}% |" if v.get("judge") is not None else " – |") if hasj else ""
            rcell = f" {v['recovered_pct']:.1f}% |" if with_recovery else ""
            status = "✅ done" if v["done"] else f"🔄 {pct:.0f}%"
            out.append(f"| {c} | {v['n']} | {pct:.0f}% | {v['em']:.1f}% | {v['f1']:.1f}% |"
                       + jcell + rcell + f" {v['medtok']:,} | {v['medsteps']} | {status} |")
        out.append("")
    ndone = sum(1 for ds in rows_by for c in rows_by[ds] if rows_by[ds][c]["done"])
    ntot = sum(len(rows_by[ds]) for ds in rows_by)
    recovery_note = (" `recovered` = % of rows whose empty final_answer was filled in by "
                     "scripts/force_answer_backfill.py's offline pass (--with-recovery)."
                     if with_recovery else "")
    out.append(f"_{ndone}/{ntot} conditions complete. Partial conditions (🔄) are still accumulating; "
               "numbers shift as n grows. Regenerate: `python scripts/make_results_table.py`."
               + recovery_note + "_")
    os.makedirs("docs", exist_ok=True)
    open("docs/results.md", "w").write("\n".join(out) + "\n")
    print(f"wrote docs/results.md ({ndone}/{ntot} conditions done)")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--with-recovery", action="store_true",
                     help="overlay scripts/force_answer_backfill.py's recovered_answers.jsonl "
                          "onto empty-final_answer rows before scoring (opt-in; default OFF).")
    _args = _ap.parse_args()
    main(with_recovery=_args.with_recovery)
