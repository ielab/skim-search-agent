#!/usr/bin/env python
"""C1 -- the flat-vs-structured comparison the paper builds the flat twin FOR and then never
reports (docs/reviews/round4_full.md, C1).

For every (run_group, model, condition) that has a rows.jsonl under BOTH a `<base>_flat` and a
`<base>_structured` dataset dir, report the paired structured-vs-flat delta: n_shared, EM and
judge accuracy each side, paired delta, exact McNemar p (on the SHARED, judged-on-both-sides
subset), plus recall%/mean tokens (count-once)/mean LLM calls each side.

Today (2026-07-22) this is exactly one pair with usable data:
    runs/_visit_uncapped/agent/browsecomp_plus_flat/Tongyi-DeepResearch-30B-A3B/agent_research_bm25
    runs/_visit_uncapped/agent/browsecomp_plus_structured/Tongyi-DeepResearch-30B-A3B/agent_research_bm25
Both are the SAME agent (BM25 search + whole-document visit) over byte-identical underlying text
(structured side additionally carries a `sections` field the agent never queries through here --
this condition does not use the structured query surface). Any structured-vs-flat gap under THIS
condition therefore isolates what the corpus/ranking alone contributes, not the interface.

hotpotqa_flat / musique_flat were "just launched and filling" as of this writing (task brief) --
this script does NOT hardcode that pair list. It walks runs/ for every `*_flat` dataset dir,
requires a same-run-group/same-model/same-condition `*_structured` sibling, and requires >=100
rows on BOTH sides (MIN_ROWS below) before including a pair -- so re-running this script later,
once those wiki twins have filled in, picks them up automatically with no code change.

REUSE (per task instruction), no reimplementation:
  - evaluation.metrics.answer_em            -- canonical EM
  - scripts.force_answer_backfill.load_rows_with_recovery -- recovery-overlaid final_answer,
    the SAME overlay scripts/compare_cells.py's headline table uses
  - scripts.compare_cells.gold_doc_recall   -- gold-corpus-id-surfaced regex (verbatim)
  - scripts.compare_cells.mcnemar_p         -- exact two-sided McNemar via binomial test
  - scripts.compare_cells.load_qrels / load_judge_cache -- qrels + judge-verdict sidecar loading

Files touched here are the SAME ones scripts/compare_cells.py already reads for its registry
(188-450MB rows.jsonl, well under the 17-31GB auto-read files the task brief warns off).

    PYTHONPATH=. envs/bin/python analysis/flat_vs_structured.py
    PYTHONPATH=. envs/bin/python analysis/flat_vs_structured.py --out analysis/flat_vs_structured.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from hashlib import sha1
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.metrics import answer_em  # noqa: E402
from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, gold_doc_recall, load_judge_cache, load_qrels, mcnemar_p, metrics, pct,
)
from scripts.force_answer_backfill import load_rows_with_recovery  # noqa: E402

MIN_ROWS = 100
FLAT_SUFFIX = "_flat"
STRUCTURED_SUFFIX = "_structured"

# Corpus-vs-interface decomposition (round8, Job 1 extension): the full method cell each
# structured baseline is compared against to get the "interface step". Same cell
# scripts/compare_cells.py's REGISTRY calls "bql+dense+snip fetch" on every dataset (verified
# against REGISTRY at the time this was written: browsecomp_plus_structured/hotpotqa_structured/
# musique_structured all point agent_research_bql_dense_snip under _headline_validation here).
FULL_METHOD_SUBDIR = "_headline_validation"
FULL_METHOD_COND = "agent_research_bql_dense_snip"
BASELINE_SUBDIR = "_visit_uncapped"
BASELINE_COND = "agent_research_bm25"


# --------------------------------------------------------------------------------------
# discovery: every (run_group, model, condition) with rows on BOTH <base>_flat and
# <base>_structured, >= MIN_ROWS on each side. Two run-group layouts exist in this repo
# (see scripts/compare_cells.cell_dir): runs/agent/<ds>/<model>/<cond> (subdir == "agent")
# and runs/<subdir>/agent/<ds>/<model>/<cond> (any other run group) -- both globbed.
# --------------------------------------------------------------------------------------

def _count_rows(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def discover_pairs(repo_root: Path) -> list[dict]:
    patterns = [
        str(repo_root / "runs" / "agent" / "*_flat" / "*" / "*" / "rows.jsonl"),
        str(repo_root / "runs" / "*" / "agent" / "*_flat" / "*" / "*" / "rows.jsonl"),
    ]
    pairs, seen = [], set()
    for pattern in patterns:
        for flat_path in sorted(glob.glob(pattern)):
            cond_dir = os.path.dirname(flat_path)
            model_dir = os.path.dirname(cond_dir)
            flat_ds_dir = os.path.dirname(model_dir)
            flat_ds = os.path.basename(flat_ds_dir)
            agent_dir = os.path.dirname(flat_ds_dir)          # .../agent
            cond = os.path.basename(cond_dir)
            model = os.path.basename(model_dir)
            key = (flat_ds_dir, cond)
            if key in seen:
                continue
            seen.add(key)
            if not flat_ds.endswith(FLAT_SUFFIX):
                continue
            structured_ds = flat_ds[: -len(FLAT_SUFFIX)] + STRUCTURED_SUFFIX
            structured_path = os.path.join(agent_dir, structured_ds, model, cond, "rows.jsonl")
            if not os.path.exists(structured_path):
                continue
            n_flat = _count_rows(flat_path)
            if n_flat < MIN_ROWS:
                continue
            n_structured = _count_rows(structured_path)
            if n_structured < MIN_ROWS:
                continue
            run_group = os.path.relpath(os.path.dirname(agent_dir), repo_root)  # "runs" or "runs/_visit_uncapped"
            pairs.append({
                "run_group": run_group, "model": model, "condition": cond,
                "flat_dataset": flat_ds, "structured_dataset": structured_ds,
                "flat_dir": cond_dir, "structured_dir": os.path.dirname(structured_path),
                "n_flat_raw": n_flat, "n_structured_raw": n_structured,
            })
    return pairs


# --------------------------------------------------------------------------------------
# per-cell metrics
# --------------------------------------------------------------------------------------

def _qid(iid: str, dataset: str) -> str:
    prefix = dataset + "__"
    return iid[len(prefix):] if iid.startswith(prefix) else iid.rsplit("__", 1)[-1]


def load_cell(cond_dir: str, dataset: str, qrels: dict) -> dict:
    """qid -> per-row metrics dict, keyed by the dataset-prefix-stripped instance id so the two
    sides of a flat/structured pair (different dataset-name prefixes, identical numeric/qid
    suffix -- verified: qrels are byte-identical between browsecomp_plus_flat and
    browsecomp_plus_structured) pair up correctly."""
    judge_cache = load_judge_cache(Path(cond_dir))
    rows = load_rows_with_recovery(cond_dir)
    out = {}
    for r in rows:
        iid = r.get("instance_id") or ""
        qid = _qid(iid, dataset)
        gold = str(r.get("gold_answer") or "")
        ans = str(r.get("final_answer") or "")
        gold_ids = set(str(x) for x in (r.get("gold_ids") or qrels.get(qid, set())))
        tok = r.get("total_tokens_once")
        if tok is None:
            tok = ((r.get("initial_prompt_tokens") or 0) + (r.get("context_once_tokens") or 0)
                   + (r.get("output_tokens") or r.get("completion_tokens") or 0))
        out[qid] = {
            "iid": iid,
            "em": bool(answer_em(ans, gold)),
            "judge": judge_cache.get((iid, sha1(ans.strip().encode()).hexdigest())),
            "recall": gold_doc_recall(r, gold_ids),
            "tok": tok,
            "llm_calls": r.get("llm_calls") or r.get("n_steps") or len(r.get("observations") or []),
        }
    return out


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(xs):
    xs = list(xs)
    return 100.0 * sum(1 for x in xs if x) / len(xs) if xs else float("nan")


def compare(flat: dict, structured: dict) -> dict:
    shared = sorted(set(flat) & set(structured))
    n_shared = len(shared)
    flat_em = [flat[q]["em"] for q in shared]
    struct_em = [structured[q]["em"] for q in shared]
    # McNemar on EM: discordant pairs where flat/structured disagree
    b_em = sum(1 for q in shared if flat[q]["em"] and not structured[q]["em"])   # flat right, struct wrong
    c_em = sum(1 for q in shared if structured[q]["em"] and not flat[q]["em"])   # struct right, flat wrong
    p_em = mcnemar_p(b_em, c_em)

    judged = [q for q in shared if flat[q]["judge"] is not None and structured[q]["judge"] is not None]
    flat_judge = [flat[q]["judge"] for q in judged]
    struct_judge = [structured[q]["judge"] for q in judged]
    b_j = sum(1 for q in judged if flat[q]["judge"] and not structured[q]["judge"])
    c_j = sum(1 for q in judged if structured[q]["judge"] and not flat[q]["judge"])
    p_judge = mcnemar_p(b_j, c_j)

    return {
        "n_shared": n_shared,
        "flat": {
            "em_pct": _pct(flat_em), "judge_pct": _pct(flat_judge),
            "recall_pct": _pct(flat[q]["recall"] for q in shared),
            "mean_tok": _mean(flat[q]["tok"] for q in shared),
            "mean_calls": _mean(flat[q]["llm_calls"] for q in shared),
        },
        "structured": {
            "em_pct": _pct(struct_em), "judge_pct": _pct(struct_judge),
            "recall_pct": _pct(structured[q]["recall"] for q in shared),
            "mean_tok": _mean(structured[q]["tok"] for q in shared),
            "mean_calls": _mean(structured[q]["llm_calls"] for q in shared),
        },
        "delta_em_pp": _pct(struct_em) - _pct(flat_em),
        "delta_judge_pp": (_pct(struct_judge) - _pct(flat_judge)) if judged else float("nan"),
        "n_judged_shared": len(judged),
        "mcnemar": {"b_em": b_em, "c_em": c_em, "p_em": p_em,
                    "b_judge": b_j, "c_judge": c_j, "p_judge": p_judge},
    }


# --------------------------------------------------------------------------------------
# corpus-vs-interface decomposition (round8 Job 1 extension)
#
# The "corpus step" is exactly the flat-vs-structured comparison above (compare()), computed
# on the SAME structure-blind agent (agent_research_bm25) reading two byte-differing corpora.
# The "interface step" is the ALREADY-PUBLISHED full-method-vs-SERP-bm25-baseline paired
# comparison every REGISTRY row in comparison_result.md reports (see also
# analysis/round5_analyses.py's M3, HEADLINE_FAMILY_RAW, which copies these exact deltas/p from
# comparison_result.md) -- recomputed HERE directly from rows.jsonl (not copied) so this script
# stays correct as the underlying cells are re-judged / re-merged.
#
# IMPORTANT ASSUMPTION, stated explicitly because it is untestable with the cells that exist:
# the full method (bql+dense+snip fetch) has never been run on the FLAT corpus, so there is no
# direct "full method on flat" number. The decomposition below therefore treats the corpus
# effect (flat baseline -> structured baseline) as additively transferring onto the interface
# effect (structured baseline -> structured full method) to get a "total" = corpus + interface.
# This is the standard 2-step decomposition of a 2-factor design WITHOUT the interaction cell
# (no flat x full-method run exists) -- if corpus and interface interact (e.g. the full method's
# gain over baseline is itself corpus-dependent), the true total could differ from this sum. We
# do not have the data to test that here; flagged, not resolved.
# --------------------------------------------------------------------------------------

def interface_step(dataset: str, qrels: dict) -> dict | None:
    """Paired EM comparison: structured SERP-bm25 baseline vs structured full method
    (bql+dense+snip fetch), on their mutual instance_id set. None if either cell is absent."""
    base_rows = cell_rows(BASELINE_SUBDIR, dataset, BASELINE_COND)
    full_rows = cell_rows(FULL_METHOD_SUBDIR, dataset, FULL_METHOD_COND)
    if base_rows is None or full_rows is None:
        return None
    base_jc = load_judge_cache(cell_dir(BASELINE_SUBDIR, dataset, BASELINE_COND))
    full_jc = load_judge_cache(cell_dir(FULL_METHOD_SUBDIR, dataset, FULL_METHOD_COND))
    base_m = metrics(base_rows, qrels, dataset, base_jc)
    full_m = metrics(full_rows, qrels, dataset, full_jc)
    shared = sorted(set(base_m) & set(full_m))
    if not shared:
        return None
    b = sum(1 for i in shared if base_m[i]["em"] and not full_m[i]["em"])   # baseline right, full wrong
    c = sum(1 for i in shared if full_m[i]["em"] and not base_m[i]["em"])   # full right, baseline wrong
    em_base = pct([base_m[i]["em"] for i in shared])
    em_full = pct([full_m[i]["em"] for i in shared])
    return {
        "n": len(shared), "em_baseline": em_base, "em_full": em_full,
        "delta": em_full - em_base, "p": mcnemar_p(b, c), "b": b, "c": c,
    }


def build_decomposition(corpus_results: list[dict]) -> list[dict]:
    """For each dataset with a computed corpus step (flat-vs-structured pair, compare()
    output), attach the interface step and the additive-total decomposition."""
    out = []
    for r in corpus_results:
        structured_ds = r["structured_dataset"]
        qrels = load_qrels(structured_ds)
        iface = interface_step(structured_ds, qrels)
        if iface is None:
            out.append({"dataset": structured_ds, "corpus": r, "interface": None})
            continue
        delta_corpus = r["delta_em_pp"]
        delta_interface = iface["delta"]
        total = delta_corpus + delta_interface
        share_corpus = (100.0 * delta_corpus / total) if total else float("nan")
        share_interface = (100.0 * delta_interface / total) if total else float("nan")
        out.append({
            "dataset": structured_ds,
            "corpus": r,
            "interface": iface,
            "delta_corpus": delta_corpus, "delta_interface": delta_interface,
            "total": total, "share_corpus_pct": share_corpus, "share_interface_pct": share_interface,
        })
    return out


def render_decomposition_markdown(decomp: list[dict]) -> str:
    lines = ["\n---\n\n# Corpus-vs-interface decomposition (extension)\n"]
    lines.append(
        "For each dataset with a flat/structured control pair (above): flat-corpus baseline EM "
        "(structure-blind BM25 agent on the flat corpus) -> structured-corpus baseline EM (SAME "
        "agent, structured corpus -- **the corpus step**, identical to the `agent_research_bm25` "
        "row above) -> structured-corpus full-method EM (bql+dense+snip fetch -- **the interface "
        "step**). Both steps' paired significance (exact McNemar) is computed on that step's own "
        "shared-instance set (the two steps use slightly different instance sets in general -- "
        "corpus step is flat∩structured, interface step is structured-baseline∩structured-full-"
        "method -- both are the FULL dataset for datasets with complete runs, so this is a minor "
        "caveat in practice here, not a real discrepancy; noted per-row below).\n"
    )
    lines.append(
        "**Assumption flagged, not tested:** the total is computed as corpus-step delta + "
        "interface-step delta (additive). The full method has never been run on the flat corpus, "
        "so there is no cell that directly measures whether the corpus effect and the interface "
        "effect interact -- this decomposition assumes they do not. Treat 'total' as an "
        "estimate under that assumption, not a directly observed quantity.\n"
    )
    lines.append(
        "| dataset | n (corpus step) | flat-corpus baseline EM% | structured-corpus baseline EM% "
        "| Δ corpus (pp) | p (corpus) | n (interface step) | structured-corpus full-method EM% "
        "| Δ interface (pp) | p (interface) | Δ total (pp) | corpus share % | interface share % |"
    )
    lines.append("|" + "---|" * 13)
    for d in decomp:
        c = d["corpus"]
        if d["interface"] is None:
            lines.append(
                f"| {d['dataset']} | {c['n_shared']} | {c['flat']['em_pct']:.1f} | "
                f"{c['structured']['em_pct']:.1f} | {c['delta_em_pp']:+.1f} | {c['mcnemar']['p_em']:.3g} | "
                "-- | -- | -- | -- | -- | -- | -- | (full-method cell not available for this dataset) |"
            )
            continue
        i = d["interface"]
        lines.append(
            f"| {d['dataset']} | {c['n_shared']} | {c['flat']['em_pct']:.1f} | "
            f"{c['structured']['em_pct']:.1f} | {d['delta_corpus']:+.1f} | {c['mcnemar']['p_em']:.3g} | "
            f"{i['n']} | {i['em_full']:.1f} | {d['delta_interface']:+.1f} | {i['p']:.3g} | "
            f"{d['total']:+.1f} | {d['share_corpus_pct']:.0f}% | {d['share_interface_pct']:.0f}% |"
        )
    lines.append("")

    computable = [d for d in decomp if d["interface"] is not None]
    lines.append("## Per-dataset reading\n")
    for d in computable:
        c, i = d["corpus"], d["interface"]
        corpus_sig = "significant" if c["mcnemar"]["p_em"] < 0.05 else "not significant"
        iface_sig = "significant" if i["p"] < 0.05 else "not significant"
        lines.append(
            f"- **{d['dataset']}**: flat-corpus baseline {c['flat']['em_pct']:.1f}% -> "
            f"structured-corpus baseline {c['structured']['em_pct']:.1f}% "
            f"({d['delta_corpus']:+.1f} pp, McNemar p={c['mcnemar']['p_em']:.3g}, {corpus_sig}, "
            f"n={c['n_shared']}) -> structured-corpus full method {i['em_full']:.1f}% "
            f"({d['delta_interface']:+.1f} pp, McNemar p={i['p']:.3g}, {iface_sig}, n={i['n']}). "
            f"Of the {d['total']:+.1f} pp total (additive estimate), the corpus step accounts for "
            f"{d['share_corpus_pct']:.0f}% and the interface step for {d['share_interface_pct']:.0f}%."
        )
    lines.append("")

    if len(computable) >= 2:
        shares = [(d["dataset"], d["share_corpus_pct"]) for d in computable]
        spread = max(s for _, s in shares) - min(s for _, s in shares)
        pattern_note = (
            "roughly consistent across datasets (corpus share spread <= 20 points)" if spread <= 20
            else "clearly DATASET-DEPENDENT (corpus share varies by more than 20 points across "
                 "datasets) -- do not describe the split as a single fixed ratio in the paper"
        )
        lines.append(
            f"**Cross-dataset pattern:** corpus share of the total ranges "
            f"{min(s for _, s in shares):.0f}%-{max(s for _, s in shares):.0f}% across "
            f"{len(computable)} datasets ({', '.join(f'{d}={s:.0f}%' for d, s in shares)}) -- "
            f"{pattern_note}.\n"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# main / rendering
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write markdown here (default: print to stdout)")
    ap.add_argument("--json-out", default=str(REPO_ROOT / "analysis" / "flat_vs_structured_data.json"))
    args = ap.parse_args()

    pairs = discover_pairs(REPO_ROOT)
    if not pairs:
        print("No flat/structured pairs with >=%d rows on both sides found under runs/." % MIN_ROWS,
              file=sys.stderr)

    results = []
    for p in pairs:
        print(f"[{p['flat_dataset']} vs {p['structured_dataset']}] `{p['condition']}` "
              f"(run_group={p['run_group']}) -- loading ...", file=sys.stderr)
        qrels = load_qrels(p["flat_dataset"])
        flat_metrics = load_cell(p["flat_dir"], p["flat_dataset"], qrels)
        structured_metrics = load_cell(p["structured_dir"], p["structured_dataset"], qrels)
        cmp = compare(flat_metrics, structured_metrics)
        results.append({**p, **cmp})

    decomp = build_decomposition(results)

    with open(args.json_out, "w") as f:
        json.dump({"pairs": results, "decomposition": decomp}, f, indent=2, default=str)
    print(f"wrote {args.json_out}", file=sys.stderr)

    md = render_markdown(results) + render_decomposition_markdown(decomp)
    if args.out:
        Path(args.out).write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(md)


def render_markdown(results: list[dict]) -> str:
    lines = ["# Flat-vs-structured comparison (C1)\n"]
    if not results:
        lines.append(f"No pairs found with >= {MIN_ROWS} rows on both the `*_flat` and matching "
                      "`*_structured` side. Re-run this script once the in-flight flat-twin runs "
                      "have filled in.\n")
        return "\n".join(lines)

    lines.append("| flat dataset | structured dataset | condition | run group | n_shared | "
                  "flat EM% | struct EM% | ΔEM (pp) | McNemar p (EM) | "
                  "flat judge% | struct judge% | Δjudge (pp) | McNemar p (judge) | n judged | "
                  "flat recall% | struct recall% | flat tok/ep | struct tok/ep | "
                  "flat calls/ep | struct calls/ep |")
    lines.append("|" + "---|" * 18)
    for r in results:
        f, s = r["flat"], r["structured"]
        if r["n_judged_shared"] == 0:
            judge_cols = "pending (no judge_cache on one/both sides) | pending | -- | -- | 0 | "
        else:
            judge_cols = (
                f"{f['judge_pct']:.1f} | {s['judge_pct']:.1f} | {r['delta_judge_pp']:+.1f} | "
                f"{r['mcnemar']['p_judge']:.3g} | {r['n_judged_shared']} | "
            )
        lines.append(
            f"| {r['flat_dataset']} | {r['structured_dataset']} | `{r['condition']}` | "
            f"`{r['run_group']}` | {r['n_shared']} | "
            f"{f['em_pct']:.1f} | {s['em_pct']:.1f} | {r['delta_em_pp']:+.1f} | "
            f"{r['mcnemar']['p_em']:.3g} | " + judge_cols +
            f"{f['recall_pct']:.1f} | {s['recall_pct']:.1f} | "
            f"{f['mean_tok']:,.0f} | {s['mean_tok']:,.0f} | "
            f"{f['mean_calls']:.1f} | {s['mean_calls']:.1f} |"
        )
    lines.append("")

    lines.append("**Interpretation.** Every condition in this table is the SAME agent run over "
                  "structured vs. flat (byte-identical) text; none of these conditions query the "
                  "structured surface (fielded/Boolean search, section-level fetch) -- they are "
                  "plain BM25 keyword search with whole-document visit on both sides. Any delta "
                  "here is therefore attributable to the corpus/ranking alone (e.g. BM25 term "
                  "statistics shifting slightly because the structured corpus's indexed text "
                  "differs in incidental ways -- section markers, metadata fields folded into the "
                  "index), NOT to the method's interface, which is not in use in any of these "
                  "cells.\n")

    for r in results:
        f, s, mc = r["flat"], r["structured"], r["mcnemar"]
        direction = ("helps" if r["delta_em_pp"] > 0 else "hurts" if r["delta_em_pp"] < 0 else "is neutral for")
        sig = "significant" if mc["p_em"] == mc["p_em"] and mc["p_em"] < 0.05 else "not significant"
        recall_note = ("nearly flat, so the EM/judge gap is not obviously a retrieval-recall effect"
                        if abs(s["recall_pct"] - f["recall_pct"]) < 2 else
                        "itself different, which is consistent with (part of) the EM/judge gap "
                        "being a retrieval-side effect")
        if r["n_judged_shared"] == 0:
            judge_sentence = "judge accuracy is not comparable (no judge_cache on the flat side yet)"
        else:
            judge_sentence = (f"{r['delta_judge_pp']:+.1f} pp judge accuracy ({s['judge_pct']:.1f}% vs "
                               f"{f['judge_pct']:.1f}%, n_judged={r['n_judged_shared']}, "
                               f"McNemar p={mc['p_judge']:.3g})")
        lines.append(
            f"- **{r['structured_dataset']} vs {r['flat_dataset']}**, `{r['condition']}` "
            f"(n_shared={r['n_shared']}): structure alone {direction} this structure-blind agent "
            f"by {r['delta_em_pp']:+.1f} pp EM ({s['em_pct']:.1f}% vs {f['em_pct']:.1f}%, "
            f"McNemar b={mc['b_em']}/c={mc['c_em']}, p={mc['p_em']:.3g}, {sig} at α=0.05) and "
            f"{judge_sentence}, "
            f"while gold-doc recall is {s['recall_pct']:.1f}% (structured) vs {f['recall_pct']:.1f}% "
            f"(flat) -- {recall_note}.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
