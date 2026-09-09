#!/usr/bin/env python
"""Per-cell mean DISTINCT (count-once) and ACCUMULATED (step-summed) tokens/episode, for the
appendix token-columns table requested for browsecomp_plus_structured / hotpotqa_structured /
musique_structured.

Field semantics mirror scripts/compare_cells.py's `_row_intrinsic`:
    tok_once    = total_tokens_once, falling back to
                  initial_prompt_tokens + context_once_tokens + (output_tokens or
                  completion_tokens) on the rare row missing the field outright.
    tok_stepsum = prompt_tokens + completion_tokens (re-counts the growing prompt every step).

Loading pattern: rows.jsonl is streamed one line at a
time (json.loads, pull scalar fields, discard) rather than loaded whole -- these files run
190MB-1GB+. No recovery/judge overlay is applied (not needed: token fields are row-intrinsic and
unaffected by answer recovery).

Usage:
    PYTHONPATH=. python analysis/appendix_token_columns.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"

OUT_JSON = Path(__file__).resolve().parent / "appendix_token_columns.json"


def agent_dir(tier: str, ds: str, cond: str) -> Path:
    # runs/agent/<ds>/... has NO extra "agent" subfolder (unlike runs/_visit_uncapped/agent/<ds>/...
    # etc, where "agent" is a real subdirectory under the tier). "agent" tier IS that subdirectory.
    if tier == "agent":
        return ROOT / "runs" / "agent" / ds / MODEL_DIR / cond / "rows.jsonl"
    return ROOT / "runs" / tier / "agent" / ds / MODEL_DIR / cond / "rows.jsonl"


def oneshot_dir(ds: str, retriever: str) -> Path:
    return ROOT / "runs" / "_oneshot" / ds / retriever / "rows.jsonl"


# dataset key -> dataset dir name
DATASETS = {
    "bcp": "browsecomp_plus_structured",
    "hp": "hotpotqa_structured",
    "mq": "musique_structured",
}

# cell -> path-builder function taking dataset dir name, or None if not applicable for that ds
CELLS = {
    "oneshot_bm25":        lambda ds: oneshot_dir(ds, "bm25"),
    "oneshot_dense":       lambda ds: oneshot_dir(ds, "dense"),
    "autoread_bm25":       lambda ds: agent_dir("_visit_uncapped", ds, "agent_research_bm25_autoread"),
    "autoread_dense":      lambda ds: agent_dir("_visit_uncapped", ds, "agent_research_dense_autoread"),
    "dci":                 lambda ds: agent_dir("agent", ds, "agent_research_dci"),
    "bm25_dci":            lambda ds: agent_dir("agent", ds, "agent_research_bm25_dci") if ds == "browsecomp_plus_structured" else None,
    "visit_bm25":          lambda ds: agent_dir("_visit_uncapped", ds, "agent_research_bm25"),
    "visit_dense":         lambda ds: agent_dir("_fullvisit", ds, "agent_research_dense"),
    "visit_hybrid":        lambda ds: agent_dir("_fullvisit", ds, "agent_research_hybrid"),
    "fetch_bm25_k10":      lambda ds: agent_dir("_fetch_k10", ds, "agent_research_bm25_fetch_snip"),
    "fetch_dense_k10":     lambda ds: agent_dir("_fetch_k10", ds, "agent_research_dense_fetch"),
    "fetch_hybrid_k10":    lambda ds: agent_dir("_fetch_k10", ds, "agent_research_hybrid_fetch_snip"),
    "fetch_dense_plain_k10": lambda ds: agent_dir("_fetch_k10", ds, "agent_research_dense_fetch_plain") if ds == "browsecomp_plus_structured" else None,
    "sieve":               lambda ds: agent_dir("_headline_validation", ds, "agent_research_bql_dense_snip"),
    "sieve_nodense":       lambda ds: agent_dir("_headline_validation", ds, "agent_research_snip"),
    "sieve_nosnip":        lambda ds: agent_dir("_headline_validation", ds, "agent_research_bql_dense_fetch"),
    "indri_visit":         lambda ds: agent_dir("_fullvisit", ds, "agent_research_indri_visit") if ds == "browsecomp_plus_structured" else None,
    "indri_dense_visit":   lambda ds: agent_dir("_fullvisit_dense", ds, "agent_research_indri_visit") if ds == "browsecomp_plus_structured" else None,
    "indri_fetch":         lambda ds: agent_dir("_headline_validation", ds, "agent_research_indri_snip") if ds == "browsecomp_plus_structured" else None,
    "indri_dense_fetch":   lambda ds: agent_dir("_dense_validation", ds, "agent_research_indri_snip") if ds == "browsecomp_plus_structured" else None,
    # "Indri without snippets" is identified by its paper-table row (n=830 bcp-only, tok=45k):
    # the _headline_validation "agent_research_indri" condition (no _snip suffix),
    # sibling of agent_research_indri_snip ("Indri", with snippet excerpt) in the same tier.
    "indri_nosnip":        lambda ds: agent_dir("_headline_validation", ds, "agent_research_indri") if ds == "browsecomp_plus_structured" else None,
}


def stream_row_stats(path: Path):
    """Streams rows.jsonl one line at a time, no whole-file
    load. Returns (n, tok_once_values list, tok_stepsum_values list)."""
    tok_once_vals = []
    tok_stepsum_vals = []
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # unparsable trailing line from a live/torn write; skip
            iid = r.get("instance_id") or ""
            if not iid:
                continue
            tok_once = (
                r["total_tokens_once"] if r.get("total_tokens_once") is not None
                else (r.get("initial_prompt_tokens") or 0)
                     + (r.get("context_once_tokens") or 0)
                     + (r.get("output_tokens") or r.get("completion_tokens") or 0)
            )
            tok_stepsum = (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
            tok_once_vals.append(float(tok_once))
            tok_stepsum_vals.append(float(tok_stepsum))
    return tok_once_vals, tok_stepsum_vals


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    out = {"cells": {}, "warnings": [], "checks": {}}

    for dskey, ds in DATASETS.items():
        out["cells"][dskey] = {}
        for cell, fn in CELLS.items():
            path = fn(ds)
            entry_key = f"{dskey}/{cell}"
            if path is None:
                out["cells"][dskey][cell] = {
                    "n": None, "tok_once_mean": None, "tok_summed_mean": None,
                    "run_dir": None, "note": "not applicable for this dataset",
                }
                continue
            rel_path = str(path.relative_to(ROOT))
            if not path.exists():
                out["cells"][dskey][cell] = {
                    "n": None, "tok_once_mean": None, "tok_summed_mean": None,
                    "run_dir": rel_path, "note": "missing (rows.jsonl does not exist)",
                }
                print(f"MISSING {entry_key}: {rel_path}")
                continue
            result = stream_row_stats(path)
            if result is None or len(result[0]) == 0:
                out["cells"][dskey][cell] = {
                    "n": 0, "tok_once_mean": None, "tok_summed_mean": None,
                    "run_dir": rel_path, "note": "missing (0 rows)",
                }
                print(f"EMPTY {entry_key}: {rel_path}")
                continue
            tok_once_vals, tok_stepsum_vals = result
            n = len(tok_once_vals)
            tok_once_mean = mean(tok_once_vals)
            tok_stepsum_mean = mean(tok_stepsum_vals)
            out["cells"][dskey][cell] = {
                "n": n,
                "tok_once_mean": round(tok_once_mean),
                "tok_summed_mean": round(tok_stepsum_mean),
                "run_dir": rel_path,
            }
            print(f"{entry_key}: n={n} tok_once={tok_once_mean:.0f} tok_stepsum={tok_stepsum_mean:.0f} ({rel_path})")

    # ---- sanity anchors (distinct tokens) --------------------------------------------------
    anchors = {
        ("bcp", "visit_bm25"): 63800,
        ("bcp", "sieve"): 44900,
        ("hp", "visit_bm25"): 19900,
        ("hp", "sieve"): 13900,
        ("mq", "visit_bm25"): 43000,
        ("mq", "sieve"): 21300,
    }
    for (dskey, cell), expected in anchors.items():
        got = out["cells"][dskey][cell].get("tok_once_mean")
        if got is None:
            out["warnings"].append(f"{dskey}/{cell}: cannot check anchor, tok_once_mean is null")
            continue
        pct_off = 100.0 * (got - expected) / expected
        if abs(pct_off) > 2.0:  # generous tolerance beyond "rounding"
            out["warnings"].append(
                f"{dskey}/{cell}: tok_once_mean={got} vs anchor {expected} "
                f"({pct_off:+.2f}% off) -- FLAGGED"
            )

    # ---- checks: step-summed Sieve vs visit_bm25 reduction ---------------------------------
    expected_pct = {"bcp": -34.9, "hp": -24.1, "mq": -50.7}
    for dskey in DATASETS:
        sieve = out["cells"][dskey]["sieve"].get("tok_summed_mean")
        base = out["cells"][dskey]["visit_bm25"].get("tok_summed_mean")
        if sieve is None or base is None or base == 0:
            out["checks"][dskey] = {"pct_change": None, "note": "cannot compute, missing cell"}
            continue
        pct_change = 100.0 * (sieve - base) / base  # negative = sieve smaller (reduction)
        exp = expected_pct[dskey]
        diff = pct_change - exp
        out["checks"][dskey] = {
            "sieve_tok_summed_mean": sieve,
            "visit_bm25_tok_summed_mean": base,
            "pct_change_sieve_vs_visit_bm25": round(pct_change, 2),
            "paper_expected_pct": exp,
            "diff_from_paper": round(diff, 2),
            "within_tolerance_0.3": abs(diff) <= 0.3,
        }
        if abs(diff) > 0.3:
            out["warnings"].append(
                f"{dskey}: step-summed Sieve-vs-visit_bm25 reduction computed "
                f"{pct_change:+.2f}% vs paper's {exp:+.1f}% (diff {diff:+.2f}pp) -- FLAGGED"
            )

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")
    if out["warnings"]:
        print("\nWARNINGS:")
        for w in out["warnings"]:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
