#!/usr/bin/env python
"""Full-corpus BCP recomputation of (a) operator adoption (Analysis 2), (b) zero-hit fallback
share of Sieve's search calls, (c) failure decomposition (Analysis 1), reusing the project's own
paper_analyses functions and fielded_flat_vs_structured.classify_obs. Wiki rows untouched.

Generates the BCP-S numbers edited into tables/operator_adoption.tex,
figures/fig_failure_decomposition.tex, and tables/failure_decomposition.tex on 2026-07-30 when
the paper switched to the FULL corpus, and the new BCP endpoint of the zero-hit fallback range
(analysis.tex / appendix_engine.tex prose). The pooled_sieve cell is a sanity anchor: it must
reproduce the paper's historical 54.6% BCP fallback share (it does, to the digit, 17930/32862).

Run: PYTHONPATH=. envs/bin/python analysis/fullcorpus_bcp_analytics.py"""
import sys
sys.path.insert(0, "${REPO_ROOT:-.}")

from pathlib import Path

from analysis.paper_analyses import (  # noqa: E402
    analysis1_for_cell, analysis2_for_cell, load_cell, summarize_categories,
)
from analysis.fielded_flat_vs_structured import classify_obs  # noqa: E402

ROOT = Path("${REPO_ROOT:-.}")
MODEL = "Tongyi-DeepResearch-30B-A3B"

CELLS = {
    "full_sieve": ROOT / "runs/_fullcorpus/agent/browsecomp_plus_structured_full" / MODEL
    / "agent_research_bql_dense_snip",
    "full_base": ROOT / "runs/_fullcorpus/agent/browsecomp_plus_structured_full" / MODEL
    / "agent_research_bm25",
    "pooled_sieve": ROOT / "runs/_headline_validation/agent/browsecomp_plus_structured" / MODEL
    / "agent_research_bql_dense_snip",  # sanity anchor: paper's 54.6% BCP endpoint
}


def fallback_split(rows):
    from collections import Counter
    c = Counter()
    for r in rows:
        acts = r.get("actions") or []
        obs = r.get("observations") or []
        for a, o in zip(acts, obs):
            if "search" not in (a or "").lower():
                continue
            body = o
            if o.startswith("search:") and " -> " in o:
                body = o.split(" -> ", 1)[1]
            c[classify_obs(body)] += 1
    total = sum(c.values())
    return c, total


def report_fallback(name, rows):
    c, total = fallback_split(rows)
    fb = c["zero_exact_soft"] + c["zero_exact_coverage"]
    print(f"[{name}] search calls={total}")
    for k in sorted(c, key=c.get, reverse=True):
        print(f"    {k:22s} {c[k]:7d}  {100.0 * c[k] / total:5.1f}%")
    print(f"    => zero-hit fallback share = {fb}/{total} = {100.0 * fb / total:.1f}%")
    print()


for name in ("pooled_sieve", "full_sieve"):
    print(f"loading {name} ...", file=sys.stderr)
    rows = load_cell(CELLS[name])
    report_fallback(name, rows)
    if name == "full_sieve":
        a2 = analysis2_for_cell(rows)
        print(f"[full_sieve] operator adoption (n={a2['n_episodes']}, "
              f"queries/ep={a2['mean_queries_per_episode']:.1f}):")
        for k in ("quoted", "field", "bool", "wildcard", "date", "proximity", "bare_only"):
            print(f"    {k:10s} {a2['pct'][k]:5.1f}")
        m_per = analysis1_for_cell(rows)
        del rows
        s = summarize_categories(m_per)
        print(f"\n[full_sieve] failure decomposition: n_wrong={s['n_wrong']} "
              f"retrieval={s['pct']['retrieval']:.1f} selection={s['pct']['selection']:.1f} "
              f"synthesis={s['pct']['synthesis']:.1f}")
    else:
        del rows

print("loading full_base ...", file=sys.stderr)
rows = load_cell(CELLS["full_base"])
b_per = analysis1_for_cell(rows)
del rows
s = summarize_categories(b_per)
print(f"[full_base ] failure decomposition: n_wrong={s['n_wrong']} "
      f"retrieval={s['pct']['retrieval']:.1f} selection={s['pct']['selection']:.1f} "
      f"synthesis={s['pct']['synthesis']:.1f}")
