#!/usr/bin/env python
"""Prose-pack paired stats on the FULL BCP corpus, overlay-aware, shared ids, exact McNemar.
Reuses analysis.make_paper_tables' gather_cells (same overlay-aware loading as the tables).
Source of the results-section paired numbers after the 2026-07-30 full-corpus switch
(sieve vs baseline / nosnip / fetch-{bm25,dense,hybrid}; AgentWorld sieve vs baseline).

Run: PYTHONPATH=. envs/bin/python analysis/fullcorpus_prose_stats.py"""
import sys
sys.path.insert(0, "${REPO_ROOT:-.}")

from analysis.make_paper_tables import BCP_DS, cell_paths, gather_cells  # noqa: E402
from scripts.compare_cells import mcnemar_p, pct  # noqa: E402

ds = BCP_DS["full"]
cp = cell_paths(ds, "full")
cells = [("bcp", ck, cd, ds) for ck, cd in cp.items()]
stats = gather_cells(cells, workers=4, use_cache=True, verbose=False)


def paired(a, b, metric):
    """(n_shared, pct_a, pct_b, delta, p) for cells a,b on shared ids."""
    pa, pb = stats[("bcp", a)]["per"], stats[("bcp", b)]["per"]
    mut = sorted(set(pa) & set(pb))
    va = [bool(pa[i][metric]) for i in mut]
    vb = [bool(pb[i][metric]) for i in mut]
    x = sum(1 for i in mut if bool(pa[i][metric]) and not bool(pb[i][metric]))
    y = sum(1 for i in mut if bool(pb[i][metric]) and not bool(pa[i][metric]))
    return len(mut), pct(va), pct(vb), pct(va) - pct(vb), mcnemar_p(x, y), x, y


def tokmeans(a, b):
    pa, pb = stats[("bcp", a)]["per"], stats[("bcp", b)]["per"]
    mut = sorted(set(pa) & set(pb))
    ta = sum(pa[i]["tok"] for i in mut) / len(mut)
    tb = sum(pb[i]["tok"] for i in mut) / len(mut)
    return ta, tb, (ta - tb) / tb * 100


def line(tag, a, b):
    for metric in ("em", "judge"):
        n, va, vb, d, p, x, y = paired(a, b, metric)
        print(f"{tag} [{metric}]: n={n} {a}={va:.1f} {b}={vb:.1f} "
              f"delta={d:+.1f} p={p:.4g} (discordant {x}/{y})")
    ta, tb, dpc = tokmeans(a, b)
    sa, sb = stats[("bcp", a)], stats[("bcp", b)]
    print(f"{tag} [tok-once]: {a}={ta:.0f} {b}={tb:.0f} delta={dpc:+.1f}%")
    print(f"{tag} [tok-acc ]: {a}={sa['tok_acc']:.0f} {b}={sb['tok_acc']:.0f} "
          f"delta={(sa['tok_acc'] - sb['tok_acc']) / sb['tok_acc'] * 100:+.1f}%")
    print(f"{tag} [calls   ]: {a}={sa['calls']:.1f} {b}={sb['calls']:.1f}")
    print()


line("SIEVE vs BASELINE", "sieve", "visit_bm25")
line("SIEVE vs NOSNIP", "sieve", "sieve_nosnip")
line("SIEVE vs FETCH-BM25", "sieve", "fetch_bm25")
line("SIEVE vs FETCH-DENSE", "sieve", "fetch_dense")
line("SIEVE vs FETCH-HYBRID", "sieve", "fetch_hybrid")
line("AW SIEVE vs AW BASELINE", "xb_sieve", "xb_visit")
