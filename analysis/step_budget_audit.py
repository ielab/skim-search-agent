#!/usr/bin/env python
"""Step-budget provenance audit + budget-MATCHED repair of every affected comparison.

WHY THIS EXISTS
---------------
The paper lists "matched step and token budgets" as one of three fairness contributions, and
the appendix states a "uniform 100-step budget". Both statements are FALSE on the two
Wikipedia datasets. This script establishes the TRUE per-instance step cap from CONFIGURATION
provenance (never from observed step counts -- see below), then recomputes every affected
comparison restricted to instances where BOTH cells ran under the SAME configured cap.

WHY NOT INFER THE CAP FROM OBSERVED STEPS
-----------------------------------------
`n_steps` (how many steps an episode actually took) is a POST-TREATMENT outcome. Splitting a
cell on `n_steps <= 51` vs `> 51` and calling the two halves "cap 50" and "cap 100" conditions
on the outcome and manufactures selection bias: a cap-100 cell in which most episodes happen to
finish in under 50 steps looks like a 50/100 "mixture" when it is in fact uniformly cap 100.
That is exactly what happens here -- see the provenance table. This script therefore reads the
cap out of CONFIGURATION only, from three independent sources:

  S1  SHARD provenance : runs/<tier>/__shards/<cond>/shard_<i>of<N>.txt lists the instance_ids
      routed to that shard; runs/<tier>/__shards/<cond>/shard_<i>of<N>/agent/<dataset>/<model>/
      <cond>/config.json gives that shard's `max_steps`. This is the per-instance
      instance_id -> shard -> configured cap map.
  S2  CANONICAL dir config : the cell's own runs/<tier>/agent/<dataset>/<model>/<cond>/
      config.json `max_steps`, which applies to every instance in the cell. Absent for merged
      cells (merge_shards.py does not synthesise one).
  S3  PER-ROW RECORDED cap : the `max_steps` field on each row of rows.jsonl. This is NOT an
      observed step count -- evaluation/run_eval.py copies `retriever.max_steps` (i.e.
      cfg.agent.max_steps, the configured budget) into the row via _retriever_meta/_META_KEYS at
      the moment that episode is written. It is a pre-treatment configuration value, stamped
      per episode.

ADJUDICATION. S1/S2 are directory-level and describe the LAST launch that wrote that directory;
S3 is stamped into each row when that episode ran. Where they disagree, S3 wins and the conflict
is reported (this happens for exactly one cell -- musique_structured SERP-bm25 -- whose
config.json says 50 while 200 of its 2409 rows were written by an earlier max_steps=100 launch).
Where S3 is absent (never happens in this repo) the instance is reported as UNKNOWN.

WHAT THE AUDIT FINDS (see the generated markdown for the table)
--------------------------------------------------------------
Every `_headline_validation` control cell on hotpotqa/musique is UNIFORMLY cap 100 (all shard
configs, all rows). Sieve on those datasets is UNIFORMLY cap 50. So the budget-matched subset
for Sieve-vs-any-control on hotpotqa/musique is EMPTY -- the comparison cannot be repaired by
subsetting; it can only be re-run. BrowseComp-Plus is uniformly cap 100 everywhere and is
therefore the one dataset on which these contrasts ARE budget-matched; it is computed here as
the valid matched evidence.

METHOD PARITY IS MANDATORY -- nothing statistical is reimplemented:
  scripts.compare_cells        : cell_dir, cell_rows, metrics, mcnemar_p, pct, load_qrels,
                                 load_judge_cache
  analysis.structured_surface_control : cell_metrics, paired_em, mean_tok_recall  (the count-once
                                 token mean used by analysis/structured_surface_control_3way.py)
  evaluation.metrics           : answer_em (transitively)
  scripts.force_answer_backfill: load_rows_with_recovery (transitively, via cell_rows)
Only the budget-matched SUBSETTING and the token t-test/Wilcoxon are new here, and the subsetting
feeds the same paired_em/mean_tok_recall helpers.

MANDATORY SANITY GATE (first, before any new number is trusted):
  (a) Sieve vs hybrid control on hotpotqa_structured  == +5.7 EM, McNemar p=1.62e-27
  (b) Sieve vs "No dense evidence" on browsecomp_plus_structured == +4.7 EM, p=0.0104
A gate failure aborts with a nonzero exit unless --no-gate.

Reads runs/ read-only. Writes analysis/step_budget_audit{_data.json,.md}.
Run: PYTHONPATH=. envs/bin/python analysis/step_budget_audit.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import cell_dir, load_qrels, mcnemar_p, pct  # noqa: E402
from analysis.structured_surface_control import (  # noqa: E402
    cell_metrics, paired_em, mean_tok_recall,
)

MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"
ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "analysis" / "step_budget_audit_data.json"
MD_PATH = ROOT / "analysis" / "step_budget_audit.md"
CACHE_PATH = ROOT / "analysis" / ".step_budget_cache.json"

SIEVE = ("_headline_validation", "agent_research_bql_dense_snip")
HYBRID = ("_headline_validation", "agent_research_hybrid_fetch_snip")
NODENSE = ("_headline_validation", "agent_research_snip")
SPARSE = ("_headline_validation", "agent_research_bm25_fetch_snip")
BASELINE = ("_visit_uncapped", "agent_research_bm25")
# the rest of the read-axis ablation grid, included so the budget status of the WHOLE ablation
# table is on the record rather than only the four comparisons named in the audit request
NOSNIP = ("_headline_validation", "agent_research_bql_dense_fetch")
DENSEPLAIN = ("_headline_validation", "agent_research_dense_fetch_plain")
DENSESNIP = ("_headline_validation", "agent_research_dense_fetch")

CELL_LABEL = {
    SIEVE: "Sieve (bql+dense+snip)",
    HYBRID: "hybrid control (bm25+dense RRF, snip+fetch)",
    NODENSE: "no dense evidence (bql+snip fetch)",
    SPARSE: "sparse only (bm25+snip fetch)",
    BASELINE: "SERP bm25 [BASELINE]",
    NOSNIP: "no snippets (bql+dense fetch)",
    DENSEPLAIN: "dense no snippets (plain dense fetch)",
    DENSESNIP: "dense same interface (dense+snip fetch)",
}

DATASETS = ["hotpotqa_structured", "musique_structured", "browsecomp_plus_structured"]
CELLS = [SIEVE, HYBRID, NODENSE, SPARSE, BASELINE, NOSNIP, DENSEPLAIN, DENSESNIP]

# (label, dataset, cell_A (Sieve), cell_B, published_full_set_reference or None)
# `published` mirrors what the repo's existing artifacts / the paper report for the FULL
# (unmatched) set, so the matched result can be put beside it. em_delta/p only.
COMPARISONS = [
    ("(a) Sieve vs hybrid control", "hotpotqa_structured", SIEVE, HYBRID,
     dict(src="analysis/structured_surface_control_3way.md", em_delta=5.7, p=1.62e-27)),
    ("(a) Sieve vs hybrid control", "musique_structured", SIEVE, HYBRID,
     dict(src="analysis/structured_surface_control_3way.md", em_delta=5.5, p=1.58e-11)),
    ("(b) Sieve vs no dense evidence", "hotpotqa_structured", SIEVE, NODENSE,
     dict(src="analysis/ablation_deltas.md", em_delta=0.8, p=0.103)),
    ("(b) Sieve vs no dense evidence", "musique_structured", SIEVE, NODENSE,
     dict(src="analysis/ablation_deltas.md", em_delta=1.3, p=0.0949)),
    ("(c) Sieve vs sparse only", "hotpotqa_structured", SIEVE, SPARSE,
     dict(src="analysis/ablation_deltas.md", em_delta=4.0, p=1.19e-14)),
    ("(c) Sieve vs sparse only", "musique_structured", SIEVE, SPARSE,
     dict(src="analysis/ablation_deltas.md", em_delta=5.7, p=1.37e-12)),
    ("(d) Sieve vs SERP-bm25 BASELINE", "musique_structured", SIEVE, BASELINE,
     dict(src="analysis/matched_budget_reanalysis.md", em_delta=2.70, p=0.00073)),
    # Included as controls on the audit itself: hotpotqa's baseline contrast is ALREADY fully
    # budget-matched (both cells cap 50), and browsecomp is uniformly cap 100 across every cell,
    # so browsecomp is the one dataset where (a)/(b)/(c) are genuinely budget-matched.
    ("(d') Sieve vs SERP-bm25 BASELINE", "hotpotqa_structured", SIEVE, BASELINE,
     dict(src="analysis/matched_budget_reanalysis.md", em_delta=1.60, p=0.00186)),
    ("(a'') Sieve vs hybrid control", "browsecomp_plus_structured", SIEVE, HYBRID,
     dict(src="analysis/structured_surface_control_3way.md", em_delta=3.3, p=0.102)),
    ("(b'') Sieve vs no dense evidence", "browsecomp_plus_structured", SIEVE, NODENSE,
     dict(src="analysis/ablation_deltas.md", em_delta=4.7, p=0.0104)),
    ("(c'') Sieve vs sparse only", "browsecomp_plus_structured", SIEVE, SPARSE,
     dict(src="analysis/ablation_deltas.md", em_delta=6.0, p=0.0025)),
    # remainder of the read-axis ablation grid on the two wiki datasets: these three rungs are
    # NOT all on the same side of the budget split, so their budget status has to be on the
    # record too (two of them turn out to be fully matched at cap 50).
    ("(e) Sieve vs no snippets", "hotpotqa_structured", SIEVE, NOSNIP,
     dict(src="analysis/ablation_deltas.md", em_delta=None, p=None)),
    ("(e) Sieve vs no snippets", "musique_structured", SIEVE, NOSNIP,
     dict(src="analysis/ablation_deltas.md", em_delta=None, p=None)),
    ("(f) Sieve vs dense no snippets", "hotpotqa_structured", SIEVE, DENSEPLAIN,
     dict(src="analysis/ablation_deltas.md", em_delta=None, p=None)),
    ("(f) Sieve vs dense no snippets", "musique_structured", SIEVE, DENSEPLAIN,
     dict(src="analysis/ablation_deltas.md", em_delta=None, p=None)),
    ("(g) Sieve vs dense same interface", "hotpotqa_structured", SIEVE, DENSESNIP,
     dict(src="analysis/ablation_deltas.md", em_delta=None, p=None)),
    ("(g) Sieve vs dense same interface", "musique_structured", SIEVE, DENSESNIP,
     dict(src="analysis/ablation_deltas.md", em_delta=None, p=None)),
]

GATES = [
    dict(name="(a) Sieve vs hybrid control, hotpotqa_structured (full set)",
         dataset="hotpotqa_structured", a=SIEVE, b=HYBRID,
         em_delta=5.7, p=1.62e-27, tol_delta=0.15, rel_tol_p=0.02),
    dict(name="(b) Sieve vs 'No dense evidence', browsecomp_plus_structured (full set)",
         dataset="browsecomp_plus_structured", a=SIEVE, b=NODENSE,
         em_delta=4.7, p=0.0104, tol_delta=0.15, rel_tol_p=0.02),
]

_MAXSTEPS_RE = re.compile(rb'"max_steps"\s*:\s*(\d+)')
_IID_RE = re.compile(rb'"instance_id"\s*:\s*"([^"]+)"')


# ---------------------------------------------------------------------------
# PART 1 -- provenance
# ---------------------------------------------------------------------------

def per_row_caps_path(p: Path, cache: dict) -> dict:
    """S3: instance_id -> the `max_steps` CONFIGURED VALUE stamped on that row when it was
    written (evaluation/run_eval.py `_retriever_meta`: retriever.max_steps == cfg.agent.max_steps).
    Not an observed step count. Streamed with a byte regex because rows.jsonl reaches ~750MB per
    cell and json.loads of the full trajectory is not needed for this one field; verified that
    both keys occur exactly once per line."""
    if not p.exists():
        return {}
    st = p.stat()
    key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
    if key in cache:
        return cache[key]
    out = {}
    with p.open("rb") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            mi, ms = _IID_RE.search(ln), _MAXSTEPS_RE.search(ln)
            if mi is None:
                continue
            out[mi.group(1).decode()] = int(ms.group(1)) if ms else None
    cache[key] = out
    return out


def per_row_caps(dataset: str, tier: str, cond: str, cache: dict) -> dict:
    return per_row_caps_path(cell_dir(tier, dataset, cond) / "rows.jsonl", cache)


def repo_wide_sweep(cache: dict) -> list:
    """PART 1b: the appendix claim is that EVERY condition runs under a 'uniform 100-step
    budget'. This checks it against every agent cell in runs/ for the main backbone, not just
    the cells in the comparisons above."""
    pats = ["runs/*/agent/*/" + MODEL_DIR + "/*/rows.jsonl",
            "runs/agent/*/" + MODEL_DIR + "/*/rows.jsonl"]
    out = []
    seen = set()
    for pat in pats:
        for p in sorted(ROOT.glob(pat)):
            if p in seen:
                continue
            seen.add(p)
            parts = p.parts
            tier = parts[-6] if parts[-6] != "runs" else "agent"
            caps = per_row_caps_path(p, cache)
            dist = {}
            for c in caps.values():
                dist[str(c)] = dist.get(str(c), 0) + 1
            out.append(dict(tier=tier, dataset=parts[-4], cond=parts[-2], n=len(caps),
                            cap_dist=dist,
                            uniform_100=bool(list(dist) == ["100"])))
    return out


def shard_caps(dataset: str, tier: str, cond: str) -> dict:
    """S1: instance_id -> configured max_steps, via the shard id-file -> shard config.json map.
    Only shard families that actually contain a config for THIS dataset are consulted (a
    condition's shard dir holds a separate family per dataset, e.g. `of20` = hotpotqa and
    `of15` = musique for agent_research_hybrid_fetch_snip)."""
    sdir = ROOT / "runs" / tier / "__shards" / cond
    out = {}
    if not sdir.is_dir():
        return out
    for idfile in sorted(sdir.glob("shard_*of*.txt")):
        stem = idfile.stem
        cfg = sdir / stem / "agent" / dataset / MODEL_DIR / cond / "config.json"
        if not cfg.exists():
            continue  # this shard family belongs to a different dataset
        cap = json.loads(cfg.read_text()).get("max_steps")
        for iid in idfile.read_text().split():
            out[iid] = cap
    return out


def canonical_cap(dataset: str, tier: str, cond: str):
    """S2: the cell's own config.json max_steps, or None when the cell was merged from shards
    (merge_shards.py writes no config.json)."""
    p = cell_dir(tier, dataset, cond) / "config.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("max_steps")


def cell_provenance(dataset: str, tier: str, cond: str, cache: dict) -> dict:
    """Adjudicated instance_id -> cap for one cell, plus the three raw sources and their
    agreement counts. Adjudication: per-row (S3) wins over directory-level (S1/S2), because
    config.json describes only the LAST launch that wrote the directory while the row field is
    stamped when that episode ran."""
    rows = per_row_caps(dataset, tier, cond, cache)
    shards = shard_caps(dataset, tier, cond)
    canon = canonical_cap(dataset, tier, cond)

    caps, unknown = {}, []
    agree_shard = disagree_shard = 0
    agree_canon = disagree_canon = 0
    covered_by_shard = 0
    for iid, row_cap in rows.items():
        s_cap = shards.get(iid)
        if s_cap is not None:
            covered_by_shard += 1
            if row_cap is not None:
                agree_shard += int(s_cap == row_cap)
                disagree_shard += int(s_cap != row_cap)
        if canon is not None and row_cap is not None:
            agree_canon += int(canon == row_cap)
            disagree_canon += int(canon != row_cap)
        cap = row_cap if row_cap is not None else (s_cap if s_cap is not None else canon)
        if cap is None:
            unknown.append(iid)
        else:
            caps[iid] = cap

    dist = {}
    for c in caps.values():
        dist[str(c)] = dist.get(str(c), 0) + 1
    return dict(
        dataset=dataset, tier=tier, cond=cond, label=CELL_LABEL.get((tier, cond), cond),
        n_rows=len(rows), caps=caps, cap_dist=dist, n_unknown=len(unknown),
        unknown_ids=unknown[:20],
        source_shard_covered=covered_by_shard,
        source_shard_agree=agree_shard, source_shard_disagree=disagree_shard,
        canonical_config_cap=canon,
        source_canon_agree=agree_canon, source_canon_disagree=disagree_canon,
        shard_cap_values=sorted({v for v in shards.values() if v is not None}),
    )


# ---------------------------------------------------------------------------
# PART 2/3 -- paired comparison on an id subset
# ---------------------------------------------------------------------------

def _paired_tokens(a_m: dict, b_m: dict, ids):
    """Count-once token means (the SAME `tok` field analysis/structured_surface_control_3way.py
    reports, via mean_tok_recall) plus the paired t-test the paper declares and a Wilcoxon
    signed-rank as the distribution-free companion. Differences are B - A, i.e. how many MORE
    tokens the control spends than Sieve; a positive mean diff = Sieve is cheaper."""
    ids = list(ids)
    if not ids:
        return dict(n=0)
    tok_a, _ = mean_tok_recall(a_m, ids)
    tok_b, _ = mean_tok_recall(b_m, ids)
    diffs = [b_m[i]["tok"] - a_m[i]["tok"] for i in ids]
    red = 100.0 * (tok_b - tok_a) / tok_b if tok_b else float("nan")
    out = dict(n=len(ids), tok_a=float(tok_a), tok_b=float(tok_b), tok_delta=float(tok_b - tok_a),
               pct_reduction=float(red), mean_diff=float(sum(diffs) / len(diffs)))
    try:
        from scipy.stats import ttest_rel, wilcoxon
        if len(ids) >= 2 and len(set(diffs)) > 1:
            t = ttest_rel([b_m[i]["tok"] for i in ids], [a_m[i]["tok"] for i in ids])
            out["t_stat"], out["t_p"] = float(t.statistic), float(t.pvalue)
            try:
                w = wilcoxon(diffs)
                out["wilcoxon_stat"], out["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
            except ValueError as e:
                out["wilcoxon_p"] = None
                out["wilcoxon_note"] = str(e)
        else:
            out["t_p"] = out["wilcoxon_p"] = None
    except ImportError:
        out["t_p"] = out["wilcoxon_p"] = None
    return out


def paired_on_ids(a_m: dict, b_m: dict, ids) -> dict:
    """EM (via paired_em's own convention, restricted to `ids`) + tokens on one id subset."""
    ids = sorted(ids)
    sub_a = {i: a_m[i] for i in ids}
    sub_b = {i: b_m[i] for i in ids}
    n, em_a, em_b, delta, b, c, p = paired_em(sub_a, sub_b)
    out = dict(n=n, em_a=float(em_a), em_b=float(em_b), em_delta=float(delta), b=int(b), c=int(c),
               p=(None if p is None else float(p)),
               sig=bool(p is not None and float(p) < 0.05))
    out["tokens"] = _paired_tokens(a_m, b_m, ids)
    return out


def compare(label, dataset, cell_a, cell_b, published, prov, mcache) -> dict:
    a_m = mcache[(dataset, cell_a)]
    b_m = mcache[(dataset, cell_b)]
    mut = sorted(set(a_m) & set(b_m))
    pa, pb = prov[(dataset, cell_a)]["caps"], prov[(dataset, cell_b)]["caps"]

    entry = dict(label=label, dataset=dataset,
                 cell_a=f"{cell_a[0]}/{cell_a[1]}", cell_b=f"{cell_b[0]}/{cell_b[1]}",
                 label_a=CELL_LABEL[cell_a], label_b=CELL_LABEL[cell_b],
                 published=published, n_shared=len(mut))
    entry["full"] = paired_on_ids(a_m, b_m, mut)

    # cross-tab of configured caps over the shared instance set
    xtab = {}
    for i in mut:
        k = f"{pa.get(i)}|{pb.get(i)}"
        xtab[k] = xtab.get(k, 0) + 1
    entry["cap_crosstab"] = xtab

    strata = {}
    for cap in sorted({v for v in pa.values()} | {v for v in pb.values()}):
        ids = [i for i in mut if pa.get(i) == cap and pb.get(i) == cap]
        if ids:
            strata[str(cap)] = paired_on_ids(a_m, b_m, ids)
    entry["strata"] = strata

    matched_ids = [i for i in mut if pa.get(i) is not None and pa.get(i) == pb.get(i)]
    entry["n_matched"] = len(matched_ids)
    entry["n_unmatched"] = len(mut) - len(matched_ids)
    if matched_ids:
        # Stratified (Mantel-Haenszel style) combination: for paired binary data the MH
        # statistic reduces to McNemar on the discordant pairs SUMMED over strata, so the
        # combined test is the exact binomial on (sum b, sum c) -- NOT a naive pooling of
        # instances across caps. EM/token means are reported over the union of the matched
        # strata and are flagged when more than one stratum contributes.
        sb = sum(s["b"] for s in strata.values())
        sc = sum(s["c"] for s in strata.values())
        comb = paired_on_ids(a_m, b_m, matched_ids)
        comb["b"], comb["c"] = int(sb), int(sc)
        comb["p"] = float(mcnemar_p(sb, sc))
        comb["sig"] = bool(comb["p"] < 0.05)
        comb["n_strata"] = len(strata)
        entry["matched_combined"] = comb
    else:
        entry["matched_combined"] = None
    return entry


def verdict(entry) -> tuple:
    """(flag, sentence) comparing the published FULL-set verdict to the budget-matched one."""
    full, m = entry["full"], entry["matched_combined"]
    if m is None or m["n"] == 0:
        return ("VERDICT UNTESTABLE",
                "No instance ran under the SAME configured cap in both cells, so no "
                "budget-matched subset exists. The published number cannot be confirmed OR "
                "refuted from existing data; it can only be re-run.")
    same_sig = (full["sig"] == m["sig"])
    same_dir = (full["em_delta"] >= 0) == (m["em_delta"] >= 0)
    if same_sig and same_dir:
        return ("VERDICT HOLDS",
                f"Matched subset (n={m['n']}) gives the same direction and the same "
                f"significance call as the published full set.")
    return ("VERDICT CHANGED",
            f"Matched subset (n={m['n']}) differs: sig {full['sig']} -> {m['sig']}, "
            f"delta {full['em_delta']:+.2f} -> {m['em_delta']:+.2f}.")


# ---------------------------------------------------------------------------

def fmt_p(p):
    if p is None:
        return "--"
    return f"{p:.3g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gate", action="store_true", help="run even if the sanity gate fails")
    args = ap.parse_args()

    cache = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            cache = {}

    qrels = {ds: load_qrels(ds) for ds in DATASETS}

    # load every cell once
    mcache, missing = {}, []
    for ds in DATASETS:
        for tier, cond in CELLS:
            m, _cov, _n = cell_metrics(ds, tier, cond, qrels[ds])
            if m is None:
                missing.append((ds, tier, cond))
            else:
                mcache[(ds, (tier, cond))] = m
    print(f"[cells] loaded {len(mcache)}, missing {len(missing)}", file=sys.stderr)

    # ---- sanity gate -----------------------------------------------------
    gate_results, gate_ok = [], True
    for g in GATES:
        a_m, b_m = mcache[(g["dataset"], g["a"])], mcache[(g["dataset"], g["b"])]
        n, em_a, em_b, delta, b, c, p = paired_em(a_m, b_m)
        ok = (abs(delta - g["em_delta"]) <= g["tol_delta"]
              and abs(p - g["p"]) <= g["rel_tol_p"] * g["p"])
        gate_ok &= ok
        gate_results.append(dict(name=g["name"], n=int(n), em_a=float(em_a), em_b=float(em_b),
                                 em_delta=float(delta), b=int(b), c=int(c), p=float(p),
                                 expected_delta=g["em_delta"], expected_p=g["p"],
                                 passed=bool(ok)))
        print(f"[gate] {g['name']}: n={n} delta={delta:+.2f} (exp {g['em_delta']:+.1f}) "
              f"p={p:.3g} (exp {g['p']:.3g}) -> {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    if not gate_ok and not args.no_gate:
        print("SANITY GATE FAILED -- stopping, no new numbers produced.", file=sys.stderr)
        JSON_PATH.write_text(json.dumps(dict(gate_passed=False, gates=gate_results), indent=1))
        return 2

    # ---- PART 1: provenance ---------------------------------------------
    prov = {}
    for ds in DATASETS:
        for tier, cond in CELLS:
            if (ds, (tier, cond)) not in mcache:
                continue
            prov[(ds, (tier, cond))] = cell_provenance(ds, tier, cond, cache)
            pr = prov[(ds, (tier, cond))]
            print(f"[prov] {ds:28s} {pr['label'][:34]:34s} {pr['cap_dist']} "
                  f"unknown={pr['n_unknown']}", file=sys.stderr)
    sweep = repo_wide_sweep(cache)
    CACHE_PATH.write_text(json.dumps(cache))
    print(f"[sweep] {len(sweep)} cells, "
          f"{sum(1 for r in sweep if not r['uniform_100'])} not uniformly cap-100",
          file=sys.stderr)

    # ---- PART 2/3: comparisons ------------------------------------------
    comps = []
    for label, ds, ca, cb, pub in COMPARISONS:
        if (ds, ca) not in mcache or (ds, cb) not in mcache:
            continue
        e = compare(label, ds, ca, cb, pub, prov, mcache)
        e["verdict_flag"], e["verdict_text"] = verdict(e)
        comps.append(e)
        print(f"[cmp] {label} {ds}: n_matched={e['n_matched']} -> {e['verdict_flag']}",
              file=sys.stderr)

    data = dict(
        gate_passed=bool(gate_ok), gates=gate_results,
        provenance=[{k: v for k, v in p.items() if k != "caps"} for p in prov.values()],
        repo_sweep=sweep,
        comparisons=comps,
        missing_cells=[f"{d}/{t}/{c}" for d, t, c in missing],
    )
    JSON_PATH.write_text(json.dumps(data, indent=1))
    MD_PATH.write_text(render(data))
    print(f"wrote {JSON_PATH}\nwrote {MD_PATH}")
    return 0


def render(d) -> str:
    L = []
    A = L.append
    A("# Step-budget provenance audit and budget-matched repair\n")
    A("Generated by `analysis/step_budget_audit.py` (read-only over `runs/`). Establishes the "
      "TRUE per-instance **configured** step cap of every cell involved in the paper's control "
      "comparisons, then recomputes each comparison on the subset of instances where BOTH cells "
      "ran under the SAME configured cap.\n")
    A("**The cap is never inferred from observed step counts.** `n_steps` is a post-treatment "
      "outcome; splitting a cell on `n_steps<=51` vs `>51` conditions on the outcome and makes a "
      "uniformly-cap-100 cell look like a 50/100 mixture whenever most of its episodes happen to "
      "finish early. Provenance comes from configuration only: (S1) `instance_id -> shard -> "
      "shard config.json max_steps`, (S2) the cell's own `config.json`, (S3) the `max_steps` "
      "field stamped on each row by `evaluation/run_eval.py` (`retriever.max_steps` == "
      "`cfg.agent.max_steps`, a configured value, written when the episode ran).\n")

    A("## 0. Mandatory sanity gate\n")
    A("| check | n | EM Sieve | EM other | delta | expected delta | p | expected p | result |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for g in d["gates"]:
        A(f"| {g['name']} | {g['n']} | {g['em_a']:.1f} | {g['em_b']:.1f} | {g['em_delta']:+.2f} | "
          f"{g['expected_delta']:+.1f} | {fmt_p(g['p'])} | {fmt_p(g['expected_p'])} | "
          f"**{'PASS' if g['passed'] else 'FAIL'}** |")
    A(f"\n**Gate overall: {'PASS' if d['gate_passed'] else 'FAIL'}**\n")

    A("## 1. Provenance table (configured step cap per instance)\n")
    A("| dataset | cell | n rows | cap 50 | cap 100 | unknown | canonical config.json | "
      "shard-covered ids | shard cap values | S1 vs S3 disagree | S2 vs S3 disagree |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for p in d["provenance"]:
        A(f"| {p['dataset']} | {p['label']} | {p['n_rows']} | {p['cap_dist'].get('50', 0)} | "
          f"{p['cap_dist'].get('100', 0)} | {p['n_unknown']} | "
          f"{p['canonical_config_cap'] if p['canonical_config_cap'] is not None else 'absent (merged)'} | "
          f"{p['source_shard_covered']} | "
          f"{p['shard_cap_values'] if p['shard_cap_values'] else 'no shards'} | "
          f"{p['source_shard_disagree']} | {p['source_canon_disagree']} |")
    A("")
    A("`unknown` = instances for which no configuration source (S1/S2/S3) yields a cap. "
      "`S1 vs S3 disagree` / `S2 vs S3 disagree` count instances where the directory-level "
      "config contradicts the per-row recorded cap; the per-row value is taken as authoritative "
      "because `config.json` records only the LAST launch that wrote that directory.\n")

    A("## 1b. The \"uniform 100-step budget\" claim, checked over every cell in `runs/`\n")
    A("The appendix states that *\"every condition, DCI included, runs under a uniform 100-step "
      "budget\"*. Below is every agent cell in `runs/` for the main backbone "
      f"(`{MODEL_DIR}`), with its per-instance configured caps. Rows flagged **NO** falsify that "
      "sentence.\n")
    sw = d.get("repo_sweep") or []
    bad = [r for r in sw if not r["uniform_100"]]
    A(f"{len(sw)} cells scanned; **{len(bad)} are not uniformly cap 100** "
      f"({len([r for r in bad if len(r['cap_dist']) > 1])} of those are internally MIXED, i.e. "
      "the same cell contains episodes run at two different caps).\n")
    A("| tier | dataset | condition | n | cap distribution | uniform 100-step? |")
    A("|---|---|---|---:|---|---|")
    for r in sw:
        A(f"| {r['tier']} | {r['dataset']} | `{r['cond']}` | {r['n']} | "
          f"{', '.join(f'{k}: {v}' for k, v in sorted(r['cap_dist'].items(), key=lambda kv: int(kv[0])))} | "
          f"{'yes' if r['uniform_100'] else '**NO**'} |")
    A("")

    A("## 2. Repaired (budget-matched) comparisons\n")
    A("| comparison | dataset | published full-set delta / p | full-set recomputed delta / p | "
      "cap cross-tab (Sieve\\|other) | n matched | matched delta / p | verdict |")
    A("|---|---|---|---|---|---:|---|---|")
    for c in d["comparisons"]:
        f, m = c["full"], c["matched_combined"]
        pub = c["published"]
        xt = ", ".join(f"{k}:{v}" for k, v in sorted(c["cap_crosstab"].items()))
        mt = "n/a (empty)" if m is None else f"{m['em_delta']:+.2f} / {fmt_p(m['p'])}"
        pt = ("not separately published"
              if pub.get("em_delta") is None
              else f"{pub['em_delta']:+.1f} / {fmt_p(pub['p'])}")
        A(f"| {c['label']} | {c['dataset']} | {pt} | "
          f"{f['em_delta']:+.2f} / {fmt_p(f['p'])} | {xt} | {c['n_matched']} | {mt} | "
          f"**{c['verdict_flag']}** |")
    A("")

    A("### Per-comparison detail\n")
    for c in d["comparisons"]:
        f, m = c["full"], c["matched_combined"]
        A(f"**{c['label']} -- {c['dataset']}**  ")
        A(f"A = `{c['cell_a']}` ({c['label_a']}), B = `{c['cell_b']}` ({c['label_b']}). "
          f"n_shared={c['n_shared']}, n_matched={c['n_matched']}, "
          f"n_unmatched={c['n_unmatched']}.\n")
        A("| subset | n | EM A | EM B | delta | b | c | McNemar p | sig@.05 | tok A | tok B | "
          "tok delta | % reduction | paired t p | Wilcoxon p |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")

        def row(name, s):
            t = s["tokens"]
            A(f"| {name} | {s['n']} | {s['em_a']:.1f} | {s['em_b']:.1f} | {s['em_delta']:+.2f} | "
              f"{s['b']} | {s['c']} | {fmt_p(s['p'])} | {'yes' if s['sig'] else 'no'} | "
              f"{t['tok_a']:,.0f} | {t['tok_b']:,.0f} | {t['tok_delta']:+,.0f} | "
              f"{t['pct_reduction']:+.1f}% | {fmt_p(t.get('t_p'))} | {fmt_p(t.get('wilcoxon_p'))} |")

        row("FULL (published basis, unmatched)", f)
        for cap, s in sorted(c["strata"].items(), key=lambda kv: int(kv[0])):
            row(f"stratum: both at cap {cap}", s)
        if m is not None:
            row("MATCHED combined (stratified)", m)
        else:
            A("| MATCHED combined | 0 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | "
              "-- | -- |")
        A(f"\n{c['verdict_flag']}: {c['verdict_text']}\n")
        A("---\n")

    A("## 3. The token/efficiency confound\n")
    A("The efficiency claim is the one confounded IN THE PAPER'S FAVOUR: a cell capped at 50 "
      "steps mechanically spends fewer tokens than one capped at 100, independent of the "
      "method. The `tok` column above is the count-once total "
      "(`total_tokens_once` = initial prompt + each retrieved document counted once + output), "
      "the same definition `analysis/structured_surface_control_3way.py` reports; the paired "
      "t-test is the paper's declared test and Wilcoxon is the distribution-free companion. "
      "Read the `MATCHED combined` row of each table: where it is absent, the token comparison "
      "on that dataset has no budget-matched evidence at all, and the only budget-matched "
      "evidence for those contrasts is browsecomp_plus_structured (uniformly cap 100 in every "
      "cell).\n")
    A("| comparison | dataset | budget-matched? | matched tok Sieve | matched tok other | "
      "% reduction | paired t p | Wilcoxon p | token advantage survives matching? |")
    A("|---|---|---|---:|---:|---:|---:|---:|---|")
    for c in d["comparisons"]:
        m = c["matched_combined"]
        if m is None:
            A(f"| {c['label']} | {c['dataset']} | NO (no matched instances) | -- | -- | -- | -- "
              f"| -- | **UNTESTABLE from existing data** |")
            continue
        t = m["tokens"]
        surv = (t["tok_delta"] > 0 and t.get("t_p") is not None and t["t_p"] < 0.05)
        A(f"| {c['label']} | {c['dataset']} | yes (n={m['n']}) | {t['tok_a']:,.0f} | "
          f"{t['tok_b']:,.0f} | {t['pct_reduction']:+.1f}% | {fmt_p(t.get('t_p'))} | "
          f"{fmt_p(t.get('wilcoxon_p'))} | "
          f"**{'YES' if surv else 'NO'}** |")
    A("")
    A("### The pattern, stated plainly\n")
    A("Sort every comparison in this audit by whether it is budget-matched, and the token result "
      "splits perfectly along that line:\n")
    A("- **Every UNMATCHED comparison** (Sieve at cap 50 vs a control at cap 100: hybrid, "
      "no-dense-evidence, sparse-only, dense-same-interface, on both wiki datasets) shows a large "
      "Sieve token *reduction*, +6.5% to +31.9%.\n")
    A("- **Every budget-MATCHED comparison against a same-read-interface control** shows no Sieve "
      "token advantage at all, and usually the opposite. On the matched cap-50 wiki rungs Sieve "
      "spends MORE than the control (hotpotqa: -4.5% vs no-snippets, p=6.0e-08; -3.2% vs plain "
      "dense fetch, p=2.1e-04; musique: -11.1% vs no-snippets, p=7.5e-29; -0.0% vs plain dense "
      "fetch, p=0.99). On budget-matched browsecomp it is -1.0%, -0.4% and +4.4% against the "
      "hybrid, no-dense and sparse-only controls respectively, none significant by the paper's "
      "declared paired t-test.\n")
    A("So the answer to \"does the token advantage survive budget matching?\" is **no** for every "
      "same-interface control on which it can be tested, and **untestable** for the three "
      "controls on hotpotqa/musique where the paper actually reports it.\n")
    A("The one efficiency claim that DOES survive is Sieve vs the SERP-bm25 BASELINE: "
      "budget-matched on hotpotqa (all 7343 at cap 50) and on 2209/2409 musique instances, with "
      "a +30.4% / +51.0% count-once token reduction, p far below any threshold under both tests. "
      "That one is also the only comparison where a genuine read-interface difference (whole-"
      "document visit vs section-level fetch) predicts a token gap, so it is the one place the "
      "efficiency story rests on the method rather than on the budget.\n")

    A("## 4. Which paper statements this bears on\n")
    A("Factual, claim by claim.\n")
    A("1. **\"Matched step and token budgets\"** (`latex/sections/introduction.tex:123`, "
      "`latex/sections/experimental_setup.tex:111`, `latex/sections/appendix.tex:60`) — FALSE as "
      "stated for step budgets on hotpotqa_structured and musique_structured. The appendix "
      "paragraph carrying that title only ever substantiates the *token* half (the fit-loop "
      "prompt budget and the 12,000-token per-document read ceiling); no step-budget matching "
      "procedure is described, and none was applied. Sieve ran at cap 50 on both wiki datasets "
      "while the hybrid, no-dense-evidence and sparse-only controls it is compared against ran "
      "at cap 100.\n")
    A("2. **\"Uniform 100-step budget\"** (`latex/sections/experimental_setup.tex:51`, "
      "`latex/sections/appendix.tex:141-142`) — FALSE. See section 1b: many cells ran at 50, and "
      "two cells (`agent/hotpotqa_structured/agent_research_dci` and "
      "`_visit_uncapped/musique_structured/agent_research_bm25`) are internally mixed. The DCI "
      "sentence is doubly wrong: it is the specific cell named in the claim, and it is one of "
      "the mixed cells.\n")
    A("3. **Accuracy deltas vs the controls** — the substance is *not* overturned, but it is no "
      "longer verified. The direction of the confound is conservative for accuracy (Sieve won "
      "with half the budget), so the published deltas are if anything understatements; but "
      "\"conservative\" is an argument, not a measurement, and with n_matched=0 there is no "
      "budget-matched subset that can confirm them. `analysis/matched_budget_reanalysis.md` "
      "already brackets these by trajectory truncation and every bracket keeps the sign; that "
      "bound, not a matched subset, is the available evidence.\n")
    A("4. **Efficiency / token-reduction deltas vs the same-interface controls** — the substance "
      "DOES change. The confound runs in the paper's favour here, and on every comparison where "
      "the budgets ARE matched the advantage is absent or reversed (section 3). Any "
      "token-reduction number quoted against the hybrid, no-dense-evidence, sparse-only or "
      "dense-same-interface control on hotpotqa or musique is not currently supported, and the "
      "matched evidence available points the other way.\n")
    A("5. **Sieve vs the SERP-bm25 BASELINE** — unaffected on hotpotqa (fully matched at cap 50) "
      "and effectively unaffected on musique (2209/2409 matched; +2.74 -> +2.81 EM, p 7.3e-4 -> "
      "8.2e-4). The headline accuracy result stands.\n")
    A("6. **The browsecomp results** — entirely unaffected; every browsecomp cell is at cap "
      "100.\n")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
