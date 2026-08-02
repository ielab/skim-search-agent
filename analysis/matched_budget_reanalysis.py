#!/usr/bin/env python
"""Matched-step-budget re-analysis by TRAJECTORY TRUNCATION — zero new GPU compute.

WHY THIS EXISTS
---------------
The paper claims matched step budgets across compared conditions as an explicit fairness
contribution. That claim is FALSE on the two Wikipedia datasets: reading the per-row
`max_steps` field out of every `runs/**/rows.jsonl`, some cells ran at `max_steps=50` and
others at `max_steps=100`, and they are compared against each other. BrowseComp-Plus is
clean (every cell at 100). This script (a) audits the budget of EVERY cell, (b) recomputes
every affected comparison at a MATCHED 50-step cap by truncating the over-budgeted side's
trajectories, and (c) reports the BrowseComp flat/structured self-replicate as a
run-to-run noise estimate.

WHY TRUNCATION IS RECONSTRUCTIBLE
---------------------------------
Every row retains full per-step structure, not just aggregates:
  `trajectory`  : list[dict], one entry per step, each with the step's raw_output
  `observations`: list[str],  len == len(trajectory) == `n_steps`
  `n_steps`     : number of trajectory entries
  `llm_calls`   : number of model turns actually consumed
  `max_steps`   : the budget THAT ROW ran under (per-row, so mixed-budget cells are detectable)
  `stopped`     : "answer" | "max_steps" | "ctx_budget" | "submit" | "stop" | "fix"
  `elicitation` : None | "nudge" | "prefill_inline" | "prefill_failed"
Verified empirically on 3000 rows of a 100-step wiki cell: n_steps == len(trajectory) ==
len(observations) always, and n_steps - llm_calls is 0 or 1 — the 1 being the injected
"STEP BUDGET REACHED" pseudo-step (agent_search/agent/loop.py, the `force_answer` block),
which consumes no model turn. So `llm_calls` is exactly "model turns used", and the
observed max of 101 on a max_steps=100 cell is 100 model turns + 1 nudge pseudo-step.

TRUNCATION RULE (per row, cap B=50)
-----------------------------------
  max_steps <= B          -> row untouched (it never had more than B to begin with)
  max_steps >  B and
      llm_calls <= B      -> row untouched (it finished inside B model turns; the trajectory
                             up to that point is identical under any budget >= it)
  max_steps >  B and
      llm_calls >  B      -> the episode had NOT produced an answer by turn B, so it is
                             scored WRONG (em := False, judge := False), exactly as an
                             unanswered episode is scored today.
This is applied ONLY to the over-budgeted side; a side already at 50 is left alone. On a
MIXED cell (rows at both 50 and 100 — several exist, see the audit) the rule applies per
row, which is the only correct treatment.

THE TWO CAVEATS, STATED PLAINLY (both make the truncated arm a LOWER bound)
--------------------------------------------------------------------------
1. A REAL 50-step run force-answers at the cap: the loop injects the budget nudge on the
   final turn, and `_elicit_inline` / the offline `force_answer_backfill` prefill then
   extract a best-effort answer. Some of those forced answers are correct. Strict
   truncation credits none of them. We therefore ALSO report `optimistic_em`: the strict
   number plus the truncated-away episodes credited at the SAME cell's own measured EM
   among its genuinely budget-exhausted episodes (`stopped` in {max_steps, ctx_budget}) —
   i.e. the empirical accuracy of a forced answer after a full grind in that cell. That is
   a point estimate only; no p-value is attached to it because it is not per-instance.
2. The system prompt tells the agent its budget verbatim (agent_search/prompts/tasks/
   research.md:37 — "You have a BUDGET of {{step_budget}} tool calls for this question.
   Pace yourself"). A 100-step agent was told 100 and paced for 100. A genuine 50-step run
   would have been told 50 and would have committed earlier. Truncation cannot simulate
   that re-pacing, so it penalises the 100-step arm harder than a real 50-step run would.
So the true matched-budget value for an over-budgeted arm lies between the strict-truncated
number (floor) and its published 100-step number (ceiling); `optimistic_em` sits inside.

METHOD PARITY (no reimplementation of EM or McNemar)
----------------------------------------------------
  scripts.compare_cells : cell_dir, cell_rows, metrics, mcnemar_p, load_qrels,
                          load_judge_cache, pct, _qid_of, _compute_cell
  analysis.ablation_deltas : paired, paired_judge, JUDGE_COVERAGE_MIN, FULL
  evaluation.metrics       : answer_em (transitively, via the above)
  scripts.force_answer_backfill : load_rows_with_recovery (transitively — the recovery
                                  overlay is applied before scoring, as everywhere else)
Per-instance scores come from `scripts.compare_cells._compute_cell`, the STREAMING,
cache-backed path (the big-file-safe twin of `metrics()`; its docstring documents the
equivalence). `--verify-equivalence` re-derives one cell through the non-streaming
`metrics(cell_rows(...))` path and asserts the two agree instance-for-instance.

MANDATORY SANITY GATE: reproduces the published Sieve vs "No dense evidence" result on
browsecomp_plus_structured (+4.7 EM, exact McNemar p=0.0104) before any new number is
trusted. A gate failure aborts the run unless --no-gate.

Reads runs/ only. Writes analysis/matched_budget_reanalysis{_data.json,.md} and a private
per-cell scan cache under analysis/.budget_cache/. Never writes under runs/, latex/,
latex_acl8/.

    PYTHONPATH=. envs/bin/python analysis/matched_budget_reanalysis.py
    PYTHONPATH=. envs/bin/python analysis/matched_budget_reanalysis.py --verify-equivalence
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_cells import (  # noqa: E402
    MODEL_DIR, REGISTRY, _compute_cell, _qid_of, _read_cache, cell_dir, cell_rows,
    load_judge_cache, load_qrels, mcnemar_p, metrics, pct,
)
from analysis.ablation_deltas import JUDGE_COVERAGE_MIN  # noqa: E402

CAP = 50                      # the matched budget: the smaller of the two budgets in use
BUDGET_CACHE = Path("analysis/.budget_cache")
_BUDGET_CACHE_VERSION = 1

SIEVE = ("_headline_validation", "agent_research_bql_dense_snip")
BASELINE = ("_visit_uncapped", "agent_research_bm25")
HYBRID = ("_headline_validation", "agent_research_hybrid_fetch_snip")
NODENSE = ("_headline_validation", "agent_research_snip")

WIKI = ["hotpotqa_structured", "musique_structured"]
ALL_DATASETS = ["browsecomp_plus_structured", "browsecomp_plus_flat",
                "hotpotqa_structured", "hotpotqa_flat",
                "musique_structured", "musique_flat"]

# Published numbers, for side-by-side reporting. Sources, verbatim:
#   Sieve vs baseline      -> comparison_result.md (scripts/compare_cells.py output)
#   Sieve vs hybrid        -> analysis/hybrid_control_ci_data.json ["gate"]["per_dataset"]
#   Sieve vs no-dense      -> analysis/no_dense_evidence_3way_data.json
#   structured vs flat     -> analysis/flat_vs_structured_data.json ["pairs"]
PUBLISHED = {
    ("sieve_vs_baseline", "browsecomp_plus_structured"): (2.5, 0.217),
    ("sieve_vs_baseline", "hotpotqa_structured"): (1.6, 0.00186),
    ("sieve_vs_baseline", "musique_structured"): (2.7, 0.00073),
    ("sieve_vs_hybrid", "browsecomp_plus_structured"): (3.2530120481927653, 0.10194275483952327),
    ("sieve_vs_hybrid", "hotpotqa_structured"): (5.746969903309271, 1.6227238882340592e-27),
    ("sieve_vs_hybrid", "musique_structured"): (5.520963055209631, 1.581553928649189e-11),
    ("sieve_vs_nodense", "browsecomp_plus_structured"): (4.69879518072289, 0.010420769409438875),
    ("sieve_vs_nodense", "hotpotqa_structured"): (0.7898679014026939, 0.10322411349940606),
    ("sieve_vs_nodense", "musique_structured"): (1.2868410128684076, 0.09491641241975839),
    ("structured_vs_flat", "browsecomp_plus_structured"): (1.5662650602409585, 0.3724648239355135),
    ("structured_vs_flat", "hotpotqa_structured"): (1.2665123246629406, 0.00977857377510718),
    ("structured_vs_flat", "musique_structured"): (1.1623080116230788, 0.13228725863201551),
}

GATE = dict(key="sieve_vs_nodense", dataset="browsecomp_plus_structured",
            expect_delta=4.69879518072289, expect_p=0.010420769409438875,
            tol_delta=0.05, tol_p_rel=0.02)


# --------------------------------------------------------------------------------------
# per-row budget scan (the ONLY thing compare_cells' own cache does not already carry)
# --------------------------------------------------------------------------------------

def _bcache_path(cond_dir: Path) -> Path:
    return BUDGET_CACHE / (hashlib.sha1(str(cond_dir).encode()).hexdigest() + ".json")


def budget_info(cond_dir: Path) -> dict:
    """{instance_id: [max_steps, n_steps, llm_calls, stopped, elicitation]} for one cell.

    Streams rows.jsonl line-by-line and keeps only these six scalars per row, so memory is
    flat regardless of file size (some of these files are >1GB). Cached to a sidecar under
    analysis/.budget_cache/ keyed on the file's byte size (rows.jsonl is append-only with
    unique instance_ids, so an unchanged size means unchanged content — the same invariant
    scripts/compare_cells.py's own cache relies on); a size change forces a full re-scan.
    NEVER writes under runs/."""
    rows_path = cond_dir / "rows.jsonl"
    if not rows_path.exists():
        return {}
    size = rows_path.stat().st_size
    cp = _bcache_path(cond_dir)
    if cp.exists():
        try:
            payload = json.loads(cp.read_text())
            if payload.get("version") == _BUDGET_CACHE_VERSION and payload.get("n_bytes") == size:
                return payload["info"]
        except (OSError, json.JSONDecodeError, TypeError, KeyError):
            pass
    info = {}
    with rows_path.open() as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue  # torn/corrupt line: skipped exactly like load_rows_tolerant does
            iid = r.get("instance_id") or ""
            if not iid:
                continue
            n_steps = r.get("n_steps")
            info[iid] = [r.get("max_steps"), n_steps,
                         r.get("llm_calls") if r.get("llm_calls") is not None else n_steps,
                         r.get("stopped"), r.get("elicitation")]
    try:
        BUDGET_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name(cp.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"version": _BUDGET_CACHE_VERSION, "n_bytes": size,
                                   "info": info}))
        tmp.replace(cp)
    except OSError:
        pass
    return info


def turns_used(rec) -> int:
    """Model turns consumed by an episode = `llm_calls` (n_steps minus the injected
    zero-call budget-nudge pseudo-step, when one fired). Falls back to n_steps."""
    lc, ns = rec[2], rec[1]
    v = lc if lc is not None else ns
    return int(v) if v is not None else 10**9   # unknown -> treat as "ran past any cap"


def row_budget(rec) -> int:
    ms = rec[0]
    return int(ms) if ms is not None else -1


# --------------------------------------------------------------------------------------
# cell loading + truncation
# --------------------------------------------------------------------------------------

def cell(dataset: str, subdir: str, cond: str, qrels: dict):
    """(per_instance_metrics, judge_coverage, budget_info) for one cell, or (None, None, None).

    `per_instance_metrics` comes from scripts.compare_cells._compute_cell — the streaming,
    recovery-overlaid, answer_em-scored per-instance dict (identical in shape and value to
    compare_cells.metrics(cell_rows(...)); see --verify-equivalence)."""
    cdir = cell_dir(subdir, dataset, cond)
    exists, m = _compute_cell(str(cdir), dataset, qrels, True)
    if not exists or not m:
        return None, None, None
    n = len(m)
    cov = sum(1 for v in m.values() if v["judge"] is not None) / n if n else 0.0
    return m, cov, budget_info(cdir)


def truncate(m: dict, binfo: dict, cap: int = CAP):
    """Replay every episode as if the cap had been `cap`. Returns (new_metrics, stats)."""
    out, n_over_budget, n_cut, n_cut_correct, n_cut_judged_correct = {}, 0, 0, 0, 0
    for iid, v in m.items():
        rec = binfo.get(iid)
        cut = False
        if rec is not None and row_budget(rec) > cap:
            n_over_budget += 1
            if turns_used(rec) > cap:
                cut = True
        if cut:
            n_cut += 1
            n_cut_correct += bool(v["em"])
            n_cut_judged_correct += bool(v["judge"])
            out[iid] = {**v, "em": False,
                        "judge": (False if v["judge"] is not None else None)}
        else:
            out[iid] = v
    stats = dict(n=len(m), n_over_budget=n_over_budget, n_truncated=n_cut,
                 pct_over_budget=100.0 * n_over_budget / len(m) if m else 0.0,
                 pct_truncated=100.0 * n_cut / len(m) if m else 0.0,
                 n_truncated_correct=n_cut_correct,
                 pct_of_all_em_lost=(100.0 * n_cut_correct / sum(1 for v in m.values() if v["em"])
                                     if any(v["em"] for v in m.values()) else 0.0),
                 n_truncated_judged_correct=n_cut_judged_correct,
                 em_lost_pp=100.0 * n_cut_correct / len(m) if m else 0.0)
    return out, stats


def forced_answer_em(m: dict, binfo: dict) -> dict:
    """EM among episodes that genuinely EXHAUSTED their own budget (`stopped` in
    {max_steps, ctx_budget}) — i.e. the measured accuracy of this cell's force-answered
    episodes. Used as the optimistic credit rate for truncated-away episodes."""
    ids = [i for i, rec in binfo.items()
           if i in m and str(rec[3]) in ("max_steps", "ctx_budget")]
    return dict(n=len(ids), em_pct=pct([m[i]["em"] for i in ids]) if ids else float("nan"))


def paired_em(a: dict, b: dict, dataset_a: str = "", dataset_b: str = ""):
    """Paired EM delta (a - b) + exact McNemar on the shared instance set.

    Pairing key is the raw instance_id when both cells are the same dataset, and the
    dataset-stripped qid (compare_cells._qid_of) when they are not — which is what the
    structured-vs-flat contrast needs, since ids carry a dataset prefix."""
    if dataset_a and dataset_b and dataset_a != dataset_b:
        ka = {_qid_of(i, dataset_a): v for i, v in a.items()}
        kb = {_qid_of(i, dataset_b): v for i, v in b.items()}
    else:
        ka, kb = a, b
    mut = set(ka) & set(kb)
    if not mut:
        return None
    nb = sum(1 for i in mut if kb[i]["em"] and not ka[i]["em"])
    nc = sum(1 for i in mut if ka[i]["em"] and not kb[i]["em"])
    em_a, em_b = pct([ka[i]["em"] for i in mut]), pct([kb[i]["em"] for i in mut])
    return dict(n_shared=len(mut), em_a=em_a, em_b=em_b, delta=em_a - em_b,
                b_disc=nb, c_disc=nc, mcnemar_p=mcnemar_p(nb, nc))


def paired_judge_em(a, b, cov_a, cov_b, dataset_a="", dataset_b=""):
    if cov_a is None or cov_b is None or cov_a < JUDGE_COVERAGE_MIN or cov_b < JUDGE_COVERAGE_MIN:
        return None
    if dataset_a and dataset_b and dataset_a != dataset_b:
        ka = {_qid_of(i, dataset_a): v for i, v in a.items()}
        kb = {_qid_of(i, dataset_b): v for i, v in b.items()}
    else:
        ka, kb = a, b
    mut = [i for i in set(ka) & set(kb)
           if ka[i]["judge"] is not None and kb[i]["judge"] is not None]
    if not mut:
        return None
    nb = sum(1 for i in mut if kb[i]["judge"] and not ka[i]["judge"])
    nc = sum(1 for i in mut if ka[i]["judge"] and not kb[i]["judge"])
    ja, jb = pct([ka[i]["judge"] for i in mut]), pct([kb[i]["judge"] for i in mut])
    return dict(n_shared=len(mut), judge_a=ja, judge_b=jb, delta=ja - jb,
                b_disc=nb, c_disc=nc, mcnemar_p=mcnemar_p(nb, nc))


# --------------------------------------------------------------------------------------
# budget audit over every cell
# --------------------------------------------------------------------------------------

_MS_RE = re.compile(rb'"max_steps": (\d+)')


def audit_max_steps(rows_path: Path) -> dict:
    """{max_steps_value: n_rows} for a cell, via a byte-level scan (no JSON parsing) so it
    stays cheap even on the 15-31GB auto-read cells. `"max_steps": N` occurs exactly once
    per row; validated against the JSON-parsed budget_info counts for every comparison cell
    (see `validate` in the emitted JSON)."""
    c = Counter()
    with rows_path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            # a row is far smaller than 16MB and the pattern is short; re-read a small
            # overlap so a pattern straddling a chunk boundary is not missed
            tail = fh.read(64)
            if tail:
                fh.seek(-64, os.SEEK_CUR)
            for mo in _MS_RE.finditer(chunk + (tail or b"")):
                c[int(mo.group(1))] += 1
    return dict(c)


def build_audit(datasets):
    """Per-cell budget audit for every REGISTRY cell in `datasets`, plus the flat twins."""
    seen, out = set(), []
    cells = [(lbl, sub, ds, cond) for lbl, sub, ds, cond, _ in REGISTRY if ds in datasets]
    for ds in datasets:  # flat twins are registry-listed only for browsecomp
        cells.append(("SERP bm25 [BASELINE]", "_visit_uncapped", ds, "agent_research_bm25"))
    for lbl, sub, ds, cond in cells:
        cdir = cell_dir(sub, ds, cond)
        if str(cdir) in seen:
            continue
        seen.add(str(cdir))
        rp = cdir / "rows.jsonl"
        if not rp.exists():
            continue
        dist = audit_max_steps(rp)
        rec = dict(label=lbl, dataset=ds, subdir=sub, condition=cond, dir=str(cdir),
                   n_rows=sum(dist.values()),
                   max_steps_dist={str(k): v for k, v in sorted(dist.items())},
                   mixed=len(dist) > 1, size_bytes=rp.stat().st_size)
        # step-usage stats, but only from a WARM compare_cells cache — never pay a
        # multi-GB parse just for the audit table
        cached = _read_cache(cdir)
        if cached and cached.get("exists") and cached.get("n_bytes") == rec["size_bytes"]:
            calls = [v.get("llm_calls") or 0 for v in (cached.get("intrinsic") or {}).values()]
            if calls:
                rec.update(mean_llm_calls=sum(calls) / len(calls), max_llm_calls=max(calls),
                           pct_calls_gt_cap=100.0 * sum(1 for c in calls if c > CAP) / len(calls))
        out.append(rec)
    return out


# --------------------------------------------------------------------------------------
# the comparisons
# --------------------------------------------------------------------------------------

def comparison_specs():
    """(key, title, dataset_a, arm_a, dataset_b, arm_b) — delta is always arm_a - arm_b."""
    specs = []
    for ds in ["browsecomp_plus_structured"] + WIKI:
        specs.append(("sieve_vs_baseline", "Sieve vs SERP-bm25 baseline", ds, SIEVE, ds, BASELINE))
        specs.append(("sieve_vs_hybrid", "Sieve vs hybrid control", ds, SIEVE, ds, HYBRID))
        specs.append(("sieve_vs_nodense", "Sieve vs 'No dense evidence'", ds, SIEVE, ds, NODENSE))
    for ds in ["browsecomp_plus_structured"] + WIKI:
        specs.append(("structured_vs_flat", "structured corpus vs flat twin (same bm25 agent)",
                      ds, BASELINE, ds.replace("_structured", "_flat"), BASELINE))
    return specs


def run_comparisons(cache: dict):
    results = []
    for key, title, ds_a, arm_a, ds_b, arm_b in comparison_specs():
        ma, cova, ba = cache[(ds_a, arm_a)]
        mb, covb, bb = cache[(ds_b, arm_b)]
        if ma is None or mb is None:
            results.append(dict(key=key, title=title, dataset=ds_a, error="cell missing"))
            continue
        ta, sa = truncate(ma, ba)
        tb, sb = truncate(mb, bb)
        pub_d, pub_p = PUBLISHED.get((key, ds_a), (None, None))
        entry = dict(
            key=key, title=title, dataset=ds_a,
            arm_a=dict(dataset=ds_a, subdir=arm_a[0], condition=arm_a[1],
                       budget_dist=Counter(str(row_budget(r)) for r in ba.values()),
                       trunc=sa, forced=forced_answer_em(ma, ba)),
            arm_b=dict(dataset=ds_b, subdir=arm_b[0], condition=arm_b[1],
                       budget_dist=Counter(str(row_budget(r)) for r in bb.values()),
                       trunc=sb, forced=forced_answer_em(mb, bb)),
            published=dict(delta=pub_d, p=pub_p),
            as_published=paired_em(ma, mb, ds_a, ds_b),
            matched=paired_em(ta, tb, ds_a, ds_b),
            affected=bool(sa["n_over_budget"] or sb["n_over_budget"]),
        )
        entry["arm_a"]["budget_dist"] = dict(entry["arm_a"]["budget_dist"])
        entry["arm_b"]["budget_dist"] = dict(entry["arm_b"]["budget_dist"])
        j_pub = paired_judge_em(ma, mb, cova, covb, ds_a, ds_b)
        j_mat = paired_judge_em(ta, tb, cova, covb, ds_a, ds_b)
        entry["as_published_judge"], entry["matched_judge"] = j_pub, j_mat
        # optimistic point estimate: credit truncated-away episodes at the cell's own
        # measured forced-answer EM rate (no p-value — not a per-instance assignment)
        def optimistic(strict, st, forced, n_shared):
            if not st["n_truncated"] or math.isnan(forced["em_pct"]):
                return strict
            return strict + (st["n_truncated"] * forced["em_pct"] / 100.0) * 100.0 / n_shared
        if entry["matched"]:
            ns = entry["matched"]["n_shared"]
            oa = optimistic(entry["matched"]["em_a"], sa, entry["arm_a"]["forced"], ns)
            ob = optimistic(entry["matched"]["em_b"], sb, entry["arm_b"]["forced"], ns)
            entry["optimistic"] = dict(em_a=oa, em_b=ob, delta=oa - ob)
        results.append(entry)
    return results


# --------------------------------------------------------------------------------------
# sanity gate + equivalence check
# --------------------------------------------------------------------------------------

def sanity_gate(results) -> dict:
    for r in results:
        if r.get("key") == GATE["key"] and r.get("dataset") == GATE["dataset"]:
            got = r["as_published"]
            dd = abs(got["delta"] - GATE["expect_delta"])
            dp = abs(got["mcnemar_p"] - GATE["expect_p"]) / max(GATE["expect_p"], 1e-300)
            ok = dd <= GATE["tol_delta"] and dp <= GATE["tol_p_rel"]
            return dict(passed=bool(ok), expected_delta=GATE["expect_delta"],
                        got_delta=got["delta"], expected_p=GATE["expect_p"],
                        got_p=got["mcnemar_p"], n_shared=got["n_shared"],
                        delta_abs_err=dd, p_rel_err=dp,
                        note=("reproduced the published Sieve vs 'No dense evidence' result on "
                              "browsecomp_plus_structured through this script's own loading path"))
    return dict(passed=False, note="gate comparison not found in results")


def verify_equivalence(dataset: str, subdir: str, cond: str, qrels: dict) -> dict:
    """Assert _compute_cell (streaming/cached) == metrics(cell_rows(...)) (from-scratch)
    instance-for-instance on em/judge for one cell — the method-parity proof."""
    cdir = cell_dir(subdir, dataset, cond)
    _, m_stream = _compute_cell(str(cdir), dataset, qrels, True)
    m_direct = metrics(cell_rows(subdir, dataset, cond), qrels, dataset, load_judge_cache(cdir))
    same_ids = set(m_stream) == set(m_direct)
    diffs = [i for i in set(m_stream) & set(m_direct)
             if m_stream[i]["em"] != m_direct[i]["em"]
             or m_stream[i]["judge"] != m_direct[i]["judge"]]
    return dict(cell=str(cdir), n=len(m_stream), same_id_sets=same_ids,
                n_disagreements=len(diffs), passed=bool(same_ids and not diffs))


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

def _f(x, spec="+.2f"):
    return "—" if x is None else format(x, spec)


def _p(x):
    return "—" if x is None else (f"{x:.3g}")


def render(data) -> str:
    L = []
    A = L.append
    A("# Matched-step-budget re-analysis (trajectory truncation, zero new compute)\n")
    A(f"Generated by `analysis/matched_budget_reanalysis.py` at git rev `{data['git_rev']}`. "
      f"Matched cap **B = {CAP}** model turns.\n")

    g = data["sanity_gate"]
    A("## 0. Mandatory sanity gate\n")
    A(f"**{'PASSED' if g['passed'] else 'FAILED'}** — reproduce the published Sieve vs "
      f"\"No dense evidence\" result on browsecomp_plus_structured through this script's own "
      f"loading path.\n")
    A("| quantity | published | recomputed here |")
    A("|---|---:|---:|")
    A(f"| EM delta (pp) | {g['expected_delta']:+.4f} | {g['got_delta']:+.4f} |")
    A(f"| exact McNemar p | {g['expected_p']:.6g} | {g['got_p']:.6g} |")
    A(f"| n_shared | 830 | {g['n_shared']} |")
    if data.get("equivalence"):
        e = data["equivalence"]
        A(f"\nMethod-parity check (`--verify-equivalence`): streaming `_compute_cell` vs "
          f"from-scratch `metrics(cell_rows(...))` on `{e['cell']}` — n={e['n']}, "
          f"identical id sets: {e['same_id_sets']}, disagreements: {e['n_disagreements']} → "
          f"**{'PASSED' if e['passed'] else 'FAILED'}**.\n")

    A("\n## 1. Per-cell budget audit\n")
    A("Per-row `max_steps` read out of every `rows.jsonl`. `mixed` = the cell contains rows "
      "run under BOTH budgets. `>cap%` = share of episodes that consumed more than "
      f"{CAP} model turns (only these can be changed by truncation); it is computed from "
      "`scripts/compare_cells.py`'s warm per-cell cache and is blank where that cache is cold.\n")
    for ds in ALL_DATASETS:
        rows = [r for r in data["audit"] if r["dataset"] == ds]
        if not rows:
            continue
        A(f"\n### {ds}\n")
        A("| cell | subdir/condition | n | max_steps distribution | mixed | mean turns | max turns | >cap% |")
        A("|---|---|---:|---|:-:|---:|---:|---:|")
        for r in sorted(rows, key=lambda r: (r["subdir"], r["condition"])):
            dist = ", ".join(f"{v}×{k}" for k, v in r["max_steps_dist"].items())
            A(f"| {r['label']} | `{r['subdir']}/{r['condition']}` | {r['n_rows']} | {dist} | "
              f"{'**YES**' if r['mixed'] else ''} | "
              f"{r.get('mean_llm_calls', float('nan')):.1f} | {r.get('max_llm_calls', 0)} | "
              f"{r.get('pct_calls_gt_cap', float('nan')):.1f} |")

    A("\n## 2. Matched-budget re-analysis\n")
    A("`as published` = the numbers in the paper today (unequal budgets). `matched@50` = the "
      "over-budgeted side's trajectories cut at 50 model turns, episodes that had not answered "
      "by then scored wrong; the already-50 side untouched. `optimistic@50` credits the "
      "truncated-away episodes at that same cell's measured EM among its own budget-exhausted "
      "(force-answered) episodes — a point estimate, no p-value.\n")
    A("| comparison | dataset | budgets (A vs B) | n_shared | published Δ | published p | "
      "as-published Δ | matched@50 Δ | matched@50 p | optimistic@50 Δ | truncated A | truncated B |")
    A("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in data["comparisons"]:
        if r.get("error"):
            A(f"| {r['title']} | {r['dataset']} | — | | | | | | | | | (missing) |")
            continue
        bd = lambda arm: "/".join(f"{v}×{k}" for k, v in sorted(arm["budget_dist"].items()))
        pub = r["published"]
        ap, mt = r["as_published"], r["matched"]
        opt = r.get("optimistic") or {}
        A(f"| {r['title']} | {r['dataset']} | {bd(r['arm_a'])} vs {bd(r['arm_b'])} | "
          f"{ap['n_shared']} | {_f(pub['delta'])} | {_p(pub['p'])} | {_f(ap['delta'])} | "
          f"{_f(mt['delta'])} | {_p(mt['mcnemar_p'])} | {_f(opt.get('delta'))} | "
          f"{r['arm_a']['trunc']['n_truncated']} ({r['arm_a']['trunc']['pct_truncated']:.1f}%) | "
          f"{r['arm_b']['trunc']['n_truncated']} ({r['arm_b']['trunc']['pct_truncated']:.1f}%) |")

    A("\n### Per-comparison detail\n")
    for r in data["comparisons"]:
        if r.get("error"):
            continue
        ap, mt = r["as_published"], r["matched"]
        A(f"\n**{r['title']} — {r['dataset']}**  \n"
          f"A = `{r['arm_a']['subdir']}/{r['arm_a']['condition']}` "
          f"({r['arm_a']['dataset']}), B = `{r['arm_b']['subdir']}/{r['arm_b']['condition']}` "
          f"({r['arm_b']['dataset']}). Budget-affected: {r['affected']}.\n")
        A("| | EM_A | EM_B | Δ | McNemar p | discordant b/c |")
        A("|---|---:|---:|---:|---:|---:|")
        A(f"| as published | {ap['em_a']:.2f} | {ap['em_b']:.2f} | {ap['delta']:+.2f} | "
          f"{ap['mcnemar_p']:.3g} | {ap['b_disc']}/{ap['c_disc']} |")
        A(f"| matched@{CAP} (strict) | {mt['em_a']:.2f} | {mt['em_b']:.2f} | {mt['delta']:+.2f} | "
          f"{mt['mcnemar_p']:.3g} | {mt['b_disc']}/{mt['c_disc']} |")
        if r.get("optimistic"):
            o = r["optimistic"]
            A(f"| optimistic@{CAP} | {o['em_a']:.2f} | {o['em_b']:.2f} | {o['delta']:+.2f} | — | — |")
        for tag in ("arm_a", "arm_b"):
            s, fo = r[tag]["trunc"], r[tag]["forced"]
            A(f"\n- {tag}: {s['n']} episodes, {s['n_over_budget']} ran under a budget > {CAP} "
              f"({s['pct_over_budget']:.1f}%), of which {s['n_truncated']} used more than {CAP} "
              f"turns and are truncated away ({s['pct_truncated']:.1f}% of the cell). Those "
              f"carried {s['n_truncated_correct']} correct answers "
              f"= {s['pct_of_all_em_lost']:.1f}% of the cell's correct answers, "
              f"= {s['em_lost_pp']:.2f} EM points. "
              f"Forced-answer EM within this cell (budget-exhausted episodes, n={fo['n']}): "
              + (f"{fo['em_pct']:.1f}%." if not math.isnan(fo['em_pct']) else "n/a."))
        if r.get("as_published_judge"):
            jp, jm = r["as_published_judge"], r.get("matched_judge")
            A(f"\n- judge: as published Δ={jp['delta']:+.2f} (p={jp['mcnemar_p']:.3g}, "
              f"n={jp['n_shared']})"
              + (f"; matched@{CAP} Δ={jm['delta']:+.2f} (p={jm['mcnemar_p']:.3g})" if jm else ""))

    A("\n## 3. BrowseComp-Plus flat/structured self-replicate — a noise-floor estimate\n")
    nf = data["noise_floor"]
    A(nf["prose"] + "\n")
    A("| metric | structured | flat | Δ (structured − flat) | exact McNemar p | n |")
    A("|---|---:|---:|---:|---:|---:|")
    A(f"| EM | {nf['em']['em_a']:.2f} | {nf['em']['em_b']:.2f} | {nf['em']['delta']:+.2f} | "
      f"{nf['em']['mcnemar_p']:.3g} | {nf['em']['n_shared']} |")
    if nf.get("judge"):
        j = nf["judge"]
        A(f"| judge | {j['judge_a']:.2f} | {j['judge_b']:.2f} | {j['delta']:+.2f} | "
          f"{j['mcnemar_p']:.3g} | {j['n_shared']} |")
    A("")
    return "\n".join(L)


def main():
    global CAP
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cap", type=int, default=CAP, help="matched step cap (default 50)")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not abort when the sanity gate fails (still reported)")
    ap.add_argument("--verify-equivalence", action="store_true",
                    help="also assert _compute_cell == metrics(cell_rows(...)) on one cell")
    ap.add_argument("--skip-audit", action="store_true", help="skip the per-cell budget audit scan")
    ap.add_argument("--json-out", default="analysis/matched_budget_reanalysis_data.json")
    ap.add_argument("--md-out", default="analysis/matched_budget_reanalysis.md")
    args = ap.parse_args()
    CAP = args.cap

    try:
        git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True, cwd=REPO_ROOT).stdout.strip() or "n/a"
    except OSError:
        git_rev = "n/a"

    qrels = {ds: load_qrels(ds) for ds in ALL_DATASETS}

    need = set()
    for _, _, ds_a, arm_a, ds_b, arm_b in comparison_specs():
        need.add((ds_a, arm_a))
        need.add((ds_b, arm_b))
    cache = {}
    for ds, arm in sorted(need):
        print(f"[load] {ds} {arm[0]}/{arm[1]}", file=sys.stderr, flush=True)
        cache[(ds, arm)] = cell(ds, arm[0], arm[1], qrels[ds])

    results = run_comparisons(cache)
    gate = sanity_gate(results)
    print(f"[gate] {'PASSED' if gate['passed'] else 'FAILED'}: delta {gate.get('got_delta')} "
          f"vs {gate.get('expected_delta')}, p {gate.get('got_p')} vs {gate.get('expected_p')}",
          file=sys.stderr)
    if not gate["passed"] and not args.no_gate:
        sys.exit("SANITY GATE FAILED — refusing to emit numbers. Re-run with --no-gate to override.")

    equiv = None
    if args.verify_equivalence:
        equiv = verify_equivalence("browsecomp_plus_structured", *SIEVE,
                                   qrels["browsecomp_plus_structured"])
        print(f"[equiv] {equiv}", file=sys.stderr)

    audit = [] if args.skip_audit else build_audit(ALL_DATASETS)

    nf_entry = next(r for r in results
                    if r["key"] == "structured_vs_flat"
                    and r["dataset"] == "browsecomp_plus_structured")
    noise = dict(
        em=nf_entry["as_published"], judge=nf_entry.get("as_published_judge"),
        both_budgets=dict(structured=nf_entry["arm_a"]["budget_dist"],
                          flat=nf_entry["arm_b"]["budget_dist"]),
        prose=("Both arms ran at max_steps=100, so this contrast is budget-clean. The two arms "
               "are the SAME agent (BM25 search + whole-document visit) over corpora whose "
               "searchable/renderable `text` is byte-identical, so — subject to the code audit "
               "in the report — this pair is one experiment run twice, and its gap is a "
               "run-to-run NOISE estimate at n=830, not a corpus effect. It is a single "
               "replicate pair, so it pins the noise floor only crudely."))

    data = dict(git_rev=git_rev, cap=CAP, sanity_gate=gate, equivalence=equiv,
                audit=audit, comparisons=results, noise_floor=noise, published=
                {f"{k[0]}|{k[1]}": dict(delta=v[0], p=v[1]) for k, v in PUBLISHED.items()})
    Path(args.json_out).write_text(json.dumps(data, indent=1, default=str))
    md = render(data)
    Path(args.md_out).write_text(md)
    print(md)
    print(f"\nwrote: {args.json_out} {args.md_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
