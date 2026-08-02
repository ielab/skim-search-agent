#!/usr/bin/env python
"""Paired significance test: full method vs the DCI (closed-book, no-retrieval) baseline.

Closes a reviewer Minor: the paper compares its method to the DCI baseline in the results
table but never reports a paired significance test between the two. This script computes,
for every dataset, the paired EM delta (exact McNemar on discordant pairs, same-instance
design per docs/03-agent-evaluation.md) between:

    full method : runs/_headline_validation/agent/<ds>/Tongyi-DeepResearch-30B-A3B/
                   agent_research_bql_dense_snip/rows.jsonl
    dci         : runs/agent/<ds>/Tongyi-DeepResearch-30B-A3B/agent_research_dci/rows.jsonl

for ds in {browsecomp_plus_structured, hotpotqa_structured, musique_structured}, plus the
BM25-filtered DCI variant (agent_research_bm25_dci) wherever it exists (currently browsecomp
only — checked at runtime, not hardcoded, so this script self-updates if that changes).

MEMORY: the dci rows.jsonl files are LARGE (~1.4GB browsecomp, ~1.9GB hotpotqa, ~1.6GB
musique) and carry heavy per-row fields (trajectory, observations) this script does not
need. `load_rows_with_recovery` (scripts/force_answer_backfill) materializes the WHOLE file
as a list of full dicts, which would multiply each file's on-disk size several-fold in
memory once JSON-parsed — fine for the method cells (188-385MB) but not for the multi-GB dci
cells. So: the method side uses `load_rows_with_recovery` directly (the intended primitive,
small enough to load whole); the dci side uses a hand-rolled STREAMING reader
(`_stream_cell_metrics` below) that reads rows.jsonl one line at a time, JSON-parses ONE row,
extracts only instance_id/final_answer/gold_answer, applies the exact same recovery-overlay
and judge-lookup semantics as `scripts/compare_cells.py`, then discards the row before
reading the next line — peak memory is O(one row), not O(file size), regardless of cell size.
The overlay/judge semantics are pulled from the same primitives compare_cells.py itself uses
(`needs_recovery`, `load_rows_tolerant`, `answer_em`, `load_judge_cache`, `mcnemar_p`) so the
two code paths can never silently disagree on what counts as empty/recovered/judged.

Usage:
    PYTHONPATH=. envs/bin/python analysis/method_vs_dci.py
"""
from __future__ import annotations

import json
import sys
from hashlib import sha1
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import answer_em  # noqa: E402
from scripts.compare_cells import load_judge_cache, mcnemar_p  # noqa: E402
from scripts.force_answer_backfill import (  # noqa: E402
    load_rows_tolerant, load_rows_with_recovery, needs_recovery,
)

MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"
DATASETS = ["browsecomp_plus_structured", "hotpotqa_structured", "musique_structured"]
METHOD_COND = "agent_research_bql_dense_snip"
METHOD_SUBDIR = "_headline_validation"
DCI_CONDS = ["agent_research_dci", "agent_research_bm25_dci"]  # bm25_dci checked per-dataset
JUDGE_COVERAGE_THRESHOLD = 0.90

OUT_MD = Path(__file__).resolve().parent / "method_vs_dci_data.json"


def method_dir(ds: str) -> Path:
    return Path("runs") / METHOD_SUBDIR / "agent" / ds / MODEL_DIR / METHOD_COND


def dci_dir(ds: str, cond: str) -> Path:
    return Path("runs") / "agent" / ds / MODEL_DIR / cond


def _load_recovered_map(cond_dir: Path) -> dict:
    """instance_id -> recovered_answer for GENUINE recoveries only (same semantics as
    scripts/force_answer_backfill.load_rows_with_recovery's overlay filter and
    scripts/compare_cells._load_recovered_map). The sidecar itself is small (tens-hundreds of
    KB) regardless of how large rows.jsonl is, so loading it whole is cheap."""
    out: dict = {}
    p = cond_dir / "recovered_answers.jsonl"
    if not p.exists():
        return out
    for rec in load_rows_tolerant(str(p)):
        iid = rec.get("instance_id")
        if iid and not needs_recovery(rec.get("recovered_answer")):
            out[iid] = rec.get("recovered_answer") or ""
    return out


def _stream_cell_metrics(cond_dir: Path) -> dict:
    """instance_id -> {em, empty, recovered, judge} read by STREAMING rows.jsonl one line at a
    time (see module docstring — this is the memory-bounded path for the large dci files).
    Equivalent in outcome to scripts.compare_cells.metrics()/_apply_overlay() run over the same
    cell, but never holds more than one parsed row in memory at once."""
    rows_path = cond_dir / "rows.jsonl"
    if not rows_path.exists():
        return {}
    recovered_by_id = _load_recovered_map(cond_dir)
    judge_cache = load_judge_cache(cond_dir)
    out: dict = {}
    with rows_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # unparsable trailing line from a live/torn write; skip like _load_rows
            iid = r.get("instance_id") or ""
            gold = str(r.get("gold_answer") or "")
            ans_raw = str(r.get("final_answer") or "")
            recovered = False
            ans = ans_raw
            if needs_recovery(ans_raw):
                rec = recovered_by_id.get(iid)
                if rec is not None:
                    ans, recovered = rec, True
            em = bool(answer_em(ans, gold))
            empty = needs_recovery(ans)
            judge = judge_cache.get((iid, sha1(ans.strip().encode()).hexdigest()))
            out[iid] = dict(em=em, empty=empty, recovered=recovered, judge=judge)
            del r  # drop the heavy fields (trajectory/observations) before the next line
    return out


def _load_recovery_overlaid_metrics(cond_dir: Path) -> dict:
    """instance_id -> {em, empty, recovered, judge} for a cell SMALL enough to load whole via
    `load_rows_with_recovery` (the intended primitive). Used for the method cells only."""
    judge_cache = load_judge_cache(cond_dir)
    out = {}
    for r in load_rows_with_recovery(str(cond_dir)):
        iid = r.get("instance_id") or ""
        gold = str(r.get("gold_answer") or "")
        ans = str(r.get("final_answer") or "")
        out[iid] = dict(
            em=bool(answer_em(ans, gold)),
            empty=needs_recovery(ans),
            recovered=bool(r.get("recovered")),
            judge=judge_cache.get((iid, sha1(ans.strip().encode()).hexdigest())),
        )
    return out


def pct(xs) -> float:
    return 100.0 * sum(xs) / len(xs) if xs else 0.0


def compare(method_m: dict, other_m: dict) -> dict:
    """Paired EM + (conditionally) paired judge comparison between two already-computed metrics
    dicts, mirroring scripts/compare_cells.py's own pairing logic exactly (mutual instance ids,
    exact two-sided McNemar on discordant pairs)."""
    mut = sorted(set(method_m) & set(other_m))
    n_shared = len(mut)
    em_method = pct([method_m[i]["em"] for i in mut])
    em_other = pct([other_m[i]["em"] for i in mut])
    b = sum(1 for i in mut if method_m[i]["em"] and not other_m[i]["em"])  # method-only wins
    c = sum(1 for i in mut if other_m[i]["em"] and not method_m[i]["em"])  # dci-only wins
    p_em = mcnemar_p(b, c)
    result = dict(
        n_shared=n_shared,
        em_method=em_method, em_other=em_other, delta_em=em_method - em_other,
        b_method_only=b, c_other_only=c, p_em=p_em,
    )

    n_method_all = len(method_m)
    n_other_all = len(other_m)
    method_judged = [v["judge"] for v in method_m.values() if v["judge"] is not None]
    other_judged = [v["judge"] for v in other_m.values() if v["judge"] is not None]
    method_cov = len(method_judged) / n_method_all if n_method_all else 0.0
    other_cov = len(other_judged) / n_other_all if n_other_all else 0.0
    result.update(judge_coverage_method=100 * method_cov, judge_coverage_other=100 * other_cov)

    if method_cov >= JUDGE_COVERAGE_THRESHOLD and other_cov >= JUDGE_COVERAGE_THRESHOLD:
        jmut = [i for i in mut if method_m[i]["judge"] is not None and other_m[i]["judge"] is not None]
        if jmut:
            jb = sum(1 for i in jmut if method_m[i]["judge"] and not other_m[i]["judge"])
            jc = sum(1 for i in jmut if other_m[i]["judge"] and not method_m[i]["judge"])
            judge_method = pct([method_m[i]["judge"] for i in jmut])
            judge_other = pct([other_m[i]["judge"] for i in jmut])
            result.update(
                judge_status="computed", n_judge_shared=len(jmut),
                judge_method=judge_method, judge_other=judge_other,
                delta_judge=judge_method - judge_other,
                b_judge_method_only=jb, c_judge_other_only=jc, p_judge=mcnemar_p(jb, jc),
            )
        else:
            result["judge_status"] = "pending (no mutual judged rows)"
    else:
        result["judge_status"] = (
            f"pending (judge coverage {100*method_cov:.1f}%/{100*other_cov:.1f}% "
            f"< {100*JUDGE_COVERAGE_THRESHOLD:.0f}% threshold on one or both sides)"
        )
    return result


def main() -> int:
    all_results = []
    for ds in DATASETS:
        m_dir = method_dir(ds)
        if not (m_dir / "rows.jsonl").exists():
            print(f"SKIP {ds}: method rows.jsonl not found at {m_dir}", file=sys.stderr)
            continue
        print(f"[{ds}] loading method cell ({m_dir}) ...", file=sys.stderr)
        method_m = _load_recovery_overlaid_metrics(m_dir)
        print(f"[{ds}] method: n={len(method_m)}", file=sys.stderr)

        for cond in DCI_CONDS:
            d_dir = dci_dir(ds, cond)
            if not (d_dir / "rows.jsonl").exists():
                if cond == "agent_research_bm25_dci":
                    print(f"[{ds}] {cond}: not present, skipping (expected: browsecomp only)",
                          file=sys.stderr)
                else:
                    print(f"[{ds}] {cond}: MISSING rows.jsonl at {d_dir}", file=sys.stderr)
                continue
            print(f"[{ds}] streaming {cond} cell ({d_dir}) ...", file=sys.stderr)
            other_m = _stream_cell_metrics(d_dir)
            print(f"[{ds}] {cond}: n={len(other_m)}", file=sys.stderr)
            res = compare(method_m, other_m)
            res.update(dataset=ds, other_cond=cond,
                       n_method_total=len(method_m), n_other_total=len(other_m))
            all_results.append(res)
            print(f"[{ds}] vs {cond}: n_shared={res['n_shared']} "
                  f"EM {res['em_method']:.1f} vs {res['em_other']:.1f} "
                  f"(delta={res['delta_em']:+.1f}, p={res['p_em']:.3g}) "
                  f"judge={res['judge_status']}", file=sys.stderr)

    OUT_MD.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {OUT_MD}", file=sys.stderr)

    # --- markdown table to stdout ---------------------------------------------------------
    print("\n| dataset | vs | n_shared | EM method% | EM dci% | ΔEM | p_em (McNemar) | "
          "judge coverage (method/dci) | Δjudge | p_judge |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_results:
        label = {"agent_research_dci": "dci", "agent_research_bm25_dci": "bm25->dci"}[r["other_cond"]]
        if r.get("judge_status") == "computed":
            jd = f"{r['delta_judge']:+.1f}"
            jp = f"{r['p_judge']:.3g}"
        else:
            jd = "pending"
            jp = "pending"
        print(f"| {r['dataset']} | {label} | {r['n_shared']} | {r['em_method']:.1f} | "
              f"{r['em_other']:.1f} | {r['delta_em']:+.1f} | {r['p_em']:.3g} | "
              f"{r['judge_coverage_method']:.1f}%/{r['judge_coverage_other']:.1f}% | {jd} | {jp} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
