#!/usr/bin/env python
"""Round-8 Job 2 -- WIKI COMPONENT ISOLATION. Component-isolation contrasts previously existed
ONLY on browsecomp_plus_structured (analysis/round5_analyses.py), which reviewers repeatedly
flagged as the dataset where the method's effect is weakest -- i.e. the isolation story was only
ever demonstrated on the paper's hardest case. This script computes the SAME family of paired,
same-interface isolation contrasts on hotpotqa_structured (now released) and musique_structured
(cells mostly not yet released -- reported as such, not fabricated), plus a HOTPOTQA-only third
contrast the task brief adds: the direct cost of removing dense fusion from the full method
(full method vs "No dense evidence").

Three contrasts, each a DIRECT paired comparison between two non-baseline cells (never either
cell vs the dataset's SERP-bm25 baseline -- that pairing is comparison_result.md's; never either
cell vs the full method except contrast 1 itself):

  1. dense_fusion_cost   -- full method (bql+dense+snip fetch) vs No dense evidence (bql+snip
                            fetch, no dense fusion): what removing dense fusion costs the full
                            method, holding the structured query surface + snip/fetch interface
                            fixed on both sides.
  2. query_surface_alone -- No dense evidence (bql+snip fetch) vs Sparse-only same interface
                            (bm25+snip fetch): isolates the structured query SURFACE alone,
                            holding the snip+fetch read interface and absence of dense fusion
                            fixed on both sides. Same pairing as round5_analyses.py's M5.
  3. dense_alone         -- Dense-only same interface (dense+snip fetch) vs Sparse-only same
                            interface (bm25+snip fetch): isolates dense retrieval alone, holding
                            the snip+fetch interface fixed and the BQL query surface absent on
                            both sides. Same pairing as round5_analyses.py's M5 dense_fusion.

GATING (this is the whole point of writing this as a re-runnable script rather than a one-shot
number pull): every cell's completion is checked AT RUN TIME by unioning canonical rows.jsonl +
every __shards/<cond>/shard_*of*/.../rows.jsonl found under the cell's run group, deduped by
instance_id, against data/<dataset>/queries.jsonl's line count. A cell below COMPLETE_THRESHOLD
(90%) of the dataset's total n is SKIPPED -- reported as "PARTIAL, not computed" with its live
row count, never computed on the partial subset. A cell with a merged rows.jsonl (canonical
file present) is treated as released and used directly via scripts.compare_cells.cell_rows,
which applies the recovery overlay.

Reuses (does not reimplement):
  - evaluation.metrics.answer_em                           -- canonical EM, via compare_cells.metrics()
  - scripts.force_answer_backfill.load_rows_with_recovery / load_rows_tolerant / needs_recovery
  - scripts.compare_cells.cell_dir / cell_rows / metrics / mcnemar_p / pct / load_qrels /
    load_judge_cache

Run: PYTHONPATH=. envs/bin/python analysis/wiki_isolation.py
Writes analysis/wiki_isolation.md and analysis/wiki_isolation_data.json.
Safe to re-run any time (e.g. once musique's isolation cells or hotpotqa's dense_fetch cell fill
further) -- picks up whatever has crossed the 90% gate with no code change.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compare_cells import (  # noqa: E402
    cell_dir, cell_rows, load_judge_cache, load_qrels, mcnemar_p, metrics, pct,
)
from scripts.force_answer_backfill import load_rows_tolerant  # noqa: E402

MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"
SUBDIR = "_headline_validation"
COMPLETE_THRESHOLD = 0.90
JUDGE_COVERAGE_MIN = 0.90
ALPHA = 0.05

DATASETS = ["hotpotqa_structured", "musique_structured", "browsecomp_plus_structured"]

# key -> (label, condition dir name)
CELLS = {
    "full":               ("Full method (bql+dense+snip fetch)", "agent_research_bql_dense_snip"),
    "no_dense":           ("No dense evidence (bql+snip fetch)", "agent_research_snip"),
    "sparse_same_iface":  ("Sparse only, same interface (bm25+snip fetch)", "agent_research_bm25_fetch_snip"),
    "dense_same_iface":   ("Dense only, same interface (dense+snip fetch)", "agent_research_dense_fetch"),
}

ISOLATIONS = [
    dict(
        key="dense_fusion_cost",
        title="Cost of removing dense fusion: full method vs No dense evidence",
        A="full", B="no_dense",
        isolates=(
            "What removing dense (embedding) fusion costs the full method, holding the "
            "structured (BQL) query surface AND the snip+fetch read interface fixed on both "
            "sides. Both cells share: BQL query surface, snippet display, section-fetch reading; "
            "they differ only in whether dense retrieval is fused in."
        ),
        does_not_isolate=(
            "Whether the structured query surface itself helps (neither removed here -- both "
            "sides have BQL); that isolation is contrast 2 below. Does not isolate dense "
            "retrieval's value WITHOUT the structured surface either -- that is contrast 3."
        ),
    ),
    dict(
        key="query_surface_alone",
        title="Query surface alone: No dense evidence vs Sparse-only, same interface",
        A="no_dense", B="sparse_same_iface",
        isolates=(
            "Whether the field-tagged Boolean query SURFACE itself (vs plain keyword BM25) "
            "helps, holding the read interface (snippet-bearing listing + named-section fetch) "
            "and the absence of dense fusion fixed on both sides. Identical pairing to "
            "round5_analyses.py's M5 query_surface isolation, now computed on this dataset."
        ),
        does_not_isolate=(
            "Anything about dense fusion (neither side has it) or the 'visit whole document' "
            "reading style (neither side uses it). Does not separate BQL's Boolean filter from "
            "BQL's field-tag vocabulary -- BQL bundles both."
        ),
    ),
    dict(
        key="dense_alone",
        title="Dense fusion alone: Dense-only same interface vs Sparse-only same interface",
        A="dense_same_iface", B="sparse_same_iface",
        isolates=(
            "Whether swapping the query engine from plain BM25 to pure dense retrieval helps, "
            "holding the read interface (snippet listing + section-fetch) fixed and the BQL "
            "structured query surface absent from both sides. Identical pairing to "
            "round5_analyses.py's M5 dense_fusion isolation, now computed on this dataset."
        ),
        does_not_isolate=(
            "Anything about the BQL structured query surface (neither side has it) or about "
            "dense fusion INSIDE a structured query (that is contrast 1's 'full' cell vs the "
            "'no_dense' cell, i.e. contrast 1 itself, not this pair)."
        ),
    ),
]


# --------------------------------------------------------------------------------------
# completion gate: union canonical + shard rows.jsonl, dedup by instance_id, vs dataset total n
# --------------------------------------------------------------------------------------

def _dataset_total_n(dataset: str) -> int | None:
    p = Path("data") / dataset / "queries.jsonl"
    if not p.exists():
        return None
    n = 0
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
    except OSError:
        return None
    return n or None


def cell_completion(dataset: str, cond: str) -> dict:
    """Live completion check for one cell: unions the canonical rows.jsonl (if any) with every
    shard rows.jsonl found under runs/<SUBDIR>/__shards/<cond>/shard_*of*/agent/<dataset>/
    <MODEL_DIR>/<cond>/rows.jsonl, dedup by instance_id. Returns row counts + fraction of the
    dataset's total n -- does NOT load full row content (cheap: instance_id field only)."""
    total_n = _dataset_total_n(dataset)
    canonical_path = cell_dir(SUBDIR, dataset, cond) / "rows.jsonl"
    ids: set[str] = set()
    canonical_n = 0
    if canonical_path.exists():
        for r in load_rows_tolerant(str(canonical_path)):
            iid = r.get("instance_id")
            if iid:
                ids.add(iid)
        canonical_n = len(ids)
    shard_glob = str(Path("runs") / SUBDIR / "__shards" / cond / f"shard_*of*" / "agent" / dataset / MODEL_DIR / cond / "rows.jsonl")
    shard_paths = sorted(glob.glob(shard_glob))
    for sp in shard_paths:
        for r in load_rows_tolerant(sp):
            iid = r.get("instance_id")
            if iid:
                ids.add(iid)
    n = len(ids)
    frac = (n / total_n) if total_n else None
    return {
        "dataset": dataset, "cond": cond, "n": n, "total_n": total_n, "frac": frac,
        "canonical_exists": canonical_path.exists(), "canonical_n": canonical_n,
        "n_shard_dirs": len(shard_paths), "complete": bool(frac is not None and frac >= COMPLETE_THRESHOLD),
    }


def load_cell_gated(dataset: str, cond: str, qrels: dict):
    """Returns (status, payload) where status in {"ok", "partial", "absent"}.
    "ok"      -> payload = (metrics_dict, judge_coverage_frac, completion_dict)
    "partial" -> payload = completion_dict (n/total_n/frac live counts; NOT loaded for metrics)
    "absent"  -> payload = completion_dict (n==0, cell does not exist at all yet)
    Metrics are only ever computed from the CANONICAL (merged) rows.jsonl, via
    scripts.compare_cells.cell_rows (recovery-overlaid) -- shard-only cells are never scored,
    even if their live row count happens to clear the 90% threshold, because they have not been
    through the merge step other released cells in this repo go through."""
    comp = cell_completion(dataset, cond)
    if comp["n"] == 0:
        return "absent", comp
    if not comp["complete"]:
        return "partial", comp
    if not comp["canonical_exists"]:
        # shards alone cleared 90% but no merge has landed -- still not a "released" cell here.
        return "partial", comp
    rows = cell_rows(SUBDIR, dataset, cond)
    if rows is None:
        return "partial", comp
    jc = load_judge_cache(cell_dir(SUBDIR, dataset, cond))
    m = metrics(rows, qrels, dataset, jc)
    n = len(m)
    njudged = sum(1 for v in m.values() if v["judge"] is not None)
    cov = (njudged / n) if n else 0.0
    return "ok", (m, cov, comp)


# --------------------------------------------------------------------------------------
# paired stats (same shape/semantics as round5_analyses.py's paired_em/paired_judge)
# --------------------------------------------------------------------------------------

def paired_em(m_a: dict, m_b: dict):
    mut = set(m_a) & set(m_b)
    if not mut:
        return None
    b = sum(1 for i in mut if m_a[i]["em"] and not m_b[i]["em"])
    c = sum(1 for i in mut if m_b[i]["em"] and not m_a[i]["em"])
    em_a = pct([m_a[i]["em"] for i in mut])
    em_b = pct([m_b[i]["em"] for i in mut])
    return {"n": len(mut), "em_a": em_a, "em_b": em_b, "delta": em_a - em_b, "p": mcnemar_p(b, c), "b": b, "c": c}


def paired_judge(m_a: dict, m_b: dict, cov_a: float, cov_b: float):
    if cov_a is None or cov_b is None or cov_a < JUDGE_COVERAGE_MIN or cov_b < JUDGE_COVERAGE_MIN:
        return None
    mut = set(m_a) & set(m_b)
    jmut = [i for i in mut if m_a[i]["judge"] is not None and m_b[i]["judge"] is not None]
    if not jmut:
        return None
    jb = sum(1 for i in jmut if m_a[i]["judge"] and not m_b[i]["judge"])
    jc = sum(1 for i in jmut if m_b[i]["judge"] and not m_a[i]["judge"])
    j_a = pct([m_a[i]["judge"] for i in jmut])
    j_b = pct([m_b[i]["judge"] for i in jmut])
    return {"n": len(jmut), "judge_a": j_a, "judge_b": j_b, "delta": j_a - j_b, "p": mcnemar_p(jb, jc)}


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def run() -> dict:
    results = {"datasets": {}}
    for dataset in DATASETS:
        qrels = load_qrels(dataset)
        cell_status = {}
        for key, (label, cond) in CELLS.items():
            status, payload = load_cell_gated(dataset, cond, qrels)
            cell_status[key] = {"label": label, "cond": cond, "status": status, "payload": payload}
        iso_results = {}
        for iso in ISOLATIONS:
            a_key, b_key = iso["A"], iso["B"]
            a_status, a_payload = cell_status[a_key]["status"], cell_status[a_key]["payload"]
            b_status, b_payload = cell_status[b_key]["status"], cell_status[b_key]["payload"]
            if a_status != "ok" or b_status != "ok":
                reasons = []
                for k, st, pl in ((a_key, a_status, a_payload), (b_key, b_status, b_payload)):
                    label = CELLS[k][0]
                    if st == "absent":
                        reasons.append(f"{label} (`{CELLS[k][1]}`): cell not present under "
                                        f"runs/{SUBDIR}/agent/{dataset}/{MODEL_DIR}/")
                    elif st == "partial":
                        comp = pl
                        tot = comp["total_n"] if comp["total_n"] else "?"
                        fracpct = f"{comp['frac']*100:.1f}%" if comp["frac"] is not None else "?"
                        reasons.append(f"{label} (`{CELLS[k][1]}`): PARTIAL, n={comp['n']} of {tot} "
                                        f"({fracpct}, threshold {int(COMPLETE_THRESHOLD*100)}%) -- "
                                        f"canonical_merged={comp['canonical_exists']}, "
                                        f"shard_dirs_found={comp['n_shard_dirs']}")
                iso_results[iso["key"]] = {"computable": False, "reasons": reasons}
                continue
            m_a, cov_a, _ = a_payload
            m_b, cov_b, _ = b_payload
            res = paired_em(m_a, m_b)
            if res is None:
                iso_results[iso["key"]] = {"computable": False, "reasons": ["no shared instances"]}
                continue
            jres = paired_judge(m_a, m_b, cov_a, cov_b)
            iso_results[iso["key"]] = {
                "computable": True, "em": res, "judge": jres,
                "cov_a": cov_a, "cov_b": cov_b,
            }
        results["datasets"][dataset] = {"cells": cell_status, "isolations": iso_results}
    return results


def _fmt_p(p: float) -> str:
    return f"{p:.3g}"


def render_markdown(results: dict) -> tuple[str, list[str]]:
    out = ["# Wiki component isolation (round8 Job 2)\n"]
    out.append(
        "Same family of DIRECT paired, same-interface isolation contrasts "
        "`analysis/round5_analyses.py` computes on browsecomp_plus_structured (the dataset where "
        "the method's headline effect is weakest), now computed on hotpotqa_structured and "
        "musique_structured wherever the required cells have crossed a 90% completion gate "
        "(checked live against runs/ at script run time, not assumed). Neither side of any "
        "contrast below is the dataset's SERP-bm25 baseline or the full method compared to that "
        "baseline -- those pairings are `comparison_result.md`'s and `analysis/round5_analyses.py`'s "
        "M3, respectively.\n"
    )

    sentences = []
    for iso in ISOLATIONS:
        out.append(f"\n## {iso['title']}\n")
        out.append(f"**Isolates:** {iso['isolates']}\n")
        out.append(f"**Does NOT isolate:** {iso['does_not_isolate']}\n")
        a_label, b_label = CELLS[iso["A"]][0], CELLS[iso["B"]][0]
        out.append(f"A = {a_label}; B = {b_label}.\n")
        out.append("| dataset | n_shared | EM_A% | EM_B% | ΔEM (A-B) | McNemar p_em | judge_A% | judge_B% | Δjudge | McNemar p_judge |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for dataset in DATASETS:
            r = results["datasets"][dataset]["isolations"][iso["key"]]
            if not r["computable"]:
                note = "; ".join(r["reasons"])
                out.append(f"| {dataset} | -- | | | | | | | | SKIPPED -- {note} |")
                continue
            em = r["em"]
            if r["judge"] is not None:
                j = r["judge"]
                j_cols = f"{j['judge_a']:.1f} | {j['judge_b']:.1f} | {j['delta']:+.1f} | {_fmt_p(j['p'])}"
            else:
                cov_note = f"cov A={r['cov_a']*100:.0f}%, B={r['cov_b']*100:.0f}%"
                j_cols = f"pending ({cov_note}) | pending | -- | --"
            out.append(
                f"| {dataset} | {em['n']} | {em['em_a']:.1f} | {em['em_b']:.1f} | {em['delta']:+.1f} | "
                f"{_fmt_p(em['p'])} | {j_cols} |"
            )
            sig = "statistically significant" if em["p"] < ALPHA else "not statistically significant"
            sentences.append(
                f"{iso['title'].split(':')[0]} on {dataset}: {a_label} scores {em['em_a']:.1f}% EM vs "
                f"{b_label}'s {em['em_b']:.1f}% EM on the {em['n']} shared instances "
                f"({em['delta']:+.1f} pts; exact McNemar p={_fmt_p(em['p'])}, {sig})."
            )
        out.append("")

    out.append("\n## Live cell-completion status (all datasets, all cells this script uses)\n")
    out.append("| dataset | cell | condition | status | n | total_n | % complete | canonical merged | shard dirs found |")
    out.append("|---|---|---|---|---:|---:|---:|---|---:|")
    for dataset in DATASETS:
        for key, info in results["datasets"][dataset]["cells"].items():
            pl = info["payload"]
            n = pl["n"] if isinstance(pl, dict) else (pl[2]["n"] if info["status"] == "ok" else 0)
            comp = pl if isinstance(pl, dict) else pl[2]
            tot = comp["total_n"] if comp["total_n"] else "?"
            fracpct = f"{comp['frac']*100:.1f}%" if comp["frac"] is not None else "?"
            out.append(f"| {dataset} | {info['label']} | `{info['cond']}` | {info['status']} | "
                       f"{comp['n']} | {tot} | {fracpct} | {comp['canonical_exists']} | {comp['n_shard_dirs']} |")
    out.append("")
    return "\n".join(out), sentences


def _json_safe(results: dict) -> dict:
    """Strips the large per-instance metrics dicts (`m` in each 'ok' cell's payload) before
    dumping to JSON -- those are an implementation detail re-derivable from rows.jsonl any time;
    keeping them out matches every other analysis/*_data.json in this repo (aggregate-only,
    KB-scale), not the multi-MB per-instance dump this would otherwise become at n=7343x4 cells."""
    safe = {"datasets": {}}
    for ds, d in results["datasets"].items():
        cells_safe = {}
        for key, info in d["cells"].items():
            payload = info["payload"]
            comp = payload if isinstance(payload, dict) else payload[2]
            cov = None if isinstance(payload, dict) else payload[1]
            cells_safe[key] = {"label": info["label"], "cond": info["cond"], "status": info["status"],
                                "completion": comp, "judge_coverage": cov}
        safe["datasets"][ds] = {"cells": cells_safe, "isolations": d["isolations"]}
    return safe


def main() -> None:
    results = run()
    md, sentences = render_markdown(results)

    data_path = Path(__file__).resolve().parent / "wiki_isolation_data.json"
    data_path.write_text(json.dumps(_json_safe(results), indent=2, default=str))
    print(f"wrote {data_path}", file=sys.stderr)

    text = md + "\n## Lift-able sentences\n\n" + "\n\n".join(f"- {s}" for s in sentences) + "\n"
    out_path = Path(__file__).resolve().parent / "wiki_isolation.md"
    out_path.write_text(text)
    print(text)
    print(f"\nwrote: {out_path}")


if __name__ == "__main__":
    main()
