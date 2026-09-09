#!/usr/bin/env python
"""Recompute every numeric value of the paper's result tables directly from run data.

    PYTHONPATH=. python analysis/make_paper_tables.py --check          # default
    PYTHONPATH=. python analysis/make_paper_tables.py --emit <outdir>
    PYTHONPATH=. python analysis/make_paper_tables.py --selftest       # pipeline vs cell_metrics

BrowseComp-Plus corpus switch (--corpus, DEFAULT full):
  --corpus full   -> every BCP cell reads the FULL-corpus runs (dataset
                     browsecomp_plus_structured_full; conditions under runs/_fullcorpus /
                     runs/_fullcorpus_indridense / runs/_oneshot/browsecomp_plus_structured_full),
                     and the two BCP-S rows of backbone_transfer.tex are UNFROZEN and recomputed
                     (Tongyi from the full-corpus visit-bm25/sieve cells; AgentWorld from the
                     full-corpus Qwen-AgentWorld-35B-A3B bm25 / bql_dense_snip cells).
  --corpus pooled -> the historical pooled-corpus mapping, kept intact for provenance; BCP-S
                     backbone rows stay FROZEN constants copied verbatim from the live .tex.
  Wiki (hotpotqa/musique) cells are identical under both settings.

Covers the paper's table sources:
  consolidated.tex, snippet_ablation.tex, backbone_transfer.tex (hotpotqa rows recomputed always;
  BCP-S rows recomputed under --corpus full, FROZEN under pooled; MuSiQue rows always FROZEN
  historical constants copied from the live .tex), main_results.tex, wiki_results.tex (two
  tables), sieve_indri.tex.  Also prints (reference-only, never counted as diffs) the headline
  stats behind fig_gainloss / fig_pareto.

Metric code is the project's own — nothing rescored here:
  - scripts.compare_cells._row_intrinsic / _apply_overlay: per-row EM / judge / recovery-overlay /
    recall / count-once tokens / llm_calls.  metrics() (and therefore
    analysis.ablation_deltas.cell_metrics) is documented + drift-guarded as expressed in exactly
    these two building blocks; we use them directly because cell_metrics loads whole rows.jsonl
    files and the AutoRead cells run 16-31GB (this script streams line-by-line instead).
    `--selftest` verifies per-instance equality against cell_metrics on two small cells.
  - scripts.compare_cells.load_qrels / mcnemar_p / pct: qrels, exact McNemar, percent.
  - Accumulated (step-summed) tokens = prompt_tokens + completion_tokens per row, exactly as
    analysis/appendix_token_columns.py's stream_row_stats.

Caching: analysis/paper_tables_cache.json keyed by (relative rows.jsonl path, mtime_ns, size).
The cached payload is the ROW-INTRINSIC half only (same split as compare_cells' own cache) plus
the per-row step-summed tokens; the overlay-dependent half (recovery / judge sidecars) is
recomputed fresh every run, so a late-landing judge or recovery pass is picked up without
re-streaming anything.  A warm scripts/compare_cells cache (analysis/.compare_cache) is reused
for the intrinsic dict when valid, so cold cells only stream rows.jsonl for the step-sum scalars.

Concurrent-symlink safety: cells are only ever addressed through the canonical runs/ paths below;
stat()/open() follow symlinks, so re-organisation via symlinks keeps resolving.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                    # `scripts/` is repo-only, not an installed package
    sys.path.insert(0, str(ROOT))
# NOTE: compare_cells / load_qrels / the compare cache use repo-relative paths, so main()
# changes into ROOT before doing any work. Importing this module never changes the cwd.

from scripts.compare_cells import (  # noqa: E402
    MODEL_DIR, _apply_overlay, _qid_of, _read_cache, _row_intrinsic, load_qrels, mcnemar_p, pct,
)

# The paper's LaTeX tree is a local checkout, not part of this release; these paths only
# resolve when that tree happens to sit alongside the repo.
TABLES_DIR = ROOT / "Boolean_agent_paper" / "tables"
FIGS_DIR = ROOT / "Boolean_agent_paper" / "figures"
CACHE_PATH = ROOT / "analysis" / "paper_tables_cache.json"
CACHE_VERSION = 1
XB_MODEL = "Qwen-AgentWorld-35B-A3B"
# backbone_transfer rows 3-4 (2026-07-31): model-dir cells under _xbackbone (wiki) / _fullcorpus
# (BCP-full); registered lazily by existence so partially-landed fleets never break the loader.
# MiroThinker removed from the paper 2026-08-01 (author decision: effectiveness too low);
# its landed cells remain under runs/_xbackbone + runs/_fullcorpus, just not table-checked.
NEW_BACKBONES = (("or", "OpenResearcher-30B-A3B"),)
XB_EXPECT = {"bcp": 830, "hp": 7343, "mq": 2409}

BCP_DS = {"pooled": "browsecomp_plus_structured", "full": "browsecomp_plus_structured_full"}


def datasets_for(corpus: str) -> dict:
    """key -> (dataset dir name, n expected).  Only the BCP entry depends on --corpus."""
    return {
        "bcp": (BCP_DS[corpus], 830),
        "hp": ("hotpotqa_structured", 7343),
        "mq": ("musique_structured", 2409),
    }


def tier_dir(tier: str, ds: str, cond: str, model: str = MODEL_DIR) -> Path:
    if tier == "agent":
        return Path("runs/agent") / ds / model / cond
    return Path("runs") / tier / "agent" / ds / model / cond


def cell_paths(ds: str, corpus: str = "full") -> dict:
    """cell key -> canonical condition dir (relative to repo root) for one dataset."""
    if ds == BCP_DS["full"]:
        return _cell_paths_bcp_full(ds)
    # pooled-corpus BCP mapping and the (corpus-independent) wiki mappings
    p = {
        "oneshot_bm25": Path("runs/_oneshot") / ds / "bm25",
        "oneshot_dense": Path("runs/_oneshot") / ds / "dense",
        "autoread_bm25": tier_dir("_visit_uncapped", ds, "agent_research_bm25_autoread"),
        "autoread_dense": tier_dir("_visit_uncapped", ds, "agent_research_dense_autoread"),
        "dci": tier_dir("agent", ds, "agent_research_dci"),
        "bm25_dci": tier_dir("agent", ds, "agent_research_bm25_dci"),
        "visit_bm25": tier_dir("_visit_uncapped", ds, "agent_research_bm25"),
        "visit_dense": tier_dir("_fullvisit", ds, "agent_research_dense"),
        "visit_hybrid": tier_dir("_fullvisit", ds, "agent_research_hybrid"),
        "fetch_bm25": tier_dir("_fetch_k5", ds, "agent_research_bm25_fetch_snip"),
        "fetch_dense": tier_dir("_fetch_k5", ds, "agent_research_dense_fetch"),
        "fetch_hybrid": tier_dir("_fetch_k5", ds, "agent_research_hybrid_fetch_snip"),
        "sieve": tier_dir("_headline_validation", ds, "agent_research_bql_dense_snip"),
        "sieve_bm25": tier_dir("_headline_validation", ds, "agent_research_snip"),
        "sieve_dense": tier_dir("_newconds", ds, "agent_research_bql_donly_snip"),
        "sieve_nosnip": tier_dir("_headline_validation", ds, "agent_research_bql_dense_fetch"),
    }
    if ds == BCP_DS["pooled"]:
        p["indri"] = tier_dir("_headline_validation", ds, "agent_research_indri_snip")
        p["indri_dense"] = tier_dir("_dense_validation", ds, "agent_research_indri_snip")
    if ds == "hotpotqa_structured":
        p["xb_visit"] = tier_dir("_xbackbone", ds, "agent_research_bm25", XB_MODEL)
        p["xb_sieve"] = tier_dir("_xbackbone", ds, "agent_research_bql_dense_snip", XB_MODEL)
    # new backbones (backbone_transfer rows 3-4): registered only once their cell dirs exist,
    # so mid-flight runs never break the loader; the checker treats absent keys as `pending`.
    for tag, model in NEW_BACKBONES:
        for ck, cond in ((f"{tag}_visit", "agent_research_bm25"),
                         (f"{tag}_sieve", "agent_research_bql_dense_snip")):
            d = tier_dir("_xbackbone", ds, cond, model)
            if (d / "rows.jsonl").exists():
                p[ck] = d
    return p


def _cell_paths_bcp_full(ds: str) -> dict:
    """FULL-corpus BCP mapping: everything lives in the _fullcorpus tier (one condition dir per
    interface), Indri+dense in its sibling _fullcorpus_indridense tier, one-shot under the full
    dataset's _oneshot dir, and the cross-backbone cells under the same _fullcorpus tier's
    Qwen-AgentWorld model dir."""
    fc = lambda cond, model=MODEL_DIR: tier_dir("_fullcorpus", ds, cond, model)  # noqa: E731
    p = {
        "oneshot_bm25": Path("runs/_oneshot") / ds / "bm25",
        "oneshot_dense": Path("runs/_oneshot") / ds / "dense",
        "autoread_bm25": fc("agent_research_bm25_autoread"),
        "autoread_dense": fc("agent_research_dense_autoread"),
        "dci": fc("agent_research_dci"),
        "bm25_dci": fc("agent_research_bm25_dci"),
        "visit_bm25": fc("agent_research_bm25"),
        "visit_dense": fc("agent_research_dense"),
        "visit_hybrid": fc("agent_research_hybrid"),
        "fetch_bm25": fc("agent_research_bm25_fetch_snip"),
        "fetch_dense": fc("agent_research_dense_fetch"),
        "fetch_hybrid": fc("agent_research_hybrid_fetch_snip"),
        "sieve": fc("agent_research_bql_dense_snip"),
        "sieve_bm25": fc("agent_research_snip"),
        "sieve_dense": fc("agent_research_bql_donly_snip"),
        "sieve_nosnip": fc("agent_research_bql_dense_fetch"),
        "indri": fc("agent_research_indri_snip"),
        "indri_dense": tier_dir("_fullcorpus_indridense", ds, "agent_research_indri_snip"),
        "xb_visit": fc("agent_research_bm25", XB_MODEL),
        "xb_sieve": fc("agent_research_bql_dense_snip", XB_MODEL),
    }
    for tag, model in NEW_BACKBONES:
        for ck, cond in ((f"{tag}_visit", "agent_research_bm25"),
                         (f"{tag}_sieve", "agent_research_bql_dense_snip")):
            d = fc(cond, model)
            if (d / "rows.jsonl").exists():
                p[ck] = d
    return p


# --- per-cell computation (intrinsic half cached; overlay half fresh every run) -------------------

def _stream_cell(cond_dir_str: str, dataset: str, qrels: dict):
    """(intrinsic, stepsum) for one cell.  Streams rows.jsonl line-by-line (never whole-file: the
    AutoRead files run 16-31GB).  When scripts/compare_cells' own incremental cache is warm and
    byte-exact for this cell, its intrinsic dict is reused and the stream only extracts the two
    step-sum scalars per row (no regex / no answer_em)."""
    cond_dir = Path(cond_dir_str)
    rows_path = cond_dir / "rows.jsonl"
    size = rows_path.stat().st_size
    cc = _read_cache(cond_dir)
    have_intrinsic = bool(cc and cc.get("exists") and cc.get("n_bytes") == size
                          and cc.get("intrinsic"))
    intrinsic = dict(cc["intrinsic"]) if have_intrinsic else {}
    stepsum = {}
    with rows_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn/corrupt trailing line — same tolerance as load_rows_tolerant
            iid = r.get("instance_id") or ""
            if not iid:
                continue
            stepsum[iid] = (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
            if not have_intrinsic:
                qid = _qid_of(iid, dataset)
                intrinsic[iid] = _row_intrinsic(r, (qrels or {}).get(qid, set()))
    return intrinsic, stepsum


def _load_my_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            c = json.loads(CACHE_PATH.read_text())
            if c.get("version") == CACHE_VERSION:
                return c
        except (OSError, json.JSONDecodeError):
            pass
    return {"version": CACHE_VERSION, "cells": {}}


def _rows_key(cond_dir: Path):
    st = (cond_dir / "rows.jsonl").stat()
    return [st.st_mtime_ns, st.st_size]


def gather_cells(all_cells: list, workers: int, use_cache: bool, verbose: bool) -> dict:
    """all_cells: [(dskey, cellkey, cond_dir, dataset)] -> {(dskey, cellkey): stats dict}.
    Cache misses stream in parallel; overlay + aggregation always run fresh in-process."""
    cache = _load_my_cache() if use_cache else {"version": CACHE_VERSION, "cells": {}}
    qrels_by_ds = {}
    todo, payloads = [], {}
    for dskey, ck, cond_dir, ds in all_cells:
        ent = cache["cells"].get(str(cond_dir))
        key = _rows_key(cond_dir)
        if ent and ent.get("key") == key:
            payloads[str(cond_dir)] = ent
        else:
            if ds not in qrels_by_ds:
                qrels_by_ds[ds] = load_qrels(ds)
            todo.append((str(cond_dir), ds))
    if todo:
        if verbose:
            print(f">> {len(todo)} cell(s) not in cache — streaming (workers={workers})",
                  file=sys.stderr)
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(_stream_cell, cd, ds, qrels_by_ds[ds]): cd for cd, ds in todo}
            for fut in as_completed(futs):
                cd = futs[fut]
                intrinsic, stepsum = fut.result()
                payloads[cd] = {"key": _rows_key(Path(cd)),
                                "intrinsic": intrinsic, "stepsum": stepsum}
                if verbose:
                    print(f"   streamed {cd} (n={len(intrinsic)})", file=sys.stderr)
        if use_cache:
            cache["cells"].update({cd: payloads[cd] for cd, _ in todo})
            tmp = CACHE_PATH.with_name(CACHE_PATH.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(cache))
            tmp.replace(CACHE_PATH)

    out = {}
    for dskey, ck, cond_dir, ds in all_cells:
        ent = payloads[str(cond_dir)]
        intrinsic, stepsum = ent["intrinsic"], ent["stepsum"]
        m = _apply_overlay(intrinsic, cond_dir)  # overlay half: fresh sidecar read every run
        # One-shot rows carry no reading trace, so trace-based recall is identically 0. For those
        # cells recall is instead the stuffed-set definition: gold doc present in retrieved_ids
        # (see the T9/T10 captions). Detected by the cell's dir living under runs/_oneshot/.
        if "/_oneshot/" in str(cond_dir):
            import json as _json
            from scripts.compare_cells import load_qrels as _lq
            _q = _lq(ds)
            _pref = None
            with open(str(cond_dir) + "/rows.jsonl") as _fh:
                for _line in _fh:
                    _r = _json.loads(_line)
                    _iid = _r["instance_id"]
                    if _pref is None:
                        for _k in _q:
                            if _iid.endswith(_k):
                                _pref = _iid[: len(_iid) - len(_k)]
                                break
                    _sh = _iid[len(_pref):] if _pref and _iid.startswith(_pref) else _iid
                    _gold = set(_q.get(_sh) or _q.get(_iid) or [])
                    _got = {str(_x) for _x in (_r.get("retrieved_ids") or [])}
                    if _iid in m:
                        m[_iid]["recall"] = bool(_gold & _got)
        vals = list(m.values())
        n = len(vals)
        judged = [v["judge"] for v in vals if v["judge"] is not None]
        cov = len(judged) / n if n else 0.0
        out[(dskey, ck)] = dict(
            n=n,
            em=pct([v["em"] for v in vals]),
            judge=pct(judged) if judged and cov >= 0.9 else None,
            judge_cov=cov,
            recall=pct([v["recall"] for v in vals]),
            tok=sum(v["tok"] for v in vals) / max(n, 1),
            tok_acc=sum(stepsum.get(i, 0) for i in m) / max(n, 1),
            calls=sum(v["llm_calls"] for v in vals) / max(n, 1),
            per=m,  # per-instance dicts (em / judge / tok) for McNemar + figure stats
        )
    return out


def add_stars(stats: dict) -> None:
    """EM-based exact McNemar vs the dataset's baseline (visit bm25, _visit_uncapped) on shared
    instance ids.  star = p < 0.05; the baseline itself never starred.
    (Under --corpus full the bcp baseline is the _fullcorpus agent_research_bm25 cell, because
    that is what visit_bm25 resolves to there.)"""
    for dskey in sorted({dk for dk, _ in stats}):
        base = stats.get((dskey, "visit_bm25"))
        if not base:
            continue
        for (dk, ck), s in stats.items():
            if dk != dskey:
                continue
            if ck == "visit_bm25":
                s["star"], s["p_em"] = False, None
                continue
            mut = set(s["per"]) & set(base["per"])
            b = sum(1 for i in mut if s["per"][i]["em"] and not base["per"][i]["em"])
            c = sum(1 for i in mut if base["per"][i]["em"] and not s["per"][i]["em"])
            p = mcnemar_p(b, c)
            s["star"], s["p_em"] = bool(p < 0.05), p


# --- formatting -----------------------------------------------------------------------------------

PENDING = "??"  # rendered for a metric that is not yet computable (judge coverage <90%);
                # run_check WARNs on these, run_emit refuses to write them into a .tex


def f1(v) -> str:
    return PENDING if v is None else f"{v:.1f}"


def fk(v) -> str:
    return PENDING if v is None else f"{v / 1000:.0f}k"


def fk1(v) -> str:
    return PENDING if v is None else f"{v / 1000:.1f}k"


class Cell:
    """One rendered table cell: plain text (number, incl. any star / % / sign), bold flag,
    math flag ($...$ wrapping on emit)."""
    def __init__(self, text: str, bold: bool = False, math: bool = False):
        self.text, self.bold, self.math = text, bold, math

    def latex(self) -> str:
        t = self.text.replace("%", "\\%")
        if self.math:
            t = f"${t}$"
        return f"\\textbf{{{t}}}" if self.bold else t


def bold_marks(values: list, minimize: bool, eligible: list) -> list:
    """Bool per row: is this row's (unrounded) value the best among eligible rows?"""
    pool = [v for v, e in zip(values, eligible) if e and v is not None]
    if not pool:
        return [False] * len(values)
    best = min(pool) if minimize else max(pool)
    return [bool(e and v is not None and abs(v - best) < 1e-9)
            for v, e in zip(values, eligible)]


# --- table row inventories ------------------------------------------------------------------------

# (cell key, LaTeX row label as it appears in the .tex, iterative-agent?)
MAIN_ROWS = [
    ("oneshot_bm25", "BM25", False),
    ("oneshot_dense", "Dense", False),
    ("autoread_bm25", "BM25", True),
    ("autoread_dense", "Dense", True),
    ("dci", "DCI (no retriever)", True),
    ("bm25_dci", "BM25-bounded DCI (RISE-style)", True),
    ("visit_bm25", "BM25", True),
    ("visit_dense", "Dense", True),
    ("visit_hybrid", "BM25+Dense", True),
    ("fetch_bm25", "BM25", True),
    ("fetch_dense", "Dense", True),
    ("fetch_hybrid", "BM25+Dense", True),
    ("sieve_bm25", "\\textsc{Sieve} (Boolean-filtered BM25)", True),
    ("sieve_dense", "\\textsc{Sieve} (Boolean-filtered Dense)", True),
    ("sieve", "\\textsc{Sieve} (Boolean-filtered BM25+Dense)", True),
]


def acc_of(s: dict, dskey: str):
    """The table-1 'accuracy': judge on BCP-S, EM on the wiki collections.
    Returns None (rendered as PENDING, warned in --check, refused in --emit) when a BCP cell's
    judge coverage is still <90% — e.g. a full-corpus cell the judge daemon hasn't reached."""
    if dskey == "bcp":
        return s["judge"]  # None if coverage <90%
    return s["em"]


def build_consolidated(stats):
    """15 rows x [Acc Tok Calls] x 3 datasets.  Bold = best iterative-agent value per column."""
    iter_flags = [it for _, _, it in MAIN_ROWS]
    grid = []
    for ck, _, _ in MAIN_ROWS:
        row = []
        for dskey in ("hp", "mq", "bcp"):
            s = stats[(dskey, ck)]
            row += [acc_of(s, dskey), s["tok"], s["calls"]]
        grid.append(row)
    out = []
    for j in range(9):
        col = [r[j] for r in grid]
        marks = bold_marks(col, minimize=(j % 3 != 0), eligible=iter_flags)
        fmt = f1 if j % 3 == 0 else (fk if j % 3 == 1 else f1)
        out.append([Cell(fmt(v), bold=b) for v, b in zip(col, marks)])
    rows = []
    for i, (ck, label, _) in enumerate(MAIN_ROWS):
        rows.append((label, [out[j][i] for j in range(9)]))
    return rows


def build_full(stats, dskey: str):
    """main_results (bcp: Judge EM* Recall Tok AccTok Calls) / wiki (EM* Recall Tok AccTok Calls).
    Bold: accuracy-ish columns best over ALL rows; token/call columns best among iterative rows.
    Stars on the EM column only (vs visit-bm25 baseline, exact McNemar p<0.05)."""
    iter_flags = [it for _, _, it in MAIN_ROWS]
    all_flags = [True] * len(MAIN_ROWS)
    cols = ([("judge", False, all_flags, f1), ("em", False, all_flags, f1),
             ("recall", False, all_flags, f1), ("tok", True, iter_flags, fk),
             ("tok_acc", True, iter_flags, fk), ("calls", True, iter_flags, f1)]
            if dskey == "bcp" else
            [("em", False, all_flags, f1), ("recall", False, all_flags, f1),
             ("tok", True, iter_flags, fk), ("tok_acc", True, iter_flags, fk),
             ("calls", True, iter_flags, f1)])
    cells_by_col = []
    for metric, minimize, elig, fmt in cols:
        col = [stats[(dskey, ck)][metric] for ck, _, _ in MAIN_ROWS]
        marks = bold_marks(col, minimize, elig)
        cc = []
        for i, (v, b) in enumerate(zip(col, marks)):
            txt = fmt(v)
            if metric == "em" and stats[(dskey, MAIN_ROWS[i][0])]["star"]:
                txt += "*"
            cc.append(Cell(txt, bold=b))
        cells_by_col.append(cc)
    return [(label, [cells_by_col[j][i] for j in range(len(cols))])
            for i, (ck, label, _) in enumerate(MAIN_ROWS)]


def build_snippet(stats):
    """2 rows x [Acc Tok AccTok Calls] x 3 datasets; bold = better of the pair per column."""
    rows_keys = [("sieve_nosnip", "\\textsc{Sieve} without snippets"), ("sieve", "\\textsc{Sieve}")]
    grid = []
    for ck, _ in rows_keys:
        row = []
        for dskey in ("hp", "mq", "bcp"):
            s = stats[(dskey, ck)]
            row += [acc_of(s, dskey), s["tok"], s["tok_acc"], s["calls"]]
        grid.append(row)
    out = []
    for j in range(12):
        col = [r[j] for r in grid]
        marks = bold_marks(col, minimize=(j % 4 != 0), eligible=[True, True])
        fmt = f1 if j % 4 == 0 else (fk if j % 4 in (1, 2) else f1)
        out.append([Cell(fmt(v), bold=b) for v, b in zip(col, marks)])
    return [(label, [out[j][i] for j in range(12)]) for i, (_, label) in enumerate(rows_keys)]


def build_sieve_indri(stats):
    """Two stacked 3-row blocks over the same conditions (bcp):
    block A = Judge / EM* / Recall, block B = Tok / AccTok / Calls; bold = best per column."""
    keys = [("indri", "Indri"), ("indri_dense", "Indri + dense belief"), ("sieve", "\\textsc{Sieve}")]
    a_cols, b_cols = [], []
    for metric, minimize, fmt in [("judge", False, f1), ("em", False, f1), ("recall", False, f1)]:
        col = [stats[("bcp", ck)][metric] for ck, _ in keys]
        marks = bold_marks(col, minimize, [True] * 3)
        cc = []
        for i, (v, b) in enumerate(zip(col, marks)):
            txt = fmt(v) + ("*" if metric == "em" and stats[("bcp", keys[i][0])]["star"] else "")
            cc.append(Cell(txt, bold=b))
        a_cols.append(cc)
    for metric, fmt in [("tok", fk), ("tok_acc", fk), ("calls", f1)]:
        col = [stats[("bcp", ck)][metric] for ck, _ in keys]
        marks = bold_marks(col, minimize=True, eligible=[True] * 3)
        b_cols.append([Cell(fmt(v), bold=b) for v, b in zip(col, marks)])
    rows = [(label, [a_cols[j][i] for j in range(3)]) for i, (_, label) in enumerate(keys)]
    rows += [(label, [b_cols[j][i] for j in range(3)]) for i, (_, label) in enumerate(keys)]
    return rows


def build_backbone(stats, corpus: str):
    """The recomputed rows of backbone_transfer.tex.  HotpotQA rows always; the two BCP-S rows
    only under --corpus full (frozen historical constants under pooled).  MuSiQue rows always
    FROZEN.  Acc is the judge verdict on BCP-S and EM elsewhere (the table's caption convention).
    Acc delta from the rounded printed values; token delta % from the unrounded means (this is the
    convention the frozen rows verifiably follow, cf. -30.4% vs 19.9k/13.9k)."""
    def mk(visit, sieve, metric):
        av, as_ = visit[metric], sieve[metric]
        if av is None or as_ is None:  # judge coverage still <90% on a BCP-S cell
            return [Cell(PENDING)] * 6
        cells = [Cell(f1(av)), Cell(f1(as_)),
                 Cell(f"{round(as_, 1) - round(av, 1):+.1f}", math=True),
                 Cell(fk1(visit["tok"])), Cell(fk1(sieve["tok"])),
                 Cell(f"{(sieve['tok'] - visit['tok']) / visit['tok'] * 100:+.1f}%", math=True)]
        return cells
    rows = {
        ("Tongyi-DeepResearch-30B-A3B", "HotpotQA"):
            mk(stats[("hp", "visit_bm25")], stats[("hp", "sieve")], "em"),
        ("Qwen-AgentWorld-35B-A3B", "HotpotQA"):
            mk(stats[("hp", "xb_visit")], stats[("hp", "xb_sieve")], "em"),
    }
    if corpus == "full":
        rows[("Tongyi-DeepResearch-30B-A3B", "\\textsc{BCP-S}")] = \
            mk(stats[("bcp", "visit_bm25")], stats[("bcp", "sieve")], "judge")
        rows[("Qwen-AgentWorld-35B-A3B", "\\textsc{BCP-S}")] = \
            mk(stats[("bcp", "xb_visit")], stats[("bcp", "xb_sieve")], "judge")
    return rows


# --- .tex walking: locate data-row statements by ordered label match ------------------------------

def _statements(lines: list, start: int):
    """Join lines[start:] until one ends with '\\\\'; returns (statement text, end line index)."""
    j = start
    stmt = [lines[j]]
    while not lines[j].rstrip().endswith("\\\\"):
        j += 1
        stmt.append(lines[j])
    return "\n".join(stmt), j


def _label_matches(stripped: str, label: str) -> bool:
    if not stripped.startswith(label):
        return False
    rest = stripped[len(label):]
    return rest == "" or rest[0] in (" ", "&", "\\", "\t")


def walk_table(path: Path, expected_labels: list):
    """Returns (lines, matches): matches[i] = (start_line, end_line, statement) for the i-th
    expected data row, walking the file top-to-bottom in order.  Raises if a row is not found."""
    lines = path.read_text().split("\n")
    matches = []
    idx = 0
    for label in expected_labels:
        found = None
        while idx < len(lines):
            s = lines[idx].strip()
            if s and not s.startswith("%") and _label_matches(s, label):
                stmt, end = _statements(lines, idx)
                found = (idx, end, stmt)
                idx = end + 1
                break
            idx += 1
        if found is None:
            raise RuntimeError(f"{path.name}: data row not found: {label!r}")
        matches.append(found)
    return lines, matches


def parse_cells(stmt: str) -> list:
    """['2.9', ...] normalized cell descriptors from one data-row statement (label dropped):
    each as (plain_text, star, bold).  Robust to \\textbf{...}, $...$, \\%, and -- cells."""
    body = stmt.replace("\n", " ").strip()
    if body.endswith("\\\\"):
        body = body[:-2]
    parts = [p.strip() for p in body.split("&")]
    out = []
    for p in parts[1:]:
        bold = False
        m = re.fullmatch(r"\\textbf\{(.*)\}", p)
        if m:
            bold, p = True, m.group(1).strip()
        p = p.replace("$", "").replace("\\%", "%").strip()
        star = p.endswith("*")
        if star:
            p = p[:-1].strip()
        out.append((p, star, bold))
    return out


# --- check / emit drivers -------------------------------------------------------------------------

TABLE_FILES = {
    "consolidated": TABLES_DIR / "consolidated.tex",
    "snippet_ablation": TABLES_DIR / "snippet_ablation.tex",
    "backbone_transfer": TABLES_DIR / "backbone_transfer.tex",
    "main_results": TABLES_DIR / "main_results.tex",
    "wiki_results": TABLES_DIR / "wiki_results.tex",
    "sieve_indri": TABLES_DIR / "sieve_indri.tex",
}


def computed_tables(stats):
    """table name -> ordered [(label, [Cell, ...])]. wiki_results carries both its tables."""
    return {
        "consolidated": build_consolidated(stats),
        "snippet_ablation": build_snippet(stats),
        "main_results": build_full(stats, "bcp"),
        "wiki_results": build_full(stats, "hp") + build_full(stats, "mq"),
        "sieve_indri": build_sieve_indri(stats),
    }


def backbone_order(corpus: str) -> list:
    """File order of the four data rows in the 2026-07-31 author-rewritten one-row-per-backbone
    layout.  Payload per collection (bcp, hp, mq order, matching the header): a stats-key source
    ("stats", dskey, visit_key, sieve_key, metric), or None = frozen, never recomputed."""
    full = corpus == "full"
    def src(dskey, vk, sk):
        metric = "judge" if dskey == "bcp" else "em"
        return ("stats", dskey, vk, sk, metric)
    return [
        ("Tongyi-DeepResearch-30B-A3B",
         [src("bcp", "visit_bm25", "sieve") if full else None,
          src("hp", "visit_bm25", "sieve"), None]),
        ("Qwen-AgentWorld-35B-A3B",
         [src("bcp", "xb_visit", "xb_sieve") if full else None,
          src("hp", "xb_visit", "xb_sieve"), None]),
        ("OpenResearcher-30B-A3B",
         [src(k, "or_visit", "or_sieve") if full else None for k in ("bcp", "hp", "mq")]),
    ]


_BB_ARROW_RE = re.compile(r"\$(\d+(?:\.\d+)?)\{\\rightarrow\}(\d+(?:\.\d+)?)\$(k?)")

# author's 2026-08-01 layout: one \\multirow block per backbone, three data lines per block in
# BCP-S / HotpotQA / MuSiQue order, each line one accuracy arrow cell + one token arrow cell.
_BB_SHORT = {
    "Tongyi-DeepResearch-30B-A3B": "Tongyi-DeepResearch",
    "Qwen-AgentWorld-35B-A3B": "Qwen-AgentWorld",
    "OpenResearcher-30B-A3B": "OpenResearcher",
}


def check_backbone(diffs: list, stats: dict, corpus: str) -> int:
    """Diffs the live backbone_transfer.tex (multirow author format) against recomputed pair
    values; returns #numeric cells compared.  A pair is recomputable only when both cells are
    loaded AND complete (n == XB_EXPECT[dskey], judge coverage >=90% on BCP)."""
    order = backbone_order(corpus)
    lines = TABLE_FILES["backbone_transfer"].read_text().split("\n")
    collnames = ("BCP-S", "HotpotQA", "MuSiQue")
    n_cells = 0
    idx = 0
    for backbone, sources in order:
        tag = f"\\multirow{{3}}{{*}}{{{_BB_SHORT[backbone]}}}"
        while idx < len(lines) and not lines[idx].strip().startswith(tag):
            idx += 1
        assert idx < len(lines), f"backbone_transfer.tex: block not found: {backbone}"
        stmts = []
        while len(stmts) < 3 and idx < len(lines):
            s = lines[idx].strip()
            if s and not s.startswith("%") and s.rstrip().endswith("\\\\"):
                stmts.append(s)
            idx += 1
        assert len(stmts) == 3, f"{backbone}: expected 3 data lines, got {len(stmts)}"
        for coll, srcspec, stmt in zip(collnames, sources, stmts):
            row = f"{backbone} / {coll}"
            assert coll in stmt, f"backbone row order mismatch: expected {coll} in {stmt!r}"
            pairs = _BB_ARROW_RE.findall(stmt)
            assert len(pairs) == 2, f"{row}: expected 2 arrow cells, got {len(pairs)}: {stmt!r}"
            (av, sv, ak), (tv, ts, tk) = [(float(a), float(b), k == "k") for a, b, k in pairs]
            assert not ak and tk, f"{row}: acc/token cell order broke: {stmt!r}"
            if srcspec is None:
                continue  # frozen historical cell — author text is authoritative
            _, dskey, vk, sk, metric = srcspec
            v, s = stats.get((dskey, vk)), stats.get((dskey, sk))
            complete = (v and s and v["n"] == XB_EXPECT[dskey] and s["n"] == XB_EXPECT[dskey]
                        and v[metric] is not None and s[metric] is not None)
            if not complete:
                diffs.append(("backbone_transfer", row, "pair", "value",
                              f"{av}->{sv}", "cell incomplete on disk"))
                continue
            for cname, want, got in (
                    ("Search-Visit Acc", f1(v[metric]), f1(av)),
                    ("Sieve Acc", f1(s[metric]), f1(sv)),
                    ("Search-Visit Tok", f1(v["tok"] / 1000), f1(tv)),
                    ("Sieve Tok", f1(s["tok"] / 1000), f1(ts))):
                n_cells += 1
                if want != got:
                    diffs.append(("backbone_transfer", row, cname, "value", got, want))
    return n_cells



def _diff_one(diffs, table, row, col, printed, cell: Cell):
    ptext, pstar, pbold = printed
    ctext = cell.text
    cstar = ctext.endswith("*")
    if cstar:
        ctext = ctext[:-1]
    if ptext != ctext:
        diffs.append((table, row, col, "value", ptext + ("*" if pstar else ""),
                      cell.text))
    elif pstar != cstar:
        diffs.append((table, row, col, "star", ptext + ("*" if pstar else ""),
                      cell.text))
    elif pbold != cell.bold:
        diffs.append((table, row, col, "bold",
                      ("\\textbf " if pbold else "plain ") + ptext,
                      ("\\textbf " if cell.bold else "plain ") + cell.text))


COLNAMES = {
    "consolidated": [f"{d} {c}" for d in ("hp", "mq", "bcp") for c in ("Acc", "Tok", "Calls")],
    "snippet_ablation": [f"{d} {c}" for d in ("hp", "mq", "bcp")
                         for c in ("Acc", "Tok", "AccTok", "Calls")],
    "main_results": ["Judge", "EM", "Recall", "Tok", "AccTok", "Calls"],
    "wiki_results": ["EM", "Recall", "Tok", "AccTok", "Calls"],
    "sieve_indri_A": ["Judge", "EM", "Recall"],
    "sieve_indri_B": ["Tok", "AccTok", "Calls"],
}


def run_check(stats, verbose: bool, corpus: str = "full") -> int:
    tables = computed_tables(stats)
    diffs = []
    n_cells = 0
    for tname, rows in tables.items():
        path = TABLE_FILES[tname]
        _, matches = walk_table(path, [label for label, _ in rows])
        for i, ((label, comp), (_, _, stmt)) in enumerate(zip(rows, matches)):
            printed = parse_cells(stmt)
            if tname == "sieve_indri":
                cols = COLNAMES["sieve_indri_A" if i < 3 else "sieve_indri_B"]
            elif tname == "wiki_results":
                cols = COLNAMES["wiki_results"]
            else:
                cols = COLNAMES[tname]
            if len(printed) != len(comp):
                diffs.append((tname, label, "-", "shape",
                              f"{len(printed)} cells", f"{len(comp)} cells"))
                continue
            rowname = label if tname != "wiki_results" else \
                ("hp: " if i < len(rows) // 2 else "mq: ") + label
            if tname == "sieve_indri":
                rowname = ("acc: " if i < 3 else "cost: ") + label
            for cname, p, c in zip(cols, printed, comp):
                n_cells += 1
                _diff_one(diffs, tname, rowname, cname, p, c)
    n_cells += check_backbone(diffs, stats, corpus)

    print(f"\n=== paper-table check: {n_cells} numeric cells recomputed ===")
    if diffs:
        print(f"{len(diffs)} DIFF(S):")
        print(f"{'table':<18} {'row':<46} {'column':<18} {'kind':<6} {'printed':<16} computed")
        for t, r, c, k, p, comp in diffs:
            print(f"{t:<18} {r:<46} {c:<18} {k:<6} {p:<16} {comp}")
    else:
        print("no diffs — every printed value matches the recomputed value.")

    # --- reference-only: figure headline stats (never counted as diffs) ---------------------------
    print("\n--- figure headline stats (reference only) ---")
    base, sieve = stats[("bcp", "visit_bm25")], stats[("bcp", "sieve")]
    mB, mS = base["per"], sieve["per"]
    shared = set(mB) & set(mS)
    n_gain = sum(1 for i in shared if bool(mS[i]["judge"]) and not bool(mB[i]["judge"]))
    n_loss = sum(1 for i in shared if bool(mB[i]["judge"]) and not bool(mS[i]["judge"]))
    n_saved = sum(1 for i in shared if mS[i]["tok"] < mB[i]["tok"])
    print(f"fig_gainloss: gained={n_gain} lost={n_loss} fewer-tokens on {n_saved}/{len(shared)}"
          f" (judge cov: base {base['judge_cov']:.3f}, sieve {sieve['judge_cov']:.3f})")
    cap = (FIGS_DIR / "fig_gainloss.tex").read_text()
    m = re.search(r"fewer tokens on \$(\d+)\$\s*\nof \$(\d+)\$ queries.*?\$(\d+)\$ queries gained "
                  r"against \$(\d+)\$ lost", cap, re.S)
    if not m:
        m = re.search(r"fewer tokens on \$(\d+)\$\s*of\s*\$(\d+)\$.*?\$(\d+)\$ queries gained\s*"
                      r"against\s*\$(\d+)\$ lost", cap, re.S)
    if m:
        cs, ct, cg, cl = (int(x) for x in m.groups())
        ok = (cs, ct, cg, cl) == (n_saved, len(shared), n_gain, n_loss)
        print(f"fig_gainloss caption says: saved {cs}/{ct}, gained {cg}, lost {cl} "
              f"-> {'MATCH' if ok else 'MISMATCH'}")
    if base["judge"] is not None and sieve["judge"] is not None:
        d_acc = round(sieve["judge"], 1) - round(base["judge"], 1)
        d_tok = (sieve["tok"] - base["tok"]) / base["tok"] * 100
        print(f"fig_pareto: Sieve vs Search-Visit arrow: {d_acc:+.1f} judge points, "
              f"{d_tok:+.1f}% tokens (caption: +2.5, -32.4%)")
    else:
        print("fig_pareto: SKIPPED (judge coverage <90% on visit_bm25 or sieve)")

    print(f"\n{'PASS' if not diffs else 'FAIL'}: {len(diffs)} diff(s) across "
          f"{len(TABLE_FILES)} table files")
    return 0 if not diffs else 1


def _refuse_pending(tname: str, label: str, cells: list) -> None:
    if any(PENDING in c.text for c in cells):
        raise RuntimeError(
            f"--emit refused: {tname} row {label!r} contains a PENDING ({PENDING}) cell "
            "(judge coverage <90% on a BCP cell) — wait for the judge daemon, then re-run")


def run_emit(stats, outdir: Path, corpus: str = "full") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tables = computed_tables(stats)
    for tname, rows in tables.items():
        path = TABLE_FILES[tname]
        lines, matches = walk_table(path, [label for label, _ in rows])
        out, prev = [], 0
        for (label, comp), (s0, s1, _) in zip(rows, matches):
            _refuse_pending(tname, label, comp)
            out += lines[prev:s0]
            out.append(f"{label} & " + " & ".join(c.latex() for c in comp) + " \\\\")
            prev = s1 + 1
        out += lines[prev:]
        (outdir / path.name).write_text("\n".join(out))
        print(f"wrote {outdir / path.name}")
    # backbone_transfer (2026-07-31): the author owns this table's compact one-row-per-
    # backbone layout — emit copies it verbatim; --check verifies its numbers instead.
    path = TABLE_FILES["backbone_transfer"]
    (outdir / path.name).write_text(path.read_text())
    print(f"wrote {outdir / path.name}")


def run_selftest(stats, corpus: str = "full") -> int:
    """Per-instance equality of this script's intrinsic+overlay path vs
    analysis.ablation_deltas.cell_metrics (which whole-file-loads via load_rows_with_recovery)
    on two small cells."""
    from analysis.ablation_deltas import cell_metrics
    checks = [("mq", "sieve", "musique_structured", "_headline_validation",
               "agent_research_bql_dense_snip"),
              ("bcp", "oneshot_dense", None, None, None)]
    ok = True
    for dskey, ck, ds, sub, cond in checks:
        mine = stats[(dskey, ck)]["per"]
        if ds is None:
            from scripts.force_answer_backfill import load_rows_with_recovery
            from scripts.compare_cells import load_judge_cache, metrics
            bcp_ds = BCP_DS[corpus]
            cdir = cell_paths(bcp_ds, corpus)["oneshot_dense"]
            rows = load_rows_with_recovery(cdir)
            ref = metrics(rows, load_qrels(bcp_ds), bcp_ds, load_judge_cache(cdir))
        else:
            ref, _cov = cell_metrics(ds, sub, cond, load_qrels(ds))
        same_ids = set(mine) == set(ref)
        bad = [i for i in mine if i in ref and (
            mine[i]["em"] != ref[i]["em"] or mine[i]["judge"] != ref[i]["judge"]
            or mine[i]["tok"] != ref[i]["tok"] or mine[i]["recall"] != ref[i]["recall"]
            or mine[i]["llm_calls"] != ref[i]["llm_calls"])]
        print(f"selftest {dskey}/{ck}: ids match={same_ids}, mismatching instances={len(bad)}")
        ok = ok and same_ids and not bad
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    os.chdir(ROOT)   # compare_cells / load_qrels / the compare cache use repo-relative paths
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="recompute + diff vs live .tex (default)")
    ap.add_argument("--emit", metavar="OUTDIR", default=None,
                    help="write regenerated .tex files here (captions/preambles verbatim)")
    ap.add_argument("--selftest", action="store_true",
                    help="verify this pipeline against analysis.ablation_deltas.cell_metrics")
    ap.add_argument("--corpus", choices=("pooled", "full"), default="full",
                    help="BrowseComp-Plus corpus for every BCP cell (default: full). 'pooled' is "
                         "the historical mapping, kept for provenance; wiki cells are identical "
                         "under both")
    ap.add_argument("--bcp-report", action="store_true",
                    help="print a per-BCP-cell stats table (n / judge cov / judge / EM / recall / "
                         "tok / acc-tok / calls / star vs baseline) before the check/emit step")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore + don't write analysis/paper_tables_cache.json")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("PT_WORKERS", "4")),
                    help="parallel streaming workers for cache misses (default 4; the AutoRead "
                         "cells are 16-31GB but are streamed, never whole-loaded)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    datasets = datasets_for(args.corpus)
    print(f">> BCP corpus: {args.corpus} (dataset {datasets['bcp'][0]})", file=sys.stderr)
    all_cells = []
    for dskey, (ds, _nexp) in datasets.items():
        for ck, cond_dir in cell_paths(ds, args.corpus).items():
            all_cells.append((dskey, ck, cond_dir, ds))
    stats = gather_cells(all_cells, args.workers, not args.no_cache, True)
    add_stars(stats)

    # sanity gates (WARN, never fail): row counts on every cell + judge coverage on BCP cells
    for dskey, (ds, nexp) in datasets.items():
        for ck, cond_dir in cell_paths(ds, args.corpus).items():
            s = stats[(dskey, ck)]
            if s["n"] != nexp:
                print(f"WARNING: {dskey}/{ck} n={s['n']} != expected {nexp} ({cond_dir})")
            if dskey == "bcp" and s["judge_cov"] < 1.0:
                sev = (f"< 0.90 — Acc/Judge columns render {PENDING} and --emit will refuse"
                       if s["judge"] is None else "< 1.00 — judge pass still filling")
                print(f"WARNING: bcp/{ck} judge coverage {s['judge_cov']:.3f} {sev} ({cond_dir})")

    if args.bcp_report:
        print(f"\n--- per-BCP-cell stats ({args.corpus} corpus) ---")
        print(f"{'cell':<16} {'n':>4} {'jcov':>6} {'judge':>6} {'em':>6} {'recall':>6} "
              f"{'tok':>8} {'acctok':>8} {'calls':>6} {'star':>5} {'p_em':>9}")
        for ck in cell_paths(datasets["bcp"][0], args.corpus):
            s = stats[("bcp", ck)]
            j = f"{s['judge']:.1f}" if s["judge"] is not None else PENDING
            p = f"{s['p_em']:.2e}" if s.get("p_em") is not None else "-"
            print(f"{ck:<16} {s['n']:>4} {s['judge_cov']:>6.3f} {j:>6} {s['em']:>6.1f} "
                  f"{s['recall']:>6.1f} {s['tok']:>8.0f} {s['tok_acc']:>8.0f} "
                  f"{s['calls']:>6.1f} {'*' if s.get('star') else '':>5} {p:>9}")

    rc = 0
    if args.selftest:
        rc = run_selftest(stats, args.corpus)
    if args.emit:
        run_emit(stats, Path(args.emit), args.corpus)
    if args.check or not (args.emit or args.selftest):
        rc = max(rc, run_check(stats, args.verbose, args.corpus))
    print(f"\nruntime: {time.time() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
