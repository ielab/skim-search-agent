#!/usr/bin/env python
"""LLM-judge gate for browsecomp cells. The canonical answer for a row comes from
`force_answer_backfill.load_rows_with_recovery`'s overlay, and `rows.jsonl` must never be
mutated in place, so this script never re-extracts or rewrites `final_answer`, it only
reads rows and records judge verdicts in a sibling cache file (see below).

WHY a SIBLING CACHE instead of writing verdicts into rows.jsonl: rows.jsonl is the raw episode
record and must stay reproducible against the exact loop.py/policies.py that produced it (same
rule `force_answer_backfill.py` follows for `recovered_answers.jsonl`). Judging is also expensive
(API calls) and answers can change out from under a condition dir (recovery backfill lands after
the fact), so `judge_cache.jsonl` is keyed on `(instance_id, sha1(final_answer.strip()))`, not
just `instance_id`: a changed answer is a cache MISS, not a stale HIT, and gets re-judged; an
unchanged answer is a HIT and is never re-billed. This makes reruns of this script (after new rows
land, or after a backfill pass changes some answers) idempotent and strictly incremental.

WHY the EM short-circuit precedes the judge call: `agent_search.evaluation.metrics.answer_em` is the
deterministic, free, canonical QA exact-match. Any row that already passes it needs no LLM opinion
,  recording it as `method="em_shortcircuit"` (distinct from `judge_answer_detail`'s own internal
normalized-exact short-circuit, see below) saves the call and keeps the ledger auditable by method.

    python scripts/judge_cells.py --dry-run                  # preview only, no API calls
    python scripts/judge_cells.py --cell bm25 --limit 5      # smoke: a few real calls
    PYTHONPATH=. python scripts/judge_cells.py --workers 16  # full gate run

Cells judged: every `scripts.compare_cells.REGISTRY` row whose dataset starts with one of the
`--datasets` prefixes (comma-separated; default "browsecomp", covering both
`browsecomp_plus_structured` and `browsecomp_plus_flat`; "all" selects every REGISTRY dataset).
The wiki datasets (`hotpotqa_structured`, `musique_structured`) have the same row shape and
`judge_answer_detail` is dataset-agnostic, so extending the selection needs no protocol change.
The two ONESHOT dirs `runs/_oneshot/browsecomp_plus_structured/{bm25,dense}` ride along whenever
their dataset is selected. Registry/rows are imported, never copied, so this script tracks
`compare_cells.py`'s cell definitions automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_search.evaluation.llm_judge import judge_answer_detail, make_judge  # noqa: E402
from agent_search.evaluation.metrics import answer_em  # noqa: E402
from scripts.compare_cells import MODEL_DIR, ONESHOT, REGISTRY, cell_rows  # noqa: E402
from scripts.force_answer_backfill import load_rows_with_recovery  # noqa: E402

CACHE_NAME = "judge_cache.jsonl"
JUDGE_MODEL = "gpt-4o-mini"


# --- cell discovery -------------------------------------------------------------------------------

def _cond_dir_for(subdir: str, dataset: str, cond: str) -> Path:
    """The condition dir for a REGISTRY row. Mirrors `compare_cells.cell_rows`' path construction
    (that function returns loaded rows, not the path, this script also needs the directory, to
    find/write the sibling `judge_cache.jsonl`), so the two small branches are duplicated here."""
    p = Path("runs") / subdir / "agent" / dataset / MODEL_DIR / cond / "rows.jsonl"
    if subdir == "agent":
        p = Path("runs/agent") / dataset / MODEL_DIR / cond / "rows.jsonl"
    return p.parent


def dataset_selected(ds: str, prefixes: list) -> bool:
    """Prefix match against the --datasets selection; the sentinel "all" selects everything."""
    return "all" in prefixes or any(ds.startswith(p) for p in prefixes)


def selected_cells(prefixes: list) -> list:
    """[(label, dataset, cond_dir, subdir, cond, rows_loader), ...] for every REGISTRY cell whose
    dataset matches a `--datasets` prefix, plus the ONESHOT dirs when their dataset is selected.
    `rows_loader` is a zero-arg callable (lazy: --dry-run and --cell filtering should not
    force-load every cell's rows.jsonl before filtering narrows the set)."""
    cells = []
    for lbl, sub, ds, cond, _is_base in REGISTRY:
        if not dataset_selected(ds, prefixes):
            continue
        cdir = _cond_dir_for(sub, ds, cond)
        cells.append((lbl, ds, cdir, sub, cond, (lambda s=sub, d=ds, c=cond: cell_rows(s, d, c))))
    if dataset_selected("browsecomp_plus_structured", prefixes):
        for lbl, variant in ONESHOT:                  # runs/_oneshot/browsecomp_plus_structured/{bm25,dense}
            cdir = Path("runs/_oneshot/browsecomp_plus_structured") / variant
            cells.append((lbl, "browsecomp_plus_structured", cdir, "_oneshot", variant,
                          (lambda d=cdir: load_rows_with_recovery(d))))
    return cells


# --- verdict cache (sibling file, append-only, resumable) -----------------------------------------

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").strip().encode("utf-8")).hexdigest()


def load_cache(cond_dir: Path) -> dict:
    """{(instance_id, answer_sha1): record} from `<cond_dir>/judge_cache.jsonl`. Tolerant of an
    unparsable trailing line (a concurrently-running job still appending), mirroring
    `force_answer_backfill.load_rows_tolerant`."""
    path = cond_dir / CACHE_NAME
    cache: dict = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cache[(rec.get("instance_id"), rec.get("answer_sha1"))] = rec
    return cache


def append_cache(cond_dir: Path, record: dict, lock: threading.Lock) -> None:
    path = cond_dir / CACHE_NAME
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


def _record(iid, sha1: str, gold: str, method: str, detail: dict) -> dict:
    """{instance_id, answer_sha1, gold_answer, judge_correct, judge_extracted, judge_reasoning,
    method, judge_model}, `detail` is exactly `judge_answer_detail`'s return shape (real LLM calls
    and the two free synthetic verdicts below all produce this shape, so the cache is uniform)."""
    rec = {"instance_id": iid, "answer_sha1": sha1, "gold_answer": gold}
    rec.update(detail)
    rec["method"] = method
    rec["judge_model"] = JUDGE_MODEL
    return rec


# --- classification (no side effects; safe under --dry-run) ---------------------------------------

def classify_row(row: dict, cache: dict) -> Optional[dict]:
    """The verdict record for one row against the CURRENT cache, or None if it needs an actual
    `judge_answer_detail` call that has not happened yet ("llm_pending"). Never calls the judge and
    never writes anything, a cached hit is returned as-is; the empty/EM-shortcircuit verdicts are
    computed directly (free, deterministic) so counts are accurate even in --dry-run, which never
    persists them. Cache key is `(instance_id, sha1(final_answer.strip()))`, so a changed answer
    (recovery backfill landing after the fact) is a miss and gets (re-)classified here, not silently
    reused from a stale cache entry."""
    iid = row.get("instance_id")
    ans = str(row.get("final_answer") or "")
    sha1 = _sha1(ans)
    key = (iid, sha1)
    if key in cache:
        return cache[key]
    gold = str(row.get("gold_answer") or "")
    if not ans.strip():
        return _record(iid, sha1, gold, "empty", {
            "judge_correct": False, "judge_extracted": "None",
            "judge_reasoning": "empty final_answer (no judge call)"})
    if bool(answer_em(ans, gold)):
        return _record(iid, sha1, gold, "em_shortcircuit", {
            "judge_correct": True, "judge_extracted": ans,
            "judge_reasoning": "canonical EM match (agent_search.evaluation.metrics.answer_em, no judge call)"})
    return None


def summarize(rows: list, cache: dict) -> dict:
    """Aggregate counts over `rows` against the CURRENT cache (a pure read via `classify_row` , 
    call this both before and after `do_live_work` to see what changed)."""
    counts = {"em_shortcircuit": 0, "llm": 0, "empty": 0}
    correct = judged = pending = 0
    for row in rows:
        rec = classify_row(row, cache)
        if rec is None:
            pending += 1
            continue
        judged += 1
        counts[rec["method"]] = counts.get(rec["method"], 0) + 1
        if rec.get("judge_correct"):
            correct += 1
    acc = 100.0 * correct / judged if judged else 0.0
    return dict(n=len(rows), judged=judged, acc=acc, em_sc=counts["em_shortcircuit"],
               llm=counts["llm"], empty=counts["empty"], pending=pending)


# --- live work: persist free verdicts, then judge the rest (threaded, capped by --limit) ----------

def do_live_work(cond_dir: Path, rows: list, cache: dict, *, workers: int,
                 limit: Optional[int], generate: Callable[[str], str]) -> int:
    """Mutates `cache` in place and appends to `<cond_dir>/judge_cache.jsonl`: every empty/EM
    verdict not yet cached is persisted immediately (free, no API); up to `limit` (None = no cap)
    of the remaining "llm_pending" rows are graded via `judge_answer_detail`, threaded over
    `workers`. Returns the number of judge calls actually issued this run."""
    lock = threading.Lock()
    llm_todo = []
    for row in rows:
        iid = row.get("instance_id")
        sha1 = _sha1(str(row.get("final_answer") or ""))
        if (iid, sha1) in cache:
            continue
        rec = classify_row(row, cache)          # cache miss confirmed above -> fresh empty/em_sc rec, or None
        if rec is None:
            llm_todo.append((row, iid, sha1))
        else:
            cache[(iid, sha1)] = rec
            append_cache(cond_dir, rec, lock)
    if limit is not None:
        llm_todo = llm_todo[:limit]
    if not llm_todo:
        return 0

    def _work(item):
        row, iid, sha1 = item
        gold = str(row.get("gold_answer") or "")
        detail = judge_answer_detail(row.get("question") or "", gold,
                                     str(row.get("final_answer") or ""), generate)
        return iid, sha1, _record(iid, sha1, gold, "llm", detail)

    n_calls = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = [pool.submit(_work, item) for item in llm_todo]
        for fut in as_completed(futs):
            iid, sha1, rec = fut.result()
            cache[(iid, sha1)] = rec
            append_cache(cond_dir, rec, lock)
            n_calls += 1
    return n_calls


def run_cell(label: str, cond_dir: Path, rows: list, *, workers: int, limit: Optional[int],
            dry_run: bool, generate: Optional[Callable[[str], str]]) -> dict:
    cache = load_cache(cond_dir)
    if not dry_run:
        do_live_work(cond_dir, rows, cache, workers=workers, limit=limit, generate=generate)
    summary = summarize(rows, cache)
    summary["label"] = label
    summary["cond_dir"] = str(cond_dir)
    return summary


# --- CLI --------------------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="classify + print per-cell counts only; NO cache writes, NO API calls, "
                         "make_judge() is never instantiated")
    ap.add_argument("--workers", type=int, default=16, help="thread pool size for judge calls (default 16)")
    ap.add_argument("--cell", default=None,
                    help="substring filter, matched case-insensitively against the condition name "
                         "or runs_subdir (e.g. 'bm25', '_headline_validation')")
    ap.add_argument("--cell-exact", default=None,
                    help="EXACT condition-name match (comma-separated for several). Prefer this over "
                         "--cell when a name is a prefix of a much larger sibling: '--cell "
                         "agent_research_bm25' also matches agent_research_bm25_autoread, whose "
                         "rows.jsonl is ~16GB and costs ~45GB RSS to load for nothing.")
    ap.add_argument("--extra-cell", action="append", default=[], metavar="DIR",
                    help="judge an arbitrary condition dir containing rows.jsonl, bypassing the "
                         "REGISTRY (repeatable). Needed for run dirs the REGISTRY cannot express, "
                         "e.g. the cross-backbone cells under runs/_xbackbone/, whose model dir is "
                         "not compare_cells.MODEL_DIR. Verdicts land in that dir's judge_cache.jsonl "
                         "exactly as for a REGISTRY cell.")
    ap.add_argument("--limit", type=int, default=None,
                    help="max NEW judge API calls per cell this run (smoke testing); does not cap "
                         "the free em_shortcircuit/empty verdicts, which are always fully persisted")
    ap.add_argument("--datasets", default="browsecomp",
                    help="comma-separated dataset-name PREFIXES to judge (default 'browsecomp'; "
                         "'all' = every REGISTRY dataset, incl. hotpotqa_structured/musique_structured)")
    args = ap.parse_args(argv)

    prefixes = [p.strip() for p in args.datasets.split(",") if p.strip()]
    cells = selected_cells(prefixes)
    if args.cell:
        needle = args.cell.lower()
        cells = [c for c in cells if needle in c[4].lower() or needle in c[3].lower()]
    if args.cell_exact:
        wanted = {s.strip().lower() for s in args.cell_exact.split(",") if s.strip()}
        cells = [c for c in cells if c[4].lower() in wanted]
    if args.extra_cell and not (args.cell or args.cell_exact):
        # An explicit dir list means THAT LIST. Appending it to the unfiltered REGISTRY instead
        # silently re-adds every registry cell, including the 16GB autoread one (caught live
        # 2026-07-23: "32 cell(s) to judge", 44.5GB RSS). Pass --cell/--cell-exact alongside
        # --extra-cell to judge registry cells in the same invocation.
        cells = []
    for d in args.extra_cell:
        cdir = Path(d)
        if not (cdir / "rows.jsonl").exists():
            print(f"--extra-cell has no rows.jsonl: {cdir}", file=sys.stderr)
            return 1
        # label from the two path parts that actually distinguish these dirs (model dir / condition)
        cells.append((f"{cdir.parent.name}/{cdir.name}", "browsecomp_plus_structured", cdir,
                      "_extra", cdir.name, (lambda p=cdir: load_rows_with_recovery(p))))
    if not cells:
        print(f"no cells matched --datasets {args.datasets!r} / --cell {args.cell!r}", file=sys.stderr)
        return 1

    generate = None if args.dry_run else make_judge(JUDGE_MODEL)
    if args.dry_run:
        print(">> DRY RUN: no cache writes, no API calls (make_judge not instantiated)", file=sys.stderr)
    print(f">> {len(cells)} cell(s) to judge (datasets={','.join(prefixes)}, model={JUDGE_MODEL})",
          file=sys.stderr)

    results = []
    for label, ds, cond_dir, subdir, cond, loader in cells:
        # wiki labels repeat across datasets ("SERP bm25 [BASELINE]" x3), disambiguate them;
        # browsecomp labels stay verbatim so the default output is unchanged.
        if not ds.startswith("browsecomp"):
            label = f"{label} ({ds})"
        rows = loader()
        if not rows:
            print(f"{label} | n=0 (no rows at {cond_dir})")
            continue
        s = run_cell(label, cond_dir, rows, workers=args.workers, limit=args.limit,
                    dry_run=args.dry_run, generate=generate)
        results.append(s)
        print(f"{s['label']} | n={s['n']} judged={s['judged']} acc={s['acc']:.1f}% "
             f"(em_sc={s['em_sc']} llm={s['llm']} empty={s['empty']} pending={s['pending']})")

    lines = ["", "| cell | n | judged | acc% | em_sc | llm | empty | pending |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in results:
        lines.append(f"| {s['label']} | {s['n']} | {s['judged']} | {s['acc']:.1f} | "
                     f"{s['em_sc']} | {s['llm']} | {s['empty']} | {s['pending']} |")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
