#!/usr/bin/env python
"""One-command comparison table across all experiment cells.

    python scripts/compare_cells.py                 # full registry, markdown to stdout
    python scripts/compare_cells.py --dataset browsecomp_plus_structured
    python scripts/compare_cells.py --out docs/comparison_$(date +%Y%m%d_%H%M).md

Every run also writes the full markdown output to comparison_result.md at the repo root
(DEFAULT_OUT below, overwritten each run regardless of --out; --out is an additional copy).
The final stdout line names the file(s) written.

Per cell: n, EM% (canonical answer_em), lenient contains-gold%, gold-surfaced-in-observations%,
mean cumulative tokens, mean steps, empty% after the recovery overlay, recovered count, plus
paired EM delta and exact McNemar p against the same-dataset baseline on mutual instance ids.
The recovery overlay comes from force_answer_backfill.load_rows_with_recovery (sidecar files,
live-safe). `empty%` uses force_answer_backfill.needs_recovery, the same predicate the backfill
script selects rows on (empty, whitespace-only, or a placeholder answer such as "..."), so this
table's `empty` count and what force_answer_backfill.py actually recovers never disagree.
REGISTRY holds the current validation tier: edit it when cells change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_search.evaluation.metrics import answer_em  # noqa: E402
from agent_search.evaluation.rows import observations_of  # noqa: E402
from scripts.force_answer_backfill import (  # noqa: E402
    load_rows_tolerant, load_rows_with_recovery, needs_recovery,
)

MODEL_DIR = "Tongyi-DeepResearch-30B-A3B"
_DOCID_RE = None

# Every run always writes the full markdown output (tables + Legend) here, repo root,
# overwritten each run, regardless of --out (which remains an additional optional path).
# A module-level constant (not inlined in main()) so tests can monkeypatch it to a tmp path
# instead of touching the real repo-root file. Never under runs/ (repo convention).
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "comparison_result.md"

# --- per-cell metrics cache: INCREMENTAL, append-aware -----------------------------------------
# rows.jsonl is APPEND-only with unique instance_ids (agent_search.evaluation.run_eval.evaluate / _load_rows:
# each finished instance is appended once via `sink = open(rows_path, "a")`; a resumed run reads
# `done` ids first and only appends new ones, earlier lines are never rewritten). ~20+ cells are
# being appended to continuously by live SLURM jobs, so the OLD cache (keyed on whole-file
# mtime+size) missed on every run for those cells and paid a full reread + full gold_doc_recall
# regex over the entire file (wiki cells now 5000-7000+ rows, some files ~1GB). Fix: split each
# row's metrics into
#   (a) ROW-INTRINSIC (expensive: the regex, tok, llm_calls, surfaced, raw final_answer/gold_answer
#       string, instance_id), depends only on that row's bytes in rows.jsonl, which never change
#       once written. Cached INCREMENTALLY per cell as {instance_id -> intrinsic}, plus how many
#       bytes of rows.jsonl have been processed so far.
#   (b) OVERLAY-DEPENDENT (cheap: em/lenient/empty after the recovery overlay, judge verdict) , 
#       depends on recovered_answers.jsonl / judge_cache.jsonl, which can change for
#       already-written rows. Recomputed fresh every run from the cached intrinsic dict, never
#       cached itself.
# On each run: if rows.jsonl's byte size is unchanged since the cached payload, trust the cache
# outright (no I/O on rows.jsonl at all, the "idle cell" fast path). If it grew, confirm the
# growth is append-only (a hash of the first `n_bytes` cached bytes still matches what's on disk
# now) before reading only the new tail bytes/lines and merging their intrinsic metrics in; a
# shrink or a prefix-hash mismatch (e.g. prune_rows.py / a surgical repair rewrote the file) falls
# back to a full re-read from byte 0, correctness first. Never written under runs/ (repo
# convention: runs/ is data, not scratch). Cache key is just the cond_dir path (stable across
# appends, the payload itself carries the byte-offset/hash provenance needed to validate a hit).
CACHE_DIR = Path("analysis/.compare_cache")
# Bumped whenever the SHAPE of a cached intrinsic entry changes (e.g. new precomputed fields) , 
# _read_cache rejects any payload whose version doesn't match as a plain miss (self-healing full
# recompute for that cell), so an old-format cache file on disk can never cause a KeyError/crash
# against code that expects the new shape.
_CACHE_VERSION = 4  # bumped: added count-once in_tok/out_tok split (row-metrics dict gained tok_in/tok_out)


def _cache_key(cond_dir: Path) -> str:
    return hashlib.sha1(str(cond_dir).encode()).hexdigest()


def _cache_path(cond_dir: Path) -> Path:
    return CACHE_DIR / f"{_cache_key(cond_dir)}.json"


def _read_cache(cond_dir: Path):
    """The cached payload dict for cond_dir, or None on any miss (file absent, unreadable/corrupt
   , e.g. a torn write, wrong-shape, or a stale cache file from a different `_CACHE_VERSION` , 
    in which case callers silently fall back to a full recompute)."""
    cpath = _cache_path(cond_dir)
    if not cpath.exists():
        return None
    try:
        payload = json.loads(cpath.read_text())
        if not (isinstance(payload, dict) and "exists" in payload
                and payload.get("version") == _CACHE_VERSION):
            return None
        return payload
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _write_cache(cond_dir: Path, payload: dict) -> None:
    """Best-effort: a failed cache write must never fail the run (e.g. a racing sibling process,
    or a read-only analysis/ in some environment), the data is only an optimization."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cpath = _cache_path(cond_dir)
        tmp = cpath.with_name(cpath.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({**payload, "version": _CACHE_VERSION}))
        tmp.replace(cpath)
    except OSError:
        pass


def _file_prefix_hash(path: Path, n: int) -> str:
    """sha1 of the first `n` bytes of `path` (n<=0 -> hash of the empty string). Cheap relative to
    JSON-parsing + regexing the same bytes, this is what makes append-only validation affordable
    even on the ~1GB rows.jsonl files this repo now has."""
    h = hashlib.sha1()
    if n > 0:
        with path.open("rb") as fh:
            remaining = n
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    return h.hexdigest()


def _read_tail_lines(path: Path, start_byte: int):
    """Complete '\\n'-terminated lines from byte offset `start_byte` to EOF, plus the byte offset
    immediately after the last complete line consumed. A torn trailing line (a live job's
    in-progress write) is left unconsumed, the returned end offset stops before it, so it gets
    re-read whole once it's complete on a later call. Mirrors `load_rows_tolerant`'s / run_eval's
    `_load_rows`'s tolerance for a live-appending file (skip an unparsable/incomplete trailing
    line, never crash)."""
    with path.open("rb") as fh:
        fh.seek(start_byte)
        buf = fh.read()
    if not buf:
        return [], start_byte
    if buf.endswith(b"\n"):
        body = buf[:-1]
        lines = body.split(b"\n") if body else []
        consumed = len(buf)
    else:
        idx = buf.rfind(b"\n")
        if idx == -1:
            return [], start_byte  # the whole remainder is one torn line
        lines = buf[:idx].split(b"\n")
        consumed = idx + 1
    return lines, start_byte + consumed


def _qid_of(iid: str, dataset: str) -> str:
    """qids may themselves contain '__' (musique '2hop__X_Y'), strip the dataset prefix only."""
    return iid.split(f"{dataset}__", 1)[-1] if dataset and iid.startswith(f"{dataset}__") \
        else iid.rsplit("__", 1)[-1]


def _row_intrinsic(r: dict, gold_ids: set) -> dict:
    """The part of a row's metrics that depends only on that row's own bytes in rows.jsonl and
    never changes once the row is written: the gold_doc_recall regex, tok, llm_calls, surfaced,
    the raw gold_answer/final_answer strings (final_answer here is pre-recovery-overlay , 
    overlaying happens later in `_apply_overlay`), and em_raw/lenient_raw/empty_raw, the
    em/lenient/empty a row would have IF NOT recovery-overlaid. These three are, strictly, a
    function of only ans_raw/gold (both already intrinsic), so they're safe to compute once here
    and cache forever too: `answer_em` (agent_search/evaluation/metrics.py, out of this script's scope to
    change) normalizes both strings from scratch on every call, a real cost at 1000s of rows/cell
   , so `_apply_overlay` reuses these precomputed values for the (vast majority) of rows the
    recovery overlay never touches, and only re-derives em/lenient/empty from scratch for the
    small subset that actually got a genuine recovered answer."""
    gold = str(r.get("gold_answer") or "")
    ans_raw = str(r.get("final_answer") or "")
    obs = " ".join(observations_of(r))
    empty_raw = needs_recovery(ans_raw)
    return dict(
        gold=gold,
        ans_raw=ans_raw,
        surfaced=bool(gold and gold.lower() in obs.lower()),
        recall=gold_doc_recall(r, gold_ids),
        tok=(r["total_tokens_once"] if r.get("total_tokens_once") is not None
             else (r.get("initial_prompt_tokens") or 0)
                  + (r.get("context_once_tokens") or 0)
                  + (r.get("output_tokens") or r.get("completion_tokens") or 0)),
        # count-once input (what the model reads: initial prompt + each retrieved doc once) and
        # output (generated), these two sum to `tok` (total_tokens_once). Split into columns.
        tok_in=(r.get("initial_prompt_tokens") or 0) + (r.get("context_once_tokens") or 0),
        tok_out=(r.get("output_tokens") or r.get("completion_tokens") or 0),
        llm_calls=r.get("llm_calls") or r.get("n_steps") or len(observations_of(r)),
        em_raw=bool(answer_em(ans_raw, gold)),
        lenient_raw=bool(gold and gold.lower() in ans_raw.lower()),
        empty_raw=empty_raw,
    )


def _load_recovered_map(cond_dir: Path) -> dict:
    """instance_id -> recovered_answer for GENUINE recoveries only, read directly from the sibling
    `recovered_answers.jsonl` sidecar WITHOUT touching rows.jsonl at all (that's the point of
    keeping the overlay cheap). Mirrors `force_answer_backfill.load_rows_with_recovery`'s own
    overlay-candidate filter exactly: a sidecar entry whose recovered_answer is itself a
    placeholder/empty (`needs_recovery` true) is not a valid overlay candidate, and a later
    genuine entry for the same id wins over an earlier one (dict overwrite in file order, same as
    the original)."""
    out: dict = {}
    p = cond_dir / "recovered_answers.jsonl"
    if not p.exists():
        return out
    for rec in load_rows_tolerant(str(p)):
        iid = rec.get("instance_id")
        if iid and not needs_recovery(rec.get("recovered_answer")):
            out[iid] = rec.get("recovered_answer") or ""
    return out


def _apply_overlay(intrinsic: dict, cond_dir: Path) -> dict:
    """The OVERLAY-DEPENDENT half of metrics(): em/lenient/empty (after the recovery overlay) and
    the judge verdict, computed FRESH every call from the cached intrinsic dict plus a fresh read
    of the (cheap) recovered_answers.jsonl / judge_cache.jsonl sidecars, so a judge/recovery pass
    landing new results is picked up correctly without ever re-reading rows.jsonl. Produces
    exactly the same per-instance dict shape as `metrics()` below (and is provably equivalent to
    it: `metrics()` is itself expressed in terms of `_row_intrinsic`, see below).

    For the (vast majority of) rows the recovery overlay never touches, `ans == ans_raw`
    identically, so em/lenient/empty are read straight off `_row_intrinsic`'s precomputed
    em_raw/lenient_raw/empty_raw rather than re-normalizing the same strings through
    `answer_em` again on every single run, that redundant renormalization (of every row, every
    warm run) was the actual remaining hot path once rows.jsonl re-parsing was eliminated. Only
    the small subset of genuinely-recovered rows pay for a fresh `answer_em` call, on their (new,
    different) recovered answer."""
    from hashlib import sha1 as _sha1
    recovered_by_id = _load_recovered_map(cond_dir)
    judge_cache = load_judge_cache(cond_dir)
    out = {}
    for iid, ri in intrinsic.items():
        ans_raw = ri["ans_raw"]
        gold = ri["gold"]
        rec = recovered_by_id.get(iid)
        if rec is not None and needs_recovery(ans_raw):
            ans, recovered = rec, True
            em = bool(answer_em(ans, gold))
            lenient = bool(gold and gold.lower() in ans.lower())
            empty = needs_recovery(ans)
        else:
            ans, recovered = ans_raw, False
            em, lenient, empty = ri["em_raw"], ri["lenient_raw"], ri["empty_raw"]
        out[iid] = dict(
            em=em,
            lenient=lenient,
            surfaced=ri["surfaced"],
            recall=ri["recall"],
            tok=ri["tok"],
            tok_in=ri["tok_in"],
            tok_out=ri["tok_out"],
            llm_calls=ri["llm_calls"],
            empty=empty,
            recovered=recovered,
            judge=judge_cache.get((iid, _sha1(ans.strip().encode()).hexdigest())),
        )
    return out


def load_qrels(dataset: str) -> dict:
    """qid -> set of gold corpus ids (strings), from data/<dataset>/qrels/*.tsv."""
    gold: dict = {}
    for p in (Path("data") / dataset / "qrels").glob("*.tsv"):
        for ln in p.read_text().splitlines()[1:]:
            parts = ln.split("\t")
            if len(parts) >= 3 and parts[2].strip() not in ("0", ""):
                gold.setdefault(parts[0].strip(), set()).add(parts[1].strip())
    return gold


def gold_doc_recall(row, gold_ids: set) -> bool:
    """True if any gold corpus id appears in a RETRIEVAL context of the observations.
    String ids (wiki 'd_ed_wood'): word-bounded anywhere (zero false positives measured).
    Numeric ids (browsecomp): only in listing lines ('  1  25898  '), fetch echoes ('[25898 ยง'),
    or dci file paths ('./25898.txt'), free-text numbers ('118 episodes') must not count
    (verified false positive browsecomp__991)."""
    if not gold_ids:
        return False
    import re
    blob = "\n".join(observations_of(row))
    for g in gold_ids:
        ge = re.escape(g)
        if g.isdigit():
            pat = rf"(?m)(?:^\s*\d+\s+{ge}\s)|(?:\[{ge}[\s\u00a7])|(?:\./{ge}\.txt)|(?:\b{ge}\b(?=\s+'))"
        else:
            pat = rf"\b{ge}\b"
        if re.search(pat, blob):
            return True
    return False
# (label, runs_subdir, dataset, condition, is_baseline_for_its_dataset)
REGISTRY = [
    # --- FULL-CORPUS BrowseComp-Plus cells (2026-07-29 fleet; dataset browsecomp_plus_structured_full) ---
    ("fc visit bm25 [baseline]", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_bm25", True),
    ("fc visit dense", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_dense", False),
    ("fc visit hybrid", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_hybrid", False),
    ("fc fetch bm25", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_bm25_fetch_snip", False),
    ("fc fetch dense", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_dense_fetch", False),
    ("fc fetch hybrid", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_hybrid_fetch_snip", False),
    ("fc autoread bm25", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_bm25_autoread", False),
    ("fc autoread dense", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_dense_autoread", False),
    ("fc dci", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_dci", False),
    ("fc bm25-dci", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_bm25_dci", False),
    ("fc sieve", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_bql_dense_snip", False),
    ("fc sieve-bm25", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_snip", False),
    ("fc sieve-dense", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_bql_donly_snip", False),
    ("fc sieve-nosnip", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_bql_dense_fetch", False),
    ("fc indri", "_fullcorpus", "browsecomp_plus_structured_full", "agent_research_indri_snip", False),
    ("fc indri+dense", "_fullcorpus_indridense", "browsecomp_plus_structured_full", "agent_research_indri_snip", False),
    ("fc sieve strictbool", "_fullcorpus_strictbool", "browsecomp_plus_structured_full", "agent_research_bql_dense_snip", False),
    ("fc sieve emb06", "_fullcorpus_emb06", "browsecomp_plus_structured_full", "agent_research_bql_dense_snip", False),
    ("fc flat baseline", "_flatbaseline", "browsecomp_plus_flat_full", "agent_research_bm25", False),
    ("fc sieve bgesmall", "_fullcorpus_bgesmall", "browsecomp_plus_structured_full", "agent_research_bql_dense_snip", False),
    ("fc sieve bgelarge", "_fullcorpus_bgelarge", "browsecomp_plus_structured_full", "agent_research_bql_dense_snip", False),
    ("fc sieve emb8b", "_fullcorpus_emb8b", "browsecomp_plus_structured_full", "agent_research_bql_dense_snip", False),
    ("fc sieve emb4b", "_fullcorpus_emb4b", "browsecomp_plus_structured_full", "agent_research_bql_dense_snip", False),
    ("SERP bm25 [BASELINE]", "_visit_uncapped", "browsecomp_plus_structured", "agent_research_bm25", True),
    ("flat-twin bm25", "_visit_uncapped", "browsecomp_plus_flat", "agent_research_bm25", True),
    ("SERP bm25 k=10", "_visit_uncapped_k10", "browsecomp_plus_structured", "agent_research_bm25", False),
    ("bm25 auto-read", "_visit_uncapped", "browsecomp_plus_structured", "agent_research_bm25_autoread", False),
    ("dense auto-read", "_visit_uncapped", "browsecomp_plus_structured", "agent_research_dense_autoread", False),
    ("dense visit", "_fullvisit", "browsecomp_plus_structured", "agent_research_dense", False),
    ("indri visit", "_fullvisit", "browsecomp_plus_structured", "agent_research_indri_visit", False),
    ("bql visit", "_fullvisit", "browsecomp_plus_structured", "agent_research_bql_visit", False),
    ("indri+dense visit", "_fullvisit_dense", "browsecomp_plus_structured", "agent_research_indri_visit", False),
    ("hybrid rrf visit", "_fullvisit", "browsecomp_plus_structured", "agent_research_hybrid", False),
    ("bql+dense visit", "_fullvisit", "browsecomp_plus_structured", "agent_research_bql_dense_visit", False),
    ("bm25+snip fetch", "_fetch_k5", "browsecomp_plus_structured", "agent_research_bm25_fetch_snip", False),
    ("dense+snip fetch", "_fetch_k5", "browsecomp_plus_structured", "agent_research_dense_fetch", False),
    ("hybrid+snip fetch", "_fetch_k5", "browsecomp_plus_structured", "agent_research_hybrid_fetch_snip", False),
    ("bql(donly)+snip fetch", "_newconds", "browsecomp_plus_structured", "agent_research_bql_donly_snip", False),
    ("hybrid+snip fetch (k10)", "_fetch_k10", "browsecomp_plus_structured", "agent_research_hybrid_fetch_snip", False),
    ("bql+dense+snip fetch", "_headline_validation", "browsecomp_plus_structured", "agent_research_bql_dense_snip", False),
    ("bql+dense fetch", "_headline_validation", "browsecomp_plus_structured", "agent_research_bql_dense_fetch", False),
    ("qwen dense visit", "_qwen_fullvisit", "browsecomp_plus_structured", "agent_research_dense", False),
    ("qwen hybrid visit", "_qwen_fullvisit", "browsecomp_plus_structured", "agent_research_hybrid", False),
    ("qwen bql+dense visit", "_qwen_fullvisit", "browsecomp_plus_structured", "agent_research_bql_dense_visit", False),
    ("qwen indri+dense visit", "_qwen_fullvisit_dense", "browsecomp_plus_structured", "agent_research_indri_visit", False),
    ("qwen indri+dense+snip fetch", "_qwen_dense_validation", "browsecomp_plus_structured", "agent_research_indri_snip", False),
    ("bql fetch", "_headline_validation", "browsecomp_plus_structured", "agent_research", False),
    ("bql+snip fetch", "_headline_validation", "browsecomp_plus_structured", "agent_research_snip", False),
    ("indri fetch", "_headline_validation", "browsecomp_plus_structured", "agent_research_indri", False),
    ("indri+snip fetch", "_headline_validation", "browsecomp_plus_structured", "agent_research_indri_snip", False),
    ("dense+snip fetch (k10)", "_fetch_k10", "browsecomp_plus_structured", "agent_research_dense_fetch", False),
    ("plain dense fetch (k10)", "_fetch_k10", "browsecomp_plus_structured", "agent_research_dense_fetch_plain", False),
    ("bm25+snip fetch (k10)", "_fetch_k10", "browsecomp_plus_structured", "agent_research_bm25_fetch_snip", False),
    ("indri+dense+snip fetch", "_dense_validation", "browsecomp_plus_structured", "agent_research_indri_snip", False),
    ("dci", "agent", "browsecomp_plus_structured", "agent_research_dci", False),
    ("bm25->dci", "agent", "browsecomp_plus_structured", "agent_research_bm25_dci", False),
    ("SERP bm25 [BASELINE]", "_visit_uncapped", "musique_structured", "agent_research_bm25", True),
    ("SERP bm25 k=10", "_visit_uncapped_k10", "musique_structured", "agent_research_bm25", False),
    ("indri+dense+snip fetch", "_dense_validation", "musique_structured", "agent_research_indri_snip", False),
    ("indri+dense visit", "_fullvisit_dense", "musique_structured", "agent_research_indri_visit", False),
    ("dense visit", "_fullvisit", "musique_structured", "agent_research_dense", False),
    ("hybrid rrf visit", "_fullvisit", "musique_structured", "agent_research_hybrid", False),
    ("bm25 auto-read", "_visit_uncapped", "musique_structured", "agent_research_bm25_autoread", False),
    ("dense auto-read", "_visit_uncapped", "musique_structured", "agent_research_dense_autoread", False),
    ("bm25+snip fetch", "_fetch_k5", "musique_structured", "agent_research_bm25_fetch_snip", False),
    ("dense+snip fetch", "_fetch_k5", "musique_structured", "agent_research_dense_fetch", False),
    ("hybrid+snip fetch", "_fetch_k5", "musique_structured", "agent_research_hybrid_fetch_snip", False),
    ("bql(donly)+snip fetch", "_newconds", "musique_structured", "agent_research_bql_donly_snip", False),
    ("bm25-dci", "agent", "musique_structured", "agent_research_bm25_dci", False),
    ("bm25+snip fetch (k10)", "_fetch_k10", "musique_structured", "agent_research_bm25_fetch_snip", False),
    ("bql+dense+snip fetch", "_headline_validation", "musique_structured", "agent_research_bql_dense_snip", False),
    ("bql+dense fetch", "_headline_validation", "musique_structured", "agent_research_bql_dense_fetch", False),
    ("plain dense fetch (k10)", "_fetch_k10", "musique_structured", "agent_research_dense_fetch_plain", False),
    ("bql+snip fetch", "_headline_validation", "musique_structured", "agent_research_snip", False),
    ("dense+snip fetch (k10)", "_fetch_k10", "musique_structured", "agent_research_dense_fetch", False),
    ("hybrid+snip fetch (k10)", "_fetch_k10", "musique_structured", "agent_research_hybrid_fetch_snip", False),
    ("dci", "agent", "musique_structured", "agent_research_dci", False),
    ("SERP bm25 [BASELINE]", "_visit_uncapped", "hotpotqa_structured", "agent_research_bm25", True),
    ("SERP bm25 k=10", "_visit_uncapped_k10", "hotpotqa_structured", "agent_research_bm25", False),
    ("indri+dense+snip fetch", "_dense_validation", "hotpotqa_structured", "agent_research_indri_snip", False),
    ("indri+dense visit", "_fullvisit_dense", "hotpotqa_structured", "agent_research_indri_visit", False),
    ("dense visit", "_fullvisit", "hotpotqa_structured", "agent_research_dense", False),
    ("hybrid rrf visit", "_fullvisit", "hotpotqa_structured", "agent_research_hybrid", False),
    ("bm25 auto-read", "_visit_uncapped", "hotpotqa_structured", "agent_research_bm25_autoread", False),
    ("dense auto-read", "_visit_uncapped", "hotpotqa_structured", "agent_research_dense_autoread", False),
    ("bm25+snip fetch", "_fetch_k5", "hotpotqa_structured", "agent_research_bm25_fetch_snip", False),
    ("dense+snip fetch", "_fetch_k5", "hotpotqa_structured", "agent_research_dense_fetch", False),
    ("hybrid+snip fetch", "_fetch_k5", "hotpotqa_structured", "agent_research_hybrid_fetch_snip", False),
    ("bql(donly)+snip fetch", "_newconds", "hotpotqa_structured", "agent_research_bql_donly_snip", False),
    ("bm25-dci", "agent", "hotpotqa_structured", "agent_research_bm25_dci", False),
    ("bm25+snip fetch (k10)", "_fetch_k10", "hotpotqa_structured", "agent_research_bm25_fetch_snip", False),
    ("bql+dense+snip fetch", "_headline_validation", "hotpotqa_structured", "agent_research_bql_dense_snip", False),
    ("bql+dense fetch", "_headline_validation", "hotpotqa_structured", "agent_research_bql_dense_fetch", False),
    ("plain dense fetch (k10)", "_fetch_k10", "hotpotqa_structured", "agent_research_dense_fetch_plain", False),
    ("bql+snip fetch", "_headline_validation", "hotpotqa_structured", "agent_research_snip", False),
    ("dense+snip fetch (k10)", "_fetch_k10", "hotpotqa_structured", "agent_research_dense_fetch", False),
    ("hybrid+snip fetch (k10)", "_fetch_k10", "hotpotqa_structured", "agent_research_hybrid_fetch_snip", False),
    ("dci", "agent", "hotpotqa_structured", "agent_research_dci", False),
]
ONESHOT = [("one-shot bm25", "bm25"), ("one-shot dense", "dense")]  # browsecomp only

# One line per distinct cell label (order = first appearance in REGISTRY/ONESHOT), rendered as
# the "## Legend" section after the tables. Wording is derived from the actual condition/toolset
# code, not guessed: agent_search/strategies/paper.py (condition -> task x strategy
# binding + tool descriptions), agent_search/tools/*/tool.py (the tools:
# Bm25Visit/DenseVisit/HybridVisit/BqlVisitWorkspace/DocSearchFetch/...), agent_search/agent/
# retriever.py (the env-knob retrofits, INDRI_DENSE=1 / BQL_DENSE=1 attach a DenseBelief onto
# the same condition's executor; these knobs are set per RUN SUBDIR, not per condition name, which
# is why e.g. "indri visit" and "indri+dense visit" share the same REGISTRY `cond` string), and
# agent_search/retrievers/dense/belief.py (DENSE_MODEL env override for the
# "qwen *" cells' embedder swap) / scripts/oneshot_rag.py (the one-shot baseline).
# A drift guard (tests/test_compare_cells.py) asserts this list's labels exactly match the
# distinct labels in REGISTRY + ONESHOT, keep both in sync when either changes.
LEGEND_CELLS = [
    ("SERP bm25 [BASELINE]",
     "The reference condition every other cell is compared against. The agent searches with "
     "plain keyword search (BM25, the classic ranked-keyword algorithm) over the document "
     "collection, gets back a listing of the top 5 results (title + a short preview snippet), "
     "then chooses results to 'visit' — open and read the entire document. No coaching/prompt "
     "hints beyond the tool descriptions themselves."),
    ("flat-twin bm25",
     "The exact same plain-keyword-search-then-visit-whole-document agent as the baseline above, "
     "but run against a different copy of the corpus where every document's internal structure "
     "(section headings, infobox fields, etc.) has been stripped out, leaving flat unstructured "
     "text. Comparing this cell to the baseline shows what having document structure is worth by "
     "itself, before any structure-aware search method is even applied."),
    ("SERP bm25 k=10",
     "Identical to the baseline, except the search listing shows 10 results per query instead of "
     "the baseline's 5. Tests whether simply showing the agent more candidates (with no change to "
     "search or reading method) changes accuracy."),
    ("bm25 auto-read",
     "Plain keyword search, but there is no separate 'open and read' step at all: each search "
     "call immediately dumps the full text of the top 5 results into the response. This removes "
     "the agent's choice of which result to open, isolating how much value that preview-then-pick "
     "step adds over just reading everything found."),
    ("dense auto-read",
     "Identical to 'bm25 auto-read' (each search dumps the full text of the top 5 results, no "
     "separate open-and-read step), except retrieval is dense (semantic/meaning-based, same "
     "embedder as 'dense visit') instead of keyword BM25 — the search-only/no-read baseline on "
     "the dense engine."),
    ("dense visit",
     "Same shape as the baseline (search a ranked listing, then visit whole documents) but the "
     "search itself is swapped from keyword matching to dense (semantic/meaning-based) search: "
     "documents are ranked by how close their meaning is to the query, via an embedding model "
     "(BAAI/bge-base-en-v1.5), rather than by shared words. Useful when the right document uses "
     "different wording than the query."),
    ("indri visit",
     "Search uses a graded query language: instead of a document either matching or not matching "
     "(like plain keyword search), each document gets a score for how well it satisfies the "
     "query's operators (required/weighted/proximity terms, field and date filters), so a query "
     "never comes back with zero results — it always returns the closest matches. The listing "
     "shows a query-relevant snippet per result, and reading is whole-document 'visit' as in the "
     "baseline. Isolates the effect of the graded search language alone, holding the reading "
     "method (whole-document) fixed for a fair comparison to the baseline."),
    ("bql visit",
     "Search uses our Boolean field-tagged query language: the agent can require or exclude "
     "specific terms and filter by fields such as title, date, or section, similar to an advanced "
     "library search box. If a strict AND query would return zero hits, it automatically falls "
     "back to ranking documents by how many of the requirements each one still satisfies, rather "
     "than giving up. Reading is whole-document 'visit', matching the baseline's reading method so "
     "only the search language differs."),
    ("indri+dense visit",
     "Same graded query-language search as 'indri visit', but each candidate's score is re-scored "
     "by blending it with a meaning-based (dense/embedding) similarity score, and a few extra "
     "meaning-similar documents that the keyword-style query missed are pulled into the candidate "
     "pool too. This is a fusion INSIDE one already-retrieved candidate list, not a merge of two "
     "separately-ranked lists (contrast with 'hybrid rrf visit' below). Reading is whole-document "
     "visit, as in 'indri visit'."),
    ("hybrid rrf visit",
     "Runs plain keyword search and dense (meaning-based) search as two completely SEPARATE "
     "ranked lists, then blends them by combining each document's rank position in each list "
     "(reciprocal rank fusion) rather than by directly mixing their scores. Because it blends by "
     "rank rather than score, it works best when the two rankers tend to agree; when they "
     "disagree, the blend can be noisy. This is the control for isolating plain "
     "keyword+meaning fusion from any benefit of a genuinely structured query language. Reading "
     "is whole-document visit."),
    ("bql+dense visit",
     "Uses the SAME Boolean field-tagged filter as 'bql visit' to decide which documents are "
     "eligible at all — a document that fails the filter can never appear here no matter how "
     "similar its meaning is. Among the documents that pass the filter, only their ORDER changes: "
     "it is re-ranked by blending plain-keyword rank with meaning-based (dense) rank, the same "
     "rank-blending as 'hybrid rrf visit' but restricted to already-filtered candidates. Reading "
     "is whole-document visit."),
    ("hybrid+snip fetch",
     "Same keyword+meaning rank-fusion search as 'hybrid rrf visit', but each listing result now "
     "also shows a one-line excerpt — the single sentence from that document that best matches "
     "the query — and instead of opening the whole document, the agent pulls specific named "
     "sections of it ('fetch'), which is cheaper to read but requires picking the right section "
     "name."),
    ("bql+dense+snip fetch",
     "Same Boolean-field-tagged-filter-then-rank-blend search as 'bql+dense visit' (filter first, "
     "then re-rank the survivors by keyword+meaning), but each listing result shows a one-line "
     "best-matching excerpt, and reading pulls specific named sections instead of the whole "
     "document."),
    ("bql+dense fetch",
     "BQL Boolean-field-tagged-filter-then-rank-blend search (SAME dense fusion as 'bql+dense "
     "visit'/'bql+dense+snip fetch'), plain listing (no per-result excerpt), reads named sections "
     "('fetch') instead of the whole document — this is 'bql+dense+snip fetch' (the winner) minus "
     "the snippet, isolating what the excerpt itself contributes on top of the dense-fused filter "
     "and section-fetch read."),
    ("qwen dense visit",
     "Identical to 'dense visit', except the meaning-based search uses a stronger embedding model "
     "(Qwen3-Embedding-0.6B) instead of the default one (bge-base) — tests whether a better "
     "embedding model changes the result."),
    ("qwen hybrid visit",
     "Identical to 'hybrid rrf visit', except the meaning-based half of the fusion uses the "
     "stronger Qwen3-Embedding-0.6B embedding model instead of the default bge-base."),
    ("qwen bql+dense visit",
     "Identical to 'bql+dense visit', except the meaning-based re-ranking uses the stronger "
     "Qwen3-Embedding-0.6B embedding model instead of the default bge-base."),
    ("qwen indri+dense visit",
     "Identical to 'indri+dense visit', except the meaning-based score blended into the graded "
     "query-language search uses the stronger Qwen3-Embedding-0.6B embedding model instead of the "
     "default bge-base."),
    ("qwen indri+dense+snip fetch",
     "Identical to 'indri+dense+snip fetch' (below), except the meaning-based score blended in "
     "uses the stronger Qwen3-Embedding-0.6B embedding model instead of the default bge-base."),
    ("bql fetch",
     "Search uses our Boolean field-tagged query language (same as 'bql visit'): the agent can "
     "require or exclude specific terms and filter by fields such as title, date, or section, and "
     "if a strict query would return zero hits it automatically falls back to ranking documents by "
     "how many of the requirements each one still satisfies. The listing is plain (title + section "
     "outline, no per-result excerpt), and reading pulls specific named sections of a document "
     "('fetch') instead of opening the whole thing. This is the direct Boolean-query analog of "
     "'indri fetch' below, and the no-snippet / no-dense sibling of 'bql+snip fetch' below and "
     "'bql+dense+snip fetch' above — included so the BQL read-interface family (visit / plain-fetch "
     "/ snippet-fetch) is complete."),
    ("bql+snip fetch",
     "Search uses our Boolean field-tagged query language (same as 'bql visit'), and each listing "
     "result also shows a one-line best-matching excerpt of its actual content — added because "
     "without it, a correct document whose match is only in its title or section names (not "
     "visible content) could get skipped by the agent. Reading pulls specific named sections "
     "instead of the whole document."),
    ("indri fetch",
     "Graded query-language search (same scoring style as 'indri visit': every candidate scored "
     "for how well it satisfies the query, never zero results), with a plain listing (no "
     "per-result excerpt). Reading pulls specific named sections of a document instead of opening "
     "the whole thing."),
    ("indri+snip fetch",
     "Same graded query-language search as 'indri fetch', plus a one-line best-matching excerpt "
     "shown per listing result. Reading pulls specific named sections instead of the whole "
     "document."),
    ("dense+snip fetch",
     "Pure meaning-based (dense/embedding) search — the query-engine analog of bm25/bql/indri "
     "'+snip fetch' — each listing result shows a one-line best-matching excerpt and reading pulls "
     "specific named sections. (Formerly labelled 'dense fetch'; the excerpt is always on, so it "
     "belongs to the snippet-fetch family.) Isolates whether the BQL structured query adds value "
     "over plain dense retrieval at the same snip+fetch interface (winner bql+dense+snip fetch vs this)."),
    ("plain dense fetch (k10)",
     "Pure meaning-based (dense/embedding) search — SAME retrieval as 'dense+snip fetch' — but "
     "the listing is PLAIN (title + section outline, no per-result excerpt), and reading pulls "
     "specific named sections instead of the whole document. This is 'dense+snip fetch' minus the "
     "snippet, isolating what the excerpt itself contributes on top of pure dense retrieval and "
     "section-fetch reading — the dense-only analog of 'bql+dense fetch' above."),
    ("bm25+snip fetch",
     "Plain keyword search (BM25), but each listing result now shows a one-line excerpt chosen for "
     "relevance to the query (instead of just the document's fixed opening lines) — a fairness fix "
     "so this baseline's listing is as informative as the other excerpt-showing cells. Reading "
     "pulls specific named sections instead of opening the whole document."),
    ("indri+dense+snip fetch",
     "Same graded-query-language-plus-meaning-based-blend search as 'indri+dense visit' (candidate "
     "scores blended with dense similarity, extra meaning-similar candidates pulled in), with a "
     "one-line best-matching excerpt shown per listing result. Reading pulls specific named "
     "sections instead of opening the whole document."),
    ("dci",
     "No search tool at all. The agent gets a plain command line (bash) and a file-reader over the "
     "raw document collection's files, and has to grep/list/read its way to the answer directly — "
     "the brute-force, closed-book baseline with no ranking or retrieval help whatsoever."),
    ("bm25->dci",
     "Plain keyword search first narrows the collection down to its top results, and only THEN "
     "does the agent get a command line and file-reader restricted to just those narrowed-down "
     "files — a middle ground between the pure 'dci' brute-force baseline (no narrowing at all) "
     "and the visit/fetch cells (a guided listing, not raw file access)."),
    ("one-shot bm25",
     "No agent loop and no tool calls at all: the top results from a single plain keyword search "
     "are pasted directly into one prompt, and the model produces one answer in a single call — "
     "the simplest possible retrieval-augmented setup, used to show what an agent loop adds over "
     "one-shot retrieval."),
    ("one-shot dense",
     "Same single-prompt, single-call setup as 'one-shot bm25', but the documents pasted into the "
     "prompt come from a meaning-based (dense/embedding, bge-base) search instead of plain keyword "
     "search."),
    ('bm25+snip fetch (k10)',
     'Search-Fetch with BM25 ranking and snippet cards at the k=10 pool size (pool-size variation of the k=5 main condition).'),
    ('dense+snip fetch (k10)',
     'Search-Fetch with dense ranking and snippet cards at k=10 (pool-size variation).'),
    ('hybrid+snip fetch (k10)',
     'Search-Fetch with BM25+Dense fusion and snippet cards at k=10 (pool-size variation).'),
    ('bm25-dci',
     'BM25-bounded direct corpus interaction (RISE-style): shell search + file reads inside a staged 10-document working set (pooled corpus).'),
    ('bql(donly)+snip fetch',
     'Sieve with dense-only ranking over the Boolean-admitted candidates (pooled corpus).'),
    ('fc autoread bm25',
     'FULL-corpus BrowseComp-Plus: Search-AutoRead with BM25 ranking.'),
    ('fc autoread dense',
     'FULL-corpus: Search-AutoRead with dense ranking.'),
    ('fc dci',
     'FULL-corpus: direct corpus interaction (no retriever).'),
    ('fc bm25-dci',
     'FULL-corpus: BM25-bounded DCI (RISE-style).'),
    ('fc visit bm25 [baseline]',
     'FULL-corpus: BM25 Search-Visit — the baseline for all full-corpus comparisons.'),
    ('fc visit dense',
     'FULL-corpus: dense Search-Visit.'),
    ('fc visit hybrid',
     'FULL-corpus: BM25+Dense Search-Visit.'),
    ('fc fetch bm25',
     'FULL-corpus: BM25 Search-Fetch (section reading).'),
    ('fc fetch dense',
     'FULL-corpus: dense Search-Fetch.'),
    ('fc fetch hybrid',
     'FULL-corpus: BM25+Dense Search-Fetch.'),
    ('fc sieve',
     'FULL-corpus: Sieve with the default BM25+Dense fused ranking.'),
    ('fc sieve-bm25',
     'FULL-corpus: Sieve with BM25 ranking.'),
    ('fc sieve-dense',
     'FULL-corpus: Sieve with dense-only ranking.'),
    ('fc sieve-nosnip',
     'FULL-corpus: Sieve without query-focused snippets (result-representation ablation).'),
    ('fc sieve strictbool',
     'FULL-corpus: strict-Boolean Sieve (BQL_SOFT_FALLBACK=0) — the zero-hit fallback disabled.'),
    ('fc sieve bgesmall',
     'FULL-corpus: Sieve with the bge-small-en-v1.5 dense encoder (retriever-sensitivity ladder).'),
    ('fc sieve bgelarge',
     'FULL-corpus: Sieve with the bge-large-en-v1.5 dense encoder (ladder).'),
    ('fc sieve emb06',
     'FULL-corpus: Sieve with Qwen3-Embedding-0.6B (ladder).'),
    ('fc sieve emb4b',
     'FULL-corpus: Sieve with Qwen3-Embedding-4B (ladder).'),
    ('fc sieve emb8b',
     'FULL-corpus: Sieve with Qwen3-Embedding-8B (ladder).'),
    ('fc indri',
     'FULL-corpus: Indri-QL structured executor with snippet cards (engine comparison).'),
    ('fc indri+dense',
     'FULL-corpus: Indri executor with dense belief fusion (engine comparison).'),
    ('fc flat baseline',
     'FULL-corpus FLAT twin: BM25 Search-Visit over the flat corpus (structure-availability control).'),
]

LEGEND_COLUMNS = [
    ("n", "Number of questions scored in this cell."),
    ("judge%", "Percent of answers a GPT-4o-mini judge model (following the BrowseComp-Plus "
               "grading protocol) rated as correct. An answer that is an exact string match to "
               "the gold answer is auto-accepted without spending a judge call."),
    ("Δjudge / p_judge", "The judge-accuracy gap versus this dataset's baseline cell, computed "
                          "only on the questions both cells answered, plus how statistically "
                          "significant that gap is (exact McNemar test — a small p-value means the "
                          "gap is unlikely to be chance; a p-value near 1 means no real "
                          "difference)."),
    ("EM%", "Percent of answers that exactly match the gold answer string (strict exact-match "
            "scoring, no judge model involved)."),
    ("ΔEM / p_em", "Same as Δjudge / p_judge, but using strict exact-match instead of the judge "
                    "model's rating."),
    ("recall%", "Percent of questions where a document actually known to contain the answer "
                "(from the gold reference list) shows up somewhere the agent retrieved — in a "
                "search result listing, a section it fetched, or a file it opened — not just "
                "mentioned in passing."),
    ("surfaced%", "Percent of questions where the gold answer's exact text appeared anywhere in "
                  "what the agent saw during the episode, regardless of whether it came from a "
                  "known-correct source document. A weaker, looser signal than recall%."),
    ("avg_tok/inst", "Average distinct tokens per question, each counted ONCE: the initial prompt "
                     "plus every retrieved/read document counted a single time (not re-counted when "
                     "the same context is re-sent on later steps) plus the model's generated output. "
                     "This measures the unique content volume the episode actually consumed, not the "
                     "step-summed processing total (which re-counts the growing prompt every step and "
                     "runs far larger). From the per-row `total_tokens_once` field."),
    ("in_tok/inst", "The INPUT half of avg_tok/inst, counted once: the initial prompt plus every "
                    "retrieved/read document counted a single time. This is the unique content the "
                    "model had to read. in_tok + out_tok = avg_tok/inst."),
    ("out_tok/inst", "The OUTPUT half of avg_tok/inst: the tokens the model generated across the "
                     "episode (all steps' completions). in_tok + out_tok = avg_tok/inst."),
    ("avg_llm_calls", "Average number of separate model calls made per question."),
    ("empty%", "Percent of final answers that came back blank, whitespace-only, or a placeholder "
               "like \"...\" — measured AFTER an offline recovery pass has already tried to "
               "refill blanks by re-asking the model for a forced answer."),
    ("recov", "Number of rows in this cell whose blank/placeholder answer was successfully "
              "refilled by that offline recovery pass."),
    ("n_mut", "Number of questions this cell has in common with the dataset's baseline cell — the "
              "shared set the Δ / p-value columns are computed over."),
]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar via binomial test on the discordant pairs."""
    try:
        from scipy.stats import binomtest
        n = b + c
        return 1.0 if n == 0 else binomtest(min(b, c), n, 0.5).pvalue
    except ImportError:
        return float("nan")


def cell_dir(subdir: str, dataset: str, cond: str) -> Path:
    if subdir == "agent":
        return Path("runs/agent") / dataset / MODEL_DIR / cond
    return Path("runs") / subdir / "agent" / dataset / MODEL_DIR / cond


def cell_rows(subdir: str, dataset: str, cond: str):
    p = cell_dir(subdir, dataset, cond) / "rows.jsonl"
    if not p.exists():
        return None
    return load_rows_with_recovery(p.parent)


def load_judge_cache(cond_dir: Path) -> dict:
    """(instance_id, sha1(final_answer.strip())) -> bool from scripts/judge_cells.py's sibling cache."""
    import json
    out = {}
    p = cond_dir / "judge_cache.jsonl"
    if p.exists():
        for ln in p.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                out[(r["instance_id"], r["answer_sha1"])] = bool(r["judge_correct"])
    return out


def metrics(rows, qrels=None, dataset="", judge_cache=None):
    """Full from-scratch computation over an already-overlaid row list (as produced by
    `load_rows_with_recovery`: `final_answer` already replaced by any genuine recovery,
    `recovered` already set). Expressed in terms of `_row_intrinsic`/`_qid_of`, the same building
    blocks the incremental cache (`_compute_cell`/`_apply_overlay`) uses, so the two computation
    paths can never silently drift apart; `ans_raw` from `_row_intrinsic` IS the (possibly already
    overlaid) `final_answer` on `r` here, since this function doesn't do its own overlaying."""
    from hashlib import sha1 as _sha1
    out = {}
    for r in rows:
        iid = r.get("instance_id") or ""
        qid = _qid_of(iid, dataset)
        ri = _row_intrinsic(r, (qrels or {}).get(qid, set()))
        ans, gold = ri["ans_raw"], ri["gold"]
        out[iid] = dict(
            em=bool(answer_em(ans, gold)),
            lenient=bool(gold and gold.lower() in ans.lower()),
            surfaced=ri["surfaced"],
            recall=ri["recall"],
            tok=ri["tok"],
            tok_in=ri["tok_in"],
            tok_out=ri["tok_out"],
            llm_calls=ri["llm_calls"],
            empty=needs_recovery(ans),
            recovered=bool(r.get("recovered")),
            judge=(judge_cache or {}).get((iid, _sha1(ans.strip().encode()).hexdigest())),
        )
    return out


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else 0.0


def _dataset_cells(ds: str) -> list:
    return [(lbl, sub, c, bl) for lbl, sub, d, c, bl in REGISTRY if d == ds]


def _compute_cell(cond_dir_str: str, dataset: str, qrels: dict, use_cache: bool):
    """Incremental worker: bring this cell's cached intrinsic dict up to date with rows.jsonl
    (reading only newly-appended bytes when the cache validates as still append-only; a full
    re-read from byte 0 otherwise), then apply the overlay-dependent fields fresh. Runs inside a
    ProcessPoolExecutor worker when dispatched from main() (the regex-heavy intrinsic pass is
    CPU-bound, so threads wouldn't help under the GIL), module-level and picklable-argument-only
    (str/dict of str/bool) so it survives the fork/pickle boundary; also safe to call directly
    in-process (tests do this, and main()'s own "trusted unchanged" fast path does too).
    Returns (exists, metrics_dict): `exists` is whether rows.jsonl was present (registry cells
    collapse "absent" and "present-but-empty" to the same "no data" outcome via `metrics_dict`
    being empty either way; the ONESHOT cells in main() additionally need `exists` on its own,
    since they render an n=0 row rather than skipping when the file exists but is empty)."""
    cond_dir = Path(cond_dir_str)
    rows_path = cond_dir / "rows.jsonl"
    if not rows_path.exists():
        if use_cache:
            _write_cache(cond_dir, {"exists": False})
        return False, {}

    cached = _read_cache(cond_dir) if use_cache else None
    start = 0
    intrinsic: dict = {}
    trusted_unchanged = False
    if cached and cached.get("exists") and "n_bytes" in cached and "prefix_hash" in cached:
        cached_n = cached["n_bytes"]
        try:
            size_now = rows_path.stat().st_size
        except OSError:
            size_now = -1
        if size_now == cached_n:
            # Unchanged byte size: nothing was appended, and this pipeline never rewrites
            # rows.jsonl in place at an identical byte length (prune/repair always change the
            # row count, hence the length), trust the cache without touching rows.jsonl at all.
            # This is the near-zero-cost path for a cell that is fully idle between two runs.
            start = cached_n
            intrinsic = dict(cached.get("intrinsic") or {})
            trusted_unchanged = True
        elif size_now > cached_n and _file_prefix_hash(rows_path, cached_n) == cached["prefix_hash"]:
            # Confirmed append-only growth: the bytes we already processed are still exactly the
            # file's prefix, so only the new tail needs the expensive regex pass.
            start = cached_n
            intrinsic = dict(cached.get("intrinsic") or {})
        # else: size shrank, or the prefix hash no longer matches (e.g. a prune/repair rewrote
        # the file) -> fall back to a full re-read from byte 0 (start/intrinsic stay at defaults).

    if trusted_unchanged:
        end_byte = start
    else:
        lines, end_byte = _read_tail_lines(rows_path, start)
        for ln in lines:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue  # a fully-written-but-corrupt line; skip it like load_rows_tolerant does
            iid = r.get("instance_id") or ""
            qid = _qid_of(iid, dataset)
            intrinsic[iid] = _row_intrinsic(r, (qrels or {}).get(qid, set()))

        if use_cache:
            new_hash = _file_prefix_hash(rows_path, end_byte)
            _write_cache(cond_dir, {"exists": True, "n_bytes": end_byte, "prefix_hash": new_hash,
                                     "intrinsic": intrinsic})

    m = _apply_overlay(intrinsic, cond_dir)
    return True, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--out", default=None,
                     help="also write markdown here (IN ADDITION to the always-written "
                          "repo-root comparison_result.md)")
    ap.add_argument("--no-cache", action="store_true",
                     help="bypass analysis/.compare_cache/: always reload rows.jsonl and recompute "
                          "(also skips writing the cache)")
    args = ap.parse_args()
    use_cache = not args.no_cache

    lines = []
    datasets = sorted({d for _, _, d, _, _ in REGISTRY if not args.dataset or d == args.dataset})
    qrels_by_ds = {ds: load_qrels(ds) for ds in datasets}

    # Enumerate every cell this run needs (registry cells across all selected datasets, plus the
    # browsecomp one-shot baselines) so cache misses can be loaded in ONE parallel batch. The main
    # process still does all cross-cell pairing (McNemar, deltas) below, unaffected by load order.
    job_dirs = {}  # cond_dir (str) -> dataset
    for ds in datasets:
        for lbl, sub, cond, bl in _dataset_cells(ds):
            job_dirs[str(cell_dir(sub, ds, cond))] = ds
    for ds in datasets:
        for lbl, variant in ONESHOT:
            job_dirs[str(Path("runs/_oneshot") / ds / variant)] = ds

    # Split cells into three tiers so the common case (a fully idle cell between two runs) never
    # pays subprocess/IPC overhead or a redundant cache-file reparse:
    #   - rows.jsonl absent -> resolved trivially in-process, no work.
    #   - cache present and rows.jsonl's byte size is unchanged since it -> "trusted unchanged":
    #     resolved in-process directly from the cache payload we already read here (only the cheap
    #     overlay is recomputed), never touches rows.jsonl, and never re-reads/re-parses the
    #     cache file a second time the way calling _compute_cell again would.
    #   - anything else (cold, grown, or shrunk/rewritten) -> genuinely needs the regex-heavy
    #     intrinsic pass, dispatched to the ProcessPoolExecutor for CPU parallelism.
    all_m = {}  # cond_dir (str) -> (exists, metrics_dict)
    work = []
    for cond_dir_str, ds in job_dirs.items():
        cond_dir = Path(cond_dir_str)
        rows_path = cond_dir / "rows.jsonl"
        if not rows_path.exists():
            all_m[cond_dir_str] = (False, {})
            continue
        cached = _read_cache(cond_dir) if use_cache else None
        trusted = False
        if cached and cached.get("exists") and "n_bytes" in cached:
            try:
                trusted = rows_path.stat().st_size == cached["n_bytes"]
            except OSError:
                trusted = False
        if trusted:
            all_m[cond_dir_str] = (True, _apply_overlay(cached.get("intrinsic") or {}, cond_dir))
        else:
            work.append((cond_dir_str, ds))

    if work:
        # COMPARE_WORKERS=1 for the giant auto-read cells (17-31GB rows.jsonl): 8 parallel
        # parses of multi-GB files OOM the shared login node (BrokenProcessPool, 2026-07-19).
        max_workers = int(os.environ.get("COMPARE_WORKERS", min(8, os.cpu_count() or 1)))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_compute_cell, cd, ds, qrels_by_ds[ds], use_cache): cd
                    for cd, ds in work}
            for fut in as_completed(futs):
                all_m[futs[fut]] = fut.result()

    def get_metrics(sub, ds, cond):
        return all_m[str(cell_dir(sub, ds, cond))][1]

    for ds in datasets:
        cells = _dataset_cells(ds)
        base = next(((lbl, sub, c) for lbl, sub, c, bl in cells if bl), None)
        qr = qrels_by_ds[ds]
        base_m = get_metrics(base[1], ds, base[2]) if base else {}
        rows_out = []
        for lbl, sub, cond, is_base in cells:
            m = get_metrics(sub, ds, cond)
            if not m:
                rows_out.append((lbl, None))
                continue
            vals = list(m.values())
            judged = [v["judge"] for v in vals if v["judge"] is not None]
            rec = dict(
                n=len(m), em=pct([v["em"] for v in vals]), lenient=pct([v["lenient"] for v in vals]),
                surfaced=pct([v["surfaced"] for v in vals]), recall=pct([v["recall"] for v in vals]),
                judge=pct(judged) if judged and len(judged) >= 0.9 * len(vals) else None, njudged=len(judged),
                tok=sum(v["tok"] for v in vals) // max(len(vals), 1), tok_in=sum(v["tok_in"] for v in vals) // max(len(vals), 1), tok_out=sum(v["tok_out"] for v in vals) // max(len(vals), 1),
                llm_calls=sum(v["llm_calls"] for v in vals) / max(len(vals), 1),
                empty=pct([v["empty"] for v in vals]), nrec=sum(v["recovered"] for v in vals),
            )
            if base_m and not is_base:
                mut = set(m) & set(base_m)
                b = sum(1 for i in mut if m[i]["em"] and not base_m[i]["em"])
                c = sum(1 for i in mut if base_m[i]["em"] and not m[i]["em"])
                em_mut = pct([m[i]["em"] for i in mut])
                em_base_mut = pct([base_m[i]["em"] for i in mut])
                rec.update(delta=em_mut - em_base_mut, nmut=len(mut), p=mcnemar_p(b, c))
                jmut = [i for i in mut if m[i]["judge"] is not None and base_m[i]["judge"] is not None]
                if jmut:
                    jb = sum(1 for i in jmut if m[i]["judge"] and not base_m[i]["judge"])
                    jc = sum(1 for i in jmut if base_m[i]["judge"] and not m[i]["judge"])
                    rec.update(jdelta=pct([m[i]["judge"] for i in jmut]) - pct([base_m[i]["judge"] for i in jmut]),
                               jp=mcnemar_p(jb, jc))
            rows_out.append((lbl, rec))
        if True:  # one-shot floors: rendered for every dataset whose runs/_oneshot/<ds>/<variant> exists
            for lbl, variant in ONESHOT:
                cond_dir_str = str(Path("runs/_oneshot") / ds / variant)
                exists, m = all_m[cond_dir_str]
                if exists:
                    vals = list(m.values())
                    judged = [v["judge"] for v in vals if v["judge"] is not None]
                    rec = dict(n=len(m), em=pct([v["em"] for v in vals]), lenient=pct([v["lenient"] for v in vals]),
                               surfaced=float("nan"), recall=pct([v["recall"] for v in vals]), tok=sum(v["tok"] for v in vals) // max(len(vals), 1), tok_in=sum(v["tok_in"] for v in vals) // max(len(vals), 1), tok_out=sum(v["tok_out"] for v in vals) // max(len(vals), 1),
                               judge=pct(judged) if judged and len(judged) >= 0.9 * len(vals) else None, njudged=len(judged),
                               llm_calls=1.0, empty=pct([v["empty"] for v in vals]), nrec=0)
                    if base_m:
                        mut = set(m) & set(base_m)
                        b = sum(1 for i in mut if m[i]["em"] and not base_m[i]["em"])
                        c = sum(1 for i in mut if base_m[i]["em"] and not m[i]["em"])
                        rec.update(delta=pct([m[i]["em"] for i in mut]) - pct([base_m[i]["em"] for i in mut]),
                                   nmut=len(mut), p=mcnemar_p(b, c))
                        jmut = [i for i in mut if m[i]["judge"] is not None and base_m[i]["judge"] is not None]
                        if jmut:
                            jb = sum(1 for i in jmut if m[i]["judge"] and not base_m[i]["judge"])
                            jc = sum(1 for i in jmut if base_m[i]["judge"] and not m[i]["judge"])
                            rec.update(jdelta=pct([m[i]["judge"] for i in jmut]) - pct([base_m[i]["judge"] for i in jmut]),
                                       jp=mcnemar_p(jb, jc))
                    rows_out.append((lbl, rec))
        lines.append(f"\n## {ds}\n")
        lines.append("| cell | n | judge% | Δjudge | p_judge | EM% | ΔEM | p_em | recall% | surfaced% | avg_tok/inst | in_tok/inst | out_tok/inst | avg_llm_calls | empty% | recov | n_mut |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        key = lambda x: -(x[1] or {}).get("judge") if (x[1] or {}).get("judge") is not None else -(x[1] or {}).get("em", -1)
        for lbl, rec in sorted(rows_out, key=key):
            if rec is None:
                lines.append(f"| {lbl} | — | | | | | | | | | | | | | (no rows) |")
                continue
            j = f"{rec['judge']:.1f}" if rec.get("judge") is not None else "—"
            jd = f"{rec['jdelta']:+.1f}" if "jdelta" in rec else "—"
            jpv = f"{rec['jp']:.3g}" if "jp" in rec else "—"
            d = f"{rec['delta']:+.1f}" if "delta" in rec else "—"
            nm = rec.get("nmut", "—")
            pv = f"{rec['p']:.3g}" if "p" in rec else "—"
            lines.append(
                f"| {lbl} | {rec['n']} | {j} | {jd} | {jpv} | {rec['em']:.1f} | {d} | {pv} | {rec['recall']:.1f} | {rec['surfaced']:.1f} | "
                f"{rec['tok']/1000:,.0f}k | {rec['tok_in']/1000:,.0f}k | {rec['tok_out']/1000:,.0f}k | {rec['llm_calls']:.1f} | {rec['empty']:.1f} | {rec['nrec']} | {nm} |")
    lines.append("\n## Legend\n")
    lines.append("### Cells\n")
    for lbl, desc in LEGEND_CELLS:
        lines.append(f"- **{lbl}**: {desc}")
    lines.append("\n### Columns\n")
    for col, desc in LEGEND_COLUMNS:
        lines.append(f"- **{col}**: {desc}")
    text = "\n".join(lines)
    print(text)
    # always write the canonical output file (repo root, overwritten each run), --out is an
    # additional optional path, not a replacement. The final stdout line names what was written.
    rendered = f"# Cell comparison (auto-generated)\n{text}\n"
    written = [DEFAULT_OUT]
    DEFAULT_OUT.write_text(rendered)
    if args.out:
        Path(args.out).write_text(rendered)
        written.append(Path(args.out))
    print("\nwrote: " + " ".join(str(p) for p in written))


if __name__ == "__main__":
    main()
