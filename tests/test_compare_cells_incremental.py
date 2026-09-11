"""End-to-end correctness tests for scripts/compare_cells.py's incremental append-aware cache
(see the CACHE_DIR docstring in scripts/compare_cells.py for the full design rationale): the
incremental path must produce results byte-identical (as dicts) to a from-scratch, --no-cache-style
full recompute, in every one of the scenarios the cache is meant to handle:

  (i)   append-only growth: incremental result equals full recompute on the grown file
  (ii)  a sidecar (judge_cache/recovered_answers) change on an unchanged rows.jsonl: overlay
        fields (em/empty/judge/recovered) update correctly without rows.jsonl's intrinsic content
        ever being re-parsed
  (iii) a non-append change (prefix rewritten in place, or the file shrinks): falls back to a
        full recompute, and the result is correct (reflects the new content, not stale cache)
  (iv)  --no-cache bypasses the cache entirely (never reads or writes analysis/.compare_cache/)
"""
import json

import scripts.compare_cells as cc


def _row(instance_id, final_answer="forty-two", gold_answer="forty-two", **extra):
    row = {
        "instance_id": instance_id, "final_answer": final_answer, "gold_answer": gold_answer,
        "observations": [f"some text mentioning {gold_answer} in passing"],
        "prompt_tokens": 10, "completion_tokens": 5, "n_steps": 3,
    }
    row.update(extra)
    return row


def _write_rows(cond_dir, rows, mode="w"):
    cond_dir.mkdir(parents=True, exist_ok=True)
    with (cond_dir / "rows.jsonl").open(mode) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _full_recompute(cond_dir):
    """The ground truth: exactly what --no-cache computes (also exactly what metrics() over
    load_rows_with_recovery computes — the pre-existing, well-tested code path)."""
    exists, m = cc._compute_cell(str(cond_dir), "ds", {}, False)
    return exists, m


# --- (i) append-only growth -----------------------------------------------------------------

def test_append_only_growth_matches_full_recompute(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row("q1"), _row("q2", final_answer="")])

    # warm the cache
    first = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert first[1]["q1"]["em"] is True
    assert first[1]["q2"]["empty"] is True

    # append more rows to the SAME file (simulating a live SLURM job)
    _write_rows(cond_dir, [_row("q3", final_answer="wrong"), _row("q4")], mode="a")

    incremental = cc._compute_cell(str(cond_dir), "ds", {}, True)
    full = _full_recompute(cond_dir)
    assert incremental == full
    assert set(incremental[1]) == {"q1", "q2", "q3", "q4"}


def test_append_only_growth_incremental_reads_only_new_tail(tmp_path, monkeypatch):
    """Directly proves the incremental result equals a from-scratch recompute of the appended
    file, AND that getting there only required parsing the new tail (not the whole file) —
    spying on _read_tail_lines's start_byte argument."""
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row(f"q{i}") for i in range(5)])
    cc._compute_cell(str(cond_dir), "ds", {}, True)
    size_before = (cond_dir / "rows.jsonl").stat().st_size

    _write_rows(cond_dir, [_row("q_new", final_answer="brand new")], mode="a")

    calls = []
    real = cc._read_tail_lines

    def spy(path, start_byte):
        calls.append(start_byte)
        return real(path, start_byte)

    monkeypatch.setattr(cc, "_read_tail_lines", spy)
    incremental = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert calls == [size_before], "must read starting at the prior byte offset, not from 0"

    monkeypatch.undo()
    full = _full_recompute(cond_dir)
    assert incremental == full


# --- (ii) sidecar-only change on an unchanged rows.jsonl -------------------------------------

def test_sidecar_change_updates_overlay_without_reparsing_rows_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row("q1", final_answer="")])  # empty -> needs_recovery

    exists, m = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert m["q1"]["empty"] is True
    assert m["q1"]["recovered"] is False

    calls = []
    real = cc._read_tail_lines

    def spy(path, start_byte):
        calls.append(start_byte)
        return real(path, start_byte)

    monkeypatch.setattr(cc, "_read_tail_lines", spy)

    # land a recovery for q1 WITHOUT touching rows.jsonl at all
    (cond_dir / "recovered_answers.jsonl").write_text(
        json.dumps({"instance_id": "q1", "recovered_answer": "forty-two"}) + "\n")

    exists2, m2 = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert calls == [], "rows.jsonl content must NOT be re-parsed for an intrinsic-only reason"
    assert m2["q1"]["empty"] is False
    assert m2["q1"]["recovered"] is True
    assert m2["q1"]["em"] is True  # "forty-two" == gold_answer "forty-two"

    monkeypatch.undo()
    full = _full_recompute(cond_dir)
    assert (exists2, m2) == full


def test_judge_cache_change_updates_overlay_without_reparsing_rows_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row("q1", final_answer="close enough")])

    exists, m = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert m["q1"]["judge"] is None

    from hashlib import sha1
    ans_sha1 = sha1("close enough".strip().encode()).hexdigest()
    (cond_dir / "judge_cache.jsonl").write_text(
        json.dumps({"instance_id": "q1", "answer_sha1": ans_sha1, "judge_correct": True}) + "\n")

    calls = []
    real = cc._read_tail_lines
    monkeypatch.setattr(cc, "_read_tail_lines", lambda p, s: (calls.append(s), real(p, s))[1])

    exists2, m2 = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert calls == [], "a judge_cache-only change must not touch rows.jsonl content"
    assert m2["q1"]["judge"] is True

    monkeypatch.undo()
    full = _full_recompute(cond_dir)
    assert (exists2, m2) == full


# --- (iii) non-append change: prefix rewrite / shrink -----------------------------------------

def test_prefix_rewrite_falls_back_and_is_correct(tmp_path, monkeypatch):
    """A rewrite that changes the EARLY content but grows the file overall (size_now > cached_n)
    must still be caught by the prefix-hash check, not mistaken for a plain append."""
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row("q1", final_answer="old")])
    cached = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert cached[1]["q1"]["em"] is False

    # a "surgical row repair" rewrites q1's own line (not just appends after it) and adds q2 —
    # the file GROWS overall, but its first bytes no longer match what the cache last saw.
    (cond_dir / "rows.jsonl").unlink()
    _write_rows(cond_dir, [_row("q1", final_answer="forty-two"), _row("q2")])

    incremental = cc._compute_cell(str(cond_dir), "ds", {}, True)
    full = _full_recompute(cond_dir)
    assert incremental == full
    assert set(incremental[1]) == {"q1", "q2"}
    assert incremental[1]["q1"]["em"] is True, "must reflect the repaired content, not stale cache"


def test_shrink_falls_back_and_is_correct(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row("q1"), _row("q2"), _row("q3")])
    cc._compute_cell(str(cond_dir), "ds", {}, True)

    # prune_rows.py-style: rewrite with strictly fewer bytes than before
    (cond_dir / "rows.jsonl").unlink()
    _write_rows(cond_dir, [_row("q1")])

    incremental = cc._compute_cell(str(cond_dir), "ds", {}, True)
    full = _full_recompute(cond_dir)
    assert incremental == full
    assert set(incremental[1]) == {"q1"}


# --- (iv) --no-cache bypasses -------------------------------------------------------------------

def test_no_cache_never_touches_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row("q1")])

    cc._compute_cell(str(cond_dir), "ds", {}, False)
    assert not (tmp_path / "cache").exists()

    _write_rows(cond_dir, [_row("q2")], mode="a")
    exists, m = cc._compute_cell(str(cond_dir), "ds", {}, False)
    assert set(m) == {"q1", "q2"}
    assert not (tmp_path / "cache").exists(), "--no-cache must never write the cache"


def test_no_cache_result_matches_cached_result(tmp_path, monkeypatch):
    """--no-cache and the cached path must agree on output, they just differ in I/O."""
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell"
    _write_rows(cond_dir, [_row("q1"), _row("q2", final_answer="")])

    cached = cc._compute_cell(str(cond_dir), "ds", {}, True)
    uncached = cc._compute_cell(str(cond_dir), "ds", {}, False)
    assert cached == uncached
