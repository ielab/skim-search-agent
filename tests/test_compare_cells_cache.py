"""Offline tests for scripts/compare_cells.py's per-cell incremental metrics cache
(analysis/.compare_cache/): a fully-idle cell (rows.jsonl byte size unchanged since the cached
payload) must never re-read rows.jsonl's new content at all; a grown rows.jsonl must only have
its new tail bytes parsed and regexed, not the whole file; a non-append change (prefix
rewritten or the file shrinks) must fall back to a full re-read from byte 0; and --no-cache
must bypass the cache path altogether, always reloading and never reading or writing
analysis/.compare_cache/.

These tests exercise `_compute_cell` (the same primitive `main()` dispatches to, whether via
the in-process "trusted unchanged" fast path or the ProcessPoolExecutor) directly, spying on
`_read_tail_lines` (the only function that ever touches rows.jsonl's content for new rows) to
distinguish "no read", "incremental tail read starting mid-file", and "full read from byte 0".
End-to-end correctness equivalence (incremental result equals full recompute) is covered by
tests/test_compare_cells_incremental.py; this file is about the mechanism.
"""
import json

import scripts.compare_cells as cc


def _write_row(cond_dir, instance_id="q1", final_answer="forty-two", gold_answer="forty-two"):
    cond_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "instance_id": instance_id, "final_answer": final_answer, "gold_answer": gold_answer,
        "observations": [], "prompt_tokens": 10, "completion_tokens": 5, "n_steps": 3,
    }
    with (cond_dir / "rows.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _spy_tail_reads(monkeypatch):
    """Wrap cc._read_tail_lines with a call log (list of start_byte args); this is the ONLY
    function in the incremental path that ever reads rows.jsonl's content, so its call log
    distinguishes a no-op ([]), an incremental tail read (start_byte > 0), and a full re-read
    (start_byte == 0)."""
    calls = []
    real = cc._read_tail_lines

    def wrapped(path, start_byte):
        calls.append(start_byte)
        return real(path, start_byte)

    monkeypatch.setattr(cc, "_read_tail_lines", wrapped)
    return calls


def test_trusted_unchanged_skips_rows_jsonl_content_read(tmp_path, monkeypatch):
    """A second run against an untouched rows.jsonl must not call _read_tail_lines at all (the
    byte-size-unchanged fast path trusts the cache outright)."""
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell1"
    _write_row(cond_dir)
    calls = _spy_tail_reads(monkeypatch)

    exists, m = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert exists is True and len(m) == 1
    assert calls == [0], "first run must be a full read from byte 0"

    exists2, m2 = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert (exists2, m2) == (exists, m)
    assert calls == [0], "unchanged rows.jsonl must never call _read_tail_lines again"


def test_growth_reads_only_the_new_tail(tmp_path, monkeypatch):
    """Appending a row must incrementally read starting at the PREVIOUS byte size, not from 0,
    and the resulting metrics dict must contain both the old and the new instance."""
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell2"
    _write_row(cond_dir, instance_id="q1")
    calls = _spy_tail_reads(monkeypatch)

    cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert calls == [0]
    size_after_first = (cond_dir / "rows.jsonl").stat().st_size

    _write_row(cond_dir, instance_id="q2")
    exists, m = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert calls == [0, size_after_first], "second read must start at the prior byte size, not 0"
    assert set(m) == {"q1", "q2"}


def test_prefix_rewrite_falls_back_to_full_reread(tmp_path, monkeypatch):
    """If rows.jsonl no longer starts with the exact bytes the cache last saw (a prune/repair
    rewrote it, even if the new content happens to have grown), the cache must fall back to a
    full re-read from byte 0 rather than silently merging garbage."""
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell3"
    _write_row(cond_dir, instance_id="q1", final_answer="forty-two")
    calls = _spy_tail_reads(monkeypatch)

    cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert calls == [0]

    # Rewrite the file's prefix in place (repair-style), instead of a pure append.
    (cond_dir / "rows.jsonl").write_text(
        json.dumps({"instance_id": "q1", "final_answer": "repaired-answer",
                    "gold_answer": "forty-two", "observations": [], "prompt_tokens": 10,
                    "completion_tokens": 5, "n_steps": 3}) + "\n"
        + json.dumps({"instance_id": "q2", "final_answer": "x", "gold_answer": "x",
                      "observations": [], "prompt_tokens": 1, "completion_tokens": 1,
                      "n_steps": 1}) + "\n")

    exists, m = cc._compute_cell(str(cond_dir), "ds", {}, True)
    assert calls[-1] == 0, "a rewritten prefix must trigger a full re-read from byte 0"
    assert set(m) == {"q1", "q2"}
    # the repaired content must actually be reflected (proves it wasn't served stale from cache)
    assert m["q1"]["em"] is False  # "repaired-answer" no longer matches gold "forty-two"


def test_no_cache_bypasses(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "CACHE_DIR", tmp_path / "cache")
    cond_dir = tmp_path / "cell4"
    _write_row(cond_dir)
    calls = _spy_tail_reads(monkeypatch)

    cc._compute_cell(str(cond_dir), "ds", {}, False)
    cc._compute_cell(str(cond_dir), "ds", {}, False)
    assert calls == [0, 0], "--no-cache must do a full read from byte 0 every time"
    assert not (tmp_path / "cache").exists(), "--no-cache must never write the cache either"
