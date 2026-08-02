"""scripts/prune_rows.py -- surgical rows.jsonl row removal.

Covers: basic prune, idempotency (second run is a no-op), backup creation,
unknown-ids no-op, and the malformed-tail-line PRESERVE policy (see the
script's module docstring for why: we cannot determine a malformed line's
instance_id, so dropping it would be silent, unrecoverable data loss --
this suite pins that a torn tail write survives a prune untouched).
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.getcwd())
from scripts.prune_rows import prune_rows, _read_ids_file  # noqa: E402


def _row(iid, **extra):
    d = {"instance_id": iid, "final_answer": "x"}
    d.update(extra)
    return d


def _write_rows(path, rows_or_lines):
    with open(path, "w") as fh:
        for r in rows_or_lines:
            if isinstance(r, str):
                fh.write(r if r.endswith("\n") else r + "\n")
            else:
                fh.write(json.dumps(r) + "\n")


def _read_rows(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def test_basic_prune_removes_only_targeted_ids(tmp_path):
    p = tmp_path / "rows.jsonl"
    _write_rows(p, [_row("a"), _row("b"), _row("c")])

    result = prune_rows(str(p), {"b"}, live_ok=True)

    assert result["removed"] == 1
    assert result["kept"] == 2
    assert result["unmatched_ids"] == []
    ids = [r["instance_id"] for r in _read_rows(p)]
    assert ids == ["a", "c"]


def test_requires_live_ok():
    with pytest.raises(SystemExit):
        prune_rows("/does/not/matter.jsonl", {"a"}, live_ok=False)


def test_backup_created_and_matches_pre_prune_content(tmp_path):
    p = tmp_path / "rows.jsonl"
    original_rows = [_row("a"), _row("b"), _row("c")]
    _write_rows(p, original_rows)

    result = prune_rows(str(p), {"b"}, live_ok=True)

    backup_path = result["backup_path"]
    assert os.path.dirname(backup_path) == str(tmp_path)
    assert os.path.basename(backup_path).startswith("rows.jsonl.pre_prune_")
    assert _read_rows(backup_path) == original_rows  # backup is the PRE-prune snapshot
    # original file has moved on; backup did not
    assert _read_rows(p) != original_rows


def test_idempotent_second_prune_is_noop(tmp_path):
    p = tmp_path / "rows.jsonl"
    _write_rows(p, [_row("a"), _row("b"), _row("c")])

    first = prune_rows(str(p), {"b"}, live_ok=True)
    assert first["removed"] == 1

    second = prune_rows(str(p), {"b"}, live_ok=True)
    assert second["removed"] == 0
    assert second["unmatched_ids"] == ["b"]
    ids = [r["instance_id"] for r in _read_rows(p)]
    assert ids == ["a", "c"]
    # each call makes its own backup, even when it's a no-op
    assert os.path.isfile(second["backup_path"])


def test_unknown_ids_are_noop_not_error(tmp_path):
    p = tmp_path / "rows.jsonl"
    _write_rows(p, [_row("a"), _row("b")])

    result = prune_rows(str(p), {"does-not-exist"}, live_ok=True)

    assert result["removed"] == 0
    assert result["kept"] == 2
    assert result["unmatched_ids"] == ["does-not-exist"]
    ids = [r["instance_id"] for r in _read_rows(p)]
    assert ids == ["a", "b"]


def test_malformed_tail_line_is_preserved_verbatim(tmp_path):
    """Policy: a torn tail write (last line only) is a live-append artifact,
    not corruption -- PRESERVE it unchanged rather than drop it, since we
    cannot know its instance_id and therefore cannot safely decide to prune
    it."""
    p = tmp_path / "rows.jsonl"
    torn_tail = '{"instance_id": "c", "final_answer": "unterm'  # no closing brace/quote
    _write_rows(p, [json.dumps(_row("a")), json.dumps(_row("b")), torn_tail])

    result = prune_rows(str(p), {"a"}, live_ok=True)

    assert result["removed"] == 1
    assert result["malformed_kept"] == 1
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert lines[-1] == torn_tail  # byte-for-byte preserved, not dropped or "fixed"


def test_malformed_non_tail_line_is_also_preserved_with_warning(tmp_path, capsys):
    p = tmp_path / "rows.jsonl"
    _write_rows(
        p,
        [
            json.dumps(_row("a")),
            "{this is not valid json at all",
            json.dumps(_row("b")),
        ],
    )

    result = prune_rows(str(p), {"a"}, live_ok=True)

    assert result["malformed_kept"] == 1
    err = capsys.readouterr().err
    assert "malformed JSON" in err
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert "{this is not valid json at all" in lines


def test_no_ids_given_raises(tmp_path):
    p = tmp_path / "rows.jsonl"
    _write_rows(p, [_row("a")])
    with pytest.raises(SystemExit):
        prune_rows(str(p), set(), live_ok=True)


def test_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit):
        prune_rows(str(tmp_path / "nope.jsonl"), {"a"}, live_ok=True)


def test_read_ids_file_skips_blank_and_comment_lines(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("a\n\n# comment\nb\n   \nc\n")
    assert _read_ids_file(str(f)) == ["a", "b", "c"]


def test_duplicate_instance_id_rows_both_removed(tmp_path):
    """Guards a resume-double-write scenario (see audit part b): if the same
    instance_id appears twice, pruning that id removes ALL matching rows,
    not just the first."""
    p = tmp_path / "rows.jsonl"
    _write_rows(p, [_row("a"), _row("a"), _row("b")])

    result = prune_rows(str(p), {"a"}, live_ok=True)

    assert result["removed"] == 2
    assert result["kept"] == 1
    ids = [r["instance_id"] for r in _read_rows(p)]
    assert ids == ["b"]
