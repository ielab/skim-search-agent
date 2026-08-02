"""Offline tests for scripts/compare_cells.py's `metrics()` — specifically that its `empty`
computation is the SAME predicate `scripts/force_answer_backfill.py` selects rows for recovery
on (`needs_recovery`), not a re-implemented inline check. This is the fix for the asymmetry bug:
force_answer_backfill used to select empty-only rows for recovery while compare_cells counted
placeholder answers ("...", ".", "..") as empty too, so placeholder rows in some cells were never
backfilled even though they were already being counted (and penalized) as empty here.
"""
from scripts.compare_cells import metrics
from scripts.force_answer_backfill import needs_recovery


def _row(instance_id, final_answer, gold_answer="forty-two"):
    return {
        "instance_id": instance_id, "final_answer": final_answer, "gold_answer": gold_answer,
        "observations": [], "prompt_tokens": 10, "completion_tokens": 5, "n_steps": 3,
    }


def test_metrics_empty_uses_needs_recovery_for_placeholders_and_true_empty():
    rows = [
        _row("truly_empty", ""),
        _row("whitespace_only", "   "),
        _row("dots3", "..."),
        _row("dot1", "."),
        _row("dots2", ".."),
        _row("real_answer", "forty-two"),
    ]
    out = metrics(rows)
    for iid in ("truly_empty", "whitespace_only", "dots3", "dot1", "dots2"):
        assert out[iid]["empty"] is True, iid
    assert out["real_answer"]["empty"] is False


def test_metrics_empty_agrees_with_needs_recovery_on_every_row():
    """metrics()'s `empty` field must never disagree with the predicate force_answer_backfill.py
    uses to SELECT rows for recovery — that disagreement was exactly the original bug."""
    answers = ["", "   ", "...", ".", "..", "  ...  ", "answer", "...and more", ". . ."]
    rows = [_row(f"r{i}", a) for i, a in enumerate(answers)]
    out = metrics(rows)
    for i, a in enumerate(answers):
        assert out[f"r{i}"]["empty"] == needs_recovery(a), repr(a)


def test_metrics_empty_is_compare_cells_own_needs_recovery_import():
    """Guard against a future re-divergence: compare_cells must IMPORT the predicate, not carry a
    parallel inline tuple check that could silently drift from force_answer_backfill's."""
    import scripts.compare_cells as cc
    import scripts.force_answer_backfill as fab
    assert cc.needs_recovery is fab.needs_recovery


def test_main_always_writes_comparison_result_md_matching_stdout(tmp_path, monkeypatch, capsys):
    """Every run must write the full markdown output (tables + Legend) to the repo-root
    comparison_result.md, regardless of --out, and the final stdout line must name it.
    DEFAULT_OUT is monkeypatched to a tmp path (never write the real repo-root file from a
    test), cwd is a tmp dir (no real runs/ or data/ is read — the one registry cell renders
    as '(no rows)'), and REGISTRY/ONESHOT are shrunk to keep the run fast and hermetic."""
    import sys
    import scripts.compare_cells as cc

    monkeypatch.chdir(tmp_path)
    out_file = tmp_path / "comparison_result.md"
    monkeypatch.setattr(cc, "DEFAULT_OUT", out_file)
    monkeypatch.setattr(cc, "REGISTRY", [
        ("SERP bm25 [BASELINE]", "_visit_uncapped", "browsecomp_plus_structured",
         "agent_research_bm25", True)])
    monkeypatch.setattr(cc, "ONESHOT", [])
    monkeypatch.setattr(sys, "argv", ["compare_cells.py", "--no-cache"])

    cc.main()

    stdout = capsys.readouterr().out
    assert out_file.exists()
    content = out_file.read_text()
    # the file is the SAME rendered text stdout printed, under the auto-generated header
    assert content.startswith("# Cell comparison (auto-generated)\n")
    body = content.split("\n", 1)[1]
    assert body.strip() in stdout
    assert "## Legend" in content
    # the FINAL stdout line names the written path
    assert stdout.rstrip().splitlines()[-1] == f"wrote: {out_file}"


def test_main_out_flag_writes_an_additional_copy(tmp_path, monkeypatch, capsys):
    """--out <path> remains an ADDITIONAL optional path: both files are written with identical
    content, and the final stdout line names both."""
    import sys
    import scripts.compare_cells as cc

    monkeypatch.chdir(tmp_path)
    default_file = tmp_path / "comparison_result.md"
    extra_file = tmp_path / "extra_copy.md"
    monkeypatch.setattr(cc, "DEFAULT_OUT", default_file)
    monkeypatch.setattr(cc, "REGISTRY", [
        ("SERP bm25 [BASELINE]", "_visit_uncapped", "browsecomp_plus_structured",
         "agent_research_bm25", True)])
    monkeypatch.setattr(cc, "ONESHOT", [])
    monkeypatch.setattr(sys, "argv", ["compare_cells.py", "--no-cache", "--out", str(extra_file)])

    cc.main()

    stdout = capsys.readouterr().out
    assert default_file.exists() and extra_file.exists()
    assert default_file.read_text() == extra_file.read_text()
    assert stdout.rstrip().splitlines()[-1] == f"wrote: {default_file} {extra_file}"


def test_legend_covers_every_registry_and_oneshot_label():
    """Guard against LEGEND_CELLS drifting out of sync with REGISTRY/ONESHOT: every distinct cell
    label the tables can render must have exactly one Legend entry, and vice versa (no orphaned
    legend entries for labels that no longer exist)."""
    import scripts.compare_cells as cc
    registry_labels = {lbl for lbl, _, _, _, _ in cc.REGISTRY}
    oneshot_labels = {lbl for lbl, _ in cc.ONESHOT}
    expected = registry_labels | oneshot_labels
    legend_labels = [lbl for lbl, _ in cc.LEGEND_CELLS]
    assert len(legend_labels) == len(set(legend_labels)), "duplicate LEGEND_CELLS label"
    assert set(legend_labels) == expected, set(legend_labels) ^ expected
