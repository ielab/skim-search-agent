"""Offline tests for scripts/judge_cells.py (no API key, no network): the judge generator is
monkeypatched, so these prove control flow (EM short-circuit skips it, cache resume skips
already-judged pairs, a changed answer re-judges) rather than judge quality."""
import json

import pytest

import scripts.judge_cells as jc
from scripts.judge_cells import (
    CACHE_NAME,
    classify_row,
    dataset_selected,
    do_live_work,
    load_cache,
    run_cell,
    selected_cells,
    summarize,
)


def _row(iid, question, gold, answer):
    return {"instance_id": iid, "question": question, "gold_answer": gold, "final_answer": answer}


def _write_rows_jsonl(cond_dir, rows):
    path = cond_dir / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


@pytest.fixture
def cond_dir(tmp_path):
    d = tmp_path / "agent_research_fake"
    d.mkdir()
    return d


def _counting_generate(calls, response_json='{"extracted_final_answer": "no match", '
                                            '"reasoning": "differs", "correct": "no", "confidence": 100}'):
    """A fake judge `generate(prompt) -> str` that records every call it receives."""
    def gen(prompt):
        calls.append(prompt)
        return response_json
    return gen


# --- --datasets prefix filter --------------------------------------------------------------------

@pytest.mark.parametrize("ds,prefixes,expected", [
    ("browsecomp_plus_structured", ["browsecomp"], True),
    ("browsecomp_plus_flat", ["browsecomp"], True),
    ("hotpotqa_structured", ["browsecomp"], False),          # wiki skipped under the default
    ("musique_structured", ["browsecomp"], False),
    ("hotpotqa_structured", ["all"], True),                  # "all" selects everything
    ("hotpotqa_structured", ["hotpotqa"], True),             # explicit prefix
    ("musique_structured", ["browsecomp", "musique"], True), # multi-prefix
])
def test_dataset_selected(ds, prefixes, expected):
    assert dataset_selected(ds, prefixes) is expected


def test_wiki_registry_cell_skipped_by_default_included_under_all(monkeypatch):
    fake_registry = [
        ("fake bc cell", "_fake", "browsecomp_plus_structured", "agent_research_fake", True),
        ("fake wiki cell", "_fake", "hotpotqa_structured", "agent_research_fake", True),
    ]
    monkeypatch.setattr(jc, "REGISTRY", fake_registry)
    default_labels = [c[0] for c in selected_cells(["browsecomp"])]
    assert "fake bc cell" in default_labels
    assert "fake wiki cell" not in default_labels            # wiki skipped under the default
    assert "one-shot bm25" in default_labels                 # ONESHOT rides along with browsecomp
    all_labels = [c[0] for c in selected_cells(["all"])]
    assert "fake wiki cell" in all_labels                    # included under --datasets all
    assert "fake bc cell" in all_labels
    wiki_only = [c[0] for c in selected_cells(["hotpotqa"])]
    assert wiki_only == ["fake wiki cell"]                   # and ONESHOT does NOT ride along here


# --- EM short-circuit never calls the generator ----------------------------------------------

def test_em_shortcircuit_skips_the_generator(cond_dir):
    rows = [_row("q1", "What is 2+2?", "four", "Four")]  # answer_em normalizes case/punct -> match
    calls = []
    gen = _counting_generate(calls)
    n_calls = do_live_work(cond_dir, rows, {}, workers=2, limit=None, generate=gen)
    assert n_calls == 0
    assert calls == []                      # the generator was NEVER invoked
    cache = load_cache(cond_dir)
    rec = next(iter(cache.values()))
    assert rec["method"] == "em_shortcircuit"
    assert rec["judge_correct"] is True


def test_empty_answer_skips_the_generator(cond_dir):
    rows = [_row("q1", "What is 2+2?", "four", "   ")]
    calls = []
    gen = _counting_generate(calls)
    n_calls = do_live_work(cond_dir, rows, {}, workers=2, limit=None, generate=gen)
    assert n_calls == 0
    assert calls == []
    cache = load_cache(cond_dir)
    rec = next(iter(cache.values()))
    assert rec["method"] == "empty" and rec["judge_correct"] is False


def test_non_em_row_does_call_the_generator(cond_dir):
    rows = [_row("q1", "Who wrote Hamlet?", "William Shakespeare", "Shakespeare wrote it.")]
    calls = []
    gen = _counting_generate(calls)
    n_calls = do_live_work(cond_dir, rows, {}, workers=2, limit=None, generate=gen)
    assert n_calls == 1
    assert len(calls) == 1
    cache = load_cache(cond_dir)
    rec = next(iter(cache.values()))
    assert rec["method"] == "llm"


# --- cache resume: idempotent, skips already-judged (instance_id, answer_sha1) pairs -----------

def test_cache_resume_skips_already_judged_pairs(cond_dir):
    rows = [_row("q1", "Who wrote Hamlet?", "William Shakespeare", "Shakespeare wrote it.")]
    calls = []
    gen = _counting_generate(calls)
    n1 = do_live_work(cond_dir, rows, load_cache(cond_dir), workers=2, limit=None, generate=gen)
    assert n1 == 1
    # re-run over the SAME rows, same cond_dir: nothing new to judge, generator not called again.
    n2 = do_live_work(cond_dir, rows, load_cache(cond_dir), workers=2, limit=None, generate=gen)
    assert n2 == 0
    assert len(calls) == 1                  # still only the one real call, ever
    cache_lines = (cond_dir / CACHE_NAME).read_text().strip().splitlines()
    assert len(cache_lines) == 1             # append-only, no duplicate record written


# --- a changed answer (different sha1) is a cache MISS and gets re-judged ----------------------

def test_changed_answer_is_a_new_cache_key_and_gets_rejudged(cond_dir):
    rows_v1 = [_row("q1", "Who wrote Hamlet?", "William Shakespeare", "Shakespeare wrote it.")]
    calls = []
    gen = _counting_generate(calls)
    do_live_work(cond_dir, rows_v1, load_cache(cond_dir), workers=2, limit=None, generate=gen)
    assert len(calls) == 1

    # simulate a recovery backfill changing the answer for the SAME instance_id
    rows_v2 = [_row("q1", "Who wrote Hamlet?", "William Shakespeare", "It was Christopher Marlowe.")]
    n2 = do_live_work(cond_dir, rows_v2, load_cache(cond_dir), workers=2, limit=None, generate=gen)
    assert n2 == 1                           # new sha1 -> cache miss -> re-judged
    assert len(calls) == 2

    cache = load_cache(cond_dir)
    assert len(cache) == 2                   # two distinct (instance_id, answer_sha1) entries, both kept
    cache_lines = (cond_dir / CACHE_NAME).read_text().strip().splitlines()
    assert len(cache_lines) == 2


def test_limit_caps_new_api_calls_but_not_free_verdicts(cond_dir):
    rows = [
        _row("q1", "2+2?", "four", "four"),                       # em_shortcircuit, free
        _row("q2", "Who wrote Hamlet?", "William Shakespeare", "Marlowe."),   # llm
        _row("q3", "Who wrote Macbeth?", "William Shakespeare", "Marlowe."),  # llm
    ]
    calls = []
    gen = _counting_generate(calls)
    n_calls = do_live_work(cond_dir, rows, load_cache(cond_dir), workers=2, limit=1, generate=gen)
    assert n_calls == 1                      # capped
    assert len(calls) == 1
    summary = summarize(rows, load_cache(cond_dir))
    assert summary["em_sc"] == 1
    assert summary["llm"] == 1
    assert summary["pending"] == 1           # the second llm-eligible row is still pending


# --- classify_row / summarize are pure (no writes) ----------------------------------------------

def test_classify_row_dry_run_never_writes(cond_dir):
    rows = [_row("q1", "2+2?", "four", "four")]
    rec = classify_row(rows[0], {})
    assert rec["method"] == "em_shortcircuit"
    assert not (cond_dir / CACHE_NAME).exists()   # classify_row alone never persists


def test_run_cell_dry_run_makes_no_generator_calls_and_no_cache_file(cond_dir):
    rows = [
        _row("q1", "2+2?", "four", "four"),
        _row("q2", "Who wrote Hamlet?", "William Shakespeare", "Marlowe."),
    ]
    summary = run_cell("fake cell", cond_dir, rows, workers=2, limit=None, dry_run=True, generate=None)
    assert summary["em_sc"] == 1
    assert summary["pending"] == 1          # the llm-eligible row is NOT judged in dry-run
    assert summary["llm"] == 0
    assert not (cond_dir / CACHE_NAME).exists()


# --- rows.jsonl is never touched -----------------------------------------------------------------

def test_rows_jsonl_is_byte_identical_before_and_after(cond_dir):
    rows = [
        _row("q1", "2+2?", "four", "four"),
        _row("q2", "Who wrote Hamlet?", "William Shakespeare", "Marlowe."),
    ]
    rows_path = _write_rows_jsonl(cond_dir, rows)
    before = rows_path.read_bytes()

    calls = []
    gen = _counting_generate(calls)
    run_cell("fake cell", cond_dir, rows, workers=2, limit=None, dry_run=False, generate=gen)
    # a second pass (idempotent resume) too
    run_cell("fake cell", cond_dir, rows, workers=2, limit=None, dry_run=False, generate=gen)

    after = rows_path.read_bytes()
    assert before == after
    assert len(calls) == 1                  # only the one non-EM row, only once (resume worked)
