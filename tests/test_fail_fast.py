"""A run that cannot work (every instance errors before any succeeds) stops early with a
SetupError instead of burning the retry budget on every remaining question."""
from __future__ import annotations

import pytest

from agent_search.errors import SetupError
from agent_search.evaluation import run_eval
from agent_search.evaluation.datasets import Instance

DOCS = [{"_id": str(i), "title": f"doc {i}", "text": f"document number {i}"} for i in range(1, 6)]


class _Broken:
    returns_full_set = False

    def index(self, units, key=None):
        return self

    def search(self, query, k):
        raise ConnectionError("Connection error.")


def _instances(n):
    return [Instance(instance_id=f"q{i}", repo="local/x", base_commit="0" * 40, problem_statement=f"q {i}",
                     patch="", docs=DOCS, gold_doc_ids={"1"}, corpus_id="fail_fast_corpus") for i in range(n)]


def test_three_consecutive_errors_abort_the_run(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS", raising=False)
    seen = []
    factory = lambda: _Broken()  # noqa: E731
    with pytest.raises(SetupError, match="3 consecutive errors"):
        run_eval.evaluate(_instances(10), factory, ks=(1,), results_dir=str(tmp_path))
    assert (tmp_path / "rows.jsonl").read_text() == ""       # nothing was scored


def test_the_check_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS", "0")
    res = run_eval.evaluate(_instances(4), lambda: _Broken(), ks=(1,), results_dir=str(tmp_path))
    assert res["n_errors"] == 4 and res["n"] == 0


def test_errors_after_a_success_do_not_abort(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS", raising=False)

    class Flaky(_Broken):
        calls = 0

        def search(self, query, k):
            Flaky.calls += 1
            if Flaky.calls == 1:
                return ["1", "2"]
            raise ConnectionError("Connection error.")

    res = run_eval.evaluate(_instances(6), lambda: Flaky(), ks=(1,), results_dir=str(tmp_path))
    assert res["n"] == 1 and res["n_errors"] == 5
