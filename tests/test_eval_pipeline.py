"""End-to-end smoke: the eval pipeline runs on the fixture and localizes the gold
function with the lexical baseline (no heavy deps)."""
from evaluation.datasets import Instance, fixture_instances
from agent_search.retrievers.lexical.bm25 import BM25Local
from evaluation.run_eval import _load_rows, evaluate


def test_pipeline_runs_and_finds_gold_function():
    res = evaluate(fixture_instances(), lambda: BM25Local(), ks=[1, 5, 10],
                   level="function")
    assert res["n"] == 1
    assert res["metrics"]["recall@10"] == 1.0   # gold function retrieved
    assert res["metrics"]["acc@10"] == 1.0       # and within top-10


def test_pipeline_file_level_runs():
    res = evaluate(fixture_instances(), lambda: BM25Local(), ks=[1, 5], level="file")
    assert res["n"] == 1
    assert res["metrics"]["recall@5"] == 1.0


def test_code_fix_agent_stub_pipeline_runs_without_model():
    from evaluation.run_eval import _make_factory

    # the code-fix arm drives search -> fetch -> <fix> with no model; the run completes
    # with 0 errors and records fix_file_ok (its value depends on the stub's fix guess).
    res = evaluate(fixture_instances(), _make_factory("agent_codefix", policy="stub"),
                   ks=[1, 10], level="function")

    assert res["n"] == 1
    assert res["n_errors"] == 0
    assert "fix_file_ok" in res["rows"][0]


def test_workers_concurrent_matches_sequential():
    # the --workers thread-pool path must produce identical metrics to sequential
    seq = evaluate(fixture_instances(), lambda: BM25Local(), ks=[1, 10],
                   level="function", workers=1)
    conc = evaluate(fixture_instances(), lambda: BM25Local(), ks=[1, 10],
                    level="function", workers=4)
    assert seq["n"] == conc["n"] and seq["metrics"] == conc["metrics"]


def test_results_persist_and_resume(tmp_path):
    import json
    d = str(tmp_path / "run")
    r1 = evaluate(fixture_instances(), lambda: BM25Local(), ks=[1, 10],
                  level="function", results_dir=d)
    assert (tmp_path / "run" / "rows.jsonl").exists()
    assert (tmp_path / "run" / "results.json").exists()
    assert r1["n"] == 1

    # second run resumes: same metrics, and rows.jsonl is NOT duplicated
    r2 = evaluate(fixture_instances(), lambda: BM25Local(), ks=[1, 10],
                  level="function", results_dir=d)
    assert r2["metrics"] == r1["metrics"]
    lines = [l for l in (tmp_path / "run" / "rows.jsonl").read_text().splitlines() if l]
    assert len(lines) == 1                       # resumed, not re-appended
    assert json.loads(lines[0])["instance_id"] == "fixture__session-expiry-1"


def test_resume_rows_deduplicate_instance_ids(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        '{"instance_id":"x","recall@1":1.0}\n'
        '{"instance_id":"x","recall@1":0.0}\n'
        '{"instance_id":"y","recall@1":1.0}\n'
    )

    loaded, done = _load_rows(str(rows))

    assert [r["instance_id"] for r in loaded] == ["x", "y"]
    assert done == {"x", "y"}


def test_shared_fixed_corpus_reuses_indexed_retriever():
    class CountingRetriever:
        index_calls = 0

        def __init__(self):
            self.doc_ids = []

        def index(self, units, key=None):
            type(self).index_calls += 1
            self.doc_ids = [u.doc_id for u in units]
            return self

        def search(self, query, k):
            return self.doc_ids[:k]

    docs = [{"doc_id": "d1", "title": "Alpha", "text": "shared document"}]
    instances = [
        Instance("q1", "local/shared", "0" * 40, "alpha", "", docs=docs,
                 gold_doc_ids={"d1"}, corpus_id="shared"),
        Instance("q2", "local/shared", "0" * 40, "alpha again", "", docs=docs,
                 gold_doc_ids={"d1"}, corpus_id="shared"),
    ]

    res = evaluate(instances, lambda: CountingRetriever(), ks=[1], level="function",
                   reuse_indexed_retriever=True)

    assert res["n"] == 2
    assert CountingRetriever.index_calls == 1
