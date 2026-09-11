"""Regression tests for the defects found in the 2026-09 review pass."""
from __future__ import annotations

import json
import threading

import pytest

from agent_search.errors import SetupError
from agent_search.corpus.docstore import JsonlDocStore, LazyUnits
from agent_search.corpus.units import units_from_documents
from tests.lucene_support import require_jvm

require_jvm()

DOCS = [{"_id": str(i), "title": f"Doc {i}", "text": f"Doc {i}\nbody of document number {i} " + "w " * 20}
        for i in range(1, 201)]


def _store(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps({"docid": d["_id"], "text": d["text"]}) for d in DOCS) + "\n")
    return JsonlDocStore(str(p))


# --- docstore -----------------------------------------------------------------------------

def test_docstore_reads_are_thread_safe(tmp_path):
    store = _store(tmp_path)
    errors = []

    def worker(seed):
        import random
        rng = random.Random(seed)
        for _ in range(300):
            d = str(rng.randint(1, 200))
            u = store.unit(d)
            if u is None or u.doc_id != d or f"number {d} " not in (u.body or ""):
                errors.append((d, None if u is None else u.doc_id))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


class _NeverIterate(LazyUnits):
    def __iter__(self):
        raise AssertionError("a workspace must not materialise a lazy corpus")


def test_workspaces_do_not_materialise_a_lazy_corpus(tmp_path):
    from agent_search.tools.base import EpisodeState
    from agent_search.tools.get_document.tool import GetDocument
    from agent_search.tools.search_bm25.tool import SearchBm25
    from agent_search.tools.search_dedup.tool import SearchDedup
    from agent_search.tools.visit.tool import Visit
    units = _NeverIterate(_store(tmp_path))

    class Engine:
        def search(self, q, k=5):
            return ["3", "7"]

    state = EpisodeState(question="x")
    search = SearchBm25(name="bm25_search").bind(state, units, units.by_id, {"bm25": Engine()})
    visit = Visit(name="visit").bind(state, units, units.by_id, {})
    assert "Doc 3" in search.run({"query": "x"})

    class PoolEngine:
        def search(self, q, k=5):
            return ["3"]

    dd_state = EpisodeState(question="x")
    dedup = SearchDedup(ranking="bm25").bind(dd_state, units, units.by_id, {"bm25": PoolEngine()})
    assert "DocID:3" in dedup.run({"query": "x"})


def test_in_memory_engines_refuse_a_lazy_corpus(tmp_path):
    """The two in-memory engines left (the grep ranker and the code Boolean executor) must
    not read an on-disk document store into memory."""
    from agent_search.retrievers.lexical.grep import GrepBaseline
    from agent_search.retrievers.bql.executor import StructuralExecutor
    units = LazyUnits(_store(tmp_path))
    for build in (lambda: GrepBaseline().index(units), lambda: StructuralExecutor(units)):
        with pytest.raises(SetupError, match="on-disk document store"):
            build()


# --- tools ------------------------------------------------------------------------------

def test_float_rank_gives_a_clean_error_not_a_crash():
    from agent_search.tools.base import EpisodeState, ToolBox
    from agent_search.tools.search_bm25.tool import SearchBm25
    from agent_search.tools.visit.tool import Visit

    class Engine:
        def search(self, q, k=5):
            return ["1", "2"]

    units = units_from_documents(DOCS[:5])
    ubyid = {u.doc_id: u for u in units}
    state = EpisodeState(question="x")
    search = SearchBm25(name="bm25_search").bind(state, units, ubyid, {"bm25": Engine()})
    visit = Visit(name="visit").bind(state, units, ubyid, {})
    ws = ToolBox([search, visit], state)
    ws.run("bm25_search", {"query": "x"})
    out = ws.run("visit", {"rank": 1.0})
    assert "AttributeError" not in out and "Doc 1" in out


# --- retrievers ---------------------------------------------------------------------------

def test_unknown_pooling_is_rejected(tmp_path, monkeypatch):
    """The check runs before any weights are loaded, so a weightless checkpoint dir is enough."""
    pytest.importorskip("sentence_transformers")
    from agent_search.retrievers.dense import DenseRetriever
    ck = tmp_path / "ckpt"; ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    monkeypatch.setenv("DENSE_POOLING", "mena")
    with pytest.raises(ValueError, match="unknown pooling 'mena'"):
        DenseRetriever(str(ck), max_seq_length=64)._shared_encoder(None)
    assert DenseRetriever(str(ck), encoder=object()).resolve_pooling() == "mena"

def test_fingerprint_sees_metadata_changes():
    from agent_search.corpus.fingerprint import corpus_fingerprint
    a = units_from_documents([{"_id": "1", "title": "T", "text": "body", "author": "A. Person"}])
    b = units_from_documents([{"_id": "1", "title": "T", "text": "body", "author": "B. Person"}])
    assert corpus_fingerprint(a) != corpus_fingerprint(b)


def test_external_index_built_with_another_model_is_refused(tmp_path, monkeypatch):
    import numpy as np
    from agent_search.retrievers.dense import DenseRetriever
    from agent_search.retrievers.dense.vector_index import build_index, save_index
    emb = np.eye(3, dtype="float32")
    idx = build_index(emb, ["1", "2", "3"])
    d = tmp_path / "cache"; d.mkdir()
    save_index(idx, str(d), extra_meta={"dense_model": "other/model"})
    monkeypatch.setenv("DENSE_INDEX_PATH", str(d))
    with pytest.raises(ValueError, match="built with 'other/model'"):
        DenseRetriever("my/model", encoder=object()).index([], key="k")
    DenseRetriever("other/model", encoder=object()).index([], key="k")     # the right model loads


def test_query_length_comes_from_the_serving_note(tmp_path):
    import numpy as np
    from agent_search.retrievers.dense import DenseRetriever
    ck = tmp_path / "ckpt"; ck.mkdir()
    (ck / "skimsearchagent_dense.json").write_text(json.dumps({"query_max_len": 2048, "max_seq_length": 512}))

    class Enc:
        max_seq_length = 512
        seen = []

        def encode(self, texts, **kw):
            Enc.seen.append(self.max_seq_length)
            return np.ones((len(texts), 4), dtype="float32")

    enc = Enc()
    r = DenseRetriever(str(ck), encoder=enc, index_root=str(tmp_path / "idx"), max_seq_length=512)
    r.index(units_from_documents(DOCS[:3]), key="k")
    assert Enc.seen and Enc.seen[-1] == 512              # documents at the document length
    r.search("q", 1)
    assert Enc.seen[-1] == 2048 and enc.max_seq_length == 512   # queries at query_max_len, then restored


# --- evaluation --------------------------------------------------------------------------

def _instances(n):
    from agent_search.evaluation.datasets import Instance
    return [Instance(instance_id=f"q{i}", repo="local/x", base_commit="0" * 40, problem_statement=f"q {i}",
                     patch="", docs=DOCS[:5], gold_doc_ids={"1"}, corpus_id="review_fix_corpus") for i in range(n)]


def test_fail_fast_ignores_errors_on_a_resumed_run_with_prior_successes(tmp_path, monkeypatch):
    from agent_search.evaluation import run_eval
    monkeypatch.delenv("AGENT_SEARCH_MAX_CONSECUTIVE_ERRORS", raising=False)
    rows = tmp_path / "rows.jsonl"
    rows.write_text("\n".join(json.dumps({"instance_id": f"q{i}", "hit@1": 1.0}) for i in range(5)) + "\n")

    class Broken:
        returns_full_set = False

        def index(self, units, key=None):
            return self

        def search(self, q, k):
            raise ConnectionError("Connection error.")

    res = run_eval.evaluate(_instances(9), lambda: Broken(), ks=(1,), results_dir=str(tmp_path))
    assert res["n_errors"] == 4 and res["n"] == 5


def test_run_config_refuses_a_different_setting_in_the_same_directory(tmp_path):
    from agent_search.evaluation import run_eval
    from agent_search.evaluation.config import DatasetArgs, EvaluationArgs, OutputArgs, RetrieverArgs, RunConfig
    out = OutputArgs(results_dir=str(tmp_path / "r"))
    first = RunConfig(dataset=DatasetArgs(name="doc_fixture"), retriever=RetrieverArgs(name="bm25_pyserini", index_root=str(tmp_path / "idx")),
                      output=out)
    run_eval.run_config(first)
    second = RunConfig(dataset=DatasetArgs(name="doc_fixture"), retriever=RetrieverArgs(name="bm25_pyserini", index_root=str(tmp_path / "idx")),
                       evaluation=EvaluationArgs(level="file"), output=out)
    with pytest.raises(SystemExit):
        run_eval.run_config(second)

def test_overrides_are_recorded_in_the_run_record(tmp_path):
    from pathlib import Path
    from agent_search import cli
    from agent_search.evaluation.build_indexes import build
    from agent_search.evaluation.datasets import load_dataset_by_name
    repo = Path(__file__).resolve().parents[1]
    f = repo / "configs" / "smoke_doc_fixture_sieve_bm25.yaml"
    # the doc run opens a prebuilt Lucene structured index under output.index_root
    build(load_dataset_by_name("doc_fixture"), index_root=str(tmp_path / "idx"),
          retriever="search_lucene", progress=False)
    rc = cli.main(["run", str(f), "dataset.limit=1", "name=override-test",
                   f"output.runs_dir={tmp_path / 'runs'}", f"output.index_root={tmp_path / 'idx'}"])
    assert rc == 0
    cfg = json.loads(next((tmp_path / "runs").rglob("config.json")).read_text())
    assert cfg["experiment"]["dataset"]["limit"] == 1 and cfg["experiment"]["name"] == "override-test"
    assert cfg["experiment_overrides"]["dataset.limit"] == "1"
    assert cfg["experiment_sha256"] != cfg["experiment_file_sha256"]


def test_qrels_header_row_is_skipped(tmp_path):
    from agent_search.evaluation import datasets as DS
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"docid": "1", "text": "T\nbody"}) + "\n")
    (tmp_path / "topics.tsv").write_text("q1\tquestion\tanswer\n")
    (tmp_path / "qrels.txt").write_text("query-id\tQ0\tcorpus-id\tscore\nq1 Q0 1 1\n")
    inst = DS._load_topics_qrels(str(tmp_path), "hdr")
    assert len(inst) == 1 and inst[0].gold_doc_ids == {"1"}


# --- training ----------------------------------------------------------------------------

def test_subset_corpus_keeps_weak_negatives():
    from agent_search.training.retriever_eval import _triple_doc_ids
    ids = _triple_doc_ids([{"pos_id": ["p"], "neg_diversity_id": ["d"], "neg_hard_id": ["h"], "neg_weak_id": ["w1", "w2"]}])
    assert ids == ["p", "d", "h", "w1", "w2"]


def test_title_shortener_counts_tokens():
    from agent_search.tokens import count_tokens
    from agent_search.training.queries import _title_of
    long_title = " ".join(f"word{i}" for i in range(40)) + "\nrest"
    short = _title_of(long_title)
    assert count_tokens(short) <= 12 and short.endswith("...")
    assert _title_of("Short title\nbody") == "Short title"


def test_is_patched_requires_every_file(tmp_path):
    from agent_search.training.retriever import _patch_markers, is_patched
    markers = _patch_markers()
    assert len(markers) >= 9
    root = tmp_path / "FlagEmbedding"
    for rel, marker in markers.items():
        f = root / rel; f.parent.mkdir(parents=True, exist_ok=True); f.write_text(marker + "\n")
    assert is_patched(root)
    first = next(iter(markers))
    (root / first).write_text("pristine\n")
    assert not is_patched(root)


def test_note_conditioned_query_is_the_same_at_training_and_inference():
    """The note the model writes after a read must be part of the query the NEXT search encodes,
    at inference exactly as in the triples the builder renders."""
    from types import SimpleNamespace
    from agent_search.training.history import QueryContext
    from agent_search.training.queries import render_query
    texts = {"d1": "Doc one text about treaties.", "d2": "Doc two text about rivers."}
    steps = [
        SimpleNamespace(name="search", args={"query": "s1"}, raw_output="<tool_call>x</tool_call>", hits=["d1", "d2"]),
        SimpleNamespace(name="visit", args={"doc": "d1"}, raw_output="<tool_call>x</tool_call>", hits=[]),
        SimpleNamespace(name="search", args={"query": "s2"},
                        raw_output="<think>The treaty was signed in 1848 and ended the war.</think><tool_call>x</tool_call>", hits=["d2"]),
    ]
    ctx = QueryContext(question="Q?", text_of=lambda d: texts.get(d), style="i4")
    rendered = []
    for st in steps:
        ctx.note(st.raw_output)                       # what the loop does before the tool runs
        if st.name == "search":
            rendered.append(ctx.render(st.args["query"]))
        ctx.observe(st, st.hits)
    expected_s2 = render_query("i4", "Q?", "s2", [{"query": "s1", "visits": [("d1", texts["d1"],
                                "The treaty was signed in 1848 and ended the war.")]}])
    assert rendered[1] == expected_s2
    assert "1848" in rendered[1]
