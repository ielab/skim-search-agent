"""ITER's setup inside the library: the dedup search strategy, answer-only datasets, the on-disk
corpus path with prebuilt indexes, pooling auto-detection, and the retriever-only evaluation."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from agent_search import cli
from agent_search import experiment as X
from agent_search.agent.tools.doc_dedup import DedupSearchWorkspace
from agent_search.corpus.docstore import JsonlDocStore, LazyUnits
from agent_search.corpus.units import units_from_documents
from agent_search.evaluation import datasets as DS

REPO = Path(__file__).resolve().parents[1]

DOCS = [
    {"_id": "1", "title": "Treaty of Guadalupe Hidalgo", "text": "Treaty of Guadalupe Hidalgo\nSigned in 1848, it ended the Mexican-American War."},
    {"_id": "2", "title": "Mexican-American War", "text": "Mexican-American War\nFought 1846 to 1848 between the United States and Mexico."},
    {"_id": "3", "title": "Nicholas Trist", "text": "Nicholas Trist\nThe American diplomat who negotiated the treaty in 1848."},
    {"_id": "4", "title": "Rio Grande", "text": "Rio Grande\nThe river that became the border after the war."},
    {"_id": "5", "title": "Gadsden Purchase", "text": "Gadsden Purchase\nA later 1853 land purchase from Mexico."},
]


def _units():
    return units_from_documents(DOCS)


# --- the dedup workspace ------------------------------------------------------------------

def test_dedup_search_hides_already_seen_and_lists_them():
    order = ["1", "2", "3", "4", "5"]
    ws = DedupSearchWorkspace(_units(), lambda q, k: order[:k], top_k=2, pool_k=4)
    first = ws.search("treaty")
    assert "DocID:1" in first and "DocID:2" in first and "DocID:3" not in first
    assert "Already-seen" not in first
    second = ws.search("treaty again")
    # the top-2 of the pool were surfaced before: hidden, and listed under Already-seen
    assert "DocID:3" in second and "DocID:4" in second
    assert "Already-seen" in second and "DocID:1" in second.split("Already-seen")[1]
    assert ws.surfaced == ["1", "2", "3", "4"]
    assert ws.last_hits == ["3", "4"]
    doc = ws.get_document("DocID:1")
    assert doc.startswith("DocID:1\n[Treaty of Guadalupe Hidalgo]")
    assert "ERROR" in ws.get_document("999")
    assert ws.run("get_document", {"docid": "2"}).startswith("DocID:2")
    assert "unknown tool" in ws.run("visit_all", {})


def test_dedup_conditions_and_strategies_are_registered():
    from agent_search.prompts import load_condition
    from agent_search.strategies import STRATEGIES, DENSE_STRATEGIES
    for cond, tools in (("research_dedup_bm25", ("bm25_search", "get_document")),
                        ("research_dedup_dense", ("search", "get_document"))):
        p = load_condition(cond)
        assert p.tool_names == tools
        assert "Already-seen" in p.system
    assert STRATEGIES["dedup_dense"] == "agent_research_dedup_dense"
    assert "dedup_dense" in DENSE_STRATEGIES and "dedup_bm25" not in DENSE_STRATEGIES


# --- answer-only datasets through the whole harness -----------------------------------------

def _stage_topics(root: Path, with_qrels: bool):
    (root / "corpus.jsonl").write_text("\n".join(json.dumps({"docid": d["_id"], "text": d["text"]}) for d in DOCS) + "\n")
    (root / "topics.tsv").write_text("q1\tWhich treaty ended the Mexican-American War?\tTreaty of Guadalupe Hidalgo\n"
                                     "q2\tWho negotiated it?\tNicholas Trist\n")
    if with_qrels:
        (root / "qrels.txt").write_text("q1 Q0 1 1\nq2 Q0 3 1\n")


def test_topics_loader_answer_only_and_with_qrels(tmp_path):
    a = tmp_path / "a"; a.mkdir(); _stage_topics(a, with_qrels=False)
    inst = DS._load_topics_qrels(str(a), "tiny", corpus_id="tinycorpus")
    assert len(inst) == 2 and inst[0].gold_doc_ids is None and inst[0].answer == "Treaty of Guadalupe Hidalgo"
    assert inst[0].docs is not None and inst[0].docs[0]["title"] == "Treaty of Guadalupe Hidalgo"
    assert inst[0].corpus_id == "tinycorpus"
    b = tmp_path / "b"; b.mkdir(); _stage_topics(b, with_qrels=True)
    inst = DS._load_topics_qrels(str(b), "tiny")
    assert inst[0].gold_doc_ids == {"1"} and inst[1].gold_doc_ids == {"3"}


def test_topics_loader_serves_a_big_corpus_from_disk(tmp_path, monkeypatch):
    a = tmp_path / "a"; a.mkdir(); _stage_topics(a, with_qrels=False)
    monkeypatch.setenv("AGENT_SEARCH_DOCSTORE", "1")
    inst = DS._load_topics_qrels(str(a), "tiny")
    assert inst[0].docs is None and inst[0].docstore is not None
    assert inst[0].docstore.get("3")["title"] == "Nicholas Trist"


def test_answer_only_run_end_to_end_with_dedup_bm25(tmp_path):
    root = tmp_path / "tiny_answer_only"; root.mkdir(); _stage_topics(root, with_qrels=False)
    if "tiny_answer_only" not in DS.available_datasets():
        DS.register_dataset("tiny_answer_only", domain="general")(DS._topics_qrels_loader("tiny_answer_only", root=str(root)))
    data = X.defaults(None, "dedup_bm25")
    data["name"] = "tiny-dedup"
    data["dataset"]["name"] = "tiny_answer_only"
    data["model"]["policy"] = "stub"
    data["output"]["runs_dir"] = str(tmp_path / "runs")
    data["output"]["index_root"] = str(tmp_path / "idx")
    X.validate(data, complete=True)
    f = tmp_path / "exp.yaml"; f.write_text(X.render(data))
    assert cli.main(["run", str(f)]) == 0
    rows = [json.loads(l) for l in next((tmp_path / "runs").rglob("rows.jsonl")).read_text().splitlines()]
    assert len(rows) == 2 and all(r.get("answer_only") for r in rows)
    assert all("recall@1" not in r and "answer_em" in r for r in rows)   # scored on the answer only
    steps = rows[0]["trajectory"]
    assert any(st["hit_ids"] for st in steps) and any(st["read_ids"] for st in steps)   # searched, then opened a DocID
    assert any("docid" in (st.get("args") or {}) for st in steps)
    summary = json.loads(next((tmp_path / "runs").rglob("results.json")).read_text())
    assert summary["n"] == 2 and summary["n_skipped"] == 0


# --- the on-disk corpus path with prebuilt indexes ---------------------------------------------

def _external_faiss(tmp_path, units, dim=8):
    faiss = pytest.importorskip("faiss")
    import numpy as np
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((len(units), dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.add(emb)
    d = tmp_path / "ext"; d.mkdir()
    faiss.write_index(index, str(d / "index.faiss"))
    import pickle
    with open(d / "index.lookup.pkl", "wb") as fh:
        pickle.dump([u.doc_id for u in units], fh)
    return d, emb


class _FakeEncoder:
    """Encodes a query to the stored vector of the doc whose title it names."""
    def __init__(self, units, emb):
        self.by_title = {u.title.lower(): emb[i] for i, u in enumerate(units)}
        self.device = "cpu"

    def encode(self, texts, **kw):
        import numpy as np
        out = []
        for t in texts:
            t = t.lower()
            vec = next((v for title, v in self.by_title.items() if title in t), None)
            out.append(vec if vec is not None else np.zeros_like(next(iter(self.by_title.values()))))
        return np.stack(out)


def test_lazy_corpus_with_external_faiss_index_never_materialises_units(tmp_path, monkeypatch):
    from agent_search.agent.retriever import AgentRetriever
    from agent_search.retrievers.dense.dense import DenseRetriever
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps({"docid": d["_id"], "text": d["text"]}) for d in DOCS) + "\n")
    units = LazyUnits(JsonlDocStore(str(p)))
    assert len(units) == 5 and units.by_id["3"].title == "Nicholas Trist" and "9" not in units.by_id
    ext, emb = _external_faiss(tmp_path, list(units))
    monkeypatch.setenv("DENSE_INDEX_PATH", str(ext))
    r = DenseRetriever("fake/model", encoder=_FakeEncoder(list(units), emb), index_root=str(tmp_path / "idx"))
    monkeypatch.setattr("agent_search.retrievers.dense.dense.query_prefix_for", lambda m: "")
    assert r.is_cached("anything")
    r.index(units, key="wiki")                     # opens the external index; no encoding, no cache dir
    assert not (tmp_path / "idx").exists()
    assert r.search("Rio Grande", 1) == ["4"]
    # the agent retriever keeps the corpus lazy and looks units up by id
    ar = AgentRetriever.__new__(AgentRetriever)
    ar._units, ar._ubyid = [], {}
    AgentRetriever.index  # noqa: B018 — sanity that the method exists
    ar.dense_model, ar.index_root, ar.rebuild = "fake/model", str(tmp_path / "idx"), False
    ar.toolset = ("search", "get_document"); ar._bql = ar._bm25 = ar._indri = ar._dense_belief = None
    ar._files = {}; ar._key = None
    import threading; ar._tl = threading.local()
    ar.index(units, key="wiki")
    assert ar._units is units and ar._ubyid is units.by_id
    assert ar._doc_text("2").startswith("Mexican-American War")


def test_in_memory_engines_refuse_a_lazy_corpus(tmp_path, monkeypatch):
    from agent_search.core.errors import SetupError
    from agent_search.retrievers.dense.dense import DenseRetriever
    from agent_search.retrievers.lexical import build_bm25_engine
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps({"docid": "1", "text": "a\nb"}) + "\n")
    units = LazyUnits(JsonlDocStore(str(p)))
    monkeypatch.delenv("DENSE_INDEX_PATH", raising=False)
    monkeypatch.delenv("BM25_BACKEND", raising=False)
    with pytest.raises(SetupError, match="BM25_INDEX_PATH"):
        build_bm25_engine(units, key="x")
    with pytest.raises(SetupError, match="DENSE_INDEX_PATH"):
        DenseRetriever("fake/model", encoder=object(), index_root=str(tmp_path)).index(units, key="x")


def test_pooling_resolution_for_released_checkpoints(tmp_path, monkeypatch):
    from agent_search.retrievers.dense.dense import resolve_pooling
    ck = tmp_path / "ckpt"; ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 1024}))
    monkeypatch.delenv("DENSE_POOLING", raising=False)
    assert resolve_pooling(str(ck)) == "last_token"           # a Qwen3-Embedding-style checkpoint
    (ck / "skimsearchagent_dense.json").write_text(json.dumps({"pooling": "mean"}))
    assert resolve_pooling(str(ck)) == "mean"                 # the serving note wins over the guess
    monkeypatch.setenv("DENSE_POOLING", "cls")
    assert resolve_pooling(str(ck)) == "cls"                  # the knob wins over everything
    assert resolve_pooling("BAAI/bge-base-en-v1.5") == "cls"
    monkeypatch.delenv("DENSE_POOLING")
    assert resolve_pooling("BAAI/bge-base-en-v1.5") is None   # hub models: sentence-transformers decides


# --- retriever-only evaluation --------------------------------------------------------------

def test_retriever_eval_recall_and_novelty():
    from agent_search.training.retriever_eval import evaluate
    triples = [
        {"query": "q1", "pos_id": ["3"], "neg_diversity_id": ["1"], "neg_hard_id": ["2"]},
        {"query": "q2", "pos_id": ["5"], "neg_diversity_id": [], "neg_hard_id": []},
        {"query": "no-pos", "pos_id": []},
    ]
    ranked = {"q1": ["1", "3", "2", "4"], "q2": ["4", "2", "1", "3"]}
    res = evaluate(triples, lambda q, k: ranked[q][:k], ks=(1, 2))
    assert res["n"] == 2
    assert res["recall@1"] == 0.0 and res["recall@2"] == 0.5
    assert res["novelty@2"] == pytest.approx((0.5 + 1.0) / 2)


def test_retriever_eval_subset_corpus(tmp_path):
    from agent_search.training.retriever_eval import corpus_units
    root = tmp_path / "tiny_eval_corpus"; root.mkdir(); _stage_topics(root, with_qrels=True)
    if "tiny_eval_corpus" not in DS.available_datasets():
        DS.register_dataset("tiny_eval_corpus", domain="general")(DS._topics_qrels_loader("tiny_eval_corpus", root=str(root)))
    triples = [{"query": "q", "pos_id": ["5"], "neg_diversity_id": ["4"]}]
    sub = corpus_units("tiny_eval_corpus", triples, subset=2)
    assert [u.doc_id for u in sub] == ["5", "4", "1", "2"]


def test_docstore_corpus_runs_through_the_harness_with_a_prebuilt_lucene_index(tmp_path, monkeypatch):
    """The whole on-disk path: a corpus forced onto the docstore, a prebuilt Lucene index named by
    BM25_INDEX_PATH, the dedup_bm25 strategy with the stub policy, answer-only scoring."""
    pytest.importorskip("pyserini")
    from agent_search.retrievers.lexical.pyserini import BM25Pyserini
    root = tmp_path / "tiny_disk"; root.mkdir(); _stage_topics(root, with_qrels=False)
    # build a Lucene index once from the in-memory units, then serve it as an external index
    try:
        BM25Pyserini(index_root=str(tmp_path / "build")).index(_units(), key="tiny")
    except Exception as e:  # noqa: BLE001 — no usable JVM on this machine
        pytest.skip(f"pyserini indexing unavailable: {e}")
    lucene = tmp_path / "build" / "bm25_pyserini" / "tiny" / "lucene"
    if "tiny_disk" not in DS.available_datasets():
        DS.register_dataset("tiny_disk", domain="general")(DS._topics_qrels_loader("tiny_disk", root=str(root)))
    monkeypatch.setenv("AGENT_SEARCH_DOCSTORE", "1")
    # the launcher exports the file's knobs into this process; register them with monkeypatch so
    # they are removed again after the test (BM25_INDEX_PATH must not leak into other tests)
    monkeypatch.setenv("BM25_INDEX_PATH", str(lucene))
    monkeypatch.setenv("BM25_BACKEND", "pyserini")
    data = X.defaults(None, "dedup_bm25")
    data["name"] = "tiny-disk"
    data["dataset"]["name"] = "tiny_disk"
    data["model"]["policy"] = "stub"
    data["retrieval"]["bm25_backend"] = "pyserini"
    data["retrieval"]["bm25_index"] = str(lucene)
    data["output"]["runs_dir"] = str(tmp_path / "runs")
    data["output"]["index_root"] = str(tmp_path / "idx")
    X.validate(data, complete=True)
    f = tmp_path / "exp.yaml"; f.write_text(X.render(data))
    assert cli.main(["run", str(f)]) == 0
    rows = [json.loads(l) for l in next((tmp_path / "runs").rglob("rows.jsonl")).read_text().splitlines()]
    assert len(rows) == 2 and all(r.get("answer_only") for r in rows)
    steps = rows[0]["trajectory"]
    assert any(st["hit_ids"] for st in steps) and any(st["read_ids"] for st in steps)
    assert rows[0]["retrieved"][0]["title"]                      # rendered from the docstore, by id
    cfg = json.loads(next((tmp_path / "runs").rglob("config.json")).read_text())
    assert cfg["env_knobs"]["BM25_INDEX_PATH"] == str(lucene)


def test_hub_ids_resolve_to_their_cached_snapshot_for_pooling(tmp_path, monkeypatch):
    from agent_search.retrievers.dense.dense import _local_snapshot, resolve_pooling
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("DENSE_POOLING", raising=False)
    assert _local_snapshot(str(tmp_path)) == str(tmp_path)
    assert _local_snapshot("nobody/definitely-not-a-cached-model") is None
    snap = _local_snapshot("Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B")
    if snap is None:
        pytest.skip("LRAT checkpoint not in the local HF cache")
    # a released decoder checkpoint without a sentence-transformers config: last-token pooling
    assert not os.path.exists(os.path.join(snap, "modules.json"))
    assert resolve_pooling("Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B") == "last_token"


def test_concurrent_episodes_build_a_shared_engine_once(monkeypatch):
    """`--workers N`: N episodes asking for the lazily built engine at the same moment must not
    each build (load) it; the first builds, the others wait and reuse it."""
    import threading
    import time
    from agent_search.agent import retriever as R

    calls = []

    def slow_build(units, index_root, rebuild, key):
        calls.append(key)
        time.sleep(0.2)
        return object()

    monkeypatch.setattr(R, "_build_bm25_engine", slow_build)

    class Holder:
        _bm25 = None

    holder = Holder()
    ctx = R.WorkspaceContext(units=_units(), ubyid={}, corpus_key="k", index_root="idx", rebuild=False,
                             query="", domain="general", retriever=holder)
    got = []
    threads = [threading.Thread(target=lambda: got.append(ctx.bm25())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(got) == 8 and all(g is got[0] for g in got)


def test_get_document_cap_is_the_configured_visit_budget():
    """ITER capped get_document at 512 tokens. Here the cap is `budgets.max_visit_tokens`
    (env MAX_VISIT_TOKENS): nothing is fixed, the ITER files set 512, the library default is 12000."""
    import subprocess
    import sys
    code = (
        "from agent_search.agent.tools.doc_dedup import DedupSearchWorkspace\n"
        "from agent_search.corpus.units import units_from_documents\n"
        "u = units_from_documents([{'_id': '1', 'title': 'T', 'text': ' '.join(f'w{i}' for i in range(400))}])\n"
        "ws = DedupSearchWorkspace(u, lambda q, k: ['1'])\n"
        "out = ws.get_document('1'); print(len(out.split()))\n")
    env = dict(os.environ, MAX_VISIT_TOKENS="20")
    n_capped = int(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip())
    env = dict(os.environ, MAX_VISIT_TOKENS="12000")
    n_full = int(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout.strip())
    assert n_capped < 40 < 400 <= n_full
    import yaml
    for f in (REPO / "configs" / "iter").glob("*.yaml"):
        d = yaml.safe_load(f.read_text())
        if "strategy" in d:
            assert d["budgets"]["max_visit_tokens"] == 512, f.name


def test_triple_builder_reads_documents_from_the_docstore(tmp_path, monkeypatch):
    from agent_search.training.build_triples import corpus_text_lookup
    root = tmp_path / "tiny_triples"; root.mkdir(); _stage_topics(root, with_qrels=False)
    if "tiny_triples" not in DS.available_datasets():
        DS.register_dataset("tiny_triples", domain="general")(DS._topics_qrels_loader("tiny_triples", root=str(root)))
    monkeypatch.setenv("AGENT_SEARCH_DOCSTORE", "1")
    text_of = corpus_text_lookup("tiny_triples")
    assert text_of("3").startswith("Nicholas Trist") and text_of("999") is None


def test_sample_dataset_and_directory_discovery(tmp_path, monkeypatch):
    from agent_search.evaluation.sample import sample_dataset
    src = tmp_path / "src"; src.mkdir(); _stage_topics(src, with_qrels=True)
    if "tiny_sample_src" not in DS.available_datasets():
        DS.register_dataset("tiny_sample_src", domain="general")(DS._topics_qrels_loader("tiny_sample_src", root=str(src)))
    data_dir = tmp_path / "data"; data_dir.mkdir()
    monkeypatch.setattr(DS, "DATA_DIR", str(data_dir))
    meta = sample_dataset("tiny_sample_src", str(data_dir / "tiny_cut"), n_topics=1, n_docs=1, seed=0,
                          pool_search=lambda q, k: ["4"], pool_k=1)
    assert meta["n_topics"] == 1 and meta["qrels"] is True
    ids = {json.loads(l)["docid"] for l in (data_dir / "tiny_cut" / "corpus.jsonl").read_text().splitlines()}
    assert "4" in ids and len(ids) >= 2                      # the pool doc, the gold doc, a random doc
    assert (data_dir / "tiny_cut" / "qrels.txt").read_text().strip()
    # a folder under data/ with topics.tsv + corpus.jsonl is a dataset without any code
    assert "tiny_cut" in DS.available_datasets()
    inst = DS.load_dataset_by_name("tiny_cut")
    assert len(inst) == 1 and inst[0].gold_doc_ids and inst[0].answer
