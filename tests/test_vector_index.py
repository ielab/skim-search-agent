"""Pluggable vector index: flat (numpy, default) + optional FAISS hnsw / ivfpq.

The flat backend is exercised fully (no dependency). The FAISS backends run only
where faiss is installed (cluster); everywhere else they're skipped. The point of
the layer is that a much larger corpus is a backend swap, not a rewrite — and that a
missing faiss degrades to flat instead of crashing.
"""
import json
import os

import numpy as np
import pytest

from agent_search.retrievers.dense import vector_index as vi


def _corpus(n=200, d=32, seed=0):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, d)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)          # normalize = cosine
    ids = [f"d{i}" for i in range(n)]
    return emb, ids


def _brute_topk(emb, qv, k):
    return [int(i) for i in np.argsort(-(emb @ qv))[:k]]


# --- backend selection ------------------------------------------------------

def test_choose_backend_defaults_to_flat_without_faiss(monkeypatch):
    monkeypatch.delenv("AGENT_SEARCH_ANN", raising=False)
    monkeypatch.setattr(vi, "_faiss", lambda: None)
    assert vi.choose_backend(10_000_000) == "flat"             # no faiss -> always flat


def test_choose_backend_env_override(monkeypatch):
    monkeypatch.setenv("AGENT_SEARCH_ANN", "hnsw")
    assert vi.choose_backend(10) == "hnsw"
    monkeypatch.setenv("AGENT_SEARCH_ANN", "flat")
    assert vi.choose_backend(10**9) == "flat"


def test_choose_backend_auto_scales_when_faiss_present(monkeypatch):
    monkeypatch.delenv("AGENT_SEARCH_ANN", raising=False)
    monkeypatch.setattr(vi, "_faiss", lambda: object())        # pretend faiss exists
    assert vi.choose_backend(100_000) == "flat"                # small -> flat
    assert vi.choose_backend(2_000_000) == "hnsw"              # large -> hnsw
    assert vi.choose_backend(20_000_000) == "ivfpq"            # huge -> compressed


# --- flat backend: correctness, persistence, float16 storage ----------------

def test_flat_search_matches_brute_force():
    emb, ids = _corpus()
    idx = vi.FlatIndex.build(emb, ids)
    qv = emb[7]                                                 # query = a known doc
    assert idx.search(qv, 1)[0] == "d7"
    got = idx.search(qv, 5)
    assert got == [ids[i] for i in _brute_topk(emb, qv, 5)]


def test_flat_stores_float16():
    emb, ids = _corpus()
    idx = vi.FlatIndex.build(emb, ids)
    assert idx.emb.dtype == np.float16                         # halved storage


def test_build_index_defaults_to_flat(monkeypatch):
    monkeypatch.setattr(vi, "_faiss", lambda: None)
    emb, ids = _corpus()
    idx = vi.build_index(emb, ids)
    assert idx.backend == "flat"


def test_save_and_load_round_trip(tmp_path):
    emb, ids = _corpus()
    idx = vi.build_index(emb, ids, backend="flat")
    vi.save_index(idx, str(tmp_path))
    assert os.path.exists(tmp_path / "meta.json")
    assert json.load(open(tmp_path / "meta.json"))["backend"] == "flat"
    loaded = vi.load_index(str(tmp_path))
    assert loaded is not None and loaded.backend == "flat"
    qv = emb[3]
    assert loaded.search(qv, 3) == idx.search(qv, 3)


def test_load_legacy_cache_without_meta(tmp_path):
    """A pre-backends cache is a bare float32 embeddings.npy + doc_ids.json."""
    emb, ids = _corpus()
    np.save(tmp_path / "embeddings.npy", emb)                   # float32, no meta.json
    json.dump(ids, open(tmp_path / "doc_ids.json", "w"))
    loaded = vi.load_index(str(tmp_path))
    assert loaded is not None and loaded.backend == "flat"
    assert loaded.search(emb[1], 1) == ["d1"]


def test_load_missing_returns_none(tmp_path):
    assert vi.load_index(str(tmp_path)) is None


def test_ann_payload_without_faiss_triggers_rebuild(tmp_path, monkeypatch):
    """meta says hnsw but faiss is absent -> load returns None (caller re-encodes
    flat) rather than crashing."""
    json.dump(["d0"], open(tmp_path / "doc_ids.json", "w"))
    json.dump({"backend": "hnsw", "n": 1}, open(tmp_path / "meta.json", "w"))
    monkeypatch.setattr(vi, "_faiss", lambda: None)
    assert vi.load_index(str(tmp_path)) is None


def test_build_index_degrades_to_flat_without_faiss(monkeypatch):
    monkeypatch.setattr(vi, "_faiss", lambda: None)
    emb, ids = _corpus()
    idx = vi.build_index(emb, ids, backend="hnsw")             # asked hnsw, no faiss
    assert idx.backend == "flat"                               # degraded, not crashed


def test_search_k_larger_than_corpus():
    emb, ids = _corpus(n=5)
    idx = vi.FlatIndex.build(emb, ids)
    assert len(idx.search(emb[0], 100)) == 5


def test_flat_ties_are_deterministic_by_doc_id():
    """All-identical embeddings = every doc tied; top-k must be a stable lexicographic
    prefix regardless of k (no argpartition/argsort layout dependence)."""
    emb = np.ones((6, 4), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = vi.FlatIndex.build(emb, [f"d{i}" for i in range(6)])
    assert idx.search(emb[0], 2) == ["d0", "d1"]
    assert idx.search(emb[0], 3) == ["d0", "d1", "d2"]


def test_ivfpq_rejects_tiny_corpus(monkeypatch):
    """Forced ivfpq on too-few vectors degrades to flat (caught), never crashes."""
    monkeypatch.setattr(vi, "_faiss", lambda: object())   # pretend faiss present
    emb, ids = _corpus(n=10)
    # build via the IvfpqIndex guard directly: <256 vectors -> ValueError (then flat)
    import pytest as _pt
    with _pt.raises(ValueError):
        vi.IvfpqIndex.build(emb, ids)


# --- FAISS backends: only where faiss is installed --------------------------

faiss_available = vi._faiss() is not None
needs_faiss = pytest.mark.skipif(not faiss_available, reason="faiss not installed")


@needs_faiss
@pytest.mark.parametrize("backend", ["hnsw", "ivfpq"])
def test_faiss_backend_round_trip_and_recall(tmp_path, backend):
    emb, ids = _corpus(n=4000, d=64)
    idx = vi.build_index(emb, ids, backend=backend)
    assert idx.backend == backend
    vi.save_index(idx, str(tmp_path))
    loaded = vi.load_index(str(tmp_path))
    assert loaded.backend == backend
    # ANN: the query's own vector should return itself near the top
    hits = loaded.search(emb[42], 10)
    assert "d42" in hits


# --- AGENT_SEARCH_FLAT_FAISS opt-in exact fast path (default OFF) -----------
#
# Performance fix: flat's numpy `emb(float16) @ qv(float32)` has no BLAS kernel for
# float16 and falls back to a slow elementwise path. AGENT_SEARCH_FLAT_FAISS=1 opts
# FlatIndex into an in-memory faiss.IndexFlatIP (still EXACT, just SIMD+threaded)
# over the SAME stored embeddings. Must be byte-identical to before when unset.

def _fp32_reference_topk(emb_f16, qv, k):
    """The reference exact brute-force: stored (fp16) embeddings cast ONCE to fp32,
    matmul'd against a fp32 query — independent of both the numpy-mixed-precision
    path and of faiss, used only to prove faiss returns EXACT nearest neighbours."""
    import numpy as np
    emb32 = emb_f16.astype(np.float32)
    scores = emb32 @ qv.astype(np.float32)
    order = np.argsort(-scores, kind="stable")[:k]
    return [int(i) for i in order]


def test_env_unset_uses_numpy_path_unchanged(monkeypatch):
    """(a) Default (env unset) -> results identical to the pre-existing numpy path,
    and the lazy faiss cache is never populated."""
    monkeypatch.delenv("AGENT_SEARCH_FLAT_FAISS", raising=False)
    emb, ids = _corpus(n=300, d=48, seed=1)
    idx = vi.FlatIndex.build(emb, ids)
    qv = emb[11]
    got = idx.search(qv, 5)
    assert got == [ids[i] for i in _brute_topk(emb.astype(np.float32), qv, 5)]
    assert idx._flat_faiss is None                     # never built when disabled


@needs_faiss
def test_env_set_uses_faiss_flat_and_matches_fp32_reference(monkeypatch):
    """(b) Env set -> the faiss.IndexFlatIP path is used, and its top-k matches the
    fp32 brute-force reference EXACTLY (exact search, not ANN) for a batch of
    random queries (well separated -> no tie ambiguity)."""
    monkeypatch.setenv("AGENT_SEARCH_FLAT_FAISS", "1")
    emb, ids = _corpus(n=2000, d=64, seed=2)
    idx = vi.FlatIndex.build(emb, ids)
    rng = np.random.default_rng(99)
    queries = list(emb[[3, 100, 1999]])                # "real" (in-corpus) queries
    for _ in range(5):                                  # + random queries
        q = rng.standard_normal(64).astype(np.float32)
        q /= np.linalg.norm(q)
        queries.append(q)
    for qv in queries:
        got = idx.search(qv, 10)
        want = [ids[i] for i in _fp32_reference_topk(idx.emb, qv, 10)]
        assert got == want
    assert idx._flat_faiss is not None                  # lazily built, then cached
    cached = idx._flat_faiss
    idx.search(queries[0], 3)
    assert idx._flat_faiss is cached                    # never rebuilt on later queries


def test_env_set_but_faiss_unavailable_falls_back_to_numpy(monkeypatch):
    """(c) Env set but faiss missing (monkeypatched away) -> falls back cleanly to
    the numpy path, never crashes."""
    monkeypatch.setenv("AGENT_SEARCH_FLAT_FAISS", "1")
    monkeypatch.setattr(vi, "_faiss", lambda: None)
    emb, ids = _corpus(n=200, d=32, seed=3)
    idx = vi.FlatIndex.build(emb, ids)
    qv = emb[5]
    got = idx.search(qv, 4)
    assert got == [ids[i] for i in _brute_topk(emb.astype(np.float32), qv, 4)]
    assert idx._flat_faiss is None


@needs_faiss
def test_numpy_and_faiss_paths_agree_on_well_separated_vectors(monkeypatch):
    """(d) Both paths (env off = numpy, env on = faiss) agree on top-k for
    well-separated vectors (no float16 rounding ties to create ambiguity)."""
    emb, ids = _corpus(n=500, d=32, seed=4)
    monkeypatch.delenv("AGENT_SEARCH_FLAT_FAISS", raising=False)
    idx_numpy = vi.FlatIndex.build(emb, ids)
    numpy_result = idx_numpy.search(emb[17], 8)

    monkeypatch.setenv("AGENT_SEARCH_FLAT_FAISS", "1")
    idx_faiss = vi.FlatIndex.build(emb, ids)
    faiss_result = idx_faiss.search(emb[17], 8)

    assert numpy_result == faiss_result
