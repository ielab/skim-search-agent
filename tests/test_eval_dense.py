import numpy as np

from agent_search.retrievers.dense.dense import DenseRetriever
from agent_search.corpus.units import CodeUnit


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        rows = []
        for text in texts:
            low = text.lower()
            rows.append([1.0, 0.0] if "token" in low else [0.0, 1.0])
        return np.array(rows, dtype=float)


def _units():
    return [
        CodeUnit("a.py::token", "a.py", "token", 1, 1, "def token(): pass"),
        CodeUnit("b.py::render", "b.py", "render", 1, 1, "def render(): pass"),
    ]


def test_dense_retriever_persists_and_reuses_embeddings(tmp_path):
    enc1 = FakeEncoder()
    r1 = DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc1).index(_units(), key="repo@abc")
    assert r1.search("token", k=1) == ["a.py::token"]
    assert enc1.calls == 2  # docs + query

    enc2 = FakeEncoder()
    r2 = DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc2).index(_units(), key="repo@abc")
    assert enc2.calls == 0  # docs loaded from cache
    assert r2.search("token", k=1) == ["a.py::token"]
    assert enc2.calls == 1  # query only


def _other_units():
    """A DIFFERENT corpus (different doc_ids) than `_units()` -- simulates either a
    stale cache (corpus changed since it was built) or a wrong-key collision
    (two different corpora sharing a cache dir)."""
    return [
        CodeUnit("c.py::alpha", "c.py", "alpha", 1, 1, "def alpha(): pass"),
        CodeUnit("d.py::beta", "d.py", "beta", 1, 1, "def beta(): pass"),
        CodeUnit("e.py::gamma", "e.py", "gamma", 1, 1, "def gamma(): pass"),
    ]


def test_dense_retriever_rejects_incongruent_cache_and_rebuilds(tmp_path, capsys):
    """Regression for the adversarial-verification HIGH-latent finding: a wrong-key
    or stale cache must NEVER be trusted silently -- `DenseRetriever.index()` used to
    load whatever `doc_ids.json` was on disk under `key` with no check that it still
    matches the CURRENT corpus, so a stale/colliding cache would silently return
    wrong doc identities for every query. Now the doc_ids must be validated against
    the passed units; a mismatch is a loud, logged rebuild -- mirroring
    `StructuralExecutor.attach_units`'s doc-id-order congruence check on the BQL
    pickle path (bql/executor.py)."""
    enc1 = FakeEncoder()
    DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc1).index(
        _units(), key="repo@abc")
    assert enc1.calls == 1  # docs only (index() alone, no search)

    # Same cache key, but a DIFFERENT corpus this time -- the cache on disk is now
    # incongruent with what the caller is asking to index.
    enc2 = FakeEncoder()
    r2 = DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc2).index(
        _other_units(), key="repo@abc")

    # Must NOT silently serve the stale corpus_A cache: it re-encodes the NEW corpus
    # (one batched `.encode()` call over all 3 new docs).
    assert enc2.calls == 1, "stale/incongruent cache was trusted instead of rebuilt"
    assert sorted(r2._doc_ids) == ["c.py::alpha", "d.py::beta", "e.py::gamma"]

    # And the failure must be loud (a clear rebuild message), not silent.
    err = capsys.readouterr().err
    assert "incongruent" in err.lower() or "mismatch" in err.lower() or "stale" in err.lower()

    # A subsequent load with the SAME (new) corpus now hits the freshly-rebuilt,
    # congruent cache again.
    enc3 = FakeEncoder()
    DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc3).index(
        _other_units(), key="repo@abc")
    assert enc3.calls == 0


def test_dense_retriever_rejects_stale_fingerprint_same_doc_ids_and_rebuilds(tmp_path, capsys):
    """Same doc_ids, same order, same COUNT -- but a unit's `code` edited in place. The
    doc-id-list congruence check above can't see this (it's still an exact match); the
    `corpus_fingerprint` (agent_search.corpus.fingerprint) check attached to the persisted
    meta.json must catch it and force a rebuild instead of silently serving embeddings for
    text that no longer matches the corpus."""
    enc1 = FakeEncoder()
    DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc1).index(
        _units(), key="repo@fp")
    assert enc1.calls == 1

    edited = [
        CodeUnit("a.py::token", "a.py", "token", 1, 1,
                 "def token(): pass  # body edited in place"),
        CodeUnit("b.py::render", "b.py", "render", 1, 1, "def render(): pass"),
    ]
    enc2 = FakeEncoder()
    DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc2).index(
        edited, key="repo@fp")
    assert enc2.calls == 1, (
        "stale cache (same doc_ids, changed content) was trusted instead of rebuilt")
    err = capsys.readouterr().err
    assert "incongruent" in err.lower() or "mismatch" in err.lower() or "stale" in err.lower()

    # An OLD cache with no `corpus_fingerprint` key at all (pre-existing artifact) must
    # still be TRUSTED as before -- nothing to compare against.
    enc3 = FakeEncoder()
    r3 = DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc3).index(
        edited, key="repo@fp")
    cache_dir = r3._cache_dir("repo@fp")
    import json
    import os
    meta_path = os.path.join(cache_dir, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert meta.get("corpus_fingerprint")           # the key IS actually written
    meta.pop("corpus_fingerprint")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    enc4 = FakeEncoder()
    DenseRetriever("fake/model", index_root=str(tmp_path), encoder=enc4).index(
        edited, key="repo@fp")
    assert enc4.calls == 0, "an old cache with no fingerprint key must be trusted as before"
