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
