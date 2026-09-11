"""Sieve's ranking invariant: Boolean is for filtering only; ranking is the arm's model, and the
0-hit fallback must rank with the same model over the same index as the exact path.

* no dense belief attached  -> exact path and fallback both order by the persisted corpus BM25;
* dense belief attached (sieve) -> both order by RRF(BM25, dense) from the persisted embeddings;
* dense-only executor (sieve_dense) -> both order purely by dense similarity.

The dense side is a stub that only exposes what `DenseBelief` exposes to the executor
(`is_ready`, `top_k_doc_ids`, `score`) and counts encoder calls: nothing is encoded online for
documents, and one search pass encodes the query once.
"""
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.bql.executor import (
    DenseOnlyStructuralExecutor, StructuralExecutor)
from agent_search.retrievers.bql.parser import parse

DOCS = [
    {"_id": "A", "title": "Alpha treaty", "text": "alpha alpha alpha treaty signed"},
    {"_id": "B", "title": "Beta accord", "text": "alpha accord signed later"},
    {"_id": "C", "title": "Gamma pact", "text": "a pact with no shared words at all"},
    {"_id": "D", "title": "Delta", "text": "unrelated delta text"},
    {"_id": "E", "title": "Epsilon", "text": "alpha epsilon one two three four five six"},
]


class StubDense:
    """Prefers B, then C (which shares NO lexical term with the query), then E, A; never D."""
    sims = {"B": 0.9, "C": 0.8, "E": 0.5, "A": 0.2, "D": -0.5}

    def __init__(self):
        self.query_encodes = 0
        self._last = None

    def is_ready(self):
        return True

    def _encode(self, q):
        if self._last != q:
            self.query_encodes += 1
            self._last = q

    def top_k_doc_ids(self, query_text, k=None):
        self._encode(query_text)
        return sorted(self.sims, key=lambda d: -self.sims[d])[: (k or 10)]

    def score(self, query_text, doc_ids=None):
        self._encode(query_text)
        ids = list(doc_ids) if doc_ids is not None else list(self.sims)
        return {d: self.sims[d] for d in ids if d in self.sims}


def _executor(dense=None, dense_only=False):
    ex = StructuralExecutor(units_from_documents(DOCS)).prewarm()
    if dense_only:
        ex.__class__ = DenseOnlyStructuralExecutor
    if dense is not None:
        ex.attach_dense(dense)
    return ex


def test_without_dense_fallback_and_exact_path_share_the_bm25_ranker():
    ex = _executor()
    exact, n = ex.run_with_count(parse("alpha").expr, k=10)
    assert n == 3 and [d for d, _ in exact] == ["A", "B", "E"]     # BM25: tf, then length
    soft = ex.soft_topk(["alpha"], k=10)
    assert [d for d, _ in soft] == ["A", "B", "E"]                  # same order, same scorer
    assert all(s > 0 for _, s in soft)


def test_with_dense_the_fallback_uses_the_same_fused_ranker():
    dense = StubDense()
    ex = _executor(dense)
    exact, _ = ex.run_with_count(parse("alpha").expr, k=10)
    assert [d for d, _ in exact] == ["B", "A", "E"]                 # RRF(bm25 [A,B,E], dense [B,E,A])
    soft = ex.soft_topk(["alpha"], k=10)
    ids = [d for d, _ in soft]
    # the fallback pool is lexical-closest ∪ dense-nearest, fused by the SAME RRF rule:
    # B and A keep the exact path's relative order, and C (dense-only neighbour) surfaces
    assert ids[:2] == ["B", "A"]
    assert "C" in ids and ids.index("C") < ids.index("D")           # dense-only neighbour ranks by the model
    assert dense.query_encodes == 1                                 # one encode per search pass


def test_dense_only_executor_fallback_is_dense_ordered():
    dense = StubDense()
    ex = _executor(dense, dense_only=True)
    exact, _ = ex.run_with_count(parse("alpha").expr, k=10)
    assert [d for d, _ in exact] == ["B", "E", "A"]                 # pure dense order
    soft = ex.soft_topk(["alpha"], k=10)
    assert [d for d, _ in soft][:3] == ["B", "C", "E"]              # pure dense order, C included


def test_fallback_pool_is_bounded_and_dense_failure_degrades_to_bm25():
    class Broken(StubDense):
        def top_k_doc_ids(self, query_text, k=None):
            raise RuntimeError("dense index unavailable")

        def score(self, query_text, doc_ids=None):
            raise RuntimeError("dense index unavailable")

    ex = _executor(Broken())
    soft = ex.soft_topk(["alpha"], k=1)
    assert [d for d, _ in soft] == ["A"]                            # BM25 order, never raises


def test_coverage_fallback_for_a_single_constraint_follows_the_same_rule():
    dense = StubDense()
    ex = _executor(dense)
    rows = ex.coverage_topk(parse("alpha").expr, k=3)
    assert [r[0] for r in rows][:2] == ["B", "A"]
