"""Tests for the dense-embedding belief source of Indri retrieval
(`agent_search/retrievers/dense/belief.py`, blended in by `LuceneIndriAdapter` when a
`DenseBelief` is attached; `agent_search/retrievers/lucene/adapters.py`, Deviation 2).

CPU-only, tiny corpus, deterministic hash-based stub encoder (no model download):

1. dense-off: an adapter with no dense source returns Lucene's own ranking, no `#dense` entry.
2. dense-on rerank: a dense source that favours Lucene's worst-ranked hit moves it to the top
   at a high `INDRI_DENSE_W`. The adapter reranks Lucene's own pool; a document with no
   lexical overlap never enters it (Deviation 3), and a test pins that too.
3. w=0: the Lucene order and scores pass through unchanged; the `#dense` diagnostics entry
   is still attached.
4. `_plain_terms` operator stripping.
5. robustness: a dense source that raises degrades silently to lexical-only.
6. one-encode-per-search: the adapter's single `score` call encodes the query once.
7. the persisted embedding cache is reused across `DenseBelief` instances.

Needs a JVM; the module skips without one.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.dense.belief import (
    DenseBelief, _plain_terms, _plain_text)
from tests.lucene_support import build_lucene_indri, require_jvm

require_jvm()

# --- stub encoder --------------------------------------------------------------


class StubEncoder:
    """Deterministic hash-based bag-of-tokens vectors, no model download. A tiny SYNONYM
    table maps paraphrase tokens onto the same hash bucket, so the stub sees 'quagga' and
    'zebra' as the same token while the lexical index does not."""
    DIM = 32
    SYNONYMS = {"quagga": "zebra"}

    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kw):
        self.calls += 1
        rows = []
        for t in texts:
            v = np.zeros(self.DIM, dtype=np.float64)
            for tok in re.findall(r"[a-z0-9]+", (t or "").lower()):
                tok = self.SYNONYMS.get(tok, tok)
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
                v[h] += 1.0
            n = np.linalg.norm(v)
            rows.append(v / n if n > 0 else v)
        return np.asarray(rows, dtype=np.float32)


class RaisingDense:
    """A dense source whose every entry point raises: the adapter must degrade silently to
    lexical-only and never crash the agent loop."""

    def top_k_doc_ids(self, query_text, k=None):
        raise RuntimeError("boom")

    def score(self, query_text, doc_ids=None):
        raise RuntimeError("boom")


class FixedDense:
    """A dense source with hand-set similarities, for engineering a rerank."""

    def __init__(self, sims: dict):
        self.sims = sims

    def score(self, query_text, doc_ids=None):
        ids = doc_ids if doc_ids is not None else list(self.sims)
        return {d: self.sims.get(d, 0.0) for d in ids}


# --- fixtures --------------------------------------------------------------------


def _mk(doc_id: str, text: str, title: str | None = None) -> CodeUnit:
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=text, body=text, title=title)


def _corpus() -> list[CodeUnit]:
    return [
        # lexical hit for 'zebra'
        _mk("zebradoc", "zebra runs fast across the open field today"),
        # the paraphrase doc: zero lexical overlap with 'zebra'; the stub maps 'quagga' onto
        # 'zebra', so its vector is close to the query's
        _mk("paraphrase", "quagga quagga quagga"),
        # unrelated fillers
        _mk("other1", "cat mouse bird fish garden"),
        _mk("other2", "train station schedule arrival"),
        _mk("other3", "red blue green paint palette"),
        _mk("other4", "bread butter jam breakfast"),
        _mk("dogdoc", "dog park walk leash morning"),
        _mk("traindog", "dog train dog train station"),
    ]


@pytest.fixture(scope="module")
def units() -> list[CodeUnit]:
    return _corpus()


@pytest.fixture(scope="module")
def plain(units):
    return build_lucene_indri(units)


@pytest.fixture()
def dense(units, tmp_path) -> DenseBelief:
    d = DenseBelief(model="stub/model", index_root=str(tmp_path), encoder=StubEncoder())
    d.build_or_load(units, key="stubcorpus")
    return d


# --- (1) dense-off: Lucene's own ranking, no #dense entry ---------------------------

def test_dense_off_is_a_pure_lucene_search(plain):
    for q in ("#combine(dog train)", "zebra", "#or(cat bread)"):
        res = plain.search(q, k=8)
        assert res.error is None, q
        assert res.hits, q
        assert not any(name == "#dense" for name, _ in res.diagnostics), q


def test_nonexistent_term_returns_no_hits(plain):
    res = plain.search("nonexistentterm", k=8)
    assert res.error is None
    assert res.hits == []


# --- (2) dense-on: rerank within Lucene's pool; no pool expansion ------------------


def test_paraphrase_doc_invisible_to_lexical_search(plain):
    res = plain.search("zebra", k=8)
    assert res.error is None
    assert [d for d, _ in res.hits] == ["zebradoc"]


def test_dense_never_expands_the_pool(units, dense, monkeypatch):
    """Deviation 3: the adapter reranks what Lucene returned. The paraphrase doc has the
    highest stub similarity to 'zebra' but no lexical overlap, so it stays out."""
    monkeypatch.setenv("INDRI_DENSE_W", "0.9")
    sims = dense.score("zebra")
    assert sims["paraphrase"] == max(sims.values())
    res = build_lucene_indri(units, dense=dense).search("zebra", k=8)
    assert res.error is None
    assert [d for d, _ in res.hits] == ["zebradoc"]


def test_dense_reranks_lucene_pool_at_high_weight(units, plain, monkeypatch):
    r0 = plain.search("#combine(dog train)", k=8)
    doc_ids = [d for d, _ in r0.hits]
    assert len(doc_ids) >= 2
    boosted = doc_ids[-1]                    # Lucene's worst-ranked returned hit
    sims = {d: (10.0 if d == boosted else 0.0) for d in doc_ids}
    monkeypatch.setenv("INDRI_DENSE_W", "0.95")
    r1 = build_lucene_indri(units, dense=FixedDense(sims)).search("#combine(dog train)", k=8)
    assert r1.error is None
    assert [d for d, _ in r1.hits][0] == boosted
    assert set(d for d, _ in r1.hits) == set(doc_ids)      # same pool, new order


# --- (3) w=0: the Lucene order and scores pass through; #dense is still reported ------


def test_w_zero_preserves_lexical_scores_and_order(units, plain, dense, monkeypatch):
    monkeypatch.setenv("INDRI_DENSE_W", "0")
    lex = plain.search("#combine(dog train)", k=8)
    on = build_lucene_indri(units, dense=dense).search("#combine(dog train)", k=8)
    assert on.hits == lex.hits


def test_diagnostics_include_dense_entry(units, dense, monkeypatch):
    monkeypatch.setenv("INDRI_DENSE_W", "0.5")
    res = build_lucene_indri(units, dense=dense).search("#combine(dog train)", k=5)
    dense_entries = [(n, v) for n, v in res.diagnostics if n == "#dense"]
    assert len(dense_entries) == 1
    v = dense_entries[0][1]
    assert isinstance(v, float) and v <= 0.0 and v == v      # log of (0,1]: finite, <= 0
    # present at w=0 as well: the contribution is visible without affecting the ranking
    monkeypatch.setenv("INDRI_DENSE_W", "0")
    res0 = build_lucene_indri(units, dense=dense).search("#combine(dog train)", k=5)
    assert any(n == "#dense" for n, _ in res0.diagnostics)


# --- (4) operator-stripping helper ---------------------------------------------------


def test_plain_terms_strips_operators_fields_punct():
    q = '#combine( #1(bank management) treaty.title )'
    assert _plain_terms(q) == ["bank", "management", "treaty"]
    assert _plain_text(q) == "bank management treaty"
    q2 = '#weight(2.0 dolly.title 1.0 #syn(sheep ewe)) #date:between(2002-01-01 2002-12-31)'
    terms = _plain_terms(q2)
    assert "dolly" in terms and "sheep" in terms and "ewe" in terms
    assert not any(t.startswith("#") for t in terms)
    assert "title" not in terms and "combine" not in terms and "weight" not in terms
    assert _plain_terms("") == [] and _plain_terms(None) == []


# --- (5) robustness + (6) efficiency ---------------------------------------------------


def test_raising_dense_degrades_to_lexical_only(units, plain, monkeypatch):
    monkeypatch.setenv("INDRI_DENSE_W", "0.5")
    broken = build_lucene_indri(units, dense=RaisingDense())
    for q in ("#combine(dog train)", "zebra"):
        a, b = plain.search(q, k=8), broken.search(q, k=8)
        assert b.error is None
        assert b.hits == a.hits, q
        assert b.diagnostics == []


def test_one_query_encode_per_search(units, dense):
    enc = dense._retriever._model
    assert isinstance(enc, StubEncoder)
    calls_after_build = enc.calls                             # doc encode(s) at build time
    build_lucene_indri(units, dense=dense).search("#combine(dog train)", k=5)
    assert enc.calls == calls_after_build + 1


# --- (7) the persisted embedding cache -------------------------------------------------


def test_dense_cache_reused_across_instances(units, tmp_path):
    enc1 = StubEncoder()
    DenseBelief(model="stub/model", index_root=str(tmp_path),
                encoder=enc1).build_or_load(units, key="reuse")
    assert enc1.calls == 1                                    # one batched doc encode
    enc2 = StubEncoder()
    d2 = DenseBelief(model="stub/model", index_root=str(tmp_path),
                     encoder=enc2).build_or_load(units, key="reuse")
    assert enc2.calls == 0                                    # loaded from the persisted cache
    assert d2.is_ready()
    sims = d2.score("zebra")                                  # query encode only
    assert enc2.calls == 1
    # the engineered similarity survives the round-trip: the paraphrase doc is the pool max
    # (not exactly 1.0: the doc blob is 'qualname\ncode', so the doc_id token dilutes the
    # vector slightly, cos = 3/sqrt(10) ~= 0.949).
    assert sims["paraphrase"] == max(sims.values())
    assert sims["paraphrase"] > 0.9
