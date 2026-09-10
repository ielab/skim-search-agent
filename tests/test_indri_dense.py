"""Tests for the DENSE-EMBEDDING belief source of the Indri graded retrieval engine
(`agent_search/retrievers/indri/dense_belief.py` + the additive, default-OFF
`dense=` hooks in `model.py`).

CPU-only, tiny corpus, deterministic hash-based STUB encoder (no model download):

1. dense-off parity guard — `IndriExecutor(units)` and `IndriExecutor(units, dense=None)`
   produce byte-identical hits + diagnostics across representative queries.
2. dense-on recall — a doc with NO lexical overlap with the query but an engineered-
   similar stub vector enters the pool (dense expansion) and ranks TOP at high w.
3. w=0 — lexical-only relative order and scores are preserved (dense only expands
   the pool; the combination is a no-op).
4. diagnostics gain a '#dense' contribution entry.
5. `_plain_terms` operator stripping (mirrors doc_indri's `_indri_query_terms`).
+ robustness: a dense source that RAISES degrades silently to lexical-only.
+ one-encode-per-search: expansion + scoring share a memoized query vector.

Real-corpus integration check (browsecomp_plus_structured + local bge-base HF
snapshot, offline): encoding ~200-300 real docs on this login node's CPU takes
~200-300s (measured: 300 docs in 298.9s, ~1 s/doc at 512 tokens), far too slow for
the default suite — gated behind INDRI_DENSE_INTEGRATION=1 and skipped otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import numpy as np
import pytest

from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.indri.dense_belief import (
    DenseBelief, _plain_terms, _plain_text)
from agent_search.retrievers.indri.model import IndriExecutor

# --- stub encoder --------------------------------------------------------------


class StubEncoder:
    """Deterministic hash-based bag-of-tokens vectors — no model download. A tiny
    SYNONYM table maps paraphrase tokens onto the same hash bucket, so "semantic
    similarity without lexical overlap" is engineerable: the stub sees 'quagga'
    and 'zebra' as the same token, the lexical index does not."""
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
    """A dense source whose every entry point raises — the executor must degrade
    silently to lexical-only (never crash the agent loop)."""

    def top_k_doc_ids(self, query_text, k=None):
        raise RuntimeError("boom")

    def score(self, query_text, doc_ids=None):
        raise RuntimeError("boom")


# --- fixtures --------------------------------------------------------------------


def _mk(doc_id: str, text: str, title: str | None = None) -> CodeUnit:
    return CodeUnit(doc_id=doc_id, path=f"{doc_id}.txt", qualname=doc_id,
                    start_line=1, end_line=1, code=text, body=text, title=title)


def _corpus() -> list[CodeUnit]:
    return [
        # lexical hit for 'zebra' (contains the term, diluted vector)
        _mk("zebradoc", "zebra runs fast across the open field today"),
        # THE paraphrase doc: zero lexical overlap with 'zebra', but the stub maps
        # 'quagga' -> 'zebra' so its vector is (near-)identical to the query's.
        _mk("paraphrase", "quagga quagga quagga"),
        # unrelated fillers (low similarity, provide the min of the pool min-max)
        _mk("other1", "cat mouse bird fish garden"),
        _mk("other2", "train station schedule arrival"),
        _mk("other3", "red blue green paint palette"),
        _mk("other4", "bread butter jam breakfast"),
        _mk("dogdoc", "dog park walk leash morning"),
        _mk("traindog", "dog train dog train station"),
    ]


@pytest.fixture()
def units() -> list[CodeUnit]:
    return _corpus()


@pytest.fixture()
def dense(units, tmp_path) -> DenseBelief:
    d = DenseBelief(model="stub/model", index_root=str(tmp_path), encoder=StubEncoder())
    d.build_or_load(units, key="stubcorpus")
    return d


# --- (1) dense-off parity guard ----------------------------------------------------

PARITY_QUERIES = [
    "#combine(dog train)",
    "zebra",
    "#weight(1.0 dog 0.5 train)",
    "#combine(zebra quagga)",
    "#or(cat bread)",
    "nonexistentterm",
]


def test_dense_off_is_byte_identical_to_plain_executor(units):
    plain = IndriExecutor(units, mu=2500)
    off = IndriExecutor(units, mu=2500, dense=None)
    for q in PARITY_QUERIES:
        a, b = plain.search(q, k=8), off.search(q, k=8)
        assert repr(a.hits) == repr(b.hits), q         # scores byte-identical, not just ~=
        assert a.error == b.error, q
        assert repr(a.diagnostics) == repr(b.diagnostics), q
        assert not any(name == "#dense" for name, _ in b.diagnostics), q


def test_attach_dense_none_is_noop_and_load_or_build_default_off(units, tmp_path):
    from agent_search.retrievers.indri.model import load_or_build
    plain = IndriExecutor(units, mu=2500)
    ex = load_or_build(units, index_root=str(tmp_path), key="parity")   # no dense kwarg
    assert ex.dense is None
    ex.attach_dense(None)
    a, b = plain.search("#combine(dog train)", k=8), ex.search("#combine(dog train)", k=8)
    assert repr(a.hits) == repr(b.hits)


# --- (2) dense-on: paraphrase doc enters the pool and ranks top at high w -----------


def test_paraphrase_doc_invisible_to_lexical_only(units):
    res = IndriExecutor(units, mu=2500).search("zebra", k=8)
    assert res.error is None
    assert "paraphrase" not in [d for d, _ in res.hits]      # never pooled lexically


def test_dense_expands_pool_and_ranks_paraphrase_top_at_high_w(units, dense, monkeypatch):
    monkeypatch.setenv("INDRI_DENSE_W", "0.9")
    ex = IndriExecutor(units, mu=2500, dense=dense)
    res = ex.search("zebra", k=8)
    assert res.error is None
    doc_ids = [d for d, _ in res.hits]
    assert "paraphrase" in doc_ids                           # pool expansion (recall)
    assert doc_ids[0] == "paraphrase"                        # dense belief dominates at w=0.9
    assert "zebradoc" in doc_ids                             # lexical hit still present


def test_dense_kwarg_via_attach_dense(units, dense, monkeypatch):
    monkeypatch.setenv("INDRI_DENSE_W", "0.9")
    ex = IndriExecutor(units, mu=2500).attach_dense(dense)
    res = ex.search("zebra", k=8)
    assert [d for d, _ in res.hits][0] == "paraphrase"


# --- (3) w=0: lexical-only order (dense expands the pool, combination is a no-op) ---


def test_w_zero_preserves_lexical_scores_and_order(units, dense, monkeypatch):
    monkeypatch.setenv("INDRI_DENSE_W", "0")
    lex = IndriExecutor(units, mu=2500).search("#combine(dog train)", k=8)
    on = IndriExecutor(units, mu=2500, dense=dense).search("#combine(dog train)", k=8)
    lex_scores = dict(lex.hits)
    on_scores = dict(on.hits)
    # every lexically-scored doc keeps its EXACT score at w=0 ...
    for doc_id, s in lex_scores.items():
        assert on_scores[doc_id] == s
    # ... and the relative order of the lexical docs is unchanged.
    lex_order = [d for d, _ in lex.hits]
    on_order_restricted = [d for d, _ in on.hits if d in lex_scores]
    assert on_order_restricted == lex_order


# --- (4) diagnostics gain a '#dense' entry ------------------------------------------


def test_diagnostics_include_dense_entry(units, dense, monkeypatch):
    monkeypatch.setenv("INDRI_DENSE_W", "0.5")
    res = IndriExecutor(units, mu=2500, dense=dense).search("#combine(dog train)", k=5)
    dense_entries = [(n, v) for n, v in res.diagnostics if n == "#dense"]
    assert len(dense_entries) == 1
    v = dense_entries[0][1]
    assert isinstance(v, float) and v <= 0.0 and v == v      # log of (0,1]: finite, <= 0
    # present even at w=0 (contribution visible without affecting ranking)
    monkeypatch.setenv("INDRI_DENSE_W", "0")
    res0 = IndriExecutor(units, mu=2500, dense=dense).search("#combine(dog train)", k=5)
    assert any(n == "#dense" for n, _ in res0.diagnostics)


# --- (5) operator-stripping helper ---------------------------------------------------


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


# --- robustness + efficiency ----------------------------------------------------------


def test_raising_dense_degrades_to_lexical_only(units):
    plain = IndriExecutor(units, mu=2500)
    broken = IndriExecutor(units, mu=2500, dense=RaisingDense())
    for q in ("#combine(dog train)", "zebra"):
        a, b = plain.search(q, k=8), broken.search(q, k=8)
        assert b.error is None
        assert repr(a.hits) == repr(b.hits), q


def test_one_query_encode_per_search(units, dense):
    enc = dense._retriever._model
    assert isinstance(enc, StubEncoder)
    calls_after_build = enc.calls                             # doc encode(s) at build time
    ex = IndriExecutor(units, mu=2500, dense=dense)
    ex.search("zebra", k=5)
    # expansion + pool scoring share the memoized query vector: exactly ONE encode.
    assert enc.calls == calls_after_build + 1


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
    # engineered similarity survives the persisted round-trip: the paraphrase doc is
    # the pool max (not exactly 1.0 — the doc blob is 'qualname\ncode', so the doc_id
    # token dilutes the vector slightly: cos = 3/sqrt(10) ~= 0.949).
    assert sims["paraphrase"] == max(sims.values())
    assert sims["paraphrase"] > 0.9


# --- real-corpus integration check (opt-in: minutes of CPU encoding) ------------------

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BC_CORPUS = os.path.join(_REPO, "data", "browsecomp_plus_structured", "corpus.jsonl")
_BGE_SNAPSHOT = os.path.join(os.environ.get("HF_HOME", ""), "hub", "models--BAAI--bge-base-en-v1.5")

integration = pytest.mark.skipif(
    not (os.environ.get("INDRI_DENSE_INTEGRATION")
         and os.path.exists(_BC_CORPUS) and os.path.isdir(_BGE_SNAPSHOT)),
    reason="opt-in real-corpus check: set INDRI_DENSE_INTEGRATION=1 with the local "
           "browsecomp corpus + bge-base HF snapshot (HF_HOME) present; encoding 200 "
           "real docs on this CPU takes ~3-5 min (measured 300 docs = 298.9s)")


@integration
def test_real_corpus_dense_surfaces_paraphrase_case(tmp_path):
    """200-doc mini-check on the REAL browsecomp_plus_structured corpus (the first
    200 corpus lines include all 5 gold docs of query 769) with the REAL bge-base
    encoder, offline. Asserts the dense arm surfaces a gold doc for an OBFUSCATED
    keyword query (paraphrased wording: 'college festival supporting palestinians'
    vs the gold doc's 'university cultural week Palestinian support') and reports
    timing."""
    from agent_search.corpus.units import units_from_documents
    docs = []
    with open(_BC_CORPUS, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 200:
                break
            docs.append(json.loads(line))
    units = units_from_documents(docs)
    gold = {"5412", "82002", "86190", "18639", "41759"}
    assert gold & {u.doc_id for u in units}

    t0 = time.time()
    dense = DenseBelief(index_root=str(tmp_path), device="cpu").build_or_load(
        units, key="bc_mini_200")
    build_s = time.time() - t0

    os.environ["INDRI_DENSE_W"] = "0.5"
    try:
        ex = IndriExecutor(units, dense=dense)
        # OBFUSCATED query: no 'university'/'cultural'/'week' tokens from doc 5412.
        q = "#combine(college festival supporting palestinians)"
        t1 = time.time()
        res = ex.search(q, k=10)
        query_s = time.time() - t1
    finally:
        os.environ.pop("INDRI_DENSE_W", None)
    assert res.error is None
    assert res.hits
    print(f"\n[integration] build(200 docs)={build_s:.1f}s query={query_s:.2f}s "
          f"hits={[d for d, _ in res.hits]}")
    assert gold & {d for d, _ in res.hits}
