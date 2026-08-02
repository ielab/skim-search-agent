"""Tests for the `BM25_BACKEND` knob (agent_search.retrievers.lexical.build_bm25_engine) that
selects the doc arm's bm25 engine: `local` (BM25Local, the dependency-free approximation) or
`pyserini` (BM25Pyserini, canonical Lucene BM25 — Porter stemming + stopwords, k1=0.9/b=0.4,
the SAME engine SWE-bench's own BM25 baseline uses).

Covers the four things "make pyserini a selectable bm25 engine" needs pinned:

  1. agreement     — BM25Local and BM25Pyserini rank documents MATERIALLY differently for the
                      same query/corpus — the reason BM25_BACKEND=pyserini exists at all.
                      Measured empirically on the REAL browsecomp_plus_structured corpus
                      (67,707 docs, this project's data/browsecomp_plus_structured/): mean
                      top-5 Jaccard between the two engines' rankings is ~0.546 (well below
                      1.0 — BM25Local's tokenizer is not a faithful stand-in for canonical
                      Lucene BM25 on prose). This test reproduces the SAME mechanism (verb
                      tense/plural variants BM25Local never stems but Lucene's analyzer does)
                      on a small, deterministic, IN-FILE corpus — fast and hermetic, no
                      dependency on staged data/ files — and pins the resulting divergence with
                      a wide tolerance band, so a future change that accidentally makes the two
                      engines agree (byte-identical rankings — the whole point of having two
                      engines would be silently lost) or disagree completely (something broke)
                      is caught.
  2. selection      — env BM25_BACKEND resolves to the right engine CLASS, both at the shared
                      `build_bm25_engine` helper and through `agent_search.agent.retriever`'s
                      real per-episode construction path.
  3. offline safety — BM25Pyserini.search() makes no network call (a local Lucene index read
                      only) — verified by blocking socket creation during a real search, and by
                      importing the module in a subprocess with NO `OPENAI_API_KEY` set (the
                      transitive-import landmine pyserini.py's module docstring documents).
  4. listing parity — Bm25Visit's rendered search listing has the IDENTICAL shape regardless of
                      which engine answered the query (a workspace only ever consumes the
                      returned doc_id list; rendering is engine-agnostic).

Needs a real JVM (Java 11+) + `pyserini` importable — both are provisioned in this project's
`envs/` (see agent_search/retrievers/lexical/pyserini.py's module docstring). If unavailable,
the whole module is skipped rather than hard-failing an environment that never opted into the
pyserini backend.
"""
from __future__ import annotations

import os
import random
import re
import socket
import subprocess
import sys

import pytest

pytest.importorskip(
    "pyserini.search.lucene", reason="pyserini not installed — BM25_BACKEND=pyserini unavailable")


def _jvm_available() -> bool:
    """A real probe, not just a python import check — pyserini's JVM starts lazily on first
    Java object construction, so `import pyserini.search.lucene` alone doesn't prove Java
    actually works (e.g. no `java` binary / bad JAVA_HOME)."""
    try:
        os.environ.setdefault("OPENAI_API_KEY", "agent-search-unused-placeholder")
        from pyserini.pyclass import autoclass
        autoclass("java.lang.String")
        return True
    except Exception:
        return False


if not _jvm_available():
    pytest.skip("no working JVM — BM25_BACKEND=pyserini needs Java 11+", allow_module_level=True)

from agent_search.agent.retriever import AgentRetriever
from agent_search.agent.tools.doc_research import Bm25Visit
from agent_search.corpus.units import units_from_documents
from agent_search.retrievers.lexical import build_bm25_engine
from agent_search.retrievers.lexical.bm25 import BM25Local
from agent_search.retrievers.lexical.pyserini import BM25Pyserini


# --- a small, deterministic corpus reproducing the browsecomp_plus divergence mechanism -------
# Verb lemmas layered with morphological variants: BM25Local's code_tokenize does NOT stem (a
# doc saying "climbed" shares no token with a query saying "climbing"), while pyserini's Lucene
# analyzer applies Porter stemming (both reduce to "climb") — which documents a query matches
# genuinely differs by engine. Distinctive proper-noun "topic tags" (~15 docs each) give queries
# a STABLE partial-overlap core (both engines tokenize a literal capitalized name identically),
# so the corpus produces PARTIAL, not total, divergence — matching the real-world pattern (some
# ranks agree, some don't), not "every result differs" or "every result agrees".
_LEMMAS = ["climb", "fish", "sail", "study", "teach", "build", "paint", "dance", "write",
          "plan", "design", "grow", "cook", "sing", "play", "travel", "explore", "research"]
_FORMS = ["{0}", "{0}ing", "{0}ed", "{0}s"]
_TAGS = ["Kestrel Valley", "Ashwood Fields", "Marrow Bay", "Thistledown Ridge",
        "Copperlake District", "Windmere Basin", "Silverpine Hollow", "Blackthorn Pass"]
_STOP = ["the", "a", "an", "of", "in", "on", "at", "with", "near", "along", "through",
        "over", "under", "between", "during", "and", "but", "for", "to"]
_ADJ = ["ancient", "remote", "vast", "rugged", "quiet", "hidden", "coastal", "historic"]
_DOCS_PER_TAG = 15


def _make_corpus_and_queries() -> tuple[list[dict], list[str]]:
    """Docs THEN queries, drawn from ONE fixed-seed `random.Random` stream (matching the
    calibration script this test's tolerance band was measured from) — a single generation
    pass, not two independently-seeded ones, so there is no fragile hand-replay of the doc
    loop's draw sequence to keep a second stream in sync."""
    rng = random.Random(11)

    def sentence(tag):
        lemma = rng.choice(_LEMMAS)
        form = rng.choice(_FORMS).format(lemma)
        adj = rng.choice(_ADJ)
        stop_a, stop_b = rng.sample(_STOP, 2)
        subj = rng.choice(["Researchers", "Travelers", "Local guides", "Volunteers",
                           "Scientists", "Villagers"])
        return f"{subj} were {form} {stop_a} {adj} {tag} {stop_b} the season."

    docs, doc_i = [], 0
    for tag in _TAGS:
        for _ in range(_DOCS_PER_TAG):
            text = " ".join(sentence(tag) for _ in range(rng.randint(2, 3)))
            docs.append({"_id": f"doc{doc_i}", "title": f"{tag} Report {doc_i}", "text": text})
            doc_i += 1

    queries = []
    for tag in _TAGS:
        lemma = rng.choice(_LEMMAS)
        form = rng.choice(_FORMS).format(lemma)
        queries.append(f"{form} {tag}")
    for lemma in _LEMMAS[:8]:
        form = rng.choice(_FORMS).format(lemma)
        tag = rng.choice(_TAGS)
        queries.append(f"{form} {tag}")
    return docs, queries


_AGREEMENT_CORPUS_KEY = "test_pyserini_backend_agreement_corpus"

# Empirically measured on THIS corpus (fixed seed 11): mean top-5 Jaccard = 0.485, min 0.250,
# max 1.000 across the 16 queries below — materially divergent (well under 1.0) but not
# uncorrelated (well over 0.0), the same qualitative shape as the real browsecomp_plus figure
# (~0.546 on the full 67,707-doc corpus). Tolerance band is wide on purpose: this pins the
# PHENOMENON (a real, partial divergence), not a library-version-fragile exact float.
_MEAN_JACCARD_FLOOR = 0.25
_MEAN_JACCARD_CEILING = 0.75


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    """Both engines built ONCE over the SAME corpus, shared by every test below — a real
    pyserini Lucene build costs real wall-clock (JVM startup + indexing subprocess), so this
    keeps the whole module's cost to one build instead of one per test."""
    docs, _queries = _make_corpus_and_queries()
    units = units_from_documents(docs)
    ubyid = {u.doc_id: u for u in units}
    local = BM25Local().index(units)
    index_root = str(tmp_path_factory.mktemp("pyserini_backend"))
    pys = BM25Pyserini(index_root=index_root, rebuild=True).index(units, key=_AGREEMENT_CORPUS_KEY)
    return units, ubyid, local, pys, index_root


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# --- 1. agreement: material, partial divergence, pinned with a tolerance band ------------------

def test_agreement_pinned_within_tolerance_band(engines):
    _units, _ubyid, local, pys, _root = engines
    _docs, queries = _make_corpus_and_queries()
    assert len(queries) == 16

    jaccards = [_jaccard(local.search(q, k=5), pys.search(q, k=5)) for q in queries]
    mean_j = sum(jaccards) / len(jaccards)

    assert _MEAN_JACCARD_FLOOR <= mean_j <= _MEAN_JACCARD_CEILING, (
        f"mean top-5 Jaccard {mean_j:.3f} outside the pinned tolerance band "
        f"[{_MEAN_JACCARD_FLOOR}, {_MEAN_JACCARD_CEILING}] — either the two BM25 engines "
        f"stopped diverging (drifted toward 1.0: BM25_BACKEND=pyserini would no longer be "
        f"doing anything different from 'local') or something broke retrieval entirely "
        f"(drifted toward 0.0).")


def test_agreement_is_not_total_and_not_zero_per_query(engines):
    """Sanity on the per-query DISTRIBUTION, not just the mean — the corpus/query design must
    actually produce a MIX of full agreement, partial agreement, and disagreement (the real-
    world shape), not e.g. every query landing at exactly the mean."""
    _units, _ubyid, local, pys, _root = engines
    _docs, queries = _make_corpus_and_queries()
    jaccards = [_jaccard(local.search(q, k=5), pys.search(q, k=5)) for q in queries]
    assert any(j < 1.0 for j in jaccards), "expected at least one query where engines disagree"
    assert any(j > 0.0 for j in jaccards), "expected at least one query where engines agree"
    assert len(set(jaccards)) > 1, "expected genuine variance across queries, not a constant"


def test_both_engines_return_real_doc_ids(engines):
    """A collapsed/degenerate engine (e.g. always empty, or ids outside the corpus) would
    trivially satisfy the Jaccard band by both returning `[]` — guard against that."""
    units, ubyid, local, pys, _root = engines
    q = "climbing Kestrel Valley"
    lr, pr = local.search(q, k=5), pys.search(q, k=5)
    assert lr and pr
    assert all(i in ubyid for i in lr)
    assert all(i in ubyid for i in pr)


# --- 2. backend selection: env knob -> engine class ---------------------------------------------

def test_build_bm25_engine_defaults_to_local(monkeypatch):
    monkeypatch.delenv("BM25_BACKEND", raising=False)
    units = units_from_documents([{"_id": "d1", "title": "T", "text": "hello world"}])
    eng = build_bm25_engine(units)
    assert isinstance(eng, BM25Local)


def test_build_bm25_engine_local_is_explicit_too(monkeypatch):
    monkeypatch.setenv("BM25_BACKEND", "local")
    units = units_from_documents([{"_id": "d1", "title": "T", "text": "hello world"}])
    assert isinstance(build_bm25_engine(units), BM25Local)


def test_build_bm25_engine_is_case_insensitive_and_strips_whitespace(monkeypatch, tmp_path):
    monkeypatch.setenv("BM25_BACKEND", "  PySerini  ")
    units = units_from_documents([{"_id": "d1", "title": "T", "text": "hello world"}])
    eng = build_bm25_engine(units, index_root=str(tmp_path))   # a real (tiny) Lucene build
    assert isinstance(eng, BM25Pyserini)


def test_build_bm25_engine_pyserini_selects_pyserini(engines, monkeypatch):
    units, _ubyid, _local, _pys, index_root = engines
    monkeypatch.setenv("BM25_BACKEND", "pyserini")
    # SAME index_root/key as the `engines` fixture's already-built index — a cache hit, so
    # this doesn't pay a second real Lucene build.
    eng = build_bm25_engine(units, index_root=index_root, key=_AGREEMENT_CORPUS_KEY)
    assert isinstance(eng, BM25Pyserini)
    assert eng.search("climbing Kestrel Valley", k=5)


def test_build_bm25_engine_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("BM25_BACKEND", "bogus")
    units = units_from_documents([{"_id": "d1", "title": "T", "text": "hello world"}])
    with pytest.raises(ValueError, match="BM25_BACKEND"):
        build_bm25_engine(units)


@pytest.mark.parametrize("toolset", [
    ("bm25_search", "visit"),          # bm25 arm
    ("bm25_search", "fetch"),          # bm25fetch arm
    ("bm25q_search", "visit_q"),       # bm25q arm
    ("bm25_search_snip", "fetch"),     # bm25fetchsnip arm
])
def test_agent_retriever_bm25_family_arms_respect_backend_local(toolset):
    """The real per-episode construction site (agent_search.agent.retriever) — every
    bm25-family arm's `self._bm25` is BM25Local when BM25_BACKEND is unset/local, whatever the
    specific arm (bm25/bm25fetch/bm25q/bm25fetchsnip all funnel through the SAME
    `_build_bm25_engine` call in AgentRetriever.index())."""
    os.environ.pop("BM25_BACKEND", None)
    units = units_from_documents([{"_id": "d1", "title": "T", "text": "hello world"}])
    r = AgentRetriever(policy_factory=lambda: None, toolset=toolset, domain="general").index(units)
    assert isinstance(r._bm25, BM25Local)


def test_agent_retriever_bm25_arm_respects_backend_pyserini(engines, monkeypatch):
    units, _ubyid, _local, _pys, index_root = engines
    monkeypatch.setenv("BM25_BACKEND", "pyserini")
    r = AgentRetriever(policy_factory=lambda: None, toolset=("bm25_search", "visit"),
                       domain="general", index_root=index_root).index(units, key=_AGREEMENT_CORPUS_KEY)
    assert isinstance(r._bm25, BM25Pyserini)


# --- 3. offline safety: no network call during search() -----------------------------------------

def test_search_makes_no_python_level_network_call(engines, monkeypatch):
    _units, _ubyid, _local, pys, _root = engines

    def _blocked(*a, **k):
        raise AssertionError("search() attempted to open a socket — should be 100% local")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    try:
        ids = pys.search("climbing Kestrel Valley", k=5)
    finally:
        monkeypatch.undo()
    assert ids  # the search still worked with sockets blocked


def test_import_succeeds_with_no_openai_api_key_in_a_clean_subprocess():
    """Regression for the transitive-import landmine pyserini.py's module docstring
    documents: `pyserini.search.lucene` transitively imports `pyserini.encode._openai`, which
    used to raise at IMPORT time if OPENAI_API_KEY was unset — and only `evaluation/__init__.py`
    (not pyserini.py itself) used to set the placeholder, so importing
    `agent_search.retrievers.lexical.pyserini` directly (bypassing `evaluation`) on a machine
    with no OPENAI_API_KEY in its environment used to crash before this module set its own
    placeholder. Runs in a REAL subprocess with OPENAI_API_KEY/GEMINI_API_KEY stripped — the
    only way to prove import-order independence (the parent test process may already have one
    set from an earlier test/import)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_API_KEY", "GEMINI_API_KEY")}
    proc = subprocess.run(
        [sys.executable, "-c",
         "from agent_search.retrievers.lexical.pyserini import BM25Pyserini; "
         "print('OK', BM25Pyserini().name)"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        f"import failed with no OPENAI_API_KEY set — stdout={proc.stdout!r} "
        f"stderr={proc.stderr[-2000:]!r}")
    assert "OK bm25_pyserini" in proc.stdout


# --- 4. listing parity: Bm25Visit's rendering shape is engine-agnostic --------------------------

_LISTING_RE = re.compile(
    r"^search: .+   \(\d+ matches\):\n(  \d+  \S+  '.*'  .*…\n?)+$")


def test_listing_shape_is_identical_across_backends(engines):
    units, ubyid, local, pys, _root = engines
    query = "climbing Kestrel Valley"

    local_out = Bm25Visit(units, engine=local, ubyid=ubyid).search(query, k=5)
    pys_out = Bm25Visit(units, engine=pys, ubyid=ubyid).search(query, k=5)

    assert _LISTING_RE.match(local_out), f"local listing didn't match the expected shape:\n{local_out}"
    assert _LISTING_RE.match(pys_out), f"pyserini listing didn't match the expected shape:\n{pys_out}"

    # same NUMBER of rendered lines (both requested k=5 and both matched >=1 doc; the exact
    # doc_ids may differ per the agreement tests above, but the RENDERING shape must not).
    local_lines = local_out.splitlines()
    pys_lines = pys_out.splitlines()
    assert local_lines[0].startswith(f"search: {query}")
    assert pys_lines[0].startswith(f"search: {query}")
    assert len(local_lines) == len(pys_lines)


def test_listing_ranks_are_1_indexed_and_sequential_for_both_backends(engines):
    units, ubyid, local, pys, _root = engines
    query = "climbing Kestrel Valley"
    for engine in (local, pys):
        ws = Bm25Visit(units, engine=engine, ubyid=ubyid)
        out = ws.search(query, k=5)
        ranks = [int(m.group(1)) for m in re.finditer(r"^  (\d+)  ", out, re.MULTILINE)]
        assert ranks == list(range(1, len(ranks) + 1))


def test_visit_whole_doc_read_works_for_both_backends(engines):
    """The read side (`visit`) is engine-agnostic by construction (it never touches `self.bm`),
    but confirm end to end: a doc surfaced by EITHER engine's search is visitable."""
    units, ubyid, local, pys, _root = engines
    query = "climbing Kestrel Valley"
    for engine in (local, pys):
        ws = Bm25Visit(units, engine=engine, ubyid=ubyid)
        ws.search(query, k=5)
        assert ws.last_hits
        out = ws.visit(1)
        assert not out.startswith("ERROR")


# --- 5. build efficiency knobs: parallel indexing + lean stored fields --------------------------

def test_index_threads_defaults_to_cpu_count_and_env_overrides(monkeypatch):
    from agent_search.retrievers.lexical.pyserini import _index_threads
    monkeypatch.delenv("BM25_PYSERINI_THREADS", raising=False)
    assert _index_threads() == max(1, os.cpu_count() or 1)
    monkeypatch.setenv("BM25_PYSERINI_THREADS", "3")
    assert _index_threads() == 3
    monkeypatch.setenv("BM25_PYSERINI_THREADS", "0")   # floor: never < 1
    assert _index_threads() == 1


def test_store_raw_is_off_by_default_and_env_enables(monkeypatch):
    from agent_search.retrievers.lexical.pyserini import _store_raw
    monkeypatch.delenv("BM25_PYSERINI_STORE_RAW", raising=False)
    assert _store_raw() is False
    monkeypatch.setenv("BM25_PYSERINI_STORE_RAW", "1")
    assert _store_raw() is True


def test_build_shards_corpus_one_jsonl_per_thread(engines):
    """Anserini's JsonCollection parallelizes across FILES — the build must have written the
    corpus as multiple shards (one per thread requested), not one monolithic docs.jsonl, or
    --threads N is silently single-threaded. The module-scope `engines` fixture built its index
    with the default thread count, so its corpus dir is the artifact to inspect."""
    _units, _ubyid, _local, _pys, index_root = engines
    corpus_dir = os.path.join(index_root, "bm25_pyserini", _AGREEMENT_CORPUS_KEY, "corpus")
    shards = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".jsonl"))
    assert shards, "no corpus shards written"
    assert "docs.jsonl" not in shards, "legacy single-file layout — sharding not applied"
    expected = min(max(1, os.cpu_count() or 1), 120)   # 120 docs in the fixture corpus
    assert len(shards) <= max(1, os.cpu_count() or 1)
    if (os.cpu_count() or 1) > 1:
        assert len(shards) > 1, (
            "corpus written as ONE shard on a multi-core machine — the parallel build "
            "regressed to effectively single-threaded ingestion")
    # every doc is in exactly one shard (no loss, no duplication across the split)
    n_lines = 0
    for s in shards:
        with open(os.path.join(corpus_dir, s)) as fh:
            n_lines += sum(1 for _ in fh)
    assert n_lines == len(_units)


def test_lean_index_has_no_stored_raw_but_searches_fine(engines):
    """Default build stores NO raw contents/docvectors — doc(id).raw() comes back empty/None —
    yet ranking works (BM25 needs only the inverted index). The tool layer renders from its own
    units, so nothing downstream misses the stored fields."""
    _units, _ubyid, _local, pys, _root = engines
    ids = pys.search("climbing Kestrel Valley", k=3)
    assert ids
    d = pys._searcher.doc(ids[0])
    assert d is None or not d.raw()


# --- doc-count congruence guard (adversarial-verification HIGH-latent finding) --------------

def _small_docs(n: int, prefix: str = "d") -> list:
    return [{"_id": f"{prefix}{i}", "title": f"{prefix}{i}", "text": f"filler content {i}"}
            for i in range(n)]


def test_is_built_rejects_stale_doc_count_and_rebuilds(tmp_path):
    """Regression: `BM25Pyserini._is_built` used to be a bare `segments_*`-file presence
    check, so an index left on disk under a cache key whose CORPUS later changed (docs
    added/removed) was silently reused -- serving BM25 hits for the wrong document set with
    no error anywhere. Now `index()` writes a `meta.json` doc-count sentinel and cross-checks
    it on every subsequent `index()` call for the same key; a mismatch is a loud, logged
    rebuild (mirrors the dense-cache congruence check in retrievers/dense/dense.py and the
    lucene_structured `is_built` doc-count check in index_builder.py)."""
    index_root = str(tmp_path)
    key = "congruence_test_corpus"

    units_a = units_from_documents(_small_docs(5, "a"))
    pys_a = BM25Pyserini(index_root=index_root, rebuild=True).index(units_a, key=key)
    hits_a = set(pys_a.search("filler", k=10))
    assert hits_a and hits_a <= {u.doc_id for u in units_a}  # sanity: searchable at all
    meta_path = os.path.join(index_root, "bm25_pyserini", key, "meta.json")
    assert os.path.exists(meta_path)

    # A DIFFERENT-sized corpus reusing the SAME key, with rebuild=False (the normal reuse
    # path) -- must detect the incongruence and rebuild rather than trust the stale index.
    units_b = units_from_documents(_small_docs(8, "b"))
    pys_b = BM25Pyserini(index_root=index_root, rebuild=False).index(units_b, key=key)
    hits = set(pys_b.search("filler", k=20))
    assert hits, "rebuilt index should be searchable"
    assert hits <= {u.doc_id for u in units_b}, (
        "stale corpus_a doc_ids leaked through -- the doc-count congruence check did not "
        "trigger a rebuild")
    assert not (hits & {u.doc_id for u in units_a}), (
        "old corpus_a doc_ids still present in results after the corpus changed")
