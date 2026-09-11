"""`BM25Pyserini` (agent_search.retrievers.lexical.pyserini): the one BM25 engine, canonical
Lucene BM25 (Porter stemming plus stopwords, k1=0.9/b=0.4, the engine SWE-bench's own BM25
baseline uses). `build_bm25_engine` and `Engines.bm25()` return it for every corpus; there is
no in-memory alternative and no backend switch.

Pinned here:

  1. index build   the index is built once per corpus under `index_root/bm25_pyserini/<key>/`,
                   reused on the next `index()` for the same corpus, and rebuilt when the corpus
                   under that key changed (doc count or content fingerprint, `_is_built`).
  2. search        `search` returns real doc ids; `search_scored` pairs them with descending
                   Lucene scores; morphological variants match through the Porter stemmer.
  3. selection     `build_bm25_engine` and `ConditionAgent.index()` (every bm25-family arm, via
                   `Engines`) hand out a `BM25Pyserini`.
  4. BM25_INDEX_PATH
                   a prebuilt Lucene directory is opened as is and nothing is built under
                   `index_root`; a path that is not an index is refused.
  5. offline       `search()` opens no socket, and the module imports in a clean subprocess with
                   no `OPENAI_API_KEY` (pyserini's transitive import-time landmine).
  6. listing       `search_bm25` renders a well-formed ranked listing over it, and `visit` reads
                   a ranked hit.
  7. build knobs   thread count and stored-field knobs, one corpus shard per indexing thread,
                   a lean index with no stored raw text.

Needs a real JVM and `pyserini`; the module skips without them (`lucene_support.require_jvm`).
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import subprocess
import sys

import pytest

from tests import lucene_support

lucene_support.require_jvm()

from agent_search.corpus.units import units_from_documents  # noqa: E402
from agent_search.evaluation.agent_runner import ConditionAgent  # noqa: E402
from agent_search.retrievers.lexical import build_bm25_engine  # noqa: E402
from agent_search.retrievers.lexical.pyserini import BM25Pyserini  # noqa: E402
from agent_search.strategies.base import STRATEGIES  # noqa: E402
from agent_search.strategies.conditions import Condition  # noqa: E402
from agent_search.tasks.base import TASKS  # noqa: E402
from agent_search.tools.base import EpisodeState, ToolBox  # noqa: E402
from agent_search.tools.search_bm25.tool import SearchBm25  # noqa: E402
from agent_search.tools.visit.tool import Visit  # noqa: E402


def _cond(strategy_name: str) -> Condition:
    """An ad-hoc research condition over a registered strategy, without touching the global
    `CONDITIONS` registry (`ConditionAgent` accepts a `Condition` object directly)."""
    return Condition(name=strategy_name, task=TASKS["research"], strategy=STRATEGIES[strategy_name])


def _bm25_visit_toolbox(units, engine):
    """`bm25_search`/`visit`: the plain search-then-visit tool pair, bound directly to `engine`."""
    state = EpisodeState(question="q")
    ubyid = {u.doc_id: u for u in units}
    search = SearchBm25(name="bm25_search").bind(state, units, ubyid, {"bm25": engine})
    visit = Visit(name="visit").bind(state, units, ubyid, {})
    return ToolBox([search, visit], state)


# --- a small deterministic corpus: verb lemmas in several morphological forms, tagged with
# proper-noun place names, 15 docs per tag (120 docs). The forms are what the stemming test
# looks at; the size is what the sharding test looks at. ------------------------------------
_LEMMAS = ["climb", "fish", "sail", "study", "teach", "build", "paint", "dance", "write",
           "plan", "design", "grow", "cook", "sing", "play", "travel", "explore", "research"]
_FORMS = ["{0}", "{0}ing", "{0}ed", "{0}s"]
_TAGS = ["Kestrel Valley", "Ashwood Fields", "Marrow Bay", "Thistledown Ridge",
         "Copperlake District", "Windmere Basin", "Silverpine Hollow", "Blackthorn Pass"]
_STOP = ["the", "a", "an", "of", "in", "on", "at", "with", "near", "along", "through",
         "over", "under", "between", "during", "and", "but", "for", "to"]
_ADJ = ["ancient", "remote", "vast", "rugged", "quiet", "hidden", "coastal", "historic"]
_DOCS_PER_TAG = 15


def _make_corpus() -> list[dict]:
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
    return docs


@pytest.fixture(scope="module")
def corpus():
    """The corpus, its units, the engine and the key of its index under the shared test index
    root. Built once for the module: a Lucene build is a JVM start plus an indexing subprocess."""
    docs = _make_corpus()
    units = units_from_documents(docs)
    assert len(units) == len(_TAGS) * _DOCS_PER_TAG
    pys = lucene_support.build_pyserini(units)
    return docs, units, {u.doc_id: u for u in units}, pys, lucene_support.corpus_key(units)


_QUERY = "climbing Kestrel Valley"


# --- 1. index build and reuse ------------------------------------------------------------------

def test_index_is_built_under_index_root_and_is_cached(corpus):
    _docs, _units, _ubyid, _pys, key = corpus
    base = os.path.join(lucene_support.index_root(), "bm25_pyserini", key)
    assert os.path.isdir(os.path.join(base, "lucene"))
    with open(os.path.join(base, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["n_docs"] == len(_units) and meta.get("corpus_fingerprint")
    probe = BM25Pyserini(index_root=lucene_support.index_root())
    assert probe.is_cached(key)
    assert not probe.is_cached("no_such_key")


def test_second_index_call_reuses_the_build(corpus, monkeypatch):
    """`index()` for the same corpus and key opens the existing index; it never runs the
    indexing subprocess again."""
    _docs, units, _ubyid, _pys, key = corpus

    def _no_build(*a, **k):
        raise AssertionError("index() re-ran the pyserini indexer on an up-to-date index")

    monkeypatch.setattr(subprocess, "run", _no_build)
    eng = BM25Pyserini(index_root=lucene_support.index_root()).index(units, key=key)
    assert eng.search(_QUERY, k=5)


def _small_docs(n: int, prefix: str = "d") -> list:
    return [{"_id": f"{prefix}{i}", "title": f"{prefix}{i}", "text": f"filler content {i}"}
            for i in range(n)]


def test_is_built_rejects_stale_doc_count_and_rebuilds(tmp_path):
    """An index left on disk under a key whose corpus later changed (docs added or removed)
    must not be reused. `index()` writes a `meta.json` doc-count sentinel and cross-checks it
    on every later `index()` for the same key; a mismatch is a logged rebuild (the same guard
    as the dense cache in retrievers/dense/base.py)."""
    index_root = str(tmp_path)
    key = "congruence_test_corpus"

    units_a = units_from_documents(_small_docs(5, "a"))
    pys_a = BM25Pyserini(index_root=index_root, rebuild=True).index(units_a, key=key)
    hits_a = set(pys_a.search("filler", k=10))
    assert hits_a and hits_a <= {u.doc_id for u in units_a}  # sanity: searchable at all
    meta_path = os.path.join(index_root, "bm25_pyserini", key, "meta.json")
    assert os.path.exists(meta_path)

    # A different-sized corpus reusing the same key with rebuild=False (the normal reuse
    # path) must detect the incongruence and rebuild rather than trust the stale index.
    units_b = units_from_documents(_small_docs(8, "b"))
    pys_b = BM25Pyserini(index_root=index_root, rebuild=False).index(units_b, key=key)
    hits = set(pys_b.search("filler", k=20))
    assert hits, "rebuilt index should be searchable"
    assert hits <= {u.doc_id for u in units_b}, (
        "stale corpus_a doc_ids leaked through: the doc-count congruence check did not "
        "trigger a rebuild")
    assert not (hits & {u.doc_id for u in units_a}), (
        "old corpus_a doc_ids still present in results after the corpus changed")


def test_is_built_rejects_stale_fingerprint_same_doc_count_and_rebuilds(tmp_path):
    """A unit's content can change in place (same doc_id, same count); the doc-count check
    alone cannot see that. `index()` also writes the `corpus_fingerprint`
    (agent_search.corpus.fingerprint) into meta.json and cross-checks it, so a same-count,
    different-content reuse still rebuilds instead of serving the old postings."""
    index_root = str(tmp_path)
    key = "fingerprint_congruence_test"

    units_a = units_from_documents([{"_id": "d0", "title": "d0", "text": "alpha content zero"}])
    pys_a = BM25Pyserini(index_root=index_root, rebuild=True).index(units_a, key=key)
    assert set(pys_a.search("alpha", k=10)) == {"d0"}
    meta_path = os.path.join(index_root, "bm25_pyserini", key, "meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert meta.get("corpus_fingerprint")          # the key is actually being written

    # same doc_id, same count, different text: a doc-count check alone would trust this.
    units_b = units_from_documents(
        [{"_id": "d0", "title": "d0", "text": "totally different beta wording"}])
    pys_b = BM25Pyserini(index_root=index_root, rebuild=False).index(units_b, key=key)
    assert pys_b.search("alpha", k=10) == [], (
        "stale content served after an in-place edit: the fingerprint check did not "
        "trigger a rebuild")
    assert set(pys_b.search("beta", k=10)) == {"d0"}


# --- 2. search and search_scored ------------------------------------------------------------

def test_search_returns_real_doc_ids(corpus):
    _docs, _units, ubyid, pys, _key = corpus
    ids = pys.search(_QUERY, k=5)
    assert ids and len(ids) == 5
    assert all(i in ubyid for i in ids)
    assert len(set(ids)) == len(ids)


def test_search_scored_pairs_the_same_ranking_with_descending_scores(corpus):
    _docs, _units, _ubyid, pys, _key = corpus
    scored = pys.search_scored(_QUERY, k=5)
    assert [d for d, _ in scored] == pys.search(_QUERY, k=5)
    scores = [s for _, s in scored]
    assert all(isinstance(s, float) and s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_porter_stemming_matches_every_morphological_form(corpus):
    """Lucene's analyzer stems, so `climbing` reaches the docs that say `climbed`, `climbs`
    or `climb`: exactly the docs holding any form, no others."""
    docs, _units, _ubyid, pys, _key = corpus
    form = re.compile(r"\bclimb(?:ing|ed|s)?\b", re.IGNORECASE)
    expected = {d["_id"] for d in docs if form.search(d["text"])}
    assert expected and any(not re.search(r"\bclimbing\b", d["text"]) for d in docs
                            if d["_id"] in expected)        # some hold only another form
    assert set(pys.search("climbing", k=len(docs))) == expected
    assert set(pys.search("climbed", k=len(docs))) == expected


def test_no_match_query_returns_empty(corpus):
    _docs, _units, _ubyid, pys, _key = corpus
    assert pys.search("zzzz_nonexistent_qqqq", k=5) == []
    assert pys.search_scored("zzzz_nonexistent_qqqq", k=5) == []


# --- 3. selection: every path hands out BM25Pyserini --------------------------------------------

def test_build_bm25_engine_returns_pyserini(corpus):
    _docs, units, _ubyid, _pys, key = corpus
    eng = build_bm25_engine(units, index_root=lucene_support.index_root(), key=key)
    assert isinstance(eng, BM25Pyserini)
    assert eng.search(_QUERY, k=5)


@pytest.mark.parametrize("strategy_name", [
    "search_visit",              # bm25 + visit
    "search_fetch_bm25_plain",   # bm25 + fetch
    "search_visit_snippets",     # bm25q + visit
    "search_fetch",              # bm25 snip + fetch
])
def test_condition_agent_bm25_family_arms_get_lucene(corpus, strategy_name):
    """The real per-episode construction site (`ConditionAgent.index()`): every bm25-family
    arm's `bm25` engine is `BM25Pyserini`, whatever the arm, since all of them go through the
    same `build_bm25_engine` call via `Engines`."""
    _docs, units, _ubyid, _pys, key = corpus
    r = ConditionAgent(_cond(strategy_name), lambda: None,
                       index_root=lucene_support.index_root()).index(units, key=key)
    assert isinstance(r.engines.get("bm25"), BM25Pyserini)


# --- 4. BM25_INDEX_PATH: a prebuilt Lucene directory, opened as is ----------------------------

def test_bm25_index_path_opens_a_prebuilt_index_and_builds_nothing(corpus, tmp_path, monkeypatch):
    _docs, units, ubyid, _pys, key = corpus
    lucene_dir = os.path.join(lucene_support.index_root(), "bm25_pyserini", key, "lucene")
    monkeypatch.setenv("BM25_INDEX_PATH", lucene_dir)

    def _no_build(*a, **k):
        raise AssertionError("BM25_INDEX_PATH is set; nothing may be indexed")

    monkeypatch.setattr(subprocess, "run", _no_build)
    eng = BM25Pyserini(index_root=str(tmp_path)).index(units, key="ignored_key")
    assert not os.path.exists(os.path.join(str(tmp_path), "bm25_pyserini"))
    ids = eng.search(_QUERY, k=5)
    assert ids and all(i in ubyid for i in ids)


def test_bm25_index_path_that_is_not_an_index_is_refused(corpus, tmp_path, monkeypatch):
    _docs, units, _ubyid, _pys, _key = corpus
    bogus = tmp_path / "not_an_index"
    bogus.mkdir()
    monkeypatch.setenv("BM25_INDEX_PATH", str(bogus))
    with pytest.raises(RuntimeError, match="not a Lucene index directory"):
        BM25Pyserini(index_root=str(tmp_path)).index(units, key="k")


# --- 5. offline safety --------------------------------------------------------------------------

def test_search_makes_no_python_level_network_call(corpus, monkeypatch):
    _docs, _units, _ubyid, pys, _key = corpus

    def _blocked(*a, **k):
        raise AssertionError("search() attempted to open a socket; it must be local only")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    try:
        ids = pys.search(_QUERY, k=5)
    finally:
        monkeypatch.undo()
    assert ids  # the search still worked with sockets blocked


def test_import_succeeds_with_no_openai_api_key_in_a_clean_subprocess():
    """Regression for the transitive-import landmine pyserini.py's module docstring
    documents: `pyserini.search.lucene` imports `pyserini.encode._openai`, which raises at
    import time if OPENAI_API_KEY is unset. The module sets its own placeholder around the
    two pyserini imports, so importing it directly on a machine with no key works. A real
    subprocess with OPENAI_API_KEY/GEMINI_API_KEY stripped is the only way to prove it (the
    parent test process may already have one set)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENAI_API_KEY", "GEMINI_API_KEY")}
    proc = subprocess.run(
        [sys.executable, "-c",
         "from agent_search.retrievers.lexical.pyserini import BM25Pyserini; "
         "print('OK', BM25Pyserini().name)"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, (
        f"import failed with no OPENAI_API_KEY set: stdout={proc.stdout!r} "
        f"stderr={proc.stderr[-2000:]!r}")
    assert "OK bm25_pyserini" in proc.stdout


# --- 6. listing: search_bm25 over the engine, visit of a ranked hit ---------------------------

_LISTING_RE = re.compile(
    r"^search: .+   \(\d+ matches\):\n(  \d+  \S+  '.*'  .*…\n?)+$")


def test_listing_is_well_formed(corpus):
    _docs, units, _ubyid, pys, _key = corpus
    out = _bm25_visit_toolbox(units, pys).run("bm25_search", {"query": _QUERY})
    assert _LISTING_RE.match(out), f"listing didn't match the expected shape:\n{out}"
    lines = out.splitlines()
    assert lines[0].startswith(f"search: {_QUERY}")
    assert len(lines) == 1 + len(pys.search(_QUERY, k=len(lines) - 1))


def test_listing_ranks_are_1_indexed_and_sequential(corpus):
    _docs, units, _ubyid, pys, _key = corpus
    out = _bm25_visit_toolbox(units, pys).run("bm25_search", {"query": _QUERY})
    ranks = [int(m.group(1)) for m in re.finditer(r"^  (\d+)  ", out, re.MULTILINE)]
    assert ranks and ranks == list(range(1, len(ranks) + 1))


def test_visit_whole_doc_read_works(corpus):
    """`visit` never touches the engine, but end to end: a doc surfaced by the search is
    visitable."""
    _docs, units, _ubyid, pys, _key = corpus
    ws = _bm25_visit_toolbox(units, pys)
    ws.run("bm25_search", {"query": _QUERY})
    assert ws.last_hits
    out = ws.run("visit", {"rank": 1})
    assert not out.startswith("ERROR")
    assert "Kestrel Valley" in out


# --- 7. build efficiency knobs: parallel indexing and lean stored fields -----------------------

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


def test_build_shards_corpus_one_jsonl_per_thread(corpus):
    """Anserini's JsonCollection parallelizes across files, so the build must write the corpus
    as several shards (one per thread), not one docs.jsonl, or `--threads N` is silently
    single-threaded. The module fixture built its index with the default thread count."""
    _docs, units, _ubyid, _pys, key = corpus
    corpus_dir = os.path.join(lucene_support.index_root(), "bm25_pyserini", key, "corpus")
    shards = sorted(f for f in os.listdir(corpus_dir) if f.endswith(".jsonl"))
    assert shards, "no corpus shards written"
    assert "docs.jsonl" not in shards, "legacy single-file layout: sharding not applied"
    assert len(shards) <= max(1, os.cpu_count() or 1)
    if (os.cpu_count() or 1) > 1:
        assert len(shards) > 1, (
            "corpus written as one shard on a multi-core machine: the parallel build "
            "regressed to single-threaded ingestion")
    # every doc is in exactly one shard (no loss, no duplication across the split)
    n_lines = 0
    for s in shards:
        with open(os.path.join(corpus_dir, s)) as fh:
            n_lines += sum(1 for _ in fh)
    assert n_lines == len(units)


def test_lean_index_has_no_stored_raw_but_searches_fine(corpus):
    """The default build stores no raw contents or docvectors (`doc(id).raw()` is empty), yet
    ranking works: BM25 needs only the inverted index, and the tool layer renders from its
    own units."""
    _docs, _units, _ubyid, pys, _key = corpus
    ids = pys.search(_QUERY, k=3)
    assert ids
    d = pys._searcher.doc(ids[0])
    assert d is None or not d.raw()
