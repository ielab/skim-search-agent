"""The deep-research doc ACI: DocSearchFetch (search -> fetch a section) + Bm25Visit baseline.

search(query) returns ranked ARTICLES + their SECTION structure (NO bodies); fetch([rank,
section]) pulls a named section (or infobox). Sections are derived LIVE from the body's `##`
markers. The bm25 baseline is search + visit-the-whole-doc. Fetch references resolve by rank
OR doc_id/title (integer doc_ids must NOT be mis-read as ranks)."""
import os

from agent_search.agent.tools.doc_research import (Bm25Visit, DenseVisit, DocSearchFetch,
                                                   sections_from_body)
from agent_search.corpus.units import units_from_documents

# a structured doc (## markers in the body) + a flat doc (no markers), one integer-id doc.
DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\n## History\nFounded in 1897 by A. Smith.\n"
             "\n## Legacy\nStill held today.",
     "infobox": "Founded: 1897; Location: Portville"},
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
    {"_id": "65405", "title": "Integer Id Doc",
     "text": "A document whose id is an integer string."},
]


def _ws():
    return DocSearchFetch(units_from_documents(DOCS))


# --- sections derived live from ## markers ----------------------------------

def test_sections_from_body_splits_on_markers():
    secs = sections_from_body("Lead.\n\n## History\nh.\n\n## Legacy\nl.")
    assert list(secs) == ["(intro)", "History", "Legacy"]
    assert secs["History"] == "h." and secs["Legacy"] == "l."


def test_flat_body_is_one_intro_section():
    secs = sections_from_body("Just one paragraph, no headings.")
    assert list(secs) == ["(intro)"]


# --- explicit MATCHED sections (phase-0 structured corpus) -------------------
# A structured corpus (wikipedia's real `##` sections, browsecomp's TOC-inserted ones) ships
# `sections` as a matched, ordered list of {heading, text}. The unit must carry those exact parts
# and `fetch` must read the matched slice DIRECTLY — not re-derive it from `##` markers in a joined
# body (which loses the heading↔text pairing the corpus already established).

MATCHED_DOC = [{
    "_id": "d_bt", "title": "Blue Train",
    "sections": [
        {"heading": "(intro)", "text": "An album by John Coltrane."},
        {"heading": "Recording", "text": "Cut in 1957 at Van Gelder."},
        {"heading": "Reception", "text": "Widely praised as a landmark."},
    ],
}]


def test_explicit_sections_carry_to_unit_as_matched_pairs():
    u = units_from_documents(MATCHED_DOC)[0]
    assert u.sections == (("(intro)", "An album by John Coltrane."),
                          ("Recording", "Cut in 1957 at Van Gelder."),
                          ("Reception", "Widely praised as a landmark."))
    # the joined-heading `section` field is derived from the parts (for IN(section,·) + the listing)
    assert "Recording" in (u.section or "") and "Reception" in (u.section or "")


def test_fetch_uses_explicit_matched_section_not_rederived():
    ws = DocSearchFetch(units_from_documents(MATCHED_DOC))
    ws.search("blue[title]", k=5)
    out = ws.fetch([["d_bt", "Recording"]])
    assert "Cut in 1957 at Van Gelder" in out and "§Recording" in out
    out2 = ws.fetch([["d_bt", "Reception"]])
    assert "landmark" in out2 and "Van Gelder" not in out2   # parts stay separated, not merged


# --- search: structure table, no bodies -------------------------------------

def test_search_lists_sections_and_infobox_no_body():
    out = _ws().search("harbor[title]", k=5)
    assert "d_harbor" in out and "'Harbor Festival'" in out
    assert "History" in out and "Legacy" in out          # section names
    assert "Founded" in out                               # infobox key
    assert "A. Smith" not in out                          # NOT the body content


def test_search_zero_hits_hints():
    out = _ws().search("zzznotarealword[title]", k=5)
    assert "0 matches" in out


def test_zero_hit_search_preserves_prior_ranking():
    """REGRESSION: the skill coaches a 0-hit pivot/loosen; that must NOT wipe the last good
    hits from under a subsequent fetch. A 0-hit search keeps the prior non-empty ranking."""
    ws = _ws()
    ws.search("harbor[title]", k=5)                       # establishes last_hits (rank 1 = harbor)
    zero = ws.search("zzznotarealword[title]", k=5)       # a coached pivot that finds nothing
    assert "0 matches" in zero
    out = ws.fetch([[1, "History"]])                      # rank 1 STILL refers to the harbor hit
    assert "Founded in 1897" in out and "no prior search" not in out.lower()


def test_bm25_zero_hit_search_preserves_prior_ranking():
    """REGRESSION (baseline parity): a 0-hit bm25 search likewise keeps the prior ranking."""
    bw = Bm25Visit(units_from_documents(DOCS))
    bw.search("harbor festival annual event", k=5)        # establishes last_hits
    zero = bw.search("zzznotarealword", k=5)
    assert "0 matches" in zero
    visited = bw.visit(1)                                 # rank 1 STILL the harbor doc
    assert "Founded in 1897" in visited


# --- fetch: named section / infobox -----------------------------------------

def test_fetch_named_section_by_rank():
    ws = _ws()
    ws.search("harbor[title]", k=5)
    out = ws.fetch([[1, "History"]])
    assert "Founded in 1897" in out and "§History" in out


def test_fetch_infobox():
    ws = _ws()
    ws.search("harbor[title]", k=5)
    out = ws.fetch([[1, "infobox"]])
    assert "Founded=1897" in out


def test_fetch_bad_section_lists_available():
    ws = _ws()
    ws.search("harbor[title]", k=5)
    out = ws.fetch([[1, "Nonexistent"]])
    assert "no section" in out and "History" in out


def test_fetch_by_doc_id_directly():
    ws = _ws()
    ws.search("harbor[title]", k=5)               # establishes some last_hits
    out = ws.fetch([["d_flat", "(intro)"]])
    assert "Adams-Onis Treaty of 1819" in out


def test_fetch_integer_doc_id_is_not_mistaken_for_rank():
    """REGRESSION (the digit-as-rank bug): an integer doc_id that is NOT a valid rank must
    resolve as a doc_id, not error as an out-of-range rank."""
    ws = _ws()
    ws.search("harbor[title]", k=1)               # last_hits has length 1
    out = ws.fetch([["65405", "(intro)"]])        # 65405 is a doc_id, not rank 65405
    assert "integer string" in out and "out of range" not in out


def test_seen_tracks_surfaced_docs():
    ws = _ws()
    ws.search("harbor[title]", k=5)
    assert "d_harbor" in ws.seen


def test_surfaced_is_first_seen_order_across_two_searches():
    """`workspace.surfaced` (agent/loop.py's retrieval ranking, read via `getattr(workspace,
    "surfaced", [])`) must list doc_ids in FIRST-SEEN order — the order a `search`/`fetch`
    call actually surfaced them in, not set iteration order. A second search that resurfaces
    an already-seen doc must not move it: `surfaced` is append-only, `OrderedSeen`-backed."""
    ws = _ws()
    ws.search("treaty[title]", k=5)                 # surfaces d_flat first
    ws.search("harbor[title]", k=5)                 # then d_harbor
    assert ws.surfaced == ["d_flat", "d_harbor"]
    # re-surfacing d_flat via fetch must not reorder it to the end.
    ws.fetch([["d_flat", "(intro)"]])
    assert ws.surfaced == ["d_flat", "d_harbor"]


# --- bm25 baseline: search + visit the whole doc ----------------------------

def test_bm25_search_then_visit_whole_doc():
    bw = Bm25Visit(units_from_documents(DOCS))
    out = bw.search("harbor festival annual event", k=5)
    assert "d_harbor" in out
    visited = bw.visit(1)
    assert "Founded in 1897" in visited           # the WHOLE doc, not a section


def test_bm25_visit_integer_doc_id_not_mistaken_for_rank():
    bw = Bm25Visit(units_from_documents(DOCS))
    bw.search("harbor", k=1)
    out = bw.visit("65405")
    assert "integer string" in out and "out of range" not in out


# =============================================================================================
# FAIRNESS: MAX_SECTION_TOKENS defaulted to 180 while
# MAX_VISIT_TOKENS defaulted to 12000 — a 6.7x read-budget gap on the exact same module's two
# read modes (a bigger gap still, 66x, against runs that raised MAX_VISIT_TOKENS to 12000
# without touching MAX_SECTION_TOKENS). 96.6% of real BQL `fetch` calls truncated as a result.
# The factorial's design intent is that fetch vs visit differ in WHAT is read (a named part vs
# the whole doc), not in HOW MUCH can be read — so `fetch`'s per-section budget must default to
# parity with `visit`'s whole-doc budget, while staying independently env-overridable.
# =============================================================================================

def test_max_section_tokens_env_resolution_fresh_process():
    """Import-time env resolution, run in a fresh subprocess (MAX_SECTION_TOKENS/MAX_VISIT_TOKENS
    are read once as module-level constants at import — see test_configs.py's
    test_config_json_env_knobs_reflect_set_env_var docstring for why reimporting in-process is
    unsafe here; same subprocess pattern as test_dense_baseline.py's
    test_default_max_tokens_env_knob_in_a_fresh_process).

    Three cases:
      1. neither var set -> both resolve to the code default, 12000 (parity).
      2. only MAX_VISIT_TOKENS overridden -> MAX_SECTION_TOKENS tracks it (still parity) —
         this is the "resolved value" behavior the fix requires, not a second hardcoded 12000.
      3. both vars independently overridden -> MAX_SECTION_TOKENS keeps ITS OWN value, proving
         the env override is not simply squashed by the parity default.
    """
    import subprocess
    import sys as _sys

    code = ("import agent_search.agent.tools.doc_research as m; "
            "print(m.MAX_SECTION_TOKENS, m.MAX_VISIT_TOKENS)")

    def _run(env_overrides):
        env = {k: v for k, v in os.environ.items()
                if k not in ("MAX_SECTION_TOKENS", "MAX_VISIT_TOKENS")}
        env.update(env_overrides)
        out = subprocess.run([_sys.executable, "-c", code], cwd=os.getcwd(),
                             env=env, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        section, visit = out.stdout.strip().split()
        return int(section), int(visit)

    # 1. both unset -> code defaults, and parity holds.
    section, visit = _run({})
    assert (section, visit) == (12000, 12000)

    # 2. only MAX_VISIT_TOKENS overridden -> MAX_SECTION_TOKENS follows the RESOLVED value.
    section, visit = _run({"MAX_VISIT_TOKENS": "500"})
    assert (section, visit) == (500, 500)

    # 3. both independently overridden -> MAX_SECTION_TOKENS keeps its own env value.
    section, visit = _run({"MAX_VISIT_TOKENS": "500", "MAX_SECTION_TOKENS": "50"})
    assert (section, visit) == (50, 500)


def test_fetch_of_a_long_flat_section_is_not_truncated_at_the_old_180_cap():
    """REGRESSION for the fairness fix: browsecomp-style flat docs (one `(intro)` section)
    with a body well over the OLD 180-token cap but under
    the parity-scale (12000-token) budget must now come back whole, not truncated — otherwise a
    `fetch("")` on a flat doc is silently still a lossy whole-doc read, defeating the fix."""
    long_body = " ".join(f"word{i}" for i in range(300))     # > 180, < 12000
    ws = DocSearchFetch(units_from_documents(
        [{"_id": "d_long_flat", "title": "Long Flat Doc", "text": long_body}]))
    ws.search("word0[title]", k=5)
    out = ws.fetch([[1, ""]])
    assert "truncated" not in out
    assert "word299" in out


# =============================================================================================
# SERP LISTING DEPTH (BM25_VISIT_TOPK / DENSE_VISIT_TOPK): the retrieve-then-visit baselines'
# search listing shows a FIXED number of results per query. tools.yaml's bm25_search/
# bm25q_search/dense_search schemas expose ONLY `query` — no `k` — so the env knob is the ONLY
# control: default 5 (byte-identical to the old hardcoded listing), settable to e.g. 10 for a
# k=10 listing family, and NEVER resizable by a hallucinated `k` in the model's tool-call args.
# =============================================================================================

SERP_DOCS = [{"_id": f"d{i:02d}", "title": f"Common Topic {i}",
              "text": f"common topic document number {i}"} for i in range(12)]


def _listing_count(out: str) -> int:
    """Number of listed hits in a Bm25Visit/DenseVisit SERP rendering (lines minus header)."""
    lines = out.strip().splitlines()
    assert lines and lines[0].startswith("search:"), out
    return len(lines) - 1


class _StubDenseEngine:
    """CPU-only stand-in exposing ONLY what DenseVisit calls (`top_k_doc_ids(query, k)`) —
    the same injection pattern test_dense_baseline.py uses; no torch import."""

    def __init__(self, ids):
        self._ids = list(ids)

    def top_k_doc_ids(self, query, k=None):
        return list(self._ids[: (k or len(self._ids))])


def test_serp_listing_default_five_and_env_ten_in_a_fresh_process():
    """Import-time env resolution for BM25_VISIT_TOPK / DENSE_VISIT_TOPK, run end-to-end
    against the RENDERED listing in a fresh subprocess (the constants are read once at module
    import — same subprocess pattern as test_max_section_tokens_env_resolution_fresh_process).

    Three cases over a 12-doc corpus where every doc matches the query:
      1. neither var set -> both constants default to 5 and both listings render exactly 5 hits.
      2. BM25_VISIT_TOPK=10 -> the bm25 listing renders 10; the dense listing stays 5
         (independent knobs, not one shared constant).
      3. DENSE_VISIT_TOPK=10 -> the dense listing renders 10; the bm25 listing stays 5.
    """
    import subprocess
    import sys as _sys

    code = (
        "import agent_search.agent.tools.doc_research as m\n"
        "from agent_search.corpus.units import units_from_documents\n"
        "docs = [{'_id': f'd{i:02d}', 'title': f'Common Topic {i}',\n"
        "         'text': f'common topic document number {i}'} for i in range(12)]\n"
        "units = units_from_documents(docs)\n"
        "bm_out = m.Bm25Visit(units).run('bm25_search', {'query': 'common topic'})\n"
        "class Stub:\n"
        "    def top_k_doc_ids(self, query, k=None):\n"
        "        ids = [u.doc_id for u in units]\n"
        "        return ids[: (k or len(ids))]\n"
        "d_out = m.DenseVisit(units, engine=Stub()).run('dense_search', {'query': 'common topic'})\n"
        "print(m.BM25_VISIT_TOPK, m.DENSE_VISIT_TOPK,\n"
        "      len(bm_out.strip().splitlines()) - 1, len(d_out.strip().splitlines()) - 1)\n")

    def _run(env_overrides):
        env = {k: v for k, v in os.environ.items()
               if k not in ("BM25_VISIT_TOPK", "DENSE_VISIT_TOPK")}
        env.update(env_overrides)
        out = subprocess.run([_sys.executable, "-c", code], cwd=os.getcwd(),
                             env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        bm_k, d_k, bm_n, d_n = out.stdout.strip().split()
        return int(bm_k), int(d_k), int(bm_n), int(d_n)

    # 1. both unset -> code defaults (5), and the rendered listings show exactly 5 hits.
    assert _run({}) == (5, 5, 5, 5)
    # 2. BM25_VISIT_TOPK=10 -> the bm25 listing renders 10; dense is untouched (5).
    assert _run({"BM25_VISIT_TOPK": "10"}) == (10, 5, 10, 5)
    # 3. DENSE_VISIT_TOPK=10 -> the dense listing renders 10; bm25 is untouched (5).
    assert _run({"DENSE_VISIT_TOPK": "10"}) == (5, 10, 5, 10)


def test_bm25_serp_listing_ignores_a_hallucinated_k_arg():
    """tools.yaml's bm25_search schema exposes ONLY `query`; a hallucinated `k` in the model's
    tool-call args must NOT resize the SERP listing — depth is the env knob's alone."""
    import agent_search.agent.tools.doc_research as m
    bw = Bm25Visit(units_from_documents(SERP_DOCS))
    baseline = _listing_count(bw.run("bm25_search", {"query": "common topic"}))
    assert baseline == m.BM25_VISIT_TOPK          # the resolved knob (5 unless env-overridden)
    assert _listing_count(bw.run("bm25_search", {"query": "common topic", "k": 2})) == baseline
    assert _listing_count(bw.run("bm25_search", {"query": "common topic", "k": 50})) == baseline


def test_dense_serp_listing_ignores_a_hallucinated_k_arg():
    """The dense twin of the bm25 test above (dense_search's schema likewise exposes only
    `query`; DENSE_VISIT_TOPK alone sets the depth)."""
    import agent_search.agent.tools.doc_research as m
    units = units_from_documents(SERP_DOCS)
    dv = DenseVisit(units, engine=_StubDenseEngine([u.doc_id for u in units]))
    baseline = _listing_count(dv.run("dense_search", {"query": "common topic"}))
    assert baseline == m.DENSE_VISIT_TOPK
    assert _listing_count(dv.run("dense_search", {"query": "common topic", "k": 2})) == baseline
    assert _listing_count(dv.run("dense_search", {"query": "common topic", "k": 50})) == baseline
