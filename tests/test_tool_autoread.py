"""`research_bm25_autoread`: the retrieve-and-read baseline (`search_bm25` with `full_text=True`,
formerly `Bm25AutoRead`), between one-shot RAG (a single stuffed prompt, no agent loop) and the
SERP retrieve-then-visit baseline (research_bm25, a listing the agent must then choose to
visit). It uses the same plain BM25 ranking as the plain `search_bm25` listing (over the flat
doc title+body, same engine and fallback), but `search()` itself renders the full text of
every one of the top `AUTOREAD_TOPK` hits (capped at MAX_VISIT_TOKENS per doc, the same
`_cap_tokens` truncation and marker the whole-doc `visit` read uses). There is no visit/fetch
tool in this condition; a search call is the read.

No network: a tiny Lucene BM25 index (`lucene_support.build_pyserini`, built once per
corpus), plus a subprocess for the import-time env knobs (AUTOREAD_TOPK, MAX_VISIT_TOKENS).
The subprocess opens the index the parent already built; it never builds one itself.
"""
from __future__ import annotations

import os

from agent_search.corpus.units import units_from_documents
from agent_search.tools.base import EpisodeState, ToolBox
from agent_search.tools.search_bm25.tool import SearchBm25

from tests import lucene_support

lucene_support.require_jvm()

DOCS = [
    {"_id": "d_harbor", "title": "Harbor Festival",
     "text": "Harbor Festival is an annual event.\n\n## History\nFounded in 1897 by A. Smith.\n"
             "\n## Legacy\nStill held today."},
    {"_id": "d_flat", "title": "Adams-Onis Treaty",
     "text": "The Adams-Onis Treaty of 1819 concerned Florida."},
    {"_id": "65405", "title": "Integer Id Doc",
     "text": "A document whose id is an integer string."},
]

# a 12-doc corpus where every doc matches "common topic"; d00 ALSO carries 1500 extra unique
# filler tokens (word0..word1499) — well over MAX_VISIT_TOKENS's default (12000) — so the
# per-doc cap is exercised on a doc that is otherwise a normal, relevant hit.
_LONG_FILLER = " ".join(f"word{i}" for i in range(1500))
SERP_DOCS = [{"_id": f"d{i:02d}", "title": f"Common Topic {i}",
              "text": f"common topic document number {i}"
                      + (f" {_LONG_FILLER}" if i == 0 else "")}
             for i in range(12)]


def _units():
    return units_from_documents(DOCS)


def _serp_units():
    return units_from_documents(SERP_DOCS)


def _engine(units):
    return lucene_support.build_pyserini(units)


def _subprocess_engine_code(units) -> str:
    """Python source that opens the Lucene index the parent built for `units` (no build in
    the child), as `engine`."""
    lucene_support.build_pyserini(units)
    root, key = lucene_support.index_root(), lucene_support.corpus_key(units)
    return ("from agent_search.retrievers.lexical.pyserini import BM25Pyserini\n"
            f"engine = BM25Pyserini(index_root={root!r}).index(units, key={key!r})\n")


def _toolbox(units):
    state = EpisodeState(question="q")
    ubyid = {u.doc_id: u for u in units}
    search = SearchBm25(name="bm25_read_search", full_text=True).bind(
        state, units, ubyid, {"bm25": _engine(units)})
    return ToolBox([search], state)


# --- 1. tools: ONE tool only, no visit/fetch --------------------------------------------

def test_tools_tuple_is_bm25_read_search_only():
    box = _toolbox(_units())
    assert box.tools == ("bm25_read_search",)


# --- 2. search renders FULL TEXT of every top-k hit, not a listing ----------------------

def test_search_renders_full_text_not_a_listing():
    box = _toolbox(_units())
    out = box.run("bm25_read_search", {"query": "harbor festival annual event"})
    assert "d_harbor" in out
    assert "Founded in 1897 by A. Smith" in out    # the FULL body, not a snippet
    assert "Still held today" in out


def test_search_renders_exactly_autoread_topk_full_docs():
    import agent_search.tools.budgets as m
    box = _toolbox(_serp_units())
    out = box.run("bm25_read_search", {"query": "common topic"})
    # one full-text block per hit, header line not included
    n_blocks = len(out.split("\n\n")) - 1     # first "\n\n"-split chunk is the header line
    assert n_blocks == m.AUTOREAD_TOPK == 5
    # every rendered block is numbered 1..5, each naming a distinct doc_id from the corpus
    ids_in_corpus = {u["_id"] for u in SERP_DOCS}
    ranks_seen = []
    for block in out.split("\n\n")[1:]:
        header_line = block.splitlines()[0]
        rank_s, doc_id, _rest = header_line.strip().split(None, 2)
        assert doc_id in ids_in_corpus
        ranks_seen.append(int(rank_s))
    assert ranks_seen == [1, 2, 3, 4, 5]


def test_search_ignores_a_hallucinated_k_arg():
    box = _toolbox(_serp_units())
    baseline = box.run("bm25_read_search", {"query": "common topic"})
    widened = _toolbox(_serp_units()).run("bm25_read_search", {"query": "common topic", "k": 50})
    narrowed = _toolbox(_serp_units()).run("bm25_read_search", {"query": "common topic", "k": 1})
    assert widened == baseline
    assert narrowed == baseline
    assert baseline.count("word0 word1") <= 1   # sanity: still bounded, not exploded


# --- 3. per-doc token cap: SAME cap/marker as the whole-doc visit read ------------------

def test_per_doc_text_is_truncated_at_max_visit_tokens_with_same_marker():
    """MAX_VISIT_TOKENS is read once at import time (like AUTOREAD_TOPK) — pin it explicitly via
    a fresh subprocess so the truncation boundary is deterministic regardless of the ambient
    environment. d00's body is "common topic document number 0" plus 1500 filler words
    (word0..word1499); with MAX_VISIT_TOKENS=50, the cap keeps the first 50 MODEL tokens of the
    body. The exact word where that lands depends on the tokenizer (some of "word0".."word1499"
    are more than one model token each), so the expected boundary is computed here with the
    same `truncate_tokens` ruler the tool uses, rather than a hardcoded word count.
    """
    import subprocess
    import sys as _sys

    from agent_search.tokens import TRUNCATED, truncate_tokens

    filler = " ".join(f"word{i}" for i in range(1500))
    body = f"common topic document number 0 {filler}"

    kept = truncate_tokens(body, 50, TRUNCATED)
    assert kept.endswith(TRUNCATED)
    kept_words = kept[: -len(TRUNCATED)].split()
    last_kept_word = kept_words[-1]
    first_cut_word = filler.split()[filler.split().index(last_kept_word) + 1]

    code = (
        "from agent_search.tools.base import EpisodeState, ToolBox\n"
        "from agent_search.tools.search_bm25.tool import SearchBm25\n"
        "from agent_search.corpus.units import units_from_documents\n"
        "filler = ' '.join(f'word{i}' for i in range(1500))\n"
        "docs = [{'_id': 'd00', 'title': 'Common Topic 0',\n"
        "         'text': f'common topic document number 0 {filler}'}]\n"
        "units = units_from_documents(docs)\n"
        "ubyid = {u.doc_id: u for u in units}\n"
        + _subprocess_engine_code(units_from_documents([{"_id": "d00", "title": "Common Topic 0",
                                                          "text": body}])) +
        "state = EpisodeState(question='q')\n"
        "search = SearchBm25(name='bm25_read_search', full_text=True).bind(state, units, ubyid, {'bm25': engine})\n"
        "box = ToolBox([search], state)\n"
        "print(box.run('bm25_read_search', {'query': 'common topic'}))\n")

    env = {k: v for k, v in os.environ.items() if k != "MAX_VISIT_TOKENS"}
    env["MAX_VISIT_TOKENS"] = "50"
    result = subprocess.run([_sys.executable, "-c", code], cwd=os.getcwd(),
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "…(truncated — this is the whole-doc cap)" in out
    assert last_kept_word in out.split()          # last kept filler word
    assert first_cut_word not in out.split()      # truncated away


def test_short_docs_are_not_marked_truncated():
    box = _toolbox(_units())
    out = box.run("bm25_read_search", {"query": "harbor festival annual"})
    assert "truncated" not in out


# --- 4. NO visit/fetch tool: a clear error, not a silent fallback -----------------------

def test_visit_call_returns_a_clear_error():
    box = _toolbox(_units())
    box.run("bm25_read_search", {"query": "harbor"})
    out = box.run("visit", {"rank": 1})
    assert out.startswith("ERROR:")
    assert "no visit tool in this condition" in out
    assert "search already returns full documents" in out


def test_fetch_call_also_returns_the_same_clear_error():
    box = _toolbox(_units())
    out = box.run("fetch", {"specs": [[1, ""]]})
    assert out.startswith("ERROR:")
    assert "no visit tool in this condition" in out


def test_unknown_tool_name_gets_the_generic_error_not_the_visit_error():
    box = _toolbox(_units())
    out = box.run("frobnicate", {})
    assert out.startswith("ERROR: unknown tool")
    assert "bm25_read_search" in out


def test_run_aliases_bare_search():
    box = _toolbox(_units())
    out = box.run("search", {"query": "harbor festival annual event"})
    assert "Founded in 1897" in out


# --- 5. env AUTOREAD_TOPK: import-time resolution, fresh subprocess --------------------

def test_autoread_topk_env_default_five_and_env_three_in_a_fresh_process():
    import subprocess
    import sys as _sys

    docs = [{"_id": f"d{i:02d}", "title": f"Common Topic {i}",
             "text": f"common topic document number {i}"} for i in range(12)]
    code = (
        "from agent_search.tools.base import EpisodeState, ToolBox\n"
        "from agent_search.tools.search_bm25.tool import SearchBm25\n"
        "import agent_search.tools.budgets as b\n"
        "from agent_search.corpus.units import units_from_documents\n"
        "docs = [{'_id': f'd{i:02d}', 'title': f'Common Topic {i}',\n"
        "         'text': f'common topic document number {i}'} for i in range(12)]\n"
        "units = units_from_documents(docs)\n"
        "ubyid = {u.doc_id: u for u in units}\n"
        + _subprocess_engine_code(units_from_documents(docs)) +
        "state = EpisodeState(question='q')\n"
        "search = SearchBm25(name='bm25_read_search', full_text=True).bind(state, units, ubyid, {'bm25': engine})\n"
        "box = ToolBox([search], state)\n"
        "out = box.run('bm25_read_search', {'query': 'common topic'})\n"
        "n_blocks = len(out.split(chr(10)+chr(10))) - 1\n"
        "print(b.AUTOREAD_TOPK, n_blocks)\n")

    def _run(env_overrides):
        env = {k: v for k, v in os.environ.items() if k != "AUTOREAD_TOPK"}
        env.update(env_overrides)
        out = subprocess.run([_sys.executable, "-c", code], cwd=os.getcwd(),
                             env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        k, n = out.stdout.strip().split()
        return int(k), int(n)

    assert _run({}) == (5, 5)
    assert _run({"AUTOREAD_TOPK": "3"}) == (3, 3)


# --- 6. condition wiring: research_bm25_autoread resolves via the retriever registry ---------

def test_research_bm25_autoread_condition_loads_uncoached():
    from agent_search.strategies import CONDITIONS
    from agent_search.tasks.render import render_manuals

    p = CONDITIONS["research_bm25_autoread"]
    assert p.strategy.toolset_name == "bm25_autoread"
    assert p.tool_names == ("bm25_read_search",)
    # UNCOACHED like research_bm25: no manual renders for this toolset.
    assert render_manuals([t.manual_path("general") for t in p.strategy.tools]) == ""
    assert "term[field]" not in p.render()


def test_research_bm25_autoread_resolves_via_registry_as_bm25autoread_arm():
    from agent_search.evaluation.agent_runner import ConditionAgent
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    r = build_factory("agent_research_bm25_autoread", RetrieverConfig(policy="stub"))()
    assert isinstance(r, ConditionAgent)
    assert r.toolset == ("bm25_read_search",)
    assert r.tool == "agent_research_bm25_autoread"
    assert r.condition.name == "research_bm25_autoread"
    assert r.domain == "general"
    assert not r.needs_files


def test_bm25_autoread_workspace_builds_end_to_end():
    from agent_search.retrievers.registry import RetrieverConfig, build_factory

    cfg = RetrieverConfig(policy="stub", index_root=lucene_support.index_root())
    r = build_factory("agent_research_bm25_autoread", cfg)()
    r.index(_units(), key=lucene_support.corpus_key(_units()))
    ws = r.toolbox("harbor festival annual event")
    assert ws.tools == ("bm25_read_search",)
    assert isinstance(ws["bm25_read_search"], SearchBm25)
    assert ws["bm25_read_search"].full_text is True
    out = ws.run("bm25_read_search", {"query": "harbor festival annual event"})
    assert "Founded in 1897" in out


def test_existing_research_bm25_condition_is_unaffected():
    from agent_search.strategies import CONDITIONS

    p = CONDITIONS["research_bm25"]
    assert p.strategy.toolset_name == "research_bm25"
    assert p.tool_names == ("bm25_search", "visit")
