"""Search-Fetch: a live search that lists structure, then fetching a named section.

Three query engines (bm25, dense, hybrid) pair their retrieval with the shared structured
`fetch` (agent_search.tools.fetch) instead of a whole-doc `visit`. The listing shows each
hit's section names and infobox keys (plus, for the *_snip arms, a best-matching excerpt),
and `fetch` pulls one named section. `search_fetch_bm25_plain`/`search_fetch_dense_plain`
are the excerpt-free siblings for the two arms that have one; hybrid has no plain sibling in
the paper's grid, so only its snippet arm is registered.
"""
from agent_search.snippets import NoSnippet, TermWindow
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_dense.tool import SearchDense
from agent_search.tools.search_hybrid.tool import SearchHybrid

search_fetch = register_strategy(Strategy(
    name="search_fetch", toolset_name="bm25_fetch_snip",
    description="search (BM25) then fetch a named section, listing carries an excerpt",
    tools=(SearchBm25(name="bm25_search_snip", structure=True, snippet=TermWindow()), Fetch(name="fetch"))))

search_fetch_dense = register_strategy(Strategy(
    name="search_fetch_dense", toolset_name="dense_fetch",
    description="search (dense) then fetch a named section, listing carries an excerpt",
    tools=(SearchDense(name="dense_search_f", structure=True, snippet=TermWindow()), Fetch(name="fetch"))))

search_fetch_hybrid = register_strategy(Strategy(
    name="search_fetch_hybrid", toolset_name="hybrid_fetch_snip",
    description="search (BM25 and dense fused by RRF) then fetch a named section, listing carries an excerpt",
    tools=(SearchHybrid(name="hybrid_search_snip", structure=True, snippet=TermWindow()), Fetch(name="fetch"))))

# the plain (no-excerpt) siblings: same retrieval and fetch, a bare structure listing.
search_fetch_bm25_plain = register_strategy(Strategy(
    name="search_fetch_bm25_plain",
    description="search (BM25) then fetch a named section, no excerpt in the listing",
    tools=(SearchBm25(name="bm25_search", structure=True, snippet=NoSnippet()), Fetch(name="fetch"))))

search_fetch_dense_plain = register_strategy(Strategy(
    name="search_fetch_dense_plain",
    description="search (dense) then fetch a named section, no excerpt in the listing",
    tools=(SearchDense(name="dense_search_fp", structure=True, snippet=NoSnippet()), Fetch(name="fetch"))))
