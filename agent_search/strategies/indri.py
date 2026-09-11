"""Indri: graded belief search, paired with either a structured fetch or a whole-doc visit.

`indri` (isearch_s/fetch) is the registered paper condition: the same graded Indri ranking
as `indri_plain`, with a listing excerpt, paired with the shared structured `fetch`
(agent_search.tools.fetch). `indri_plain` (isearch/fetch) is the same pairing with the
excerpt left off. `indri_visit` (isearch_v/visit_v) crosses the same graded search (excerpt
forced on, for fairness parity with the bm25 baseline's opening-snippet listing) with a
whole-doc `visit` instead, disentangling the search engine from the read granularity.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_indri.tool import SearchIndri
from agent_search.tools.visit.tool import Visit

indri = register_strategy(Strategy(
    name="indri", toolset_name="indri_snip",
    description="search (Indri graded query language, with excerpt) then fetch a named section",
    tools=(SearchIndri(name="isearch_s", snippets=True), Fetch(name="fetch"))))

indri_plain = register_strategy(Strategy(
    name="indri_plain",
    description="search (Indri graded query language) then fetch a named section",
    tools=(SearchIndri(name="isearch"), Fetch(name="fetch"))))

indri_visit = register_strategy(Strategy(
    name="indri_visit",
    description="search (Indri graded query language, with excerpt) then read the whole document",
    tools=(SearchIndri(name="isearch_v", snippets=True), Visit(name="visit_v"))))
