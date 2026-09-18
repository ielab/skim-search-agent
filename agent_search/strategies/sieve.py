"""Sieve: Boolean structural search that lists structure, then a named-section fetch.

One `search_bql` and one `fetch` per variant. The search lists each candidate's section
names and infobox keys (no bodies); the fetch pulls one named section. Options on the search
decide the ranking model behind the Boolean filter (`bm25`, `fused` = BM25 and the dense
model by RRF, `dense` only), whether a query-biased snippet is shown per candidate, and the
manual the agent reads. The tool names are the ones the paper prompts used.
"""
from agent_search.snippets import TermWindow
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.fetch.tool import Fetch
from agent_search.tools.search_bql.tool import SearchBql
from agent_search.tools.visit.tool import Visit

# the method: Boolean search with snippets, BM25 ranking
sieve_bm25 = register_strategy(Strategy(
    name="sieve_bm25", toolset_name="search_fetch_s",
    description="Sieve: Boolean search with snippets (BM25 ranking) then fetch a section",
    tools=(SearchBql(name="search_s", snippet=TermWindow()), Fetch(name="fetch_s"))))

# the method with the dense model fused into the ranking (the paper's headline Sieve)
sieve = register_strategy(Strategy(
    name="sieve", toolset_name="bql_dense_snip",
    description="Sieve: Boolean search with snippets (BM25 and dense fused) then fetch a section",
    tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow()), Fetch(name="fetch_bqlds"))))

# the headline Sieve with the paper's BrowseComp manual verbatim (it describes an unsegmented
# corpus: no sections, fetch the body); the default manual above describes the sectioned corpus
sieve_paper_manual = register_strategy(Strategy(
    name="sieve_paper_manual", toolset_name="bql_dense_snip",
    description="Sieve (BM25 and dense fused) with the paper's BrowseComp manual verbatim",
    tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), manual_set="paper"), Fetch(name="fetch_bqlds"))))

# the manual ablation on the headline Sieve: what the agent is told about the query language.
# nomanual: a 21-word stub; syntax: the reference only (mechanics, fields, fetch, worked examples);
# noconstruct: the full manual without its query-construction advice
sieve_nomanual = register_strategy(Strategy(
    name="sieve_nomanual", toolset_name="bql_dense_snip",
    description="Sieve (BM25 and dense fused) with no manual, the tool declarations only",
    tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), manual_set="nomanual"), Fetch(name="fetch_bqlds"))))

sieve_card = register_strategy(Strategy(
    name="sieve_card", toolset_name="bql_dense_snip",
    description="Sieve (BM25 and dense fused) with a card: the query language, fields and fetch call only",
    tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), manual_set="card"), Fetch(name="fetch_bqlds"))))

sieve_syntax = register_strategy(Strategy(
    name="sieve_syntax", toolset_name="bql_dense_snip",
    description="Sieve (BM25 and dense fused) with the reference part of the manual only",
    tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), manual_set="syntax"), Fetch(name="fetch_bqlds"))))

sieve_noconstruct = register_strategy(Strategy(
    name="sieve_noconstruct", toolset_name="bql_dense_snip",
    description="Sieve (BM25 and dense fused) with the manual minus its query-construction advice",
    tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), manual_set="noconstruct"), Fetch(name="fetch_bqlds"))))

# leave-one-section-out cuts of the corrected manual (scripts/derive_manual_cuts.py): which part
# of the manual carries the effect. Each strategy drops exactly one `## ` section.
_CUTS = {"nohowto": "How to search", "nofields": "The fields", "nofetch": "Fetch",
         "nohops": "Hops", "noexamples": "Worked examples", "nomistakes": "Common mistakes"}
sieve_manual_cuts = {
    ms: register_strategy(Strategy(
        name=f"sieve_{ms}", toolset_name="bql_dense_snip",
        description=f"Sieve (BM25 and dense fused) with the manual minus its '{heading}' section",
        tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), manual_set=ms), Fetch(name="fetch_bqlds"))))
    for ms, heading in _CUTS.items()}

# the reference-only manual plus one advice section added back: which advice helps on top of
# the reference (the mirror of the cuts, starting from the best manual of the ablation)
_ADDS = {"refhowto": "How to search", "refhops": "Hops", "refmistakes": "Common mistakes"}
sieve_manual_adds = {
    ms: register_strategy(Strategy(
        name=f"sieve_{ms}", toolset_name="bql_dense_snip",
        description=f"Sieve (BM25 and dense fused) with the reference manual plus its '{heading}' section",
        tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), manual_set=ms), Fetch(name="fetch_bqlds"))))
    for ms, heading in _ADDS.items()}

# dense-only ranking inside the Boolean filter
sieve_dense = register_strategy(Strategy(
    name="sieve_dense", toolset_name="bql_donly_snip",
    description="Sieve: Boolean search with snippets (dense ranking only) then fetch a section",
    tools=(SearchBql(name="search_bqldos", ranking="dense", snippet=TermWindow()), Fetch(name="fetch_bqldos"))))

# the fused ranking without snippets (an ablation)
sieve_nosnip = register_strategy(Strategy(
    name="sieve_nosnip", toolset_name="bql_dense_fetch",
    description="Sieve without snippets: Boolean search (BM25 and dense fused) then fetch a section",
    tools=(SearchBql(name="search_bqldf", ranking="fused"), Fetch(name="fetch_bqldf"))))

# the plain Boolean search (no snippets, BM25 ranking) and its coverage-tiered v2 manual
sieve_plain = register_strategy(Strategy(
    name="sieve_plain", toolset_name="search_fetch",
    description="Boolean search (BM25 ranking, no snippets) then fetch a section",
    tools=(SearchBql(name="search"), Fetch(name="fetch"))))

sieve_v2 = register_strategy(Strategy(
    name="sieve_v2", toolset_name="search_fetch_v2",
    description="Boolean search with coverage tiers and a date nudge (v2 manual) then fetch a section",
    tools=(SearchBql(name="search_v2", coverage=True, date_nudge=True, manual_set="v2"), Fetch(name="fetch_v2"))))

# Boolean search then reading whole documents (the visit read on the Sieve search)
sieve_visit = register_strategy(Strategy(
    name="sieve_visit", toolset_name="bql_visit",
    description="Boolean search (BM25 ranking) then read whole documents",
    tools=(SearchBql(name="search_bv"), Visit(name="visit_bv"))))

sieve_visit_fused = register_strategy(Strategy(
    name="sieve_visit_fused", toolset_name="bql_dense_visit",
    description="Boolean search (BM25 and dense fused) then read whole documents",
    tools=(SearchBql(name="search_bqld", ranking="fused"), Visit(name="visit_bqld"))))

sieve_visit_dense = register_strategy(Strategy(
    name="sieve_visit_dense", toolset_name="bql_donly_visit",
    description="Boolean search (dense ranking only) then read whole documents",
    tools=(SearchBql(name="search_bqldo", ranking="dense"), Visit(name="visit_bqldo"))))

# Sieve with the listing filled to k: exact matches first, then the ranker's closest documents
# over the query's terms (marked ~), so a tight filter never leaves the agent short of candidates
sieve_fill = register_strategy(Strategy(
    name="sieve_fill", toolset_name="bql_dense_snip_fill",
    description="Sieve, listing filled to k (BM25 and dense fused): exact Boolean matches first, then the closest documents",
    tools=(SearchBql(name="search_bqlds", ranking="fused", snippet=TermWindow(), fill=True), Fetch(name="fetch_bqlds"))))

sieve_bm25_fill = register_strategy(Strategy(
    name="sieve_bm25_fill", toolset_name="search_fetch_s_fill",
    description="Sieve, listing filled to k (BM25 ranking): exact Boolean matches first, then the closest documents",
    tools=(SearchBql(name="search_s", snippet=TermWindow(), fill=True), Fetch(name="fetch_s"))))

sieve_dense_fill = register_strategy(Strategy(
    name="sieve_dense_fill", toolset_name="bql_donly_snip_fill",
    description="Sieve, listing filled to k (dense ranking only): exact Boolean matches first, then the closest documents",
    tools=(SearchBql(name="search_bqldos", ranking="dense", snippet=TermWindow(), fill=True), Fetch(name="fetch_bqldos"))))
