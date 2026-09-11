"""Search-Visit: a search that lists documents, then reading whole documents.

Four variants, one per ranker (BM25, dense, hybrid, reranked). The tools are declared here with the exact text the paper
prompts used (`bm25_search`/`visit`, `dense_search`/`visit_d`, `hybrid_search`/`visit_h`).
"""
from agent_search.snippets import TermWindow
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.search_bm25.tool import SearchBm25
from agent_search.tools.search_dense.tool import SearchDense
from agent_search.tools.search_hybrid.tool import SearchHybrid
from agent_search.tools.search_reranked.tool import SearchReranked
from agent_search.tools.visit.tool import Visit

search_visit = register_strategy(Strategy(
    name="search_visit", toolset_name="research_bm25",
    description="search (BM25) then read whole documents",
    tools=(SearchBm25(name="bm25_search"), Visit(name="visit"))))

search_visit_dense = register_strategy(Strategy(
    name="search_visit_dense", toolset_name="dense_visit",
    description="search (dense) then read whole documents",
    tools=(SearchDense(name="dense_search"), Visit(name="visit_d"))))

search_visit_hybrid = register_strategy(Strategy(
    name="search_visit_hybrid", toolset_name="hybrid_visit",
    description="search (BM25 and dense fused by RRF) then read whole documents",
    tools=(SearchHybrid(name="hybrid_search"), Visit(name="visit_h"))))

search_visit_reranked = register_strategy(Strategy(
    name="search_visit_reranked", toolset_name="reranked_visit",
    description="search (a retriever's pool reordered by a reranker) then read whole documents",
    tools=(SearchReranked(name="reranked_search"), Visit(name="visit_r"))))

# the BM25 listing with query-biased snippets (a fairness variant of search_visit)
search_visit_snippets = register_strategy(Strategy(
    name="search_visit_snippets", toolset_name="bm25q_visit",
    description="search (BM25, query-biased snippets) then read whole documents",
    tools=(SearchBm25(name="bm25q_search", snippet=TermWindow()), Visit(name="visit_q"))))
