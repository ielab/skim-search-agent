"""QUEST's tool: one search that returns long passages, and no read tool.

`search` takes an array of queries and returns, for each, the top five documents with a score
and the first 512 tokens of their text. QUEST was trained to answer from these passages, so it
runs with this tool and with its own prompt (task `research_quest`). Two variants, one per
ranker behind the search.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.search_passages.tool import SearchPassages

passages_dense = register_strategy(Strategy(
    name="passages_dense", toolset_name="passages_dense", sdk=False,
    description="QUEST's tool over the run's dense model: a batched search that returns long passages",
    tools=(SearchPassages(ranking="dense"),)))

passages_bm25 = register_strategy(Strategy(
    name="passages_bm25", toolset_name="passages_bm25", sdk=False,
    description="QUEST's tool over BM25: a batched search that returns long passages",
    tools=(SearchPassages(ranking="bm25"),)))
