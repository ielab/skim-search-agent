"""Environment-tunable size knobs for the doc-research tool families.

Every constant here is read from `os.environ` at import time, so a config file that sets
the env var before import controls it. `agent_search/tokens.py` defines the token
counter these budgets are measured in; nothing here is a character cap.
"""
import os

# listing-snippet width, in model tokens (agent_search/snippets)
SNIPPET_TOKENS = int(os.environ.get("SNIPPET_TOKENS", "32"))
# how many section names and infobox keys a result card shows per hit (0 = all). The paper's
# cards showed 8 and 6; on BrowseComp-Plus structured 52% of the documents have more than 8
# named sections, so a hidden section cannot be fetched by name
LISTING_SECTIONS = int(os.environ.get("LISTING_SECTIONS", "8"))
LISTING_INFOBOX_KEYS = int(os.environ.get("LISTING_INFOBOX_KEYS", "6"))
# whole-doc visit/autoread read cap, in tokens
MAX_VISIT_TOKENS = int(os.environ.get("MAX_VISIT_TOKENS", "12000"))
# per-section fetch read cap, in tokens; tracks MAX_VISIT_TOKENS unless set on its own
MAX_SECTION_TOKENS = int(os.environ.get("MAX_SECTION_TOKENS", str(MAX_VISIT_TOKENS)))

# search_bm25's search listing depth when structure=False and full_text=False (the search_visit strategy)
BM25_VISIT_TOPK = int(os.environ.get("BM25_VISIT_TOPK", "5"))
# shared top-k for the full_text=True mode of search_bm25, search_dense and search_hybrid (the autoread strategies)
AUTOREAD_TOPK = int(os.environ.get("AUTOREAD_TOPK", "5"))
# search_dense's search listing depth when structure=False and full_text=False (the search_visit_dense strategy)
DENSE_VISIT_TOPK = int(os.environ.get("DENSE_VISIT_TOPK", "5"))
# search_bm25's listing depth when structure=True (the search_fetch strategy); the Sieve paper's k=5
# for every search call (its Sec. 4.4), the same as the visit strategies
BM25_FETCH_TOPK = int(os.environ.get("BM25_FETCH_TOPK", "5"))
# search_dense's retrieval pool size when structure=True (the search_fetch_dense strategy)
DENSE_FETCH_TOPK = int(os.environ.get("DENSE_FETCH_TOPK", "5"))

# Reciprocal Rank Fusion constant (Cormack, Clarke and Buettcher 2009); read by
# agent_search.retrievers.fusion.rrf, kept here so the run record lists it with the other knobs
RRF_K = int(os.environ.get("RRF_K", "60"))
# per-ranker pool depth queried before RRF fusion (bm25 pool and dense pool, each this deep)
HYBRID_POOL = int(os.environ.get("HYBRID_POOL", "100"))
# search_hybrid's post-fusion search listing depth when structure=False and full_text=False (the search_visit_hybrid strategy)
HYBRID_VISIT_TOPK = int(os.environ.get("HYBRID_VISIT_TOPK", "5"))
# search_hybrid's post-fusion retrieval pool size when structure=True (the search_fetch_hybrid strategy)
HYBRID_FETCH_TOPK = int(os.environ.get("HYBRID_FETCH_TOPK", "5"))
RERANK_VISIT_TOPK = int(os.environ.get("RERANK_VISIT_TOPK", "5"))
RERANK_FETCH_TOPK = int(os.environ.get("RERANK_FETCH_TOPK", "5"))
# ITER's listing: each hit shows the passage cut to this many model tokens (ITER's runs: 64)
DEDUP_SNIPPET_TOKENS = int(os.environ.get("DEDUP_SNIPPET_TOKENS", "64"))
