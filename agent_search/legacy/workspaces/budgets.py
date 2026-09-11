"""Pre-0.3 environment-tunable size knobs for the doc-research tool families.

Kept so the parity tests can compare against it. `agent_search/tools/budgets.py` replaces it.

Every constant here is read from `os.environ` at import time, so a config file that sets
the env var before import controls it. `agent_search/core/tokens.py` defines the token
counter these budgets are measured in; nothing here is a character cap.
"""
import os

# listing-snippet width, in whitespace tokens (best_line/opening_line in common.py)
SNIPPET_TOKENS = int(os.environ.get("SNIPPET_TOKENS", "32"))
# whole-doc visit/autoread read cap, in tokens
MAX_VISIT_TOKENS = int(os.environ.get("MAX_VISIT_TOKENS", "12000"))
# per-section fetch read cap, in tokens; tracks MAX_VISIT_TOKENS unless set on its own
MAX_SECTION_TOKENS = int(os.environ.get("MAX_SECTION_TOKENS", str(MAX_VISIT_TOKENS)))

# Bm25Visit/Bm25AutoRead's search listing depth
BM25_VISIT_TOPK = int(os.environ.get("BM25_VISIT_TOPK", "5"))
# retrieve-and-read baselines' (Bm25AutoRead/DenseAutoRead/HybridAutoRead) shared top-k
AUTOREAD_TOPK = int(os.environ.get("AUTOREAD_TOPK", "5"))
# DenseVisit's search listing depth
DENSE_VISIT_TOPK = int(os.environ.get("DENSE_VISIT_TOPK", "5"))
# Bm25FetchWorkspace/Bm25FetchSnipWorkspace's retrieval pool size
BM25_FETCH_TOPK = int(os.environ.get("BM25_FETCH_TOPK", "10"))
# DenseFetchWorkspace/DenseFetchPlainWorkspace's retrieval pool size
DENSE_FETCH_TOPK = int(os.environ.get("DENSE_FETCH_TOPK", "10"))

# Reciprocal Rank Fusion constant (Cormack, Clarke & Buettcher 2009); see common.rrf_fuse
RRF_K = int(os.environ.get("RRF_K", "60"))
# per-ranker pool depth queried before RRF fusion (bm25 pool and dense pool, each this deep)
HYBRID_POOL = int(os.environ.get("HYBRID_POOL", "100"))
# HybridVisit's post-fusion search listing depth
HYBRID_VISIT_TOPK = int(os.environ.get("HYBRID_VISIT_TOPK", "5"))
# HybridFetchSnipWorkspace's post-fusion retrieval pool size
HYBRID_FETCH_TOPK = int(os.environ.get("HYBRID_FETCH_TOPK", "10"))
