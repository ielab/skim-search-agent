"""Retrievers that read every page with a neural model ahead of a run and rank the corpus
exactly: `splade` (learned sparse), `colbert` (late interaction) and `diffretriever` (a diffusion
language model's multi-vector output, served from its own environment). The shared frame is
`base.py`; one file per model family."""
from agent_search.retrievers.learned.base import LearnedIndexRetriever, page_texts
from agent_search.retrievers.learned.colbert import ColbertRetriever
from agent_search.retrievers.learned.diffretriever import DiffRetrieverRetriever
from agent_search.retrievers.learned.splade import SpladeRetriever

__all__ = ["LearnedIndexRetriever", "SpladeRetriever", "ColbertRetriever", "DiffRetrieverRetriever", "page_texts"]
