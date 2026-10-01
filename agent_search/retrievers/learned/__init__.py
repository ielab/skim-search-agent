"""Retrievers that read every page with a neural model ahead of a run and rank the corpus
exactly: `splade` (learned sparse) and `colbert` (late interaction). The shared frame is
`base.py`; one file per model family."""
from agent_search.retrievers.learned.base import LearnedIndexRetriever, page_texts
from agent_search.retrievers.learned.colbert import ColbertRetriever
from agent_search.retrievers.learned.splade import SpladeRetriever

__all__ = ["LearnedIndexRetriever", "SpladeRetriever", "ColbertRetriever", "page_texts"]
