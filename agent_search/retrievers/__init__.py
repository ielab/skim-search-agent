"""Retriever families (a query to ranked units). lexical/ (the BM25 scorer, in-memory BM25,
Lucene BM25 through Pyserini, grep), dense/ (a base class and one file per encoder family),
bql/ (the Boolean structural method), indri/ (the Indri query language), lucene/ (both query
languages compiled to Lucene), backend.py (which engine serves BQL and Indri), engines.py (the
per-corpus engine registry the tools share), registry.py (the names a run can ask for)."""
