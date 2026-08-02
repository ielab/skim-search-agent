import os

# Pyserini's `pyserini.encode` subpackage constructs an openai.OpenAI() client at
# import time (pulled in transitively by LuceneSearcher), which raises without a key.
# This project never uses OpenAI for Pyserini BM25; this placeholder only lets that dead import succeed;
# no OpenAI request is ever made. Set once here so every eval entry point is covered.
os.environ.setdefault("OPENAI_API_KEY", "agent-search-unused-placeholder")
# HF tokenizers' Rust threadpool otherwise leaks POSIX semaphores at interpreter
# exit under our ThreadPoolExecutor workers (the resource_tracker warning). We never
# need intra-call tokenizer parallelism here. setdefault: a shell-level value wins.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
