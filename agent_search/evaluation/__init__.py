"""Evaluation harness: dataset -> corpus -> retriever/agent -> metrics -> run record.

Entry points:

* ``python -m agent_search.evaluation.run_eval`` (or the ``skimsearchagent-eval`` console
  script) — the fully-flagged experiment runner.
* ``python -m agent_search.evaluation.build_indexes`` — pre-build persistent indexes.
* ``python -m agent_search.evaluation.llm_judge`` — grade a finished run with an LLM judge.
* ``agent_search.evaluation.run_eval.run_config(RunConfig)`` — the programmatic equivalent.

Importing this package has no side effects on the process environment beyond
``TOKENIZERS_PARALLELISM`` (set only if unset), which prevents the HF tokenizers thread pool
from leaking POSIX semaphores under the harness's worker threads.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
