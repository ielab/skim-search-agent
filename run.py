#!/usr/bin/env python
"""Repository-local alias for the ``skimsearchagent`` console script.

    python run.py dataset=doc_fixture strategy=sieve_bm25
    python run.py dataset=doc_fixture strategy=search_visit model=gpt-4o-mini limit=1

Equivalent to ``skimsearchagent ...`` (installed with the package). See
``agent_search/cli.py`` and docs/CONFIGURATION.md for every accepted key.
"""
from __future__ import annotations

import sys

from agent_search.cli import ENV_KNOBS, STRATEGIES, main  # noqa: F401 (re-exported for tests)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
