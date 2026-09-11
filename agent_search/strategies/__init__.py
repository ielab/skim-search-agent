"""Strategies: named combinations of tools (Sieve, Search-Visit, DCI, ...), one file each.

`base.py` holds the contract and the registry; `conditions.py` pairs strategies with tasks
and keeps the paper's condition names. The friendly-name table that the CLI and experiment
files resolve strategy names through lives in `agent_search.strategies.names`.
"""
from agent_search.strategies.base import STRATEGIES, Strategy, register_strategy
from agent_search.strategies.names import (DEFAULT_STRATEGY, DENSE_STRATEGIES, condition_of,  # noqa: F401
                                            resolve_strategy)
from agent_search.strategies.names import STRATEGIES as FRIENDLY_NAMES  # noqa: F401
from agent_search.strategies import paper  # noqa: E402,F401  (the built-in conditions)
from agent_search.strategies.conditions import CONDITIONS, Condition, get_condition  # noqa: E402,F401

__all__ = ["Strategy", "STRATEGIES", "register_strategy", "DEFAULT_STRATEGY", "DENSE_STRATEGIES",
           "condition_of", "resolve_strategy", "FRIENDLY_NAMES"]
