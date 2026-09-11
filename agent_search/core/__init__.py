"""Core contracts and shared primitives: the library's extension surface.

* :mod:`agent_search.core.interfaces`: ``Retriever``, ``Model``, ``Policy``, ``Workspace``,
  ``Hit``, ``Observation``.
* :mod:`agent_search.core.units`: ``Unit``, the retrievable atom.
* :mod:`agent_search.core.tokens`: the single token ruler for every length limit.
* :mod:`agent_search.core.seen`: ``OrderedSeen``, what an episode surfaced, in order.
* :mod:`agent_search.core.errors`: ``SetupError``.
"""
from .errors import SetupError
from .interfaces import Hit, Model, Observation, Policy, Retriever, Workspace
from .seen import OrderedSeen
from .tokens import cap_tokens, count_tokens, truncate_tokens
from .units import CodeUnit, Unit

__all__ = ["CodeUnit", "Unit", "Hit", "Model", "Observation", "Policy", "Retriever", "Workspace",
           "OrderedSeen", "SetupError", "cap_tokens", "count_tokens", "truncate_tokens"]
