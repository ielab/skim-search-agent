"""The contracts of the framework — everything you implement to extend it, in one
place.

agent_search answers an information need by **searching a corpus**. The pieces that
vary across tasks are the *dials*; each is one interface here:

- ``CorpusSource`` — where the units come from. Two regimes: a **per-query** corpus
  (a code repo @ commit, built fresh per instance) or a **shared** corpus (one fixed
  document/passage collection searched by every query).
- ``Retriever`` — query -> ranked unit ids. The zoo: grep, BM25, dense, SPLADE, an
  LLM reranker, the structural BQL engine. Index lifecycle is the retriever's own
  business (none / inverted / embeddings / sparse).
- ``Model`` — an LLM provider: ``generate(messages) -> text``. Local (vLLM) or API
  (OpenAI / Claude / Gemini). The agent is provider-agnostic because tools are
  described in the prompt and tool calls are parsed from text.
- ``Executor`` / ``Observation`` / ``Hit`` — a retriever's execution result and the
  feedback an agent reads after a tool call.

Adding a dial value = implement the interface + register it; no harness change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from .units import Unit


# --- corpus -----------------------------------------------------------------

@runtime_checkable
class CorpusSource(Protocol):
    """Yields the retrievable units for a query instance. A per-query source
    returns that instance's repo units; a shared source ignores the instance and
    returns the one fixed collection (cached). ``key`` is a stable corpus identity
    used to reuse a built index across instances that share a corpus."""

    def units_for(self, instance: Any) -> Sequence[Unit]: ...

    def key(self, instance: Any) -> str: ...


# --- retrieval --------------------------------------------------------------

class Retriever(ABC):
    """index a corpus of units, then rank them for a query. Built once per corpus
    identity (per SWE-bench instance for code, once for a shared collection)."""

    name: str = "retriever"

    @abstractmethod
    def index(self, units: Sequence[Unit], key: Optional[str] = None) -> "Retriever":
        """Build/load the index for a corpus. ``key`` lets persistent backends cache
        and reuse an index keyed by corpus identity (e.g. "repo@commit")."""
        ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[str]:
        """Return up to k unit doc_ids ("path::qualname"), best first."""
        ...


# --- agent tools ------------------------------------------------------------

@runtime_checkable
class Tool(Protocol):
    """An agent-callable action: given JSON arguments, return a text observation the
    agent reads next turn. Search tools wrap a Retriever; workspace tools open/scroll
    a file. The available tools per condition are listed in ``prompts/tools.yaml`` and
    dispatched by the agent's action executor / workspace."""

    def __call__(self, **arguments: Any) -> str: ...


# --- execution result + agent feedback --------------------------------------

@dataclass
class Hit:
    doc_id: str
    score: float
    path: str | None = None
    line: int | None = None
    region: str | None = None
    snippet: str | None = None


@dataclass
class Observation:
    """What gets serialized into the agent's context after a query."""
    n_hits: int
    hits: Sequence[Hit]
    clause_df: dict[str, int] = field(default_factory=dict)  # per-clause/variant df
    typecheck_ok: bool = True
    error: str | None = None  # parse/type error -> agent reacts instead of crashing


class Executor(ABC):
    """Runs a compiled retrieval program and ranks (BM25). CPU-only by design.
    The structural BQL engine is the reference implementation."""

    @abstractmethod
    def run(self, program: Any, k: int = 100) -> Observation:
        ...


# --- model provider ---------------------------------------------------------

@runtime_checkable
class Model(Protocol):
    """An LLM provider. The agent calls it with a chat message list and reads back
    raw text; tool calls live in that text (DeepResearch-style), so any provider —
    local vLLM or a hosted OpenAI/Claude/Gemini endpoint — plugs in as one adapter."""

    def __call__(self, messages: list[dict]) -> str: ...
