"""The contracts of the library — what you implement to extend it, in one place.

SkimSearchAgent answers an information need by letting an agent **search a corpus** over
several steps. The parts that vary between experiments are the *dials*; each is one
interface here, and every built-in implementation goes through the same interface as a
user-provided one (nothing in the harness special-cases the built-ins):

==================  ====================================================================
dial                contract
==================  ====================================================================
corpus              a sequence of ``Unit`` (``agent_search.corpus.units.CodeUnit``): the
                    retrievable atom — ``doc_id``, ``title``, ``body``, optional named
                    ``sections`` and ``metadata``. Build one from plain dicts with
                    ``units_from_documents``; register a loader with
                    ``agent_search.evaluation.datasets.register_dataset``.
retriever           ``Retriever`` — index a corpus, rank it for a query. Register with
                    ``agent_search.retrievers.registry.register``.
model               ``Model`` — ``generate(messages) -> text``: any callable taking an
                    OpenAI-style chat message list and returning the raw generation.
                    ``agent_search.models.backends.make_generate`` builds the built-ins.
policy              ``Policy`` — decides the next raw generation from the task and the
                    step history. ``AgentPolicy`` (prompted model), ``KeywordPolicy``
                    (no model) and ``ScriptPolicy`` (replay) are the built-ins.
workspace / tools   ``Workspace`` — the tool surface an episode drives: dispatch one tool
                    call by name and return the text observation; remember what was
                    surfaced. ``agent_search.tools.base.ToolBox`` is the built-in: the
                    bound ``Tool`` instances of a strategy (``agent_search/strategies``)
                    over one episode state. A tool is a ``Tool`` subclass in its own
                    folder under ``agent_search/tools/``.
evaluator           functions over the run record: ``agent_search.evaluation.metrics``,
                    ``doc_scoring``, ``llm_judge``.
==================  ====================================================================

Every interface here is either an ABC the built-ins subclass or a ``runtime_checkable``
Protocol the harness reads through ``getattr``; there are no decorative contracts. The
episode loop (``agent_search.agent.loop.run_episode``) consumes exactly ``Policy`` and
``Workspace``; the harness (``agent_search.evaluation.run_eval``) consumes exactly
``Retriever`` plus the optional capability flags documented on it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from .units import Unit


# --- retrieval --------------------------------------------------------------

class Retriever(ABC):
    """Index a corpus of units, then rank them for a query.

    Built once per corpus identity (``key``) and reused across queries. Optional
    capabilities the harness reads with ``getattr``:

    * ``returns_full_set = True`` — ``search`` returns the retriever's own complete ranking
      (an agent's surfaced set), so the harness must not pad it to the set-metric pool.
    * ``is_cached(key) -> bool`` — a persistent index for ``key`` already exists on disk
      (lets ``build_indexes`` skip parsing the corpus).
    * ``search_with_scores(query, k) -> list[tuple[str, float]]`` — scored ranking, when
      the engine has scores to show.
    """

    name: str = "retriever"

    @abstractmethod
    def index(self, units: Sequence[Unit], key: Optional[str] = None) -> "Retriever":
        """Build or load the index for a corpus. ``key`` is the corpus identity persistent
        backends cache under (e.g. the dataset name); ``None`` means in-memory only."""
        ...

    @abstractmethod
    def search(self, query: str, k: int) -> list[str]:
        """Return up to ``k`` unit ``doc_id``s, best first."""
        ...


@dataclass
class Hit:
    """One ranked result with optional provenance (used by the structured executors)."""
    doc_id: str
    score: float
    path: str | None = None
    line: int | None = None
    region: str | None = None
    snippet: str | None = None


@dataclass
class Observation:
    """A structured retrieval result: what a search tool renders for the agent."""
    n_hits: int
    hits: Sequence[Hit]
    clause_df: dict[str, int] = field(default_factory=dict)  # per-clause document frequency
    typecheck_ok: bool = True
    error: str | None = None  # parse/type error -> the agent reacts instead of crashing


# --- model provider ---------------------------------------------------------

@runtime_checkable
class Model(Protocol):
    """An LLM provider: ``generate(messages) -> text``.

    ``messages`` is an OpenAI-style chat list (``[{"role": ..., "content": ...}, ...]``); the
    return value is the raw generation. Tool calls live in that text (``<tool_call>...``), so
    any provider plugs in as one callable — a served vLLM, the OpenAI or Gemini APIs, or a
    lambda in a test. ``agent_search.models.backends.make_generate`` returns one of these.

    Optional attributes the loop reads with ``getattr`` (never required): ``client`` and
    ``model`` (an OpenAI-compatible client + model id) enable the forced-answer elicitation
    call on budget exhaustion; without them the episode simply ends with what it has."""

    def __call__(self, messages: list[dict]) -> str: ...


# --- policy -----------------------------------------------------------------

@runtime_checkable
class Policy(Protocol):
    """Decides the next raw generation from the task and the steps so far. The loop
    parses ONE tool call (or a terminal ``<answer>``) out of what it returns."""

    def propose(self, task: Any, history: Sequence[Any]) -> str: ...


# --- workspace / tools ------------------------------------------------------

@runtime_checkable
class Workspace(Protocol):
    """The tool surface an episode drives.

    ``run`` dispatches one tool call by name and returns the text observation fed back to
    the policy — a tool error is itself a returned observation, never a raised exception.
    ``tools`` lists the tool names this workspace answers to (the condition's toolset).
    ``surfaced`` is the doc ids the episode has surfaced so far in first-seen order: the
    agent's retrieval ranking for rank metrics (built-ins keep an
    ``agent_search.core.seen.OrderedSeen`` as ``seen`` and expose it as ``surfaced``)."""

    tools: Sequence[str]

    def run(self, name: str, args: dict) -> str: ...

    @property
    def surfaced(self) -> Sequence[str]: ...


__all__ = ["Unit", "Retriever", "Hit", "Observation", "Model", "Policy", "Workspace"]
