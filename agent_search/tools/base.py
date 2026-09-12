"""The tool contract and the episode state.

A tool is one atomic action the agent can call. It owns its declaration (the name the model
sees, a description, JSON parameters), its implementation (`run(args)` returning the
observation text; an error is text, never an exception) and, when the agent has to learn a
syntax, its manual (a markdown file next to the tool, rendered into the system prompt).

Tools share an `EpisodeState`: what the last search listed, what has been surfaced in
first-seen order (the agent's ranking for the rank metrics), what was read. A strategy binds its
tools to the state, the corpus and the engines it needs; the bound set is a `ToolBox`, which is
what the agent loop talks to.
"""
from __future__ import annotations

import copy
import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from agent_search.tools.seen import OrderedSeen
from typing import Protocol, runtime_checkable

ENGINE_KINDS = ("bm25", "dense", "bql", "bql_fused", "bql_dense", "bql_plain", "indri")


@dataclass
class EpisodeState:
    """Per-question state shared by a strategy's tools."""
    question: str = ""
    seen: OrderedSeen = field(default_factory=OrderedSeen)   # surfaced ids, first-seen order
    last_hits: list = field(default_factory=list)            # ids listed by the last search, rank order
    listing: dict = field(default_factory=dict)              # per id: what the last search rendered (sections, infobox keys)
    previous_queries: list = field(default_factory=list)
    reads: list = field(default_factory=list)                # ids opened by a read tool, in order
    scratch: dict = field(default_factory=dict)              # tool-private state, keyed by tool name

    @property
    def surfaced(self) -> list:
        return list(self.seen)



@runtime_checkable
class Workspace(Protocol):
    """The tool surface an episode drives.

    ``run`` dispatches one tool call by name and returns the text observation fed back to
    the policy. A tool error is itself a returned observation, never a raised exception.
    ``tools`` lists the tool names this workspace answers to (the condition's toolset).
    ``surfaced`` is the doc ids the episode has surfaced so far in first-seen order: the
    agent's retrieval ranking for rank metrics (built-ins keep an
    ``agent_search.tools.seen.OrderedSeen`` as ``seen`` and expose it as ``surfaced``)."""

    tools: Sequence[str]

    def run(self, name: str, args: dict) -> str: ...

    @property
    def surfaced(self) -> Sequence[str]: ...


def query_text(args: dict, *keys: str) -> str:
    """The query argument as one string. A backbone sometimes passes a list of queries (ITER's
    tool accepts one) or a number; a list is joined with spaces, anything else is str()."""
    value = ""
    for k in keys or ("query", "q"):
        if args.get(k) not in (None, ""):
            value = args[k]
            break
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value if v not in (None, ""))
    return str(value).strip()


class Tool:
    """Subclass per tool. Class attributes are the declaration; `run` is the implementation."""

    name: str = ""                       # the name the model sees (a strategy may rename it)
    description: str = ""
    parameters: dict = {"type": "object", "properties": {}}
    manual: Any = None                   # a markdown file next to the tool, or {domain: file}
    engines: tuple = ()                  # engine kinds this tool needs (see ENGINE_KINDS)
    needs_files: bool = False            # reads the raw repository files (code tasks)
    aliases: tuple = ()                  # other names a model may call this tool by
    refusals: dict = {}                  # {name: message} for calls this tool answers with an error

    def __init__(self, name: Optional[str] = None, **options):
        if name:
            self.name = name
        for k, v in options.items():
            if not hasattr(type(self), k):
                raise TypeError(f"{type(self).__name__} has no option {k!r}")
            setattr(self, k, v)
        self.options = dict(options)
        self.state: EpisodeState = EpisodeState()
        self.units: Sequence = ()
        self.ubyid: Mapping = {}
        self.engine: dict = {}
        self.files: dict = {}
        self.corpus_key: Optional[str] = None

    # --- binding -----------------------------------------------------------------------
    def bind(self, state: EpisodeState, units, ubyid, engines: Mapping[str, Any],
             files: Optional[dict] = None, corpus_key: Optional[str] = None) -> "Tool":
        """Attach the episode state, the corpus and the engines (`{kind: engine}`)."""
        self.state, self.units, self.ubyid = state, units, ubyid
        self.engine = {k: engines[k] for k in self.engines}
        self.files = files or {}
        self.corpus_key = corpus_key
        self.on_bind()
        return self

    def on_bind(self) -> None:
        """Per-episode setup after binding (staging, exports). Nothing by default."""

    # --- the declaration ---------------------------------------------------------------
    def declaration(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def manual_path(self, domain: str) -> Optional[str]:
        """The manual for `domain`: a plain file serves every domain, a mapping selects per
        domain (falling back to `code`, then any). Paths are relative to the tool's folder."""
        spec = self.manual
        if isinstance(spec, dict):
            spec = spec.get(domain) or spec.get("code") or next(iter(spec.values()), None)
        if not spec:
            return None
        p = Path(spec)
        if not p.is_absolute():
            p = Path(inspect.getfile(type(self))).resolve().parent / p
        return str(p)

    # --- the implementation --------------------------------------------------------------
    def run(self, args: dict) -> str:
        raise NotImplementedError

    def fresh(self) -> "Tool":
        """A copy for a new episode (tools are prototypes on a strategy)."""
        return copy.copy(self)


class ToolBox:
    """A strategy's tools bound for one episode: what the agent loop calls."""

    def __init__(self, tools: Sequence[Tool], state: EpisodeState):
        self._tools = {t.name: t for t in tools}
        self.tools = tuple(t.name for t in tools)
        self.state = state

    def run(self, name: str, args: Optional[dict]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            for t in self._tools.values():
                if name in t.aliases:
                    tool = t
                    break
        if tool is None:
            for t in self._tools.values():
                if name in t.refusals:
                    return t.refusals[name]
            return f"ERROR: unknown tool {name!r}. Available tools: {', '.join(self.tools)}."
        try:
            return tool.run(args or {})
        except Exception as e:  # noqa: BLE001: a tool error is an observation, never a crash
            return f"ERROR: {type(e).__name__}: {e}"

    def __getitem__(self, name: str) -> Tool:
        return self._tools[name]

    # what the harness records
    @property
    def seen(self) -> OrderedSeen:
        return self.state.seen

    @property
    def surfaced(self) -> list:
        return self.state.surfaced

    @property
    def last_hits(self) -> list:
        return self.state.last_hits

    @property
    def previous_queries(self) -> list:
        return self.state.previous_queries


__all__ = ["Tool", "ToolBox", "EpisodeState", "ENGINE_KINDS"]
