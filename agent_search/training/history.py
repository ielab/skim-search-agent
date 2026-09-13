"""The agent's search history at inference, so a trained retriever sees the same query it was
trained on.

`agent_search.evaluation.agent_runner.ConditionAgent` opens a `QueryContext` for every episode and
feeds it each step as it happens (what was searched, what was read, what the model said
afterwards). `agent_search.retrievers.dense.belief.DenseBelief` asks `current_query_for` how
to write the retriever query for the sub-query it is about to encode. When the dense query style
is ``plain`` (the default) nothing changes; with any other style the query is rendered by
`queries.render_query` with the history so far, byte-identical to how `triples.py` rendered it
for training.

The context is a `contextvars.ContextVar`, so concurrent episodes in worker threads never see
each other's history and shared engines need no per-episode state.
"""
from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent_search.training.queries import DEFAULT_STYLE, STYLES, render_query, reasoning_text
from agent_search.training.triples import is_read_action, is_search_action, read_ids


def dense_query_style() -> str:
    style = os.environ.get("DENSE_QUERY_STYLE", "plain").strip() or "plain"
    if style not in STYLES:
        raise ValueError(f"DENSE_QUERY_STYLE={style!r}; choose from {STYLES}")
    return style


@dataclass
class QueryContext:
    question: str
    text_of: Callable[[str], Optional[str]]
    style: str = "plain"
    interactions: list[dict] = field(default_factory=list)   # [{"query", "visits": [(id, text, reasoning)]}]
    _last_hits: list[str] = field(default_factory=list)
    _pending_reads: list[str] = field(default_factory=list)  # docs read at the previous step
    current_reasoning: str = ""                              # the <think> of the turn now running

    def render(self, current: str) -> str:
        if self.style == "plain":
            return current
        return render_query(self.style, self.question, current, self.interactions,
                            pre_reasoning=self.current_reasoning)

    def note(self, raw_output: str) -> None:
        """Attach the model's latest generation as the note on the documents read at the
        previous step. Called before the next tool runs, so a search issued in this generation
        already sees the note (the same order `triples.py` uses when it builds training data)."""
        self.current_reasoning = reasoning_text(raw_output or "")   # i9: the pre-search reasoning
        if self._pending_reads and self.interactions:
            note = self.current_reasoning
            visits = self.interactions[-1]["visits"]
            for k, (d, t, _) in enumerate(visits):
                if d in self._pending_reads and not _:
                    visits[k] = (d, t, note)
        self._pending_reads = []

    def observe(self, step: Any, last_hits: list[str]) -> None:
        """Record one finished step. `step` is a loop `Step` (name/args/raw_output); `last_hits`
        the toolbox's listing after the step."""
        raw = getattr(step, "raw_output", "") or ""
        # the model's generation at this step is the reasoning AFTER the previous step's reads;
        # `note()` normally handled it before the tool ran, this covers callers without the hook
        if self._pending_reads and self.interactions:
            self.note(raw)
        name = getattr(step, "name", "") or ""
        args = getattr(step, "args", {}) or {}
        if is_search_action(name):
            q = args.get("query") or args.get("q") or args.get("pattern") or ""
            if isinstance(q, (list, tuple)):                 # the tool ran the first query of a list
                q = next((x for x in q if x not in (None, "")), "")
            q = " ".join(str(q).split())
            self.interactions.append({"query": q, "visits": []})
            self._last_hits = list(last_hits)
        elif is_read_action(name):
            ids = read_ids({"args": args}, self._last_hits)
            if self.interactions:
                for d in ids:
                    self.interactions[-1]["visits"].append((d, self.text_of(d) or "", ""))
            self._pending_reads = ids


CURRENT: contextvars.ContextVar[Optional[QueryContext]] = contextvars.ContextVar(
    "skimsearchagent_query_context", default=None)


def current_query_for(sub_query: str) -> str:
    """The retriever query for `sub_query` under the active episode's context (or `sub_query`
    itself outside an episode / in the plain style)."""
    ctx = CURRENT.get()
    if ctx is None:
        return sub_query
    return ctx.render(sub_query)


__all__ = ["QueryContext", "CURRENT", "current_query_for", "dense_query_style", "DEFAULT_STYLE"]
