"""Programmatic use: build and run a research agent in-process.

    from agent_search import research

    docs = [{"_id": "d1", "title": "Treaty of Guadalupe Hidalgo",
             "text": "The Treaty of Guadalupe Hidalgo ended the Mexican-American War in 1848."},
            ...]
    result = research("Which treaty ended the Mexican-American War?", docs,
                      strategy="sieve_bm25", model="gpt-4o-mini")
    result.answer, result.ranking, result.steps, result.usage

The command line goes through the same objects used here: a strategy resolves to a
*condition* (a task paired with a strategy of tools), and ``run_episode`` drives a *policy*
over the strategy's tools. Pass a model as any ``messages -> text``
callable (``generate=``), or name one (``model=``) and let
``agent_search.agent.backbone.make_generate`` route it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from agent_search.corpus.units import CodeUnit, units_from_documents
from agent_search.strategies.names import DEFAULT_STRATEGY, condition_of, resolve_strategy


@dataclass
class ResearchResult:
    """What one episode produced. ``steps`` is the same per-step record the harness writes to
    ``rows.jsonl`` (``trajectory``); ``usage`` the same token/cost fields."""
    answer: str
    ranking: list[str]
    stopped: str
    steps: list[dict]
    usage: dict
    surfaced: list[str] = field(default_factory=list)
    condition: str = ""

    @property
    def observations(self) -> list[str]:
        return [str(s.get("observation") or "") for s in self.steps]


def build_agent(strategy: str = DEFAULT_STRATEGY, *,
                generate: Optional[Callable[[list], str]] = None,
                model: Optional[str] = None, backend: str = "api",
                api_base: str = "http://localhost:8000/v1", tp: int = 1,
                temperature: float = 0.6, seed: Optional[int] = 42,
                max_steps: int = 50, index_root: str = "indexes", rebuild: bool = False,
                field_profile: Optional[str] = None, driver: Optional[str] = None,
                dense_model: Optional[str] = None):
    """An un-indexed retriever agent for ``strategy`` (a friendly name or ``agent_<condition>``).

    * ``generate``: your own ``messages -> text`` callable (the ``Model`` contract). Wins
      over ``model``. Runs the text-parsed loop driver.
    * ``model``: a model id routed by ``make_generate`` (OpenAI ``gpt-*``, Gemini
      ``gemini-*``, a served OpenAI-compatible endpoint with ``backend="api"``, or in-process
      vLLM with ``backend="vllm"``).
    * neither: the dependency-free ``KeywordPolicy`` (a scripted smoke policy).

    Call ``.index(units, key=...)`` then ``.search(question, k)``; the episode record is on
    ``.last_trajectory_meta``. :func:`research` wraps exactly that."""
    from agent_search.evaluation.agent_runner import ConditionAgent
    from agent_search.strategies.conditions import CONDITIONS

    retriever_name = resolve_strategy(strategy)
    cond_name = condition_of(retriever_name) or (strategy if strategy in CONDITIONS else None)
    cond = CONDITIONS.get(cond_name) if cond_name else None
    if cond is None:
        raise ValueError(f"{strategy!r} is not a registered condition; choose from "
                         f"{sorted(CONDITIONS)}")
    if cond.strategy.retriever:
        raise ValueError(f"{strategy!r} is a retrieval-only floor, not an agent strategy; "
                         f"use agent_search.retrievers.registry.build_factory for it")
    domain = cond.domain
    profile = field_profile or domain
    from agent_search.evaluation.datasets import default_dense_model
    dense = dense_model or default_dense_model(domain)

    if generate is None and model is not None:
        from agent_search.agent.backbone import make_generate
        generate = make_generate(model=model, backend=backend, api_base=api_base, tp=tp,
                                 temperature=temperature, seed=seed)

    if generate is None:
        from agent_search.agent.policies import KeywordPolicy
        policy_for = lambda c: KeywordPolicy(c.tool_names)  # noqa: E731
        chosen_driver = "loop"
    else:
        from agent_search.agent.policies import AgentPolicy
        gen = generate
        policy_for = lambda c: AgentPolicy(generate=gen, system=c.render(profile))  # noqa: E731
        chosen_driver = driver or "loop"

    return ConditionAgent(cond, policy_for, generate=generate, max_steps=max_steps, dense_model=dense,
                          index_root=index_root, rebuild=rebuild, driver=chosen_driver,
                          field_profile=profile, model=model, api_base=api_base)


def as_units(docs: Iterable[Any]) -> list[CodeUnit]:
    """Plain dicts (``_id``/``id``, ``title``, ``text``/``body``, optional ``sections``
    ``[{"heading", "text"}, ...]`` and metadata) or ready ``CodeUnit`` objects -> units."""
    docs = list(docs)
    if docs and isinstance(docs[0], CodeUnit):
        return docs  # type: ignore[return-value]
    return units_from_documents(docs)


def research(question: str, docs: Iterable[Any], *, strategy: str = DEFAULT_STRATEGY,
             corpus_key: Optional[str] = None, k: int = 10, **agent_kwargs) -> ResearchResult:
    """Run one research episode over ``docs`` and return its answer, ranking, and record.

    ``corpus_key`` names the corpus for its persistent indexes (Lucene BM25, the Lucene
    structured index, dense embeddings) under ``index_root``; with ``None`` the key is derived
    from the corpus content, so a second call over the same documents reuses the indexes."""
    units = as_units(docs)
    agent = build_agent(strategy, **agent_kwargs)
    agent.index(units, key=corpus_key)
    ranking = list(agent.search(question, k))
    meta = dict(agent.last_trajectory_meta or {})
    usage = {key: meta.get(key, 0) for key in (
        "llm_calls", "n_steps", "prompt_tokens", "completion_tokens",
        "cached_input_tokens", "reasoning_tokens")}
    return ResearchResult(
        answer=meta.get("final_answer", ""), ranking=ranking, stopped=meta.get("stopped", ""),
        steps=list(meta.get("trajectory") or []), usage=usage,
        surfaced=list(meta.get("surfaced_docs") or []), condition=agent.tool or "")


__all__ = ["ResearchResult", "build_agent", "research", "as_units"]
