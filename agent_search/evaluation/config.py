"""Structured arguments for programmatic evaluation runs.

The CLI in ``agent_search.evaluation.run_eval`` is intentionally flat for shell use, but both
CLI and Python runs map to these layers: dataset, retriever, agent, evaluation,
and output.
"""
from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Sequence

from agent_search.retrievers.registry import RetrieverConfig


@dataclass(frozen=True)
class DatasetArgs:
    name: str = "fixture"
    limit: int | None = None
    corpus_limit: int | None = None
    repo_cache: str = "data/repos"
    allow_clone: bool = False


@dataclass(frozen=True)
class RetrieverArgs:
    name: str = "bm25_local"
    model: str | None = None
    dense_model: str | None = None
    index_root: str = "indexes"
    rebuild: bool = False


@dataclass(frozen=True)
class AgentArgs:
    policy: str = "stub"                  # "stub" | "llm"
    backend: str = "vllm"                 # "vllm" | "api"
    tp: int = 1
    api_base: str = "http://localhost:8000/v1"
    domain: str | None = None             # None = infer from dataset registry
    field_profile: str | None = None      # None = infer from dataset registry (BQL manual)
    max_steps: int = 50
    prompt_profile: str | None = None
    temperature: float = 0.6              # LLM sampling temperature
    seed: int | None = 42                 # sampling seed; fixed 42 for reproducibility


@dataclass(frozen=True)
class EvaluationArgs:
    level: str = "function"               # "function" | "file"
    k: Sequence[int] = (1, 3, 5, 10)
    workers: int = 1


@dataclass(frozen=True)
class OutputArgs:
    runs_dir: str = "runs"
    results_dir: str | None = None


@dataclass(frozen=True)
class RunConfig:
    dataset: DatasetArgs = field(default_factory=DatasetArgs)
    retriever: RetrieverArgs = field(default_factory=RetrieverArgs)
    agent: AgentArgs = field(default_factory=AgentArgs)
    evaluation: EvaluationArgs = field(default_factory=EvaluationArgs)
    output: OutputArgs = field(default_factory=OutputArgs)

    def resolved(self) -> "RunConfig":
        """Fill dataset-derived defaults shared by CLI and Python callers.

        The CLI auto-detects the prompt/embedder domain from the dataset. Python
        users should get the same behavior unless they explicitly override it.
        """
        from agent_search.evaluation.datasets import dataset_domain, default_dense_model

        domain = dataset_domain(self.dataset.name, self.agent.domain)
        dense_model = self.retriever.dense_model
        if dense_model is None:
            if self.retriever.name == "dense" and self.retriever.model:
                dense_model = self.retriever.model
            else:
                dense_model = default_dense_model(domain)

        if self.agent.domain == domain and self.retriever.dense_model == dense_model:
            return self
        return RunConfig(
            dataset=self.dataset,
            retriever=RetrieverArgs(
                name=self.retriever.name,
                model=self.retriever.model,
                dense_model=dense_model,
                index_root=self.retriever.index_root,
                rebuild=self.retriever.rebuild,
            ),
            agent=AgentArgs(
                policy=self.agent.policy,
                backend=self.agent.backend,
                tp=self.agent.tp,
                api_base=self.agent.api_base,
                domain=domain,
                field_profile=self.agent.field_profile,
                max_steps=self.agent.max_steps,
                prompt_profile=self.agent.prompt_profile,
                temperature=self.agent.temperature,
                seed=self.agent.seed,
            ),
            evaluation=self.evaluation,
            output=self.output,
        )

    def retriever_config(self) -> RetrieverConfig:
        from agent_search.evaluation.datasets import dataset_field_profile
        cfg = self.resolved()
        profile = dataset_field_profile(cfg.dataset.name, cfg.agent.field_profile)
        return RetrieverConfig(
            model=cfg.retriever.model,
            dense_model=cfg.retriever.dense_model,
            index_root=cfg.retriever.index_root,
            rebuild=cfg.retriever.rebuild,
            policy=cfg.agent.policy,
            backend=cfg.agent.backend,
            tp=cfg.agent.tp,
            api_base=cfg.agent.api_base,
            domain=cfg.agent.domain or "code",
            field_profile=profile,
            max_steps=cfg.agent.max_steps,
            prompt_override=cfg.agent.prompt_profile,
            temperature=cfg.agent.temperature,
            seed=cfg.agent.seed,
        )

    def as_namespace(self) -> Namespace:
        """CLI-shaped object for provenance writing."""
        cfg = self.resolved()
        return Namespace(
            dataset=cfg.dataset.name,
            retriever=cfg.retriever.name,
            model=cfg.retriever.model,
            dense_model=cfg.retriever.dense_model,
            domain=cfg.agent.domain,
            policy=cfg.agent.policy,
            check_complete=False,
            max_steps=cfg.agent.max_steps,
            prompt_profile=cfg.agent.prompt_profile,
            temperature=cfg.agent.temperature,
            seed=cfg.agent.seed,
            backend=cfg.agent.backend,
            tp=cfg.agent.tp,
            workers=cfg.evaluation.workers,
            repo_cache=cfg.dataset.repo_cache,
            allow_clone=cfg.dataset.allow_clone,
            api_base=cfg.agent.api_base,
            level=cfg.evaluation.level,
            k=list(cfg.evaluation.k),
            limit=cfg.dataset.limit,
            corpus_limit=cfg.dataset.corpus_limit,
            index_root=cfg.retriever.index_root,
            rebuild=cfg.retriever.rebuild,
            runs_dir=cfg.output.runs_dir,
            results_dir=cfg.output.results_dir,
        )


def results_dir_for(config: RunConfig) -> str:
    """The run-directory convention: <runs_dir>/<kind>/<dataset>/<model>/<retriever>.
    Everything else (level, k, max_steps, seed, corpus_limit, git rev, ...) is recorded in
    that directory's config.json instead of being flattened into a long folder name.
    `summarize_runs` reads those from config.json, so the path stays short and browsable.

    Note: since steps and seed do not appear in the path, a max_steps or multi-seed ablation
    lands in the same directory as the main run. Point those at a separate `runs_dir=` so
    they do not overwrite it (the main sweep uses one fixed steps/seed value, so this rarely
    comes up)."""
    config = config.resolved()
    if config.output.results_dir:
        return config.output.results_dir
    kind = "agent" if config.retriever.name.startswith("agent") else "retrieval_only"
    model = (config.retriever.model or config.retriever.dense_model or "default").split("/")[-1]
    return os.path.join(config.output.runs_dir, kind,
                        config.dataset.name, model, config.retriever.name)
