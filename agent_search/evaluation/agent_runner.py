"""Run a condition (a task with a strategy) as a retriever the evaluation can score.

`ConditionAgent` builds the strategy's engines once per corpus and, per question, hands the
strategy's harness (`agent_search/harness/`: ReAct, one-shot RAG, a team) a `HarnessContext`
with the corpus, the engines and the model, then keeps the result's trajectory and its record.
It returns the surfaced documents in first-seen order as the ranking, so every harness is
scored by the same metrics; a team's member episodes are kept in full under `members`.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional, Sequence

from agent_search.agent.record import trajectory_meta
from agent_search.corpus.units import CodeUnit
from agent_search.harness.base import HarnessContext
from agent_search.retrievers.base import Retriever
from agent_search.retrievers.engines import Engines
from agent_search.strategies.conditions import Condition, get_condition


class ConditionAgent(Retriever):
    name = "agent"
    returns_full_set = True

    def __init__(self, condition: Condition | str, policy_for: Optional[Callable] = None, *,
                 generate: Optional[Callable] = None, max_steps: int = 50, dense_model: Optional[str] = None,
                 index_root: str = "indexes", rebuild: bool = False, driver: str = "loop",
                 field_profile: Optional[str] = None, model: Optional[str] = None, api_base: Optional[str] = None):
        self.condition = get_condition(condition) if isinstance(condition, str) else condition
        self.task, self.strategy = self.condition.task, self.condition.strategy
        self.domain = self.task.domain
        self.toolset = self.strategy.tool_names
        self.tool = f"agent_{self.condition.name}"          # the registered retriever name
        self.prompt_path = self.condition.name                # recorded as the row's prompt_profile_path
        self.tool_description = (f"toolset: {', '.join(self.toolset)}" if self.toolset
                                 else f"harness: {self.strategy.harness.name}")
        self._policy_for = policy_for                          # condition -> a policy (None: the keyword stub)
        self._generate = generate                              # messages -> text (None: no model)
        self.max_steps = max_steps
        self.dense_model = dense_model
        self.index_root, self.rebuild = index_root, rebuild
        self._driver = driver
        self._field_profile = field_profile
        self._model, self._api_base = model, api_base
        self._units: Sequence = ()
        self._ubyid: dict = {}
        self._key: Optional[str] = None
        self._files: dict = {}
        self.engines: Optional[Engines] = None
        self._tl = threading.local()

    # --- what the evaluation asks ---------------------------------------------------------------
    @property
    def needs_files(self) -> bool:
        from agent_search.strategies.base import STRATEGIES
        return self.strategy.needs_files or any(STRATEGIES[m].needs_files for m in self.strategy.harness.members)

    @property
    def last_trajectory(self):
        return getattr(self._tl, "traj", None)

    @property
    def last_trajectory_meta(self):
        return getattr(self._tl, "meta", None)

    def set_files(self, files: dict) -> None:
        self._files = files or {}

    def system_prompt(self) -> str:
        """The prompt text hashed into the run identity: the condition's rendered prompt, the
        harness's own prompts, and for a team every member condition's rendered prompt."""
        from agent_search.harness.plan_and_search import member_condition
        parts = [self.condition.render(self._field_profile)] if self.toolset else []
        parts += list(self.strategy.harness.prompts())
        parts += [member_condition(m).render(self._field_profile) for m in self.strategy.harness.members]
        return "\n\n".join(parts)

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "ConditionAgent":
        if getattr(units, "lazy", False):
            self._units, self._ubyid = units, units.by_id
        else:
            self._units = list(units)
            self._ubyid = {u.doc_id: u for u in self._units}
        self._key = key
        self.engines = Engines(self._units, key, index_root=self.index_root, rebuild=self.rebuild,
                               dense_model=self.dense_model, domain=self.domain)
        self.engines.build(self.strategy.engines)      # a missing artifact stops the run here
        return self

    def toolbox(self, query: str):
        """The bound tools for one question (tests read this; an episode binds its own)."""
        from agent_search.tools.base import EpisodeState
        state = EpisodeState(question=query)
        engine_map = {k: self.engines.get(k) for k in self.strategy.engines}
        return self.strategy.toolbox(state, self._units, self._ubyid, engine_map, files=self._files, corpus_key=self._key)

    # --- one question -----------------------------------------------------------------------------
    def search(self, query: str, k: int) -> list:
        import agent_search.agent.backbone as backends
        backends.reset_usage()
        ctx = HarnessContext(condition=self.condition, units=self._units, ubyid=self._ubyid,
                             engines={e: self.engines.get(e) for e in self.strategy.engines},
                             generate=self._generate, policy_for=self._policy_for, max_steps=self.max_steps,
                             files=self._files, corpus_key=self._key, driver=self._driver,
                             model=self._model, api_base=self._api_base, field_profile=self._field_profile)
        res = self.strategy.harness.run(query, ctx)
        meta = trajectory_meta(res.trajectory, res.surfaced)
        if res.members:
            meta["members"] = [trajectory_meta(m.trajectory, m.surfaced) for m in res.members]
        self._tl.traj, self._tl.meta = res.trajectory, meta
        return res.located


def build_condition_agent(cfg, condition_name: str):
    """A factory for the evaluation: `agent_<condition>` from a RetrieverConfig. A floor is the
    registered retriever its strategy names; anything else runs through `ConditionAgent`."""
    cond = get_condition(condition_name)
    if cfg.prompt_override:
        cond = Condition(name=cond.name, task=task_for_override(cfg.prompt_override, cond), strategy=cond.strategy)
    if cond.strategy.retriever:
        from agent_search.retrievers.registry import build_factory
        return build_factory(cond.strategy.retriever, cfg)
    from agent_search.evaluation.datasets import default_dense_model
    dense_model = cfg.dense_model or default_dense_model(cond.domain)
    profile = cfg.field_profile or cond.domain
    driver = "loop"
    gen = None
    if cfg.policy == "stub":
        from agent_search.agent.policies import KeywordPolicy
        policy_for = lambda c: KeywordPolicy(c.tool_names)  # noqa: E731
    else:
        from agent_search.agent.policies import AgentPolicy
        from agent_search.agent.backbone import DEFAULT_MODEL, is_gemini_model, is_openai_model, make_generate
        mdl = cfg.model or DEFAULT_MODEL
        gen = make_generate(model=mdl, backend=cfg.backend, api_base=cfg.api_base, tp=cfg.tp,
                            temperature=cfg.temperature, seed=cfg.seed)
        policy_for = lambda c: AgentPolicy(generate=gen, system=c.render(profile))  # noqa: E731
        if is_openai_model(mdl) or is_gemini_model(mdl) or cfg.backend == "api":
            try:
                import agents  # noqa: F401
                driver = "sdk"
            except ImportError:
                print("[agent] optional OpenAI Agents SDK not installed (`pip install openai-agents`) — "
                      "falling back to the text-parsed loop driver. Set AGENT_DRIVER=sdk after installing it "
                      "to silence this.")
    return lambda: ConditionAgent(cond, policy_for, generate=gen, max_steps=cfg.max_steps, dense_model=dense_model,
                                  index_root=cfg.index_root, rebuild=cfg.rebuild, driver=driver,
                                  field_profile=profile, model=cfg.model, api_base=cfg.api_base)


def task_for_override(override: str, cond: Condition):
    """The task a `--prompt-profile` override names: a registered task, a registered condition's
    task, or a path to a template file (front matter + body) rendered with the strategy's tools."""
    from agent_search.strategies.conditions import CONDITIONS
    from agent_search.tasks.base import TASKS, Task
    from agent_search.tasks.render import split_front_matter
    if override in TASKS:
        return TASKS[override]
    if override in CONDITIONS:
        return CONDITIONS[override].task
    path = os.path.abspath(override)
    if not os.path.exists(path):
        raise ValueError(f"--prompt-profile {override!r} is not a task, a condition, or a template file")
    with open(path, encoding="utf-8") as fh:
        fm, _ = split_front_matter(fh.read())
    attrs = {"name": str(fm.get("name") or os.path.splitext(os.path.basename(path))[0]),
             "domain": str(fm.get("domain") or cond.domain),
             "message_format": str(fm.get("message_format") or cond.task.message_format),
             "terminal": str(fm.get("terminal") or cond.task.terminal),
             "prompt_file": path}
    return type("OverrideTask", (Task,), attrs)()


def register_conditions() -> None:
    """Make sure every condition in `agent_search.strategies` is registered as the retriever
    `agent_<name>` (registration happens when a condition is declared; this covers a registry
    that was reset). Called by the retriever registry after the built-ins load."""
    from agent_search.strategies.conditions import CONDITIONS, _register_as_retriever
    for cond in CONDITIONS.values():
        _register_as_retriever(cond)


__all__ = ["ConditionAgent", "build_condition_agent", "register_conditions"]
