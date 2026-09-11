"""Run a condition (a task with a strategy) as a retriever the harness can score.

`ConditionAgent` builds the strategy's engines once per corpus, renders the task prompt from
the strategy's tools, and per question binds fresh tools to an episode state and runs one
episode through the agent loop (or the Agents-SDK driver). It returns the surfaced documents
in first-seen order as the ranking, and keeps the trajectory and its metadata for the record.
"""
from __future__ import annotations

import os
import threading
from typing import Callable, Optional, Sequence

from agent_search.core.interfaces import Retriever
from agent_search.corpus.units import CodeUnit
from agent_search.retrievers.engines import Engines
from agent_search.strategies.conditions import Condition, get_condition
from agent_search.tools.base import EpisodeState


class ConditionAgent(Retriever):
    name = "agent"
    returns_full_set = True

    def __init__(self, condition: Condition | str, policy_factory: Callable[[], object], *,
                 max_steps: int = 50, dense_model: Optional[str] = None, index_root: str = "indexes",
                 rebuild: bool = False, driver: str = "loop", field_profile: Optional[str] = None,
                 model: Optional[str] = None, api_base: Optional[str] = None):
        self.condition = get_condition(condition) if isinstance(condition, str) else condition
        self.task, self.strategy = self.condition.task, self.condition.strategy
        self.domain = self.task.domain
        self.toolset = self.strategy.tool_names
        self.tool = f"agent_{self.condition.name}"          # the registered retriever name
        self.prompt_path = self.condition.name                # recorded as the row's prompt_profile_path
        self.tool_description = "toolset: " + ", ".join(self.toolset)
        self._policy_factory = policy_factory
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

    # --- what the harness asks ---------------------------------------------------------------
    @property
    def needs_files(self) -> bool:
        return self.strategy.needs_files

    @property
    def last_trajectory(self):
        return getattr(self._tl, "traj", None)

    @property
    def last_trajectory_meta(self):
        return getattr(self._tl, "meta", None)

    def set_files(self, files: dict) -> None:
        self._files = files or {}

    def system_prompt(self) -> str:
        return self.condition.render(self._field_profile)

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

    def _doc_text(self, doc_id: str):
        u = self._ubyid.get(doc_id)
        if u is None:
            return None
        return "\n".join(p for p in (u.title, u.body or u.code) if p)

    def toolbox(self, query: str):
        state = EpisodeState(question=query)
        engine_map = {k: self.engines.get(k) for k in self.strategy.engines}
        return self.strategy.toolbox(state, self._units, self._ubyid, engine_map, files=self._files, corpus_key=self._key)

    # --- one episode --------------------------------------------------------------------------
    def search(self, query: str, k: int) -> list:
        ws = self.toolbox(query)
        driver = os.environ.get("AGENT_DRIVER") or self._driver
        if driver == "sdk" and self.strategy.sdk:
            traj = self._run_sdk(ws, query)
            self._tl.traj, self._tl.meta = traj, _trajectory_meta(traj, ws)
            return traj.located
        from agent_search.agent.loop import Task as LoopTask, run_episode
        from agent_search.tasks.codefix.guards import fix_guard_for
        import agent_search.models as backends
        from agent_search.training.history import CURRENT, QueryContext, dense_query_style
        from agent_search.training.triples import is_read_action, is_search_action, read_ids
        backends.reset_usage()
        policy = self._policy_factory()
        guard = None
        if self.task.terminal in ("fix", "patch"):
            guard = fix_guard_for(self.toolset, ws)
        ctx = QueryContext(question=query, text_of=self._doc_text, style=dense_query_style())

        def _on_step(step) -> None:
            hits = list(getattr(ws, "last_hits", []) or [])
            step.hit_ids = hits if is_search_action(step.name) else []
            step.read_ids = read_ids({"args": step.args}, hits) if is_read_action(step.name) else []
            ctx.observe(step, hits)

        def _before_tool(name, args, raw) -> None:
            ctx.note(raw)

        token = CURRENT.set(ctx)
        try:
            traj = run_episode(policy, LoopTask(task_id="q", query=query), ws, self._units,
                               max_steps=self.max_steps, usage_fn=backends.usage_events,
                               domain=self.domain, before_tool=_before_tool, fix_guard=guard, on_step=_on_step)
        finally:
            CURRENT.reset(token)
        self._tl.traj, self._tl.meta = traj, _trajectory_meta(traj, ws)
        return traj.located

    def _run_sdk(self, ws, query: str):
        import re as _re
        from datetime import date
        from agent_search.agent.loop import Step, Trajectory, resolve_locations
        from agent_search.agent.sdk_driver import run_episode_sdk
        from agent_search.models import DEFAULT_MODEL
        system = _re.sub(r"\n{3,}", "\n\n", self.system_prompt()).strip()
        system = system.replace("{{step_budget}}", str(self.max_steps))
        user_input = f"Current date: {date.today().isoformat()}\n\n{query}"
        sdk = run_episode_sdk(ws, user_input, model=self._model or DEFAULT_MODEL, api_base=self._api_base,
                              max_turns=self.max_steps, instructions=system)
        steps = [Step(name=n, args=a, observation=o) for (n, a, o) in sdk.steps]
        if steps:
            steps[0].prompt_tokens = sdk.first_input_tokens
        located = list(getattr(ws, "last_hits", []) or []) or list(getattr(ws, "seen", []) or [])
        if self.task.terminal in ("fix", "patch") and sdk.fix_text:
            from agent_search.evaluation.fix_scoring import fix_file
            f = fix_file(sdk.fix_text)
            located = resolve_locations([f], self._units) if f else located
        return Trajectory(task_id="q", steps=steps, located=located, declared=[], llm_calls=sdk.requests,
                          prompt_tokens=sdk.input_tokens, completion_tokens=sdk.output_tokens,
                          cached_input_tokens=sdk.cached_input_tokens, reasoning_tokens=sdk.reasoning_tokens,
                          stopped_reason=("fix" if sdk.fix_text else getattr(sdk, "stopped_reason", "answer")),
                          final_answer=sdk.final_answer, fix_text=sdk.fix_text, elicitation=sdk.elicitation)


class ProcedureAgent(Retriever):
    """A loop-free strategy (one-shot RAG): the procedure ranks, prompts once, and the answer is
    read from the output the way the task reads it. Recorded as one step named `retrieve`."""
    name = "agent"
    returns_full_set = True

    def __init__(self, condition: Condition | str, generate: Optional[Callable], *,
                 dense_model: Optional[str] = None, index_root: str = "indexes", rebuild: bool = False):
        self.condition = get_condition(condition) if isinstance(condition, str) else condition
        self.task, self.strategy = self.condition.task, self.condition.strategy
        self.domain = self.task.domain
        self.toolset = ()
        self.tool = f"agent_{self.condition.name}"          # the registered retriever name
        self.prompt_path = self.condition.name                # recorded as the row's prompt_profile_path
        self.tool_description = "procedure: " + self.strategy.description
        self._generate = generate
        self.dense_model, self.index_root, self.rebuild = dense_model, index_root, rebuild
        self._units: Sequence = ()
        self._ubyid: dict = {}
        self.engines: Optional[Engines] = None
        self._tl = threading.local()

    needs_files = False

    @property
    def last_trajectory(self):
        return getattr(self._tl, "traj", None)

    @property
    def last_trajectory_meta(self):
        return getattr(self._tl, "meta", None)

    def set_files(self, files: dict) -> None:
        pass

    def system_prompt(self) -> str:
        from agent_search.strategies.rag import SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def index(self, units: Sequence[CodeUnit], key: Optional[str] = None) -> "ProcedureAgent":
        if getattr(units, "lazy", False):
            self._units, self._ubyid = units, units.by_id
        else:
            self._units = list(units)
            self._ubyid = {u.doc_id: u for u in self._units}
        self.engines = Engines(self._units, key, index_root=self.index_root, rebuild=self.rebuild,
                               dense_model=self.dense_model, domain=self.domain)
        self.engines.build(self.strategy.engines)
        return self

    def search(self, query: str, k: int) -> list:
        from agent_search.agent.loop import Step, Trajectory, _extract_answer
        import agent_search.models as backends
        backends.reset_usage()
        engine_map = {e: self.engines.get(e) for e in self.strategy.engines}
        doc_ids, msgs, raw = self.strategy.procedure.run(query, engine_map, self._ubyid, self._generate)
        totals = backends.usage_totals()
        step = Step(name="retrieve", args={"query": query, "k": len(doc_ids)},
                    observation=f"({len(doc_ids)} matches stuffed into one prompt)", raw_output=raw or "",
                    prompt_tokens=int(totals.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(totals.get("completion_tokens", 0) or 0))
        traj = Trajectory(task_id="q", steps=[step], located=list(doc_ids), declared=[],
                          llm_calls=int(totals.get("llm_calls", 0) or 0),
                          prompt_tokens=step.prompt_tokens, completion_tokens=step.completion_tokens,
                          cached_input_tokens=int(totals.get("cached_input_tokens", 0) or 0),
                          reasoning_tokens=int(totals.get("reasoning_tokens", 0) or 0),
                          stopped_reason="answer" if raw else "no_model",
                          final_answer=_extract_answer(raw or ""))
        self._tl.traj, self._tl.meta = traj, _trajectory_meta(traj, None)
        return traj.located


def trajectory_meta(traj, ws=None) -> dict:
    """Serialize an episode into the rows.jsonl shape the analysis tools expect.

    `ws` (the episode's toolbox) supplies the surfaced-doc set used for gold-doc coverage
    on the document domain; the code domain carries its <fix> text for fix-scoring."""
    import re
    steps = []
    for s in traj.steps:
        # the count from a search-shaped observation header: "(N units in M files ...)"
        # (code search->fetch), "(N matches ...)" (doc), or "N units matched /pat/" (code
        # grep baseline); 0 otherwise (dci's bash/read have no such header). Advisory only.
        m = (re.search(r"\((\d+)\s+(?:units|matches)\b", s.observation)
             or re.match(r"(\d+)\s+units matched\b", s.observation))
        n_hits = int(m.group(1)) if m else 0
        steps.append({
            "action": s.name, "args": s.args,
            "query": (s.args.get("query") or s.args.get("q") or ""),
            "observation": s.observation, "raw_output": s.raw_output, "n_hits": n_hits,
            "hit_ids": list(getattr(s, "hit_ids", []) or []),
            "read_ids": list(getattr(s, "read_ids", []) or []),
            "t_llm_s": round(s.t_llm, 3), "t_tool_s": round(s.t_tool, 3),
            "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens,
        })
    meta = {
        "queries": [s["query"] for s in steps],
        "actions": [s["action"] for s in steps],
        "hits_per_step": [s["n_hits"] for s in steps],
        "stopped": traj.stopped_reason, "final_answer": traj.final_answer,
        # provenance of a non-empty final_answer on a force-answer-gated episode: None (the
        # episode never reached the reserved final turn), "nudge" (model complied with the
        # inline budget nudge directly), "prefill_inline" (the shared forced-answer-elicitation call
        # filled it in, agent_search/agent/forced_answer.py), "prefill_failed" (that call fired but
        # produced nothing usable). SDK-driven episodes carry the analogous "ask_retry_inline"/
        # "ask_retry_failed" (see agent_search/agent/sdk_driver.py). See loop.py::Trajectory.
        "elicitation": getattr(traj, "elicitation", None),
        "declared": traj.declared, "llm_calls": traj.llm_calls, "n_steps": len(traj.steps),
        "prompt_tokens": traj.prompt_tokens, "completion_tokens": traj.completion_tokens,
        "cached_input_tokens": traj.cached_input_tokens,
        "reasoning_tokens": traj.reasoning_tokens,
        "fix_text": traj.fix_text,
        # docs surfaced this episode (search hits + fetch/visit targets), for gold-doc coverage.
        "surfaced_docs": sorted(getattr(ws, "seen", set()) or set()),
        # every tool response the agent saw lives, in full and uncapped, on
        # `trajectory[i]["observation"]` (see agent_search.evaluation.rows.observations_of).
        "trajectory": steps,
    }
    return meta


_trajectory_meta = trajectory_meta


def build_condition_agent(cfg, condition_name: str):
    """A factory for the harness: `agent_<condition>` from a RetrieverConfig."""
    cond = get_condition(condition_name)
    if os.environ.get("SKIMSEARCHAGENT_LEGACY_RUNNER") == "1":
        # the pre-0.3 agent implementation in agent_search.legacy, for a like-for-like
        # comparison against the current code on one setting.
        from agent_search.legacy.retriever import _build_legacy_agent
        return _build_legacy_agent(cfg, f"agent_{condition_name}")
    if cfg.prompt_override:
        cond = Condition(name=cond.name, task=task_for_override(cfg.prompt_override, cond), strategy=cond.strategy)
    from agent_search.evaluation.datasets import default_dense_model
    dense_model = cfg.dense_model or default_dense_model(cond.domain)
    if not cond.strategy.loop:
        return _build_loop_free(cfg, cond, dense_model)
    profile = cfg.field_profile or cond.domain
    system = cond.render(profile)
    driver = "loop"
    if cfg.policy == "stub":
        from agent_search.agent.policies import KeywordPolicy
        policy_factory = lambda: KeywordPolicy(cond.tool_names)  # noqa: E731
    else:
        from agent_search.agent.policies import AgentPolicy
        from agent_search.models import DEFAULT_MODEL, is_gemini_model, is_openai_model, make_generate
        mdl = cfg.model or DEFAULT_MODEL
        gen = make_generate(model=mdl, backend=cfg.backend, api_base=cfg.api_base, tp=cfg.tp,
                            temperature=cfg.temperature, seed=cfg.seed)
        policy_factory = lambda: AgentPolicy(generate=gen, system=system)  # noqa: E731
        if is_openai_model(mdl) or is_gemini_model(mdl) or cfg.backend == "api":
            try:
                import agents  # noqa: F401
                driver = "sdk"
            except ImportError:
                print("[agent] optional OpenAI Agents SDK not installed (`pip install openai-agents`) — "
                      "falling back to the text-parsed loop driver. Set AGENT_DRIVER=sdk after installing it "
                      "to silence this.")
    return lambda: ConditionAgent(cond, policy_factory, max_steps=cfg.max_steps, dense_model=dense_model,
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


def _build_loop_free(cfg, cond: Condition, dense_model: str):
    """A retrieval-only floor is the registered retriever it names; a procedure runs through
    `ProcedureAgent` with one generate callable (None under the stub policy: the answer is empty)."""
    strat = cond.strategy
    if strat.retriever:
        from agent_search.retrievers.registry import build_factory
        return build_factory(strat.retriever, cfg)
    gen = None
    if cfg.policy != "stub":
        from agent_search.models import DEFAULT_MODEL, make_generate
        gen = make_generate(model=cfg.model or DEFAULT_MODEL, backend=cfg.backend, api_base=cfg.api_base,
                            tp=cfg.tp, temperature=cfg.temperature, seed=cfg.seed)
    return lambda: ProcedureAgent(cond, gen, dense_model=dense_model, index_root=cfg.index_root, rebuild=cfg.rebuild)


def register_conditions() -> None:
    """Make sure every condition in `agent_search.strategies` is registered as the retriever
    `agent_<name>` (registration happens when a condition is declared; this covers a registry
    that was reset). Called by the retriever registry after the built-ins load."""
    from agent_search.strategies.conditions import CONDITIONS, _register_as_retriever
    for cond in CONDITIONS.values():
        _register_as_retriever(cond)


__all__ = ["ConditionAgent", "ProcedureAgent", "build_condition_agent", "register_conditions"]
