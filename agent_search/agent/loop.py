"""The agent loop — ONE loop for every condition.

The agent drives a *workspace* — any condition's tool surface (code_fix.py,
code_grep.py, doc_research.py, doc_dci.py — see `WorkspaceLike` below for the exact
contract) — whose enabled tools are the run's *toolset*: a one-element toolset like
``(search_bql,)`` is the single-tool isolation condition; the full SWE-agent set
(± search_bql) is the additive condition. There is no "single vs multi tool" code
path — only configuration.

Each turn: the policy proposes a raw generation, the loop parses ONE tool call and
runs it on the workspace, the observation is fed back. The episode ends when the
agent ``submit``s (or emits ``<answer>``) or hits ``max_steps``. The ranking is the
agent's **declared** locations if it submitted, else the **accumulated** units its
search tools surfaced — so a search-only toolset (no submit) still yields a ranking.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

from agent_search.corpus.units import CodeUnit
from agent_search.agent.actions import parse_tool_call

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FIX = re.compile(r"<fix>(.*?)</fix>", re.DOTALL | re.IGNORECASE)

# --- proactive context-budget early stop -------------------------------------------------------
# The reserved-final-turn nudge below (`force_answer` block) only fires once `max_steps` is
# exhausted — REACTIVE. On a deep-research episode whose accumulated search/fetch history is
# already huge, the turn that finally hits `max_steps - 1` can itself already be sitting near the
# model's context window; injecting the nudge THEN and getting one more (possibly ~12k-token,
# see MAX_VISIT_TOKENS in agent_search/agent/tools/doc_research.py) observation before it is what
# trips vLLM's "prompt + requested_output > max-model-len" 400 (see `run_eval.py`'s "dead cell"
# churn write-up). These two knobs make the SAME nudge/elicitation machinery fire PROACTIVELY,
# by observed token budget instead of step count, while there is still headroom:
#   AGENT_CTX_WINDOW    — the model's context window in tokens (adaptive to the backbone actually
#                          serving this run — e.g. a GPT run sets its own window). Default 131072,
#                          Tongyi's served --max-model-len.
#   AGENT_CTX_STOP_FRAC — fraction of that window at which to force the answer. Default 0.90
#                          (deliberately BIG — let the agent use ~90% of its budget). >= 1.0
#                          disables the proactive early stop; only the reactive max_steps
#                          nudge remains.
# Read once per episode (not at import time) so a per-run env override (e.g. a different served
# backbone) is honored without a process restart.
DEFAULT_CTX_WINDOW = 131072
DEFAULT_CTX_STOP_FRAC = 0.90


def _last_prompt_tokens(usage_fn: Optional[Callable[[], list]]) -> int:
    """The most recently OBSERVED prompt_tokens (the size of the last real generate() call's
    prompt) — read live off `usage_fn` (in production, `agent_search.models.backends.
    usage_events`, a thread-local list appended to by every real backend call; see
    `agent_search/models/backends.py::_record_usage`). This is the running context-size proxy the
    early-stop check below compares against the budget threshold. Never raises — an episode with
    no usage_fn (or one that hasn't produced any events yet) simply never trips the early stop."""
    if usage_fn is None:
        return 0
    try:
        events = usage_fn()
    except Exception:
        return 0
    return int(events[-1][0]) if events else 0


def _extract_answer(text: str) -> str:
    """Content of the LAST-opened `<answer>` tag. Models routinely NAME the tag in prose first
    ('...put the short answer span inside <answer> tags... Thus: <answer>Galați</answer>'), and a
    non-greedy `<answer>(.*?)</answer>` then pairs the PROSE `<answer>` with the REAL closing tag,
    capturing all the junk in between (observed: ~37% of answers mis-extracted, real answer buried
    at the end). Anchoring on the last `<answer>` opening recovers the true span."""
    if not text:
        return ""
    idx = text.rfind("<answer>")
    if idx < 0:
        return ""
    tail = text[idx + len("<answer>"):]
    end = tail.find("</answer>")
    return (tail[:end] if end >= 0 else tail).strip()


class WorkspaceLike(Protocol):
    """The ONLY contract `run_episode` needs from a workspace — every condition's
    tool surface (CodeFixWorkspace, GrepReadWorkspace, DocSearchFetch/Bm25Visit,
    DciWorkspace, ...) satisfies this by duck typing; none of them inherit from a
    common base class, so this Protocol (not a concrete class) IS the interface."""

    def run(self, name: str, args: dict) -> str:
        """Dispatch one tool call by name; return the text observation fed back
        to the policy (a tool error is itself a returned observation, never a
        raised exception)."""
        ...

    @property
    def surfaced(self) -> Sequence[str]:
        """Doc ids the episode has surfaced so far, in first-seen order — the agent's
        retrieval ranking for rank metrics. Every document workspace provides it (see
        ``agent_search.core.seen.OrderedSeen``); a workspace with no location ranking
        (the code-fix task, scored on its <fix> block) may omit it, in which case the
        loop reads an empty ranking via ``getattr(..., "surfaced", [])``."""
        ...


@dataclass
class Task:
    task_id: str
    query: str                       # the issue text / information need


@dataclass
class Step:
    name: str                        # the tool called this turn
    args: dict
    observation: str
    raw_output: str = ""
    t_llm: float = 0.0
    t_tool: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Trajectory:
    task_id: str
    steps: List[Step] = field(default_factory=list)
    located: List[str] = field(default_factory=list)    # final doc_id ranking
    declared: List[str] = field(default_factory=list)   # raw declared locations
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0     # of prompt_tokens, the prefix-cache hits (billed ~10x cheaper)
    reasoning_tokens: int = 0        # of completion_tokens, the thinking subset (OpenAI reasoning
                                     # models; Tongyi/vLLM put it in <think>…</think> — counted in run_eval)
    stopped_reason: str = "max_steps"
    final_answer: str = ""
    fix_text: str = ""                                  # the <fix> block (code-fix task)
    # provenance of a non-empty final_answer on a force-answer-gated (non-code) episode:
    #   None              — the episode never reached the reserved final turn (answered/
    #                        submitted/stopped organically before the budget nudge), or this
    #                        field wasn't recorded for the row.
    #   "nudge"           — the model complied with the inline "STEP BUDGET REACHED" nudge directly
    #                       (produced a usable <answer> on that same forced turn, no extra call).
    #   "prefill_inline"  — the nudge turn still had no <answer>; the shared forced-answer-
    #                       elicitation call (agent_search/agent/forced_answer.py, assistant-prefill
    #                       via vLLM's continue_final_message) filled it in.
    #   "prefill_failed"  — the inline elicitation call fired but produced nothing usable (or the
    #                       episode's policy has no client/model to call it with, e.g. in-process
    #                       vLLM) — final_answer stays "". Episode never crashes either way.
    # The SDK driver's analogous field is sdk_driver.SdkTrajectory.elicitation ("ask_retry_inline"/
    # "ask_retry_failed") — see that module's docstring for why it can't use this same mechanism
    # (a hosted API model has no assistant-prefill affordance).
    elicitation: Optional[str] = None


class Policy:
    """Returns the model's raw generation for (task, history); the loop parses it."""

    def propose(self, task, history: List[Step]) -> str:  # pragma: no cover - iface
        raise NotImplementedError


def run_episode(policy: Policy, task: Task, workspace: WorkspaceLike,
                units: Sequence[CodeUnit], max_steps: int = 50,
                usage_fn: Optional[Callable[[], list]] = None,
                domain: str = "code",
                fix_guard: Optional[Callable[[str, List["Step"]], tuple]] = None,
                on_step: Optional[Callable[[Step], None]] = None,
                before_tool: Optional[Callable[[str, dict, str], None]] = None) -> Trajectory:
    """Drive one episode over the workspace's enabled toolset.

    `fix_guard` (code-fix task only) gates the terminal <fix> block: called with
    (fix_text, steps) it returns (ok, why); an un-ok fix is bounced back as a
    tool_response (so a guess never ends the episode) instead of terminating. When None,
    a <fix> is accepted as-is (or there is no <fix> terminal at all for this task).

    `before_tool(name, args, raw_output)`, when given, is called right before a tool runs.
    `on_step`, when given, is called with the just-appended Step after every step the episode
    records (tool/nudge/terminal) — the live-demo streaming hook. A raising listener is
    swallowed (an episode must never die because a spectator did). None (the default) runs no
    listener and leaves the loop otherwise unaffected."""
    steps: List[Step] = []

    def _push(step: Step) -> None:
        steps.append(step)
        if on_step is not None:
            try:
                on_step(step)
            except Exception:  # noqa: BLE001 — a broken listener must never kill the episode
                pass
    reason = "max_steps"
    declared: List[str] = []
    final_answer = ""
    fix_text = ""
    llm_calls = 0
    nudge_injected = False    # True once the reserved-final-turn budget nudge is appended below
    nudge_reason = None       # "max_steps" or "ctx_budget" — WHY the nudge above was injected;
                              # feeds the post-loop stopped_reason relabel (see below the for-loop).

    # Reserve the LAST turn to FORCE a best-effort answer: agents (esp. on hard deep-research
    # questions) otherwise burn the whole budget searching and emit nothing (observed: browsecomp
    # runs to 50 steps with an empty <answer> even when a candidate was surfaced). On the final
    # allowed step we inject a terminal nudge so the model commits what it has instead of searching
    # again. `force_answer` gates it to the answer-terminal arms (research/doc); the code arm's
    # <fix> terminal is handled separately, so we only force when there IS an <answer> contract.
    force_answer = domain != "code"
    # Fill the {{step_budget}} placeholder in the system prompt with the ACTUAL budget so the agent
    # can pace itself (know how many tool calls it has before it must answer). Done here because
    # run_episode is where max_steps is authoritative; a no-op if the prompt has no placeholder.
    if isinstance(getattr(policy, "system", None), str) and "{{step_budget}}" in policy.system:
        policy.system = policy.system.replace("{{step_budget}}", str(max_steps))

    # Proactive context-budget threshold (see the module-level comment above _last_prompt_tokens).
    # ctx_stop_frac >= 1.0 is the opt-out sentinel: ctx_threshold stays None and the block below
    # never fires, so the forced-final-turn check below only ever triggers on max_steps.
    ctx_window = int(os.environ.get("AGENT_CTX_WINDOW", str(DEFAULT_CTX_WINDOW)))
    ctx_stop_frac = float(os.environ.get("AGENT_CTX_STOP_FRAC", str(DEFAULT_CTX_STOP_FRAC)))
    ctx_threshold = ctx_stop_frac * ctx_window if ctx_stop_frac < 1.0 else None

    for _step_i in range(max_steps):
        # PROACTIVE early-stop check: fires BEFORE another normal search/fetch step is taken, using
        # the last OBSERVED prompt_tokens (the size of the most recent real generate() call) as the
        # running context-size proxy — a new observation could add up to MAX_VISIT_TOKENS (~12k,
        # agent_search/agent/tools/doc_research.py) more, so we stop reaching for one once already
        # past the threshold rather than waiting for max_steps to overflow reactively.
        ctx_budget_hit = (ctx_threshold is not None and not nudge_injected and steps
                          and _last_prompt_tokens(usage_fn) >= ctx_threshold)
        is_forced_final_turn = (_step_i == max_steps - 1) or ctx_budget_hit
        if force_answer and is_forced_final_turn and steps and not nudge_injected:
            _push(Step(
                name="budget", args={},
                observation=("<tool_response>STEP BUDGET REACHED — this is your FINAL turn. Do NOT "
                             "search or fetch again. Give your single best-effort answer NOW based on "
                             "everything you have seen, as <answer>your answer</answer>. If unsure, "
                             "commit your most likely answer — a best guess scores better than an "
                             "empty answer.</tool_response>"),
                raw_output="", t_llm=0.0))
            nudge_injected = True
            nudge_reason = "ctx_budget" if ctx_budget_hit else "max_steps"
        t0 = time.monotonic()
        raw = policy.propose(task, steps)
        t_llm = time.monotonic() - t0
        llm_calls += 1
        call = parse_tool_call(raw)
        name = call[0] if call else ""
        args = call[1] if call else {}

        # the code-fix task ends with a <fix> block (not a tool call). It is checked BEFORE
        # tool-call parsing so a <fix> that also mentions a call in reasoning still terminates.
        # A grounding guard (fix_guard) may bounce an unfounded fix back for a retry.
        if fix_guard is not None or _FIX.search(_THINK.sub("", raw or "")):
            fm = _FIX.search(_THINK.sub("", raw or ""))
            if fm:
                cand = fm.group(1).strip()
                ok, why = (fix_guard(cand, steps) if fix_guard is not None else (True, ""))
                if not ok:
                    # the rejection is fed back to the policy as its next observation
                    _push(Step(name="fix_rejected", args={},
                               observation=f"<tool_response>REJECTED: {why}</tool_response>",
                               raw_output=raw or "", t_llm=t_llm))
                    continue
                reason = "fix"
                fix_text = cand
                final_answer = cand
                _push(Step(name="fix", args={}, observation="(episode ended)",
                          raw_output=raw or "", t_llm=t_llm))
                break

        finals = _final_locations(raw, name, args)
        # the single-tool isolation task terminates with a bare `STOP` (no tool call) once
        # the top hits look right — the loop must honor it, or the episode burns every
        # max_step. Its ranking is the surfaced hits (no submit). (Retained for the
        # deep-research arm's <answer>/STOP terminals; the code arm uses <fix> above.)
        is_stop = (finals is None and not call
                   and _THINK.sub("", raw or "").strip().upper() == "STOP")
        if finals is not None or is_stop:       # submit / <answer> / STOP -> episode end
            reason = "submit" if name == "submit" else ("stop" if is_stop else "answer")
            declared = finals or []
            if name == "submit":
                final_answer = args.get("answer") or ""
            elif is_stop:
                final_answer = ""
            else:                                   # <answer>...</answer>, ignoring <think>
                final_answer = _extract_answer(_THINK.sub("", raw or ""))
            _push(Step(name=reason, args=args, observation="(episode ended)",
                      raw_output=raw or "", t_llm=t_llm))
            break

        t_tool = 0.0
        if not call:
            obs = ('ERROR: no tool call found. Emit ONE <tool_call>{"name":...,'
                   '"arguments":{...}}</tool_call>, or submit your answer.')
        else:
            if before_tool is not None:
                # what the model said before this call (its notes on the last read) must be
                # visible to a history-conditioned retriever while the tool runs
                before_tool(name, args, raw or "")
            t1 = time.monotonic()
            obs = workspace.run(name, args)
            t_tool = time.monotonic() - t1
        _push(Step(name=name or "none", args=args, observation=obs,
                  raw_output=raw or "", t_llm=t_llm, t_tool=t_tool))
        if nudge_injected and is_forced_final_turn:
            # The forced final turn (max_steps- or ctx_budget-triggered) did not terminate
            # organically — the model tool-called again instead of answering. On a max_steps
            # turn this break is a no-op (the for-loop has no further iterations anyway); on an
            # earlier ctx_budget turn it is load-bearing — without it the loop would keep taking
            # normal steps up to max_steps, defeating the point of stopping early.
            break

    # INLINE FORCED-ANSWER ELICITATION. `nudge_injected` is only True when force_answer's
    # reserved-final-turn nudge (above) actually fired; `reason` stays the loop's initial
    # "max_steps" value only when the for-loop ran out without ever hitting a terminal branch —
    # i.e. the model tool-called (or emitted nothing parseable) on its forced final turn instead
    # of answering. When the model complies with the nudge directly (reason is "answer"/"submit"
    # on that same forced turn), tag it "nudge" and do nothing further.
    elicitation: Optional[str] = None
    if force_answer and nudge_injected:
        if reason == "max_steps":
            # `reason` is still the loop's initial default here ONLY because the forced final turn
            # (whichever triggered it) never hit a terminal branch. Relabel to "ctx_budget" when
            # THAT was the trigger, so a proactively-stopped episode is distinguishable in
            # provenance from one that genuinely ran out its full step budget.
            if nudge_reason == "ctx_budget":
                reason = "ctx_budget"
            answer, elicitation = _elicit_inline(policy, task, steps)
            if answer:
                final_answer = answer
        elif reason in ("answer", "submit"):
            elicitation = "nudge"

    # Ranking: resolve DECLARED locations when the agent declared any — a `submit`, or
    # (in a CODE run) an `<answer>path:func`. For a research `<answer>` the declared
    # text is PROSE, not locations, so the retrieval ranking is the surfaced evidence;
    # likewise for `STOP` / max_steps (no declaration). Same @k / set metrics either way.
    # The code-fix task has no location ranking — it is scored on the <fix> block
    # (agent_search.evaluation.fix_scoring), and its workspace exposes no `surfaced` list.
    surfaced = list(getattr(workspace, "surfaced", []) or [])
    declares_locations = bool(declared) and (reason == "submit" or domain != "general")
    located = resolve_locations(declared, units) if declares_locations else surfaced
    traj = Trajectory(task_id=getattr(task, "task_id", "q"), steps=steps,
                      located=located, declared=list(declared), llm_calls=llm_calls,
                      stopped_reason=reason, final_answer=final_answer, fix_text=fix_text,
                      elicitation=elicitation)
    if usage_fn is not None:
        events = usage_fn()
        # events are (prompt_tokens, completion_tokens, cached_input_tokens) per call
        traj.prompt_tokens = sum(e[0] for e in events)
        traj.completion_tokens = sum(e[1] for e in events)
        traj.cached_input_tokens = sum((e[2] if len(e) > 2 else 0) for e in events)
        traj.reasoning_tokens = sum((e[3] if len(e) > 3 else 0) for e in events)
        # One usage event per model call. The injected "budget" nudge step is pushed WITHOUT a
        # model call, so it must be skipped when attributing per-step usage — otherwise every
        # step after the nudge would be credited with the previous turn's tokens.
        llm_steps = [s for s in traj.steps if s.name != "budget"]
        for step, ev in zip(llm_steps, events):
            step.prompt_tokens, step.completion_tokens = ev[0], ev[1]
    return traj


def _elicit_inline(policy, task: Task, steps: List[Step]) -> tuple:
    """Best-effort inline forced-answer elicitation for a budget-exhausted episode whose reserved
    final nudge still produced no `<answer>` (see the `force_answer` block above). Reuses the
    same mechanism `scripts/force_answer_backfill.py` runs offline
    (`agent_search.agent.forced_answer.elicit_final_answer`, assistant-prefill via vLLM's
    `continue_final_message`), live, over the episode's own message list and its
    already-configured `AgentPolicy` (`task`/`steps` are the live objects the episode has been
    accumulating, already including the "budget" nudge Step) — so this path matches what the
    offline backfill script does when it reconstructs the same call from a persisted row.

    Returns `(answer, elicitation_tag)`. `elicitation_tag` is `"prefill_inline"` on success,
    `"prefill_failed"` on any failure (no client/model available on this policy's generate
    callable — e.g. in-process vLLM, which has no assistant-prefill affordance — the call itself
    raising, or both the prefill and its one plain-ask fallback coming back empty). Never raises.

    OVERFLOW SHRINK-RETRY: this call carries the episode's full accumulated history (the reserved
    final turn fires only once the budget is otherwise exhausted, so this is the largest prompt
    any turn in the episode builds), making it the call most likely to trip vLLM's "maximum
    context length" 400. Mirrors `AgentPolicy.propose()`'s shrink loop (and
    `scripts/force_answer_backfill.py::process_condition_dir`'s offline `_work`): it retries up
    to 3 times, shrinking `ctx_chars` 15% each time, so an overflowing final turn gets the same
    chance to fit before falling back to `"prefill_failed"`."""
    try:
        generate = getattr(policy, "generate", None)
        client = getattr(generate, "client", None)
        model = getattr(generate, "model", None)
        build_messages = getattr(policy, "build_messages", None)
        if client is None or model is None or build_messages is None:
            return "", "prefill_failed"
        from agent_search.agent.forced_answer import FORCE_MSG, elicit_final_answer
        from agent_search.agent.policies import default_ctx_tokens
        ctx = getattr(policy, "ctx_tokens", None) or default_ctx_tokens()
        for shrink in range(4):
            messages = build_messages(task, steps, ctx_tokens=ctx)
            messages.append({"role": "user", "content": f"<tool_response>\n{FORCE_MSG}\n</tool_response>"})
            try:
                answer, _method_tag, _raw = elicit_final_answer(messages, client, model)
                return (answer, "prefill_inline") if answer else ("", "prefill_failed")
            except Exception as e:
                if "maximum context length" not in str(e) or shrink == 3:
                    raise
                ctx = int(ctx * 0.85)
    except Exception:  # noqa: BLE001 — the extra call must never crash an otherwise-complete episode
        return "", "prefill_failed"


# --- declared-location resolution -------------------------------------------

def _final_locations(raw: str, name: str, args: dict) -> Optional[list]:
    """A turn is terminal if it submits or answers. Returns the declared location
    list (possibly empty) when terminal, else None."""
    if name == "submit":
        locs = args.get("locations") or args.get("files") or args.get("functions") or []
        if isinstance(locs, str):
            locs = [x for x in re.split(r"[\n,]+", locs) if x.strip()]
        return list(locs)
    # an <answer> the model merely QUOTES inside its <think> reasoning must not end
    # the episode — only a real <answer> in the output does. Strip <think> blocks first.
    # In a CODE run the <answer> carries declared locations (path:func); in a research
    # run it carries prose — the caller decides per domain how to rank (see run_episode).
    visible = _THINK.sub("", raw or "")
    m = _ANSWER.search(visible)
    if m:
        return [x for x in re.split(r"[\n,]+", m.group(1).strip()) if x.strip()]
    return None


def resolve_locations(items: Sequence[str], units: Sequence[CodeUnit]) -> List[str]:
    """Map declared free-form locations ("path:func", "func", "path::qual", "path")
    to unit doc_ids, order + dedup preserved. A file-only declaration resolves to that
    file's units; a bare name matches by qualname suffix ("merge" -> "Media.merge"). A
    `path:func` whose func matches no unit falls back to ALL units in that file
    (deliberately recall-favoring: a near-miss function name still points at the right
    file rather than resolving to nothing)."""
    by_id = {u.doc_id: u for u in units}
    out: List[str] = []
    seen: set = set()

    def add(doc_id: str) -> None:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)

    for raw in items:
        item = (raw or "").strip().strip("`'\"")
        if not item:
            continue
        if item in by_id:
            add(item)
            continue
        path, qual = _split_loc(item)
        cands = units
        if path:
            cands = [u for u in cands if _path_match(u.path, path)]
        if qual and not qual.isdigit():
            qcands = [u for u in cands if _qual_match(u.qualname, qual)]
            cands = qcands or cands if path else qcands
        elif qual and qual.isdigit() and path:
            ln = int(qual)
            cands = [u for u in cands if u.start_line <= ln <= u.end_line] or cands
        for u in cands:
            add(u.doc_id)
    return out


def _split_loc(item: str) -> tuple[Optional[str], Optional[str]]:
    if "::" in item:
        p, q = item.split("::", 1)
        return (p or None), (q or None)
    if ":" in item:
        p, q = item.rsplit(":", 1)
        return (p or None), (q or None)
    if "/" in item or item.endswith(".py"):
        return item, None
    return None, item


def _path_match(unit_path: str, path: str) -> bool:
    path = path.strip().lstrip("./")
    return unit_path == path or unit_path.endswith("/" + path) \
        or unit_path.split("/")[-1] == path.split("/")[-1]


def _qual_match(qualname: str, qual: str) -> bool:
    qual = qual.strip()
    return qualname == qual or qualname.endswith("." + qual) \
        or qualname.split(".")[-1] == qual.split(".")[-1]
