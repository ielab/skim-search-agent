"""The agent loop: one episode, driven the same way for every condition.

The agent drives a tool box (`agent_search.tools.base.ToolBox`, see `WorkspaceLike`
below for the exact contract this loop needs from it), whose bound tools are the
run's toolset, set by the condition's strategy. There is no separate code path for a
one-tool toolset versus a larger one; the toolset is just configuration.

Each turn, the policy proposes a raw generation, the loop parses one tool call and
runs it on the tool box, and the observation is fed back. The episode ends when the
agent submits, emits an `<answer>`, or hits `max_steps`. The ranking is the agent's
declared locations if it submitted, otherwise the accumulated units its search tools
surfaced, so a search-only toolset with no submit still yields a ranking.
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
_EMPTY_CALL = re.compile(r"<tool_call>\s*</tool_call>", re.IGNORECASE)
# BrowseComp's own answer format ("Explanation: ... Exact Answer: X  Confidence: 95%"), which
# some backbones (OpenResearcher) write instead of the <answer> tag: a reply with no tool call
# and such a line is the final answer, not a turn to nudge
_LABELLED_ANSWER = re.compile(r"^\s*\**\s*(?:Exact|Final)\s+Answer\s*\**\s*:\s*\**\s*(.+?)\s*$",
                              re.IGNORECASE | re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FIX = re.compile(r"<fix>(.*?)</fix>", re.DOTALL | re.IGNORECASE)

# --- proactive context-budget early stop -------------------------------------------------------
# The reserved-final-turn nudge below (the `force_answer` block) only fires once `max_steps` is
# exhausted, which is reactive. On a deep-research episode whose accumulated search/fetch history
# is already large, the turn that finally hits `max_steps - 1` can already be sitting near the
# model's context window; injecting the nudge then and requesting one more (possibly ~12k-token,
# see MAX_VISIT_TOKENS in agent_search/tools/budgets.py) observation before it can trip vLLM's
# "prompt + requested_output > max-model-len" 400 error. These two knobs make the same nudge and
# elicitation machinery fire proactively, by observed token budget instead of step count, while
# there is still headroom:
#   AGENT_CTX_WINDOW    - the model's context window in tokens (adaptive to the backbone actually
#                          serving this run; a GPT run sets its own window). Default 131072,
#                          Tongyi's served --max-model-len.
#   AGENT_CTX_STOP_FRAC - fraction of that window at which to force the answer. Default 0.90,
#                          deliberately generous so the agent can use most of its budget. A value
#                          >= 1.0 disables the proactive early stop; only the reactive max_steps
#                          nudge remains.
# Read once per episode, not at import time, so a per-run env override (a different served
# backbone, for instance) is honored without a process restart.
DEFAULT_CTX_WINDOW = 131072
DEFAULT_CTX_STOP_FRAC = 0.90

# The reserved final turn's message. FORCED_ANSWER_NUDGE (agent.forced_answer_nudge) replaces it;
# the ITER files carry DIVER's "Retrieval complete. You are forbidden to call any tools now. ..."
DEFAULT_BUDGET_NUDGE = ("STEP BUDGET REACHED — this is your FINAL turn. Do NOT search or fetch again. Give your "
                        "single best-effort answer NOW based on everything you have seen, as <answer>your "
                        "answer</answer>. If unsure, commit your most likely answer — a best guess scores "
                        "better than an empty answer.")


def budget_nudge() -> str:
    return os.environ.get("FORCED_ANSWER_NUDGE") or DEFAULT_BUDGET_NUDGE


def _last_prompt_tokens(usage_fn: Optional[Callable[[], list]]) -> int:
    """The most recently observed prompt_tokens: the size of the last real generate() call's
    prompt. Read live off `usage_fn` (in production, `agent_search.agent.backbone.usage_events`, a
    thread-local list appended to by every real backend call; see
    `agent_search/agent/backbone/usage.py::_record_usage`). This is the running context-size proxy the
    early-stop check below compares against the budget threshold. Never raises: an episode with
    no usage_fn, or one that has not produced any events yet, never trips the early stop."""
    if usage_fn is None:
        return 0
    try:
        events = usage_fn()
    except Exception:
        return 0
    return int(events[-1][0]) if events else 0


def _extract_answer(text: str) -> str:
    """Content of the last-opened `<answer>` tag. Models routinely name the tag in prose first
    ('...put the short answer span inside <answer> tags... Thus: <answer>Galați</answer>'). A
    non-greedy `<answer>(.*?)</answer>` would then pair that prose tag with the real closing tag,
    capturing everything in between as the match, which misextracts a large share of answers with
    the real one buried at the end. Anchoring on the last `<answer>` opening recovers the true
    span."""
    if not text:
        return ""
    idx = text.rfind("<answer>")
    if idx < 0:
        return ""
    tail = text[idx + len("<answer>"):]
    end = tail.find("</answer>")
    return (tail[:end] if end >= 0 else tail).strip()


class WorkspaceLike(Protocol):
    """The contract `run_episode` needs from a tool box: a `run` method and a `surfaced`
    property. `agent_search.tools.base.ToolBox` satisfies this by duck typing rather than
    by inheriting from a common base class, so this Protocol is the interface."""

    def run(self, name: str, args: dict) -> str:
        """Dispatch one tool call by name; return the text observation fed back
        to the policy (a tool error is itself a returned observation, never a
        raised exception)."""
        ...

    @property
    def surfaced(self) -> Sequence[str]:
        """Doc ids the episode has surfaced so far, in first-seen order: the agent's
        retrieval ranking for the rank metrics. Every document tool box provides it (see
        ``agent_search.tools.seen.OrderedSeen``). A tool box with no location ranking (the
        code-fix task, scored on its <fix> block) may omit it; the loop then reads an
        empty ranking via ``getattr(..., "surfaced", [])``."""
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
    finish_reason: Optional[str] = None   # the served model's finish reason for this turn (stop | length | ...)


@dataclass
class Trajectory:
    task_id: str
    steps: List[Step] = field(default_factory=list)
    located: List[str] = field(default_factory=list)    # final doc_id ranking
    declared: List[str] = field(default_factory=list)   # raw declared locations
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_input_tokens: int = 0     # of prompt_tokens, the prefix-cache hits (billed at a fraction of full price)
    reasoning_tokens: int = 0        # of completion_tokens, the thinking subset (OpenAI reasoning
                                     # models; Tongyi/vLLM put it in <think>...</think>, counted in run_eval)
    stopped_reason: str = "max_steps"
    final_answer: str = ""
    fix_text: str = ""                                  # the <fix> block (code-fix task)
    # provenance of a non-empty final_answer on a force-answer-gated (non-code) episode:
    #   None              - the episode never reached the reserved final turn (answered,
    #                        submitted, or stopped organically before the budget nudge), or this
    #                        field was not recorded for the row.
    #   "nudge"           - the model complied with the inline "STEP BUDGET REACHED" nudge directly:
    #                       it produced a usable <answer> on that same forced turn, no extra call.
    #   "prefill_inline"  - the nudge turn still had no <answer>. The shared forced-answer
    #                       elicitation call (agent_search/agent/forced_answer.py, assistant-prefill
    #                       via vLLM's continue_final_message) filled it in.
    #   "prefill_failed"  - the inline elicitation call fired but produced nothing usable (or the
    #                       episode's policy has no client or model to call it with, e.g. in-process
    #                       vLLM). final_answer stays "". The episode never crashes either way.
    # The SDK driver's analogous field is sdk_driver.SdkTrajectory.elicitation ("ask_retry_inline"
    # or "ask_retry_failed"). See that module's docstring for why it cannot use this same mechanism:
    # a hosted API model has no assistant-prefill affordance.
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
                before_tool: Optional[Callable[[str, dict, str], None]] = None,
                terminal: str = "answer") -> Trajectory:
    """Drive one episode over the tool box's enabled toolset.

    `fix_guard` (code-fix task only) gates the terminal <fix> block: called with
    (fix_text, steps) it returns (ok, why). An un-ok fix is bounced back as a
    tool_response, so a guess never ends the episode, instead of terminating. When None,
    a <fix> is accepted as-is (or there is no <fix> terminal at all for this task).

    `before_tool(name, args, raw_output)`, when given, is called right before a tool runs.
    `on_step`, when given, is called with the just-appended Step after every step the episode
    records (tool, nudge, or terminal); this is the live-demo streaming hook. A raising
    listener is swallowed, since an episode must never die because a spectator did. None
    (the default) runs no listener and leaves the loop otherwise unaffected.

    `terminal` is the task's answer protocol: `answer` (the default) ends the episode on an
    `<answer>` block or a submit call; `text` ends it on the first reply that carries no tool
    call, that reply being the answer, as DIVER's Qwen3.5 and WebExplorer clients do. Under
    `text` a reply the server cut off mid-thought (finish reason `length`) is not an answer: the
    model is told its thought was discarded and the next turn runs with thinking off."""
    steps: List[Step] = []

    def _push(step: Step) -> None:
        steps.append(step)
        if on_step is not None:
            try:
                on_step(step)
            except Exception:  # noqa: BLE001 - a broken listener must never kill the episode
                pass
    reason = "max_steps"
    declared: List[str] = []
    final_answer = ""
    fix_text = ""
    llm_calls = 0
    nudge_injected = False    # True once the reserved-final-turn budget nudge is appended below
    nudge_reason = None       # "max_steps" or "ctx_budget": why the nudge above was injected.
                              # Feeds the post-loop stopped_reason relabel (see below the for-loop).

    # Reserve the last turn to force a best-effort answer: agents, especially on hard
    # deep-research questions, otherwise burn the whole budget searching and emit nothing (a
    # browsecomp run can reach 50 steps with an empty <answer> even when a candidate was
    # surfaced). On the final allowed step, the loop injects a terminal nudge so the model
    # commits what it has instead of searching again. `force_answer` gates this to the
    # answer-terminal conditions (research and doc tasks); the code condition's <fix> terminal
    # is handled separately, so the loop only forces an answer when there is an <answer> contract.
    force_answer = domain != "code"
    # Fill the {{step_budget}} placeholder in the system prompt with the actual budget so the
    # agent can pace itself: it knows how many tool calls it has before it must answer. Done here
    # because run_episode is where max_steps is authoritative; a no-op if the prompt has no
    # placeholder.
    if isinstance(getattr(policy, "system", None), str) and "{{step_budget}}" in policy.system:
        policy.system = policy.system.replace("{{step_budget}}", str(max_steps))

    # Proactive context-budget threshold (see the module-level comment above _last_prompt_tokens).
    # ctx_stop_frac >= 1.0 is the opt-out sentinel: ctx_threshold stays None and the block below
    # never fires, so the forced-final-turn check below only ever triggers on max_steps.
    ctx_window = int(os.environ.get("AGENT_CTX_WINDOW", str(DEFAULT_CTX_WINDOW)))
    ctx_stop_frac = float(os.environ.get("AGENT_CTX_STOP_FRAC", str(DEFAULT_CTX_STOP_FRAC)))
    ctx_threshold = ctx_stop_frac * ctx_window if ctx_stop_frac < 1.0 else None

    for _step_i in range(max_steps):
        # Proactive early-stop check: fires before another normal search/fetch step is taken,
        # using the last observed prompt_tokens (the size of the most recent real generate() call)
        # as the running context-size proxy. A new observation could add up to MAX_VISIT_TOKENS
        # (~12k, agent_search/tools/budgets.py) more, so the loop stops reaching for one once
        # already past the threshold, rather than waiting for max_steps to overflow reactively.
        ctx_budget_hit = (ctx_threshold is not None and not nudge_injected and steps
                          and _last_prompt_tokens(usage_fn) >= ctx_threshold)
        is_forced_final_turn = (_step_i == max_steps - 1) or ctx_budget_hit
        if force_answer and is_forced_final_turn and steps and not nudge_injected:
            _push(Step(
                name="budget", args={},
                observation=f"<tool_response>{budget_nudge()}</tool_response>",
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
        if name == "answer" and "answer" not in (getattr(workspace, "tools", ()) or ()):
            # the model "calls" a tool named answer with its answer as the argument: that is a
            # final answer, not an unknown tool (Tongyi does this a few hundred times per 830 runs)
            text = next((str(v) for v in args.values() if isinstance(v, (str, int, float)) and str(v).strip()), "")
            name, args = "submit", {"answer": text}

        # the code-fix task ends with a <fix> block, not a tool call. It is checked before
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
        # A bare `STOP` (no tool call, no <answer>) is also treated as a terminal signal, so a
        # prompt protocol that ends this way still terminates the episode instead of burning the
        # full step budget. Its ranking is the surfaced hits, since there is no submit or
        # <answer> to declare locations from.
        is_stop = (finals is None and not call
                   and _THINK.sub("", raw or "").strip().upper() == "STOP")
        cut_off = (not call and getattr(policy, "last_finish_reason", None) == "length")
        plain_text = _THINK.sub("", raw or "").strip()
        # a reply that still carries a tool-call block the parser could not read is a malformed
        # call, not an answer (DIVER's client answers it with an invalid-JSON error)
        is_text_answer = (terminal == "text" and finals is None and not call and not is_stop
                          and not cut_off and bool(plain_text) and "<tool_call>" not in plain_text)
        labelled = (_LABELLED_ANSWER.search(plain_text)
                    if (terminal != "text" and finals is None and not call and not is_stop
                        and not cut_off and "<tool_call>" not in plain_text) else None)
        if finals is not None or is_stop or is_text_answer or labelled:   # submit / <answer> / STOP / plain reply / "Exact Answer:"
            reason = "submit" if name == "submit" else ("stop" if is_stop else "answer")
            declared = finals or []
            if name == "submit":
                final_answer = args.get("answer") or ""
            elif is_stop:
                final_answer = ""
            elif is_text_answer:                    # the reply itself, ignoring <think>
                final_answer = plain_text
            elif labelled:                          # "Exact Answer: X" without the tag
                final_answer = labelled.group(1).strip().strip("*").strip()
            else:                                   # <answer>...</answer>, ignoring <think>
                final_answer = _extract_answer(_THINK.sub("", raw or ""))
            _push(Step(name=reason, args=args, observation="(episode ended)",
                      raw_output=raw or "", t_llm=t_llm,
                      finish_reason=getattr(policy, "last_finish_reason", None)))
            break

        t_tool = 0.0
        if cut_off:
            # DIVER's Qwen3.5 client: a turn cut off mid-thought is discarded, and the next turn
            # runs with thinking off so the model gets to the call
            obs = ("ERROR: Your previous thought was too long and has been discarded. Now, skip all "
                   "reasoning and directly provide the <tool_call> or "
                   + ("final answer." if terminal == "text" else "<answer>."))
            if hasattr(policy, "suppress_thinking_once"):
                policy.suppress_thinking_once = True
        elif not call:
            # name the tools: a backbone that knows other names (Tongyi's own are search and
            # visit) otherwise loops on an empty <tool_call></tool_call>
            tool_names = list(getattr(workspace, "tools", ()) or ())
            names = ", ".join(tool_names)
            empty = bool(_EMPTY_CALL.search(raw or ""))
            malformed = "<tool_call>" in (raw or "") and not empty
            if empty:
                # Tongyi opens the Indri episodes with an empty block and repeats it for a median
                # of four turns (16% of that cell's steps). A filled one-line skeleton for the
                # first tool plus one turn without thinking gets it to the call.
                first = tool_names[0] if tool_names else "<tool>"
                obs = ('ERROR: your <tool_call> block was empty. Write the call on one line, for example '
                       f'<tool_call>{{"name":"{first}","arguments":{{"query":"<your query>"}}}}</tool_call>'
                       + (f" (tools: {names})" if names else "") + ', or submit your answer.')
                if hasattr(policy, "suppress_thinking_once"):
                    policy.suppress_thinking_once = True
            else:
                obs = (('ERROR: the JSON inside your <tool_call> did not parse (check the colon after each '
                        'key and the quotes around the value). ' if malformed else 'ERROR: no tool call found. ')
                       + 'Emit ONE <tool_call>{"name":"<tool>","arguments":{"<arg>":"<value>"}}</tool_call>'
                       + (f" using one of these tools: {names}" if names else "")
                       + ', or submit your answer.')
        else:
            if before_tool is not None:
                # what the model said before this call (its notes on the last read) must be
                # visible to a history-conditioned retriever while the tool runs
                before_tool(name, args, raw or "")
            t1 = time.monotonic()
            obs = workspace.run(name, args)
            t_tool = time.monotonic() - t1
        _push(Step(name=name or "none", args=args, observation=obs,
                  raw_output=raw or "", t_llm=t_llm, t_tool=t_tool,
                  finish_reason=getattr(policy, "last_finish_reason", None)))
        if nudge_injected and is_forced_final_turn:
            # The forced final turn (triggered by max_steps or by the context budget) did not
            # terminate organically: the model tool-called again instead of answering. On a
            # max_steps turn this break is a no-op, since the for-loop has no further iterations
            # anyway; on an earlier ctx_budget turn it is load-bearing, since without it the loop
            # would keep taking normal steps up to max_steps, defeating the point of stopping early.
            break

    # Inline forced-answer elicitation. `nudge_injected` is only True when force_answer's
    # reserved-final-turn nudge (above) actually fired. `reason` stays the loop's initial
    # "max_steps" value only when the for-loop ran out without ever hitting a terminal branch,
    # meaning the model tool-called (or emitted nothing parseable) on its forced final turn
    # instead of answering. When the model complies with the nudge directly (reason is "answer"
    # or "submit" on that same forced turn), tag it "nudge" and do nothing further.
    elicitation: Optional[str] = None
    if force_answer and nudge_injected:
        if reason == "max_steps":
            # `reason` is still the loop's initial default here only because the forced final
            # turn, whichever triggered it, never hit a terminal branch. Relabel to "ctx_budget"
            # when that was the trigger, so a proactively-stopped episode is distinguishable in
            # provenance from one that genuinely ran out its full step budget.
            if nudge_reason == "ctx_budget":
                reason = "ctx_budget"
            answer, elicitation = _elicit_inline(policy, task, steps, terminal=terminal)
            if answer:
                final_answer = answer
        elif reason in ("answer", "submit"):
            elicitation = "nudge"

    # Ranking: resolve declared locations when the agent declared any, either a `submit`, or
    # (in a code run) an `<answer>path:func`. For a research `<answer>` the declared text is
    # prose, not locations, so the retrieval ranking is the surfaced evidence; likewise for
    # `STOP` or max_steps, where there is no declaration. Either way, the same @k and set
    # metrics apply. The code-fix task has no location ranking: it is scored on the <fix> block
    # (agent_search.evaluation.fix_scoring), and its tool box exposes no `surfaced` list.
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
        # One usage event per model call. The injected "budget" nudge step is pushed without a
        # model call, so it must be skipped when attributing per-step usage; otherwise every
        # step after the nudge would be credited with the previous turn's tokens.
        llm_steps = [s for s in traj.steps if s.name != "budget"]
        for step, ev in zip(llm_steps, events):
            step.prompt_tokens, step.completion_tokens = ev[0], ev[1]
    return traj


def _elicit_inline(policy, task: Task, steps: List[Step], terminal: str = "answer") -> tuple:
    """Best-effort inline forced-answer elicitation for a budget-exhausted episode whose reserved
    final nudge still produced no `<answer>` (see the `force_answer` block above). Uses the same
    mechanism `scripts/force_answer_backfill.py` runs offline
    (`agent_search.agent.forced_answer.elicit_final_answer`, assistant-prefill via vLLM's
    `continue_final_message`), live, over the episode's own message list and its
    already-configured `AgentPolicy` (`task` and `steps` are the live objects the episode has
    been accumulating, already including the "budget" nudge Step), so this path builds the same
    call the offline backfill script reconstructs from a persisted row.

    Returns `(answer, elicitation_tag)`. `elicitation_tag` is `"prefill_inline"` on success,
    `"prefill_failed"` on any failure: no client or model available on this policy's generate
    callable (in-process vLLM, for instance, has no assistant-prefill affordance), the call
    itself raising, or both the prefill and its one plain-ask fallback coming back empty. Never
    raises.

    Overflow shrink-retry: this call carries the episode's full accumulated history (the reserved
    final turn fires only once the budget is otherwise exhausted, so this is the largest prompt
    any turn in the episode builds), which makes it the call most likely to trip vLLM's "maximum
    context length" 400. It mirrors `AgentPolicy.propose()`'s shrink loop (and
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
                answer, _method_tag, _raw = elicit_final_answer(messages, client, model, terminal=terminal)
                return (answer, "prefill_inline") if answer else ("", "prefill_failed")
            except Exception as e:
                if "maximum context length" not in str(e) or shrink == 3:
                    raise
                ctx = int(ctx * 0.85)
    except Exception:  # noqa: BLE001 - the extra call must never crash an otherwise-complete episode
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
    # the episode. Only a real <answer> in the output does. Strip <think> blocks first.
    # In a CODE run the <answer> carries declared locations (path:func); in a research
    # run it carries prose. The caller decides per domain how to rank (see run_episode).
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
