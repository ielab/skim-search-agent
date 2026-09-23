"""Query-formulation policies: the only place that decides what tool call to emit.

All policies return a raw generation string (a `<tool_call>...`/`<answer>...` text);
the loop parses it. Three flavours:
  - AgentPolicy   : an LLM driven by a rendered system prompt (the paper experiments).
  - ScriptPolicy  : replays a fixed list of (name, args) calls (deterministic tests).
  - KeywordPolicy : no model, one cheap search over the toolset, then submit (if the
                    toolset has submit) or end (so a search-only toolset accumulates).
"""
from __future__ import annotations

import json as _json
import os
from collections import Counter
from datetime import date
from typing import Callable, List, Sequence

from agent_search.tokens import count_tokens, truncate_tokens
from agent_search.corpus.units import code_tokenize

# History budget (model tokens) for AgentPolicy: see `default_ctx_tokens`.
DEFAULT_CTX_TOKENS = 115_000

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "be", "to", "of", "and", "or", "in", "on", "for",
    "with", "should", "would", "that", "this", "it", "its", "so", "not", "no", "but",
    "if", "when", "then", "as", "at", "by", "from", "was", "were", "has", "have", "had",
    "def", "return", "self", "none", "true", "false", "class", "import", "you", "we",
    "never", "out", "time", "up", "down", "all", "any", "into", "over",
    # interrogatives: a question's first token is otherwise picked as the "salient" query
    # (e.g. "Which treaty ended ..." -> bm25_search("which") -> 0 hits on every doc fixture)
    "which", "what", "who", "whom", "whose", "where", "why", "how",
}


def salient_keywords(text: str, n: int = 6) -> List[str]:
    toks = [t for t in code_tokenize(text) if t not in _STOP_WORDS and len(t) > 2]
    return [w for w, _ in Counter(toks).most_common(n)]



def _tool_call(name: str, **args) -> str:
    return "<tool_call>%s</tool_call>" % _json.dumps({"name": name, "arguments": args})


# --- LLM policy --------------------------------------------------------------

def max_tokens_schedule() -> list:
    """LLM_MAX_TOKENS_SCHEDULE as a list of ints ("4096,2048,1024"); empty when unset."""
    raw = (os.environ.get("LLM_MAX_TOKENS_SCHEDULE") or "").strip()
    if not raw:
        return []
    return [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]


def default_max_history() -> "int | None":
    """How many of the most recent (assistant, observation) pairs the prompt may carry.

    ``None`` (the default) means no count cap: the token budget ``ctx_tokens`` alone decides
    how much history fits. ``AGENT_MAX_HISTORY`` sets an explicit cap.

    This used to default to 40 steps. That is a silent count cap, unrelated to the token
    budget, so an episode longer than 40 steps dropped its oldest work from every prompt even
    when the whole conversation fitted the window: on a 51-step BrowseComp episode the agent
    answered without seeing its first 11 steps, and in 20% of those it had already opened the
    gold document in the part it could no longer see. It also broke prefix caching, since the
    window slid by one pair per turn and the prompt prefix changed every time.
    """
    raw = os.environ.get("AGENT_MAX_HISTORY", "").strip()
    if not raw or raw.lower() in ("none", "0", "off", "unlimited"):
        return None
    return int(raw)


def default_ctx_tokens() -> int:
    """The history budget in model tokens. ``AGENT_CTX_TOKENS`` overrides; the default leaves
    headroom inside a 131k-token window (the paper's served backbone) for the system prompt,
    the question, and the next generation. Read at policy construction, not at import."""
    return int(os.environ.get("AGENT_CTX_TOKENS", str(DEFAULT_CTX_TOKENS)))


class AgentPolicy:
    """A rendered-system-prompt policy. `generate(messages) -> raw text`.

    Length is governed in tokens only. ``ctx_tokens`` is the total token budget for the
    (assistant, observation) history pairs kept in the prompt, never a per-observation
    cap and never a character count. Tokens are counted on the library's measurement ruler
    (``agent_search.tokens.count_tokens``: tiktoken ``o200k_base`` when installed,
    whitespace tokens otherwise)."""

    def __init__(self, generate: Callable[[list], str], system: str,
                 max_history: "int | None" = -1, ctx_tokens: int | None = None):
        self.generate = generate
        # `system`: the rendered system prompt (a condition's render).
        self.system = system
        # -1 is the sentinel for 'not specified': take the environment default (no cap).
        self.max_history = default_max_history() if max_history == -1 else max_history
        self.ctx_tokens = int(ctx_tokens) if ctx_tokens is not None else default_ctx_tokens()
        self.last_raw = ""
        # per-turn generation budgets (LLM_MAX_TOKENS_SCHEDULE, "4096,2048,1024": the last value
        # repeats); empty = the backend's own budget every turn
        self.max_tokens_schedule = max_tokens_schedule()
        # set by the loop after a turn was cut off mid-thought: the next call runs with thinking
        # off (DIVER's Qwen3.5 client), then the switch clears
        self.suppress_thinking_once = False
        self.calls = 0
        self.last_finish_reason = None

    def build_messages(self, task, history, ctx_tokens: int | None = None) -> list:
        system = _re.sub(r"\n{3,}", "\n\n", self.system).strip() + "\n"
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Current date: {date.today().isoformat()}\n\n{task.query}"},
        ]
        # Walk history newest -> oldest, keeping whole (assistant, observation) pairs while
        # the running token total stays under the budget; older steps are dropped once the
        # budget is hit. The TOKEN budget is the only limit by default; `max_history` adds an
        # optional count cap on top of it (see `default_max_history`). Only a single
        # observation that alone exceeds the entire remaining budget gets truncated (rare:
        # e.g. a ~930k-token document). Everything else is kept in full, chronologically
        # ordered, in the final message list.
        kept: list[tuple[str, str]] = []   # (raw_output, observation) chronological once reversed
        budget = int(ctx_tokens) if ctx_tokens is not None else self.ctx_tokens
        recent = history if not self.max_history else history[-self.max_history:]
        for s in reversed(recent):
            raw = s.raw_output or ""
            obs = s.observation or ""
            raw_len = count_tokens(raw)
            obs_len = count_tokens(obs)
            pair_len = raw_len + obs_len
            if pair_len <= budget:
                kept.append((raw, obs))
                budget -= pair_len
            elif not kept and obs_len > budget:
                # this is the newest step, and its observation alone overflows the whole
                # budget: truncate just it (by tokens), rather than dropping it outright,
                # since the model needs some view of its most recent tool call. Then stop:
                # no room remains for any older step.
                room = max(budget - raw_len, 0)
                obs = truncate_tokens(obs, room, "\n...(truncated)")
                kept.append((raw, obs))
                budget = 0
                break
            else:
                break   # budget exhausted: older steps are dropped
        for raw, obs in reversed(kept):
            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "user", "content": f"<tool_response>\n{obs}\n</tool_response>"})
        return msgs

    def propose(self, task, history) -> str:
        # The measurement ruler can differ from the serving model's tokenizer, so the budget
        # can still overshoot the true window on token-dense content: the server then rejects
        # the request with "maximum context length". Rather than killing the instance, shrink
        # the window 15% and retry, up to 3 times. Episodes that never trip the error are
        # unaffected by this loop.
        ctx = self.ctx_tokens
        opts = {}
        if self.max_tokens_schedule:
            opts["max_tokens"] = self.max_tokens_schedule[min(self.calls, len(self.max_tokens_schedule) - 1)]
        if self.suppress_thinking_once:
            opts["thinking"] = False
            self.suppress_thinking_once = False
        for shrink in range(4):
            try:
                messages = self.build_messages(task, history, ctx_tokens=ctx)
                self.last_raw = self.generate(messages, **opts) if opts else self.generate(messages)
                self.calls += 1
                self.last_finish_reason = getattr(self.generate, "last_finish_reason", None)
                return self.last_raw
            except Exception as e:
                if "maximum context length" not in str(e) or shrink == 3:
                    raise
                ctx = int(ctx * 0.85)
        return self.last_raw  # unreachable


# --- no-model policies (tests / dependency-light fixture runs) ---------------

class ScriptPolicy:
    """Replays a fixed script of (name, args) tool calls; defaults to an empty submit."""

    def __init__(self, script: Sequence[tuple]):
        self._script = list(script)
        self.last_raw = ""

    def propose(self, task, history) -> str:
        i = len(history)
        if i >= len(self._script):
            self.last_raw = _tool_call("submit", locations=[])
        else:
            name, args = self._script[i]
            self.last_raw = _tool_call(name, **args)
        return self.last_raw


class KeywordPolicy:
    """No model: drive any toolset with cheap keyword queries so a dependency-light run
    (the stub policy, `cfg.policy == "stub"`) exercises a whole condition without a model.
    It walks a script keyed by toolset, never a hardcoded condition name: search->fetch (the
    method), bm25_search->visit (retrieve-then-visit), grep->read (the code baseline), or
    bash->read (the DCI baseline), then the condition's terminal (<fix> for code, <answer>
    for docs). No API, no embedder."""

    def __init__(self, toolset: Sequence[str], max_keywords: int = 6):
        self.toolset = tuple(toolset)
        self.max_keywords = max_keywords
        self.last_raw = ""

    def propose(self, task, history) -> str:
        ts = set(self.toolset)
        kws = salient_keywords(task.query, self.max_keywords)
        step = len(history)
        if step == 0:                                    # first move: the condition's own opener
            q = kws[0] if kws else "the"
            if "grep" in ts and "search" not in ts:
                self.last_raw = _tool_call("grep", pattern=q)
            elif "bash" in ts and "search" not in ts:
                # a broad recursive case-insensitive grep for the first keyword (-i, since the
                # source doc likely capitalizes it, e.g. a title). The Bash tool surfaces any
                # matched filename from the command or its output for gold-doc-coverage bookkeeping.
                self.last_raw = _tool_call("bash", command=f"grep -ril {q!r} .")
            elif "bm25_search" in ts:
                self.last_raw = _tool_call("bm25_search", query=q)
            else:
                self.last_raw = _tool_call("search", query=q)
            return self.last_raw
        # second move: open the first candidate (fetch a part / visit the doc / read a hit).
        # Every condition names its read tool differently (visit, visit_d, visit_h, fetch,
        # fetch_s, fetch_bqld*): pick whichever this toolset has, so the smoke run exercises a read.
        visit_tool = next((t for t in self.toolset if t.startswith("visit")), None)
        fetch_tool = next((t for t in self.toolset if t.startswith("fetch")), None)
        if step == 1:
            if "get_document" in ts:
                m = _re.search(r"DocID:\s*(\S+)", history[-1].observation or "")
                self.last_raw = _tool_call("get_document", docid=m.group(1)) if m else "<answer></answer>"
            elif visit_tool and history[-1].name != "read":
                self.last_raw = _tool_call(visit_tool, rank=1)
            elif fetch_tool:
                part = _first_fetch_part(history[-1].observation) or "(intro)"
                self.last_raw = _tool_call(fetch_tool, rank=1, section=part)
            elif "grep" in ts and "read" in ts:
                path = _first_grep_hit_path(history[-1].observation)
                self.last_raw = _tool_call("read", path=path) if path else "<answer></answer>"
            elif "bash" in ts and "read" in ts:
                path = _first_bash_txt_hit(history[-1].observation)
                self.last_raw = _tool_call("read", path=path) if path else "<answer></answer>"
            else:
                self.last_raw = "<answer></answer>"
            return self.last_raw
        # terminal: the condition's answer shape
        code_arm_via_fetch = (self.toolset[:1] == ("search",) and "fetch" in ts
                              and "visit" not in ts and _is_code_arm(history))
        code_arm_via_grep = "grep" in ts and "read" in ts
        if code_arm_via_fetch:
            path = _first_fetch_path(history)
        elif code_arm_via_grep:
            path = _first_read_path(history)
        else:
            path = ""
        if (code_arm_via_fetch or code_arm_via_grep) and path:
            self.last_raw = (f"<fix>\nfile: {path}\nfunction: (stub)\n"
                             f"change: (stub — no-model policy)\n</fix>")
        else:
            self.last_raw = "<answer></answer>"
        return self.last_raw


import re as _re


from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    """Decides the next raw generation from the task and the steps so far. The loop
    parses ONE tool call (or a terminal ``<answer>``) out of what it returns."""

    def propose(self, task: Any, history: Sequence[Any]) -> str: ...

# a code fetch renders "[1] path/to/file.py :: Qual.name": pull the first part/path.
_FETCH_HEAD = _re.compile(r"^\[\d+\]\s+(\S+)\s+::\s*(.*)$", _re.MULTILINE)
# a code search lists "  1  path   defs:[A . B]": first def name in the top file.
_SEARCH_DEFS = _re.compile(r"defs:\[([^\].]+)")


def _first_fetch_part(search_obs: str) -> str:
    """The first def name from a code search observation (for the stub's fetch)."""
    m = _SEARCH_DEFS.search(search_obs or "")
    return m.group(1).strip().split(" . ")[0].strip() if m else ""


def _is_code_arm(history) -> bool:
    """True if a prior fetch produced a code '[n] path :: part' block (the code condition's
    observation shape)."""
    return any(_FETCH_HEAD.search(s.observation or "") for s in history)


def _first_fetch_path(history) -> str:
    for s in history:
        m = _FETCH_HEAD.search(s.observation or "")
        if m and not m.group(2).lstrip().startswith("ERROR"):
            return m.group(1)
    return ""


# the code grep baseline: `grep()` renders "  path:line: text" per hit line.
_GREP_HIT = _re.compile(r"^\s*(\S+):\d+:", _re.MULTILINE)
# `read()` echoes "path lines s-e of N:" as its first line.
_READ_HEAD = _re.compile(r"^(\S+)\s+lines\s+\d+-\d+\s+of\s+\d+:", _re.MULTILINE)
# the doc DCI baseline: any exported "<doc_id>.txt" name appearing in bash's output
# (e.g. an `ls`/`grep -rl` hit like "./d1.txt" or "d1.txt:3:...").
_TXT_HIT = _re.compile(r"(?:^|[\s/])(\S+\.txt)\b", _re.MULTILINE)


def _first_grep_hit_path(grep_obs: str) -> str:
    """The first path:line hit from a code-grep observation (for the stub's read)."""
    m = _GREP_HIT.search(grep_obs or "")
    return m.group(1) if m else ""


def _first_read_path(history) -> str:
    """The first successfully-read path (code grep baseline's read() echo), for the stub's <fix>."""
    for s in history:
        if s.name != "read" or s.observation.lstrip().startswith("ERROR"):
            continue
        m = _READ_HEAD.search(s.observation or "")
        if m:
            return m.group(1)
    return ""


def _first_bash_txt_hit(bash_obs: str) -> str:
    """The first exported '<doc_id>.txt' filename in a DCI bash observation, for the stub's read."""
    m = _TXT_HIT.search(bash_obs or "")
    return m.group(1) if m else ""


# back-compat aliases (older imports / tests)
LocalizationPolicy = AgentPolicy
StubLocPolicy = ScriptPolicy
KeywordLocPolicy = KeywordPolicy
