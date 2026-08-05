"""Query-formulation policies — the only place that decides what tool call to emit.

All policies return a raw generation string (a `<tool_call>...`/`<answer>...` text);
the loop parses it. Three flavours:
  - AgentPolicy   : prompt-profile-driven LLM (the paper experiments).
  - ScriptPolicy  : replays a fixed list of (name, args) calls (deterministic tests).
  - KeywordPolicy : no model — one cheap search over the toolset, then submit (if the
                    toolset has submit) or end (so a search-only toolset accumulates).
"""
from __future__ import annotations

import json as _json
from collections import Counter
from datetime import date
from typing import Callable, List, Sequence

from agent_search.corpus.units import code_tokenize
from agent_search.prompts import load_prompt_text

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

class AgentPolicy:
    """Prompt-profile-driven policy. `generate(messages) -> raw text`."""

    def __init__(self, generate: Callable[[list], str], prompt_path: str,
                 max_history: int = 40, ctx_chars: int = 450_000,
                 field_profile: str | None = None):
        self.generate = generate
        self.prompt_path = prompt_path
        # field_profile selects the per-dataset field-tagged manual variant (e.g. structured
        # "wiki"/"browsecomp" vs flat "general"); None -> the task's own domain.
        self.system = load_prompt_text(prompt_path, field_profile)
        self.max_history = max_history
        # ctx_chars: a TOTAL char budget (default ~450k chars =~ 115k tokens) for the
        # kept (assistant, observation) history pairs, not a per-observation cap. This is
        # what makes the model's real 128k-token window (Tongyi-DeepResearch-30B-A3B's
        # max_position_embeddings=131072) usable: sections (median 100-170 tokens) always
        # fit whole; whole docs (median 550-1938 tokens, p95 up to 22k, max ~930k on
        # browsecomp) mostly fit whole too, instead of being silently clipped to ~500
        # tokens by a flat per-observation cap.
        self.ctx_chars = ctx_chars
        self.last_raw = ""

    def build_messages(self, task, history, ctx_chars: int | None = None) -> list:
        system = _re.sub(r"\n{3,}", "\n\n", self.system).strip() + "\n"
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": f"Current date: {date.today().isoformat()}\n\n{task.query}"},
        ]
        # Walk history newest -> oldest, keeping whole (assistant, observation) pairs while
        # a running char total stays under ctx_chars; older steps are dropped once the
        # budget is hit. Only a SINGLE observation that alone exceeds the entire remaining
        # budget gets truncated (rare: e.g. a ~930k-token browsecomp doc) — everything else
        # is kept in full, chronologically ordered, in the final message list.
        kept: list[tuple[str, str]] = []   # (raw_output, observation) chronological once reversed
        budget = ctx_chars if ctx_chars is not None else self.ctx_chars
        for s in reversed(history[-self.max_history:]):
            raw = s.raw_output or ""
            obs = s.observation or ""
            pair_len = len(raw) + len(obs)
            if pair_len <= budget:
                kept.append((raw, obs))
                budget -= pair_len
            elif not kept and len(obs) > budget:
                # this is the newest step, and its observation ALONE overflows the whole
                # budget (rare: e.g. a ~930k-token browsecomp doc): truncate just it,
                # rather than dropping it outright (the model needs SOME view of its most
                # recent tool call), then stop — no room remains for any older step.
                room = max(budget - len(raw), 0)
                obs = obs[:room] + "\n...(truncated)"
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
        # The char budget (~3.9 chars/token heuristic) can overshoot the model's true token
        # window on token-dense content (tables, id lists): vLLM then 400s with "maximum
        # context length". Rather than killing the instance (permanent n<200 hole — e.g.
        # browsecomp__1012 at 127,073 tokens), shrink the window 15% and retry, up to 3
        # times. Episodes that never trip the error build identical messages to before.
        ctx = self.ctx_chars
        for shrink in range(4):
            try:
                self.last_raw = self.generate(self.build_messages(task, history, ctx_chars=ctx))
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
    """No model: drive ANY toolset with cheap keyword queries so a dependency-light run
    (check_conditions / stub tests) exercises the whole arm without a model. It walks the
    arm's script by TOOLSET, never a hardcoded condition name — search->fetch (the method),
    bm25_search->visit (retrieve-then-visit), grep->read (the code baseline), or
    bash->read (the DCI baseline) — then the arm's terminal (<fix> for code, <answer> for
    docs). No API, no embedder."""

    def __init__(self, toolset: Sequence[str], max_keywords: int = 6):
        self.toolset = tuple(toolset)
        self.max_keywords = max_keywords
        self.last_raw = ""

    def propose(self, task, history) -> str:
        ts = set(self.toolset)
        kws = salient_keywords(task.query, self.max_keywords)
        step = len(history)
        if step == 0:                                    # first move: the arm's own opener
            q = kws[0] if kws else "the"
            if "grep" in ts and "search" not in ts:
                self.last_raw = _tool_call("grep", pattern=q)
            elif "bash" in ts and "search" not in ts:
                # a broad recursive CASE-INSENSITIVE grep for the first keyword (-i: the
                # source doc likely capitalizes it, e.g. a title); DciWorkspace surfaces any
                # matched filename from the command/output for gold-doc-coverage bookkeeping.
                self.last_raw = _tool_call("bash", command=f"grep -ril {q!r} .")
            elif "bm25_search" in ts:
                self.last_raw = _tool_call("bm25_search", query=q)
            else:
                self.last_raw = _tool_call("search", query=q)
            return self.last_raw
        # second move: open the first candidate (fetch a part / visit the doc / read a hit)
        if step == 1:
            if "visit" in ts:
                self.last_raw = _tool_call("visit", rank=1)
            elif "fetch" in ts:
                part = _first_fetch_part(history[-1].observation)
                self.last_raw = _tool_call("fetch", specs=[[1, part]] if part else [])
            elif "grep" in ts and "read" in ts:
                path = _first_grep_hit_path(history[-1].observation)
                self.last_raw = _tool_call("read", path=path) if path else "<answer></answer>"
            elif "bash" in ts and "read" in ts:
                path = _first_bash_txt_hit(history[-1].observation)
                self.last_raw = _tool_call("read", path=path) if path else "<answer></answer>"
            else:
                self.last_raw = "<answer></answer>"
            return self.last_raw
        # terminal: the arm's answer shape
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

# a code fetch renders "[1] path/to/file.py :: Qual.name" — pull the first part/path.
_FETCH_HEAD = _re.compile(r"^\[\d+\]\s+(\S+)\s+::\s*(.*)$", _re.MULTILINE)
# a code search lists "  1  path   defs:[A . B]" — first def name in the top file.
_SEARCH_DEFS = _re.compile(r"defs:\[([^\].]+)")


def _first_fetch_part(search_obs: str) -> str:
    """The first def name from a code search observation (for the stub's fetch)."""
    m = _SEARCH_DEFS.search(search_obs or "")
    return m.group(1).strip().split(" . ")[0].strip() if m else ""


def _is_code_arm(history) -> bool:
    """True if a prior fetch produced a code '[n] path :: part' block (code arm shape)."""
    return any(_FETCH_HEAD.search(s.observation or "") for s in history)


def _first_fetch_path(history) -> str:
    for s in history:
        m = _FETCH_HEAD.search(s.observation or "")
        if m and not m.group(2).lstrip().startswith("ERROR"):
            return m.group(1)
    return ""


# the code GREP baseline: `grep()` renders "  path:line: text" per hit line.
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
