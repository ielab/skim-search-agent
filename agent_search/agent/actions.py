"""Tool-call parsing for the agent loop.

Each turn the agent emits a Tongyi-style tool call —
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` — or a bare JSON object
(common with non-Tongyi models). ``parse_tool_call`` extracts the LAST one as
(name, arguments); the loop dispatches it to the workspace, and ``submit`` / an
``<answer>`` ends the episode.

The model frequently truncates a call (generation cut off, or a dropped brace)
so the JSON is near-valid but not quite — most commonly a single missing closing
``}``. ``_repair_load`` recovers these: it closes brackets/strings left open and
strips dangling trailing commas, then re-validates with ``json.loads`` so garbage
(e.g. a `<think>` block that just narrates a call in prose) is never accepted.
"""
from __future__ import annotations

import json
import re

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_CALL_START = re.compile(r'\{\s*"name"\s*:', re.IGNORECASE)


def _last_json_object(s: str) -> str | None:
    """The LAST balanced {...} object in s, string-literal-aware (braces inside a
    JSON string value are not structural). Scans forward, keeping the last top-level
    object — so a <think>-quoted call doesn't win over the real trailing call, and a
    value like {"query": "calls(x) }"} parses correctly."""
    best = None
    i, n = 0, len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            c = s[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    best = s[i:j + 1]
                    break
            j += 1
        i = j + 1 if j > i else i + 1
    return best


def _repair_load(raw: str | None) -> dict | None:
    """Best-effort repair of a near-valid, truncated JSON object: string-literal-aware
    (brackets/commas inside string values are never touched, backslash-escapes are
    honored), it closes any ``{``/``[`` left open at the point of truncation, closes
    an unterminated trailing string, and strips a dangling trailing comma before a
    ``}``/``]`` (or at the very end). Re-validates with ``json.loads`` afterwards, so
    a repair that still doesn't yield valid JSON (e.g. truncation mid-key, or plain
    prose that was never real JSON) correctly returns None rather than fabricating
    a call."""
    if not raw:
        return None
    out: list[str] = []
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            # a trailing comma immediately before this closer is dangling - drop it.
            k = len(out) - 1
            while k >= 0 and out[k].isspace():
                k -= 1
            if k >= 0 and out[k] == ",":
                del out[k:]
            if stack:
                stack.pop()
            out.append(ch)
        else:
            out.append(ch)
    repaired = "".join(out)
    if in_str:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired)  # dangling trailing comma at truncation
    closers = {"{": "}", "[": "]"}
    repaired += "".join(closers[b] for b in reversed(stack))
    try:
        data = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _repair_candidate(text: str) -> str | None:
    """Locate a plausible (but unbalanced) call to repair when neither a <tool_call>
    block nor a balanced bare object was found - e.g. generation was cut off before
    a closing </tool_call>. Scoped to the text after the last <tool_call> opening
    tag when one exists (so unrelated prose, e.g. inside <think>, that never actually
    opened a call isn't misread as one); otherwise the whole text."""
    opens = list(re.finditer(r"<tool_call>", text, re.IGNORECASE))
    scope = text[opens[-1].end():] if opens else text
    starts = list(_CALL_START.finditer(scope))
    return scope[starts[-1].start():] if starts else None


def parse_tool_call(text: str | None) -> tuple[str, dict] | None:
    """Extract the LAST tool call from a generation as (name, arguments). Prefers
    <tool_call> blocks (official Tongyi shape); falls back to the last balanced JSON
    object (a bare call without the wrapper). Returns None when there is no
    parseable call — a <think> block that merely quotes a call does not count,
    because the real call is always the last balanced object."""
    if not text:
        return None
    blocks = _TOOL_CALL.findall(text)
    raw = blocks[-1] if blocks else (_last_json_object(text) if '"name"' in text else None)
    if not raw:
        # Nothing balanced anywhere (e.g. generation was cut off before a closing
        # brace AND before a closing </tool_call>) -> last-resort repair on the
        # trailing, unbalanced call-shaped span.
        raw = _repair_candidate(text)
        data = _repair_load(raw) if raw else None
        if not isinstance(data, dict):
            return None
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # a <tool_call> block whose JSON didn't parse (e.g. the non-greedy regex
            # truncated at a literal </tool_call> inside a string value) -> retry with
            # the string-literal-aware balanced-object scanner over the whole text.
            alt = _last_json_object(text) if '"name"' in text else None
            data = None
            if alt and alt != raw:
                try:
                    data = json.loads(alt)
                except json.JSONDecodeError:
                    data = _repair_load(alt)
            if data is None:
                # last resort: repair near-valid JSON (e.g. a single missing closing
                # brace, a dangling trailing comma, an unterminated string) and
                # re-validate - the primary fix for the ~20% of steps a strict parser
                # drops as "none".
                data = _repair_load(raw)
            if not isinstance(data, dict):
                return None
    if not isinstance(data, dict):
        return None
    name = str(data.get("name", "")).strip()
    args = data.get("arguments")
    if not isinstance(args, dict):
        args = {}
    return (name, args) if name else None
