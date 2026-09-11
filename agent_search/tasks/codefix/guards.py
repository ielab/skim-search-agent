"""Grounding guards for the code task: a <fix> is accepted only after a successful search and a
read of the file it names. One guard per strategy shape (search then fetch; grep then read),
the same policy in both.
"""
from __future__ import annotations


def _code_fix_guard(ws):
    """Build a fix_guard(fix_text, steps) -> (ok, why) closure for the search/fetch strategy.

    `ws` is the episode's bound ToolBox; it is accepted for a uniform guard signature but
    the guard itself only inspects the step history. A <fix> is accepted only after the
    episode has (a) run a successful search and (b) fetched the exact file the fix names.
    Otherwise it is a guess, and the guard bounces it back with an actionable reason. Path
    matching is suffix-lenient, matching agent_search.evaluation.fix_scoring."""
    from agent_search.evaluation.fix_scoring import fix_file, is_grounded

    def guard(fix_text: str, steps) -> tuple:
        searched = any(
            s.name == "search" and not s.observation.startswith(("ERROR", "empty query"))
            and "(0 hits)" not in s.observation
            for s in steps)
        # a path is 'read' only if that fetch spec's own body was not an ERROR line.
        read_paths: set = set()
        import re as _re
        for s in steps:
            if s.name != "fetch":
                continue
            for block in _re.split(r"\n(?=\[\d+\] )", s.observation.strip()):
                fm = _re.match(r"\[\d+\]\s+(\S+)\s+::[^\n]*\n?(.*)", block, _re.DOTALL)
                if fm and not fm.group(2).lstrip().startswith("ERROR"):
                    read_paths.add(fm.group(1))
        file = fix_file(fix_text)
        if not searched:
            return False, ("you have not run a successful search yet. Do NOT guess the fix. "
                           "Your next message must be a single <tool_call> that searches "
                           "for a symbol from the issue.")
        if not file:
            return False, ("your <fix> block has no parseable 'file:' line. Re-emit it with a "
                           "line exactly like 'file: path/to/file.py' (a path you have "
                           "fetched), then 'function:' and 'change:'.")
        if not is_grounded(file, read_paths):
            return False, (f"'{file}' is not a file you have fetched. fetch the exact file you "
                           "intend to change (a path from a search hit), then re-emit <fix> "
                           "with that path.")
        return True, ""

    return guard


def _code_grep_guard(ws):
    """The grep baseline's fix_guard(fix_text, steps) -> (ok, why). A <fix> is accepted only
    after a successful `grep` (at least one match) and a `read` of the exact file named in
    `file:`. Same policy as `_code_fix_guard`, reading `grep`/`read` observation shapes
    instead of `search`/`fetch` ones."""
    from agent_search.evaluation.fix_scoring import fix_file, is_grounded

    def guard(fix_text: str, steps) -> tuple:
        searched = any(
            s.name == "grep" and not s.observation.startswith("0 matches")
            for s in steps)
        # The `read` tool (source="repo") echoes "{path} lines s-e of N:\n..."; the resolved
        # path is the observation's first token, unless it errored (no such file / ambiguous name).
        read_paths: set = set()
        for s in steps:
            if s.name != "read" or s.observation.lstrip().startswith("ERROR"):
                continue
            first_line = s.observation.split("\n", 1)[0]
            path = first_line.split(" lines ", 1)[0].strip()
            if path:
                read_paths.add(path)
        file = fix_file(fix_text)
        if not searched:
            return False, ("you have not run a successful grep yet. Do NOT guess the fix. "
                           "Your next message must be a single <tool_call> that greps "
                           "for a symbol from the issue.")
        if not file:
            return False, ("your <fix> block has no parseable 'file:' line. Re-emit it with a "
                           "line exactly like 'file: path/to/file.py' (a path you have "
                           "read), then 'function:' and 'change:'.")
        if not is_grounded(file, read_paths):
            return False, (f"'{file}' is not a file you have read. read the exact file you "
                           "intend to change (a path from a grep hit), then re-emit <fix> "
                           "with that path.")
        return True, ""

    return guard


def fix_guard_for(tool_names, toolbox):
    """The guard for a strategy: the grep shape when `grep` is among the tools, else search/fetch."""
    return _code_grep_guard(toolbox) if "grep" in tool_names else _code_fix_guard(toolbox)


code_fix_guard = _code_fix_guard
code_grep_guard = _code_grep_guard

__all__ = ["fix_guard_for", "code_fix_guard", "code_grep_guard"]
