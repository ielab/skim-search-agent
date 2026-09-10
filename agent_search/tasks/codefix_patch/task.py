"""The code-fix task in patch mode: the `<fix>` block is a SEARCH/REPLACE edit compiled to a
unified diff and graded by the SWE-bench harness."""
from agent_search.tasks.base import Task, register_task


@register_task
class CodeFixPatch(Task):
    name = "codefix_patch"
    domain = "code"
    message_format = "deepresearch_tool_call"
    terminal = "patch"
    description = ("Code fix (PATCH mode): investigate a bug with search -> fetch, then emit an APPLYABLE edit as a <fix> "
                   "SEARCH/REPLACE block. Compiled to a unified diff and graded by the real SWE-bench harness "
                   "(FAIL_TO_PASS / PASS_TO_PASS).")
