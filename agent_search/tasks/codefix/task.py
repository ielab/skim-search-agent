"""The code-fix task: investigate a reported bug in a repository and name the fix in a `<fix>`
block (scored on whether the right file and function were named)."""
from agent_search.tasks.base import Task, register_task


@register_task
class CodeFix(Task):
    name = "codefix"
    domain = "code"
    message_format = "deepresearch_tool_call"
    terminal = "fix"
    description = "Code fix: investigate a reported bug with search -> fetch, then propose the concrete fix in a <fix> block (scored fix-file-ok)."
