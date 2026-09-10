"""Tasks: what the agent is asked to do and the protocol it answers in, one folder each
(`prompt.md` + `task.py`). Importing the package registers the built-in tasks."""
from agent_search.tasks.base import TASKS, Task, register_task
from agent_search.tasks.research.task import Research
from agent_search.tasks.research_dedup.task import ResearchDedup
from agent_search.tasks.codefix.task import CodeFix
from agent_search.tasks.codefix_patch.task import CodeFixPatch

__all__ = ["Task", "TASKS", "register_task", "Research", "ResearchDedup", "CodeFix", "CodeFixPatch"]
