"""ITER's research task: the same goal as `research`, with ITER's strong system prompt for its
de-duplicated search and `get_document` tools; answer in `<answer>` tags."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchDedup(Task):
    name = "research_dedup"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "ITER's search strategy: keyword search with de-duplicated results, then get_document by id; answer in <answer> tags."
