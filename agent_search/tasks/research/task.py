"""The research task: answer a hard question from a document collection, in `<answer>` tags."""
from agent_search.tasks.base import Task, register_task


@register_task
class Research(Task):
    name = "research"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "Deep-research multi-hop info-seeking over a fixed document corpus; answer with <answer>."
