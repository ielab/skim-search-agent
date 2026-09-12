"""ITER's strong prompt: DIVER's `--strong` system prompt for general backbones (a meticulous
multi-constraint research agent). Tongyi-DeepResearch runs use `research_dedup` instead."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchDedupStrong(Task):
    name = "research_dedup_strong"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "ITER's strong prompt for general backbones over the de-duplicated search and get_document tools; answer in <answer> tags."
