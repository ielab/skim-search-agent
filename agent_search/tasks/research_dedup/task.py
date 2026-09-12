"""ITER's research task as DIVER evaluated it on Tongyi-DeepResearch: Tongyi's deep-research system
prompt, the strict tool rules, and the dedup notice, over the de-duplicated search and
`get_document` tools; answer in `<answer>` tags. `research_dedup_strong` is DIVER's `--strong`
prompt for general backbones."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchDedup(Task):
    name = "research_dedup"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "ITER's evaluation prompt (Tongyi): search with de-duplicated results, then get_document by id; answer in <answer> tags."
