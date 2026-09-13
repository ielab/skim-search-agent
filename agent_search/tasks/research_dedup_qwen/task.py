"""ITER's research task as DIVER evaluated it on the Qwen3.5 and WebExplorer backbones: the
Qwen-Agent system prompt ("You are a helpful assistant." plus the tools block and the tool-call
format), then the dedup notice. No answer tags: the first reply without a tool call is the
answer, as in DIVER's `qwen35_utils/react_agent.py` (terminal `text`)."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchDedupQwen(Task):
    name = "research_dedup_qwen"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "text"
    description = "ITER's evaluation prompt for the Qwen3.5 and WebExplorer backbones: search with de-duplicated results, then get_document by id; a reply without a tool call is the answer."
