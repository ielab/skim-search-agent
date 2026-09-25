"""ITER's research task as DIVER evaluated it on the Qwen3.5 and WebExplorer backbones: the
Qwen-Agent system prompt ("You are a helpful assistant." plus the tools block and the tool-call
format), without the dedup notice: the no-dedup twin of research_dedup_qwen, as research_tongyi is of research_dedup. No answer tags: the first reply without a tool call is the
answer, as in DIVER's `qwen35_utils/react_agent.py` (terminal `text`)."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchQwen(Task):
    name = "research_qwen"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "text"
    description = "The ITER paper's evaluation prompt for the Qwen3.5 and WebExplorer backbones: standard search (no dedup), get_document by id; a reply without a tool call is the answer."
