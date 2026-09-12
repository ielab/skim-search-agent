"""The Sieve paper's research prompt, kept for reproduction: the deep-research agent that answers
in as few turns as possible with the short answer span. The paper-era conditions
(`agent_search/strategies/paper.py`) bind to it; the friendly strategy names run `research`, the
library's default prompt."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchPaper(Task):
    name = "research_paper"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "The Sieve paper's prompt: answer a question over a fixed corpus with the search tools, short answer span in <answer> tags."
