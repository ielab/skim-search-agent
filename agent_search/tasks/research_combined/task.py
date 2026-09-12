"""The combined research prompt: the deep-research persona and strict tool rules of DIVER's Tongyi
prompt, the clue-by-clue verification of DIVER's strong prompt, and the budget pacing and short
answer span of the paper's research prompt. Tool rules are generated from the declarations, so the
task fits any toolset."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchCombined(Task):
    name = "research_combined"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "Deep research over a local corpus: verify every clue in retrieved text, strict tool rules, pace the budget, answer with the short span in <answer> tags."
