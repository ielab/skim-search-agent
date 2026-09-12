"""The library's default research prompt: a deep-research agent over a fixed corpus that verifies
each clue in retrieved text, follows strict tool rules generated from the declarations, paces its
budget and answers with the short span in `<answer>` tags. It combines DIVER's Tongyi prompt, DIVER's
strong prompt and the Sieve paper's prompt; on BrowseComp-Plus it lifted Sieve by 9 to 13 points at
equal token cost. The paper's own prompt is `research_paper`."""
from agent_search.tasks.base import Task, register_task


@register_task
class Research(Task):
    name = "research"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "Deep research over a local corpus: verify every clue in retrieved text, strict tool rules, pace the budget, answer with the short span in <answer> tags."
