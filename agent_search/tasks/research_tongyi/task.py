"""The ITER paper's evaluation task on Tongyi-DeepResearch: Tongyi's deep-research system prompt
with DIVER's strict tool rules and no dedup notice, over the standard top-10 `search` and
`get_document` (strategies `iter_dense` / `iter_bm25`); answer in `<answer>` tags. `research_dedup`
is the de-duplicated variant DIVER collected trajectories with."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchTongyi(Task):
    name = "research_tongyi"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "The ITER paper's evaluation prompt (Tongyi): standard top-10 search, then get_document by id; answer in <answer> tags."
