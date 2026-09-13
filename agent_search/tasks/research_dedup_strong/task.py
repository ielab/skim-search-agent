"""ITER's strong prompt: DIVER's `--strong` system prompt for general backbones (a meticulous
multi-constraint research agent) with its dedup notice, and DIVER's QUERY_TEMPLATE as the user turn
for the native-tool clients. Tongyi-DeepResearch runs use `research_dedup` instead, the Qwen3.5 and
WebExplorer runs `research_dedup_qwen`."""
from agent_search.tasks.base import Task, register_task


@register_task
class ResearchDedupStrong(Task):
    name = "research_dedup_strong"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    description = "ITER's strong prompt for general backbones over the de-duplicated search and get_document tools; answer in <answer> tags."
    # DIVER's QUERY_TEMPLATE: the user turn its native-tool clients (gpt-oss) send with --strong
    user_template = (
        "You are a deep research agent. You need to answer the given question by interacting with a search engine, "
        "using the search and get_document tools provided. Please perform reasoning and use the tools step by step, "
        "in an interleaved manner. You may use the search and get_document tools multiple times.\n\n"
        "Question: {question}\n\n"
        "Your response should be in the following format:\n"
        "Explanation: {{your explanation for your final answer. For this explanation section only, you should cite "
        "your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. "
        "For example, [20].}}\n"
        "Exact Answer: {{your succinct, final answer}}\n"
        "Confidence: {{your confidence score between 0% and 100% for your answer}}")
