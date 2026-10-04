"""OpenSeeker's own task (`PolarSeeker/OpenSeeker-v2-30B-SFT`): the prompt and the protocol of
its inference code, for the `queries_visit_*` strategies.

The official loop (github.com/PolarSeeker/OpenSeeker, `src/llm_tool_openseeker_v2.py`) sends a
one-line system message and lets the chat template list seven tools: `search`, `visit` and five
sandbox tools. `prompt.md` is that system message exactly as the template writes it. The
sandbox tools stay in the listing, as in the official prompt, and answer with the official
error for a missing sandbox key (the strategy's refusals).

The protocol: the user turn is the question plus the official box instruction; a call is JSON
inside `<tool_call>` tags; the episode ends on `<answer>` tags, as the official loop stops on
`</answer>`. The default history format gives the official prompt byte for byte through the
model's chat template, so nothing else is set.
"""
from typing import Optional, Sequence

from agent_search.tasks.base import Task, register_task

# eval/generate_answer_v2.py, BOX_FORMAT_INSTRUCTION: the model reads two backslashes
BOX_INSTRUCTION = ("\n\nYou should follow the format instruction in the request strictly "
                   "and wrap the final answer in \\\\boxed{}.")


@register_task
class ResearchOpenSeeker(Task):
    name = "research_openseeker"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    loop_user_template = "{question}" + BOX_INSTRUCTION
    prompt_verbatim = True
    description = "OpenSeeker's own prompt and protocol: batched search, visit with a goal; answer in <answer> tags."

    def render(self, tools: Sequence, profile: Optional[str] = None,
               toolset_name: Optional[str] = None) -> str:
        """The system message of the official code. The tool listing is part of the template
        text; `tools` must be the two it can run."""
        names = [t.name for t in tools]
        if names != ["search", "visit"]:
            raise ValueError(f"research_openseeker lists search and visit; the strategy has {names}")
        _, body = self.template()
        return body
