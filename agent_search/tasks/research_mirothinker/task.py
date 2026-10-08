"""MiroThinker's own task (`miromind-ai/MiroThinker-1.7-mini`): the prompt and the protocol of
MiroFlow's BrowseComp runs, for the `google_scrape_*` strategies.

MiroFlow (github.com/MiroMindAI/MiroThinker, config `mirothinker_1.7_keep5_max300`) writes the
tool servers and their schemas into the system prompt as text and asks for calls as
`<use_mcp_tool>` blocks. `prompt.md` is that system prompt as the framework renders it: the
search server, the scrape server and the five sandbox tools. The sandbox tools stay in the
listing, as in the official prompt, and answer that no sandbox is available (the strategy's
refusals).

The protocol: the user turn is the question plus the official box instruction; a call is an
MCP block (`model.tool_call_format: mcp`); a tool result is a user message with the result
text alone, and only the newest five results stay in the prompt (`plain_results`); a reply
without a call ends the episode and is the answer, which carries the model's `\\boxed{}`.
MiroFlow then asks once more for a boxed summary and restarts an episode that gives no box;
neither step is reproduced.
"""
import os
from datetime import date
from typing import Optional, Sequence

from agent_search.tasks.base import Task, register_task

# src/io/input_handler.py: the sentence appended to the question
BOX_INSTRUCTION = ("\nYou should follow the format instruction in the request strictly "
                   "and wrap the final answer in \\boxed{}.")


@register_task
class ResearchMiroThinker(Task):
    name = "research_mirothinker"
    domain = "general"
    message_format = "plain_results"
    terminal = "text"
    loop_user_template = "{question}" + BOX_INSTRUCTION
    prompt_verbatim = True
    description = "MiroThinker's own prompt and protocol: google_search, scrape_and_extract_info, MCP calls; a reply without a call is the answer."

    def render(self, tools: Sequence, profile: Optional[str] = None,
               toolset_name: Optional[str] = None) -> str:
        """The system prompt MiroFlow renders. The tool schemas are part of the copied text;
        `tools` must be the two it can run."""
        names = [t.name for t in tools]
        if names != ["google_search", "scrape_and_extract_info"]:
            raise ValueError("research_mirothinker lists google_search and scrape_and_extract_info; "
                             f"the strategy has {names}")
        _, body = self.template()
        # MIROTHINKER_DATE pins the date so a cell resumed on a later day keeps the same prompt.
        return body.replace("{{date}}", os.environ.get("MIROTHINKER_DATE") or date.today().isoformat())
