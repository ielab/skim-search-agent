"""OpenResearcher's own task (`OpenResearcher/OpenResearcher-30B-A3B`): the prompt and the
protocol of its inference code, for the `browser_*` strategies.

The model was trained on gpt-oss-120b trajectories over three browser tools. Its inference code
(github.com/TIGER-AI-Lab/OpenResearcher, `deploy_agent.py`) sends one system message, the
developer text plus the date, and lets the chat template list the tools. `prompt.md` is that
system message exactly as the template writes it for `browser.search`, `browser.open` and
`browser.find`, so the tools are named in the template's own words and not through `{{tools}}`.

The protocol: the question alone is the user turn; a call is the template's XML function block
(`model.tool_call_format: qwen_xml`); a reply without a call is the answer, whatever its format
(`<answer>` tags, "Exact Answer:", "Final Answer:"); the history goes out as tool messages, as
the official code sends it, so the template keeps the reasoning of earlier turns.
"""
from datetime import date
from typing import Optional, Sequence

from agent_search.tasks.base import Task, register_task


@register_task
class ResearchOpenResearcher(Task):
    name = "research_openresearcher"
    domain = "general"
    message_format = "tool_messages"
    terminal = "text"
    loop_user_template = "{question}"
    prompt_verbatim = True
    description = "OpenResearcher's own prompt and protocol: browser.search, browser.open, browser.find; a reply without a call is the answer."

    def render(self, tools: Sequence, profile: Optional[str] = None,
               toolset_name: Optional[str] = None) -> str:
        """The system message of the official code. The tool listing is part of the template
        text, written for the three browser tools; `tools` must be those three."""
        names = [t.name for t in tools]
        if names != ["browser.search", "browser.open", "browser.find"]:
            raise ValueError(f"research_openresearcher lists browser.search, browser.open and "
                             f"browser.find; the strategy has {names}")
        _, body = self.template()
        return body.replace("{{date}}", date.today().isoformat())
