"""QUEST's own task (`osunlp/QUEST-35B-RL`): the prompt and the protocol of its BrowseComp-Plus
run, for the `passages_*` strategies.

The official script (github.com/OSU-NLP-Group/QUEST, `inference/scripts/run_react_infer_bcp.sh`)
runs the agent with the visit tool off: the model searches and answers from the passages the
search returns. `prompt.md` is the system message that `inference/prompt.py` builds for that
run, copied as the code produces it. Removing the visit tool there leaves a stray fragment of
its declaration in the tool listing and puts the notice about it inside a sentence; both are
kept, because that is the prompt the released code sends.

The protocol: the question alone is the user turn; a call is JSON inside `<tool_call>` tags and
may carry several queries; the episode ends on `<answer>` tags. The official memory step, a
state summary written by another model once the history passes 80,000 tokens, is not
reproduced: the run keeps the newest turns that fit `agent.ctx_tokens`.
"""
from datetime import date
from typing import Optional, Sequence

from agent_search.tasks.base import Task, register_task

# inference/react_agent.py: the user message sent when the call limit is reached
CALL_LIMIT_MESSAGE = ("You have now reached the maximum number of reasoning turns you can use. "
                      "You must stop making tool calls and, based on all the information above, "
                      "think again and provide what you consider the most likely answer in the "
                      "following format:<think>your final thinking</think>\n<answer>your answer</answer>")


@register_task
class ResearchQuest(Task):
    name = "research_quest"
    domain = "general"
    message_format = "deepresearch_tool_call"
    terminal = "answer"
    loop_user_template = "{question}"
    prompt_verbatim = True
    description = "QUEST's own prompt and protocol on BrowseComp-Plus: search only, long passages; answer in <answer> tags."

    def render(self, tools: Sequence, profile: Optional[str] = None,
               toolset_name: Optional[str] = None) -> str:
        """The system message of the official code. The tool listing is part of the copied
        text; `tools` must be the one search tool."""
        names = [t.name for t in tools]
        if names != ["search"]:
            raise ValueError(f"research_quest lists search only; the strategy has {names}")
        _, body = self.template()
        return body.replace("{{date}}", date.today().isoformat())
