"""The task contract: what the agent is asked to do and the protocol it answers in.

A task is a folder: `prompt.md` (the system prompt template, with front matter and the
`{{tools}}` / `{{tool_manuals}}` placeholders) and `task.py` (a `Task` subclass naming the
domain, the message format and the terminal). The paper's templates are copied here without a
byte changed.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Optional, Sequence

from agent_search.tasks.render import compose, split_front_matter

TASKS: dict[str, "Task"] = {}


def register_task(cls):
    """Register a Task subclass under its `name`."""
    inst = cls()
    TASKS[inst.name] = inst
    return cls


class Task:
    name: str = ""
    domain: str = "general"                       # general (documents) | code
    # how the history is sent: `deepresearch_tool_call` puts each reply back as written and its
    # observation in a user turn inside <tool_response> tags; `tool_messages` sends the reply as
    # an assistant message with its reasoning and its call as fields and the observation as a
    # tool message, so the model's own chat template writes the history; `plain_results` puts
    # the observation in a user turn without tags and keeps only the newest ones (agent/policies.py)
    message_format: str = "deepresearch_tool_call"
    terminal: str = "answer"                      # answer | fix | patch
    description: str = ""
    prompt_file: str = "prompt.md"                # next to the task's task.py
    # the user turn that carries the question, with `{question}`; None = the date line plus the
    # question (the loop driver's default). The Responses driver honours it; DIVER's strong prompt
    # pairs with its QUERY_TEMPLATE (Explanation / Exact Answer / Confidence lines).
    user_template: Optional[str] = None
    # the line the user template asks the final answer to carry (DIVER's `Exact Answer:`). The
    # Responses driver reminds the model when a turn has no call and no such line; None = the
    # first turn without a call is the answer.
    answer_line: Optional[str] = None
    # the same for the loop driver, which otherwise sends the date line plus the question. A
    # task that puts the date in its system prompt, as its model's own code does, sets this.
    loop_user_template: Optional[str] = None
    # the system prompt goes out byte for byte (a model's own prompt, copied from its code).
    # Otherwise the loop folds blank runs and ends the prompt with one newline, as it always has.
    prompt_verbatim: bool = False
    prompt_path: Optional[str] = None             # or an explicit path (a plugin's own template)

    def template_path(self) -> Path:
        if self.prompt_path:
            return Path(self.prompt_path)
        return Path(inspect.getfile(type(self))).resolve().parent / self.prompt_file

    def template(self) -> tuple[dict, str]:
        return split_front_matter(self.template_path().read_text(encoding="utf-8"))

    def render(self, tools: Sequence, profile: Optional[str] = None,
               toolset_name: Optional[str] = None) -> str:
        """The system prompt for `tools` (Tool instances). `profile` selects the manual
        variant (a dataset's field profile); it defaults to the task's domain."""
        _, body = self.template()
        dom = profile or self.domain
        return compose(body, [t.declaration() for t in tools],
                       [t.manual_path(dom) for t in tools], toolset_name)

    @staticmethod
    def sha256_of(system: str) -> str:
        return hashlib.sha256(system.encode("utf-8")).hexdigest()[:16]


__all__ = ["Task", "TASKS", "register_task"]
