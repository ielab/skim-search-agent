"""`visit`: read one or more pages for a goal and return what a reader model extracts.

This is the visit tool of the WebAgent family (Tongyi DeepResearch, OpenSeeker's
`src/tools/visit.py`): the page text and the caller's goal go to a reader model with one fixed
prompt, the reader answers with JSON (`rational`, `evidence`, `summary`), and the tool returns

    The useful information in <url> for user goal <goal> as follows:

    Evidence in page:
    <evidence>

    Summary:
    <summary>

The agent never sees the raw page. Over a corpus the page is a document, addressed by the
link a search result printed (`https://corpus/<doc id>`), by its id, or by its rank in the last
listing. Several pages are read one after another and joined by a line of three dashes.

The reader is the endpoint of `agent_search/tools/page_reader.py`, at temperature 0.7 as in the
official tool. The page is cut at `VISIT_PAGE_TOKENS` model tokens (95,000 in the official tool).
"""
from __future__ import annotations

import json
import os

from agent_search.tokens import truncate_tokens
from agent_search.tools.base import Tool

PAGE_TOKENS = int(os.environ.get("VISIT_PAGE_TOKENS", "95000"))
CORPUS_URL = "https://corpus/"

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

# The official summariser call sets no limit. The prompt asks for the page's original text, so a
# reader without a limit copies whole pages until the call times out.
REPLY_TOKENS = 4096

_HEAD = "The useful information in {url} for user goal {goal} as follows: \n\n"
_FAILED = ("Evidence in page: \nThe provided webpage content could not be accessed. Please check the URL or "
           "file format.\n\nSummary: \nThe webpage content could not be processed, and therefore, no "
           "information is available.\n\n")


def read_with_goal(content: str, goal: str) -> dict:
    """The reader model's JSON for one page, or {} when the call or the JSON fails."""
    from agent_search.tools.page_reader import ask_reader
    prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
    for _ in range(2):
        try:
            text = ask_reader(prompt, temperature=0.7, max_tokens=REPLY_TOKENS)
            left, right = text.find("{"), text.rfind("}")
            data = json.loads(text[left:right + 1]) if 0 <= left <= right else None
            if isinstance(data, dict) and "evidence" in data and "summary" in data:
                return data
        except ValueError as e:
            if "VISIT_READER" in str(e):
                raise
        except Exception:  # noqa: BLE001 - a failed read is an observation, never a crash
            continue
    return {}


class VisitGoal(Tool):
    name = "visit"
    aliases = ("visit_summary",)
    description = "Parse webpage(s) and return the summary of the content according to the goal."
    parameters = {"type": "object",
                  "properties": {"url": {"type": ["string", "array"], "items": {"type": "string"}, "minItems": 1,
                                         "description": "The URL(s) of the webpage(s) to visit. Can be a "
                                                        "single URL or an array of URLs."},
                                 "goal": {"type": "string", "description": "The goal of the visit for webpage(s)."}},
                  "required": ["url", "goal"]}

    def _doc_id(self, url: str):
        ref = str(url).strip()
        if ref.startswith(CORPUS_URL):
            ref = ref[len(CORPUS_URL):]
        if ref in self.ubyid:
            return ref
        last = self.state.last_hits
        if ref.isdigit() and 1 <= int(ref) <= len(last):          # a rank in the last listing
            return last[int(ref) - 1]
        return None

    def _read(self, url: str, goal: str) -> str:
        head = _HEAD.format(url=url, goal=goal)
        doc_id = self._doc_id(url)
        if doc_id is None:
            return head + _FAILED
        u = self.ubyid[doc_id]
        self.state.seen.add(doc_id)
        if doc_id not in self.state.reads:
            self.state.reads.append(doc_id)
        title = u.title or u.qualname or ""
        page = truncate_tokens(f"{title}\n\n{u.body or u.code or ''}".strip(), PAGE_TOKENS, tail="")
        data = read_with_goal(page, goal)
        if not data:
            return head + _FAILED
        return head + f"Evidence in page: \n{data['evidence']}\n\nSummary: \n{data['summary']}\n\n"

    def run(self, args: dict) -> str:
        args = args or {}
        url, goal = args.get("url"), args.get("goal")
        if not url or goal is None:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"
        urls = [url] if isinstance(url, str) else [str(x) for x in url]
        return "\n---\n".join(self._read(x, str(goal)) for x in urls).strip()


__all__ = ["VisitGoal", "EXTRACTOR_PROMPT", "read_with_goal", "PAGE_TOKENS"]
