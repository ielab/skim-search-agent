"""The page stack behind the `browser.*` tools: gpt-oss's simple browser over a fixed corpus.

OpenResearcher (and gpt-oss, its teacher) read through a small text browser: every result of
`browser.search`, `browser.open` and `browser.find` is a page pushed on a stack, shown as a
window of numbered lines with a cursor in front:

    [3] Title (url)
    **viewing lines [0 - 41] of 212**

    L0:
    L1: URL: https://corpus/5412
    L2: ...

`cursor` is the page's position on the stack. A search page lists its results as links,
`【0†Title†domain】`, and `browser.open` follows a link id or moves the window on a page.
The behaviour follows `gpt_oss/tools/simple_browser/simple_browser_tool.py`: lines wrapped at
80 columns, a window of `VIEW_TOKENS` model tokens when no line count is given, the same
header, the same error messages. A corpus document has no web address, so a page's URL is
`https://corpus/<doc id>`.
"""
from __future__ import annotations

import itertools
import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from agent_search.tokens import truncate_tokens

VIEW_TOKENS = 1024          # tokens shown per window when the call names no line count
WRAP_COLUMNS = 80
CORPUS_URL = "https://corpus/"
_LINK = re.compile(r"【(\d+)†")
# characters the browser reserves for its own link and citation marks
_RESERVED = {"【": "〖", "】": "〗", "◼": "◾", "​": ""}
_EMPTY_LINE = re.compile(r"^\s+$", flags=re.MULTILINE)
_EXTRA_NEWLINES = re.compile(r"\n(\s*\n)+")


class BrowserError(Exception):
    """A call the browser answers with a message instead of a page."""


@dataclass
class Page:
    url: str
    title: str
    text: str
    links: dict = field(default_factory=dict)       # link id -> url
    doc_ids: dict = field(default_factory=dict)     # link id -> corpus doc id (a search page)
    link_lines: dict = field(default_factory=dict)  # link id -> line to open at (a find page)
    is_listing: bool = False                        # a search or find page: `find` refuses it
    doc_id: Optional[str] = None                    # the corpus document this page shows


def clean_text(text: str) -> str:
    """Document text as the browser shows it: its reserved marks replaced, blank runs folded."""
    for old, new in _RESERVED.items():
        text = text.replace(old, new)
    text = _EMPTY_LINE.sub("", text)
    return _EXTRA_NEWLINES.sub("\n\n", text).strip()


def doc_url(doc_id: str) -> str:
    return f"{CORPUS_URL}{doc_id}"


def wrap_lines(text: str) -> list[str]:
    wrapped = itertools.chain.from_iterable(
        (textwrap.wrap(line, width=WRAP_COLUMNS, replace_whitespace=False, drop_whitespace=False)
         if line else [""])
        for line in text.split("\n"))
    return list(wrapped)


def numbered(lines: list[str], offset: int = 0) -> str:
    return "\n".join(f"L{i + offset}: {line}" for i, line in enumerate(lines))


class PageStack:
    """One episode's pages, in the order they were shown."""

    def __init__(self):
        self.pages: list[Page] = []

    @property
    def cursor(self) -> int:
        return len(self.pages) - 1

    def page(self, cursor: int = -1) -> Page:
        if not self.pages:
            raise BrowserError("No pages to access!")
        if cursor == -1 or cursor == self.cursor:
            return self.pages[-1]
        if not isinstance(cursor, int) or isinstance(cursor, bool):
            raise BrowserError(f"`cursor` should be an integer, not `{type(cursor).__name__}`")
        if not 0 <= cursor < len(self.pages):
            raise BrowserError(f"Cursor `{cursor}` is out of range. "
                               f"Available cursor indices: [0 - {self.cursor}].")
        return self.pages[cursor]

    def show(self, page: Page, loc: int = 0, num_lines: int = -1) -> tuple[str, list[str]]:
        """Push `page` and return its display from line `loc`, with the link ids in view."""
        lines = wrap_lines(page.text)
        total = len(lines)
        if loc >= total:
            raise BrowserError(f"Invalid location parameter: `{loc}`. "
                               f"Cannot exceed page maximum of {total - 1}.")
        if num_lines <= 0:
            # the lines that fit the token window, at least one
            window = truncate_tokens(numbered(lines[loc:], offset=loc), VIEW_TOKENS, tail="")
            num_lines = max(1, window.count("\n") + 1)
        end = min(loc + num_lines, total)
        self.pages.append(page)
        body = numbered(lines[loc:end], offset=loc)
        header = page.title + (f" ({page.url})" if page.url else "")
        text = f"[{self.cursor}] {header}\n**viewing lines [{loc} - {end - 1}] of {total - 1}**\n\n{body}"
        return text, _LINK.findall(body)


def stack_of(state) -> PageStack:
    """The episode's page stack, shared by the three browser tools."""
    return state.scratch.setdefault("browser_pages", PageStack())


def as_int(value, default: int = -1):
    """A call argument that should be a whole number, as the model may write it ("3", 3.0)."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else value


__all__ = ["Page", "PageStack", "BrowserError", "stack_of", "clean_text", "doc_url", "wrap_lines",
           "numbered", "as_int", "VIEW_TOKENS", "CORPUS_URL"]
