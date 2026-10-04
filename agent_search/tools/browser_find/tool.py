"""`browser.find`: exact matches of a pattern on an opened page, as gpt-oss's browser does.

The page at `cursor` is searched line by line, case folded. Each match becomes a link,
`# 【id†match at L<line>】`, followed by the four lines from the match on; at most 50 matches.
The matches are a page of their own, and `browser.open` with a match's id shows the document
from just above that line.
"""
from __future__ import annotations

import re
from urllib.parse import quote

from agent_search.tools.base import Tool
from agent_search.tools.browser_pages import BrowserError, Page, as_int, stack_of, wrap_lines

MAX_MATCHES = 50
LINES_PER_MATCH = 4
_LINK = re.compile(r"【\d+†(?P<content>[^†】]+)(?:†[^†】]+)?】")


class BrowserFind(Tool):
    name = "browser.find"
    aliases = ("find",)
    description = "Finds exact matches of a pattern in the current page or a specified page by cursor."
    parameters = {"type": "object",
                  "properties": {
                      "pattern": {"type": "string", "description": "The exact text pattern to search for"},
                      "cursor": {"type": "integer", "default": -1,
                                 "description": "Page cursor to search in. If not provided, searches in "
                                                "the current page"}},
                  "required": ["pattern"]}

    def run(self, args: dict) -> str:
        args = args or {}
        pattern = str(args.get("pattern") or "").lower()
        cursor = as_int(args.get("cursor"), -1)
        stack = stack_of(self.state)
        try:
            page = stack.page(cursor)
            if page.is_listing:
                raise BrowserError("Cannot run `find` on search results page or find results page")
            source = stack.pages.index(page)
            lines = _LINK.sub(lambda m: m.group("content"), "\n".join(wrap_lines(page.text))).split("\n")
            chunks, links, link_lines, i = [], {}, {}, 0
            while i < len(lines) and len(chunks) < MAX_MATCHES:
                if not pattern or pattern not in lines[i].lower():
                    i += 1
                    continue
                key = str(len(chunks))
                links[key], link_lines[key] = page.url, (source, i)
                chunks.append(f"# 【{key}†match at L{i}】\n" + "\n".join(lines[i:i + LINES_PER_MATCH]))
                i += LINES_PER_MATCH
            text = "\n\n".join(chunks) if chunks else f"No `find` results for pattern: `{pattern}`"
            found = Page(url=f"{page.url}/find?pattern={quote(pattern)}",
                         title=f"Find results for text: `{pattern}` in `{page.title}`", text=text,
                         links=links, link_lines=link_lines, is_listing=True)
            shown, _ = stack.show(found, loc=0)
        except BrowserError as e:
            return str(e)
        return shown


__all__ = ["BrowserFind", "MAX_MATCHES"]
