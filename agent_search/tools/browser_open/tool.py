"""`browser.open`: follow a link or move the window, as gpt-oss's browser does.

`id` is a link id on the page at `cursor` (a search result, a find match) or a page address;
without one the call moves the window on the page at `cursor`. `loc` is the first line to
show and `num_lines` how many; unset, the window starts at the top (or just above a find
match) and holds `VIEW_TOKENS` model tokens. Every call pushes a page, so the next cursor is
the one in the display's header.
"""
from __future__ import annotations

from agent_search.tools.base import Tool
from agent_search.tools.browser_pages import (CORPUS_URL, BrowserError, Page, as_int, clean_text,
                                              doc_url, stack_of)


class BrowserOpen(Tool):
    name = "browser.open"
    aliases = ("open",)
    description = ("Opens a link from the current page or a fully qualified URL. Can scroll to a specific "
                   "location and display a specific number of lines. Valid link ids are displayed with "
                   "the formatting: 【{id}†.*】.")
    parameters = {"type": "object",
                  "properties": {
                      "id": {"type": ["integer", "string"], "default": -1,
                             "description": "Link id from current page (integer) or fully qualified URL "
                                            "(string). Default is -1 (most recent page)"},
                      "cursor": {"type": "integer", "default": -1,
                                 "description": "Page cursor to operate on. If not provided, the most "
                                                "recent page is implied"},
                      "loc": {"type": "integer", "default": -1,
                              "description": "Starting line number. If not provided, viewport will be "
                                             "positioned at the beginning or centered on relevant passage"},
                      "num_lines": {"type": "integer", "default": -1,
                                    "description": "Number of lines to display"},
                      "view_source": {"type": "boolean", "default": False,
                                      "description": "Whether to view page source"},
                      "source": {"type": "string", "description": "The source identifier (e.g., 'web')"}},
                  "required": []}

    def _document_page(self, url: str) -> Page:
        doc_id = url[len(CORPUS_URL):] if url.startswith(CORPUS_URL) else url
        u = self.ubyid.get(str(doc_id))
        if u is None:
            raise BrowserError(f"Error fetching URL `{url}`: URL not found in corpus: {url}")
        text = clean_text(u.body or u.code or "")
        return Page(url=doc_url(u.doc_id), title=u.title or u.qualname or f"Doc {u.doc_id}",
                    text=f"\nURL: {doc_url(u.doc_id)}\n{text}", doc_id=str(u.doc_id))

    def run(self, args: dict) -> str:
        args = args or {}
        link, cursor = as_int(args.get("id"), -1), as_int(args.get("cursor"), -1)
        loc, num_lines = as_int(args.get("loc"), -1), as_int(args.get("num_lines"), -1)
        stack, state = stack_of(self.state), self.state
        try:
            if not isinstance(loc, int) or not isinstance(num_lines, int):
                raise BrowserError("`loc` and `num_lines` should be integers")
            if isinstance(link, str):                       # a page address
                page = self._document_page(link.strip())
            else:
                current = stack.page(cursor)
                if link >= 0:                               # follow a link on that page
                    key = str(link)
                    if key not in current.links:
                        raise BrowserError(f"Invalid link id `{link}`.")
                    if key in current.link_lines:           # a find match: the same page at that line
                        page = stack.page(current.link_lines[key][0])
                        if loc < 0:                         # just above the match
                            line = current.link_lines[key][1]
                            loc = line - 4 if line > 4 else line
                    else:
                        page = self._document_page(current.links[key])
                else:                                       # move the window on that page
                    page = current
            text, in_view = stack.show(page, loc=max(loc, 0), num_lines=num_lines)
        except BrowserError as e:
            return str(e)
        if page.doc_id is not None:
            state.seen.add(page.doc_id)
            if page.doc_id not in state.reads:
                state.reads.append(page.doc_id)
        if page.doc_ids:                                    # a search page moved: more results in view
            state.seen.update(page.doc_ids[i] for i in in_view if i in page.doc_ids)
        return text


__all__ = ["BrowserOpen"]
