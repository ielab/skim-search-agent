"""`scrape_and_extract_info`: MiroThinker's read tool, a page read by a model for one request.

MiroFlow's scrape server (`jina_scrape_llm_summary`) fetches a page and has a second model
extract what the agent asked for (`info_to_extract`, usually a question). The agent gets a
JSON string with the extraction and a few statistics of the page, never the page. Over a
corpus the page is a document, addressed by the link a search result printed
(`https://corpus/<doc id>`) or by its id. The extraction prompt, its temperature (1.0) and its
8,192-token reply budget are the official tool's; the reader is the endpoint of
`agent_search/tools/page_reader.py`.
"""
from __future__ import annotations

import json

from agent_search.tools.base import Tool

CORPUS_URL = "https://corpus/"
REPLY_TOKENS = 8192

EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def extract(content: str, info_to_extract: str) -> str:
    """The reader model's extraction for one page."""
    from agent_search.tools.page_reader import ask_reader
    return ask_reader(EXTRACT_INFO_PROMPT.format(info_to_extract, content), temperature=1.0, max_tokens=REPLY_TOKENS)


class ScrapeExtract(Tool):
    name = "scrape_and_extract_info"
    aliases = ("scrape", "visit")
    description = ("Scrape content from a URL and extract meaningful information using an LLM.")
    parameters = {"type": "object",
                  "properties": {"url": {"title": "Url", "type": "string"},
                                 "info_to_extract": {"title": "Info To Extract", "type": "string"}},
                  "required": ["url", "info_to_extract"]}

    def _failure(self, url: str, error: str, stats: dict) -> str:
        from agent_search.tools.page_reader import reader_model
        return json.dumps({"success": False, "url": url, "extracted_info": "", "error": error,
                           "scrape_stats": stats, "model_used": reader_model(), "tokens_used": 0},
                          ensure_ascii=False)

    def run(self, args: dict) -> str:
        from agent_search.tools.page_reader import reader_model
        args = args or {}
        url = str(args.get("url") or "").strip()
        # the official executor renames these two when the model writes them for the request
        wanted = args.get("info_to_extract") or args.get("description") or args.get("introduction") or ""
        ref = url[len(CORPUS_URL):] if url.startswith(CORPUS_URL) else url
        u = self.ubyid.get(ref)
        if u is None:
            return self._failure(url, "Scraping failed (both Jina and Python): "
                                      f"URL not found in the corpus: {url or 'URL cannot be empty'}", {})
        self.state.seen.add(ref)
        if ref not in self.state.reads:
            self.state.reads.append(ref)
        content = f"{u.title or u.qualname or ''}\n\n{u.body or u.code or ''}".strip()
        lines = content.count("\n") + 1
        stats = {"line_count": lines, "char_count": len(content), "last_char_line": lines,
                 "all_content_displayed": True}
        try:
            extracted = extract(content, str(wanted))
        except Exception as e:  # noqa: BLE001 - a failed read is an observation, never a crash
            return self._failure(url, f"Jina Scrape and Extract Info: Unexpected error during LLM API call: {e}", stats)
        return json.dumps({"success": True, "url": url, "extracted_info": extracted, "error": "",
                           "scrape_stats": stats, "model_used": reader_model(), "tokens_used": 0},
                          ensure_ascii=False)


__all__ = ["ScrapeExtract", "EXTRACT_INFO_PROMPT", "extract"]
