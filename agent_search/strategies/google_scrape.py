"""MiroThinker's tools: a search that returns JSON results and a read done by a second model.

`google_search` returns the organic results of a query as JSON. `scrape_and_extract_info` takes
a link and a request and returns what a reader model extracts from the page, never the page.
MiroThinker was trained in MiroFlow over these tools, so it runs with them and with its own
prompt (task `research_mirothinker`). Its official prompt also lists five sandbox tools; a call
to one is answered with an error, since no sandbox runs here. Two variants, one per ranker
behind the search.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.google_search.tool import GoogleSearch
from agent_search.tools.scrape_extract.tool import ScrapeExtract

_NO_SANDBOX = {name: "[ERROR]: E2B_API_KEY is not set."
               for name in ("create_sandbox", "run_command", "run_python_code",
                            "upload_file_from_local_to_sandbox", "download_file_from_internet_to_sandbox")}

google_scrape_dense = register_strategy(Strategy(
    name="google_scrape_dense", toolset_name="google_scrape_dense", sdk=False,
    description="MiroThinker's tools over the run's dense model: google_search, scrape_and_extract_info",
    tools=(GoogleSearch(ranking="dense"), ScrapeExtract(refusals=_NO_SANDBOX))))

google_scrape_bm25 = register_strategy(Strategy(
    name="google_scrape_bm25", toolset_name="google_scrape_bm25", sdk=False,
    description="MiroThinker's tools over BM25: google_search, scrape_and_extract_info",
    tools=(GoogleSearch(ranking="bm25"), ScrapeExtract(refusals=_NO_SANDBOX))))
