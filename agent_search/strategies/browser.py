"""OpenResearcher's tools: gpt-oss's text browser over the run's retriever.

`browser.search` lists results as links, `browser.open` follows a link or moves the window on a
page, `browser.find` lists the lines of a page that hold a pattern. The three share one page
stack per episode (`agent_search/tools/browser_pages.py`). OpenResearcher was trained on
trajectories over these tools, so it runs with them and with its own prompt (task
`research_openresearcher`). Two variants, one per ranker behind the search.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.browser_find.tool import BrowserFind
from agent_search.tools.browser_open.tool import BrowserOpen
from agent_search.tools.browser_search.tool import BrowserSearch

browser_dense = register_strategy(Strategy(
    name="browser_dense", toolset_name="browser_dense", sdk=False,
    description="OpenResearcher's browser tools over the run's dense model: search, open, find",
    tools=(BrowserSearch(ranking="dense"), BrowserOpen(), BrowserFind())))

browser_bm25 = register_strategy(Strategy(
    name="browser_bm25", toolset_name="browser_bm25", sdk=False,
    description="OpenResearcher's browser tools over BM25: search, open, find",
    tools=(BrowserSearch(ranking="bm25"), BrowserOpen(), BrowserFind())))
