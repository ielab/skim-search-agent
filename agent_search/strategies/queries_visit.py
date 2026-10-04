"""OpenSeeker's tools: a batched search and a visit that reads for a goal.

`search` takes an array of queries and lists the top results of each. `visit` takes a link and
a goal and returns what a reader model extracts from the page for that goal, never the raw
page. OpenSeeker was trained on trajectories over these two tools, so it runs with them and
with its own prompt (task `research_openseeker`). Its official prompt also lists five sandbox
tools; a call to one gets the official tool's answer for a missing sandbox key. Two variants,
one per ranker behind the search.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.search_queries.tool import SearchQueries
from agent_search.tools.visit_goal.tool import VisitGoal

_NO_SANDBOX = {name: "[ERROR]: E2B_API_KEY is not set."
               for name in ("create_sandbox", "run_command", "run_python_code",
                            "upload_file_from_local_to_sandbox", "download_file_from_internet_to_sandbox")}

queries_visit_dense = register_strategy(Strategy(
    name="queries_visit_dense", toolset_name="queries_visit_dense", sdk=False,
    description="OpenSeeker's tools over the run's dense model: batched search, visit with a goal",
    tools=(SearchQueries(ranking="dense"), VisitGoal(refusals=_NO_SANDBOX))))

queries_visit_bm25 = register_strategy(Strategy(
    name="queries_visit_bm25", toolset_name="queries_visit_bm25", sdk=False,
    description="OpenSeeker's tools over BM25: batched search, visit with a goal",
    tools=(SearchQueries(ranking="bm25"), VisitGoal(refusals=_NO_SANDBOX))))
