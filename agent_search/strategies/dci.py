"""DCI (Direct Corpus Interaction): a shell over the exported corpus, no retriever.

`dci` gives the agent `bash` and `read` over a directory holding every document as a file.
`bounded_dci` adds a BM25 `bm25_search`; the directory then starts with the question's top
hits and grows by one batch per search call, so the shell only ever sees what was retrieved.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.bash.tool import Bash
from agent_search.tools.read.tool import Read
from agent_search.tools.search_bm25_dci.tool import SearchBm25Dci

dci = register_strategy(Strategy(
    name="dci", toolset_name="dci",
    description="a shell (bash, read) over the whole exported corpus, no retriever",
    tools=(Bash(name="bash"), Read(name="read"))))

bounded_dci = register_strategy(Strategy(
    name="bounded_dci", toolset_name="bm25_dci",
    description="BM25 search, then a shell (bash, read) over the retrieved documents only",
    tools=(SearchBm25Dci(name="bm25_search"), Bash(name="bash", bounded=True), Read(name="read"))))
