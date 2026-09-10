"""Code strategies: how the agent looks through a repository.

`codefix` is Boolean search over code units (grouped by file) then a fetch of one function or
line range. `codefix_grep` is a regex grep over the repository files then a line-range read,
with no manual. Both are for the `code` domain and pair with the `codefix` tasks.
"""
from agent_search.strategies.base import Strategy, register_strategy
from agent_search.tools.fetch_code.tool import FetchCode, SearchCode
from agent_search.tools.grep.tool import Grep
from agent_search.tools.read.tool import Read

codefix = register_strategy(Strategy(
    name="codefix", toolset_name="code_fix", domain="code",
    description="Boolean search over code units, then fetch a function or a line range",
    tools=(SearchCode(name="search"), FetchCode(name="fetch"))))

codefix_grep = register_strategy(Strategy(
    name="codefix_grep", toolset_name="code_grep", domain="code",
    description="regex grep over the repository files, then read a line range",
    tools=(Grep(name="grep"), Read(name="read", source="repo"))))
