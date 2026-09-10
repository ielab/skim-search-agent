"""What the 0.3 restructure replaced, kept for one release so old imports, old run records and
the parity tests keep working.

`workspaces/` are the pre-0.3 tool workspaces (one class per condition; `agent_search.tools`
holds the atomic tools now). `retriever.py` is the pre-0.3 `AgentRetriever` and its arm switch
(`agent_search.evaluation.agent_runner.ConditionAgent` now). `prompts/` is the YAML prompt
registry with the task templates and manuals (`agent_search.tasks` and the tools' manual files
now; the files here are byte-identical copies that the prompt pins were taken from).

The old import paths (`agent_search.agent.tools`, `agent_search.agent.retriever`,
`agent_search.prompts`) resolve to these modules. Nothing new should import from here.
"""
