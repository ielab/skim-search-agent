"""The pre-0.3 code, kept under its own name so the parity tests can compare it against the
new code. `tests/test_tool_parity.py` and `tests/test_episode_parity.py` run both sides on
the same calls, and `scripts/replay_check.py` replays a recorded run through both.

`workspaces/` are the pre-0.3 tool workspaces (one class per condition; `agent_search.tools`
holds the atomic tools now). `retriever.py` is the pre-0.3 `AgentRetriever` and its arm switch
(`agent_search.evaluation.agent_runner.ConditionAgent` now). `prompts/` is the YAML prompt
registry with the task templates and manuals (`agent_search.tasks` and the tools' manual files
now; the files here are byte-identical copies that the prompt pins were taken from).

Nothing outside the parity tests should import from here. Nothing else in the codebase is
aliased to an old path: every other module lives at one place, and every importer uses that
place directly.
"""
