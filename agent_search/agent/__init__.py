"""The agent machinery a harness is built from: the model providers (backbone/), the policies
that propose each step (policies.py), the ReAct step loop (loop.py), the parsing of tool calls
and answers (actions.py), the forced final answer (forced_answer.py), the Agents-SDK driver
(sdk_driver.py) and the run record (record.py). What to run is a condition
(agent_search.strategies); how the model is put to work is a harness (agent_search.harness)."""
