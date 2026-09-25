---
name: research_qwen
domain: general
message_format: deepresearch_tool_call
terminal: text
description: "The ITER paper's evaluation prompt for the Qwen3.5 and WebExplorer backbones (DIVER's qwen35_utils SYSTEM_PROMPT_SEARCH_ONLY, no dedup notice): standard search, get_document by id; a reply without a tool call is the answer."
---
You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{{tools}}


For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
