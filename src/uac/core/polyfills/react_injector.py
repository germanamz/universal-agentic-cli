"""ReAct prompt injection for models without native tool calling.

Converts OpenAI-format tool definitions into a system prompt that
instructs the model to use the Thought / Action / Action Input /
Observation / Final Answer format.
"""

import json
from typing import Any

_REACT_PREAMBLE = """\
You have access to the following tools:

{tool_list}

To use a tool, respond with EXACTLY this format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON object with the tool arguments>

After the tool runs you will receive an Observation with the result.
You may repeat the Thought/Action/Action Input cycle as many times as needed.

When you have enough information to answer the user, respond with:

Thought: <your final reasoning>
Final Answer: <your response to the user>

IMPORTANT:
- Always start with a Thought.
- Use EXACTLY the tool names listed above.
- Action Input MUST be valid JSON.
- Do NOT wrap your answer in any other format.\
"""

_TOOL_TEMPLATE = """\
- {name}: {description}
  Parameters: {parameters}\
"""


class ReActInjector:
    """Builds a ReAct system prompt from OpenAI-format tool definitions."""

    def inject(self, tools: list[dict[str, Any]]) -> str:
        """Return a ReAct system prompt block describing *tools*."""
        tool_lines: list[str] = []
        for tool_def in tools:
            fn = tool_def.get("function", tool_def)
            name = fn.get("name", "unknown")
            description = fn.get("description", "No description provided.")
            parameters = json.dumps(fn.get("parameters", {}), indent=2)
            tool_lines.append(
                _TOOL_TEMPLATE.format(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )

        tool_list = "\n".join(tool_lines)
        return _REACT_PREAMBLE.format(tool_list=tool_list)
