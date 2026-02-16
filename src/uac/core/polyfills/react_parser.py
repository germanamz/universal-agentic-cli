"""ReAct output parser — extracts tool calls from free-form text.

Parses the Thought / Action / Action Input / Final Answer patterns
produced by models following the ReAct system prompt.
"""

import json
import re
from dataclasses import dataclass

from uac.core.interface.models import ToolCall

# Patterns accept optional whitespace and work across multi-line text.
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


@dataclass
class ReActParseResult:
    """Result of parsing a ReAct-formatted response."""

    thought: str | None = None
    tool_call: ToolCall | None = None
    final_answer: str | None = None


class ReActParser:
    """Extracts structured data from ReAct-formatted model output."""

    def parse(self, text: str) -> ReActParseResult:
        """Parse *text* for ReAct patterns.

        Returns a ``ReActParseResult`` with either a ``tool_call`` or a
        ``final_answer``.  If no recognisable pattern is found the raw
        text is returned as the final answer (graceful degradation).
        """
        thought = self._extract_thought(text)

        # Check for Final Answer first — it takes priority if both appear
        final = _FINAL_ANSWER_RE.search(text)
        if final:
            return ReActParseResult(
                thought=thought,
                final_answer=final.group(1).strip(),
            )

        # Check for Action / Action Input
        action_match = _ACTION_RE.search(text)
        if action_match:
            tool_name = action_match.group(1).strip()
            arguments = self._extract_arguments(text)
            return ReActParseResult(
                thought=thought,
                tool_call=ToolCall(name=tool_name, arguments=arguments),
            )

        # Graceful degradation — raw text as final answer
        return ReActParseResult(
            thought=thought,
            final_answer=text.strip(),
        )

    @staticmethod
    def _extract_thought(text: str) -> str | None:
        m = _THOUGHT_RE.search(text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_arguments(text: str) -> dict[str, object]:
        m = _ACTION_INPUT_RE.search(text)
        if not m:
            return {}
        raw = m.group(1).strip()
        try:
            result: dict[str, object] = json.loads(raw)
            return result
        except (json.JSONDecodeError, TypeError):
            return {"input": raw}
