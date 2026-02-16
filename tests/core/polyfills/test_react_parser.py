"""Tests for ReActParser."""

from uac.core.polyfills.react_parser import ReActParser


class TestReActParser:
    def setup_method(self) -> None:
        self.parser = ReActParser()

    def test_parse_action_with_json_input(self) -> None:
        text = (
            "Thought: I need to calculate 2+2\n"
            "Action: calculator\n"
            'Action Input: {"expression": "2+2"}'
        )
        result = self.parser.parse(text)
        assert result.thought == "I need to calculate 2+2"
        assert result.tool_call is not None
        assert result.tool_call.name == "calculator"
        assert result.tool_call.arguments == {"expression": "2+2"}
        assert result.final_answer is None

    def test_parse_final_answer(self) -> None:
        text = "Thought: I have enough information\nFinal Answer: The result is 42."
        result = self.parser.parse(text)
        assert result.thought == "I have enough information"
        assert result.final_answer == "The result is 42."
        assert result.tool_call is None

    def test_parse_action_with_invalid_json_fallback(self) -> None:
        text = "Thought: Let me try\nAction: search\nAction Input: not valid json"
        result = self.parser.parse(text)
        assert result.tool_call is not None
        assert result.tool_call.name == "search"
        assert result.tool_call.arguments == {"input": "not valid json"}

    def test_parse_no_pattern_graceful_degradation(self) -> None:
        text = "Just a plain response with no formatting."
        result = self.parser.parse(text)
        assert result.final_answer == "Just a plain response with no formatting."
        assert result.tool_call is None
        assert result.thought is None

    def test_parse_action_without_input(self) -> None:
        text = "Thought: I should act\nAction: do_something"
        result = self.parser.parse(text)
        assert result.tool_call is not None
        assert result.tool_call.name == "do_something"
        assert result.tool_call.arguments == {}

    def test_final_answer_takes_priority(self) -> None:
        text = "Thought: Done reasoning\nFinal Answer: Here is the answer."
        result = self.parser.parse(text)
        assert result.final_answer == "Here is the answer."
        assert result.tool_call is None

    def test_parse_thought_only(self) -> None:
        text = "Thought: I'm thinking about this problem"
        result = self.parser.parse(text)
        assert result.thought == "I'm thinking about this problem"
        # Falls through to graceful degradation
        assert result.final_answer is not None
