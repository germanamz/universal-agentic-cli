"""Tests for ReActInjector."""

from uac.core.polyfills.react_injector import ReActInjector


class TestReActInjector:
    def setup_method(self) -> None:
        self.injector = ReActInjector()

    def test_inject_single_tool(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                    },
                },
            }
        ]
        prompt = self.injector.inject(tools)
        assert "calculator" in prompt
        assert "Evaluate a math expression" in prompt
        assert "Thought:" in prompt
        assert "Action:" in prompt
        assert "Action Input:" in prompt
        assert "Final Answer:" in prompt

    def test_inject_multiple_tools(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Do math",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        prompt = self.injector.inject(tools)
        assert "search" in prompt
        assert "calculator" in prompt

    def test_inject_tool_without_wrapper(self) -> None:
        tools = [
            {
                "name": "simple_tool",
                "description": "A simple tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        prompt = self.injector.inject(tools)
        assert "simple_tool" in prompt

    def test_inject_missing_description(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mystery",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        prompt = self.injector.inject(tools)
        assert "mystery" in prompt
        assert "No description provided." in prompt
