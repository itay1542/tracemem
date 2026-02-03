"""Tests for message adapters."""

import pytest



class TestLangChainAdapter:
    """Test LangChainAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a LangChainAdapter instance."""
        # Import inside test to avoid import errors when langchain is not installed
        from tracemem_core.adapters.langchain import LangChainAdapter

        return LangChainAdapter()

    def test_convert_human_message(self, adapter) -> None:
        """Convert HumanMessage to user Message."""
        from langchain_core.messages import HumanMessage

        msg = HumanMessage(content="Hello")
        result = adapter.convert_single(msg)

        assert result.role == "user"
        assert result.content == "Hello"
        assert result.tool_calls == []
        assert result.tool_call_id is None

    def test_convert_system_message(self, adapter) -> None:
        """Convert SystemMessage to system Message."""
        from langchain_core.messages import SystemMessage

        msg = SystemMessage(content="You are helpful")
        result = adapter.convert_single(msg)

        assert result.role == "system"
        assert result.content == "You are helpful"

    def test_convert_ai_message_simple(self, adapter) -> None:
        """Convert simple AIMessage to assistant Message."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="Hello there!")
        result = adapter.convert_single(msg)

        assert result.role == "assistant"
        assert result.content == "Hello there!"
        assert result.tool_calls == []

    def test_convert_ai_message_with_tool_calls(self, adapter) -> None:
        """Convert AIMessage with tool_calls to assistant Message."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="I'll read the file",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "read_file",
                    "args": {"path": "/home/user/file.py"},
                }
            ],
        )
        result = adapter.convert_single(msg)

        assert result.role == "assistant"
        assert result.content == "I'll read the file"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[0].args == {"path": "/home/user/file.py"}

    def test_convert_tool_message(self, adapter) -> None:
        """Convert ToolMessage to tool Message."""
        from langchain_core.messages import ToolMessage

        msg = ToolMessage(content="file content here", tool_call_id="call_1")
        result = adapter.convert_single(msg)

        assert result.role == "tool"
        assert result.content == "file content here"
        assert result.tool_call_id == "call_1"

    def test_convert_list_content(self, adapter) -> None:
        """Handle list content (multimodal messages)."""
        from langchain_core.messages import HumanMessage

        msg = HumanMessage(
            content=[
                {"type": "text", "text": "First line"},
                {"type": "text", "text": "Second line"},
            ]
        )
        result = adapter.convert_single(msg)

        assert result.content == "First line\nSecond line"

    def test_convert_list_content_with_strings(self, adapter) -> None:
        """Handle list content with raw strings."""
        from langchain_core.messages import HumanMessage

        msg = HumanMessage(content=["Hello", "World"])
        result = adapter.convert_single(msg)

        assert result.content == "Hello\nWorld"

    def test_convert_batch(self, adapter) -> None:
        """Convert multiple messages at once."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]
        results = adapter.convert(messages)

        assert len(results) == 2
        assert results[0].role == "user"
        assert results[1].role == "assistant"

    def test_convert_unknown_message_type(self, adapter) -> None:
        """Unknown message types default to user role."""
        from langchain_core.messages import BaseMessage

        # Create a custom message type
        class CustomMessage(BaseMessage):
            type: str = "custom"

        msg = CustomMessage(content="Custom content")
        result = adapter.convert_single(msg)

        assert result.role == "user"
        assert result.content == "Custom content"

    def test_convert_ai_message_empty_tool_calls(self, adapter) -> None:
        """AIMessage with empty tool_calls list should have empty list."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="Hello", tool_calls=[])
        result = adapter.convert_single(msg)

        assert result.tool_calls == []

    def test_convert_tool_call_with_all_fields(self, adapter) -> None:
        """Handle tool_calls with all required fields."""
        from langchain_core.messages import AIMessage

        msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_123", "name": "test_tool", "args": {"key": "value"}}],
        )
        result = adapter.convert_single(msg)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "test_tool"
        assert result.tool_calls[0].args == {"key": "value"}
