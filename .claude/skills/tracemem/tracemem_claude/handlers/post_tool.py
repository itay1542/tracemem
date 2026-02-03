"""Handler for PostToolUse hook events."""

import json
from typing import Any

from tracemem_core import Message, ToolCall, TraceMem

from tracemem_claude.handlers.base import BaseHandler
from tracemem_claude.state.session import SessionState


class PostToolHandler(BaseHandler):
    """Handles PostToolUse events from Claude Code.

    Writes assistant tool calls and tool results to TraceMem.
    Creates AgentText nodes with tool_calls and corresponding tool Messages.
    """

    async def _process(self, tm: TraceMem, data: dict[str, Any]) -> None:
        """Process a tool use completion.

        Args:
            tm: The TraceMem instance.
            data: The hook event data containing:
                - session_id: The Claude Code session ID
                - tool_use_id: Unique ID for this tool invocation
                - tool_name: Name of the tool that was used
                - tool_input: The input arguments to the tool
                - tool_response: The result from the tool
        """
        session_id = data.get("session_id", "")
        tool_use_id = data.get("tool_use_id", "")
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})
        tool_response = data.get("tool_response")

        if not session_id or not tool_use_id:
            return

        # Add tool result FIRST so it's available when the assistant message
        # triggers _process_tool_call (which needs the content hash)
        content = self._extract_content(tool_response)
        tool_message = Message(
            role="tool",
            content=content,
            tool_call_id=tool_use_id,
        )
        await tm.add_message(session_id, tool_message)

        # Now create assistant message with tool call
        # _process_tool_call will find the tool result in _tool_results
        tool_call = ToolCall(
            id=tool_use_id,
            name=tool_name,
            args=tool_input if isinstance(tool_input, dict) else {},
        )
        assistant_message = Message(
            role="assistant",
            content="",
            tool_calls=[tool_call],
        )
        result = await tm.add_message(session_id, assistant_message)

        # Track agent ID for later content update
        if "agent_text" in result:
            state = SessionState(session_id)
            state.add_pending_agent_id(str(result["agent_text"]))

    def _extract_content(self, tool_response: Any) -> str:
        """Extract content string from tool response.

        Tool responses may be strings, dicts, or other types.

        Args:
            tool_response: The raw tool response.

        Returns:
            A string representation of the content.
        """
        if tool_response is None:
            return ""

        if isinstance(tool_response, str):
            return tool_response

        if isinstance(tool_response, dict):
            # Check for common content fields
            if "content" in tool_response:
                content = tool_response["content"]
                if isinstance(content, str):
                    return content
                return json.dumps(content)
            if "result" in tool_response:
                result = tool_response["result"]
                if isinstance(result, str):
                    return result
                return json.dumps(result)
            # Fall back to JSON representation
            return json.dumps(tool_response)

        if isinstance(tool_response, (list, tuple)):
            return json.dumps(tool_response)

        return str(tool_response)
