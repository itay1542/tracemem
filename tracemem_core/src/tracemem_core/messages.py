"""Internal message representation for TraceMem.

This module provides framework-agnostic message types. Use adapters to convert
from framework-specific formats (LangChain, OpenAI, etc.) to these internal types.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool invocation within a message."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Internal message representation.

    Framework-agnostic. Adapters convert external formats to this type.

    Attributes:
        role: The role of the message sender (user, assistant, tool, system).
        content: The text content of the message.
        tool_calls: List of tool invocations (for assistant messages).
        tool_call_id: ID of the tool call this message responds to (for tool messages).
    """

    role: Literal["user", "assistant", "tool", "system"]
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
