from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class NodeBase(BaseModel):
    """Base properties shared by all nodes in the graph."""

    conversation_id: str
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UserText(NodeBase):
    """User message node - indexed in LanceDB for hybrid search."""

    text: str
    turn_index: int = 0  # Turn starts at user message


class ToolUseRecord(BaseModel):
    """Record of a tool invocation stored on AgentText node.

    This captures all tool calls from an assistant message, including those
    that don't produce resources (e.g., bash, web_search, grep).
    """

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentText(NodeBase):
    """Agent response node - reached via graph traversal from UserText."""

    text: str
    turn_index: int = 0  # Same turn as associated user message
    tool_uses: list[ToolUseRecord] = Field(default_factory=list)


class ResourceVersion(NodeBase):
    """Snapshot of a resource at a specific point in time."""

    content_hash: str
    uri: str


class Resource(NodeBase):
    """Hypernode representing resource identity across versions (identified by URI)."""

    uri: str
    current_content_hash: str | None = None
