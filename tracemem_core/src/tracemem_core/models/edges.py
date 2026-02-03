from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EdgeBase(BaseModel):
    """Base properties shared by all edges in the graph."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Relationship(EdgeBase):
    """A procedure/tool invocation that connects nodes.

    The relationship type maps 1:1 with tool name (e.g., READ, EDIT, BASH, MESSAGE).
    """

    source_id: UUID
    target_id: UUID
    relationship_type: str = "MESSAGE"
    conversation_id: str
    properties: dict[str, Any] = Field(default_factory=dict)


class VersionOf(EdgeBase):
    """Connects a ResourceVersion to its parent Resource hypernode."""

    version_id: UUID
    resource_id: UUID
