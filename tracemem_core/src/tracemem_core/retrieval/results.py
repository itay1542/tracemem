"""Result models for retrieval operations."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Configuration for retrieval operations.

    This model provides fine-grained control over search and context retrieval.

    Attributes:
        limit: Maximum number of results to return (1-100).
        include_context: Whether to expand results with full context.
        vector_weight: Weight for vector similarity vs text search (0.0-1.0).
            0.0 = pure text search, 1.0 = pure vector search.
        expand_tool_uses: Include tool invocations in context expansion.
        expand_resources: Include resource info in context expansion.
        sort_by: Field to sort resource query results by.
        sort_order: Sort direction for resource queries.
        exclude_conversation_id: Exclude a specific conversation from results.
    """

    limit: int = Field(default=10, ge=1, le=100)
    include_context: bool = True

    # Hybrid search tuning (0.0 = pure text, 1.0 = pure vector)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    # Expansion options
    expand_tool_uses: bool = True
    expand_resources: bool = True

    # Sorting (for resource queries)
    sort_by: Literal["created_at", "last_accessed_at"] = "created_at"
    sort_order: Literal["asc", "desc"] = "desc"

    # Filtering
    exclude_conversation_id: str | None = None
    unique_conversations: bool = False

    # Trajectory
    trajectory_max_depth: int = Field(default=100, ge=1, le=500)


class ResourceInfo(BaseModel):
    """Information about a resource."""

    id: str
    uri: str


class ResourceVersionInfo(BaseModel):
    """Information about a resource version."""

    id: str
    uri: str
    content_hash: str


class ToolUse(BaseModel):
    """A tool invocation with its associated resource."""

    tool_name: str
    properties: dict[str, str | int | float | bool | None] = {}
    resource_version: ResourceVersionInfo | None = None
    resource: ResourceInfo | None = None

    def __str__(self) -> str:
        uri = self.resource_version.uri if self.resource_version else "n/a"
        rv_id = self.resource_version.id[:8] if self.resource_version else None
        res_id = self.resource.id[:8] if self.resource else None
        ids = f" rv={rv_id} res={res_id}" if rv_id or res_id else ""
        return f"{self.tool_name}({uri}{ids})"


class UserTextInfo(BaseModel):
    """Information about a user message."""

    id: str
    text: str
    conversation_id: str


class AgentTextInfo(BaseModel):
    """Information about an agent response."""

    id: str
    text: str


class ContextResult(BaseModel):
    """Full context for a user query.

    Contains the user message, agent response, and any tool invocations
    with their associated resources.
    """

    user_text: UserTextInfo | None = None
    agent_text: AgentTextInfo | None = None
    tool_uses: list[ToolUse] = []

    def __str__(self) -> str:
        user_id = self.user_text.id[:8] if self.user_text else None
        agent_id = self.agent_text.id[:8] if self.agent_text else None
        user = self.user_text.text[:80] if self.user_text else "n/a"
        agent = self.agent_text.text[:80] if self.agent_text else "n/a"
        tools = ", ".join(str(t) for t in self.tool_uses) if self.tool_uses else "none"
        return f"Context(user[{user_id}]={user!r}, agent[{agent_id}]={agent!r}, tools=[{tools}])"


class RetrievalResult(BaseModel):
    """A single retrieval result from search.

    Combines vector similarity score with graph context.
    """

    node_id: UUID
    text: str
    conversation_id: str
    score: float
    created_at: datetime | None = None
    context: ContextResult | None = None

    def __str__(self) -> str:
        nid = str(self.node_id)[:8]
        text_preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        ctx = "yes" if self.context else "no"
        ts = self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "n/a"
        return f"Result({nid}, score={self.score:.3f}, conv={self.conversation_id}, ts={ts}, text={text_preview!r}, context={ctx})"


class ConversationReference(BaseModel):
    """Reference to a conversation that accessed a resource."""

    conversation_id: str
    user_text_id: str
    user_text: str
    agent_text: str | None = None
    created_at: datetime | None = None

    def __str__(self) -> str:
        uid = self.user_text_id[:8]
        text_preview = (
            self.user_text[:60] + "..." if len(self.user_text) > 60 else self.user_text
        )
        ts = self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "n/a"
        return f"ConvRef({uid}, conv={self.conversation_id}, ts={ts}, user={text_preview!r})"


class TrajectoryStep(BaseModel):
    """A single step in a conversation trajectory."""

    node_id: str
    node_type: Literal["UserText", "AgentText"]
    text: str
    conversation_id: str
    created_at: datetime | None = None
    tool_uses: list[ToolUse] = []

    def __str__(self) -> str:
        nid = self.node_id[:8]
        ts = self.created_at.strftime("%Y-%m-%d %H:%M") if self.created_at else "n/a"
        text_preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        tools = (
            f" tools=[{', '.join(t.tool_name for t in self.tool_uses)}]"
            if self.tool_uses
            else ""
        )
        return f"Step({nid} {ts} {self.node_type}: {text_preview!r}{tools})"


class TrajectoryResult(BaseModel):
    """Full trajectory from one UserText to the next.

    Contains all steps (UserText and AgentText nodes) in order,
    representing one complete user turn including agent responses.
    """

    steps: list[TrajectoryStep] = []

    def __str__(self) -> str:
        if not self.steps:
            return "Trajectory(empty)"
        step_lines = "\n  ".join(str(s) for s in self.steps)
        return f"Trajectory({len(self.steps)} steps):\n  {step_lines}"
