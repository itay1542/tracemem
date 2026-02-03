from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel

from tracemem_core.models.edges import EdgeBase
from tracemem_core.models.nodes import AgentText, NodeBase, Resource, ResourceVersion, UserText
from tracemem_core.retrieval.results import (
    ContextResult,
    ConversationReference,
)


class VectorSearchResult(BaseModel):
    """Result from vector search."""

    node_id: UUID
    text: str
    conversation_id: str
    created_at: datetime
    last_accessed: datetime
    score: float


class GraphStore(Protocol):
    """Protocol for graph storage operations.

    This protocol defines low-level CRUD operations on the graph database.
    Complex retrieval logic should be implemented in retrieval strategies.
    """

    async def connect(self) -> None:
        """Connect to the graph database."""
        ...

    async def close(self) -> None:
        """Close the connection."""
        ...

    async def initialize_schema(self) -> None:
        """Create constraints and indexes."""
        ...

    # Polymorphic node/edge operations
    async def create_node(self, node: NodeBase) -> NodeBase:
        """Create a node. Dispatches to correct handler based on type."""
        ...

    async def create_edge(self, edge: EdgeBase) -> EdgeBase:
        """Create an edge. Dispatches to correct handler based on type."""
        ...

    # Resource-specific operations
    async def get_resource_by_uri(self, uri: str) -> Resource | None:
        """Get a Resource by its URI."""
        ...

    async def update_resource_hash(self, uri: str, content_hash: str) -> None:
        """Update the current content hash of a Resource."""
        ...

    # Basic retrieval operations
    async def get_user_text(self, node_id: UUID) -> UserText | None:
        """Get a UserText node by ID."""
        ...

    async def get_last_user_text(self, conversation_id: str) -> UserText | None:
        """Get the most recent UserText node in a conversation by timestamp."""
        ...

    async def get_last_agent_text(self, conversation_id: str) -> AgentText | None:
        """Get the most recent AgentText node in a conversation by timestamp."""
        ...

    async def get_last_message_node(
        self, conversation_id: str
    ) -> UserText | AgentText | None:
        """Get the most recent message node (UserText or AgentText) by timestamp.

        This is used to maintain conversation continuity when there are multiple
        assistant messages in a row (e.g., with tool usage).
        """
        ...

    async def get_resource_version_by_hash(
        self, uri: str, content_hash: str
    ) -> "ResourceVersion | None":
        """Get a ResourceVersion by its URI and content hash."""
        ...

    async def get_max_turn_index(self, conversation_id: str) -> int:
        """Get the maximum turn index in a conversation.

        Returns -1 if no turns exist (first user message will be turn 0).
        """
        ...

    async def get_last_node_in_turn(
        self, conversation_id: str, turn_index: int
    ) -> UserText | AgentText | None:
        """Get the most recent node in a specific turn by timestamp."""
        ...

    async def update_last_accessed(self, node_ids: list[UUID]) -> None:
        """Update last_accessed_at for the given nodes."""
        ...

    # Retrieval query operations
    async def get_node_context(self, node_id: UUID) -> ContextResult:
        """Get full context for a UserText node (user text, agent response, tool uses)."""
        ...

    async def get_resource_conversations(
        self,
        uri: str,
        *,
        limit: int = 10,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        exclude_conversation_id: str | None = None,
    ) -> list[ConversationReference]:
        """Find all conversations that accessed a resource by URI."""
        ...

    async def get_trajectory_nodes(
        self,
        node_id: UUID,
        *,
        max_depth: int = 100,
    ) -> list[dict[str, Any]]:
        """Get raw nodes reachable from a UserText via MESSAGE edges.

        Returns list of dicts with 'n' (node props) and 'node_labels' (list[str]).
        """
        ...

    # Raw query execution
    async def execute_cypher(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a raw Cypher query and return results."""
        ...


class VectorStore(Protocol):
    """Protocol for vector storage operations."""

    async def connect(self) -> None:
        """Connect to the vector store."""
        ...

    async def close(self) -> None:
        """Close the connection."""
        ...

    async def add(
        self,
        node_id: UUID,
        text: str,
        vector: list[float],
        conversation_id: str,
    ) -> None:
        """Add a vector entry."""
        ...

    async def update_last_accessed(self, node_id: UUID) -> None:
        """Update last_accessed timestamp for a vector entry."""
        ...

    async def search(
        self,
        query_vector: list[float],
        query_text: str,
        limit: int = 10,
        exclude_conversation_id: str | None = None,
        vector_weight: float = 0.7,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using hybrid search.

        Args:
            query_vector: Query embedding vector.
            query_text: Query text for full-text search.
            limit: Maximum number of results.
            exclude_conversation_id: Optional conversation to exclude.
            vector_weight: Weight for vector vs text search (0.0-1.0).
                0.0 = pure text search, 1.0 = pure vector search.

        Returns:
            List of VectorSearchResult ordered by relevance.
        """
        ...

    async def delete_by_conversation(self, conversation_id: str) -> int:
        """Delete all vectors for a conversation. Returns count deleted."""
        ...
