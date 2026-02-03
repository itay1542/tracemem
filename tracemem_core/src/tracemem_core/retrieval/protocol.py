"""Protocol for retrieval strategies."""

from typing import Protocol
from uuid import UUID

from tracemem_core.retrieval.results import (
    ConversationReference,
    ContextResult,
    RetrievalConfig,
    RetrievalResult,
    TrajectoryResult,
)


class RetrievalStrategy(Protocol):
    """Protocol for retrieval strategies.

    Retrieval strategies combine vector search and graph traversal to
    retrieve relevant context from the knowledge graph.
    """

    async def get_context(self, node_id: UUID) -> ContextResult:
        """Get full context for a UserText node.

        Traverses the graph to find the agent response and any tool
        invocations with their associated resources.

        Args:
            node_id: UUID of the UserText node.

        Returns:
            ContextResult with user text, agent text, and tool uses.
        """
        ...

    async def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant past interactions.

        Performs hybrid search (vector + text) and optionally enriches
        results with graph context.

        Args:
            query: Search query text.
            config: Optional RetrievalConfig for fine-grained control.

        Returns:
            List of RetrievalResult ordered by relevance.
        """
        ...

    async def get_conversations_for_resource(
        self,
        uri: str,
        config: RetrievalConfig | None = None,
    ) -> list[ConversationReference]:
        """Find all conversations that accessed a resource.

        Args:
            uri: Canonical URI of the resource.
            config: Optional RetrievalConfig for sorting and filtering.

        Returns:
            List of ConversationReference with sorting applied.
        """
        ...

    async def get_trajectory(
        self,
        node_id: UUID,
        config: RetrievalConfig | None = None,
    ) -> TrajectoryResult:
        """Get the full trajectory from a UserText node to the next UserText.

        Args:
            node_id: UUID of the starting UserText node.
            config: Optional RetrievalConfig for trajectory settings.

        Returns:
            TrajectoryResult with all steps in chronological order.
        """
        ...
