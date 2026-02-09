"""Hybrid retrieval strategy combining vector search and graph traversal."""

import json
import logging
from datetime import datetime
from uuid import UUID

from tracemem_core.embedders.protocol import Embedder
from tracemem_core.retrieval.results import (
    ConversationReference,
    ContextResult,
    RetrievalConfig,
    RetrievalResult,
    ToolUse,
    TrajectoryResult,
    TrajectoryStep,
)
from tracemem_core.storage.protocols import GraphStore, VectorStore

logger = logging.getLogger(__name__)


class HybridRetrievalStrategy:
    """Hybrid retrieval strategy combining vector search and graph traversal.

    This strategy:
    1. Uses vector search to find relevant UserText nodes
    2. Delegates graph queries to the GraphStore
    3. Combines results with relevance scoring

    Example:
        ```python
        strategy = HybridRetrievalStrategy(graph_store, vector_store, embedder)

        # Search for relevant past interactions
        results = await strategy.search("authentication bug", config=RetrievalConfig(limit=5))

        # Get full context for a specific node
        context = await strategy.get_context(node_id)
        ```
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedder: Embedder,
    ) -> None:
        """Initialize the hybrid retrieval strategy.

        Args:
            graph_store: Graph storage for traversal queries.
            vector_store: Vector storage for similarity search.
            embedder: Embedder for converting queries to vectors.
        """
        self._graph_store = graph_store
        self._vector_store = vector_store
        self._embedder = embedder

    async def get_context(self, node_id: UUID) -> ContextResult:
        """Get full context for a UserText node.

        Traverses the graph to find:
        - The user message
        - The connected agent response
        - Any tool invocations and their resources

        Args:
            node_id: UUID of the UserText node.

        Returns:
            ContextResult with full conversation context.
        """
        return await self._graph_store.get_node_context(node_id)

    async def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant past interactions.

        Performs hybrid search combining vector similarity and full-text search.
        Optionally enriches results with graph context.

        Args:
            query: Search query text.
            config: Optional RetrievalConfig for fine-grained control.

        Returns:
            List of RetrievalResult ordered by relevance.
        """
        cfg = config or RetrievalConfig()

        logger.debug(
            "search query=%r limit=%d include_context=%s",
            query,
            cfg.limit,
            cfg.include_context,
        )

        # Get query embedding
        query_vector = await self._embedder.embed(query)

        # Fetch more when deduplicating to ensure enough unique conversations
        fetch_limit = cfg.limit * 3 if cfg.unique_conversations else cfg.limit

        # Perform vector search with vector_weight
        vector_results = await self._vector_store.search(
            query_vector=query_vector,
            query_text=query,
            limit=fetch_limit,
            exclude_conversation_id=cfg.exclude_conversation_id,
            vector_weight=cfg.vector_weight,
        )

        # Deduplicate by conversation if requested (keep best score per conv)
        if cfg.unique_conversations:
            seen: dict[str, int] = {}
            deduped: list = []
            for vr in vector_results:
                if vr.conversation_id not in seen:
                    seen[vr.conversation_id] = len(deduped)
                    deduped.append(vr)
                elif vr.score > deduped[seen[vr.conversation_id]].score:
                    deduped[seen[vr.conversation_id]] = vr
            vector_results = deduped[: cfg.limit]

        # Convert to retrieval results
        results: list[RetrievalResult] = []
        for vr in vector_results:
            result = RetrievalResult(
                node_id=vr.node_id,
                text=vr.text,
                conversation_id=vr.conversation_id,
                score=vr.score,
                created_at=vr.created_at,
            )

            if cfg.include_context:
                result.context = await self.get_context(vr.node_id)

            results.append(result)

        # Update last accessed timestamps
        if results:
            node_ids = [r.node_id for r in results]
            await self._graph_store.update_last_accessed(node_ids)
            # Also update in vector store
            for node_id in node_ids:
                await self._vector_store.update_last_accessed(node_id)

        logger.debug("search query=%r results=%d", query, len(results))
        return results

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
        cfg = config or RetrievalConfig()
        return await self._graph_store.get_resource_conversations(
            uri,
            limit=cfg.limit,
            sort_by=cfg.sort_by,
            sort_order=cfg.sort_order,
            exclude_conversation_id=cfg.exclude_conversation_id,
        )

    async def get_trajectory(
        self,
        node_id: UUID,
        config: RetrievalConfig | None = None,
    ) -> TrajectoryResult:
        """Get the full trajectory from a UserText node to the next UserText.

        Traverses the conversation graph starting from the given UserText node,
        collecting all AgentText nodes until the next UserText (exclusive) or
        end of conversation.

        Args:
            node_id: UUID of the starting UserText node.
            config: Optional RetrievalConfig for trajectory settings.

        Returns:
            TrajectoryResult with all steps in chronological order.
        """
        cfg = config or RetrievalConfig()
        records = await self._graph_store.get_trajectory_nodes(
            node_id,
            max_depth=cfg.trajectory_max_depth,
        )
        return self._parse_trajectory(node_id, records)

    def _parse_trajectory(
        self,
        node_id: UUID,
        records: list[dict],
    ) -> TrajectoryResult:
        """Parse raw trajectory records into a TrajectoryResult.

        Finds the start node, collects steps until the next UserText,
        and deserializes tool_uses JSON from AgentText nodes.
        """
        result = TrajectoryResult()

        if not records:
            return result

        found_start = False
        for rec in records:
            node = rec["n"]
            labels = rec["node_labels"]

            if "UserText" in labels:
                node_type = "UserText"
            elif "AgentText" in labels:
                node_type = "AgentText"
            else:
                continue

            # Skip until we see the start node, then stop at the next UserText
            if node_type == "UserText" and node["id"] == str(node_id):
                found_start = True
            elif node_type == "UserText" and found_start:
                # This is the follow-up UserText — include it and stop
                result.steps.append(
                    TrajectoryStep(
                        node_id=node["id"],
                        node_type="UserText",
                        text=node.get("text", ""),
                        conversation_id=node.get("conversation_id", ""),
                        created_at=self._parse_created_at(node),
                    )
                )
                break

            if not found_start:
                continue

            # Parse tool_uses from AgentText nodes
            tool_uses: list[ToolUse] = []
            if node_type == "AgentText" and node.get("tool_uses"):
                raw_tool_uses = node["tool_uses"]
                if isinstance(raw_tool_uses, str):
                    raw_tool_uses = json.loads(raw_tool_uses)
                for tu in raw_tool_uses:
                    tool_uses.append(
                        ToolUse(
                            tool_name=tu.get("name", ""),
                            properties=tu.get("args", {}),
                        )
                    )

            step = TrajectoryStep(
                node_id=node["id"],
                node_type=node_type,
                text=node.get("text", ""),
                conversation_id=node.get("conversation_id", ""),
                created_at=self._parse_created_at(node),
                tool_uses=tool_uses,
            )
            result.steps.append(step)

        logger.debug("get_trajectory node_id=%s steps=%d", node_id, len(result.steps))
        return result

    @staticmethod
    def _parse_created_at(node: dict) -> datetime | None:
        """Parse created_at ISO string from a node dict."""
        raw = node.get("created_at")
        if raw and isinstance(raw, str):
            return datetime.fromisoformat(raw)
        return None
