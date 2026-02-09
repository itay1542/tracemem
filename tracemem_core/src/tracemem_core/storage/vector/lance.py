from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import lancedb
import pyarrow as pa

from tracemem_core.storage.protocols import VectorSearchResult
from tracemem_core.storage.vector.rerankers import get_reranker


class LanceDBVectorStore:
    """LanceDB implementation of VectorStore with hybrid search."""

    TABLE_NAME = "user_texts"

    def __init__(
        self,
        path: Path,
        embedding_dimensions: int = 1536,
        reranker: str | Any = "rrf",
    ) -> None:
        self._path = path
        self._embedding_dimensions = embedding_dimensions
        self._reranker = get_reranker(reranker)
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None

    async def connect(self) -> None:
        """Connect to the LanceDB database."""
        self._path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._path))

        # Check if table exists
        table_names = [t for t in self._db.list_tables()]
        if self.TABLE_NAME in table_names:
            self._table = self._db.open_table(self.TABLE_NAME)
        else:
            # Create table with schema
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("node_id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field(
                        "vector", pa.list_(pa.float32(), self._embedding_dimensions)
                    ),
                    pa.field("conversation_id", pa.string()),
                    pa.field("created_at", pa.timestamp("us", tz="UTC")),
                    pa.field("last_accessed", pa.timestamp("us", tz="UTC")),
                ]
            )
            self._table = self._db.create_table(
                self.TABLE_NAME, schema=schema, exist_ok=True
            )

            # Create FTS index for hybrid search
            self._table.create_fts_index("text", replace=True)

    async def close(self) -> None:
        """Close the connection."""
        self._db = None
        self._table = None

    async def add(
        self,
        node_id: UUID,
        text: str,
        vector: list[float],
        conversation_id: str,
    ) -> None:
        """Add a vector entry."""
        if self._table is None:
            raise RuntimeError("Not connected")

        now = datetime.now(UTC)
        row = {
            "id": str(node_id),
            "node_id": str(node_id),
            "text": text,
            "vector": vector,
            "conversation_id": conversation_id,
            "created_at": now,
            "last_accessed": now,
        }
        self._table.add([row])

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
        if self._table is None:
            raise RuntimeError("Not connected")

        # Build the query with hybrid search (vector + FTS)
        query = (
            self._table.search(query_type="hybrid")
            .vector(query_vector)
            .text(query_text)
            .rerank(reranker=self._reranker)
            .limit(limit * 2)  # Get more results before filtering
        )

        # Execute search
        results = query.to_pandas()

        # Filter out excluded conversation
        if exclude_conversation_id:
            results = results[results["conversation_id"] != exclude_conversation_id]

        # Limit results
        results = results.head(limit)

        # Convert to VectorSearchResult
        search_results = []
        for _, row in results.iterrows():
            search_results.append(
                VectorSearchResult(
                    node_id=UUID(row["node_id"]),
                    text=row["text"],
                    conversation_id=row["conversation_id"],
                    created_at=row["created_at"].to_pydatetime(),
                    last_accessed=row["last_accessed"].to_pydatetime(),
                    score=float(row.get("_relevance_score", row.get("_distance", 0.0))),
                )
            )

        return search_results

    async def update_last_accessed(self, node_id: UUID) -> None:
        """Update last_accessed timestamp for a vector entry."""
        if self._table is None:
            raise RuntimeError("Not connected")

        self._table.update(
            where=f'node_id = "{node_id}"',
            values={"last_accessed": datetime.now(UTC)},
        )

    async def delete_by_conversation(self, conversation_id: str) -> int:
        """Delete all vectors for a conversation. Returns count deleted."""
        if self._table is None:
            raise RuntimeError("Not connected")

        # Get count before delete
        df = self._table.to_pandas()
        count = len(df[df["conversation_id"] == conversation_id])

        if count > 0:
            self._table.delete(f'conversation_id = "{conversation_id}"')

        return count
