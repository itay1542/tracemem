"""Unit tests for LanceDBVectorStore wrapper."""

import random
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from lancedb.rerankers import LinearCombinationReranker, RRFReranker

from tracemem_core.storage.protocols import VectorSearchResult
from tracemem_core.storage.vector.lance import LanceDBVectorStore


@pytest.fixture
def make_vector():
    """Create deterministic mock vectors for unit tests."""

    def _make(seed: int, dims: int = 1536) -> list[float]:
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(dims)]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    return _make


@pytest.fixture
async def vector_store(tmp_path: Path):
    """Provide a connected LanceDBVectorStore."""
    store = LanceDBVectorStore(path=tmp_path / "lancedb")
    await store.connect()
    yield store
    await store.close()


class TestLanceDBVectorStore:
    """Unit tests for LanceDBVectorStore wrapper behavior."""

    async def test_connect_and_close(self, tmp_path: Path):
        """Test that connect initializes and close cleans up."""
        store = LanceDBVectorStore(path=tmp_path / "lancedb")

        # Before connect, internal state is None
        assert store._db is None
        assert store._table is None

        await store.connect()

        # After connect, internal state is initialized
        assert store._db is not None
        assert store._table is not None

        await store.close()

        # After close, internal state is cleaned up
        assert store._db is None
        assert store._table is None

    async def test_operations_raise_when_not_connected(
        self, tmp_path: Path, make_vector
    ):
        """Test that operations raise RuntimeError when not connected."""
        store = LanceDBVectorStore(path=tmp_path / "lancedb")
        node_id = uuid4()
        vector = make_vector(42)

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.add(
                node_id=node_id,
                text="test",
                vector=vector,
                conversation_id="conv-1",
            )

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.search(
                query_vector=vector,
                query_text="test",
            )

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.update_last_accessed(node_id)

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.delete_by_conversation("conv-1")

    async def test_add_and_search_roundtrip(self, vector_store, make_vector):
        """Test adding a vector and finding it via search."""
        node_id = uuid4()
        vector = make_vector(42)
        text = "The quick brown fox jumps over the lazy dog"

        await vector_store.add(
            node_id=node_id,
            text=text,
            vector=vector,
            conversation_id="conv-1",
        )

        # Search with same vector should find the entry
        results = await vector_store.search(
            query_vector=vector,
            query_text="quick fox",
            limit=10,
        )

        assert len(results) >= 1
        assert any(r.node_id == node_id for r in results)

    async def test_search_respects_limit(self, vector_store, make_vector):
        """Test that search returns at most `limit` results."""
        # Add 5 entries
        for i in range(5):
            await vector_store.add(
                node_id=uuid4(),
                text=f"Document number {i} about testing",
                vector=make_vector(i),
                conversation_id="conv-1",
            )

        # Search with limit=2
        results = await vector_store.search(
            query_vector=make_vector(0),
            query_text="testing",
            limit=2,
        )

        assert len(results) <= 2

    async def test_search_excludes_conversation(self, vector_store, make_vector):
        """Test that exclude_conversation_id filters out results from that conversation."""
        # Add entries from two conversations with same vector
        shared_vector = make_vector(42)

        node_conv1 = uuid4()
        await vector_store.add(
            node_id=node_conv1,
            text="Message from conversation one",
            vector=shared_vector,
            conversation_id="conv-1",
        )

        node_conv2 = uuid4()
        await vector_store.add(
            node_id=node_conv2,
            text="Message from conversation two",
            vector=shared_vector,
            conversation_id="conv-2",
        )

        # Search excluding conv-1
        results = await vector_store.search(
            query_vector=shared_vector,
            query_text="conversation",
            limit=10,
            exclude_conversation_id="conv-1",
        )

        # Should only find conv-2
        assert all(r.conversation_id != "conv-1" for r in results)
        assert any(r.node_id == node_conv2 for r in results)

    async def test_delete_by_conversation(self, vector_store, make_vector):
        """Test deleting all entries for a conversation."""
        # Add entries from two conversations
        for i in range(3):
            await vector_store.add(
                node_id=uuid4(),
                text=f"Conv1 message {i}",
                vector=make_vector(i),
                conversation_id="conv-to-delete",
            )

        node_to_keep = uuid4()
        await vector_store.add(
            node_id=node_to_keep,
            text="Conv2 message",
            vector=make_vector(100),
            conversation_id="conv-to-keep",
        )

        # Delete conv-to-delete
        count = await vector_store.delete_by_conversation("conv-to-delete")

        assert count == 3

        # Search should only find conv-to-keep
        results = await vector_store.search(
            query_vector=make_vector(100),
            query_text="message",
            limit=10,
        )

        assert all(r.conversation_id != "conv-to-delete" for r in results)
        assert any(r.node_id == node_to_keep for r in results)

    async def test_delete_nonexistent_conversation_returns_zero(self, vector_store):
        """Test that deleting a nonexistent conversation returns 0."""
        count = await vector_store.delete_by_conversation("nonexistent-conv")
        assert count == 0

    async def test_search_returns_vector_search_result(self, vector_store, make_vector):
        """Test that search returns properly structured VectorSearchResult objects."""
        node_id = uuid4()
        vector = make_vector(42)
        text = "Test document for structure verification"

        await vector_store.add(
            node_id=node_id,
            text=text,
            vector=vector,
            conversation_id="conv-1",
        )

        results = await vector_store.search(
            query_vector=vector,
            query_text="test document",
            limit=1,
        )

        assert len(results) == 1
        result = results[0]

        # Verify it's a VectorSearchResult with correct fields
        assert isinstance(result, VectorSearchResult)
        assert result.node_id == node_id
        assert result.text == text
        assert result.conversation_id == "conv-1"
        assert isinstance(result.last_accessed, datetime)
        assert isinstance(result.score, float)

    async def test_update_last_accessed(self, vector_store, make_vector):
        """Test that update_last_accessed updates the timestamp."""
        import asyncio

        node_id = uuid4()
        vector = make_vector(42)

        await vector_store.add(
            node_id=node_id,
            text="Test document",
            vector=vector,
            conversation_id="conv-1",
        )

        # Get initial last_accessed
        results = await vector_store.search(
            query_vector=vector,
            query_text="test",
            limit=1,
        )
        initial_last_accessed = results[0].last_accessed

        # Wait a bit and update
        await asyncio.sleep(0.01)
        await vector_store.update_last_accessed(node_id)

        # Verify last_accessed was updated
        results = await vector_store.search(
            query_vector=vector,
            query_text="test",
            limit=1,
        )
        updated_last_accessed = results[0].last_accessed

        assert updated_last_accessed > initial_last_accessed

    async def test_custom_reranker_instance(self, tmp_path: Path, make_vector):
        """Test that a custom reranker instance is used during search."""
        custom_reranker = LinearCombinationReranker(weight=0.5)
        store = LanceDBVectorStore(
            path=tmp_path / "lancedb",
            reranker=custom_reranker,
        )
        await store.connect()

        node_id = uuid4()
        vector = make_vector(42)
        await store.add(
            node_id=node_id, text="test doc", vector=vector, conversation_id="c1"
        )

        results = await store.search(
            query_vector=vector, query_text="test", vector_weight=0.3
        )

        assert store._reranker is custom_reranker
        assert len(results) >= 1
        await store.close()

    async def test_default_reranker_is_rrf(self, tmp_path: Path):
        """Default reranker resolves to RRFReranker via registry."""
        store = LanceDBVectorStore(path=tmp_path / "lancedb")
        assert isinstance(store._reranker, RRFReranker)

    async def test_string_reranker_resolution(self, tmp_path: Path):
        """String reranker key is resolved via get_reranker registry."""
        store = LanceDBVectorStore(path=tmp_path / "lancedb", reranker="linear")
        assert isinstance(store._reranker, LinearCombinationReranker)

    async def test_unknown_reranker_string_raises(self, tmp_path: Path):
        """Unknown reranker string raises ValueError at init."""
        with pytest.raises(ValueError, match="Unknown reranker"):
            LanceDBVectorStore(path=tmp_path / "lancedb", reranker="unknown")
