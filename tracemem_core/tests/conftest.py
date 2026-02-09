from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tracemem_core.config import TraceMemConfig
from tracemem_core.retrieval.results import ContextResult
from tracemem_core.tracemem import TraceMem


class MockEmbedder:
    """Mock embedder for testing."""

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Return a deterministic mock embedding based on text hash."""
        hash_val = hash(text)
        return [(hash_val % (i + 1)) / 1000.0 for i in range(self._dimensions)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return [await self.embed(text) for text in texts]


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Provide a mock embedder for tests."""
    return MockEmbedder()


@pytest.fixture
def temp_lancedb_path(tmp_path: Path) -> Path:
    """Provide a temporary path for LanceDB."""
    return tmp_path / "lancedb"


@pytest.fixture
def sample_conversation_id() -> str:
    """Provide a sample conversation ID."""
    return f"conv-{uuid4()}"


@pytest.fixture
def mock_graph_store(mocker) -> AsyncMock:
    """Provide a mock graph store."""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock.initialize_schema = AsyncMock()
    # Polymorphic node/edge creation
    mock.create_node = AsyncMock(side_effect=lambda x: x)
    mock.create_edge = AsyncMock(side_effect=lambda x: x)
    # Resource operations
    mock.get_resource_by_uri = AsyncMock(return_value=None)
    mock.update_resource_hash = AsyncMock()
    mock.get_resource_version_by_hash = AsyncMock(return_value=None)
    # Basic retrieval operations
    mock.get_user_text = AsyncMock(return_value=None)
    mock.get_last_user_text = AsyncMock(return_value=None)
    mock.get_last_agent_text = AsyncMock(return_value=None)
    mock.get_last_message_node = AsyncMock(return_value=None)
    mock.update_last_accessed = AsyncMock()
    # Turn-based operations
    mock.get_max_turn_index = AsyncMock(return_value=-1)
    mock.get_last_node_in_turn = AsyncMock(return_value=None)
    # Retrieval query operations
    mock.get_node_context = AsyncMock(return_value=ContextResult())
    mock.get_resource_conversations = AsyncMock(return_value=[])
    mock.get_trajectory_nodes = AsyncMock(return_value=[])
    # Raw Cypher execution
    mock.execute_cypher = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def mock_vector_store(mocker) -> AsyncMock:
    """Provide a mock vector store."""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.close = AsyncMock()
    mock.add = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.update_last_accessed = AsyncMock()
    mock.delete_by_conversation = AsyncMock(return_value=0)
    return mock


@pytest.fixture
async def tracemem_kuzu(tmp_path: Path, mock_embedder: MockEmbedder):
    """TraceMem instance with embedded Kuzu and LanceDB for integration tests.

    No Docker required — uses embedded Kuzu and LanceDB in tmp_path.
    """
    config = TraceMemConfig(home=tmp_path / ".tracemem")
    tm = TraceMem(config=config, embedder=mock_embedder)
    async with tm:
        yield tm


@pytest.fixture
def retrieval_strategy_kuzu(tracemem_kuzu: TraceMem):
    """Provide a HybridRetrievalStrategy using Kuzu-backed TraceMem's stores."""
    from tracemem_core.retrieval import HybridRetrievalStrategy

    return HybridRetrievalStrategy(
        graph_store=tracemem_kuzu._graph_store,
        vector_store=tracemem_kuzu._vector_store,
        embedder=tracemem_kuzu._embedder,
    )


@pytest.fixture
def neo4j_config():
    """Neo4j connection config for tests."""
    return {
        "uri": "bolt://localhost:17687",
        "user": "neo4j",
        "password": "testpassword",
        "database": "neo4j",
    }


@pytest.fixture
async def tracemem_integration(
    tmp_path: Path, mock_embedder: MockEmbedder, neo4j_config
):
    """TraceMem instance with real Neo4j and LanceDB for integration tests.

    Requires Neo4j to be running:
        docker compose up -d neo4j

    Run with:
        uv run pytest tests/integration/test_tracemem_integration.py -v
    """
    from tracemem_core.storage.graph.neo import Neo4jGraphStore

    config = TraceMemConfig(
        graph_store="neo4j",
        neo4j_uri=neo4j_config["uri"],
        neo4j_user=neo4j_config["user"],
        neo4j_password=neo4j_config["password"],
        neo4j_database=neo4j_config["database"],
        lancedb_path=tmp_path / "lancedb",
    )
    tm = TraceMem(config=config, embedder=mock_embedder)
    async with tm:
        yield tm

    # Cleanup: delete all nodes using a fresh connection
    cleanup_store = Neo4jGraphStore(**neo4j_config)
    await cleanup_store.connect()
    async with cleanup_store._driver.session(
        database=neo4j_config["database"]
    ) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await cleanup_store.close()


@pytest.fixture
def retrieval_strategy(tracemem_integration: TraceMem):
    """Provide a HybridRetrievalStrategy using TraceMem's stores.

    This fixture depends on tracemem_integration and provides a strategy
    that uses the same graph store, vector store, and embedder.
    """
    from tracemem_core.retrieval import HybridRetrievalStrategy

    return HybridRetrievalStrategy(
        graph_store=tracemem_integration._graph_store,
        vector_store=tracemem_integration._vector_store,
        embedder=tracemem_integration._embedder,
    )
