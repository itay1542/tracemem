"""Construction tests for local/neo4j mode store selection.

No Docker required — these tests verify store instantiation without connecting.

Run with:
    uv run pytest tests/integration/test_local_mode.py -v
"""

from tracemem_core.config import TraceMemConfig
from tracemem_core.storage.graph.kuzu_store import KuzuGraphStore
from tracemem_core.tracemem import TraceMem

from ..conftest import MockEmbedder


class TestLocalModeConstruction:
    """Tests for store construction and path setup."""

    async def test_creates_kuzu_store(self, tmp_path):
        """Verify default config uses KuzuGraphStore."""
        config = TraceMemConfig(home=tmp_path / ".tracemem")
        tm = TraceMem(config=config, embedder=MockEmbedder())
        assert isinstance(tm._graph_store, KuzuGraphStore)

    async def test_local_mode_storage_paths_created(self, tmp_path):
        """Verify that storage directories are created."""
        home = tmp_path / ".tracemem"
        config = TraceMemConfig(home=home)
        async with TraceMem(config=config, embedder=MockEmbedder()):
            pass

        assert (home / "graph").exists()
        assert (home / "vectors").exists()

    async def test_neo4j_mode_creates_neo4j_store(self, tmp_path):
        """Verify that graph_store='neo4j' creates Neo4jGraphStore."""
        config = TraceMemConfig(
            home=tmp_path / ".tracemem",
            graph_store="neo4j",
        )
        tm = TraceMem(config=config, embedder=MockEmbedder())

        from tracemem_core.storage.graph.neo import Neo4jGraphStore

        assert isinstance(tm._graph_store, Neo4jGraphStore)
