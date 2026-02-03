"""Integration tests for local vs global mode URI canonicalization.

Tests end-to-end URI storage behavior with different extractor modes,
verifying that files are stored with the correct URI format and that
cross-conversation lookups work correctly.

No Docker required — uses embedded Kuzu and LanceDB.

Run with:
    uv run pytest tests/integration/test_extractor_modes.py -v
"""

import pytest

from tracemem_core.config import TraceMemConfig
from tracemem_core.extractors import DefaultResourceExtractor, _canonicalize_file_uri
from tracemem_core.messages import Message, ToolCall
from tracemem_core.retrieval import HybridRetrievalStrategy
from tracemem_core.tracemem import TraceMem

from ..conftest import MockEmbedder


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


@pytest.fixture
async def tracemem_global(tmp_path, mock_embedder):
    """TraceMem in global mode (absolute URIs)."""
    config = TraceMemConfig(home=tmp_path / "global_home" / ".tracemem")
    extractor = DefaultResourceExtractor(mode="global")
    tm = TraceMem(config=config, embedder=mock_embedder, resource_extractor=extractor)
    async with tm:
        yield tm


@pytest.fixture
async def tracemem_local(tmp_path, mock_embedder):
    """TraceMem in local mode (relative URIs under project root).

    Project root is derived as home.parent = tmp_path / "local_home".
    """
    home = tmp_path / "local_home" / ".tracemem"
    config = TraceMemConfig(home=home)
    extractor = DefaultResourceExtractor(mode="local", home=home)
    tm = TraceMem(config=config, embedder=mock_embedder, resource_extractor=extractor)
    async with tm:
        yield tm


def _import_read_trace(file_path: str, conv_id: str, content: str = "# content"):
    """Helper to build a simple read_file trace."""
    return [
        Message(role="user", content=f"Read {file_path}"),
        Message(
            role="assistant",
            content="Reading...",
            tool_calls=[
                ToolCall(id=f"call_{conv_id}", name="read_file", args={"path": file_path})
            ],
        ),
        Message(role="tool", content=content, tool_call_id=f"call_{conv_id}"),
    ]


class TestGlobalModeURIs:
    """Tests for global mode — all file URIs stored as absolute paths."""

    async def test_stores_absolute_uri(self, tracemem_global: TraceMem, tmp_path):
        """Global mode stores absolute file:// URIs."""
        test_file = tmp_path / "local_home" / "src" / "app.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# app")

        messages = _import_read_trace(str(test_file), "conv-1")
        await tracemem_global.import_trace("conv-1", messages)

        expected_uri = f"file://{test_file.resolve()}"
        resource = await tracemem_global._graph_store.get_resource_by_uri(expected_uri)
        assert resource is not None
        assert resource.uri == expected_uri

    async def test_cross_conversation_lookup_absolute_uri(
        self, tracemem_global: TraceMem, tmp_path
    ):
        """Cross-conversation lookup works with absolute URIs in global mode."""
        test_file = tmp_path / "local_home" / "src" / "shared.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# shared")

        retrieval = HybridRetrievalStrategy(
            graph_store=tracemem_global._graph_store,
            vector_store=tracemem_global._vector_store,
            embedder=tracemem_global._embedder,
        )

        for i in range(2):
            messages = _import_read_trace(str(test_file), f"conv-{i}", "# shared")
            await tracemem_global.import_trace(f"conv-{i}", messages)

        expected_uri = f"file://{test_file.resolve()}"
        conversations = await retrieval.get_conversations_for_resource(expected_uri)
        assert len(conversations) == 2

    async def test_files_outside_project_use_absolute_uri(
        self, tracemem_global: TraceMem, tmp_path
    ):
        """Files outside any project root use absolute URIs in global mode."""
        external_file = tmp_path / "external" / "lib.py"
        external_file.parent.mkdir(parents=True, exist_ok=True)
        external_file.write_text("# external")

        messages = _import_read_trace(str(external_file), "conv-1")
        await tracemem_global.import_trace("conv-1", messages)

        expected_uri = f"file://{external_file.resolve()}"
        resource = await tracemem_global._graph_store.get_resource_by_uri(expected_uri)
        assert resource is not None


class TestLocalModeURIs:
    """Tests for local mode — file URIs stored relative to project root."""

    async def test_stores_relative_uri(self, tracemem_local: TraceMem, tmp_path):
        """Local mode stores relative file:// URIs for files under project root."""
        # Project root is tmp_path / "local_home"
        test_file = tmp_path / "local_home" / "src" / "app.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# app")

        messages = _import_read_trace(str(test_file), "conv-1")
        await tracemem_local.import_trace("conv-1", messages)

        expected_uri = "file://src/app.py"
        resource = await tracemem_local._graph_store.get_resource_by_uri(expected_uri)
        assert resource is not None
        assert resource.uri == expected_uri

    async def test_cross_conversation_lookup_relative_uri(
        self, tracemem_local: TraceMem, tmp_path
    ):
        """Cross-conversation lookup works with relative URIs in local mode."""
        test_file = tmp_path / "local_home" / "src" / "shared.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# shared")

        retrieval = HybridRetrievalStrategy(
            graph_store=tracemem_local._graph_store,
            vector_store=tracemem_local._vector_store,
            embedder=tracemem_local._embedder,
        )

        for i in range(2):
            messages = _import_read_trace(str(test_file), f"conv-{i}", "# shared")
            await tracemem_local.import_trace(f"conv-{i}", messages)

        expected_uri = "file://src/shared.py"
        conversations = await retrieval.get_conversations_for_resource(expected_uri)
        assert len(conversations) == 2

    async def test_file_outside_root_falls_back_to_absolute(
        self, tracemem_local: TraceMem, tmp_path
    ):
        """Files outside project root fall back to absolute URIs in local mode."""
        # Project root is tmp_path / "local_home", file is in tmp_path / "external"
        external_file = tmp_path / "external" / "lib.py"
        external_file.parent.mkdir(parents=True, exist_ok=True)
        external_file.write_text("# external")

        messages = _import_read_trace(str(external_file), "conv-1")
        await tracemem_local.import_trace("conv-1", messages)

        # Should fall back to absolute URI since external is not under project root
        expected_uri = f"file://{external_file.resolve()}"
        resource = await tracemem_local._graph_store.get_resource_by_uri(expected_uri)
        assert resource is not None

    async def test_portability_same_relative_path_matches(self, tmp_path, mock_embedder):
        """Same relative path matches across different absolute homes.

        Simulates moving a project to a different directory — relative URIs
        should still match because they're relative to project root.
        """
        # First "location" of the project
        home1 = tmp_path / "location1" / ".tracemem"
        config1 = TraceMemConfig(home=home1)
        extractor1 = DefaultResourceExtractor(mode="local", home=home1)

        project_file1 = tmp_path / "location1" / "src" / "main.py"
        project_file1.parent.mkdir(parents=True, exist_ok=True)
        project_file1.write_text("# main")

        async with TraceMem(config=config1, embedder=mock_embedder, resource_extractor=extractor1) as tm1:
            messages = _import_read_trace(str(project_file1), "conv-1")
            await tm1.import_trace("conv-1", messages)

            # Verify relative URI is stored
            resource = await tm1._graph_store.get_resource_by_uri("file://src/main.py")
            assert resource is not None

        # The key insight: a second project at a different absolute path
        # would produce the same relative URI "file://src/main.py"
        # This is the portability benefit of local mode
        uri1 = _canonicalize_file_uri(
            f"file://{project_file1}",
            root=(tmp_path / "location1").resolve(),
        )

        project_file2 = tmp_path / "location2" / "src" / "main.py"
        project_file2.parent.mkdir(parents=True, exist_ok=True)
        uri2 = _canonicalize_file_uri(
            f"file://{project_file2}",
            root=(tmp_path / "location2").resolve(),
        )

        assert uri1 == uri2 == "file://src/main.py"


class TestModeInteraction:
    """Tests for interaction between local and global modes."""

    async def test_global_and_local_produce_different_uris(self, tmp_path, mock_embedder):
        """Global and local modes produce different URIs for the same file.

        This means a Resource stored in global mode cannot be found by
        a local-mode query (and vice versa) — they are separate namespaces.
        """
        shared_path = tmp_path / "project" / "src" / "api.py"
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        shared_path.write_text("# api")

        # Global mode
        global_home = tmp_path / "global_store" / ".tracemem"
        global_config = TraceMemConfig(home=global_home)
        global_extractor = DefaultResourceExtractor(mode="global")

        async with TraceMem(config=global_config, embedder=mock_embedder, resource_extractor=global_extractor) as tm_global:
            messages = _import_read_trace(str(shared_path), "conv-g")
            await tm_global.import_trace("conv-g", messages)

            global_uri = f"file://{shared_path.resolve()}"
            resource_global = await tm_global._graph_store.get_resource_by_uri(global_uri)
            assert resource_global is not None

        # Local mode (home under project)
        local_home = tmp_path / "project" / ".tracemem"
        local_config = TraceMemConfig(home=local_home)
        local_extractor = DefaultResourceExtractor(mode="local", home=local_home)

        async with TraceMem(config=local_config, embedder=mock_embedder, resource_extractor=local_extractor) as tm_local:
            messages = _import_read_trace(str(shared_path), "conv-l")
            await tm_local.import_trace("conv-l", messages)

            local_uri = "file://src/api.py"
            resource_local = await tm_local._graph_store.get_resource_by_uri(local_uri)
            assert resource_local is not None

        # The URIs should be different
        assert global_uri != local_uri
