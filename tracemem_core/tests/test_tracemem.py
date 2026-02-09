from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from lancedb.rerankers import LinearCombinationReranker, RRFReranker

from tracemem_core.config import TraceMemConfig
from tracemem_core.messages import Message, ToolCall
from tracemem_core.models.nodes import AgentText, ToolUseRecord, UserText
from tracemem_core.retrieval.results import (
    ContextResult,
    RetrievalConfig,
    TrajectoryResult,
)
from tracemem_core.tracemem import TraceMem

from .conftest import MockEmbedder


class TestTraceMemMessageAPI:
    """Test Message-based API."""

    @pytest.mark.asyncio
    async def test_add_message_user(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Add a user message."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        message = Message(role="user", content="Hello")
        result = await tm.add_message("conv-1", message)

        assert "user_text" in result
        mock_graph_store.create_node.assert_called_once()
        mock_vector_store.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_assistant(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Add an assistant message after a user message."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        # First add a user message
        user_result = await tm.add_message(
            "conv-1", Message(role="user", content="Hello")
        )
        user_text_id = user_result["user_text"]

        # Configure turn-based methods to return appropriate values
        # After adding user message, max turn is 0
        mock_graph_store.get_max_turn_index = AsyncMock(return_value=0)
        mock_graph_store.get_last_node_in_turn = AsyncMock(
            return_value=UserText(
                id=user_text_id, text="Hello", conversation_id="conv-1", turn_index=0
            )
        )

        # Then add assistant message
        result = await tm.add_message(
            "conv-1", Message(role="assistant", content="Hi there!")
        )

        assert "agent_text" in result
        # create_node called for user + agent, create_edge for MESSAGE relationship
        assert mock_graph_store.create_node.call_count == 2
        mock_graph_store.create_edge.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_tool_stores_result(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Tool messages should store their content for later use."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        message = Message(
            role="tool", content="file content here", tool_call_id="call_123"
        )
        await tm.add_message("conv-1", message)

        assert tm._tool_results["call_123"] == "file content here"

    @pytest.mark.asyncio
    async def test_import_trace_with_tool_calls(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Import a trace with tool calls and results."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        # Create a test file
        test_file = tmp_path / "auth.py"
        test_file.touch()

        messages = [
            Message(role="user", content="Read auth.py"),
            Message(
                role="assistant",
                content="I'll read the file",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        args={"file_path": str(test_file)},
                    )
                ],
            ),
            Message(
                role="tool", content="def authenticate(): pass", tool_call_id="call_1"
            ),
        ]

        result = await tm.import_trace("conv-1", messages)

        assert "user_text" in result
        assert "agent_text" in result
        # Resource nodes should be created
        mock_graph_store.create_node.assert_called()

    @pytest.mark.asyncio
    async def test_import_trace_collects_tool_results_first(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Import trace should collect tool results before processing."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        messages = [
            Message(role="user", content="Hello"),
            Message(role="tool", content="result1", tool_call_id="call_1"),
            Message(role="tool", content="result2", tool_call_id="call_2"),
        ]

        await tm.import_trace("conv-1", messages)

        # Tool results should be collected
        assert "call_1" in tm._tool_results
        assert "call_2" in tm._tool_results


class TestTraceMemToolUses:
    """Test tool_uses tracking on AgentText nodes."""

    @pytest.mark.asyncio
    async def test_agent_text_stores_tool_uses(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """AgentText node should have tool_uses populated from message."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        # Configure turn-based methods
        mock_graph_store.get_max_turn_index = AsyncMock(return_value=0)
        mock_graph_store.get_last_node_in_turn = AsyncMock(
            return_value=UserText(text="Hello", conversation_id="conv-1", turn_index=0)
        )

        message = Message(
            role="assistant",
            content="I'll run that command",
            tool_calls=[
                ToolCall(id="call_1", name="bash", args={"command": "git status"}),
                ToolCall(id="call_2", name="read_file", args={"path": "config.py"}),
            ],
        )

        await tm.add_message("conv-1", message)

        # Verify create_node was called with AgentText containing tool_uses
        create_node_call = mock_graph_store.create_node.call_args
        agent_text = create_node_call[0][0]

        assert isinstance(agent_text, AgentText)
        assert len(agent_text.tool_uses) == 2
        assert agent_text.tool_uses[0].id == "call_1"
        assert agent_text.tool_uses[0].name == "bash"
        assert agent_text.tool_uses[0].args == {"command": "git status"}
        assert agent_text.tool_uses[1].id == "call_2"
        assert agent_text.tool_uses[1].name == "read_file"
        assert agent_text.tool_uses[1].args == {"path": "config.py"}

    @pytest.mark.asyncio
    async def test_tool_uses_empty_when_no_tools(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """AgentText without tools should have empty tool_uses list."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        mock_graph_store.get_max_turn_index = AsyncMock(return_value=0)
        mock_graph_store.get_last_node_in_turn = AsyncMock(
            return_value=UserText(text="Hello", conversation_id="conv-1", turn_index=0)
        )

        message = Message(role="assistant", content="Hello, how can I help?")
        await tm.add_message("conv-1", message)

        create_node_call = mock_graph_store.create_node.call_args
        agent_text = create_node_call[0][0]

        assert isinstance(agent_text, AgentText)
        assert agent_text.tool_uses == []

    @pytest.mark.asyncio
    async def test_tool_uses_includes_non_resource_tools(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Tools that don't produce resources (bash, web_search) should be tracked."""
        config = TraceMemConfig()
        tm = TraceMem(config=config, embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        mock_graph_store.get_max_turn_index = AsyncMock(return_value=0)
        mock_graph_store.get_last_node_in_turn = AsyncMock(
            return_value=UserText(text="Hello", conversation_id="conv-1", turn_index=0)
        )

        # These tools don't create resources via the extractor
        message = Message(
            role="assistant",
            content="Let me check that",
            tool_calls=[
                ToolCall(id="c1", name="bash", args={"command": "echo hello"}),
                ToolCall(id="c2", name="web_search", args={"query": "python docs"}),
                ToolCall(id="c3", name="grep", args={"pattern": "TODO"}),
            ],
        )

        await tm.add_message("conv-1", message)

        create_node_call = mock_graph_store.create_node.call_args
        agent_text = create_node_call[0][0]

        assert isinstance(agent_text, AgentText)
        assert len(agent_text.tool_uses) == 3
        tool_names = [tu.name for tu in agent_text.tool_uses]
        assert tool_names == ["bash", "web_search", "grep"]


class TestToolUseRecord:
    """Test ToolUseRecord model."""

    def test_tool_use_record_creation(self) -> None:
        """ToolUseRecord can be created with id, name, args."""
        record = ToolUseRecord(
            id="call_123",
            name="read_file",
            args={"path": "/path/to/file.py"},
        )

        assert record.id == "call_123"
        assert record.name == "read_file"
        assert record.args == {"path": "/path/to/file.py"}

    def test_tool_use_record_default_args(self) -> None:
        """ToolUseRecord args defaults to empty dict."""
        record = ToolUseRecord(id="call_123", name="bash")
        assert record.args == {}


class TestTraceMemConfig:
    """Test configuration."""

    def test_default_config(self) -> None:
        """Default config should have sensible defaults."""
        config = TraceMemConfig()

        assert config.neo4j_uri == "bolt://localhost:7687"
        assert config.embedding_dimensions == 1536

    def test_default_graph_store_is_kuzu(self) -> None:
        """Default graph store should be kuzu."""
        config = TraceMemConfig()
        assert config.graph_store == "kuzu"

    def test_lancedb_path_default(self) -> None:
        """Default LanceDB path should be in home directory."""
        config = TraceMemConfig()

        path = config.get_lancedb_path()
        assert path == Path.home() / ".tracemem" / "vectors"

    def test_lancedb_path_custom(self, tmp_path: Path) -> None:
        """Custom LanceDB path should override defaults."""
        custom_path = tmp_path / "custom" / "vectors"
        config = TraceMemConfig(lancedb_path=custom_path)

        path = config.get_lancedb_path()
        assert path == custom_path

    def test_default_reranker_is_rrf(self) -> None:
        """Default reranker config is 'rrf'."""
        config = TraceMemConfig()
        assert config.reranker == "rrf"

    def test_reranker_config_accepts_string(self) -> None:
        """Reranker config accepts string keys."""
        config = TraceMemConfig(reranker="linear")
        assert config.reranker == "linear"

    def test_default_retrieval_config(self) -> None:
        """Default retrieval config is a default RetrievalConfig."""
        config = TraceMemConfig()
        assert isinstance(config.retrieval, RetrievalConfig)
        assert config.retrieval.limit == 10

    def test_get_home_default(self) -> None:
        """Default home should be ~/.tracemem."""
        config = TraceMemConfig()
        assert config.get_home() == Path.home() / ".tracemem"

    def test_get_home_custom(self, tmp_path: Path) -> None:
        """Custom home should override default."""
        custom_home = tmp_path / "custom"
        config = TraceMemConfig(home=custom_home)
        assert config.get_home() == custom_home

    def test_get_graph_path(self, tmp_path: Path) -> None:
        """Graph path should be home/graph."""
        config = TraceMemConfig(home=tmp_path / ".tracemem")
        assert config.get_graph_path() == tmp_path / ".tracemem" / "graph"

    def test_get_vector_path(self, tmp_path: Path) -> None:
        """Vector path should be home/vectors."""
        config = TraceMemConfig(home=tmp_path / ".tracemem")
        assert config.get_vector_path() == tmp_path / ".tracemem" / "vectors"

    def test_get_vector_path_backward_compat_lancedb(self, tmp_path: Path) -> None:
        """Custom lancedb_path should override get_vector_path."""
        custom_path = tmp_path / "old_lancedb"
        config = TraceMemConfig(lancedb_path=custom_path)
        assert config.get_vector_path() == custom_path
        assert config.get_lancedb_path() == custom_path


class TestTraceMemStoreSelection:
    """Test that TraceMem selects the correct graph store based on config."""

    def test_default_creates_kuzu_store(
        self, tmp_path: Path, mock_embedder: MockEmbedder
    ) -> None:
        """Default config should create KuzuGraphStore."""
        from tracemem_core.storage.graph.kuzu_store import KuzuGraphStore

        config = TraceMemConfig(home=tmp_path / ".tracemem")
        tm = TraceMem(config=config, embedder=mock_embedder)
        assert isinstance(tm._graph_store, KuzuGraphStore)

    @pytest.mark.neo4j
    def test_neo4j_config_creates_neo4j_store(
        self, tmp_path: Path, mock_embedder: MockEmbedder
    ) -> None:
        """graph_store='neo4j' should create Neo4jGraphStore."""
        from tracemem_core.storage.graph.neo import Neo4jGraphStore

        config = TraceMemConfig(
            home=tmp_path / ".tracemem",
            graph_store="neo4j",
        )
        tm = TraceMem(config=config, embedder=mock_embedder)
        assert isinstance(tm._graph_store, Neo4jGraphStore)


class TestTraceMemReranker:
    """Test reranker wiring through TraceMem."""

    def test_default_reranker_from_config(self, mock_embedder: MockEmbedder) -> None:
        """TraceMem passes reranker string from config to LanceDBVectorStore."""
        config = TraceMemConfig(reranker="rrf")
        tm = TraceMem(config=config, embedder=mock_embedder)
        assert isinstance(tm._vector_store._reranker, RRFReranker)

    def test_config_reranker_linear(self, mock_embedder: MockEmbedder) -> None:
        """Config reranker='linear' resolves to LinearCombinationReranker."""
        config = TraceMemConfig(reranker="linear")
        tm = TraceMem(config=config, embedder=mock_embedder)
        assert isinstance(tm._vector_store._reranker, LinearCombinationReranker)

    def test_explicit_reranker_overrides_config(
        self, mock_embedder: MockEmbedder
    ) -> None:
        """Explicit reranker param overrides config string."""
        config = TraceMemConfig(reranker="rrf")
        tm = TraceMem(config=config, embedder=mock_embedder, reranker="linear")
        assert isinstance(tm._vector_store._reranker, LinearCombinationReranker)

    def test_explicit_reranker_instance_overrides_config(
        self, mock_embedder: MockEmbedder
    ) -> None:
        """A custom reranker instance overrides config string."""
        custom = MagicMock()
        config = TraceMemConfig(reranker="rrf")
        tm = TraceMem(config=config, embedder=mock_embedder, reranker=custom)
        assert tm._vector_store._reranker is custom


class TestTraceMemRetrieval:
    """Test top-level retrieval delegate methods on TraceMem."""

    @pytest.mark.asyncio
    async def test_search_delegates_to_retrieval(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """TraceMem.search() delegates to retrieval.search() with default config."""
        tm = TraceMem(config=TraceMemConfig(), embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        mock_retrieval = MagicMock()
        mock_retrieval.search = AsyncMock(return_value=[])
        tm._retrieval = mock_retrieval

        await tm.search("test query")

        mock_retrieval.search.assert_called_once()
        call_args = mock_retrieval.search.call_args
        assert call_args[0][0] == "test query"
        assert isinstance(call_args.kwargs["config"], RetrievalConfig)

    @pytest.mark.asyncio
    async def test_search_uses_custom_config(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """TraceMem.search(config=custom) uses custom config over default."""
        tm = TraceMem(config=TraceMemConfig(), embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        mock_retrieval = MagicMock()
        mock_retrieval.search = AsyncMock(return_value=[])
        tm._retrieval = mock_retrieval

        custom_config = RetrievalConfig(limit=3)
        await tm.search("test query", config=custom_config)

        call_args = mock_retrieval.search.call_args
        assert call_args.kwargs["config"] is custom_config

    @pytest.mark.asyncio
    async def test_get_context_delegates(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """TraceMem.get_context() delegates to retrieval.get_context()."""
        tm = TraceMem(config=TraceMemConfig(), embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        mock_retrieval = MagicMock()
        mock_retrieval.get_context = AsyncMock(return_value=ContextResult())
        tm._retrieval = mock_retrieval

        node_id = uuid4()
        await tm.get_context(node_id)

        mock_retrieval.get_context.assert_called_once_with(node_id)

    @pytest.mark.asyncio
    async def test_get_conversations_for_resource_delegates(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """TraceMem.get_conversations_for_resource() delegates correctly."""
        tm = TraceMem(config=TraceMemConfig(), embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        mock_retrieval = MagicMock()
        mock_retrieval.get_conversations_for_resource = AsyncMock(return_value=[])
        tm._retrieval = mock_retrieval

        await tm.get_conversations_for_resource("file://src/auth.py")

        mock_retrieval.get_conversations_for_resource.assert_called_once()
        call_args = mock_retrieval.get_conversations_for_resource.call_args
        assert call_args[0][0] == "file://src/auth.py"
        assert isinstance(call_args.kwargs["config"], RetrievalConfig)

    @pytest.mark.asyncio
    async def test_get_trajectory_delegates(
        self,
        mock_embedder: MockEmbedder,
        mock_graph_store: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """TraceMem.get_trajectory() delegates correctly."""
        tm = TraceMem(config=TraceMemConfig(), embedder=mock_embedder)
        tm._graph_store = mock_graph_store
        tm._vector_store = mock_vector_store

        mock_retrieval = MagicMock()
        mock_retrieval.get_trajectory = AsyncMock(return_value=TrajectoryResult())
        tm._retrieval = mock_retrieval

        node_id = uuid4()
        custom_config = RetrievalConfig(trajectory_max_depth=50)
        await tm.get_trajectory(node_id, config=custom_config)

        mock_retrieval.get_trajectory.assert_called_once()
        call_args = mock_retrieval.get_trajectory.call_args
        assert call_args[0][0] == node_id
        assert call_args.kwargs["config"] is custom_config
