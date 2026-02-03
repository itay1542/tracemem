"""Unit tests for retrieval strategies."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tracemem_core.retrieval import (
    ConversationReference,
    ContextResult,
    HybridRetrievalStrategy,
    RetrievalConfig,
    RetrievalResult,
)
from tracemem_core.retrieval.results import (
    AgentTextInfo,
    ResourceInfo,
    ResourceVersionInfo,
    ToolUse,
    UserTextInfo,
)
from tracemem_core.storage.protocols import VectorSearchResult


class TestRetrievalConfig:
    """Tests for RetrievalConfig model."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = RetrievalConfig()

        assert config.limit == 10
        assert config.include_context is True
        assert config.vector_weight == 0.5
        assert config.expand_tool_uses is True
        assert config.expand_resources is True
        assert config.sort_by == "created_at"
        assert config.sort_order == "desc"
        assert config.exclude_conversation_id is None
        assert config.trajectory_max_depth == 100

    def test_custom_values(self):
        """Verify custom configuration values are set."""
        config = RetrievalConfig(
            limit=5,
            include_context=False,
            vector_weight=0.9,
            expand_tool_uses=False,
            expand_resources=False,
            sort_by="last_accessed_at",
            sort_order="asc",
            exclude_conversation_id="conv-123",
            trajectory_max_depth=50,
        )

        assert config.limit == 5
        assert config.include_context is False
        assert config.vector_weight == 0.9
        assert config.expand_tool_uses is False
        assert config.expand_resources is False
        assert config.sort_by == "last_accessed_at"
        assert config.sort_order == "asc"
        assert config.exclude_conversation_id == "conv-123"
        assert config.trajectory_max_depth == 50

    def test_limit_validation_min(self):
        """Verify limit must be at least 1."""
        with pytest.raises(ValueError):
            RetrievalConfig(limit=0)

    def test_limit_validation_max(self):
        """Verify limit must be at most 100."""
        with pytest.raises(ValueError):
            RetrievalConfig(limit=101)

    def test_vector_weight_validation_min(self):
        """Verify vector_weight must be at least 0.0."""
        with pytest.raises(ValueError):
            RetrievalConfig(vector_weight=-0.1)

    def test_vector_weight_validation_max(self):
        """Verify vector_weight must be at most 1.0."""
        with pytest.raises(ValueError):
            RetrievalConfig(vector_weight=1.1)

    def test_trajectory_max_depth_validation_min(self):
        """Verify trajectory_max_depth must be at least 1."""
        with pytest.raises(ValueError):
            RetrievalConfig(trajectory_max_depth=0)

    def test_trajectory_max_depth_validation_max(self):
        """Verify trajectory_max_depth must be at most 500."""
        with pytest.raises(ValueError):
            RetrievalConfig(trajectory_max_depth=501)

    def test_model_copy_update(self):
        """Verify model_copy with update works."""
        config = RetrievalConfig(limit=5)
        updated = config.model_copy(update={"include_context": False})

        assert config.include_context is True  # Original unchanged
        assert updated.include_context is False
        assert updated.limit == 5  # Other fields preserved


class TestConversationReference:
    """Tests for ConversationReference model."""

    def test_required_fields(self):
        """Verify required fields are enforced."""
        ref = ConversationReference(
            conversation_id="conv-1",
            user_text_id="user-123",
            user_text="Hello, world!",
        )

        assert ref.conversation_id == "conv-1"
        assert ref.user_text_id == "user-123"
        assert ref.user_text == "Hello, world!"

    def test_optional_agent_text(self):
        """Verify agent_text is optional and defaults to None."""
        ref = ConversationReference(
            conversation_id="conv-1",
            user_text_id="user-123",
            user_text="Hello",
        )

        assert ref.agent_text is None

    def test_agent_text_included(self):
        """Verify agent_text can be set."""
        ref = ConversationReference(
            conversation_id="conv-1",
            user_text_id="user-123",
            user_text="Hello",
            agent_text="Hi there!",
        )

        assert ref.agent_text == "Hi there!"

    def test_created_at_optional(self):
        """Verify created_at is optional."""
        ref = ConversationReference(
            conversation_id="conv-1",
            user_text_id="user-123",
            user_text="Hello",
        )

        assert ref.created_at is None

    def test_created_at_with_datetime(self):
        """Verify created_at accepts datetime."""
        now = datetime.now(UTC)
        ref = ConversationReference(
            conversation_id="conv-1",
            user_text_id="user-123",
            user_text="Hello",
            created_at=now,
        )

        assert ref.created_at == now

    def test_str(self):
        """Verify __str__ produces readable output."""
        ref = ConversationReference(
            conversation_id="conv-1",
            user_text_id="user-123",
            user_text="Hello",
        )

        assert "conv-1" in str(ref)
        assert "Hello" in str(ref)


class TestHybridRetrievalStrategySearch:
    """Tests for HybridRetrievalStrategy.search method."""

    @pytest.fixture
    def strategy(self, mock_graph_store, mock_vector_store, mock_embedder):
        """Create a HybridRetrievalStrategy with mocks."""
        return HybridRetrievalStrategy(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

    async def test_search_calls_embedder(self, strategy, mock_embedder):
        """Verify search embeds the query."""
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)

        await strategy.search("test query")

        mock_embedder.embed.assert_called_once_with("test query")

    async def test_search_calls_vector_store(self, strategy, mock_vector_store, mock_embedder):
        """Verify search uses vector store with correct params."""
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_vector_store.search = AsyncMock(return_value=[])

        config = RetrievalConfig(limit=5, vector_weight=0.8)
        await strategy.search("test query", config=config)

        mock_vector_store.search.assert_called_once_with(
            query_vector=[0.1] * 1536,
            query_text="test query",
            limit=5,
            exclude_conversation_id=None,
            vector_weight=0.8,
        )

    async def test_search_with_config_exclude_conversation(
        self, strategy, mock_vector_store, mock_embedder
    ):
        """Verify config.exclude_conversation_id is passed to vector store."""
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_vector_store.search = AsyncMock(return_value=[])

        config = RetrievalConfig(exclude_conversation_id="conv-exclude")
        await strategy.search("test query", config=config)

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["exclude_conversation_id"] == "conv-exclude"

    async def test_search_defaults_to_retrieval_config(
        self, strategy, mock_vector_store, mock_embedder
    ):
        """Verify search uses default RetrievalConfig when none provided."""
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_vector_store.search = AsyncMock(return_value=[])

        await strategy.search("test query")

        mock_vector_store.search.assert_called_once()
        call_kwargs = mock_vector_store.search.call_args.kwargs
        assert call_kwargs["limit"] == 10  # Default
        assert call_kwargs["exclude_conversation_id"] is None

    async def test_search_returns_retrieval_results(
        self, strategy, mock_vector_store, mock_embedder
    ):
        """Verify search returns properly formatted results."""
        node_id = uuid4()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_vector_store.search = AsyncMock(
            return_value=[
                VectorSearchResult(
                    node_id=node_id,
                    text="matching text",
                    conversation_id="conv-1",
                    created_at=datetime.now(UTC),
                    last_accessed=datetime.now(UTC),
                    score=0.95,
                )
            ]
        )

        config = RetrievalConfig(include_context=False)
        results = await strategy.search("test query", config=config)

        assert len(results) == 1
        assert results[0].node_id == node_id
        assert results[0].text == "matching text"
        assert results[0].conversation_id == "conv-1"
        assert results[0].score == 0.95
        assert results[0].context is None

    async def test_search_with_context_delegates_to_graph_store(
        self, strategy, mock_graph_store, mock_vector_store, mock_embedder
    ):
        """Verify search with include_context=True calls get_node_context."""
        node_id = uuid4()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_vector_store.search = AsyncMock(
            return_value=[
                VectorSearchResult(
                    node_id=node_id,
                    text="query text",
                    conversation_id="conv-1",
                    created_at=datetime.now(UTC),
                    last_accessed=datetime.now(UTC),
                    score=0.9,
                )
            ]
        )
        mock_graph_store.get_node_context = AsyncMock(
            return_value=ContextResult(
                user_text=UserTextInfo(id=str(node_id), text="query text", conversation_id="conv-1"),
                agent_text=AgentTextInfo(id="agent-1", text="response text"),
            )
        )

        config = RetrievalConfig(include_context=True)
        results = await strategy.search("test query", config=config)

        assert len(results) == 1
        assert results[0].context is not None
        assert results[0].context.user_text.text == "query text"
        assert results[0].context.agent_text.text == "response text"
        mock_graph_store.get_node_context.assert_called_once_with(node_id)

    async def test_search_updates_last_accessed(
        self, strategy, mock_graph_store, mock_vector_store, mock_embedder
    ):
        """Verify search updates last_accessed timestamps."""
        node_id = uuid4()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_vector_store.search = AsyncMock(
            return_value=[
                VectorSearchResult(
                    node_id=node_id,
                    text="text",
                    conversation_id="conv-1",
                    created_at=datetime.now(UTC),
                    last_accessed=datetime.now(UTC),
                    score=0.9,
                )
            ]
        )

        config = RetrievalConfig(include_context=False)
        await strategy.search("test query", config=config)

        mock_graph_store.update_last_accessed.assert_called_once_with([node_id])
        mock_vector_store.update_last_accessed.assert_called_once_with(node_id)


class TestHybridRetrievalStrategyGetContext:
    """Tests for HybridRetrievalStrategy.get_context method."""

    @pytest.fixture
    def strategy(self, mock_graph_store, mock_vector_store, mock_embedder):
        """Create a HybridRetrievalStrategy with mocks."""
        return HybridRetrievalStrategy(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

    async def test_get_context_delegates_to_graph_store(self, strategy, mock_graph_store):
        """Verify get_context delegates to graph_store.get_node_context."""
        node_id = uuid4()
        expected = ContextResult(
            user_text=UserTextInfo(id=str(node_id), text="question", conversation_id="c1"),
            agent_text=AgentTextInfo(id="a1", text="answer"),
        )
        mock_graph_store.get_node_context = AsyncMock(return_value=expected)

        result = await strategy.get_context(node_id)

        mock_graph_store.get_node_context.assert_called_once_with(node_id)
        assert result.user_text.text == "question"
        assert result.agent_text.text == "answer"

    async def test_get_context_returns_empty_when_not_found(self, strategy, mock_graph_store):
        """Verify get_context returns empty context when node not found."""
        mock_graph_store.get_node_context = AsyncMock(return_value=ContextResult())

        context = await strategy.get_context(uuid4())

        assert context.user_text is None
        assert context.agent_text is None
        assert context.tool_uses == []


class TestHybridRetrievalStrategyGetConversationsForResource:
    """Tests for HybridRetrievalStrategy.get_conversations_for_resource method."""

    @pytest.fixture
    def strategy(self, mock_graph_store, mock_vector_store, mock_embedder):
        """Create a HybridRetrievalStrategy with mocks."""
        return HybridRetrievalStrategy(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

    async def test_get_conversations_returns_empty_list(self, strategy, mock_graph_store):
        """Verify returns empty list when no conversations found."""
        mock_graph_store.get_resource_conversations = AsyncMock(return_value=[])

        result = await strategy.get_conversations_for_resource("file:///test.py")

        assert result == []

    async def test_get_conversations_delegates_to_graph_store(self, strategy, mock_graph_store):
        """Verify delegates to graph_store.get_resource_conversations."""
        expected = [
            ConversationReference(
                conversation_id="conv-1",
                user_text_id="user-1",
                user_text="Read the file",
                agent_text="Here's the content",
                created_at=datetime.now(UTC),
            )
        ]
        mock_graph_store.get_resource_conversations = AsyncMock(return_value=expected)

        result = await strategy.get_conversations_for_resource("file:///test.py")

        assert len(result) == 1
        assert result[0].conversation_id == "conv-1"
        assert result[0].user_text == "Read the file"

    async def test_get_conversations_passes_config(self, strategy, mock_graph_store):
        """Verify config values are passed to graph_store."""
        mock_graph_store.get_resource_conversations = AsyncMock(return_value=[])

        config = RetrievalConfig(
            limit=5,
            sort_by="last_accessed_at",
            sort_order="asc",
            exclude_conversation_id="conv-exclude",
        )
        await strategy.get_conversations_for_resource("file:///test.py", config=config)

        mock_graph_store.get_resource_conversations.assert_called_once_with(
            "file:///test.py",
            limit=5,
            sort_by="last_accessed_at",
            sort_order="asc",
            exclude_conversation_id="conv-exclude",
        )

    async def test_get_conversations_uses_default_config(self, strategy, mock_graph_store):
        """Verify default config is used when none provided."""
        mock_graph_store.get_resource_conversations = AsyncMock(return_value=[])

        await strategy.get_conversations_for_resource("file:///test.py")

        mock_graph_store.get_resource_conversations.assert_called_once_with(
            "file:///test.py",
            limit=10,
            sort_by="created_at",
            sort_order="desc",
            exclude_conversation_id=None,
        )


class TestHybridRetrievalStrategyGetTrajectory:
    """Tests for HybridRetrievalStrategy.get_trajectory method."""

    @pytest.fixture
    def strategy(self, mock_graph_store, mock_vector_store, mock_embedder):
        """Create a HybridRetrievalStrategy with mocks."""
        return HybridRetrievalStrategy(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

    async def test_get_trajectory_delegates_to_graph_store(self, strategy, mock_graph_store):
        """Verify get_trajectory delegates data fetch to graph_store."""
        node_id = uuid4()
        mock_graph_store.get_trajectory_nodes = AsyncMock(return_value=[])

        await strategy.get_trajectory(node_id)

        mock_graph_store.get_trajectory_nodes.assert_called_once_with(
            node_id, max_depth=100,
        )

    async def test_get_trajectory_passes_max_depth_from_config(self, strategy, mock_graph_store):
        """Verify trajectory_max_depth from config is passed."""
        node_id = uuid4()
        mock_graph_store.get_trajectory_nodes = AsyncMock(return_value=[])

        config = RetrievalConfig(trajectory_max_depth=50)
        await strategy.get_trajectory(node_id, config=config)

        mock_graph_store.get_trajectory_nodes.assert_called_once_with(
            node_id, max_depth=50,
        )


class TestContextResultModel:
    """Tests for ContextResult model."""

    def test_empty_context(self):
        """Verify empty context has correct defaults."""
        context = ContextResult()

        assert context.user_text is None
        assert context.agent_text is None
        assert context.tool_uses == []

    def test_context_with_all_fields(self):
        """Verify context with all fields populated."""
        context = ContextResult(
            user_text=UserTextInfo(id="u1", text="question", conversation_id="c1"),
            agent_text=AgentTextInfo(id="a1", text="answer"),
            tool_uses=[
                ToolUse(
                    tool_name="READ_FILE",
                    properties={"path": "/test.py"},
                    resource_version=ResourceVersionInfo(
                        id="v1", uri="file:///test.py", content_hash="abc"
                    ),
                    resource=ResourceInfo(id="r1", uri="file:///test.py"),
                )
            ],
        )

        assert context.user_text is not None
        assert context.user_text.text == "question"
        assert context.agent_text is not None
        assert context.agent_text.text == "answer"
        assert len(context.tool_uses) == 1
        assert context.tool_uses[0].tool_name == "READ_FILE"

    def test_str(self):
        """Verify __str__ produces readable output."""
        context = ContextResult(
            user_text=UserTextInfo(id="u1", text="question", conversation_id="c1"),
            agent_text=AgentTextInfo(id="a1", text="answer"),
        )

        s = str(context)
        assert "question" in s
        assert "answer" in s


class TestRetrievalResultModel:
    """Tests for RetrievalResult model."""

    def test_required_fields(self):
        """Verify required fields are enforced."""
        node_id = uuid4()
        result = RetrievalResult(
            node_id=node_id,
            text="matching text",
            conversation_id="conv-1",
            score=0.95,
        )

        assert result.node_id == node_id
        assert result.text == "matching text"
        assert result.conversation_id == "conv-1"
        assert result.score == 0.95
        assert result.context is None

    def test_with_context(self):
        """Verify context can be attached."""
        result = RetrievalResult(
            node_id=uuid4(),
            text="text",
            conversation_id="c1",
            score=0.8,
            context=ContextResult(
                user_text=UserTextInfo(id="u1", text="q", conversation_id="c1")
            ),
        )

        assert result.context is not None
        assert result.context.user_text is not None
        assert result.context.user_text.text == "q"

    def test_str(self):
        """Verify __str__ produces readable output."""
        result = RetrievalResult(
            node_id=uuid4(),
            text="matching text",
            conversation_id="conv-1",
            score=0.95,
        )

        s = str(result)
        assert "0.950" in s
        assert "conv-1" in s
        assert "matching text" in s
