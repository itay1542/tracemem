"""Backend-agnostic integration tests, parameterized for Kuzu and Neo4j.

These tests exercise the TraceMem public API and protocol methods without
raw driver queries, so they run identically against both graph backends.

Run Kuzu-only (no Docker):
    uv run pytest tests/integration/test_common.py -m "not neo4j" -v

Run all (Docker required for Neo4j):
    uv run pytest tests/integration/test_common.py -v
"""

import hashlib

import pytest

from tracemem_core.extractors import _canonicalize_file_uri
from tracemem_core.messages import Message, ToolCall
from tracemem_core.retrieval import HybridRetrievalStrategy, RetrievalConfig
from tracemem_core.tracemem import TraceMem


@pytest.fixture(params=[
    "kuzu",
    pytest.param("neo4j", marks=pytest.mark.neo4j),
])
def tracemem(request):
    """Yield a connected TraceMem instance for the requested backend.

    Uses lazy fixture resolution so that the Neo4j fixture is only
    instantiated when the neo4j param is selected (and vice versa).
    """
    if request.param == "kuzu":
        return request.getfixturevalue("tracemem_kuzu")
    return request.getfixturevalue("tracemem_integration")


@pytest.fixture
def retrieval(tracemem: TraceMem):
    """Provide a HybridRetrievalStrategy wired to the parameterized TraceMem."""
    return HybridRetrievalStrategy(
        graph_store=tracemem._graph_store,
        vector_store=tracemem._vector_store,
        embedder=tracemem._embedder,
    )


# ---------------------------------------------------------------------------
# TestUserMessageIngestion
# ---------------------------------------------------------------------------

class TestUserMessageIngestion:
    """Tests for user message ingestion into graph and vector stores."""

    async def test_add_user_message_creates_node(self, tracemem: TraceMem):
        """Verify user message creates UserText node in the graph."""
        message = Message(role="user", content="Hello world")
        result = await tracemem.add_message("conv-1", message)

        assert "user_text" in result
        user_text_id = result["user_text"]

        user_text = await tracemem._graph_store.get_user_text(user_text_id)
        assert user_text is not None
        assert user_text.text == "Hello world"
        assert user_text.conversation_id == "conv-1"

    async def test_add_user_message_indexes_in_vector_store(self, tracemem: TraceMem):
        """Verify user message is indexed in the vector store."""
        message = Message(role="user", content="Search test content")
        result = await tracemem.add_message("conv-1", message)

        user_text_id = result["user_text"]

        query_vector = await tracemem._embedder.embed("Search test")
        search_results = await tracemem._vector_store.search(
            query_vector=query_vector,
            query_text="Search test",
            limit=10,
        )
        assert len(search_results) >= 1
        assert any(r.node_id == user_text_id for r in search_results)

    async def test_add_multiple_user_messages_all_searchable(self, tracemem: TraceMem):
        """Verify multiple user messages are all indexed and searchable."""
        messages = [
            ("conv-1", "First message about Python programming"),
            ("conv-1", "Second message about database queries"),
            ("conv-2", "Third message about API design"),
        ]

        created_ids = []
        for conv_id, content in messages:
            result = await tracemem.add_message(
                conv_id, Message(role="user", content=content)
            )
            created_ids.append(result["user_text"])

        query_vector = await tracemem._embedder.embed("programming")
        results = await tracemem._vector_store.search(
            query_vector=query_vector,
            query_text="programming",
            limit=10,
        )

        found_ids = {r.node_id for r in results}
        assert len(found_ids) == 3


# ---------------------------------------------------------------------------
# TestAssistantMessageIngestion
# ---------------------------------------------------------------------------

class TestAssistantMessageIngestion:
    """Tests for assistant message ingestion and graph relationships."""

    async def test_add_assistant_message_creates_agent_text_node(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify assistant message creates AgentText node."""
        user_result = await tracemem.add_message(
            "conv-1", Message(role="user", content="Hi")
        )
        user_text_id = user_result["user_text"]

        result = await tracemem.add_message(
            "conv-1", Message(role="assistant", content="Hello!")
        )

        assert "agent_text" in result

        context = await retrieval.get_context(user_text_id)
        assert context.user_text is not None
        assert context.user_text.text == "Hi"
        assert context.agent_text is not None
        assert context.agent_text.text == "Hello!"


# ---------------------------------------------------------------------------
# TestFullTraceImport
# ---------------------------------------------------------------------------

class TestFullTraceImport:
    """Tests for importing complete traces with tool calls."""

    async def test_import_trace_creates_full_graph(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify import_trace creates complete graph structure."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        messages = [
            Message(role="user", content="Read test.py"),
            Message(
                role="assistant",
                content="Reading the file...",
                tool_calls=[
                    ToolCall(id="call_1", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content="def hello(): pass", tool_call_id="call_1"),
        ]

        result = await tracemem.import_trace("conv-1", messages)

        assert "user_text" in result
        assert "agent_text" in result

        resource_keys = [k for k in result.keys() if k.startswith("resource_") and not k.startswith("resource_version_")]
        version_keys = [k for k in result.keys() if k.startswith("resource_version_")]

        assert len(resource_keys) == 1
        assert len(version_keys) == 1

        context = await retrieval.get_context(result["user_text"])
        assert context.user_text.text == "Read test.py"
        assert context.agent_text.text == "Reading the file..."
        assert len(context.tool_uses) == 1

    async def test_import_trace_with_multiple_tool_calls(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify import_trace handles multiple tool calls in one message."""
        file1 = tmp_path / "auth.py"
        file2 = tmp_path / "utils.py"
        file1.write_text("def login(): pass")
        file2.write_text("def helper(): pass")

        messages = [
            Message(role="user", content="Read auth.py and utils.py"),
            Message(
                role="assistant",
                content="Reading both files...",
                tool_calls=[
                    ToolCall(id="call_1", name="read_file", args={"path": str(file1)}),
                    ToolCall(id="call_2", name="read_file", args={"path": str(file2)}),
                ],
            ),
            Message(role="tool", content="def login(): pass", tool_call_id="call_1"),
            Message(role="tool", content="def helper(): pass", tool_call_id="call_2"),
        ]

        result = await tracemem.import_trace("conv-1", messages)

        resource_keys = [k for k in result.keys() if k.startswith("resource_") and not k.startswith("resource_version_")]
        version_keys = [k for k in result.keys() if k.startswith("resource_version_")]

        assert len(resource_keys) == 2
        assert len(version_keys) == 2

        context = await retrieval.get_context(result["user_text"])
        assert len(context.tool_uses) == 2


# ---------------------------------------------------------------------------
# TestResourceVersioning
# ---------------------------------------------------------------------------

class TestResourceVersioning:
    """Tests for resource versioning when content changes."""

    async def test_resource_versioning_on_content_change(
        self, tracemem: TraceMem, tmp_path
    ):
        """Verify new ResourceVersion created when content changes."""
        test_file = tmp_path / "code.py"

        messages1 = [
            Message(role="user", content="Read code.py"),
            Message(
                role="assistant",
                content="Reading...",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content="v1", tool_call_id="c1"),
        ]
        await tracemem.import_trace("conv-1", messages1)

        messages2 = [
            Message(role="user", content="Read code.py again"),
            Message(
                role="assistant",
                content="Reading again...",
                tool_calls=[
                    ToolCall(id="c2", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content="v2", tool_call_id="c2"),
        ]
        await tracemem.import_trace("conv-2", messages2)

        canonical_uri = _canonicalize_file_uri(f"file://{test_file}", root=None)
        resource = await tracemem._graph_store.get_resource_by_uri(canonical_uri)
        assert resource is not None

        expected_hash = hashlib.sha256("v2".encode()).hexdigest()
        assert resource.current_content_hash == expected_hash

    async def test_same_content_does_not_create_new_version(
        self, tracemem: TraceMem, tmp_path
    ):
        """Verify same content hash does not create duplicate versions."""
        test_file = tmp_path / "stable.py"

        messages1 = [
            Message(role="user", content="Read stable.py"),
            Message(
                role="assistant",
                content="Reading...",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content="same content", tool_call_id="c1"),
        ]
        result1 = await tracemem.import_trace("conv-1", messages1)

        messages2 = [
            Message(role="user", content="Read stable.py again"),
            Message(
                role="assistant",
                content="Reading again...",
                tool_calls=[
                    ToolCall(id="c2", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content="same content", tool_call_id="c2"),
        ]
        result2 = await tracemem.import_trace("conv-2", messages2)

        version_keys_1 = [k for k in result1.keys() if k.startswith("resource_version_")]
        assert len(version_keys_1) == 1

        version_keys_2 = [k for k in result2.keys() if k.startswith("resource_version_")]
        assert len(version_keys_2) == 0, "Same content should not create new version"


# ---------------------------------------------------------------------------
# TestMultipleConversations
# ---------------------------------------------------------------------------

class TestMultipleConversations:
    """Tests for conversation isolation and cross-conversation queries."""

    async def test_multiple_conversations_isolated(self, tracemem: TraceMem):
        """Verify conversations are properly isolated in search."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="Message from conversation 1")
        )
        await tracemem.add_message(
            "conv-2", Message(role="user", content="Message from conversation 2")
        )

        query_vector = await tracemem._embedder.embed("Message")
        results = await tracemem._vector_store.search(
            query_vector=query_vector,
            query_text="Message",
            limit=10,
            exclude_conversation_id="conv-1",
        )

        assert len(results) >= 1
        assert all(r.conversation_id != "conv-1" for r in results)

    async def test_conversations_share_resources(
        self, tracemem: TraceMem, tmp_path
    ):
        """Verify multiple conversations accessing same file share Resource node."""
        shared_file = tmp_path / "shared.py"

        messages1 = [
            Message(role="user", content="Read shared.py"),
            Message(
                role="assistant",
                content="Reading...",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(shared_file)})
                ],
            ),
            Message(role="tool", content="shared content", tool_call_id="c1"),
        ]
        result1 = await tracemem.import_trace("conv-1", messages1)

        messages2 = [
            Message(role="user", content="Read shared.py too"),
            Message(
                role="assistant",
                content="Reading again...",
                tool_calls=[
                    ToolCall(id="c2", name="read_file", args={"path": str(shared_file)})
                ],
            ),
            Message(role="tool", content="shared content", tool_call_id="c2"),
        ]
        result2 = await tracemem.import_trace("conv-2", messages2)

        resource_keys_1 = [k for k in result1.keys() if k.startswith("resource_") and not k.startswith("resource_version_")]
        resource_keys_2 = [k for k in result2.keys() if k.startswith("resource_") and not k.startswith("resource_version_")]

        assert len(resource_keys_1) == 1
        assert len(resource_keys_2) == 0, "Second conversation should reuse existing resource"

    async def test_get_conversations_for_resource(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify we can find all conversations that touched a resource."""
        shared_file = tmp_path / "popular.py"

        for i in range(3):
            messages = [
                Message(role="user", content=f"Read popular.py - conv {i}"),
                Message(
                    role="assistant",
                    content="Reading...",
                    tool_calls=[
                        ToolCall(
                            id=f"c{i}",
                            name="read_file",
                            args={"path": str(shared_file)},
                        )
                    ],
                ),
                Message(role="tool", content=f"content v{i}", tool_call_id=f"c{i}"),
            ]
            await tracemem.import_trace(f"conv-{i}", messages)

        canonical_uri = _canonicalize_file_uri(f"file://{shared_file}", root=None)
        conversations = await retrieval.get_conversations_for_resource(canonical_uri)

        assert len(conversations) == 3
        conv_ids = {c.conversation_id for c in conversations}
        assert conv_ids == {"conv-0", "conv-1", "conv-2"}


# ---------------------------------------------------------------------------
# TestCrossConversationSharedResource
# ---------------------------------------------------------------------------

class TestCrossConversationSharedResource:
    """Tests for cross-conversation queries when multiple conversations access the same file."""

    async def test_two_conversations_same_file_same_content(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Two conversations reading same file with same content should both be queryable."""
        config_file = tmp_path / "config.py"
        file_content = "API_KEY = 'secret'\nDEBUG = True"

        messages1 = [
            Message(role="user", content="Show me the config file"),
            Message(
                role="assistant",
                content="I'll read the configuration file for you.",
                tool_calls=[
                    ToolCall(id="call_1", name="read_file", args={"path": str(config_file)})
                ],
            ),
            Message(role="tool", content=file_content, tool_call_id="call_1"),
            Message(
                role="assistant",
                content="Here's your config - it has an API key and debug flag.",
            ),
        ]
        result1 = await tracemem.import_trace("conv-alice", messages1)

        messages2 = [
            Message(role="user", content="What's in the config?"),
            Message(
                role="assistant",
                content="Let me check the configuration.",
                tool_calls=[
                    ToolCall(id="call_2", name="read_file", args={"path": str(config_file)})
                ],
            ),
            Message(role="tool", content=file_content, tool_call_id="call_2"),
            Message(
                role="assistant",
                content="The config contains API_KEY and DEBUG settings.",
            ),
        ]
        result2 = await tracemem.import_trace("conv-bob", messages2)

        resource_keys_1 = [
            k for k in result1.keys()
            if k.startswith("resource_") and not k.startswith("resource_version_")
        ]
        resource_keys_2 = [
            k for k in result2.keys()
            if k.startswith("resource_") and not k.startswith("resource_version_")
        ]
        assert len(resource_keys_1) == 1
        assert len(resource_keys_2) == 0

        version_keys_1 = [k for k in result1.keys() if k.startswith("resource_version_")]
        version_keys_2 = [k for k in result2.keys() if k.startswith("resource_version_")]
        assert len(version_keys_1) == 1
        assert len(version_keys_2) == 0

        canonical_uri = _canonicalize_file_uri(f"file://{config_file}", root=None)
        conversations = await retrieval.get_conversations_for_resource(canonical_uri)

        conv_ids = {c.conversation_id for c in conversations}
        assert conv_ids == {"conv-alice", "conv-bob"}

    async def test_cross_conversation_query_finds_all_users(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify cross-conversation query returns user text from each conversation."""
        auth_file = tmp_path / "auth.py"
        content = "def login(user, password): pass"

        user_messages = [
            ("conv-1", "How does authentication work?"),
            ("conv-2", "Show me the login code"),
            ("conv-3", "Review the auth implementation"),
        ]

        for conv_id, user_msg in user_messages:
            messages = [
                Message(role="user", content=user_msg),
                Message(
                    role="assistant",
                    content="Let me read that file.",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{conv_id}",
                            name="read_file",
                            args={"path": str(auth_file)},
                        )
                    ],
                ),
                Message(role="tool", content=content, tool_call_id=f"call_{conv_id}"),
            ]
            await tracemem.import_trace(conv_id, messages)

        canonical_uri = _canonicalize_file_uri(f"file://{auth_file}", root=None)
        conversations = await retrieval.get_conversations_for_resource(canonical_uri)

        assert len(conversations) == 3

        user_texts = {c.user_text for c in conversations}
        expected_texts = {msg for _, msg in user_messages}
        assert user_texts == expected_texts

    async def test_resource_version_lookup_by_hash(
        self, tracemem: TraceMem, tmp_path
    ):
        """Verify get_resource_version_by_hash returns the correct version."""
        test_file = tmp_path / "test.py"
        content = "def test(): pass"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        messages = [
            Message(role="user", content="Read test.py"),
            Message(
                role="assistant",
                content="Reading...",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content=content, tool_call_id="c1"),
        ]
        await tracemem.import_trace("conv-1", messages)

        canonical_uri = _canonicalize_file_uri(f"file://{test_file}", root=None)
        version = await tracemem._graph_store.get_resource_version_by_hash(
            canonical_uri, content_hash
        )

        assert version is not None
        assert version.uri == canonical_uri
        assert version.content_hash == content_hash

        no_version = await tracemem._graph_store.get_resource_version_by_hash(
            canonical_uri, "nonexistent_hash"
        )
        assert no_version is None


# ---------------------------------------------------------------------------
# TestTurnIndex
# ---------------------------------------------------------------------------

class TestTurnIndex:
    """Tests for turn_index tracking in conversation nodes."""

    async def test_user_message_starts_new_turn(self, tracemem: TraceMem):
        """First user message should be turn 0."""
        result = await tracemem.add_message(
            "conv-1", Message(role="user", content="Hello")
        )

        user_text = await tracemem._graph_store.get_user_text(result["user_text"])
        assert user_text is not None
        assert user_text.turn_index == 0

    async def test_second_user_increments_turn(self, tracemem: TraceMem):
        """Second user message should be turn 1."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="User 1")
        )
        await tracemem.add_message(
            "conv-1", Message(role="assistant", content="Agent 1")
        )

        result = await tracemem.add_message(
            "conv-1", Message(role="user", content="User 2")
        )

        user_text = await tracemem._graph_store.get_user_text(result["user_text"])
        assert user_text is not None
        assert user_text.turn_index == 1

    async def test_get_max_turn_index(self, tracemem: TraceMem):
        """Verify get_max_turn_index returns correct value."""
        max_turn = await tracemem._graph_store.get_max_turn_index("conv-empty")
        assert max_turn == -1

        for i in range(3):
            await tracemem.add_message(
                "conv-1", Message(role="user", content=f"User {i}")
            )
            await tracemem.add_message(
                "conv-1", Message(role="assistant", content=f"Agent {i}")
            )

        max_turn = await tracemem._graph_store.get_max_turn_index("conv-1")
        assert max_turn == 2

    async def test_get_last_node_in_turn(self, tracemem: TraceMem):
        """Verify get_last_node_in_turn returns correct node."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="User message")
        )
        await tracemem.add_message(
            "conv-1", Message(role="assistant", content="Agent message")
        )

        last_node = await tracemem._graph_store.get_last_node_in_turn("conv-1", 0)
        assert last_node is not None
        assert last_node.text == "Agent message"


# ---------------------------------------------------------------------------
# TestRetrievalStrategy
# ---------------------------------------------------------------------------

class TestRetrievalStrategy:
    """Tests for retrieval strategy and context retrieval."""

    async def test_get_context_returns_complete_chain(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify get_context returns complete chain."""
        test_file = tmp_path / "context_test.py"
        test_file.write_text("def test(): pass")

        messages = [
            Message(role="user", content="What does context_test.py do?"),
            Message(
                role="assistant",
                content="Let me read it for you.",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content="def test(): pass", tool_call_id="c1"),
        ]
        result = await tracemem.import_trace("conv-1", messages)

        context = await retrieval.get_context(result["user_text"])

        assert context.user_text.text == "What does context_test.py do?"
        assert context.agent_text.text == "Let me read it for you."
        assert len(context.tool_uses) == 1

        tool_use = context.tool_uses[0]
        assert tool_use.resource_version is not None
        assert tool_use.resource_version.content_hash == hashlib.sha256(
            "def test(): pass".encode()
        ).hexdigest()

    async def test_search_finds_relevant_messages(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify search returns relevant results."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="How do I fix the authentication bug?")
        )
        await tracemem.add_message(
            "conv-2", Message(role="user", content="What's the weather today?")
        )
        await tracemem.add_message(
            "conv-3", Message(role="user", content="Debug the login issue")
        )

        results = await retrieval.search(
            "authentication problem",
            config=RetrievalConfig(limit=5),
        )

        assert len(results) >= 1

    async def test_search_excludes_conversation(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify search excludes specified conversation."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="Test message one")
        )
        await tracemem.add_message(
            "conv-2", Message(role="user", content="Test message two")
        )

        results = await retrieval.search(
            "Test message",
            config=RetrievalConfig(exclude_conversation_id="conv-1", limit=10),
        )

        assert all(r.conversation_id != "conv-1" for r in results)

    async def test_search_with_context(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify search can include full context."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="Help with code")
        )
        await tracemem.add_message(
            "conv-1", Message(role="assistant", content="Sure, I can help!")
        )

        results = await retrieval.search(
            "code help",
            config=RetrievalConfig(limit=5, include_context=True),
        )

        assert len(results) >= 1
        result = results[0]
        assert result.context is not None
        assert result.context.user_text is not None


# ---------------------------------------------------------------------------
# TestRetrievalConfig
# ---------------------------------------------------------------------------

class TestRetrievalConfig:
    """Tests for RetrievalConfig-based retrieval."""

    async def test_search_with_config(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify search works with RetrievalConfig."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="How do I authenticate users?")
        )
        await tracemem.add_message(
            "conv-1", Message(role="assistant", content="Use JWT tokens.")
        )

        config = RetrievalConfig(limit=5, include_context=True)
        results = await retrieval.search("authentication", config=config)

        assert len(results) >= 1
        assert results[0].context is not None

    async def test_search_with_config_excludes_conversation(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify config.exclude_conversation_id works."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="Message in conv 1")
        )
        await tracemem.add_message(
            "conv-2", Message(role="user", content="Message in conv 2")
        )

        config = RetrievalConfig(exclude_conversation_id="conv-1")
        results = await retrieval.search("Message", config=config)

        assert all(r.conversation_id != "conv-1" for r in results)

    async def test_search_with_include_context(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify search with include_context=True populates context."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="Code review request")
        )
        await tracemem.add_message(
            "conv-1", Message(role="assistant", content="I'll review it.")
        )

        config = RetrievalConfig(include_context=True)
        results = await retrieval.search("review", config=config)

        assert len(results) >= 1
        assert results[0].context is not None

    async def test_get_conversations_for_resource_with_limit(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify limit parameter works for resource queries."""
        shared_file = tmp_path / "limited.py"
        content = "# limited test"

        for i in range(5):
            messages = [
                Message(role="user", content=f"Read limited.py - request {i}"),
                Message(
                    role="assistant",
                    content="Reading...",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{i}",
                            name="read_file",
                            args={"path": str(shared_file)},
                        )
                    ],
                ),
                Message(role="tool", content=content, tool_call_id=f"call_{i}"),
            ]
            await tracemem.import_trace(f"conv-{i}", messages)

        canonical_uri = _canonicalize_file_uri(f"file://{shared_file}", root=None)

        config = RetrievalConfig(limit=3)
        conversations = await retrieval.get_conversations_for_resource(
            canonical_uri, config=config
        )

        assert len(conversations) == 3

    async def test_get_conversations_for_resource_sort_order(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify sort_order parameter affects result order."""
        shared_file = tmp_path / "sorted.py"
        content = "# sorted test"

        for i in range(3):
            messages = [
                Message(role="user", content=f"Read sorted.py - request {i}"),
                Message(
                    role="assistant",
                    content="Reading...",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{i}",
                            name="read_file",
                            args={"path": str(shared_file)},
                        )
                    ],
                ),
                Message(role="tool", content=content, tool_call_id=f"call_{i}"),
            ]
            await tracemem.import_trace(f"conv-{i}", messages)

        canonical_uri = _canonicalize_file_uri(f"file://{shared_file}", root=None)

        config_desc = RetrievalConfig(sort_order="desc")
        conversations_desc = await retrieval.get_conversations_for_resource(
            canonical_uri, config=config_desc
        )

        config_asc = RetrievalConfig(sort_order="asc")
        conversations_asc = await retrieval.get_conversations_for_resource(
            canonical_uri, config=config_asc
        )

        desc_ids = [c.conversation_id for c in conversations_desc]
        asc_ids = [c.conversation_id for c in conversations_asc]
        assert desc_ids == list(reversed(asc_ids))

    async def test_conversation_reference_includes_agent_text(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify ConversationReference includes agent_text field."""
        shared_file = tmp_path / "agent_response.py"

        messages = [
            Message(role="user", content="Read the agent_response file"),
            Message(
                role="assistant",
                content="Here is the file content analysis.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        args={"path": str(shared_file)},
                    )
                ],
            ),
            Message(role="tool", content="# agent response test", tool_call_id="call_1"),
        ]
        await tracemem.import_trace("conv-agent", messages)

        canonical_uri = _canonicalize_file_uri(f"file://{shared_file}", root=None)
        conversations = await retrieval.get_conversations_for_resource(canonical_uri)

        assert len(conversations) == 1
        assert conversations[0].agent_text == "Here is the file content analysis."

    async def test_get_conversations_for_resource_with_config(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
        tmp_path,
    ):
        """Verify get_conversations_for_resource works with config."""
        shared_file = tmp_path / "history.py"

        messages = [
            Message(role="user", content="Read history file"),
            Message(
                role="assistant",
                content="Reading history...",
                tool_calls=[
                    ToolCall(
                        id="call_h",
                        name="read_file",
                        args={"path": str(shared_file)},
                    )
                ],
            ),
            Message(role="tool", content="# history test", tool_call_id="call_h"),
        ]
        await tracemem.import_trace("conv-history", messages)

        canonical_uri = _canonicalize_file_uri(f"file://{shared_file}", root=None)

        config = RetrievalConfig(limit=5)
        conversations = await retrieval.get_conversations_for_resource(
            canonical_uri, config=config
        )

        assert len(conversations) == 1
        assert conversations[0].conversation_id == "conv-history"

    async def test_search_with_exclude_conversation_in_config(
        self,
        tracemem: TraceMem,
        retrieval: HybridRetrievalStrategy,
    ):
        """Verify exclude_conversation_id in config filters results."""
        await tracemem.add_message(
            "conv-1", Message(role="user", content="Config exclude test message 1")
        )
        await tracemem.add_message(
            "conv-2", Message(role="user", content="Config exclude test message 2")
        )

        config = RetrievalConfig(limit=10, exclude_conversation_id="conv-1")
        results = await retrieval.search(
            "Config exclude test",
            config=config,
        )

        assert all(r.conversation_id != "conv-1" for r in results)
