"""Unit tests for KuzuGraphStore.

These tests use a real embedded Kùzu database in tmp_path — no Docker needed.

Run with:
    uv run pytest tests/storage/graph/test_kuzu.py -v
"""

from uuid import uuid4

import pytest

from tracemem_core.models.edges import Relationship, VersionOf
from tracemem_core.models.nodes import AgentText, Resource, ResourceVersion, UserText
from tracemem_core.storage.graph.kuzu_store import KuzuGraphStore


@pytest.fixture
async def graph_store(tmp_path):
    """Connected KuzuGraphStore with fresh database."""
    store = KuzuGraphStore(db_path=tmp_path / "graph")
    await store.connect()
    await store.initialize_schema()
    yield store
    await store.close()


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestKuzuGraphStoreLifecycle:
    """Tests for connection lifecycle management."""

    async def test_connect_and_close(self, tmp_path):
        """Test connecting and closing the store."""
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        assert store._conn is None

        await store.connect()
        assert store._conn is not None

        await store.close()
        assert store._conn is None

    async def test_operations_raise_when_not_connected(self, tmp_path):
        """Test that operations raise RuntimeError when not connected."""
        store = KuzuGraphStore(db_path=tmp_path / "graph")

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.initialize_schema()

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.create_node(UserText(text="test", conversation_id="c1"))

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.create_edge(
                Relationship(
                    source_id=uuid4(),
                    target_id=uuid4(),
                    relationship_type="TEST",
                    conversation_id="c1",
                )
            )

    async def test_initialize_schema_is_idempotent(self, tmp_path):
        """Test that schema can be initialized multiple times."""
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        await store.connect()
        await store.initialize_schema()
        await store.initialize_schema()  # Should not raise
        await store.close()


# =============================================================================
# Node Creation Tests
# =============================================================================


class TestKuzuGraphStoreCreateNode:
    """Tests for create_node polymorphic method."""

    async def test_create_user_text(self, graph_store):
        """Test creating a UserText node."""
        node = UserText(text="Hello, world!", conversation_id="conv1")
        result = await graph_store.create_node(node)

        assert isinstance(result, UserText)
        assert result.id == node.id
        assert result.text == node.text

    async def test_create_agent_text(self, graph_store):
        """Test creating an AgentText node."""
        node = AgentText(text="Hello back!", conversation_id="conv1")
        result = await graph_store.create_node(node)

        assert isinstance(result, AgentText)
        assert result.id == node.id

    async def test_create_resource_version(self, graph_store):
        """Test creating a ResourceVersion node."""
        node = ResourceVersion(
            content_hash="abc123", uri="file:///test.py", conversation_id="conv1"
        )
        result = await graph_store.create_node(node)

        assert isinstance(result, ResourceVersion)
        assert result.content_hash == "abc123"

    async def test_create_resource(self, graph_store):
        """Test creating a Resource node."""
        node = Resource(
            uri="file:///test.py", current_content_hash="abc123", conversation_id="conv1"
        )
        result = await graph_store.create_node(node)

        assert isinstance(result, Resource)
        assert result.uri == "file:///test.py"

    async def test_create_resource_returns_existing_on_duplicate_uri(self, graph_store):
        """Test MERGE behavior for Resource with existing URI."""
        res1 = Resource(
            uri="file:///shared.py", current_content_hash="hash1", conversation_id="conv1"
        )
        res2 = Resource(
            uri="file:///shared.py", current_content_hash="hash2", conversation_id="conv2"
        )

        result1 = await graph_store.create_node(res1)
        result2 = await graph_store.create_node(res2)

        assert result1.id == result2.id
        assert result2.current_content_hash == "hash1"

    async def test_user_text_persists_all_fields(self, graph_store):
        """Test that UserText fields are correctly persisted."""
        node = UserText(text="Test message", conversation_id="conv123", turn_index=5)
        await graph_store.create_node(node)

        retrieved = await graph_store.get_user_text(node.id)
        assert retrieved is not None
        assert retrieved.text == "Test message"
        assert retrieved.conversation_id == "conv123"
        assert retrieved.turn_index == 5

    async def test_create_node_raises_for_unknown_type(self, graph_store):
        """Test that unknown node types raise TypeError."""
        with pytest.raises(TypeError, match="Unknown node type"):
            await graph_store.create_node("not a node")  # type: ignore


# =============================================================================
# Edge Creation Tests
# =============================================================================


class TestKuzuGraphStoreCreateEdge:
    """Tests for create_edge polymorphic method."""

    async def test_create_message_edge(self, graph_store):
        """Test creating a MESSAGE relationship."""
        user = UserText(text="User msg", conversation_id="conv1")
        agent = AgentText(text="Agent msg", conversation_id="conv1")
        await graph_store.create_node(user)
        await graph_store.create_node(agent)

        edge = Relationship(
            source_id=user.id, target_id=agent.id, conversation_id="conv1"
        )
        result = await graph_store.create_edge(edge)

        assert isinstance(result, Relationship)
        assert result.id == edge.id

    async def test_create_tool_use_edge(self, graph_store):
        """Test creating a TOOL_USE edge (non-MESSAGE relationship)."""
        agent = AgentText(text="Reading file", conversation_id="conv1")
        version = ResourceVersion(
            content_hash="abc", uri="file:///test.py", conversation_id="conv1"
        )
        await graph_store.create_node(agent)
        await graph_store.create_node(version)

        edge = Relationship(
            source_id=agent.id,
            target_id=version.id,
            relationship_type="READ",
            conversation_id="conv1",
            properties={"path": "test.py"},
        )
        result = await graph_store.create_edge(edge)

        assert isinstance(result, Relationship)

    async def test_create_version_of_edge(self, graph_store):
        """Test creating a VERSION_OF relationship."""
        version = ResourceVersion(
            content_hash="abc", uri="file:///test.py", conversation_id="conv1"
        )
        resource = Resource(
            uri="file:///test.py", current_content_hash="abc", conversation_id="conv1"
        )
        await graph_store.create_node(version)
        await graph_store.create_node(resource)

        edge = VersionOf(version_id=version.id, resource_id=resource.id)
        result = await graph_store.create_edge(edge)

        assert isinstance(result, VersionOf)

    async def test_create_agent_to_agent_message_edge(self, graph_store):
        """Test MESSAGE edge between two AgentText nodes."""
        a1 = AgentText(text="First", conversation_id="conv1")
        a2 = AgentText(text="Second", conversation_id="conv1")
        await graph_store.create_node(a1)
        await graph_store.create_node(a2)

        edge = Relationship(
            source_id=a1.id, target_id=a2.id, conversation_id="conv1"
        )
        await graph_store.create_edge(edge)

    async def test_create_edge_raises_for_unknown_type(self, graph_store):
        """Test that unknown edge types raise TypeError."""
        with pytest.raises(TypeError, match="Unknown edge type"):
            await graph_store.create_edge("not an edge")  # type: ignore


# =============================================================================
# Retrieval Tests
# =============================================================================


class TestKuzuGraphStoreRetrieval:
    """Tests for retrieval operations."""

    async def test_get_user_text(self, graph_store):
        """Test retrieving a UserText by ID."""
        node = UserText(text="Hello", conversation_id="c1")
        await graph_store.create_node(node)

        result = await graph_store.get_user_text(node.id)
        assert result is not None
        assert result.text == "Hello"

    async def test_get_user_text_not_found(self, graph_store):
        """Test retrieving a non-existent UserText."""
        result = await graph_store.get_user_text(uuid4())
        assert result is None

    async def test_get_last_user_text(self, graph_store):
        """Test getting the most recent UserText."""
        u1 = UserText(text="First", conversation_id="c1")
        u2 = UserText(text="Second", conversation_id="c1")
        await graph_store.create_node(u1)
        await graph_store.create_node(u2)

        result = await graph_store.get_last_user_text("c1")
        assert result is not None
        assert result.text == "Second"

    async def test_get_last_agent_text(self, graph_store):
        """Test getting the most recent AgentText."""
        a1 = AgentText(text="First", conversation_id="c1")
        a2 = AgentText(text="Second", conversation_id="c1")
        await graph_store.create_node(a1)
        await graph_store.create_node(a2)

        result = await graph_store.get_last_agent_text("c1")
        assert result is not None
        assert result.text == "Second"

    async def test_get_last_message_node(self, graph_store):
        """Test getting the most recent message node of either type."""
        u = UserText(text="User", conversation_id="c1")
        a = AgentText(text="Agent", conversation_id="c1")
        await graph_store.create_node(u)
        await graph_store.create_node(a)

        result = await graph_store.get_last_message_node("c1")
        assert result is not None
        assert isinstance(result, AgentText)
        assert result.text == "Agent"

    async def test_get_max_turn_index(self, graph_store):
        """Test getting max turn index."""
        u = UserText(text="User", conversation_id="c1", turn_index=3)
        a = AgentText(text="Agent", conversation_id="c1", turn_index=3)
        await graph_store.create_node(u)
        await graph_store.create_node(a)

        result = await graph_store.get_max_turn_index("c1")
        assert result == 3

    async def test_get_max_turn_index_empty(self, graph_store):
        """Test max turn index returns -1 for empty conversation."""
        result = await graph_store.get_max_turn_index("nonexistent")
        assert result == -1

    async def test_get_last_node_in_turn(self, graph_store):
        """Test getting the last node in a specific turn."""
        u = UserText(text="User", conversation_id="c1", turn_index=0)
        a = AgentText(text="Agent", conversation_id="c1", turn_index=0)
        await graph_store.create_node(u)
        await graph_store.create_node(a)

        result = await graph_store.get_last_node_in_turn("c1", 0)
        assert result is not None
        assert isinstance(result, AgentText)

    async def test_get_resource_by_uri(self, graph_store):
        """Test getting a Resource by URI."""
        res = Resource(
            uri="file://test.py", current_content_hash="hash1", conversation_id="c1"
        )
        await graph_store.create_node(res)

        result = await graph_store.get_resource_by_uri("file://test.py")
        assert result is not None
        assert result.uri == "file://test.py"

    async def test_get_resource_by_uri_not_found(self, graph_store):
        """Test that missing resource returns None."""
        result = await graph_store.get_resource_by_uri("file://nonexistent.py")
        assert result is None

    async def test_get_resource_version_by_hash(self, graph_store):
        """Test getting a ResourceVersion by hash."""
        rv = ResourceVersion(
            content_hash="hash1", uri="file://test.py", conversation_id="c1"
        )
        await graph_store.create_node(rv)

        result = await graph_store.get_resource_version_by_hash("file://test.py", "hash1")
        assert result is not None
        assert result.content_hash == "hash1"

    async def test_update_resource_hash(self, graph_store):
        """Test updating a resource's content hash."""
        res = Resource(
            uri="file://test.py", current_content_hash="hash1", conversation_id="c1"
        )
        await graph_store.create_node(res)

        await graph_store.update_resource_hash("file://test.py", "hash2")

        result = await graph_store.get_resource_by_uri("file://test.py")
        assert result is not None
        assert result.current_content_hash == "hash2"

    async def test_update_last_accessed(self, graph_store):
        """Test updating last_accessed timestamps."""
        u = UserText(text="User", conversation_id="c1")
        await graph_store.create_node(u)

        # Should not raise
        await graph_store.update_last_accessed([u.id])

    async def test_update_last_accessed_empty_list(self, graph_store):
        """Test update_last_accessed with empty list does nothing."""
        await graph_store.update_last_accessed([])


# =============================================================================
# Context & Trajectory Tests
# =============================================================================


class TestKuzuGraphStoreContext:
    """Tests for context and trajectory queries."""

    async def test_get_node_context_basic(self, graph_store):
        """Test basic context retrieval."""
        u = UserText(text="Read my file", conversation_id="c1")
        a = AgentText(text="Here it is", conversation_id="c1")
        await graph_store.create_node(u)
        await graph_store.create_node(a)
        await graph_store.create_edge(
            Relationship(source_id=u.id, target_id=a.id, conversation_id="c1")
        )

        ctx = await graph_store.get_node_context(u.id)
        assert ctx.user_text is not None
        assert ctx.user_text.text == "Read my file"
        assert ctx.agent_text is not None
        assert ctx.agent_text.text == "Here it is"

    async def test_get_node_context_with_tool_uses(self, graph_store):
        """Test context retrieval with tool uses."""
        u = UserText(text="Read file", conversation_id="c1")
        a = AgentText(text="Reading", conversation_id="c1")
        rv = ResourceVersion(
            content_hash="h1", uri="file://test.py", conversation_id="c1"
        )
        res = Resource(
            uri="file://test.py", current_content_hash="h1", conversation_id="c1"
        )
        await graph_store.create_node(u)
        await graph_store.create_node(a)
        await graph_store.create_node(rv)
        await graph_store.create_node(res)

        await graph_store.create_edge(
            Relationship(source_id=u.id, target_id=a.id, conversation_id="c1")
        )
        await graph_store.create_edge(VersionOf(version_id=rv.id, resource_id=res.id))
        await graph_store.create_edge(
            Relationship(
                source_id=a.id,
                target_id=rv.id,
                relationship_type="READ",
                conversation_id="c1",
            )
        )

        ctx = await graph_store.get_node_context(u.id)
        assert len(ctx.tool_uses) == 1
        assert ctx.tool_uses[0].tool_name == "READ"
        assert ctx.tool_uses[0].resource_version is not None
        assert ctx.tool_uses[0].resource is not None

    async def test_get_node_context_not_found(self, graph_store):
        """Test context for non-existent node."""
        ctx = await graph_store.get_node_context(uuid4())
        assert ctx.user_text is None
        assert ctx.agent_text is None

    async def test_get_trajectory_nodes(self, graph_store):
        """Test trajectory traversal."""
        u = UserText(text="User", conversation_id="c1", turn_index=0)
        a1 = AgentText(text="Agent1", conversation_id="c1", turn_index=0)
        a2 = AgentText(text="Agent2", conversation_id="c1", turn_index=0)
        await graph_store.create_node(u)
        await graph_store.create_node(a1)
        await graph_store.create_node(a2)

        await graph_store.create_edge(
            Relationship(source_id=u.id, target_id=a1.id, conversation_id="c1")
        )
        await graph_store.create_edge(
            Relationship(source_id=a1.id, target_id=a2.id, conversation_id="c1")
        )

        nodes = await graph_store.get_trajectory_nodes(u.id)
        assert len(nodes) == 3

        labels = [n["node_labels"][0] for n in nodes]
        assert "UserText" in labels
        assert "AgentText" in labels

    async def test_get_resource_conversations(self, graph_store):
        """Test finding conversations that accessed a resource."""
        u = UserText(text="Read file", conversation_id="c1")
        a = AgentText(text="Reading", conversation_id="c1")
        rv = ResourceVersion(
            content_hash="h1", uri="file://test.py", conversation_id="c1"
        )
        res = Resource(
            uri="file://test.py", current_content_hash="h1", conversation_id="c1"
        )
        await graph_store.create_node(u)
        await graph_store.create_node(a)
        await graph_store.create_node(rv)
        await graph_store.create_node(res)

        await graph_store.create_edge(
            Relationship(source_id=u.id, target_id=a.id, conversation_id="c1")
        )
        await graph_store.create_edge(VersionOf(version_id=rv.id, resource_id=res.id))
        await graph_store.create_edge(
            Relationship(
                source_id=a.id,
                target_id=rv.id,
                relationship_type="READ",
                conversation_id="c1",
            )
        )

        convs = await graph_store.get_resource_conversations("file://test.py")
        assert len(convs) == 1
        assert convs[0].conversation_id == "c1"
        assert convs[0].user_text == "Read file"

    async def test_get_resource_conversations_with_exclude(self, graph_store):
        """Test excluding a conversation from resource results."""
        u = UserText(text="Read file", conversation_id="c1")
        a = AgentText(text="Reading", conversation_id="c1")
        rv = ResourceVersion(
            content_hash="h1", uri="file://test.py", conversation_id="c1"
        )
        res = Resource(
            uri="file://test.py", current_content_hash="h1", conversation_id="c1"
        )
        await graph_store.create_node(u)
        await graph_store.create_node(a)
        await graph_store.create_node(rv)
        await graph_store.create_node(res)

        await graph_store.create_edge(
            Relationship(source_id=u.id, target_id=a.id, conversation_id="c1")
        )
        await graph_store.create_edge(VersionOf(version_id=rv.id, resource_id=res.id))
        await graph_store.create_edge(
            Relationship(
                source_id=a.id,
                target_id=rv.id,
                relationship_type="READ",
                conversation_id="c1",
            )
        )

        convs = await graph_store.get_resource_conversations(
            "file://test.py", exclude_conversation_id="c1"
        )
        assert len(convs) == 0


# =============================================================================
# Raw Cypher Execution Tests
# =============================================================================


class TestKuzuGraphStoreExecuteCypher:
    """Tests for execute_cypher raw query method."""

    async def test_execute_cypher_returns_results(self, graph_store):
        """Test that execute_cypher returns query results."""
        user = UserText(text="Test", conversation_id="conv1")
        await graph_store.create_node(user)

        results = await graph_store.execute_cypher(
            "MATCH (n:UserText) WHERE n.id = $id RETURN n.text as text",
            {"id": str(user.id)},
        )
        assert len(results) == 1
        assert results[0]["text"] == "Test"

    async def test_execute_cypher_raises_when_not_connected(self, tmp_path):
        """Test that execute_cypher raises RuntimeError when not connected."""
        store = KuzuGraphStore(db_path=tmp_path / "graph")
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.execute_cypher("MATCH (n) RETURN n")
