"""Integration tests for Neo4jGraphStore.

These tests require a running Neo4j instance. Start it with:
    docker compose up -d neo4j

Run tests with:
    uv run pytest tests/storage/graph/ -v
"""

import json
from uuid import uuid4

import pytest

from tracemem_core.models.edges import Relationship, VersionOf
from tracemem_core.models.nodes import AgentText, Resource, ResourceVersion, UserText
from tracemem_core.storage.graph.neo import Neo4jGraphStore


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
async def graph_store(neo4j_config):
    """Connected Neo4jGraphStore with clean database."""
    store = Neo4jGraphStore(**neo4j_config)
    await store.connect()
    await store.initialize_schema()
    yield store
    # Cleanup: delete all nodes and relationships
    async with store._driver.session(database=neo4j_config["database"]) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    await store.close()


# =============================================================================
# Lifecycle Tests (3 tests)
# =============================================================================


class TestNeo4jGraphStoreLifecycle:
    """Tests for connection lifecycle management."""

    async def test_connect_and_close(self, neo4j_config):
        """Test connecting and closing the store."""
        store = Neo4jGraphStore(**neo4j_config)

        # Not connected initially
        assert store._driver is None

        # Connect
        await store.connect()
        assert store._driver is not None

        # Close
        await store.close()
        assert store._driver is None

    async def test_operations_raise_when_not_connected(self):
        """Test that operations raise RuntimeError when not connected."""
        store = Neo4jGraphStore()

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

    async def test_initialize_schema_creates_constraints(self, graph_store):
        """Test that initialize_schema creates expected constraints and indexes."""
        async with graph_store._driver.session(database="neo4j") as session:
            # Check constraints
            result = await session.run("SHOW CONSTRAINTS")
            constraints = await result.data()
            constraint_names = [c["name"] for c in constraints]
            assert "resource_uri" in constraint_names

            # Check indexes
            result = await session.run("SHOW INDEXES")
            indexes = await result.data()
            index_names = [i["name"] for i in indexes]
            assert "resource_version_hash" in index_names
            assert "user_text_conversation" in index_names
            assert "user_text_id" in index_names
            assert "agent_text_id" in index_names
            assert "resource_version_id" in index_names
            assert "resource_id" in index_names


# =============================================================================
# Node Creation Tests (8 tests)
# =============================================================================


class TestNeo4jGraphStoreCreateNode:
    """Tests for create_node polymorphic method."""

    # Type dispatch tests

    async def test_create_node_dispatches_user_text(self, graph_store):
        """Test that create_node correctly handles UserText."""
        node = UserText(text="Hello, world!", conversation_id="conv1")

        result = await graph_store.create_node(node)

        assert isinstance(result, UserText)
        assert result.id == node.id
        assert result.text == node.text

    async def test_create_node_dispatches_agent_text(self, graph_store):
        """Test that create_node correctly handles AgentText."""
        node = AgentText(text="Hello back!", conversation_id="conv1")

        result = await graph_store.create_node(node)

        assert isinstance(result, AgentText)
        assert result.id == node.id
        assert result.text == node.text

    async def test_create_node_dispatches_resource_version(self, graph_store):
        """Test that create_node correctly handles ResourceVersion."""
        node = ResourceVersion(
            content_hash="abc123",
            uri="file:///test.py",
            conversation_id="conv1",
        )

        result = await graph_store.create_node(node)

        assert isinstance(result, ResourceVersion)
        assert result.id == node.id
        assert result.content_hash == node.content_hash

    async def test_create_node_dispatches_resource(self, graph_store):
        """Test that create_node correctly handles Resource."""
        node = Resource(
            uri="file:///test.py",
            current_content_hash="abc123",
            conversation_id="conv1",
        )

        result = await graph_store.create_node(node)

        assert isinstance(result, Resource)
        assert result.uri == node.uri

    # Field persistence tests

    async def test_create_node_user_text_persists_all_fields(self, graph_store):
        """Test that UserText fields are correctly persisted in Neo4j."""
        node = UserText(text="Test message", conversation_id="conv123")

        await graph_store.create_node(node)

        # Retrieve and verify
        async with graph_store._driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH (n:UserText {id: $id}) RETURN n",
                id=str(node.id),
            )
            record = await result.single()
            assert record is not None
            n = record["n"]
            assert n["text"] == "Test message"
            assert n["conversation_id"] == "conv123"
            assert n["created_at"] is not None
            assert n["last_accessed_at"] is not None

    async def test_create_node_agent_text_persists_all_fields(self, graph_store):
        """Test that AgentText fields are correctly persisted in Neo4j."""
        node = AgentText(text="Agent response", conversation_id="conv456")

        await graph_store.create_node(node)

        # Retrieve and verify
        async with graph_store._driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH (n:AgentText {id: $id}) RETURN n",
                id=str(node.id),
            )
            record = await result.single()
            assert record is not None
            n = record["n"]
            assert n["text"] == "Agent response"
            assert n["conversation_id"] == "conv456"

    async def test_create_node_resource_version_persists_all_fields(self, graph_store):
        """Test that ResourceVersion fields are correctly persisted in Neo4j."""
        node = ResourceVersion(
            content_hash="sha256:abc123",
            uri="file:///path/to/file.py",
            conversation_id="conv789",
        )

        await graph_store.create_node(node)

        # Retrieve and verify
        async with graph_store._driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH (n:ResourceVersion {id: $id}) RETURN n",
                id=str(node.id),
            )
            record = await result.single()
            assert record is not None
            n = record["n"]
            assert n["content_hash"] == "sha256:abc123"
            assert n["uri"] == "file:///path/to/file.py"
            assert n["conversation_id"] == "conv789"

    # Resource MERGE behavior test

    async def test_create_node_resource_returns_existing_on_duplicate_uri(self, graph_store):
        """Test that creating a Resource with existing URI returns the existing one."""
        resource1 = Resource(
            uri="file:///shared.py",
            current_content_hash="hash1",
            conversation_id="conv1",
        )
        resource2 = Resource(
            uri="file:///shared.py",
            current_content_hash="hash2",
            conversation_id="conv2",
        )

        result1 = await graph_store.create_node(resource1)
        result2 = await graph_store.create_node(resource2)

        # Should return the same resource (MERGE behavior)
        assert result1.id == result2.id
        assert result1.uri == result2.uri
        # Original hash should be preserved (ON CREATE SET)
        assert result2.current_content_hash == "hash1"


# =============================================================================
# Edge Creation Tests (5 tests)
# =============================================================================


class TestNeo4jGraphStoreCreateEdge:
    """Tests for create_edge polymorphic method."""

    # Type dispatch tests

    async def test_create_edge_dispatches_relationship(self, graph_store):
        """Test that create_edge correctly handles Relationship."""
        # Create source and target nodes first
        user = UserText(text="User message", conversation_id="conv1")
        agent = AgentText(text="Agent response", conversation_id="conv1")
        await graph_store.create_node(user)
        await graph_store.create_node(agent)

        edge = Relationship(
            source_id=user.id,
            target_id=agent.id,
            relationship_type="MESSAGE",
            conversation_id="conv1",
        )

        result = await graph_store.create_edge(edge)

        assert isinstance(result, Relationship)
        assert result.id == edge.id

    async def test_create_edge_dispatches_version_of(self, graph_store):
        """Test that create_edge correctly handles VersionOf."""
        # Create ResourceVersion and Resource first
        version = ResourceVersion(
            content_hash="abc123",
            uri="file:///test.py",
            conversation_id="conv1",
        )
        resource = Resource(
            uri="file:///test.py",
            current_content_hash="abc123",
            conversation_id="conv1",
        )
        await graph_store.create_node(version)
        await graph_store.create_node(resource)

        edge = VersionOf(version_id=version.id, resource_id=resource.id)

        result = await graph_store.create_edge(edge)

        assert isinstance(result, VersionOf)
        assert result.id == edge.id

    # Relationship variations tests

    async def test_create_edge_relationship_with_properties(self, graph_store):
        """Test that Relationship properties are correctly persisted."""
        user = UserText(text="Edit file", conversation_id="conv1")
        agent = AgentText(text="Done", conversation_id="conv1")
        await graph_store.create_node(user)
        await graph_store.create_node(agent)

        edge = Relationship(
            source_id=user.id,
            target_id=agent.id,
            relationship_type="EDIT",
            conversation_id="conv1",
            properties={"line_start": 10, "line_end": 20},
        )

        await graph_store.create_edge(edge)

        # Verify properties are persisted (as JSON string)
        async with graph_store._driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH ()-[r:EDIT {id: $id}]->() RETURN r",
                id=str(edge.id),
            )
            record = await result.single()
            assert record is not None
            r = record["r"]
            properties = json.loads(r["properties"])
            assert properties["line_start"] == 10
            assert properties["line_end"] == 20

    async def test_create_edge_relationship_different_types(self, graph_store):
        """Test creating relationships with different types."""
        agent = AgentText(text="Agent response", conversation_id="conv1")
        version = ResourceVersion(
            content_hash="abc123",
            uri="file:///test.py",
            conversation_id="conv1",
        )
        await graph_store.create_node(agent)
        await graph_store.create_node(version)

        # Create READ relationship
        read_edge = Relationship(
            source_id=agent.id,
            target_id=version.id,
            relationship_type="READ",
            conversation_id="conv1",
        )
        await graph_store.create_edge(read_edge)

        # Create WRITE relationship
        write_edge = Relationship(
            source_id=agent.id,
            target_id=version.id,
            relationship_type="WRITE",
            conversation_id="conv1",
        )
        await graph_store.create_edge(write_edge)

        # Verify both exist with correct types
        async with graph_store._driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH (a:AgentText)-[r]->(v:ResourceVersion) RETURN type(r) as rel_type",
            )
            records = await result.data()
            rel_types = {r["rel_type"] for r in records}
            assert "READ" in rel_types
            assert "WRITE" in rel_types

    # VersionOf linking test

    async def test_create_edge_version_of_links_version_to_resource(self, graph_store):
        """Test that VersionOf correctly links ResourceVersion to Resource."""
        version = ResourceVersion(
            content_hash="def456",
            uri="file:///linked.py",
            conversation_id="conv1",
        )
        resource = Resource(
            uri="file:///linked.py",
            current_content_hash="def456",
            conversation_id="conv1",
        )
        await graph_store.create_node(version)
        await graph_store.create_node(resource)

        edge = VersionOf(version_id=version.id, resource_id=resource.id)
        await graph_store.create_edge(edge)

        # Verify the relationship exists and connects correctly
        async with graph_store._driver.session(database="neo4j") as session:
            result = await session.run(
                """
                MATCH (v:ResourceVersion {id: $version_id})-[r:VERSION_OF]->(res:Resource {id: $resource_id})
                RETURN r, v.uri as version_uri, res.uri as resource_uri
                """,
                version_id=str(version.id),
                resource_id=str(resource.id),
            )
            record = await result.single()
            assert record is not None
            assert record["version_uri"] == "file:///linked.py"
            assert record["resource_uri"] == "file:///linked.py"


# =============================================================================
# Raw Cypher Execution Tests (2 tests)
# =============================================================================


class TestNeo4jGraphStoreExecuteCypher:
    """Tests for execute_cypher raw query method."""

    async def test_execute_cypher_returns_results(self, graph_store):
        """Test that execute_cypher returns query results."""
        # Create a node first
        user = UserText(text="Test message", conversation_id="conv1")
        await graph_store.create_node(user)

        # Query it with raw Cypher
        results = await graph_store.execute_cypher(
            "MATCH (n:UserText {id: $id}) RETURN n.text as text, n.conversation_id as conv",
            {"id": str(user.id)},
        )

        assert len(results) == 1
        assert results[0]["text"] == "Test message"
        assert results[0]["conv"] == "conv1"

    async def test_execute_cypher_raises_when_not_connected(self):
        """Test that execute_cypher raises RuntimeError when not connected."""
        store = Neo4jGraphStore()

        with pytest.raises(RuntimeError, match="Not connected"):
            await store.execute_cypher("MATCH (n) RETURN n")


# =============================================================================
# Namespace Tests (5 tests)
# =============================================================================


class TestNeo4jGraphStoreNamespace:
    """Tests for namespace-based multi-user isolation."""

    @pytest.fixture
    async def ns_graph_store(self, neo4j_config):
        """Connected Neo4jGraphStore with namespace set."""
        store = Neo4jGraphStore(**neo4j_config, namespace="team-alpha")
        await store.connect()
        await store.initialize_schema()
        yield store
        async with store._driver.session(database=neo4j_config["database"]) as session:
            await session.run("MATCH (n) DETACH DELETE n")
        await store.close()

    async def test_namespace_stored_on_user_text(self, ns_graph_store):
        """UserText nodes should have namespace property set."""
        node = UserText(text="Hello", conversation_id="conv1")
        await ns_graph_store.create_node(node)

        results = await ns_graph_store.execute_cypher(
            "MATCH (n:UserText {id: $id}) RETURN n.namespace as ns",
            {"id": str(node.id)},
        )
        assert results[0]["ns"] == "team-alpha"

    async def test_namespace_stored_on_agent_text(self, ns_graph_store):
        """AgentText nodes should have namespace property set."""
        node = AgentText(text="Response", conversation_id="conv1")
        await ns_graph_store.create_node(node)

        results = await ns_graph_store.execute_cypher(
            "MATCH (n:AgentText {id: $id}) RETURN n.namespace as ns",
            {"id": str(node.id)},
        )
        assert results[0]["ns"] == "team-alpha"

    async def test_namespace_stored_on_resource_version(self, ns_graph_store):
        """ResourceVersion nodes should have namespace property set."""
        node = ResourceVersion(
            content_hash="abc123", uri="file:///test.py", conversation_id="conv1"
        )
        await ns_graph_store.create_node(node)

        results = await ns_graph_store.execute_cypher(
            "MATCH (n:ResourceVersion {id: $id}) RETURN n.namespace as ns",
            {"id": str(node.id)},
        )
        assert results[0]["ns"] == "team-alpha"

    async def test_namespace_stored_on_resource(self, ns_graph_store):
        """Resource nodes should have namespace property set on creation."""
        node = Resource(
            uri="file:///test.py",
            current_content_hash="abc123",
            conversation_id="conv1",
        )
        await ns_graph_store.create_node(node)

        results = await ns_graph_store.execute_cypher(
            "MATCH (n:Resource {uri: $uri}) RETURN n.namespace as ns",
            {"uri": "file:///test.py"},
        )
        assert results[0]["ns"] == "team-alpha"

    async def test_namespace_filters_queries(self, neo4j_config):
        """Queries with namespace should only see nodes in that namespace."""
        # Create two stores with different namespaces
        store_a = Neo4jGraphStore(**neo4j_config, namespace="team-a")
        store_b = Neo4jGraphStore(**neo4j_config, namespace="team-b")
        await store_a.connect()
        await store_b.connect()
        await store_a.initialize_schema()

        try:
            # Create a node in each namespace with the same conversation_id
            user_a = UserText(text="From team A", conversation_id="shared-conv")
            user_b = UserText(text="From team B", conversation_id="shared-conv")
            await store_a.create_node(user_a)
            await store_b.create_node(user_b)

            # Each store should only see its own namespace
            result_a = await store_a.get_last_user_text("shared-conv")
            result_b = await store_b.get_last_user_text("shared-conv")

            assert result_a is not None
            assert result_a.text == "From team A"
            assert result_b is not None
            assert result_b.text == "From team B"
        finally:
            async with store_a._driver.session(
                database=neo4j_config["database"]
            ) as session:
                await session.run("MATCH (n) DETACH DELETE n")
            await store_a.close()
            await store_b.close()
