"""Neo4j-specific integration tests that use raw driver queries.

These tests verify graph structure details that require direct Neo4j
Cypher queries via `_driver.session(database="neo4j")`.

Requires Neo4j to be running:
    docker compose up -d neo4j

Run with:
    uv run pytest tests/integration/test_neo4j.py -v
"""

import hashlib
import json

import pytest

from tracemem_core.messages import Message, ToolCall
from tracemem_core.tracemem import TraceMem

pytestmark = pytest.mark.neo4j


class TestAssistantMessageLinking:
    """Tests for MESSAGE edge between UserText and AgentText."""

    async def test_assistant_message_links_to_user_message(
        self, tracemem_integration: TraceMem
    ):
        """Verify MESSAGE edge connects UserText to AgentText."""
        user_result = await tracemem_integration.add_message(
            "conv-1", Message(role="user", content="What is Python?")
        )
        user_text_id = user_result["user_text"]

        await tracemem_integration.add_message(
            "conv-1",
            Message(role="assistant", content="Python is a programming language."),
        )

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (u:UserText {id: $id})-[r:MESSAGE]->(a:AgentText)
                RETURN u.text as user_text, a.text as agent_text
                """,
                id=str(user_text_id),
            )
            record = await result.single()
            assert record is not None
            assert record["user_text"] == "What is Python?"
            assert record["agent_text"] == "Python is a programming language."


class TestResourceVersionEdges:
    """Tests for VERSION_OF edge verification via raw queries."""

    async def test_version_of_edges_created(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify VERSION_OF edges link ResourceVersion to Resource."""
        test_file = tmp_path / "versioned.py"

        messages = [
            Message(role="user", content="Read versioned.py"),
            Message(
                role="assistant",
                content="Reading...",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(test_file)})
                ],
            ),
            Message(role="tool", content="version 1", tool_call_id="c1"),
        ]
        await tracemem_integration.import_trace("conv-1", messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (v:ResourceVersion)-[r:VERSION_OF]->(res:Resource)
                RETURN v.content_hash as version_hash, res.uri as resource_uri
                """
            )
            records = await result.data()
            assert len(records) == 1
            assert (
                records[0]["version_hash"]
                == hashlib.sha256("version 1".encode()).hexdigest()
            )


class TestSharedResourceGraphStructure:
    """Tests for graph structure when two conversations share a resource."""

    async def test_shared_resource_graph_structure(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify the graph structure when two conversations share a resource."""
        shared_file = tmp_path / "shared.py"
        content = "# shared code"

        for conv_id in ["conv-alice", "conv-bob"]:
            messages = [
                Message(role="user", content=f"Read shared.py ({conv_id})"),
                Message(
                    role="assistant",
                    content="Reading...",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{conv_id}",
                            name="read_file",
                            args={"path": str(shared_file)},
                        )
                    ],
                ),
                Message(role="tool", content=content, tool_call_id=f"call_{conv_id}"),
            ]
            await tracemem_integration.import_trace(conv_id, messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (v:ResourceVersion)
                RETURN count(v) as version_count
                """
            )
            record = await result.single()
            assert record["version_count"] == 1

            result = await session.run(
                """
                MATCH (r:Resource)
                RETURN count(r) as resource_count
                """
            )
            record = await result.single()
            assert record["resource_count"] == 1

            result = await session.run(
                """
                MATCH (a:AgentText)-[r:READ_FILE]->(v:ResourceVersion)
                RETURN count(r) as edge_count, collect(DISTINCT a.conversation_id) as conversations
                """
            )
            record = await result.single()
            assert record["edge_count"] == 2
            assert set(record["conversations"]) == {"conv-alice", "conv-bob"}


class TestConversationChain:
    """Tests for conversation chain continuity (fully connected subgraphs)."""

    async def test_conversation_forms_connected_chain(
        self, tracemem_integration: TraceMem
    ):
        """Verify alternating messages form a fully connected chain."""
        for i in range(3):
            await tracemem_integration.add_message(
                "conv-1", Message(role="user", content=f"User message {i + 1}")
            )
            await tracemem_integration.add_message(
                "conv-1", Message(role="assistant", content=f"Agent response {i + 1}")
            )

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH path = (start)-[:MESSAGE*]->(end)
                WHERE start.conversation_id = 'conv-1'
                  AND NOT ()-[:MESSAGE]->(start)
                RETURN length(path) as chain_length
                ORDER BY chain_length DESC
                LIMIT 1
                """
            )
            record = await result.single()
            assert record is not None
            assert record["chain_length"] == 5

    async def test_new_user_message_links_from_last_agent(
        self, tracemem_integration: TraceMem
    ):
        """Verify a new user message creates edge from last agent node."""
        await tracemem_integration.add_message(
            "conv-1", Message(role="user", content="First user message")
        )
        await tracemem_integration.add_message(
            "conv-1", Message(role="assistant", content="First agent response")
        )

        await tracemem_integration.add_message(
            "conv-1", Message(role="user", content="Second user message")
        )

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (a:AgentText {text: 'First agent response'})
                      -[:MESSAGE]->
                      (u:UserText {text: 'Second user message'})
                RETURN a.text as agent_text, u.text as user_text
                """
            )
            record = await result.single()
            assert record is not None
            assert record["agent_text"] == "First agent response"
            assert record["user_text"] == "Second user message"

    async def test_conversation_chain_uses_timestamp_ordering(
        self, tracemem_integration: TraceMem
    ):
        """Verify chain links use timestamp-based node lookup."""
        await tracemem_integration.add_message(
            "conv-1", Message(role="user", content="User 1")
        )
        await tracemem_integration.add_message(
            "conv-1", Message(role="assistant", content="Agent 1")
        )

        await tracemem_integration.add_message(
            "conv-1", Message(role="user", content="User 2")
        )
        await tracemem_integration.add_message(
            "conv-1", Message(role="assistant", content="Agent 2")
        )

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (u1:UserText {text: 'User 1'})
                      -[:MESSAGE]->(a1:AgentText {text: 'Agent 1'})
                      -[:MESSAGE]->(u2:UserText {text: 'User 2'})
                      -[:MESSAGE]->(a2:AgentText {text: 'Agent 2'})
                RETURN count(*) as chain_exists
                """
            )
            record = await result.single()
            assert record is not None
            assert record["chain_exists"] == 1

    async def test_tool_usage_creates_connected_chain(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify tool usage with multiple assistant messages forms connected chain."""
        test_file = tmp_path / "code.py"
        test_file.write_text("print('hello')")

        messages = [
            Message(role="user", content="Read code.py and explain it"),
            Message(
                role="assistant",
                content="I'll read the file for you.",
                tool_calls=[
                    ToolCall(
                        id="call_1", name="read_file", args={"path": str(test_file)}
                    )
                ],
            ),
            Message(role="tool", content="print('hello')", tool_call_id="call_1"),
            Message(
                role="assistant",
                content="The file contains a simple print statement.",
            ),
        ]

        await tracemem_integration.import_trace("conv-1", messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (u:UserText {text: 'Read code.py and explain it'})
                      -[:MESSAGE]->(a1:AgentText {text: "I'll read the file for you."})
                      -[:MESSAGE]->(a2:AgentText {text: 'The file contains a simple print statement.'})
                RETURN count(*) as chain_exists
                """
            )
            record = await result.single()
            assert record is not None
            assert record["chain_exists"] == 1

    async def test_multiple_tool_calls_with_final_response(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify multiple tool calls followed by final response creates proper chain."""
        file1 = tmp_path / "a.py"
        file2 = tmp_path / "b.py"
        file1.write_text("# file a")
        file2.write_text("# file b")

        messages = [
            Message(role="user", content="Compare a.py and b.py"),
            Message(
                role="assistant",
                content="Let me read both files.",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(file1)}),
                    ToolCall(id="c2", name="read_file", args={"path": str(file2)}),
                ],
            ),
            Message(role="tool", content="# file a", tool_call_id="c1"),
            Message(role="tool", content="# file b", tool_call_id="c2"),
            Message(
                role="assistant", content="Both files are similar comment headers."
            ),
        ]

        await tracemem_integration.import_trace("conv-1", messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH path = (u:UserText)-[:MESSAGE*]->(final:AgentText)
                WHERE u.conversation_id = 'conv-1'
                  AND u.text = 'Compare a.py and b.py'
                  AND NOT (final)-[:MESSAGE]->()
                RETURN length(path) as chain_length, final.text as final_text
                """
            )
            record = await result.single()
            assert record is not None
            assert record["chain_length"] == 2
            assert record["final_text"] == "Both files are similar comment headers."


class TestTurnIndexNeo4j:
    """Turn index tests that require raw Neo4j queries."""

    async def test_assistant_same_turn_as_user(self, tracemem_integration: TraceMem):
        """Assistant gets same turn_index as preceding user."""
        await tracemem_integration.add_message(
            "conv-1", Message(role="user", content="Hello")
        )
        result = await tracemem_integration.add_message(
            "conv-1", Message(role="assistant", content="Hi!")
        )

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            query_result = await session.run(
                """
                MATCH (a:AgentText {id: $id})
                RETURN a.turn_index as turn_index
                """,
                id=str(result["agent_text"]),
            )
            record = await query_result.single()
            assert record is not None
            assert record["turn_index"] == 0

    async def test_multiple_assistants_same_turn(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Tool flow keeps all assistants in same turn."""
        test_file = tmp_path / "code.py"
        test_file.write_text("print('hello')")

        messages = [
            Message(role="user", content="Read code.py and explain it"),
            Message(
                role="assistant",
                content="I'll read the file for you.",
                tool_calls=[
                    ToolCall(
                        id="call_1", name="read_file", args={"path": str(test_file)}
                    )
                ],
            ),
            Message(role="tool", content="print('hello')", tool_call_id="call_1"),
            Message(
                role="assistant",
                content="The file contains a simple print statement.",
            ),
        ]

        await tracemem_integration.import_trace("conv-1", messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (a:AgentText {conversation_id: 'conv-1'})
                RETURN a.turn_index as turn_index
                ORDER BY a.created_at
                """
            )
            records = await result.data()
            assert len(records) == 2
            assert all(r["turn_index"] == 0 for r in records)

    async def test_turn_based_chain_query(self, tracemem_integration: TraceMem):
        """Can query all nodes in a specific turn."""
        for i in range(3):
            await tracemem_integration.add_message(
                "conv-1", Message(role="user", content=f"User {i}")
            )
            await tracemem_integration.add_message(
                "conv-1", Message(role="assistant", content=f"Agent {i}")
            )

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (n)
                WHERE (n:UserText OR n:AgentText)
                  AND n.conversation_id = 'conv-1'
                  AND n.turn_index = 1
                RETURN n.text as text
                ORDER BY n.created_at
                """
            )
            records = await result.data()
            assert len(records) == 2
            assert records[0]["text"] == "User 1"
            assert records[1]["text"] == "Agent 1"


class TestToolUsesIntegration:
    """Tests for tool_uses property persistence in Neo4j."""

    async def test_tool_uses_persisted_in_neo4j(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify tool_uses are correctly stored and retrieved from Neo4j."""
        test_file = tmp_path / "code.py"
        test_file.write_text("print('hello')")

        messages = [
            Message(role="user", content="Check git status and read the file"),
            Message(
                role="assistant",
                content="I'll check git and read the file for you.",
                tool_calls=[
                    ToolCall(id="call_1", name="bash", args={"command": "git status"}),
                    ToolCall(
                        id="call_2", name="read_file", args={"path": str(test_file)}
                    ),
                ],
            ),
            Message(role="tool", content="On branch main", tool_call_id="call_1"),
            Message(role="tool", content="print('hello')", tool_call_id="call_2"),
        ]

        await tracemem_integration.import_trace("conv-1", messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (a:AgentText {conversation_id: 'conv-1'})
                RETURN a.tool_uses as tool_uses
                """
            )
            record = await result.single()
            assert record is not None

            tool_uses = json.loads(record["tool_uses"])
            assert len(tool_uses) == 2
            assert tool_uses[0]["id"] == "call_1"
            assert tool_uses[0]["name"] == "bash"
            assert tool_uses[0]["args"] == {"command": "git status"}
            assert tool_uses[1]["id"] == "call_2"
            assert tool_uses[1]["name"] == "read_file"

    async def test_tool_uses_empty_json_when_no_tools(
        self, tracemem_integration: TraceMem
    ):
        """Verify empty tool_uses is stored as empty JSON array."""
        await tracemem_integration.add_message(
            "conv-1", Message(role="user", content="Hello")
        )
        await tracemem_integration.add_message(
            "conv-1", Message(role="assistant", content="Hi there!")
        )

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (a:AgentText {conversation_id: 'conv-1'})
                RETURN a.tool_uses as tool_uses
                """
            )
            record = await result.single()
            assert record is not None

            tool_uses = json.loads(record["tool_uses"])
            assert tool_uses == []

    async def test_non_resource_tools_tracked_in_tool_uses(
        self, tracemem_integration: TraceMem
    ):
        """Verify tools that don't create resources are still tracked in tool_uses."""
        messages = [
            Message(role="user", content="Run a command"),
            Message(
                role="assistant",
                content="Let me run that command.",
                tool_calls=[
                    ToolCall(id="c1", name="bash", args={"command": "echo hello"}),
                    ToolCall(id="c2", name="web_search", args={"query": "python docs"}),
                ],
            ),
            Message(role="tool", content="hello", tool_call_id="c1"),
            Message(role="tool", content="Search results...", tool_call_id="c2"),
        ]

        result = await tracemem_integration.import_trace("conv-1", messages)

        resource_keys = [k for k in result.keys() if k.startswith("resource")]
        assert len(resource_keys) == 0

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            query_result = await session.run(
                """
                MATCH (a:AgentText {conversation_id: 'conv-1'})
                RETURN a.tool_uses as tool_uses
                """
            )
            record = await query_result.single()

            tool_uses = json.loads(record["tool_uses"])
            assert len(tool_uses) == 2
            tool_names = [tu["name"] for tu in tool_uses]
            assert tool_names == ["bash", "web_search"]


class TestCrossConversationResourceEdges:
    """Tests verifying multiple conversations create proper edges to shared resources."""

    async def test_two_conversations_both_create_edges_to_shared_resource(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Two conversations reading the same file should both have edges to ResourceVersion."""
        auth_file = tmp_path / "auth.py"
        content = "def login(user, pwd): pass"

        messages1 = [
            Message(role="user", content="Show me the auth code"),
            Message(
                role="assistant",
                content="Reading auth.py...",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", args={"path": str(auth_file)})
                ],
            ),
            Message(role="tool", content=content, tool_call_id="c1"),
        ]
        await tracemem_integration.import_trace("conv-1", messages1)

        messages2 = [
            Message(role="user", content="What's in the auth file?"),
            Message(
                role="assistant",
                content="Let me check auth.py...",
                tool_calls=[
                    ToolCall(id="c2", name="read_file", args={"path": str(auth_file)})
                ],
            ),
            Message(role="tool", content=content, tool_call_id="c2"),
        ]
        await tracemem_integration.import_trace("conv-2", messages2)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (a:AgentText)-[r:READ_FILE]->(v:ResourceVersion)
                RETURN count(r) as edge_count
                """
            )
            record = await result.single()
            assert record["edge_count"] == 2

            result = await session.run(
                """
                MATCH (a:AgentText)-[:READ_FILE]->(v:ResourceVersion)
                RETURN count(DISTINCT v) as version_count
                """
            )
            record = await result.single()
            assert record["version_count"] == 1

            result = await session.run(
                """
                MATCH (a:AgentText)-[:READ_FILE]->(v:ResourceVersion)
                RETURN collect(DISTINCT a.conversation_id) as conversations
                """
            )
            record = await result.single()
            assert set(record["conversations"]) == {"conv-1", "conv-2"}

    async def test_conversations_connected_via_shared_resource(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify a graph path exists between two conversations via shared resource."""
        shared_file = tmp_path / "config.py"
        content = "DEBUG = True"

        for conv_id in ["conv-alice", "conv-bob"]:
            messages = [
                Message(role="user", content=f"Read config ({conv_id})"),
                Message(
                    role="assistant",
                    content="Reading...",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{conv_id}",
                            name="read_file",
                            args={"path": str(shared_file)},
                        )
                    ],
                ),
                Message(role="tool", content=content, tool_call_id=f"call_{conv_id}"),
            ]
            await tracemem_integration.import_trace(conv_id, messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (a1:AgentText {conversation_id: 'conv-alice'})
                      -[:READ_FILE]->(v:ResourceVersion)<-[:READ_FILE]-
                      (a2:AgentText {conversation_id: 'conv-bob'})
                RETURN count(*) as path_exists
                """
            )
            record = await result.single()
            assert record["path_exists"] == 1

    async def test_version_of_edge_exists_for_shared_resource(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify VERSION_OF edge links ResourceVersion to Resource hypernode."""
        shared_file = tmp_path / "utils.py"
        content = "def helper(): pass"

        for conv_id in ["conv-1", "conv-2"]:
            messages = [
                Message(role="user", content="Read utils"),
                Message(
                    role="assistant",
                    content="Reading...",
                    tool_calls=[
                        ToolCall(
                            id=f"call_{conv_id}",
                            name="read_file",
                            args={"path": str(shared_file)},
                        )
                    ],
                ),
                Message(role="tool", content=content, tool_call_id=f"call_{conv_id}"),
            ]
            await tracemem_integration.import_trace(conv_id, messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (v:ResourceVersion)-[:VERSION_OF]->(r:Resource)
                RETURN count(*) as version_of_count,
                       count(DISTINCT r) as resource_count,
                       count(DISTINCT v) as version_count
                """
            )
            record = await result.single()
            assert record["version_of_count"] == 1
            assert record["resource_count"] == 1
            assert record["version_count"] == 1

    async def test_three_conversations_all_connected_to_same_resource(
        self, tracemem_integration: TraceMem, tmp_path
    ):
        """Verify three or more conversations all create edges to shared resource."""
        shared_file = tmp_path / "main.py"
        content = "if __name__ == '__main__': main()"

        for i in range(3):
            messages = [
                Message(role="user", content=f"Read main.py - request {i}"),
                Message(
                    role="assistant",
                    content="Reading main.py...",
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
            await tracemem_integration.import_trace(f"conv-{i}", messages)

        async with tracemem_integration._graph_store._driver.session(
            database="neo4j"
        ) as session:
            result = await session.run(
                """
                MATCH (a:AgentText)-[r:READ_FILE]->(v:ResourceVersion)
                RETURN count(r) as edge_count,
                       collect(DISTINCT a.conversation_id) as conversations
                """
            )
            record = await result.single()
            assert record["edge_count"] == 3
            assert set(record["conversations"]) == {"conv-0", "conv-1", "conv-2"}
