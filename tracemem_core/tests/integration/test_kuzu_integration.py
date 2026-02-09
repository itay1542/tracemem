"""Kuzu-specific graph structure verification tests.

Kuzu equivalents of Neo4j graph structure tests, using execute_cypher()
with Kuzu-compatible Cypher syntax.

Key Kuzu Cypher differences:
- label(e) not type(e), label(n) not labels(n)
- [:MESSAGE*1..30] explicit bounds on variable-length paths
- TOOL_USE relationship with tool_name property (no READ_FILE relation type)
- No multi-MATCH with shared variables

No Docker required — uses embedded Kuzu and LanceDB in tmp_path.

Run with:
    uv run pytest tests/integration/test_kuzu_integration.py -v
"""

import json

import pytest

from tracemem_core.messages import Message, ToolCall
from tracemem_core.tracemem import TraceMem

pytestmark = pytest.mark.kuzu


class TestKuzuConversationChain:
    """Tests for conversation chain connectivity in Kuzu graph."""

    async def test_conversation_forms_connected_chain(self, tracemem_kuzu: TraceMem):
        """Verify alternating messages form a fully connected chain."""
        for i in range(3):
            await tracemem_kuzu.add_message(
                "conv-1", Message(role="user", content=f"User message {i + 1}")
            )
            await tracemem_kuzu.add_message(
                "conv-1", Message(role="assistant", content=f"Agent response {i + 1}")
            )

        # Count all MESSAGE edges in the conversation
        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:UserText {conversation_id: 'conv-1'})"
            "-[r:MESSAGE]->(b:AgentText)"
            " RETURN count(r) AS edge_count"
        )
        ua_edges = records[0]["edge_count"]

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {conversation_id: 'conv-1'})"
            "-[r:MESSAGE]->(b:UserText)"
            " RETURN count(r) AS edge_count"
        )
        au_edges = records[0]["edge_count"]

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {conversation_id: 'conv-1'})"
            "-[r:MESSAGE]->(b:AgentText)"
            " RETURN count(r) AS edge_count"
        )
        aa_edges = records[0]["edge_count"]

        # 3 user + 3 agent = 6 nodes, 5 edges total
        # U->A (3), A->U (2), A->A (0)
        assert ua_edges + au_edges + aa_edges == 5

    async def test_new_user_message_links_from_last_agent(
        self, tracemem_kuzu: TraceMem
    ):
        """Verify a new user message creates edge from last agent node."""
        await tracemem_kuzu.add_message(
            "conv-1", Message(role="user", content="First user message")
        )
        await tracemem_kuzu.add_message(
            "conv-1", Message(role="assistant", content="First agent response")
        )
        await tracemem_kuzu.add_message(
            "conv-1", Message(role="user", content="Second user message")
        )

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {text: 'First agent response'})"
            "-[:MESSAGE]->"
            "(u:UserText {text: 'Second user message'})"
            " RETURN a.text AS agent_text, u.text AS user_text"
        )
        assert len(records) == 1
        assert records[0]["agent_text"] == "First agent response"
        assert records[0]["user_text"] == "Second user message"

    async def test_tool_usage_creates_connected_chain(
        self, tracemem_kuzu: TraceMem, tmp_path
    ):
        """Verify tool usage with multiple assistant messages forms connected chain.

        Flow: User -> Assistant (with tool_call) -> Tool result -> Assistant (final)
        Graph: UserText -> AgentText1 -> AgentText2
        """
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

        await tracemem_kuzu.import_trace("conv-1", messages)

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (u:UserText {text: 'Read code.py and explain it'})"
            "-[:MESSAGE]->(a1:AgentText)"
            "-[:MESSAGE]->(a2:AgentText)"
            ' WHERE a1.text = "I\'ll read the file for you."'
            " AND a2.text = 'The file contains a simple print statement.'"
            " RETURN count(*) AS chain_exists"
        )
        assert len(records) == 1
        assert records[0]["chain_exists"] == 1

    async def test_multiple_tool_calls_with_final_response(
        self, tracemem_kuzu: TraceMem, tmp_path
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

        await tracemem_kuzu.import_trace("conv-1", messages)

        # Chain: UserText -> AgentText1 -> AgentText2 = 2 edges
        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (u:UserText {text: 'Compare a.py and b.py'})"
            "-[:MESSAGE*1..10]->(final:AgentText)"
            " WHERE NOT EXISTS { MATCH (final)-[:MESSAGE]->() }"
            " RETURN final.text AS final_text"
        )
        assert len(records) == 1
        assert records[0]["final_text"] == "Both files are similar comment headers."


class TestKuzuTurnIndex:
    """Turn index tests using Kuzu Cypher queries."""

    async def test_assistant_same_turn_as_user(self, tracemem_kuzu: TraceMem):
        """Assistant gets same turn_index as preceding user."""
        await tracemem_kuzu.add_message("conv-1", Message(role="user", content="Hello"))
        result = await tracemem_kuzu.add_message(
            "conv-1", Message(role="assistant", content="Hi!")
        )

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {id: $id}) RETURN a.turn_index AS turn_index",
            parameters={"id": str(result["agent_text"])},
        )
        assert len(records) == 1
        assert records[0]["turn_index"] == 0

    async def test_multiple_assistants_same_turn(
        self, tracemem_kuzu: TraceMem, tmp_path
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

        await tracemem_kuzu.import_trace("conv-1", messages)

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {conversation_id: 'conv-1'})"
            " RETURN a.turn_index AS turn_index"
            " ORDER BY a.created_at"
        )
        assert len(records) == 2
        assert all(r["turn_index"] == 0 for r in records)

    async def test_turn_based_query(self, tracemem_kuzu: TraceMem):
        """Can query all nodes in a specific turn."""
        for i in range(3):
            await tracemem_kuzu.add_message(
                "conv-1", Message(role="user", content=f"User {i}")
            )
            await tracemem_kuzu.add_message(
                "conv-1", Message(role="assistant", content=f"Agent {i}")
            )

        # Query UserText nodes in turn 1
        user_records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (n:UserText {conversation_id: 'conv-1', turn_index: 1})"
            " RETURN n.text AS text ORDER BY n.created_at"
        )
        # Query AgentText nodes in turn 1
        agent_records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (n:AgentText {conversation_id: 'conv-1', turn_index: 1})"
            " RETURN n.text AS text ORDER BY n.created_at"
        )

        all_records = user_records + agent_records
        texts = sorted([r["text"] for r in all_records])
        assert texts == ["Agent 1", "User 1"]


class TestKuzuToolUsesStorage:
    """Tests for tool_uses property persistence in Kuzu."""

    async def test_tool_uses_persisted(self, tracemem_kuzu: TraceMem, tmp_path):
        """Verify tool_uses are correctly stored and retrieved from Kuzu."""
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

        await tracemem_kuzu.import_trace("conv-1", messages)

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {conversation_id: 'conv-1'})"
            " RETURN a.tool_uses AS tool_uses"
        )
        assert len(records) == 1

        tool_uses = json.loads(records[0]["tool_uses"])
        assert len(tool_uses) == 2
        assert tool_uses[0]["id"] == "call_1"
        assert tool_uses[0]["name"] == "bash"
        assert tool_uses[0]["args"] == {"command": "git status"}
        assert tool_uses[1]["id"] == "call_2"
        assert tool_uses[1]["name"] == "read_file"

    async def test_tool_uses_empty_when_no_tools(self, tracemem_kuzu: TraceMem):
        """Verify empty tool_uses is stored as empty JSON array."""
        await tracemem_kuzu.add_message("conv-1", Message(role="user", content="Hello"))
        await tracemem_kuzu.add_message(
            "conv-1", Message(role="assistant", content="Hi there!")
        )

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {conversation_id: 'conv-1'})"
            " RETURN a.tool_uses AS tool_uses"
        )
        assert len(records) == 1

        tool_uses = json.loads(records[0]["tool_uses"])
        assert tool_uses == []

    async def test_non_resource_tools_tracked(self, tracemem_kuzu: TraceMem):
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

        result = await tracemem_kuzu.import_trace("conv-1", messages)

        resource_keys = [k for k in result.keys() if k.startswith("resource")]
        assert len(resource_keys) == 0

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText {conversation_id: 'conv-1'})"
            " RETURN a.tool_uses AS tool_uses"
        )

        tool_uses = json.loads(records[0]["tool_uses"])
        assert len(tool_uses) == 2
        tool_names = [tu["name"] for tu in tool_uses]
        assert tool_names == ["bash", "web_search"]


class TestKuzuCrossConversationEdges:
    """Tests verifying cross-conversation resource edges in Kuzu."""

    async def test_two_conversations_both_create_edges_to_shared_resource(
        self, tracemem_kuzu: TraceMem, tmp_path
    ):
        """Two conversations reading the same file should both have TOOL_USE edges."""
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
        await tracemem_kuzu.import_trace("conv-1", messages1)

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
        await tracemem_kuzu.import_trace("conv-2", messages2)

        # Count TOOL_USE edges to ResourceVersion
        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText)-[r:TOOL_USE]->(v:ResourceVersion)"
            " RETURN count(r) AS edge_count"
        )
        assert records[0]["edge_count"] == 2

        # Verify both point to same ResourceVersion
        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText)-[:TOOL_USE]->(v:ResourceVersion)"
            " RETURN count(DISTINCT v.id) AS version_count"
        )
        assert records[0]["version_count"] == 1

    async def test_conversations_connected_via_shared_resource(
        self, tracemem_kuzu: TraceMem, tmp_path
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
            await tracemem_kuzu.import_trace(conv_id, messages)

        # Verify path: AgentText(alice) -[TOOL_USE]-> ResourceVersion <-[TOOL_USE]- AgentText(bob)
        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a1:AgentText {conversation_id: 'conv-alice'})"
            "-[:TOOL_USE]->(v:ResourceVersion)<-[:TOOL_USE]-"
            "(a2:AgentText {conversation_id: 'conv-bob'})"
            " RETURN count(*) AS path_exists"
        )
        assert records[0]["path_exists"] == 1

    async def test_version_of_edge_exists_for_shared_resource(
        self, tracemem_kuzu: TraceMem, tmp_path
    ):
        """Verify VERSION_OF edge links ResourceVersion to Resource."""
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
            await tracemem_kuzu.import_trace(conv_id, messages)

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (v:ResourceVersion)-[:VERSION_OF]->(r:Resource)"
            " RETURN count(*) AS version_of_count,"
            " count(DISTINCT r.id) AS resource_count,"
            " count(DISTINCT v.id) AS version_count"
        )
        assert records[0]["version_of_count"] == 1
        assert records[0]["resource_count"] == 1
        assert records[0]["version_count"] == 1

    async def test_three_conversations_all_connected_to_same_resource(
        self, tracemem_kuzu: TraceMem, tmp_path
    ):
        """Verify three conversations all create TOOL_USE edges to shared resource."""
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
            await tracemem_kuzu.import_trace(f"conv-{i}", messages)

        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText)-[r:TOOL_USE]->(v:ResourceVersion)"
            " RETURN count(r) AS edge_count"
        )
        assert records[0]["edge_count"] == 3

        # Verify all conversations are represented
        records = await tracemem_kuzu._graph_store.execute_cypher(
            "MATCH (a:AgentText)-[:TOOL_USE]->(v:ResourceVersion)"
            " RETURN DISTINCT a.conversation_id AS conv_id"
        )
        conv_ids = {r["conv_id"] for r in records}
        assert conv_ids == {"conv-0", "conv-1", "conv-2"}
