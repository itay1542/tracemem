from datetime import datetime
from uuid import UUID

from tracemem_core.messages import Message, ToolCall
from tracemem_core.models.edges import Relationship, VersionOf
from tracemem_core.models.nodes import AgentText, Resource, ResourceVersion, UserText


class TestNodeModels:
    """Test node models."""

    def test_user_text_defaults(self) -> None:
        """UserText should have auto-generated defaults."""
        user_text = UserText(
            text="Hello",
            conversation_id="conv-1",
        )

        assert isinstance(user_text.id, UUID)
        assert isinstance(user_text.created_at, datetime)
        assert isinstance(user_text.last_accessed_at, datetime)
        assert user_text.text == "Hello"
        assert user_text.conversation_id == "conv-1"

    def test_agent_text_defaults(self) -> None:
        """AgentText should have auto-generated defaults."""
        agent_text = AgentText(
            text="Hi there!",
            conversation_id="conv-1",
        )

        assert isinstance(agent_text.id, UUID)
        assert agent_text.text == "Hi there!"

    def test_resource_version_defaults(self) -> None:
        """ResourceVersion should have auto-generated defaults."""
        version = ResourceVersion(
            content_hash="abc123",
            uri="file:///project/src/auth.py",
            conversation_id="conv-1",
        )

        assert isinstance(version.id, UUID)
        assert version.content_hash == "abc123"
        assert version.uri == "file:///project/src/auth.py"

    def test_resource_defaults(self) -> None:
        """Resource should have auto-generated defaults."""
        resource = Resource(
            uri="file:///project/src/auth.py",
            conversation_id="conv-1",
        )

        assert isinstance(resource.id, UUID)
        assert resource.uri == "file:///project/src/auth.py"
        assert resource.current_content_hash is None

    def test_resource_with_hash(self) -> None:
        """Resource can be created with a current hash."""
        resource = Resource(
            uri="file:///project/src/auth.py",
            current_content_hash="xyz789",
            conversation_id="conv-1",
        )

        assert resource.current_content_hash == "xyz789"


class TestEdgeModels:
    """Test edge models."""

    def test_relationship_defaults(self) -> None:
        """Relationship should have auto-generated defaults."""
        source_id = UUID("12345678-1234-1234-1234-123456789012")
        target_id = UUID("87654321-4321-4321-4321-210987654321")

        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type="READ",
            conversation_id="conv-1",
        )

        assert isinstance(rel.id, UUID)
        assert isinstance(rel.created_at, datetime)
        assert rel.source_id == source_id
        assert rel.target_id == target_id
        assert rel.relationship_type == "READ"
        assert rel.properties == {}

    def test_relationship_with_properties(self) -> None:
        """Relationship can have custom properties."""
        rel = Relationship(
            source_id=UUID("12345678-1234-1234-1234-123456789012"),
            target_id=UUID("87654321-4321-4321-4321-210987654321"),
            relationship_type="EDIT",
            conversation_id="conv-1",
            properties={"line_start": 10, "line_end": 20},
        )

        assert rel.properties == {"line_start": 10, "line_end": 20}

    def test_version_of_defaults(self) -> None:
        """VersionOf should have auto-generated defaults."""
        version_id = UUID("12345678-1234-1234-1234-123456789012")
        resource_id = UUID("87654321-4321-4321-4321-210987654321")

        edge = VersionOf(
            version_id=version_id,
            resource_id=resource_id,
        )

        assert isinstance(edge.id, UUID)
        assert isinstance(edge.created_at, datetime)
        assert edge.version_id == version_id
        assert edge.resource_id == resource_id


class TestMessageModels:
    """Test message models."""

    def test_message_user(self) -> None:
        """Create a user message."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_call_id is None

    def test_message_assistant(self) -> None:
        """Create an assistant message."""
        msg = Message(role="assistant", content="Hi there!")

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_message_assistant_with_tool_calls(self) -> None:
        """Create an assistant message with tool calls."""
        msg = Message(
            role="assistant",
            content="I'll read the file",
            tool_calls=[
                ToolCall(id="call_1", name="read_file", args={"path": "/file.py"})
            ],
        )

        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_1"
        assert msg.tool_calls[0].name == "read_file"
        assert msg.tool_calls[0].args == {"path": "/file.py"}

    def test_message_tool(self) -> None:
        """Create a tool message."""
        msg = Message(role="tool", content="file content", tool_call_id="call_1")

        assert msg.role == "tool"
        assert msg.content == "file content"
        assert msg.tool_call_id == "call_1"

    def test_message_system(self) -> None:
        """Create a system message."""
        msg = Message(role="system", content="You are helpful")

        assert msg.role == "system"
        assert msg.content == "You are helpful"

    def test_tool_call_defaults(self) -> None:
        """ToolCall has default empty args."""
        tc = ToolCall(id="call_1", name="test_tool")

        assert tc.id == "call_1"
        assert tc.name == "test_tool"
        assert tc.args == {}
