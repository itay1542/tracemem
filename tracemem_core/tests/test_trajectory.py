"""Tests for trajectory expansion and TraceMem.retrieval property."""

import json
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from tracemem_core.retrieval import (
    HybridRetrievalStrategy,
    TrajectoryResult,
    TrajectoryStep,
)
from tracemem_core.retrieval.results import ToolUse
from tracemem_core.tracemem import TraceMem


class TestTrajectoryModels:
    """Tests for TrajectoryStep and TrajectoryResult models."""

    def test_trajectory_step_user_text(self):
        """TrajectoryStep can represent a UserText node."""
        step = TrajectoryStep(
            node_id="user-1",
            node_type="UserText",
            text="Fix the auth bug",
            conversation_id="conv-1",
        )

        assert step.node_id == "user-1"
        assert step.node_type == "UserText"
        assert step.text == "Fix the auth bug"
        assert step.tool_uses == []

    def test_trajectory_step_agent_text_with_tools(self):
        """TrajectoryStep can represent an AgentText with tool uses."""
        step = TrajectoryStep(
            node_id="agent-1",
            node_type="AgentText",
            text="I'll fix the bug",
            conversation_id="conv-1",
            tool_uses=[
                ToolUse(tool_name="Read", properties={"file_path": "auth.py"}),
                ToolUse(tool_name="Edit", properties={"file_path": "auth.py"}),
            ],
        )

        assert step.node_type == "AgentText"
        assert len(step.tool_uses) == 2
        assert step.tool_uses[0].tool_name == "Read"

    def test_trajectory_result_empty(self):
        """Empty TrajectoryResult has empty steps list."""
        result = TrajectoryResult()
        assert result.steps == []

    def test_trajectory_result_with_steps(self):
        """TrajectoryResult can hold multiple steps."""
        result = TrajectoryResult(
            steps=[
                TrajectoryStep(
                    node_id="u1",
                    node_type="UserText",
                    text="Fix bug",
                    conversation_id="c1",
                ),
                TrajectoryStep(
                    node_id="a1",
                    node_type="AgentText",
                    text="Fixed it",
                    conversation_id="c1",
                ),
            ]
        )

        assert len(result.steps) == 2
        assert result.steps[0].node_type == "UserText"
        assert result.steps[1].node_type == "AgentText"


class TestGetTrajectory:
    """Tests for HybridRetrievalStrategy.get_trajectory method."""

    @pytest.fixture
    def strategy(self, mock_graph_store, mock_vector_store, mock_embedder):
        """Create a HybridRetrievalStrategy with mocks."""
        return HybridRetrievalStrategy(
            graph_store=mock_graph_store,
            vector_store=mock_vector_store,
            embedder=mock_embedder,
        )

    async def test_get_trajectory_empty_when_not_found(self, strategy, mock_graph_store):
        """Returns empty trajectory when node not found."""
        mock_graph_store.get_trajectory_nodes = AsyncMock(return_value=[])

        result = await strategy.get_trajectory(uuid4())

        assert result.steps == []

    async def test_get_trajectory_full_turn_with_tools_and_response(
        self, strategy, mock_graph_store
    ):
        """Collects UserText, tool-carrying AgentTexts, final response, and follow-up."""
        user_id = str(uuid4())
        tool_agent_id = str(uuid4())
        response_agent_id = str(uuid4())
        next_user_id = str(uuid4())

        mock_graph_store.get_trajectory_nodes = AsyncMock(
            return_value=[
                {
                    "n": {
                        "id": user_id,
                        "text": "Fix the bug",
                        "conversation_id": "conv-1",
                        "created_at": "2024-01-01T00:00:00",
                    },
                    "node_labels": ["UserText"],
                },
                {
                    "n": {
                        "id": tool_agent_id,
                        "text": "",
                        "conversation_id": "conv-1",
                        "created_at": "2024-01-01T00:01:00",
                        "tool_uses": json.dumps([
                            {"name": "Read", "args": {"file_path": "auth.py"}},
                            {"name": "Edit", "args": {"file_path": "auth.py"}},
                        ]),
                    },
                    "node_labels": ["AgentText"],
                },
                {
                    "n": {
                        "id": response_agent_id,
                        "text": "I fixed the bug by updating token validation",
                        "conversation_id": "conv-1",
                        "created_at": "2024-01-01T00:02:00",
                    },
                    "node_labels": ["AgentText"],
                },
                {
                    "n": {
                        "id": next_user_id,
                        "text": "Now update the tests",
                        "conversation_id": "conv-1",
                        "created_at": "2024-01-01T00:03:00",
                    },
                    "node_labels": ["UserText"],
                },
            ]
        )

        result = await strategy.get_trajectory(UUID(user_id))

        assert len(result.steps) == 4
        assert result.steps[0].node_type == "UserText"
        assert result.steps[0].text == "Fix the bug"
        assert result.steps[1].node_type == "AgentText"
        assert result.steps[1].text == ""
        assert len(result.steps[1].tool_uses) == 2
        assert result.steps[1].tool_uses[0].tool_name == "Read"
        assert result.steps[2].node_type == "AgentText"
        assert result.steps[2].text == "I fixed the bug by updating token validation"
        assert result.steps[3].node_type == "UserText"
        assert result.steps[3].text == "Now update the tests"

    async def test_get_trajectory_stops_at_next_user_text(
        self, strategy, mock_graph_store
    ):
        """Trajectory stops at follow-up UserText, doesn't continue further."""
        user_id = str(uuid4())
        agent_id = str(uuid4())
        next_user_id = str(uuid4())
        third_agent_id = str(uuid4())

        mock_graph_store.get_trajectory_nodes = AsyncMock(
            return_value=[
                {
                    "n": {"id": user_id, "text": "Q1", "conversation_id": "c1", "created_at": "2024-01-01T00:00:00"},
                    "node_labels": ["UserText"],
                },
                {
                    "n": {"id": agent_id, "text": "A1", "conversation_id": "c1", "created_at": "2024-01-01T00:01:00"},
                    "node_labels": ["AgentText"],
                },
                {
                    "n": {"id": next_user_id, "text": "Q2", "conversation_id": "c1", "created_at": "2024-01-01T00:02:00"},
                    "node_labels": ["UserText"],
                },
                {
                    "n": {"id": third_agent_id, "text": "A2", "conversation_id": "c1", "created_at": "2024-01-01T00:03:00"},
                    "node_labels": ["AgentText"],
                },
            ]
        )

        result = await strategy.get_trajectory(UUID(user_id))

        # Should stop after Q2, not include A2
        assert len(result.steps) == 3
        assert result.steps[2].text == "Q2"

    async def test_get_trajectory_end_of_conversation(self, strategy, mock_graph_store):
        """Returns full trajectory when no follow-up UserText exists."""
        user_id = str(uuid4())
        agent_id = str(uuid4())

        mock_graph_store.get_trajectory_nodes = AsyncMock(
            return_value=[
                {
                    "n": {"id": user_id, "text": "Last question", "conversation_id": "c1", "created_at": "2024-01-01T00:00:00"},
                    "node_labels": ["UserText"],
                },
                {
                    "n": {"id": agent_id, "text": "Final answer", "conversation_id": "c1", "created_at": "2024-01-01T00:01:00"},
                    "node_labels": ["AgentText"],
                },
            ]
        )

        result = await strategy.get_trajectory(UUID(user_id))

        assert len(result.steps) == 2
        assert result.steps[0].node_type == "UserText"
        assert result.steps[1].node_type == "AgentText"
        assert result.steps[1].text == "Final answer"

    async def test_get_trajectory_parses_tool_uses_from_json_string(
        self, strategy, mock_graph_store
    ):
        """Parses tool_uses when stored as JSON string."""
        user_id = str(uuid4())
        agent_id = str(uuid4())

        mock_graph_store.get_trajectory_nodes = AsyncMock(
            return_value=[
                {
                    "n": {"id": user_id, "text": "Run ls", "conversation_id": "c1", "created_at": "2024-01-01T00:00:00"},
                    "node_labels": ["UserText"],
                },
                {
                    "n": {
                        "id": agent_id,
                        "text": "",
                        "conversation_id": "c1",
                        "created_at": "2024-01-01T00:01:00",
                        "tool_uses": '[{"name": "Bash", "args": {"command": "ls"}}]',
                    },
                    "node_labels": ["AgentText"],
                },
            ]
        )

        result = await strategy.get_trajectory(UUID(user_id))

        assert len(result.steps) == 2
        assert len(result.steps[1].tool_uses) == 1
        assert result.steps[1].tool_uses[0].tool_name == "Bash"

    async def test_get_trajectory_parses_tool_uses_from_list(
        self, strategy, mock_graph_store
    ):
        """Parses tool_uses when already a list (from Neo4j driver)."""
        user_id = str(uuid4())
        agent_id = str(uuid4())

        mock_graph_store.get_trajectory_nodes = AsyncMock(
            return_value=[
                {
                    "n": {"id": user_id, "text": "Search", "conversation_id": "c1", "created_at": "2024-01-01T00:00:00"},
                    "node_labels": ["UserText"],
                },
                {
                    "n": {
                        "id": agent_id,
                        "text": "",
                        "conversation_id": "c1",
                        "created_at": "2024-01-01T00:01:00",
                        "tool_uses": [{"name": "Grep", "args": {"pattern": "TODO"}}],
                    },
                    "node_labels": ["AgentText"],
                },
            ]
        )

        result = await strategy.get_trajectory(UUID(user_id))

        assert len(result.steps[1].tool_uses) == 1
        assert result.steps[1].tool_uses[0].tool_name == "Grep"

    async def test_get_trajectory_handles_no_tool_uses(
        self, strategy, mock_graph_store
    ):
        """Agent nodes without tool_uses produce empty tool list."""
        user_id = str(uuid4())
        agent_id = str(uuid4())

        mock_graph_store.get_trajectory_nodes = AsyncMock(
            return_value=[
                {
                    "n": {"id": user_id, "text": "Hello", "conversation_id": "c1", "created_at": "2024-01-01T00:00:00"},
                    "node_labels": ["UserText"],
                },
                {
                    "n": {"id": agent_id, "text": "Hi", "conversation_id": "c1", "created_at": "2024-01-01T00:01:00"},
                    "node_labels": ["AgentText"],
                },
            ]
        )

        result = await strategy.get_trajectory(UUID(user_id))

        assert result.steps[1].tool_uses == []

    async def test_get_trajectory_skips_unknown_labels(
        self, strategy, mock_graph_store
    ):
        """Nodes with unknown labels are skipped."""
        user_id = str(uuid4())

        mock_graph_store.get_trajectory_nodes = AsyncMock(
            return_value=[
                {
                    "n": {"id": user_id, "text": "Q", "conversation_id": "c1", "created_at": "2024-01-01T00:00:00"},
                    "node_labels": ["UserText"],
                },
                {
                    "n": {"id": str(uuid4()), "text": "resource", "conversation_id": "c1", "created_at": "2024-01-01T00:01:00"},
                    "node_labels": ["Resource"],
                },
            ]
        )

        result = await strategy.get_trajectory(UUID(user_id))

        assert len(result.steps) == 1
        assert result.steps[0].node_type == "UserText"


class TestTraceMemRetrievalProperty:
    """Tests for TraceMem.retrieval property."""

    def test_retrieval_returns_hybrid_strategy(self, mock_embedder):
        """TraceMem.retrieval returns a HybridRetrievalStrategy."""
        tm = TraceMem(embedder=mock_embedder)

        retrieval = tm.retrieval

        assert isinstance(retrieval, HybridRetrievalStrategy)

    def test_retrieval_is_lazy_singleton(self, mock_embedder):
        """TraceMem.retrieval returns the same instance on repeated access."""
        tm = TraceMem(embedder=mock_embedder)

        first = tm.retrieval
        second = tm.retrieval

        assert first is second

    def test_retrieval_uses_tracemem_stores(self, mock_embedder):
        """Retrieval strategy uses TraceMem's stores and embedder."""
        tm = TraceMem(embedder=mock_embedder)

        retrieval = tm.retrieval

        assert retrieval._graph_store is tm._graph_store
        assert retrieval._vector_store is tm._vector_store
        assert retrieval._embedder is tm._embedder
