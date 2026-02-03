"""Session state management for cross-hook communication.

State is persisted in JSON files at ~/.tracemem/sessions/{session_id}.json.
This allows different hook invocations to share state within a session.
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from tracemem_claude.config import get_hook_config


class SessionStateData(BaseModel):
    """Persisted session state data."""

    session_id: str
    transcript_path: str | None = None
    last_turn_index: int = -1
    pending_agent_ids: list[str] = Field(default_factory=list)
    last_user_message_uuid: str | None = None


class SessionState:
    """Manages session state across hook invocations.

    State is stored in JSON files for cross-process communication.
    Each hook invocation reads/writes to the same file for a session.
    """

    def __init__(self, session_id: str) -> None:
        """Initialize session state manager.

        Args:
            session_id: The Claude Code session ID.
        """
        self._session_id = session_id
        self._config = get_hook_config()
        self._state_dir = self._config.state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / f"{session_id}.json"

    def _load(self) -> SessionStateData:
        """Load state from file."""
        if self._state_file.exists():
            data = json.loads(self._state_file.read_text())
            return SessionStateData(**data)
        return SessionStateData(session_id=self._session_id)

    def _save(self, data: SessionStateData) -> None:
        """Save state to file."""
        self._state_file.write_text(data.model_dump_json(indent=2))

    def get_transcript_path(self) -> str | None:
        """Get the transcript file path for this session."""
        return self._load().transcript_path

    def set_transcript_path(self, path: str) -> None:
        """Set the transcript file path for this session."""
        data = self._load()
        data.transcript_path = path
        self._save(data)

    def get_last_turn_index(self) -> int:
        """Get the last processed turn index."""
        return self._load().last_turn_index

    def set_last_turn_index(self, index: int) -> None:
        """Set the last processed turn index."""
        data = self._load()
        data.last_turn_index = index
        self._save(data)

    def get_pending_agent_ids(self) -> list[str]:
        """Get IDs of AgentText nodes awaiting content updates."""
        return self._load().pending_agent_ids

    def add_pending_agent_id(self, agent_id: str) -> None:
        """Add an AgentText node ID that needs content update."""
        data = self._load()
        if agent_id not in data.pending_agent_ids:
            data.pending_agent_ids.append(agent_id)
        self._save(data)

    def clear_pending_agent_ids(self) -> None:
        """Clear all pending agent IDs."""
        data = self._load()
        data.pending_agent_ids = []
        self._save(data)

    def get_last_user_message_uuid(self) -> str | None:
        """Get the UUID of the last user message processed."""
        return self._load().last_user_message_uuid

    def set_last_user_message_uuid(self, uuid: str) -> None:
        """Set the UUID of the last user message processed."""
        data = self._load()
        data.last_user_message_uuid = uuid
        self._save(data)

    def clear_turn_state(self) -> None:
        """Clear turn-specific state after Stop handler processes."""
        data = self._load()
        data.pending_agent_ids = []
        self._save(data)

    def get_all(self) -> dict[str, Any]:
        """Get all session state as a dictionary."""
        return self._load().model_dump()
