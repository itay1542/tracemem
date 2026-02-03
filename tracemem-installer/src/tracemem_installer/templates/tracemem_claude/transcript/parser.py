"""Parser for Claude Code JSONL transcript files.

Claude Code writes session transcripts as JSONL with entries containing:
- type: "user", "assistant", "file-history-snapshot", etc.
- message: The message content with role and content blocks
- uuid: Unique identifier for the entry
- parentUuid: Parent entry UUID for threading
- sessionId: The session identifier
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TextBlock:
    """A text content block from an assistant message."""

    text: str
    entry_index: int
    uuid: str


@dataclass
class ToolUseBlock:
    """A tool use block from an assistant message."""

    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]
    entry_index: int
    uuid: str


class TranscriptParser:
    """Parses Claude Code JSONL transcript files.

    Claude Code transcript format:
    - Each line is a JSON object with 'type' and 'message' fields
    - type="user" entries contain user prompts
    - type="assistant" entries contain assistant responses with content blocks
    - Content blocks can be: text, thinking, tool_use, tool_result
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize the parser.

        Args:
            path: Path to the JSONL transcript file.
        """
        self._path = Path(path)
        self._entries: list[dict[str, Any]] | None = None

    def _load_entries(self) -> list[dict[str, Any]]:
        """Load and cache transcript entries."""
        if self._entries is None:
            self._entries = []
            if self._path.exists():
                content = self._path.read_text()
                for line in content.strip().split("\n"):
                    if line:
                        try:
                            self._entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return self._entries

    def get_last_user_entry_index(self) -> int:
        """Find the index of the last user message entry.

        Returns:
            Index of the last user entry, or -1 if not found.
        """
        entries = self._load_entries()
        for i in range(len(entries) - 1, -1, -1):
            if entries[i].get("type") == "user":
                return i
        return -1

    def get_assistant_texts_since_last_user(self) -> list[str]:
        """Extract assistant text content from the latest turn.

        Collects all text blocks from assistant messages after
        the most recent user message.

        Returns:
            List of text strings from assistant messages.
        """
        entries = self._load_entries()
        last_user_idx = self.get_last_user_entry_index()

        texts: list[str] = []
        for entry in entries[last_user_idx + 1 :]:
            if entry.get("type") != "assistant":
                continue

            message = entry.get("message", {})
            content = message.get("content", [])

            if not isinstance(content, list):
                continue

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        texts.append(text)

        return texts

    def get_text_blocks_since_last_user(self) -> list[TextBlock]:
        """Get detailed text blocks since the last user message.

        Returns:
            List of TextBlock objects with metadata.
        """
        entries = self._load_entries()
        last_user_idx = self.get_last_user_entry_index()

        blocks: list[TextBlock] = []
        for i, entry in enumerate(entries[last_user_idx + 1 :], start=last_user_idx + 1):
            if entry.get("type") != "assistant":
                continue

            message = entry.get("message", {})
            content = message.get("content", [])
            uuid = entry.get("uuid", "")

            if not isinstance(content, list):
                continue

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        blocks.append(TextBlock(text=text, entry_index=i, uuid=uuid))

        return blocks

    def get_tool_uses_since_last_user(self) -> list[ToolUseBlock]:
        """Get tool use blocks since the last user message.

        Returns:
            List of ToolUseBlock objects with metadata.
        """
        entries = self._load_entries()
        last_user_idx = self.get_last_user_entry_index()

        blocks: list[ToolUseBlock] = []
        for i, entry in enumerate(entries[last_user_idx + 1 :], start=last_user_idx + 1):
            if entry.get("type") != "assistant":
                continue

            message = entry.get("message", {})
            content = message.get("content", [])
            uuid = entry.get("uuid", "")

            if not isinstance(content, list):
                continue

            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    blocks.append(
                        ToolUseBlock(
                            tool_use_id=block.get("id", ""),
                            tool_name=block.get("name", ""),
                            tool_input=block.get("input", {}),
                            entry_index=i,
                            uuid=uuid,
                        )
                    )

        return blocks

    def get_full_assistant_text_since_last_user(self) -> str:
        """Get combined assistant text since last user message.

        Concatenates all text blocks with newlines.

        Returns:
            Combined text from all assistant text blocks.
        """
        texts = self.get_assistant_texts_since_last_user()
        return "\n".join(texts)

    def get_entries_since_index(self, start_index: int) -> list[dict[str, Any]]:
        """Get all entries since a given index.

        Args:
            start_index: The index to start from (exclusive).

        Returns:
            List of entries after the start index.
        """
        entries = self._load_entries()
        return entries[start_index + 1 :]
