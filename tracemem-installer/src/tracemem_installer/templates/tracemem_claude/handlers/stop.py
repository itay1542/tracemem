"""Handler for Stop hook events."""

from typing import Any

from tracemem_core import Message, TraceMem

from tracemem_claude.handlers.base import BaseHandler
from tracemem_claude.state.session import SessionState
from tracemem_claude.transcript.parser import TranscriptParser


class StopHandler(BaseHandler):
    """Handles Stop events from Claude Code.

    The Stop event fires when Claude Code finishes responding.
    This handler:
    1. Parses the transcript for assistant text content
    2. Writes an AgentText node with the combined assistant text
    3. Clears turn-specific session state
    """

    async def _process(self, tm: TraceMem, data: dict[str, Any]) -> None:
        """Process a stop event.

        Args:
            tm: The TraceMem instance.
            data: The hook event data containing:
                - session_id: The Claude Code session ID
        """
        session_id = data.get("session_id", "")
        if not session_id:
            return

        state = SessionState(session_id)
        transcript_path = state.get_transcript_path()

        if transcript_path:
            # Parse transcript for assistant text content
            parser = TranscriptParser(transcript_path)
            combined_text = parser.get_full_assistant_text_since_last_user()

            # Write the assistant text as an AgentText node
            if combined_text.strip():
                message = Message(role="assistant", content=combined_text)
                await tm.add_message(session_id, message)

        # Clear turn-specific state
        state.clear_turn_state()
