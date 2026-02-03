"""Handler for UserPromptSubmit hook events."""

import asyncio
import sys
from typing import Any

from tracemem_core import Message, RetrievalConfig, TraceMem

from tracemem_claude.config import get_hook_config
from tracemem_claude.formatters import format_similar_queries
from tracemem_claude.handlers.base import BaseHandler
from tracemem_claude.state.session import SessionState


class UserPromptHandler(BaseHandler):
    """Handles UserPromptSubmit events from Claude Code.

    Searches for similar past queries with trajectory expansion,
    outputs context to stdout, then writes user message to TraceMem.
    """

    async def _process(self, tm: TraceMem, data: dict[str, Any]) -> None:
        """Process a user prompt submission.

        Flow:
        1. Search similar past queries (excluding current session)
        2. Expand top results to full trajectories
        3. Output formatted context to stdout (Claude sees it)
        4. Write user message to TraceMem

        Args:
            tm: The TraceMem instance.
            data: The hook event data containing:
                - session_id: The Claude Code session ID
                - prompt: The user's prompt text
                - transcript_path: Path to the session transcript file
        """
        session_id = data.get("session_id", "")
        prompt = data.get("prompt", "")
        transcript_path = data.get("transcript_path", "")

        if not session_id or not prompt:
            return

        # Retrieve similar past queries (fail silently on timeout)
        hook_config = get_hook_config()
        try:
            await asyncio.wait_for(
                self._output_similar_queries(tm, prompt, session_id, hook_config),
                timeout=hook_config.retrieval_timeout_seconds,
            )
        except (TimeoutError, Exception):
            if hook_config.debug:
                import traceback

                print(
                    f"TraceMem retrieval error: {traceback.format_exc()}",
                    file=sys.stderr,
                )

        # Write user message to TraceMem
        message = Message(role="user", content=prompt)
        await tm.add_message(session_id, message)

        # Save transcript path for Stop handler
        state = SessionState(session_id)
        if transcript_path:
            state.set_transcript_path(transcript_path)

    async def _output_similar_queries(
        self,
        tm: TraceMem,
        prompt: str,
        session_id: str,
        hook_config: Any,
    ) -> None:
        """Search for similar past queries and output context to stdout."""
        config = RetrievalConfig(
            limit=hook_config.retrieval_max_results,
            include_context=False,
            exclude_conversation_id=session_id,
            vector_weight=0.3
        )
        results = await tm.search(prompt, config=config)

        if not results:
            return

        # Expand each result to full trajectory
        pairs = []
        for result in results:
            trajectory = await tm.get_trajectory(result.node_id)
            pairs.append((result, trajectory))

        formatted = format_similar_queries(pairs)
        if formatted:
            print(formatted)
