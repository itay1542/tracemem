"""Handler for PreToolUse hook events."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from tracemem_core import RetrievalConfig, TraceMem
from tracemem_core.extractors import _canonicalize_file_uri

from tracemem_claude.config import get_hook_config
from tracemem_claude.formatters import format_resource_history
from tracemem_claude.handlers.base import BaseHandler


class PreToolHandler(BaseHandler):
    """Handles PreToolUse events for Read/Write/Edit tools.

    Injects file history context as additionalContext JSON so Claude
    has awareness of past interactions with the file being accessed.
    """

    async def _process(self, tm: TraceMem, data: dict[str, Any]) -> None:
        """Process a PreToolUse event.

        Extracts file_path from tool_input, queries TraceMem for past
        conversations involving that file, and outputs additionalContext JSON.

        Args:
            tm: The TraceMem instance.
            data: The hook event data containing:
                - tool_input: Dict with file_path key
                - session_id: The Claude Code session ID
        """
        file_path = data.get("tool_input", {}).get("file_path")
        session_id = data.get("session_id", "")
        project_root = Path(data.get("cwd", ".")).resolve()

        if not file_path:
            return

        hook_config = get_hook_config()
        if hook_config.mode == "local":
            home = project_root / ".tracemem"
            root = home.parent.resolve()
        else:
            root = None
        try:
            await asyncio.wait_for(
                self._output_resource_context(
                    tm, file_path, session_id, hook_config, root
                ),
                timeout=hook_config.retrieval_timeout_seconds,
            )
        except (TimeoutError, Exception):
            if hook_config.debug:
                import traceback

                print(
                    f"TraceMem PreToolUse error: {traceback.format_exc()}",
                    file=sys.stderr,
                )

    async def _output_resource_context(
        self,
        tm: TraceMem,
        file_path: str,
        session_id: str,
        hook_config: Any,
        root: Path | None,
    ) -> None:
        """Query resource history and output additionalContext JSON."""
        uri = _canonicalize_file_uri(f"file://{file_path}", root=root)
        config = RetrievalConfig(
            limit=hook_config.pre_tool_max_results,
            exclude_conversation_id=session_id,
            sort_by="created_at",
            sort_order="desc",
        )
        refs = await tm.get_conversations_for_resource(uri, config=config)

        if refs:
            context = format_resource_history(file_path, refs)
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "additionalContext": context,
                }
            }))
