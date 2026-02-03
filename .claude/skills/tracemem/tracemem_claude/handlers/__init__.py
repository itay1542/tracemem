"""Hook handlers for Claude Code events."""

from tracemem_claude.handlers.post_tool import PostToolHandler
from tracemem_claude.handlers.pre_tool import PreToolHandler
from tracemem_claude.handlers.stop import StopHandler
from tracemem_claude.handlers.user_prompt import UserPromptHandler

HANDLERS: dict[str, type] = {
    "UserPromptSubmit": UserPromptHandler,
    "PreToolUse": PreToolHandler,
    "PostToolUse": PostToolHandler,
    "Stop": StopHandler,
}

__all__ = [
    "HANDLERS",
    "UserPromptHandler",
    "PreToolHandler",
    "PostToolHandler",
    "StopHandler",
]
