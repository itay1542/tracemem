#!/usr/bin/env python3
"""tracemem-claude CLI entry point.

Receives Claude Code hook events via stdin JSON and dispatches to handlers.
Always exits 0 to avoid blocking Claude Code.

Usage:
    tracemem-claude UserPromptSubmit < event.json
    tracemem-claude PostToolUse < event.json
    tracemem-claude Stop < event.json
"""

import asyncio
import json
import sys

from tracemem_claude.handlers import HANDLERS


def main() -> None:
    """Main entry point for the CLI.

    Reads hook event type from argv[1] and JSON data from stdin.
    Dispatches to the appropriate handler.
    Always exits 0 to never block Claude Code.
    """
    # Get event type from command line
    if len(sys.argv) < 2:
        sys.exit(0)

    event = sys.argv[1]

    # Check if we have a handler for this event
    if event not in HANDLERS:
        sys.exit(0)

    # Read event data from stdin
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    # Handle the event
    try:
        handler_class = HANDLERS[event]
        handler = handler_class()
        asyncio.run(handler.handle(data))
    except Exception as e:
        # Log error but never block Claude Code
        print(f"TraceMem error ({event}): {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
