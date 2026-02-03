"""Claude Code resource extractor.

Extracts resource URIs from Claude Code tool calls (Read, Write, Edit, Bash, etc.).
"""

import re
from pathlib import Path
from typing import Any, Literal

from tracemem_core.extractors import _canonicalize_file_uri


class ClaudeCodeResourceExtractor:
    """Resource extractor for Claude Code tools.

    Handles Read, Write, Edit, Glob, Grep, and Bash tools,
    extracting file paths from their arguments. Returns
    already-canonicalized URIs.

    Args:
        mode: "local" makes file URIs relative to the project root (derived
              from home's parent). "global" keeps absolute URIs.
        home: TraceMem home directory (e.g. project/.tracemem). When mode is
              "local", the project root is derived as home.parent.
              Defaults to cwd/.tracemem if not specified in local mode.
    """

    # Claude Code tools that operate on files
    FILE_TOOLS = {"Read", "Write", "Edit", "NotebookEdit"}
    # Tools that search/match files (extract path argument)
    SEARCH_TOOLS = {"Glob", "Grep"}

    # Pattern to find absolute file paths in bash commands
    _BASH_PATH_RE = re.compile(r"(?:^|\s)(\/[\w./-]+\.\w+)")

    def __init__(
        self,
        *,
        mode: Literal["local", "global"] = "global",
        home: Path | None = None,
    ) -> None:
        self.mode = mode
        if mode == "local":
            _home = home or Path.cwd() / ".tracemem"
            self._root: Path | None = _home.parent.resolve()
        else:
            self._root = None

    def extract(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Extract a resource URI from Claude Code tool arguments.

        Args:
            tool_name: Name of the Claude Code tool.
            args: Arguments passed to the tool.

        Returns:
            Canonicalized resource URI with file:// scheme, or None.
        """
        raw_uri = self._extract_raw(tool_name, args)
        if raw_uri and raw_uri.startswith("file://"):
            return _canonicalize_file_uri(raw_uri, self._root)
        return raw_uri

    def _extract_raw(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Extract a raw (uncanonicalized) resource URI."""
        if tool_name in self.FILE_TOOLS:
            # Read, Write, Edit, NotebookEdit use file_path or notebook_path
            path = args.get("file_path") or args.get("notebook_path")
            if path and isinstance(path, str):
                return f"file://{path}" if not path.startswith("file://") else path

        if tool_name in self.SEARCH_TOOLS:
            # Glob and Grep use path argument for the search location
            path = args.get("path")
            if path and isinstance(path, str):
                return f"file://{path}" if not path.startswith("file://") else path

        if tool_name == "Bash":
            return self._extract_from_bash(args)

        return None

    def _extract_from_bash(self, args: dict[str, Any]) -> str | None:
        """Extract the first absolute file path from a bash command string.

        Looks for paths like /Users/foo/bar.py in the command.
        Only matches paths with a file extension to avoid matching
        bare directories like /usr/bin.

        Args:
            args: Bash tool arguments containing 'command'.

        Returns:
            Resource URI for the first file path found, or None.
        """
        command = args.get("command", "")
        if not command or not isinstance(command, str):
            return None

        match = self._BASH_PATH_RE.search(command)
        if match:
            path = match.group(1)
            return f"file://{path}"

        return None
