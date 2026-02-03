"""Resource extraction from tool calls.

This module provides protocols and implementations for extracting resource URIs
from tool call arguments.
"""

from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse


def _canonicalize_file_uri(uri: str, root: Path | None) -> str:
    """Canonicalize a file:// URI. Internal helper.

    - If root is set and path is under root: relative file://path
    - If root is None or path is outside root: absolute file:///path
    - Non-file URIs: pass through as-is

    Args:
        uri: The URI to canonicalize.
        root: Optional root directory for making paths relative.

    Returns:
        Canonicalized URI string.
    """
    parsed = urlparse(uri)

    # Non-file URIs pass through unchanged
    if parsed.scheme and parsed.scheme not in ("file", ""):
        return uri

    # Extract the file path
    if parsed.scheme == "file":
        path = Path(parsed.path)
    else:
        path = Path(uri)

    # Resolve symlinks and normalize
    path = path.resolve()

    if root is not None:
        try:
            rel_path = path.relative_to(root.resolve())
            return f"file://{rel_path}"
        except ValueError:
            # Path is outside root, use absolute
            return f"file://{path}"

    return f"file://{path}"


class ResourceExtractor(Protocol):
    """Protocol for extracting resource URIs from tool calls.

    Implementations can customize how resource URIs are extracted from tool
    arguments based on tool name and argument patterns.
    """

    def extract(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Extract a resource URI from tool call arguments.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments passed to the tool.

        Returns:
            Resource URI if one can be extracted, None otherwise.
        """
        ...


class DefaultResourceExtractor:
    """Default extractor for common file and URL patterns.

    Looks for common argument names like 'path', 'file_path', 'url', etc.
    and extracts URIs from them. Returns already-canonicalized URIs.

    Args:
        mode: "local" makes file URIs relative to the project root (derived
              from home's parent). "global" keeps absolute URIs.
        home: TraceMem home directory (e.g. project/.tracemem). When mode is
              "local", the project root is derived as home.parent.
              Defaults to cwd/.tracemem if not specified in local mode.
    """

    FILE_ARGS = {"path", "file_path", "filepath", "file", "filename"}
    URL_ARGS = {"url", "uri", "endpoint"}

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
        """Extract a resource URI from tool call arguments.

        Checks for file path arguments first, then URL arguments.
        File URIs are canonicalized based on mode.

        Args:
            tool_name: Name of the tool being called (unused in default impl).
            args: Arguments passed to the tool.

        Returns:
            Canonicalized resource URI, or None.
        """
        raw_uri = self._extract_raw(tool_name, args)
        if raw_uri and raw_uri.startswith("file://"):
            return _canonicalize_file_uri(raw_uri, self._root)
        return raw_uri

    def _extract_raw(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """Extract a raw (uncanonicalized) resource URI."""
        for arg in self.FILE_ARGS:
            if arg in args and args[arg]:
                path = args[arg]
                if isinstance(path, str):
                    return f"file://{path}" if not path.startswith("file://") else path

        for arg in self.URL_ARGS:
            if arg in args and args[arg]:
                url = args[arg]
                if isinstance(url, str):
                    return url

        return None
