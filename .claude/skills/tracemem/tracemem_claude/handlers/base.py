"""Base handler for Claude Code hook events."""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from tracemem_core import TraceMem
from tracemem_core.config import TraceMemConfig

from tracemem_claude.config import get_hook_config
from tracemem_claude.extractors import ClaudeCodeResourceExtractor


class BaseHandler(ABC):
    """Base class for hook event handlers.

    Provides TraceMem connection management and common utilities.
    Subclasses implement _process() for specific event handling.
    """

    def __init__(self) -> None:
        """Initialize the handler."""
        self._hook_config = get_hook_config()

    async def handle(self, data: dict[str, Any]) -> None:
        """Handle a hook event.

        Establishes TraceMem connection and delegates to _process().
        Errors are logged but don't block Claude Code.

        Args:
            data: The hook event data from Claude Code.
        """
        cwd = data.get("cwd", ".")
        project_root = Path(cwd).resolve()

        # Mode determines home directory and URI canonicalization
        mode = self._hook_config.mode
        if mode == "global":
            home = None  # TraceMemConfig defaults to ~/.tracemem
        else:
            home = project_root / ".tracemem"

        config = TraceMemConfig(
            home=home,
            graph_store=self._hook_config.graph_store,
            embedding_model=self._hook_config.embedding_model,
            embedding_dimensions=self._hook_config.embedding_dimensions,
            openai_api_key=self._hook_config.openai_api_key,
            namespace=self._hook_config.namespace,
            reranker=self._hook_config.reranker,
            # Neo4j fields only used when graph_store="neo4j"
            neo4j_uri=self._hook_config.neo4j_uri,
            neo4j_user=self._hook_config.neo4j_user,
            neo4j_password=self._hook_config.neo4j_password,
            neo4j_database=self._hook_config.neo4j_database,
        )

        resource_extractor = ClaudeCodeResourceExtractor(mode=mode, home=home)

        try:
            async with TraceMem(config=config, resource_extractor=resource_extractor) as tm:
                await self._process(tm, data)
        except Exception as e:
            if self._hook_config.debug:
                print(f"TraceMem handler error: {e}", file=sys.stderr)
            raise

    @abstractmethod
    async def _process(self, tm: TraceMem, data: dict[str, Any]) -> None:
        """Process the hook event.

        Subclasses implement this method for specific event handling.

        Args:
            tm: The TraceMem instance.
            data: The hook event data.
        """
        ...
