"""Retrieval strategies for TraceMem.

This module provides strategies for retrieving context from the knowledge graph.
Strategies combine vector search with graph traversal for different use cases.
"""

from tracemem_core.retrieval.hybrid import HybridRetrievalStrategy
from tracemem_core.retrieval.protocol import RetrievalStrategy
from tracemem_core.retrieval.results import (
    ConversationReference,
    ContextResult,
    RetrievalConfig,
    RetrievalResult,
    ToolUse,
    TrajectoryResult,
    TrajectoryStep,
)

__all__ = [
    "RetrievalStrategy",
    "HybridRetrievalStrategy",
    "RetrievalConfig",
    "RetrievalResult",
    "ContextResult",
    "ToolUse",
    "ConversationReference",
    "TrajectoryResult",
    "TrajectoryStep",
]
