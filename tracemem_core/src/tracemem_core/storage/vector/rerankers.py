"""Reranker registry for LanceDB hybrid search.

Provides named presets for common reranking strategies that ship with LanceDB
and don't require external API keys or extra dependencies.

Custom reranker instances (e.g. CohereReranker, CrossEncoderReranker) can be
passed directly — they bypass the registry.
"""

from typing import Any

from lancedb.rerankers import LinearCombinationReranker, RRFReranker

RERANKER_REGISTRY: dict[str, Any] = {
    "rrf": RRFReranker(),
    "linear": LinearCombinationReranker(weight=0.5),
}


def get_reranker(reranker: str | Any) -> Any:
    """Resolve a reranker by name or pass through an instance.

    Args:
        reranker: Either a string key from RERANKER_REGISTRY (e.g. "rrf",
            "linear") or a reranker instance to use directly.

    Returns:
        A reranker instance ready for use with LanceDB hybrid search.

    Raises:
        ValueError: If a string key is not found in the registry.
    """
    if isinstance(reranker, str):
        if reranker not in RERANKER_REGISTRY:
            raise ValueError(
                f"Unknown reranker {reranker!r}. "
                f"Available: {list(RERANKER_REGISTRY)}"
            )
        return RERANKER_REGISTRY[reranker]
    return reranker
