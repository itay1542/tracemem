from typing import Protocol


class Embedder(Protocol):
    """Protocol for embedding text into vectors."""

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors."""
        ...
