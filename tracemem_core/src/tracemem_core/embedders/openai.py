from openai import AsyncOpenAI


class OpenAIEmbedder:
    """OpenAI embedding implementation."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dimensions,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts into vectors."""
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]
