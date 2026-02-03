from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tracemem_core.retrieval.results import RetrievalConfig


class TraceMemConfig(BaseSettings):
    """Configuration for TraceMem.

    Settings can be provided via environment variables with TRACEMEM_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRACEMEM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Graph store backend
    graph_store: Literal["kuzu", "neo4j"] = "kuzu"

    # Home directory for storage (graph + vectors)
    # Default: ~/.tracemem
    home: Path | None = None

    # Neo4j configuration (only used when graph_store="neo4j")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Namespace for Neo4j multi-user isolation
    namespace: str | None = None

    # LanceDB configuration (deprecated — use home instead)
    lancedb_path: Path | None = None

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1)
    openai_api_key: str | None = None

    # Reranker strategy (string key from registry, e.g. "rrf", "linear")
    reranker: str = "rrf"

    # Default retrieval config (used when no per-call config is provided)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    def get_home(self) -> Path:
        """Get the home directory for storage."""
        return self.home or Path.home() / ".tracemem"

    def get_graph_path(self) -> Path:
        """Get the Kùzu graph DB path (only used when graph_store='kuzu')."""
        return self.get_home() / "graph"

    def get_vector_path(self) -> Path:
        """Get the vector store path."""
        if self.lancedb_path:
            return self.lancedb_path
        return self.get_home() / "vectors"

    def get_lancedb_path(self) -> Path:
        """Get the LanceDB storage path based on mode.

        Deprecated: use get_vector_path() instead.
        """
        return self.get_vector_path()
