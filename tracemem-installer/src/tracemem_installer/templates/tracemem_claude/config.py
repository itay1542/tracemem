"""Configuration for TraceMem Claude Code hooks.

Loads settings from config.yaml (alongside the skill directory)
with env var overrides for secrets and select settings.
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel


class HookConfig(BaseModel):
    """Configuration for Claude Code hooks."""

    # Session state storage
    state_dir: Path = Path.home() / ".tracemem" / "sessions"

    # Graph store backend
    graph_store: Literal["kuzu", "neo4j"] = "kuzu"

    # Neo4j configuration (only used when graph_store="neo4j")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    openai_api_key: str | None = None

    # Retrieval settings
    retrieval_timeout_seconds: float = 3.0
    retrieval_max_results: int = 3
    pre_tool_max_results: int = 5

    # Mode: "local" (per-project storage) or "global" (shared ~/.tracemem)
    mode: Literal["local", "global"] = "local"

    # Namespace for multi-user isolation
    namespace: str | None = None

    # Reranker strategy
    reranker: str = "linear"

    # Hook behavior
    debug: bool = False


def _find_config_path() -> Path | None:
    """Find config.yaml relative to this module.

    config.yaml sits in the skill directory (parent of tracemem_claude/).
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        return config_path
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config and flatten nested sections into HookConfig field names."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            # Flatten nested sections: neo4j.uri -> neo4j_uri, etc.
            for sub_key, sub_value in value.items():
                flat[f"{key}_{sub_key}"] = sub_value
        else:
            flat[key] = value

    # Resolve ~ in path values
    if "state_dir" in flat:
        flat["state_dir"] = Path(str(flat["state_dir"])).expanduser()

    return flat


def _load_dotenv(path: Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def get_hook_config() -> HookConfig:
    """Get the hook configuration from YAML + .env + env var overrides."""
    data: dict[str, Any] = {}

    config_path = _find_config_path()
    if config_path:
        data = _load_yaml(config_path)

        # Load .env from same directory as config.yaml
        dotenv = _load_dotenv(config_path.parent / ".env")
        if api_key := dotenv.get("TRACEMEM_OPENAI_API_KEY"):
            data["openai_api_key"] = api_key

    # Env var overrides (highest priority)
    if api_key := os.environ.get("TRACEMEM_OPENAI_API_KEY"):
        data["openai_api_key"] = api_key
    if mode := os.environ.get("TRACEMEM_MODE"):
        data["mode"] = mode
    if graph_store := os.environ.get("TRACEMEM_GRAPH_STORE"):
        data["graph_store"] = graph_store
    if debug := os.environ.get("TRACEMEM_DEBUG"):
        data["debug"] = debug.lower() in ("1", "true", "yes")

    return HookConfig(**data)
