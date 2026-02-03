# Configuration

TraceMem is configured using the `TraceMemConfig` model.

## Quick Start

### Embedded (Zero Config)

```python
from tracemem_core import TraceMemConfig, TraceMem

# Just works — uses embedded Kùzu + LanceDB in ~/.tracemem/
async with TraceMem() as tm:
    ...
```

No Docker, no external databases. Data stored in `~/.tracemem/` by default.

### Project-Specific Storage

```python
from pathlib import Path
from tracemem_core import TraceMemConfig, TraceMem, DefaultResourceExtractor

config = TraceMemConfig(home=Path.cwd() / ".tracemem")
tm = TraceMem(
    config=config,
    resource_extractor=DefaultResourceExtractor(mode="local", home=config.home),
)
```

Data stored in `{project}/.tracemem/`. The `mode="local"` parameter on the extractor
makes file URIs relative to the project directory (derived from `home.parent`).

### Neo4j (Optional)

```python
config = TraceMemConfig(
    graph_store="neo4j",
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
)
```

Requires a running Neo4j instance. See [Docker Compose Example](#docker-compose-example) below.

## TraceMemConfig

```python
from tracemem_core import TraceMemConfig, TraceMem

config = TraceMemConfig(
    # Graph store backend: "kuzu" (default, embedded) or "neo4j"
    graph_store="kuzu",

    # Home directory for all storage (graph + vectors)
    # Default: ~/.tracemem
    home=None,

    # Neo4j connection (only used when graph_store="neo4j")
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    neo4j_database="neo4j",

    # Namespace for Neo4j multi-user isolation
    namespace=None,
)

async with TraceMem(config=config) as tm:
    ...
```

## Configuration Options

### graph_store

Controls which graph database backend to use:

- **`kuzu`** (default): Embedded Kùzu graph database
  - Zero external dependencies — no Docker required
  - Data stored in `{home}/graph/`
  - Synchronous API wrapped with `asyncio.to_thread()`

- **`neo4j`**: Neo4j graph database
  - Requires a running Neo4j instance
  - Supports multi-user isolation via `namespace`
  - Dynamic relationship types (READ_FILE, EDIT_FILE, etc.)

### home

Base directory for all storage. Defaults to `~/.tracemem` if not set.

The home directory contains:
- `graph/` — Kùzu database files (when `graph_store="kuzu"`)
- `vectors/` — LanceDB vector store

### namespace

Optional namespace for Neo4j multi-user isolation. When set:
- All nodes include a `namespace` property
- All queries filter by namespace
- Enables multiple users/teams on a shared Neo4j instance

```python
config = TraceMemConfig(
    graph_store="neo4j",
    namespace="team-alpha",
)
```

### Neo4j Connection

Only used when `graph_store="neo4j"`.

| Option | Default | Description |
|--------|---------|-------------|
| `neo4j_uri` | `bolt://localhost:7687` | Neo4j connection URI |
| `neo4j_user` | `neo4j` | Neo4j username |
| `neo4j_password` | `password` | Neo4j password |
| `neo4j_database` | `neo4j` | Neo4j database name |

### LanceDB Path (Deprecated)

| Option | Default | Description |
|--------|---------|-------------|
| `lancedb_path` | Auto | Path to LanceDB directory |

Deprecated — use `home` instead. If set, overrides the auto-configured vector store path.

## Environment Variables

TraceMem can also be configured via environment variables:

```bash
export TRACEMEM_GRAPH_STORE="kuzu"         # or "neo4j"
export TRACEMEM_NEO4J_URI="bolt://localhost:7687"
export TRACEMEM_NEO4J_USER="neo4j"
export TRACEMEM_NEO4J_PASSWORD="your-password"
export TRACEMEM_NEO4J_DATABASE="neo4j"
export TRACEMEM_NAMESPACE="team-alpha"
export TRACEMEM_RERANKER="rrf"             # or "linear"
```

## Custom Embedder

By default, TraceMem uses OpenAI's `text-embedding-3-small` model. You can provide a custom embedder:

```python
from tracemem_core import TraceMem, Embedder

class MyEmbedder(Embedder):
    @property
    def dimensions(self) -> int:
        return 768

    async def embed(self, text: str) -> list[float]:
        # Your embedding logic
        return [0.0] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

tm = TraceMem(embedder=MyEmbedder())
```

## Reranker

TraceMem uses a reranker for hybrid search (combining vector and full-text results). Built-in presets require no API keys:

| Preset | Reranker | Description |
|--------|----------|-------------|
| `"rrf"` (default) | `RRFReranker` | Reciprocal Rank Fusion |
| `"linear"` | `LinearCombinationReranker` | Linear combination (weight=0.5) |

```python
# Via config (env-configurable as TRACEMEM_RERANKER=linear)
config = TraceMemConfig(reranker="linear")
tm = TraceMem(config=config)

# Via constructor (overrides config)
tm = TraceMem(reranker="linear")

# Custom reranker instance (for API-key rerankers like Cohere, Jina)
from lancedb.rerankers import CohereReranker
tm = TraceMem(reranker=CohereReranker(api_key="..."))
```

## Default Retrieval Config

Set instance-level defaults for all retrieval calls:

```python
from tracemem_core import TraceMemConfig, RetrievalConfig

config = TraceMemConfig(
    retrieval=RetrievalConfig(limit=5, vector_weight=0.7),
)

async with TraceMem(config=config) as tm:
    # Uses config.retrieval defaults (limit=5, vector_weight=0.7)
    results = await tm.search("query")

    # Per-call override
    results = await tm.search("query", config=RetrievalConfig(limit=20))
```

## Docker Compose Example

Only needed when using `graph_store="neo4j"`:

```yaml
version: "3.8"
services:
  neo4j:
    image: neo4j:5.15.0
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

Start with:
```bash
docker compose up -d neo4j
```
