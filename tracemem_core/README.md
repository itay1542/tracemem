# TraceMem Core

[![PyPI](https://img.shields.io/pypi/v/tracemem-core)](https://pypi.org/project/tracemem-core/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Knowledge graph memory system for AI agents. Stores coding interactions in a hybrid graph + vector store for contextual recall.

## Installation

```bash
pip install tracemem-core
```

### Requirements

- Python 3.12+
- OpenAI API key (for embeddings, or provide a custom embedder)
- **No external services required** — uses embedded Kuzu (graph) and LanceDB (vectors) by default

Optional: Neo4j 5.x for remote graph storage (`pip install tracemem-core[neo4j]`)

## Quick Start

```python
from tracemem_core import TraceMem, TraceMemConfig, Message, ToolCall

config = TraceMemConfig(
    graph_store="kuzu",          # "kuzu" (embedded, default) or "neo4j"
    # home=Path("~/.tracemem"),  # storage directory (default: ~/.tracemem)
)

async with TraceMem(config=config) as tm:
    # Store a conversation
    await tm.add_message("conv-1", Message(role="user", content="Fix the auth bug in login.py"))
    await tm.add_message("conv-1", Message(
        role="assistant",
        content="I'll check the authentication code...",
        tool_calls=[ToolCall(id="c1", name="read_file", args={"path": "src/login.py"})],
    ))
    await tm.add_message("conv-1", Message(
        role="tool", content="def login(): ...", tool_call_id="c1",
    ))

    # Search similar past interactions
    results = await tm.search("authentication issues")
    for r in results:
        print(r.text, r.score)

    # Get full trajectory from a user message
    trajectory = await tm.get_trajectory(results[0].node_id)
    for step in trajectory.steps:
        print(step.node_type, step.text[:80], step.tool_uses)
```

## Configuration

All settings use the `TRACEMEM_` prefix:

```bash
export OPENAI_API_KEY="sk-..."                    # Required for embeddings
export TRACEMEM_GRAPH_STORE="kuzu"                # kuzu (default) or neo4j
export TRACEMEM_HOME="~/.tracemem"                # Storage directory
export TRACEMEM_EMBEDDING_MODEL="text-embedding-3-small"
export TRACEMEM_RERANKER="rrf"                    # rrf (default) or linear
```

### TraceMemConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `graph_store` | `"kuzu"` \| `"neo4j"` | `"kuzu"` | Graph backend |
| `home` | `Path` | `~/.tracemem` | Storage directory for graph + vectors |
| `embedding_model` | `str` | `"text-embedding-3-small"` | OpenAI embedding model |
| `embedding_dimensions` | `int` | `1536` | Embedding vector dimensions |
| `openai_api_key` | `str` | `None` | OpenAI API key (or use env var) |
| `reranker` | `str` | `"rrf"` | Reranker strategy (`rrf` or `linear`) |
| `neo4j_uri` | `str` | `"bolt://localhost:7687"` | Neo4j URI (only if `graph_store="neo4j"`) |
| `neo4j_user` | `str` | `"neo4j"` | Neo4j username |
| `neo4j_password` | `str` | `"password"` | Neo4j password |
| `neo4j_database` | `str` | `"neo4j"` | Neo4j database name |
| `namespace` | `str` | `None` | Namespace for multi-user isolation (Neo4j) |

## Graph Schema

### Node Types

| Node | Key Properties | Description |
|------|---------------|-------------|
| `UserText` | `id`, `text`, `conversation_id`, `turn_index` | User messages (searchable via vector store) |
| `AgentText` | `id`, `text`, `tool_uses`, `conversation_id`, `turn_index` | Agent responses and tool invocations |
| `ResourceVersion` | `id`, `uri`, `content_hash` | Snapshot of a resource at a point in time |
| `Resource` | `id`, `uri`, `current_content_hash` | Canonical resource identity (deduplicated by URI) |

### Relationship Types

Only 3 relationship types:

| Type | From → To | Description |
|------|-----------|-------------|
| `MESSAGE` | UserText ↔ AgentText | Conversation flow (chronological) |
| `TOOL_USE` | AgentText → ResourceVersion | Agent used a tool that touched this resource |
| `VERSION_OF` | ResourceVersion → Resource | Links snapshot to canonical resource |

## API Reference

### TraceMem

Must be used as an async context manager:

```python
async with TraceMem(config=config, embedder=embedder, resource_extractor=extractor) as tm:
    ...
```

**Parameters:**
- `config` (TraceMemConfig, optional): Configuration settings
- `embedder` (Embedder, optional): Custom embedder (defaults to OpenAI)
- `resource_extractor` (ResourceExtractor, optional): Custom resource URI extractor

#### `await tm.add_message(conversation_id, message)`

Add a single message to the knowledge graph.

#### `await tm.import_trace(conversation_id, messages)`

Import a full conversation from a list of Messages.

#### `await tm.search(query, config=None)`

Search for similar past interactions via vector similarity.

#### `await tm.get_trajectory(node_id)`

Get the full trajectory from a UserText node through all agent responses until the next user message.

### Data Models

```python
from tracemem_core import Message, ToolCall

Message(role="user", content="Fix the bug")
Message(role="assistant", content="...", tool_calls=[ToolCall(id="c1", name="read_file", args={"path": "auth.py"})])
Message(role="tool", content="file contents...", tool_call_id="c1")
```

## Resource Extraction

The `DefaultResourceExtractor` extracts URIs from tool call arguments:

- **File arguments**: `path`, `file_path`, `filepath`, `file`, `filename`
- **URL arguments**: `url`, `uri`, `endpoint`

```python
ToolCall(name="read_file", args={"path": "src/auth.py"})       # → file://src/auth.py
ToolCall(name="fetch", args={"url": "https://api.example.com"}) # → https://api.example.com
```

### Custom Extractor

```python
from tracemem_core import ResourceExtractor

class MyExtractor:
    def extract(self, tool_name: str, args: dict) -> str | None:
        if tool_name == "query_database":
            return f"db://{args.get('table')}"
        return None

async with TraceMem(resource_extractor=MyExtractor()) as tm:
    ...
```

## Custom Embedder

```python
from tracemem_core import Embedder, TraceMem

class MyEmbedder:
    @property
    def dimensions(self) -> int:
        return 768

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

async with TraceMem(embedder=MyEmbedder()) as tm:
    ...
```

## Adapters

### LangChain

```python
from tracemem_core.adapters.langchain import LangChainAdapter

adapter = LangChainAdapter()
messages = adapter.convert(langchain_messages)
await tm.import_trace("conv-123", messages)
```

Requires `langchain-core`: `pip install langchain-core`

## Documentation

- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [Retrieval](docs/retrieval.md)

## License

MIT
