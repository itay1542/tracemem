# TraceMem

[![PyPI - tracemem-core](https://img.shields.io/pypi/v/tracemem-core?label=tracemem-core)](https://pypi.org/project/tracemem-core/)
[![PyPI - tracemem-claude](https://img.shields.io/pypi/v/tracemem-claude?label=tracemem-claude)](https://pypi.org/project/tracemem-claude/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Knowledge graph memory for AI agents. TraceMem captures coding interactions into a hybrid graph + vector store, enabling contextual recall of past conversations, file history, and tool usage patterns.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Your Agent  │────▶│   TraceMem Core  │────▶│  Kuzu / Neo4j   │
│  (or Claude) │     │                  │     │  (knowledge     │
└─────────────┘     │  Messages ─▶     │     │   graph)        │
                    │  Extraction ─▶   │     └─────────────────┘
                    │  Retrieval  ◀─   │     ┌─────────────────┐
                    │                  │────▶│  LanceDB        │
                    └──────────────────┘     │  (vector search) │
                                             └─────────────────┘
```

- **Graph store** (Kuzu embedded or Neo4j): stores conversation nodes, tool use relationships, and resource version history
- **Vector store** (LanceDB): enables semantic similarity search over past user queries
- **Hybrid retrieval**: combines graph traversal with vector search, reranked with RRF

## Packages

| Package | Description |
|---------|-------------|
| [`tracemem-core`](https://pypi.org/project/tracemem-core/) | Core library — graph/vector stores, retrieval, extractors |
| [`tracemem-claude`](https://pypi.org/project/tracemem-claude/) | Claude Code hooks installer — captures interactions automatically |

## Quick Start

### Install

```bash
pip install tracemem-core
```

### Usage

```python
from tracemem_core import TraceMem, TraceMemConfig, Message, ToolCall

config = TraceMemConfig(
    graph_store="kuzu",         # "kuzu" (embedded, default) or "neo4j"
    # home=Path("~/.tracemem"), # storage directory (default: ~/.tracemem)
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

    # Retrieve similar past interactions
    results = await tm.retrieval.search("authentication issues")
    for r in results.conversations:
        print(r.user_text, r.score)

    # Query file history
    history = await tm.retrieval.resource_history("file:///src/login.py")
```

### Environment Variables

All settings use the `TRACEMEM_` prefix (see [`.env.example`](.env.example)):

```bash
export OPENAI_API_KEY="sk-..."               # Required for embeddings
export TRACEMEM_GRAPH_STORE="kuzu"            # kuzu (default) or neo4j
export TRACEMEM_HOME="~/.tracemem"            # Storage directory
```

## Claude Code Integration

TraceMem integrates with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via hooks, automatically capturing every coding session into the knowledge graph.

### Install hooks

```bash
# Install per-project hooks
uvx tracemem-claude init

# Or install globally
uvx tracemem-claude init --global
```

### What it does

- **`UserPromptSubmit`** — searches past queries for relevant context, surfaces it before Claude responds
- **`PreToolUse`** — queries file history for Read/Write/Edit operations, provides version context
- **`PostToolUse`** — stores tool results and assistant messages into the graph
- **`Stop`** — parses the full transcript at session end

### Uninstall

```bash
uvx tracemem-claude uninstall
uvx tracemem-claude uninstall --global
```

## Documentation

- **[Architecture](tracemem_core/docs/architecture.md)** — system design, graph schema, storage backends
- **[Configuration](tracemem_core/docs/configuration.md)** — all config options and environment variables
- **[Retrieval](tracemem_core/docs/retrieval.md)** — search strategies, reranking, and query APIs

## Examples

Jupyter notebooks in [`tracemem_core/examples/`](tracemem_core/examples/):

| Notebook | Description |
|----------|-------------|
| [01_quickstart](tracemem_core/examples/01_quickstart.ipynb) | Basic setup and message import |
| [02_retrieval_configs](tracemem_core/examples/02_retrieval_configs.ipynb) | Search and retrieval configuration |
| [03_custom_extractors](tracemem_core/examples/03_custom_extractors.ipynb) | Custom resource extraction |
| [04_graph_exploration](tracemem_core/examples/04_graph_exploration.ipynb) | Exploring the knowledge graph |

## Development

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- Docker (optional, for Neo4j integration tests)

### Setup

```bash
git clone https://github.com/ItayVerkworworko/tracemem.git
cd tracemem
uv sync --all-packages
```

### Running tests

```bash
# Unit + Kuzu tests (no external deps needed)
uv run pytest tracemem_core/tests/ -m "not neo4j and not openai" -v

# All tests (requires Neo4j + OpenAI key)
docker compose up -d neo4j
uv run pytest tracemem_core/tests/ -v
```

### Lint & format

```bash
uv run ruff check tracemem_core/src/ tracemem_core/tests/
uv run ruff format tracemem_core/src/ tracemem_core/tests/
```

### Build

```bash
uv build --package tracemem-core
uv build --package tracemem-claude
```

## License

[MIT](LICENSE)
