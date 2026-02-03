# TraceMem Core

Knowledge graph memory system for AI agents using Neo4j and LanceDB.

## Installation

```bash
# Using UV (recommended)
uv add tracemem-core

# Using pip
pip install tracemem-core
```

### Requirements

- Python 3.12+
- Neo4j 5.x instance (local or cloud)
- OpenAI API key (for embeddings, or provide a custom embedder)

## Quick Start

### With LangChain Adapter

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tracemem_core import TraceMem, TraceMemConfig
from tracemem_core.adapters.langchain import LangChainAdapter

adapter = LangChainAdapter()

config = TraceMemConfig(
    mode="local",
    project_root=Path.cwd(),
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your-password",
)

async with TraceMem(config=config) as tm:
    # Convert LangChain messages to internal format
    messages = adapter.convert([
        HumanMessage(content="Fix the auth bug in login.py"),
        AIMessage(
            content="I'll check the authentication code...",
            tool_calls=[{"id": "call_1", "name": "read_file", "args": {"path": "src/login.py"}}],
        ),
        ToolMessage(content="def login(): ...", tool_call_id="call_1"),
    ])

    # Import the conversation
    await tm.import_trace("conv-123", messages)
```

### Direct Usage (No Framework)

```python
from tracemem_core import TraceMem, Message, ToolCall

async with TraceMem() as tm:
    # Add messages directly
    await tm.add_message("conv-123", Message(role="user", content="Read auth.py"))
    await tm.add_message("conv-123", Message(
        role="assistant",
        content="I'll read that file",
        tool_calls=[ToolCall(id="call_1", name="read_file", args={"path": "auth.py"})],
    ))
    await tm.add_message("conv-123", Message(
        role="tool",
        content="def authenticate(): ...",
        tool_call_id="call_1",
    ))
```

## Configuration

### Environment Variables

All settings can be configured via environment variables with the `TRACEMEM_` prefix:

```bash
export TRACEMEM_NEO4J_URI="bolt://localhost:7687"
export TRACEMEM_NEO4J_USER="neo4j"
export TRACEMEM_NEO4J_PASSWORD="password"
export TRACEMEM_OPENAI_API_KEY="sk-..."
export TRACEMEM_MODE="local"
```

### TraceMemConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mode` | `"local"` \| `"global"` | `"local"` | Operating mode for path handling |
| `project_root` | `Path` | `None` | Root directory for local mode |
| `neo4j_uri` | `str` | `"bolt://localhost:7687"` | Neo4j connection URI |
| `neo4j_user` | `str` | `"neo4j"` | Neo4j username |
| `neo4j_password` | `str` | `"password"` | Neo4j password |
| `neo4j_database` | `str` | `"neo4j"` | Neo4j database name |
| `lancedb_path` | `Path` | Auto | Vector store location |
| `embedding_model` | `str` | `"text-embedding-3-small"` | OpenAI embedding model |
| `embedding_dimensions` | `int` | `1536` | Embedding vector dimensions |
| `openai_api_key` | `str` | `None` | OpenAI API key |

## Operating Modes

### Local Mode

Paths are stored relative to `project_root`. The project can be moved or renamed and memory remains valid.

```python
config = TraceMemConfig(
    mode="local",
    project_root=Path("/home/user/my-project"),
)

# File at /home/user/my-project/src/auth.py
# Stored as: file://src/auth.py
```

### Global Mode

Paths are stored as absolute paths. Memory spans across projects.

```python
config = TraceMemConfig(mode="global")

# File at /home/user/my-project/src/auth.py
# Stored as: file:///home/user/my-project/src/auth.py
```

## API Reference

### TraceMem

The main interface for the knowledge graph memory system.

#### `async with TraceMem(config, embedder, resource_extractor) as tm:`

Create a TraceMem instance. Must be used as an async context manager.

**Parameters:**
- `config` (TraceMemConfig, optional): Configuration settings
- `embedder` (Embedder, optional): Custom embedder implementation
- `resource_extractor` (ResourceExtractor, optional): Custom resource extractor

---

#### `await tm.import_trace(conversation_id, messages)`

Import a conversation trace from a list of Messages.

**Parameters:**
- `conversation_id` (str): Unique identifier for the conversation
- `messages` (list[Message]): List of Message objects to import

**Returns:** `dict[str, UUID]` - Created node IDs

```python
from tracemem_core.adapters.langchain import LangChainAdapter

adapter = LangChainAdapter()
messages = adapter.convert(langchain_messages)
result = await tm.import_trace("conv-123", messages)
# result = {"user_text": UUID(...), "agent_text": UUID(...), ...}
```

---

#### `await tm.add_message(conversation_id, message)`

Add a single message to the knowledge graph.

**Parameters:**
- `conversation_id` (str): Unique identifier for the conversation
- `message` (Message): The Message to add

**Returns:** `dict[str, UUID]` - Created node IDs

```python
await tm.add_message("conv-123", Message(role="user", content="Hello"))
await tm.add_message("conv-123", Message(
    role="assistant",
    content="I'll read that file",
    tool_calls=[ToolCall(id="c1", name="read_file", args={"path": "auth.py"})],
))
```

### Data Models

#### Message

Internal message representation. Framework-agnostic.

```python
from tracemem_core import Message, ToolCall

# User message
Message(role="user", content="Hello")

# Assistant message with tool calls
Message(
    role="assistant",
    content="I'll read that file",
    tool_calls=[ToolCall(id="call_1", name="read_file", args={"path": "auth.py"})],
)

# Tool result
Message(role="tool", content="file contents...", tool_call_id="call_1")

# System message
Message(role="system", content="You are a helpful assistant")
```

#### ToolCall

A tool invocation within an assistant message.

```python
ToolCall(
    id="call_123",           # Unique tool call ID
    name="read_file",        # Tool name
    args={"path": "auth.py"} # Tool arguments (default: {})
)
```

## Adapters

Adapters convert framework-specific messages to TraceMem's internal `Message` format.

### LangChainAdapter

Converts LangChain messages (HumanMessage, AIMessage, ToolMessage, SystemMessage).

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tracemem_core.adapters.langchain import LangChainAdapter

adapter = LangChainAdapter()

# Convert a list of messages
messages = adapter.convert([
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!"),
])

# Convert a single message
message = adapter.convert_single(HumanMessage(content="Hello"))
```

### Custom Adapter

Implement the `TraceAdapter` protocol:

```python
from tracemem_core.adapters.protocol import TraceAdapter
from tracemem_core import Message

class MyAdapter:
    def convert(self, messages: list) -> list[Message]:
        return [self.convert_single(m) for m in messages]

    def convert_single(self, message) -> Message:
        # Your conversion logic
        return Message(role="user", content=str(message))
```

## Resource Extraction

TraceMem automatically extracts resource URIs from tool calls using the `ResourceExtractor` protocol.

### Default Extractor

The `DefaultResourceExtractor` looks for common argument patterns:

- **File arguments**: `path`, `file_path`, `filepath`, `file`, `filename`
- **URL arguments**: `url`, `uri`, `endpoint`

```python
# These tool calls will have resources extracted:
ToolCall(name="read_file", args={"path": "src/auth.py"})      # -> file://src/auth.py
ToolCall(name="fetch", args={"url": "https://api.example.com"}) # -> https://api.example.com
```

### Custom Extractor

Implement the `ResourceExtractor` protocol for custom extraction:

```python
from tracemem_core import ResourceExtractor

class MyExtractor:
    def extract(self, tool_name: str, args: dict) -> str | None:
        if tool_name == "query_database":
            return f"db://{args.get('table')}"
        if tool_name == "read_file" and "path" in args:
            return f"file://{args['path']}"
        return None

async with TraceMem(resource_extractor=MyExtractor()) as tm:
    ...
```

## Custom Embedder

Implement the `Embedder` protocol to use a custom embedding model:

```python
from tracemem_core import Embedder, TraceMem

class MyEmbedder:
    @property
    def dimensions(self) -> int:
        return 768

    async def embed(self, text: str) -> list[float]:
        # Your embedding logic
        return [0.1] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

async with TraceMem(embedder=MyEmbedder()) as tm:
    ...
```

## Neo4j Setup

### Option 1: Docker Compose (Recommended for Development)

The repository includes a `docker-compose.yml` for local development:

```bash
# Start Neo4j (uses port 17687 to avoid conflicts)
docker compose up -d neo4j

# Check status
docker compose ps

# Stop
docker compose down
```

Connection settings for development:
- URI: `bolt://localhost:17687`
- User: `neo4j`
- Password: `testpassword`

### Option 2: Neo4j Desktop

1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new project and database
3. Start the database
4. Use the bolt URI shown in the connection details (default: `bolt://localhost:7687`)

### Option 3: Docker (Manual)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:5-community
```

### Option 4: Neo4j Aura (Cloud)

1. Create a free instance at [Neo4j Aura](https://neo4j.com/cloud/aura/)
2. Use the provided connection URI

## Graph Schema

TraceMem automatically creates the following schema:

### Node Labels

| Label | Description | Key Properties |
|-------|-------------|----------------|
| `UserText` | User messages (searchable) | `id`, `text`, `conversation_id` |
| `AgentText` | Agent responses | `id`, `text`, `conversation_id` |
| `ResourceVersion` | Resource snapshots | `id`, `content_hash`, `uri` |
| `Resource` | Resource identity | `id`, `uri`, `current_content_hash` |

### Relationships

| Type | From | To | Description |
|------|------|-----|-------------|
| `MESSAGE` | UserText | AgentText | Links user query to agent response |
| `VERSION_OF` | ResourceVersion | Resource | Links version to resource identity |
| `{TOOL_NAME}` | AgentText | ResourceVersion | Tool invocation (READ_FILE, EDIT, etc.) |

## Development

### Setup

```bash
# Clone and install dependencies
git clone <repo-url>
cd tracemem/tracemem_core
uv sync --dev
```

### Running Tests

```bash
# Start Neo4j for integration tests
docker compose up -d neo4j

# Run all tests
uv run pytest tests/ -v

# Run unit tests only (no Neo4j required)
uv run pytest tests/ -v --ignore=tests/storage/graph/ --ignore=tests/integration/

# Run integration tests only
uv run pytest tests/integration/test_tracemem_integration.py -v

# Run with coverage
uv run pytest tests/ --cov=src/tracemem_core --cov-report=html
```

### Test Markers

- `neo4j`: Tests requiring a Neo4j database
- `openai`: Tests requiring an OpenAI API key

```bash
# Skip Neo4j tests
uv run pytest tests/ -v -m "not neo4j"

# Skip OpenAI tests
uv run pytest tests/ -v -m "not openai"
```

## License

MIT
