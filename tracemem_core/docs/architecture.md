# Architecture

TraceMem uses a hybrid architecture combining graph and vector databases for efficient storage and retrieval of conversation context.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        TraceMem                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Message   │  │  Resource   │  │  Retrieval Strategy │  │
│  │   Ingestion │  │  Extraction │  │  (Hybrid Search)    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         ▼                ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Storage Layer                        ││
│  │  ┌─────────────────┐  ┌──────────────────────────────┐ ││
│  │  │   GraphStore    │  │        VectorStore           │ ││
│  │  │  Kùzu (default) │  │        (LanceDB)             │ ││
│  │  │  Neo4j (opt-in) │  │                              │ ││
│  │  └─────────────────┘  └──────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Graph Store Backends

### Kùzu (Default)

Kùzu is an embedded graph database — no server required. It's the default backend.

- **Zero dependencies**: No Docker, no external services
- **Storage**: `{home}/graph/` directory
- **API**: Synchronous, wrapped with `asyncio.to_thread()` for async interface
- **Schema**: Explicit table creation (`CREATE NODE TABLE`, `CREATE REL TABLE`)
- **Relationships**: Uses `TOOL_USE` rel table with `tool_name` property (since Kùzu requires pre-declared rel tables)

### Neo4j (Optional)

Neo4j is a client-server graph database. Opt in with `graph_store="neo4j"`.

- **Requires**: Running Neo4j instance (Docker or managed)
- **Multi-user**: Namespace-based isolation on shared instances
- **Relationships**: Dynamic relationship types (READ_FILE, EDIT_FILE, etc.)

### Kùzu vs Neo4j Cypher Differences

| Feature | Neo4j | Kùzu |
|---------|-------|------|
| Node labels | `labels(n)` → list | `label(n)` → string |
| Multi-label match | `WHERE (n:A OR n:B)` | `UNION ALL` of separate queries |
| Relationship types | Dynamic (any string) | Pre-declared `TOOL_USE` with `tool_name` |
| Variable-length paths | Unlimited depth | Max depth of 30 |
| `MERGE` | Full support | Supported for nodes |

## Data Model

### Graph Schema

```
┌──────────────┐     MESSAGE      ┌──────────────┐
│   UserText   │ ───────────────► │  AgentText   │
│              │                  │              │
│ - id         │                  │ - id         │
│ - text       │                  │ - text       │
│ - conv_id    │                  │ - conv_id    │
│ - turn_index │                  │ - turn_index │
│ - created_at │                  │ - tool_uses  │
└──────────────┘                  └──────┬───────┘
                                         │
                                    TOOL_USE *
                                         │
                                         ▼
                               ┌─────────────────┐
                               │ ResourceVersion │
                               │                 │
                               │ - id            │
                               │ - uri           │
                               │ - content_hash  │
                               └────────┬────────┘
                                        │
                                   VERSION_OF
                                        │
                                        ▼
                                ┌───────────────┐
                                │   Resource    │
                                │               │
                                │ - id          │
                                │ - uri         │
                                │ - current_hash│
                                └───────────────┘

* Kùzu: TOOL_USE rel with tool_name property
  Neo4j: Dynamic types (READ_FILE, EDIT_FILE, etc.)
```

### Node Types

| Node | Description |
|------|-------------|
| `UserText` | User message in a conversation |
| `AgentText` | Agent (LLM) response |
| `ResourceVersion` | Specific version of a resource (file content at point in time) |
| `Resource` | Canonical resource identifier (hypernode) |

### Edge Types

| Edge | From | To | Description |
|------|------|-----|-------------|
| `MESSAGE` | UserText/AgentText | AgentText/UserText | Conversation flow |
| `TOOL_USE` (Kùzu) | AgentText | ResourceVersion | Tool operation (with `tool_name` property) |
| Dynamic types (Neo4j) | AgentText | ResourceVersion | `READ_FILE`, `EDIT_FILE`, etc. |
| `VERSION_OF` | ResourceVersion | Resource | Version relationship |

### Kùzu Schema Details

Kùzu requires explicit table declarations:

```
Node Tables:
  UserText(id STRING, text STRING, conversation_id STRING, turn_index INT64, ...)
  AgentText(id STRING, text STRING, conversation_id STRING, turn_index INT64, tool_uses STRING, ...)
  ResourceVersion(id STRING, content_hash STRING, uri STRING, ...)
  Resource(id STRING, uri STRING, current_content_hash STRING, ...)

Rel Tables:
  MESSAGE (REL TABLE GROUP): UserText→AgentText, AgentText→UserText, AgentText→AgentText
  VERSION_OF: ResourceVersion→Resource
  TOOL_USE: AgentText→ResourceVersion (tool_name STRING, properties STRING)
```

### Vector Schema (LanceDB)

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique identifier |
| `node_id` | string | Corresponding UserText node ID |
| `text` | string | User message text |
| `vector` | float[1536] | Embedding vector |
| `conversation_id` | string | Conversation identifier |
| `created_at` | timestamp | Creation time |
| `last_accessed` | timestamp | Last access time |

## Conversation Flow

```
Turn 0:
  User: "Read auth.py"
  Agent: "Reading..." [TOOL_USE(read_file) → auth.py v1]

Turn 1:
  User: "Fix the bug on line 42"
  Agent: "I'll fix it..." [TOOL_USE(edit_file) → auth.py v2]
  Agent: "Done, here's what I changed..."
```

Graph representation:
```
(U1)-[MESSAGE]->(A1)-[TOOL_USE]->(V1)-[VERSION_OF]->(R)
       │                                              ▲
       └-[MESSAGE]->(U2)-[MESSAGE]->(A2)-[TOOL_USE]->(V2)-[VERSION_OF]─┘
                                     │
                                     └-[MESSAGE]->(A3)
```

## Hybrid Search

TraceMem uses hybrid search combining:

1. **Vector Search** - Semantic similarity using embeddings
2. **Full-Text Search (FTS)** - Keyword matching using BM25

The `vector_weight` parameter controls the balance:
- `0.0` = Pure FTS (keyword matching)
- `1.0` = Pure vector (semantic similarity)
- `0.7` = Default (favor semantic, include keywords)

### Reranking

LanceDB's `LinearCombinationReranker` combines scores:

```
final_score = vector_weight * vector_score + (1 - vector_weight) * fts_score
```

## Cross-Conversation Queries

Resources act as "hypernodes" connecting multiple conversations:

```
Conversation A                    Conversation B
     │                                 │
     ▼                                 ▼
  AgentText ──TOOL_USE──► ResourceVersion ◄──TOOL_USE── AgentText
                                  │
                             VERSION_OF
                                  │
                                  ▼
                              Resource
```

Query (Kùzu):
```cypher
MATCH (u:UserText)-[:MESSAGE*1..30]->(a:AgentText)-[:TOOL_USE]->(v:ResourceVersion)-[:VERSION_OF]->(res:Resource {uri: $uri})
RETURN DISTINCT u.conversation_id AS cid, u.text AS user_text, a.text AS agent_text
```

Query (Neo4j):
```cypher
MATCH (res:Resource {uri: $uri})<-[:VERSION_OF]-(v:ResourceVersion)
MATCH (v)<-[r]-(a:AgentText)<-[:MESSAGE*]-(u:UserText)
WHERE type(r) <> 'VERSION_OF'
RETURN DISTINCT u.conversation_id, u.text, a.text
```

## Turn-Based Organization

Each conversation is organized into turns:
- **Turn 0**: First user message + all agent responses
- **Turn 1**: Second user message + all agent responses
- etc.

This enables queries like:
- "Get all messages in turn 2"
- "Get the last node in turn 0"

## Storage Protocols

TraceMem uses protocols for storage abstraction:

```python
class GraphStore(Protocol):
    # Node/edge CRUD
    async def create_node(self, node: NodeBase) -> NodeBase: ...
    async def create_edge(self, edge: EdgeBase) -> EdgeBase: ...
    async def execute_cypher(self, query: str, params: dict) -> list[dict]: ...

    # Retrieval queries
    async def get_node_context(self, node_id: UUID) -> ContextResult: ...
    async def get_resource_conversations(
        self, uri: str, *, limit: int, sort_by: str,
        sort_order: str, exclude_conversation_id: str | None,
    ) -> list[ConversationReference]: ...
    async def get_trajectory_nodes(
        self, node_id: UUID, *, max_depth: int,
    ) -> list[dict[str, Any]]: ...
    # ...

class VectorStore(Protocol):
    async def add(self, node_id: UUID, text: str, vector: list[float], ...) -> None: ...
    async def search(self, query_vector: list[float], query_text: str, ...) -> list[VectorSearchResult]: ...
    # ...
```

Both `KuzuGraphStore` and `Neo4jGraphStore` implement the `GraphStore` protocol. The retrieval query methods (`get_node_context`, `get_resource_conversations`, `get_trajectory_nodes`) keep graph queries encapsulated in the storage layer, while the `HybridRetrievalStrategy` acts as a thin orchestrator.

`LanceDBVectorStore` also supports injecting a custom reranker (e.g., `CohereReranker`, `CrossEncoderReranker`) instead of the default `LinearCombinationReranker`.
