---
name: tracemem
description: Search past coding interactions for relevant context. Use when recalling how similar problems were solved, what decisions were made about files/features, or finding relevant patterns from past conversations.
user_invocable: true
context: fork
argument-hint: [search query or --stats or --file-history <path>]
---

# TraceMem Search

Search the TraceMem knowledge graph for relevant past interactions.

## Graph Schema

TraceMem stores coding interactions as a directed graph. The backend is configurable — **Kuzu** (embedded, default) or **Neo4j** (remote).

```
UserText --MESSAGE--> AgentText --MESSAGE--> AgentText --MESSAGE--> ... --MESSAGE--> UserText (next turn)
                          |                      |
                       TOOL_USE              TOOL_USE
                          |                      |
                          v                      v
                    ResourceVersion        ResourceVersion
                          |                      |
                       VERSION_OF             VERSION_OF
                          |                      |
                          v                      v
                       Resource              Resource
```

### Node Types

| Node | Properties | Description |
|------|-----------|-------------|
| **UserText** | `id`, `text`, `conversation_id`, `turn_index`, `created_at`, `last_accessed_at` | A user message/prompt |
| **AgentText** | `id`, `text`, `tool_uses` (JSON string), `conversation_id`, `turn_index`, `created_at`, `last_accessed_at` | An agent response or tool invocation. Most AgentText nodes have empty `text` and carry `tool_uses` JSON; the final AgentText in a turn has the actual response text. |
| **Resource** | `id`, `uri`, `current_content_hash`, `conversation_id`, `created_at`, `last_accessed_at` | A file or external resource (deduplicated by URI) |
| **ResourceVersion** | `id`, `uri`, `content_hash`, `conversation_id`, `created_at`, `last_accessed_at` | A snapshot of a resource at a point in time |

### Relationship Types (only 3)

| Relationship | From → To | Description |
|-------------|-----------|-------------|
| **MESSAGE** | UserText → AgentText, AgentText → AgentText, AgentText → UserText | Conversation flow in chronological order |
| **TOOL_USE** | AgentText → ResourceVersion | Agent used a tool that touched this resource (Read, Write, Edit, Bash, Grep, etc.) |
| **VERSION_OF** | ResourceVersion → Resource | Links a snapshot to its canonical resource |

### Kuzu (Embedded, Default)

Kuzu is the default graph backend. Its Cypher dialect has important differences from Neo4j:

- Use `label(e)` not `type(e)` for relationship types
- Use `label(n)` not `labels(n)[0]` for node labels
- `CONTAINS` works for string matching
- Variable-length paths: `[:MESSAGE*1..30]` — always use explicit bounds (Kuzu caps at 30)
- No `collect(DISTINCT ...)` — use subqueries or separate aggregation
- **No multi-MATCH with shared variables**: Kuzu does NOT support `MATCH (...)->(a) MATCH (u)->(a)` — variables from the first MATCH are not in scope for the second. Always use a **single MATCH path** instead:
  ```
  # WRONG (fails in Kuzu):
  MATCH (a:AgentText)-[:TOOL_USE]->(v) MATCH (u:UserText)-[:MESSAGE*]->(a) ...

  # CORRECT (single path):
  MATCH (u:UserText)-[:MESSAGE*1..30]->(a:AgentText)-[:TOOL_USE]->(v:ResourceVersion) ...
  ```
- Use inline property filters `(r:Resource {uri: '...'})` instead of separate WHERE when possible

### Neo4j (Remote)

When configured with Neo4j (`TRACEMEM_NEO4J_URI`), standard Neo4j Cypher applies:

- Use `type(r)` for relationship types, `labels(n)[0]` for node labels
- Multi-MATCH with shared variables works normally
- Variable-length paths: `[:MESSAGE*]` (no bound needed)
- `collect(DISTINCT ...)` works
- Standard `WITH` clause for passing variables between MATCH clauses

Check `config.yaml` or env vars to determine which backend is active.

### URIs

File URIs are stored relative to the project root in local mode: `file://tracemem_core/src/tracemem_core/tracemem.py`

## Query Script

Use the query script at `.claude/skills/tracemem/query_graph.py` via Bash:

```bash
# Graph overview: node counts, conversations, resources
uv run .claude/skills/tracemem/query_graph.py --stats

# File history: past interactions with a specific file
uv run .claude/skills/tracemem/query_graph.py --file-history path/to/file.py

# Raw Cypher query
uv run .claude/skills/tracemem/query_graph.py "MATCH (u:UserText) RETURN u.text LIMIT 5"

# JSON output for structured processing
uv run .claude/skills/tracemem/query_graph.py --json "MATCH (n:Resource) RETURN n.uri"
```

### Example Cypher Queries

Find all user questions and their agent responses:
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (u:UserText)-[:MESSAGE]->(a:AgentText)
  WHERE a.text <> ''
  RETURN u.text, left(a.text, 200)
  ORDER BY u.created_at DESC LIMIT 5
"
```

Find what files were accessed during a conversation:
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (u:UserText)-[:MESSAGE*]->(a:AgentText)-[:TOOL_USE]->(v:ResourceVersion)
  RETURN u.text, v.uri
  ORDER BY u.created_at DESC LIMIT 10
"
```

Find all conversations that touched a specific file (single-path traversal — Kuzu requires all nodes in one MATCH):
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (u:UserText)-[:MESSAGE*1..30]->(a:AgentText)-[:TOOL_USE]->(v:ResourceVersion)-[:VERSION_OF]->(r:Resource {uri: 'file://tracemem_core/src/tracemem_core/tracemem.py'})
  RETURN DISTINCT u.conversation_id, u.text
  ORDER BY u.created_at DESC LIMIT 10
"
```

Find the most frequently accessed files:
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (v:ResourceVersion)<-[:TOOL_USE]-(a:AgentText)
  RETURN v.uri, count(*) AS access_count
  ORDER BY access_count DESC LIMIT 10
"
```

Get full trajectory for a user turn (all tool uses until next user message):
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (u:UserText {id: 'NODE_ID_HERE'})-[:MESSAGE*]->(a:AgentText)
  WHERE a.conversation_id = u.conversation_id
  RETURN a.text, a.tool_uses
  ORDER BY a.created_at ASC
"
```

Find resources matching a pattern:
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (r:Resource) WHERE r.uri CONTAINS 'extractors' RETURN r.uri LIMIT 20
"
```

## Workflow

1. **Start with `--stats`** to understand what's in the graph
2. **Use `--file-history`** if the question is about a specific file
3. **Write Cypher queries** for more specific exploration using the schema above
4. **Summarize findings** for the user, highlighting:
   - How similar problems were solved before
   - What tools/files were involved
   - Key decisions or patterns from past interactions
