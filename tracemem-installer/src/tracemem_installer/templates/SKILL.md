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

TraceMem stores coding interactions as a directed graph in Neo4j:

```
UserText --MESSAGE--> AgentText --MESSAGE--> AgentText --MESSAGE--> ... --MESSAGE--> UserText (next turn)
                          |                      |
                        READ/WRITE/EDIT        BASH/GREP
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
| **AgentText** | `id`, `text`, `tool_uses` (JSON), `conversation_id`, `turn_index`, `created_at`, `last_accessed_at` | An agent response or tool invocation. Most AgentText nodes have empty `text` and carry `tool_uses`; the final AgentText in a turn has the actual response text. |
| **Resource** | `id`, `uri`, `current_content_hash`, `conversation_id`, `created_at`, `last_accessed_at` | A file or external resource (deduplicated by URI) |
| **ResourceVersion** | `id`, `uri`, `content_hash`, `conversation_id`, `created_at`, `last_accessed_at` | A snapshot of a resource at a point in time |

### Relationship Types

| Relationship | From → To | Description |
|-------------|-----------|-------------|
| **MESSAGE** | UserText → AgentText, AgentText → AgentText, AgentText → UserText | Conversation flow in chronological order |
| **READ** | AgentText → ResourceVersion | Agent read a file |
| **WRITE** | AgentText → ResourceVersion | Agent wrote/created a file |
| **EDIT** | AgentText → ResourceVersion | Agent edited a file |
| **BASH** | AgentText → ResourceVersion | Agent ran a bash command |
| **GREP** | AgentText → ResourceVersion | Agent searched in files |
| **VERSION_OF** | ResourceVersion → Resource | Links a snapshot to its canonical resource |

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
  RETURN u.text AS question, left(a.text, 200) AS answer
  ORDER BY u.created_at DESC LIMIT 5
"
```

Find what files were accessed during a conversation:
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (u:UserText)-[:MESSAGE*]->(a:AgentText)-[r]->(v:ResourceVersion)
  WHERE type(r) IN ['READ', 'WRITE', 'EDIT']
  RETURN u.text AS question, type(r) AS action, v.uri AS file
  ORDER BY u.created_at DESC LIMIT 10
"
```

Find all conversations that touched a specific file:
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (res:Resource {uri: 'file://tracemem_core/src/tracemem_core/tracemem.py'})
        <-[:VERSION_OF]-(v:ResourceVersion)<-[r]-(a:AgentText)
  MATCH (u:UserText)-[:MESSAGE*]->(a)
  WHERE type(r) <> 'VERSION_OF'
  RETURN DISTINCT u.conversation_id AS conv, u.text AS question, type(r) AS action
  ORDER BY u.created_at DESC LIMIT 10
"
```

Find the most frequently accessed files:
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (v:ResourceVersion)<-[r]-(a:AgentText)
  WHERE type(r) IN ['READ', 'WRITE', 'EDIT']
  RETURN v.uri AS file, count(*) AS access_count, collect(DISTINCT type(r)) AS actions
  ORDER BY access_count DESC LIMIT 10
"
```

Get full trajectory for a user turn (all tool uses until next user message):
```bash
uv run .claude/skills/tracemem/query_graph.py "
  MATCH (u:UserText {id: 'NODE_ID_HERE'})-[:MESSAGE*]->(n)
  WHERE n.conversation_id = u.conversation_id
  RETURN labels(n)[0] AS type, n.text AS text, n.tool_uses AS tools
  ORDER BY n.created_at ASC
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
