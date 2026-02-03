# Retrieval Guide

TraceMem provides hybrid retrieval combining vector similarity search with graph traversal to find relevant past interactions.

## RetrievalConfig

The `RetrievalConfig` model provides fine-grained control over search and retrieval operations.

```python
from tracemem_core import RetrievalConfig

config = RetrievalConfig(
    # Search parameters
    limit=10,                    # Max results (1-100, default: 10)
    include_context=True,        # Expand with full context (default: True)

    # Hybrid search tuning
    vector_weight=0.5,           # 0.0=pure text, 1.0=pure vector (default: 0.5)

    # Expansion options
    expand_tool_uses=True,       # Include tool invocations (default: True)
    expand_resources=True,       # Include resource info (default: True)

    # Sorting (for resource queries)
    sort_by="created_at",        # "created_at" or "last_accessed_at"
    sort_order="desc",           # "asc" or "desc"

    # Filtering
    exclude_conversation_id="conv-123",  # Exclude specific conversation

    # Trajectory
    trajectory_max_depth=100,    # Max MESSAGE hops for trajectory (1-500, default: 100)
)
```

## Retrieval Strategy

The `RetrievalStrategy` protocol defines the retrieval interface. `HybridRetrievalStrategy` is the default implementation combining vector search and graph traversal.

TraceMem exposes retrieval methods directly on the instance:

```python
from tracemem_core import TraceMem, RetrievalConfig

async with TraceMem() as tm:
    # Search with config
    config = RetrievalConfig(limit=5, include_context=True)
    results = await tm.search("authentication error", config=config)

    for r in results:
        print(r)  # Result(a1b2c3d4, score=0.850, conv=abc, text='How do I auth...', context=yes)
        if r.context and r.context.agent_text:
            print(f"Answer: {r.context.agent_text.text}")
```

For advanced usage, the underlying `HybridRetrievalStrategy` is available via `tm.retrieval`.

### Search with Vector Weight Tuning

Control the balance between semantic similarity (vector) and keyword matching (text):

```python
# Balanced hybrid (default behavior)
config = RetrievalConfig(vector_weight=0.5)

# Pure keyword/BM25 search
config = RetrievalConfig(vector_weight=0.0)

# Pure vector/semantic search
config = RetrievalConfig(vector_weight=1.0)

# Balanced hybrid
config = RetrievalConfig(vector_weight=0.5)
```

### Reranker Selection

TraceMem ships with built-in reranker presets that require no API keys:

```python
# Default: Reciprocal Rank Fusion
async with TraceMem() as tm:
    results = await tm.search("query")  # Uses "rrf" reranker

# Linear combination reranker
from tracemem_core import TraceMemConfig
config = TraceMemConfig(reranker="linear")
async with TraceMem(config=config) as tm:
    results = await tm.search("query")

# Or override at construction time
async with TraceMem(reranker="linear") as tm:
    results = await tm.search("query")
```

Available presets: `"rrf"` (RRFReranker), `"linear"` (LinearCombinationReranker).

For rerankers requiring API keys (Cohere, Jina, etc.), pass an instance directly:

```python
from lancedb.rerankers import CohereReranker

async with TraceMem(reranker=CohereReranker(api_key="...")) as tm:
    results = await tm.search("query")
```

## Resource History

Find all conversations that accessed a specific resource (file, URL, etc.).

### Get Conversations for Resource

```python
config = RetrievalConfig(
    limit=20,
    sort_by="last_accessed_at",
    sort_order="desc",
)
conversations = await tm.get_conversations_for_resource(
    "file:///src/auth.py",
    config=config,
)

for conv in conversations:
    print(conv)  # ConvRef(a1b2c3d4, conv=abc, user='How do I fix auth?')
    print(f"  Agent said: {conv.agent_text}")
    print(f"  When: {conv.created_at}")
```

## Get Context for Node

Retrieve full context for a specific UserText node:

```python
from uuid import UUID

context = await tm.get_context(UUID("abc123..."))

print(context)  # Context(user[a1b2c3d4]='How do I auth...', agent[e5f6g7h8]='Use JWT tokens...', tools=[Read(auth.py)])

for tool_use in context.tool_uses:
    print(f"Tool: {tool_use}")  # Read(file:///src/auth.py)
    print(f"  Properties: {tool_use.properties}")
```

## Get Trajectory

Retrieve the full trajectory from a UserText node — the user message, all subsequent agent responses (with tool uses), and the follow-up user message:

```python
from uuid import UUID

config = RetrievalConfig(trajectory_max_depth=200)
trajectory = await tm.get_trajectory(UUID("abc123..."), config=config)

print(trajectory)
# Trajectory(4 steps):
#   Step(a1b2c3d4 UserText: 'Fix the auth bug')
#   Step(e5f6a7b8 AgentText: '' tools=[read_file, edit_file])
#   Step(c9d0e1f2 AgentText: 'I fixed the bug by updating token validation')
#   Step(g3h4i5j6 UserText: 'Now update the tests')

for step in trajectory.steps:
    print(f"[{step.node_type}] {step.text[:80]}")
    for tool in step.tool_uses:
        print(f"  Tool: {tool}")
```

## Result Models

### RetrievalResult

```python
class RetrievalResult:
    node_id: UUID           # UserText node ID
    text: str               # User message text
    conversation_id: str    # Conversation identifier
    score: float            # Relevance score
    context: ContextResult | None  # Full context if requested
```

### ContextResult

```python
class ContextResult:
    user_text: UserTextInfo | None     # User message info
    agent_text: AgentTextInfo | None   # Agent response info
    tool_uses: list[ToolUse]           # Tool invocations
```

### ConversationReference

```python
class ConversationReference:
    conversation_id: str         # Conversation identifier
    user_text_id: str           # UserText node ID
    user_text: str              # User message text
    agent_text: str | None      # Agent response text
    created_at: datetime | None # When created
```

### TrajectoryStep

```python
class TrajectoryStep:
    node_id: str                     # Node ID
    node_type: "UserText" | "AgentText"  # Node label
    text: str                        # Message text
    conversation_id: str             # Conversation identifier
    tool_uses: list[ToolUse]         # Tool invocations (AgentText only)
```

### TrajectoryResult

```python
class TrajectoryResult:
    steps: list[TrajectoryStep]      # Ordered steps in the trajectory
```
