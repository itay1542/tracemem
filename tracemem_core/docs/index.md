# TraceMem Core Documentation

TraceMem is a conversation memory system that stores and retrieves context from agent interactions using a hybrid graph + vector architecture.

## Overview

TraceMem captures:
- **User messages** - Questions and requests from users
- **Agent responses** - LLM-generated replies
- **Tool invocations** - Tools used during conversations (file reads, API calls, etc.)
- **Resources** - Files and external resources accessed during conversations

This data is stored in:
- **Neo4j** - Graph database for relationships and traversal
- **LanceDB** - Vector database for semantic similarity search

## Quick Start

```python
from tracemem_core import TraceMem, Message, RetrievalConfig

# Initialize TraceMem
async with TraceMem() as tm:
    # Add messages to a conversation
    await tm.add_message("conv-1", Message(role="user", content="How do I authenticate?"))
    await tm.add_message("conv-1", Message(role="assistant", content="Use JWT tokens..."))

    # Search for similar past interactions
    config = RetrievalConfig(limit=5, include_context=True)
    results = await tm.search("authentication help", config=config)

    for r in results:
        print(r)  # Result(a1b2c3d4, score=0.850, conv=conv-1, text='How do I authenticate?', context=yes)
```

## Documentation

- [Retrieval Guide](./retrieval.md) - Search and context retrieval
- [Configuration](./configuration.md) - TraceMem configuration options
- [Architecture](./architecture.md) - System architecture and data model

## Installation

```bash
uv add tracemem-core
```

## Requirements

- Python 3.12+
- Neo4j 5.x
- OpenAI API key (for embeddings)
