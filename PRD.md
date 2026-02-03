# agent-memory-graph — Product Requirements Document

**Version:** 0.4
**Author:** Itay Knaan Harpaz
**Date:** February 2026
**Status:** Design Phase

---

## 1. Problem Statement

Current agent memory systems rely on markdown files that the agent is expected to edit to maintain its own memory. This approach fails fundamentally:

- **LLMs cannot maintain documents.** They duplicate entries, lose old context, write inconsistent summaries, or append indefinitely until the file is noise.
- **Unstructured text is not queryable.** You cannot ask "what did the user decide about the database?" without re-reading everything and hoping the answer is there.
- **No temporal awareness.** A decision from six months ago carries equal weight to one from yesterday. There is no decay, no versioning, no expiry.
- **LLM-based maintenance does not scale.** Systems that use LLMs to summarize, consolidate, or curate memory introduce sequential pipelines, compounding costs, and fragile heuristics.
- **The agent has no intrinsic motivation to remember.** It must be prompted to write memories, and it either over-records or under-records.

These problems exist across all current agentic systems — coding agents (Cursor, Windsurf, Claude Projects), trading assistants, research tools, and general AI assistants.

## 2. Core Insight

Don't try to be smart about what to remember. Store raw interactions as a structured graph and let retrieval-time similarity do the work.

The "memory" is not a curated knowledge base. It is a structured interaction log where edges represent the concrete actions that connected inputs to outputs. This eliminates the maintenance problem entirely: no consolidation jobs, no decay scoring, no LLM-driven fact updates. The graph grows append-only, and intelligence lives at retrieval time.

The corollary: if all the intelligence is on the retrieval path, the retrieval path must be expressive. The graph backend must support flexible, multi-hop traversal patterns that can evolve as the retrieval system matures — not just the two or three query shapes we can predict today.

## 3. Product Summary

A Python package that gives any AI agent long-term memory through a knowledge graph. The graph is stored in Neo4j for expressive retrieval and LanceDB for hybrid vector search. Install with pip, connect to a Neo4j instance, and the agent has persistent memory across conversations.

## 4. Design Principles

1. **Minimal setup.** Single pip install. Neo4j connection and optional embedding configuration are the only setup steps.
2. **Graph-native storage.** The memory is a graph. Store it in a graph database. Nodes, edges, hypernodes, and multi-hop traversals are first-class operations, not simulated through SQL joins.
3. **Pluggable embedding.** Ships with a default local model for zero-config use. Accepts any custom embedding function.
4. **Framework-agnostic.** Works with any agent framework or no framework at all.
5. **Read path is smart, write path is dumb.** Writing is fast, append-only, no analysis. Retrieval is where scoring, traversal, and ranking happen — and the graph backend must support that sophistication.

## 5. Goals

### Primary Goals

1. **Zero LLM maintenance.** The graph never requires an LLM call to maintain, update, prune, or reorganize. Small models may be used for zero-shot parallel feature extraction at ingestion (embedding generation) but never for sequential curation.
2. **Domain-agnostic core.** The graph structure works identically for coding agents, trading assistants, research tools, and general conversation. Domain specificity enters only through natural token weighting in the sparse vector.
3. **Evolving graph from computable signals.** The graph changes over time through algorithmic signals derived from its own structure — recency, connectivity, document versioning. No feedback loops depending on unmeasurable signals.
4. **Append-only with minimal batch maintenance.** Write path is fast and parallelizable. Maintenance is limited to deduplication, index rebuilds, and cold storage migration.
5. **Expressive retrieval.** The retrieval path can evolve in sophistication — multi-hop traversals, subgraph extraction, pattern matching — without rewriting query logic from scratch.

### Non-Goals (This Version)

- Shared world knowledge graph across users
- Cross-user learning or anonymized pattern sharing
- LLM-generated summaries or fact extraction at any pipeline stage
- Explicit reinforcement learning on retrieval quality
- Measuring whether a retrieval was "useful" (this signal does not reliably exist)

## 6. Operating Modes

The memory graph operates in one of two modes, determined at initialization. The mode controls how document identifiers are resolved for hypernode canonicalization.

### Local Mode

Initialized for a specific project directory. All document paths are stored as **relative paths** from the project root.

This means the project can be moved, renamed, or cloned and the memory graph remains valid. Two files at the same relative path in the same project are the same document. A file at `src/auth.py` in project A and `src/auth.py` in project B are different documents because they are scoped to different local mode instances.

Suited for: coding agents, project-scoped assistants, any agent tied to a specific workspace.

### Global Mode

Initialized at a user level. All document paths are stored as **absolute paths**.

This means the memory graph spans across projects and directories. A file at `/home/user/project-a/src/auth.py` and `/home/user/project-b/src/auth.py` are two different documents. The same file accessed from different paths is the same document.

Suited for: general-purpose assistants, cross-project memory, trading agents, research tools.

### Path Canonicalization

The hypernode identifier for a document is derived from the canonical path:

- **Local mode:** resolve the path relative to the project root, normalize (remove `../`, resolve symlinks).
- **Global mode:** resolve to absolute path, normalize.

URLs and URIs follow the same pattern — normalized and stored as-is in both modes, since they have no local/global distinction.

## 7. Graph Data Model

### 7.1 Node Types

The graph contains three node types, stored as Neo4j node labels:

| Node Type | Label | Searchable | Description |
|---|---|---|---|
| **User Text** | `:UserText` | Yes — hybrid semantic + keyword via LanceDB | The user's message. The only node type indexed for vector search. |
| **Agent Text** | `:AgentText` | No | The agent's response. Reached only via graph traversal from a matched user text node. |
| **Document Version** | `:DocVersion` | No | A snapshot of a document at a specific point in time. Each observed version is its own node. Reached via graph traversal or hypernode lookup. |

All nodes carry `id`, `created_at`, `conversation_id`, and `turn_index` properties. User and agent text nodes carry a `text` property. Document version nodes carry `content_hash` and optionally `content`.

### 7.2 Hypernodes (Document Identity)

A hypernode is a Neo4j node with the `:Document` label. It represents the identity of a document across all its versions.

Each hypernode holds the document's canonical identifier (relative or absolute path depending on mode, or URL/URI), its current content hash, and document-level metadata. Document version nodes are connected to their hypernode via a `:VERSION_OF` relationship.

A document version node is created only when the agent interacts with the document (reads or writes it) and the content has changed since the last observed version. Intermediate external changes that the agent never witnesses do not produce version nodes. This keeps the node count proportional to agent interactions, not to file change frequency.

A hypernode with a single version node is effectively a static document. A hypernode with multiple version nodes is dynamic. No explicit classification is needed — the structure reveals it.

### 7.3 Edges = Procedures

Every relationship is a **procedure** — a concrete tool invocation that occurred during an interaction. Relationships are stored without direction constraint in the query layer (Neo4j stores directed edges, but Cypher queries traverse them bidirectionally).

- Relationship type maps 1:1 with tool name (e.g., `:READ`, `:EDIT`, `:BASH`, `:WEB_SEARCH`)
- Relationship properties hold the tool arguments, conversation_id, turn_index, and timestamp
- No abstract or typed edges — every relationship is a real action that happened
- Initial user message connects to the agent response via a `:MESSAGE` relationship

Bidirectional traversal allows flexible queries: from a user query forward to the documents it touched, or from a document version backward to all conversations that interacted with it. Temporal ordering is preserved by timestamps on nodes and relationships, not by relationship direction.

### 7.4 Graph Topology

Each conversation forms a subgraph. Within a conversation, nodes from the same turn are connected through procedure relationships.

Conversations connect through **shared hypernodes**. When two conversations interact with the same document, their procedure relationships point to version nodes that are connected to the same `:Document` hypernode via `:VERSION_OF`. This creates natural cross-conversation bridges.

The version gap between two version nodes under the same hypernode carries information. A past interaction pointing to v3 when the latest version is v13 tells you the file changed significantly since the agent last saw it. A large gap where the agent was absent (no version nodes created) indicates external changes the agent didn't witness.

Retrieval **never returns nodes from the current conversation** — that context is already in the prompt.

### 7.5 Discovery Paths

There are two ways to find relevant past context:

1. **Similarity search.** Hybrid dense + sparse vector search over user text nodes in LanceDB. Finds past interactions where the user asked something semantically or lexically similar. Matched node IDs are then used to anchor graph traversals in Neo4j.

2. **Hypernode lookup.** The current conversation touches a file → match its `:Document` hypernode in Neo4j → traverse `:VERSION_OF` to find all version nodes → traverse procedure relationships to find all conversations that interacted with any version. Discovers relevant past context even when the user's question is phrased completely differently, as long as it involves the same files.

The graph database makes both paths natural. The second path in particular benefits from Cypher's ability to express multi-hop traversals concisely.

## 8. Document Versioning

### 8.1 Read-Wrapped Versioning

Every Read tool invocation is wrapped with a versioning layer:

1. Resolve the document path to a canonical identifier (relative or absolute depending on mode).
2. Match the `:Document` hypernode by canonical identifier in Neo4j.
3. If no hypernode exists, create one and create the first `:DocVersion` node connected via `:VERSION_OF`.
4. If the hypernode exists, compare the current content hash against the latest version node's hash.
5. Same hash — return the existing version node. No new node created.
6. Different hash — create a new `:DocVersion` node under the hypernode. Previous versions preserved.

### 8.2 Provenance Tracking

Each version node records whether the change was agent-caused or external:

- **Agent-caused:** The version node has an inbound procedure relationship (e.g., `:EDIT`). Full provenance is available — what changed, why, and in response to which user query.
- **External:** The version node was created because a subsequent `:READ` detected a hash change, but no agent procedure caused the change. This signals that something outside the agent's knowledge modified the file.

### 8.3 Computable Signals from Versioning

| Signal | Computation | Retrieval Impact |
|---|---|---|
| **Staleness** | Number of version nodes between a past interaction's version and the latest version under the same hypernode | Higher staleness → lower confidence in past context |
| **Volatility** | Total version node count under a hypernode / time span between first and latest version | High volatility → context expires faster, but the document is likely important |
| **Familiarity** | Total number of conversation subgraphs with edges to any version node under a hypernode | More interactions → deeper accumulated understanding |

All signals are computable from graph structure with Cypher queries. No LLM analysis required.

## 9. Retrieval System

### 9.1 Hybrid Vector Search

User text nodes are indexed in LanceDB with both dense and sparse representations:

- **Dense vector** (semantic embedding) captures meaning — "fix the login bug" matches "resolve authentication issue"
- **Sparse / BM25** (LanceDB built-in full-text search) captures exact token matches — naturally weights domain terms like `TSLA`, `auth.py`, `iron condor`, `BigQuery` without requiring entity extraction or controlled vocabularies

Score fusion is handled natively by LanceDB. No explicit entity extraction pipeline needed.

### 9.2 Retrieval Flow

1. **Embed** the new user query into dense + sparse representations.
2. **Search** via hybrid ANN in LanceDB over user text nodes, excluding the current conversation. Retrieve top-K similar past user queries with their node IDs.
3. **Augment** with hypernode lookup in Neo4j: identify documents the current conversation touches, traverse `:VERSION_OF` and procedure relationships to find other conversations that interacted with the same documents.
4. **Traverse** in Neo4j from each matched user text node, walk procedure relationships bidirectionally to collect: agent response, document version nodes referenced, procedure metadata (relationship type + properties).
5. **Score** retrieved subgraphs by combining search similarity, recency decay, document staleness penalty, and document familiarity bonus.
6. **Pack** results into the context token budget.
7. **Inject** structured context block into the prompt.

Steps 3 and 4 are where the graph database earns its keep. Multi-hop traversals, subgraph extraction around a match point, and pattern-based queries across conversations are native Cypher operations rather than hand-assembled SQL joins.

### 9.3 Scoring

| Factor | Source | Description |
|---|---|---|
| **Search score** | LanceDB hybrid fusion | Semantic + keyword match quality |
| **Recency decay** | Node timestamp | Exponential decay favoring recent interactions |
| **Staleness penalty** | Hypernode version chain (Cypher) | Penalize results whose connected documents have changed significantly |
| **Familiarity bonus** | Hypernode interaction count (Cypher) | Boost results involving documents the user frequently works with |

All parameters configurable with sensible defaults.

### 9.4 Injected Context Format

Retrieved context is structured so the LLM can see why each piece of context was surfaced:

- The past user query (relevance anchor — explains why this was retrieved)
- Procedures used (relationship type + arguments — shows how the problem was approached)
- Agent response (the actual past answer, truncated if needed)
- Document references with staleness information (which files were involved, and whether they've changed since)

## 10. API Design

The API has three layers:

### Recording

A conversation/turn-based interface for writing interactions to the graph. The caller registers user messages, agent responses, tool invocations (procedures), and document references. Turns are committed atomically as Neo4j transactions. A context manager pattern is available for convenience.

### Retrieval

A query interface that accepts a text query and returns ranked results. Each result contains the matched past user query, agent response, procedures used, document version nodes referenced (with staleness info), and scoring metadata. A formatted output option returns a string ready for direct prompt injection with a configurable token budget.

Two discovery paths are available: similarity search over user text nodes (LanceDB → Neo4j traversal), and hypernode lookup for document-based discovery (Neo4j only).

### Document Management

An interface for reading documents with automatic versioning (read-wrapper pattern), querying document history via hypernodes, and checking staleness of past interactions against current document state. All queries run against Neo4j.

## 11. Storage

| Component | Technology | Purpose |
|---|---|---|
| Graph | Neo4j | Nodes, relationships, hypernodes, version chains, multi-hop traversal |
| Vector index | LanceDB | Hybrid dense + BM25 search over user text nodes |

### Neo4j Deployment

The package connects to a Neo4j instance via the Bolt protocol. Deployment options:

- **Neo4j Desktop** — free, local, single-click install for individual developers
- **Neo4j Community Edition** — free, self-hosted, Docker or bare metal
- **Neo4j Aura Free** — managed cloud, no infrastructure to maintain
- **Neo4j Aura or Enterprise** — for production / team use

The package provides a connection string and handles schema initialization (constraints, indexes) on first connection.

### LanceDB Storage

LanceDB stores the vector index as a local directory of Lance files alongside the project or at the user-level location depending on operating mode:

- **Local mode:** `.amg/vectors/` inside the project root
- **Global mode:** `~/.amg/vectors/` or user-specified path

### Portability

The Neo4j graph can be exported and imported via Neo4j's dump/load tooling. The LanceDB directory is portable by copying. Together they form the complete memory, though portability requires both components.

## 12. Maintenance

The graph is append-only. No edge weight updates, no LLM summarization, no entity extraction, no reinforcement loops.

Available maintenance operations (all user-triggered or scheduled):

| Operation | Purpose |
|---|---|
| **Deduplicate documents** | Merge document version nodes with identical content hashes across hypernodes (Cypher query) |
| **Archive old conversations** | Detach and archive conversation subgraphs with no recent retrieval hits |
| **Rebuild vector index** | Optimize LanceDB index after large batch imports |
| **Stats** | Report node/relationship/hypernode counts and index sizes |

## 13. Dependencies

**Core:** Neo4j Python driver, LanceDB, PyArrow (required by LanceDB).

**Optional:** Sentence-transformers for local embedding, or OpenAI SDK for API-based embedding. The embedding function is pluggable.

**Infrastructure:** A Neo4j instance (local, Docker, or cloud). Neo4j Desktop and Aura Free provide zero-cost options for individual use.

## 14. Open Questions

1. **Context budget allocation.** When retrieval surfaces many relevant results, how to truncate for the token budget? Top-K by score with full content, truncated responses, or a two-pass approach?
2. **Embedding model selection.** General-purpose vs domain-fine-tuned. Trade-off between zero-config simplicity and retrieval quality.
3. **Cold start.** New users have an empty graph. Bootstrap from onboarding, import from existing tools, or build organically?
4. **Large document storage.** Full document content as Neo4j node properties, or store on disk with Neo4j holding only path + hash?
5. **Multi-agent concurrency.** Multiple agents writing to the same Neo4j instance. Neo4j handles concurrent transactions natively, but write coordination at the application level may still be needed for turn atomicity.
6. **Hypernode lookup scaling.** Popular files touched by many conversations could return large result sets. May need capping or scoring at the Cypher level.
7. **Local-global bridging.** A user may want to reference global memory from within a local project, or promote local memory to global. Deferred but worth considering for future versions.
8. **Turn boundaries.** In streaming agent scenarios, when does a turn start and end? Explicit commit works for structured use, but less structured integrations may need auto-flush.
9. **Neo4j version targeting.** Which minimum Neo4j version to support? Community Edition vs Enterprise feature differences that matter for this use case.

## 15. Why Neo4j Over SQLite

The system's design thesis is that all intelligence lives on the retrieval path. This has implications for the storage backend:

**The retrieval path must be expressive and evolvable.** Today we can predict two or three query patterns. But if the system works, retrieval gets more sophisticated over time. Multi-hop context chains (user query → document → hypernode → other versions → other conversations), subgraph extraction around a match point, pattern-based discovery across conversations — these are queries we'll want to write and iterate on. In Cypher, each is a single expression. In SQL, each is a growing thicket of self-joins maintained in application code.

**The data model is a graph.** Nodes with labels, typed relationships with properties, hypernodes grouping version chains, cross-conversation bridges through shared documents. Modeling this in a relational schema means maintaining the graph abstraction in Python rather than in the database. The impedance mismatch works against us on every query.

**Operational overhead is manageable.** Neo4j Desktop and Aura Free provide zero-cost, low-friction options. Docker Compose for self-hosted. The setup is one connection string, not a full infrastructure project. The trade-off is real — this is no longer a single-file portable system — but the retrieval flexibility justifies it for a system whose entire value proposition is retrieval quality.

## 16. Success Criteria

- Install with pip and have working memory within 10 minutes (including Neo4j setup)
- Write latency under 50ms per turn (excluding embedding)
- Retrieval latency under 200ms at 10K+ nodes
- Zero LLM calls in any code path
- Works across domains (coding, trading, general) with no domain-specific configuration
- Local mode works correctly when project is moved or cloned
- Global mode correctly deduplicates documents accessed from the same absolute path across conversations
- Retrieval queries are expressible in Cypher without application-level graph simulation