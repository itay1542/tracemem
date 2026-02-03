"""TraceMem - Knowledge graph memory system for AI agents.

Usage with adapter:
    ```python
    from tracemem_core import TraceMem
    from tracemem_core.adapters.langchain import LangChainAdapter
    from langchain_core.messages import HumanMessage, AIMessage

    adapter = LangChainAdapter()

    async with TraceMem() as tm:
        # Convert and import
        messages = adapter.convert([
            HumanMessage(content="Read auth.py"),
            AIMessage(content="Here's the file..."),
        ])
        await tm.import_trace("conv-1", messages)
    ```

Direct usage:
    ```python
    from tracemem_core import TraceMem, Message

    async with TraceMem() as tm:
        await tm.add_message("conv-1", Message(role="user", content="Hello"))
    ```
"""

import hashlib
from typing import Any
from uuid import UUID

from tracemem_core.config import TraceMemConfig
from tracemem_core.embedders.openai import OpenAIEmbedder
from tracemem_core.embedders.protocol import Embedder
from tracemem_core.extractors import DefaultResourceExtractor, ResourceExtractor
from tracemem_core.messages import Message, ToolCall
from tracemem_core.models.edges import Relationship, VersionOf
from tracemem_core.models.nodes import (
    AgentText,
    Resource,
    ResourceVersion,
    ToolUseRecord,
    UserText,
)
from tracemem_core.retrieval.hybrid import HybridRetrievalStrategy
from tracemem_core.retrieval.protocol import RetrievalStrategy
from tracemem_core.retrieval.results import (
    ConversationReference,
    ContextResult,
    RetrievalConfig,
    RetrievalResult,
    TrajectoryResult,
)
from tracemem_core.storage.graph.kuzu_store import KuzuGraphStore
from tracemem_core.storage.vector import LanceDBVectorStore


class TraceMem:
    """Knowledge graph memory system for AI agents.

    TraceMem stores agent interactions as a knowledge graph with
    vector indexing via LanceDB. It provides:

    - **Structured memory**: User messages, agent responses, and tool uses
      are stored as graph nodes with relationships.
    - **Resource versioning**: Files and documents are tracked across versions,
      enabling detection of changes and staleness.

    URI canonicalization is handled by the resource extractor, not by TraceMem
    itself. Pass a ``root`` to your extractor to get relative file URIs.

    Example with adapter:
        ```python
        from tracemem_core import TraceMem
        from tracemem_core.adapters.langchain import LangChainAdapter
        from langchain_core.messages import HumanMessage, AIMessage

        adapter = LangChainAdapter()

        async with TraceMem() as tm:
            messages = adapter.convert([
                HumanMessage(content="Read auth.py"),
                AIMessage(content="Here's the file...", tool_calls=[...]),
            ])
            await tm.import_trace("conv-1", messages)
        ```

    Example direct usage:
        ```python
        from tracemem_core import TraceMem, Message, ToolCall

        async with TraceMem() as tm:
            await tm.add_message("conv-1", Message(role="user", content="Hello"))
            await tm.add_message("conv-1", Message(
                role="assistant",
                content="I'll read that file",
                tool_calls=[ToolCall(id="c1", name="read_file", args={"path": "auth.py"})]
            ))
        ```
    """

    def __init__(
        self,
        config: TraceMemConfig | None = None,
        embedder: Embedder | None = None,
        resource_extractor: ResourceExtractor | None = None,
        reranker: str | Any | None = None,
    ) -> None:
        """Initialize TraceMem.

        Args:
            config: Configuration settings. Uses defaults if not provided.
            embedder: Custom embedder implementation. Uses OpenAI embedder if
                not provided (requires OPENAI_API_KEY or config.openai_api_key).
            resource_extractor: Custom resource extractor for tool calls.
                Uses DefaultResourceExtractor if not provided.
            reranker: Reranker strategy. Can be a string key ("rrf", "linear")
                or a reranker instance. If None, uses config.reranker.
        """
        self._config = config or TraceMemConfig()
        self._resource_extractor = resource_extractor or DefaultResourceExtractor()

        # Initialize embedder
        if embedder:
            self._embedder = embedder
        else:
            self._embedder = OpenAIEmbedder(
                model=self._config.embedding_model,
                dimensions=self._config.embedding_dimensions,
                api_key=self._config.openai_api_key,
            )

        # Resolve reranker: explicit param > config string
        resolved_reranker = reranker if reranker is not None else self._config.reranker

        # Initialize stores
        if self._config.graph_store == "neo4j":
            from tracemem_core.storage.graph.neo import Neo4jGraphStore

            self._graph_store = Neo4jGraphStore(
                uri=self._config.neo4j_uri,
                user=self._config.neo4j_user,
                password=self._config.neo4j_password,
                database=self._config.neo4j_database,
                namespace=self._config.namespace,
            )
        else:
            self._graph_store = KuzuGraphStore(
                db_path=self._config.get_graph_path(),
            )

        self._vector_store = LanceDBVectorStore(
            path=self._config.get_vector_path(),
            embedding_dimensions=self._config.embedding_dimensions,
            reranker=resolved_reranker,
        )

        # Lazy-initialized retrieval strategy
        self._retrieval: HybridRetrievalStrategy | None = None

        # State for message processing
        self._tool_results: dict[str, str] = {}

    async def __aenter__(self) -> "TraceMem":
        """Async context manager entry."""
        await self._graph_store.connect()
        await self._graph_store.initialize_schema()
        await self._vector_store.connect()
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit - closes all connections."""
        await self._graph_store.close()
        await self._vector_store.close()

    @property
    def retrieval(self) -> RetrievalStrategy:
        """Lazy-constructed retrieval strategy.

        Returns a RetrievalStrategy backed by this TraceMem instance's
        graph store, vector store, and embedder.
        """
        if self._retrieval is None:
            self._retrieval = HybridRetrievalStrategy(
                graph_store=self._graph_store,
                vector_store=self._vector_store,
                embedder=self._embedder,
            )
        return self._retrieval

    async def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant past interactions.

        Args:
            query: Search query text.
            config: Optional RetrievalConfig. Falls back to config.retrieval default.

        Returns:
            List of RetrievalResult ordered by relevance.
        """
        return await self.retrieval.search(query, config=config or self._config.retrieval)

    async def get_context(self, node_id: UUID) -> ContextResult:
        """Get full context for a UserText node.

        Args:
            node_id: UUID of the UserText node.

        Returns:
            ContextResult with user text, agent text, and tool uses.
        """
        return await self.retrieval.get_context(node_id)

    async def get_conversations_for_resource(
        self,
        uri: str,
        config: RetrievalConfig | None = None,
    ) -> list[ConversationReference]:
        """Find all conversations that accessed a resource.

        Args:
            uri: Canonical URI of the resource.
            config: Optional RetrievalConfig. Falls back to config.retrieval default.

        Returns:
            List of ConversationReference with sorting applied.
        """
        return await self.retrieval.get_conversations_for_resource(
            uri, config=config or self._config.retrieval,
        )

    async def get_trajectory(
        self,
        node_id: UUID,
        config: RetrievalConfig | None = None,
    ) -> TrajectoryResult:
        """Get trajectory from a UserText node.

        Args:
            node_id: UUID of the starting UserText node.
            config: Optional RetrievalConfig. Falls back to config.retrieval default.

        Returns:
            TrajectoryResult with all steps in chronological order.
        """
        return await self.retrieval.get_trajectory(
            node_id, config=config or self._config.retrieval,
        )

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    async def import_trace(
        self,
        conversation_id: str,
        messages: list[Message],
    ) -> dict[str, UUID]:
        """Import a conversation trace from a list of Messages.

        This is the primary method for importing conversation history.
        Use an adapter to convert framework-specific messages first.

        Args:
            conversation_id: Unique identifier for the conversation.
            messages: List of Message objects to import.

        Returns:
            Dict mapping created node types to their UUIDs.

        Example:
            ```python
            from tracemem_core.adapters.langchain import LangChainAdapter

            adapter = LangChainAdapter()
            messages = adapter.convert(langchain_messages)
            await tm.import_trace("conv-1", messages)
            ```
        """
        created: dict[str, UUID] = {}

        # First pass: collect tool results
        self._tool_results.clear()
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id:
                self._tool_results[msg.tool_call_id] = msg.content

        # Second pass: process messages
        for msg in messages:
            result = await self.add_message(conversation_id, msg)
            created.update(result)

        return created

    async def add_message(
        self,
        conversation_id: str,
        message: Message,
    ) -> dict[str, UUID]:
        """Add a single message to the knowledge graph.

        Args:
            conversation_id: Unique identifier for the conversation.
            message: The Message to add.

        Returns:
            Dict mapping created node types to their UUIDs.

        Example:
            ```python
            await tm.add_message("conv-1", Message(role="user", content="Hello"))
            ```
        """
        created: dict[str, UUID] = {}

        if message.role == "user":
            user_text = await self._add_user_message(conversation_id, message)
            created["user_text"] = user_text.id

        elif message.role == "assistant":
            agent_text, tool_ids = await self._add_assistant_message(
                conversation_id, message
            )
            created["agent_text"] = agent_text.id
            created.update(tool_ids)

        elif message.role == "tool" and message.tool_call_id:
            # Store tool result for later use when processing assistant messages
            self._tool_results[message.tool_call_id] = message.content

        # System messages are not stored in the graph

        return created

    async def _add_user_message(
        self, conversation_id: str, message: Message
    ) -> UserText:
        """Add a user message to the graph.

        User messages start a new turn. If there's a previous turn, creates an edge
        from the last agent in that turn to this user text.
        """
        # Get max turn and increment for new user message
        max_turn = await self._graph_store.get_max_turn_index(conversation_id)
        turn_index = max_turn + 1

        # Link from previous turn's last agent (if exists)
        last_agent = await self._graph_store.get_last_agent_text(conversation_id)

        user_text = UserText(
            text=message.content,
            conversation_id=conversation_id,
            turn_index=turn_index,
        )
        await self._graph_store.create_node(user_text)

        # Link from previous agent message if exists (maintains conversation chain)
        if last_agent:
            edge = Relationship(
                source_id=last_agent.id,
                target_id=user_text.id,
                conversation_id=conversation_id,
            )
            await self._graph_store.create_edge(edge)

        vector = await self._embedder.embed(user_text.text)
        await self._vector_store.add(
            node_id=user_text.id,
            text=user_text.text,
            vector=vector,
            conversation_id=conversation_id,
        )

        return user_text

    async def _add_assistant_message(
        self, conversation_id: str, message: Message
    ) -> tuple[AgentText, dict[str, UUID]]:
        """Add an assistant message to the graph.

        Assistant messages stay in the same turn as the most recent user message.
        Links from the most recent node in the current turn to maintain continuity.
        """
        created: dict[str, UUID] = {}

        # Get current turn (same as last user message)
        max_turn = await self._graph_store.get_max_turn_index(conversation_id)
        turn_index = max(0, max_turn)  # Use 0 if no turns exist

        # Get last node in this turn (could be UserText or AgentText)
        last_node = await self._graph_store.get_last_node_in_turn(
            conversation_id, turn_index
        )

        # Convert message tool_calls to ToolUseRecord
        tool_uses = [
            ToolUseRecord(id=tc.id, name=tc.name, args=tc.args)
            for tc in message.tool_calls
        ]

        agent_text = AgentText(
            text=message.content,
            conversation_id=conversation_id,
            turn_index=turn_index,
            tool_uses=tool_uses,
        )
        await self._graph_store.create_node(agent_text)

        # Link from previous node in turn (could be UserText or AgentText for tool flows)
        if last_node:
            edge = Relationship(
                source_id=last_node.id,
                target_id=agent_text.id,
                conversation_id=conversation_id,
            )
            await self._graph_store.create_edge(edge)

        # Process tool calls
        for tool_call in message.tool_calls:
            tool_ids = await self._process_tool_call(
                agent_text, tool_call, conversation_id
            )
            created.update(tool_ids)

        return agent_text, created

    async def _process_tool_call(
        self,
        agent_text: AgentText,
        tool_call: ToolCall,
        conversation_id: str,
    ) -> dict[str, UUID]:
        """Process a tool call and create resource nodes if applicable."""
        created: dict[str, UUID] = {}

        # Extract resource URI from tool call args
        resource_uri = self._resource_extractor.extract(tool_call.name, tool_call.args)

        if not resource_uri:
            return created

        # Get content hash from tool result if available
        content_hash: str | None = None
        if tool_call.id in self._tool_results:
            content = self._tool_results[tool_call.id]
            content_hash = self._compute_content_hash(content)

        if not content_hash:
            return created

        # URI is already canonicalized by the extractor
        canonical_uri = resource_uri

        # Get or create Resource hypernode
        existing_resource = await self._graph_store.get_resource_by_uri(canonical_uri)

        if existing_resource:
            resource = existing_resource
            # Check if content has changed
            if existing_resource.current_content_hash != content_hash:
                # Create new version
                version = ResourceVersion(
                    content_hash=content_hash,
                    uri=canonical_uri,
                    conversation_id=conversation_id,
                )
                await self._graph_store.create_node(version)
                created[f"resource_version_{canonical_uri}"] = version.id

                # Update resource hash
                await self._graph_store.update_resource_hash(
                    canonical_uri, content_hash
                )

                # Create VERSION_OF edge
                version_edge = VersionOf(
                    version_id=version.id,
                    resource_id=resource.id,
                )
                await self._graph_store.create_edge(version_edge)

                # Create tool relationship
                tool_edge = Relationship(
                    source_id=agent_text.id,
                    target_id=version.id,
                    relationship_type=tool_call.name.upper(),
                    conversation_id=conversation_id,
                    properties=tool_call.args,
                )
                await self._graph_store.create_edge(tool_edge)
            else:
                # Same content - still create tool relationship to existing version
                existing_version = await self._graph_store.get_resource_version_by_hash(
                    canonical_uri, content_hash
                )
                if existing_version:
                    tool_edge = Relationship(
                        source_id=agent_text.id,
                        target_id=existing_version.id,
                        relationship_type=tool_call.name.upper(),
                        conversation_id=conversation_id,
                        properties=tool_call.args,
                    )
                    await self._graph_store.create_edge(tool_edge)
        else:
            # Create new resource and version
            resource = Resource(
                uri=canonical_uri,
                current_content_hash=content_hash,
                conversation_id=conversation_id,
            )
            resource = await self._graph_store.create_node(resource)
            created[f"resource_{canonical_uri}"] = resource.id

            version = ResourceVersion(
                content_hash=content_hash,
                uri=canonical_uri,
                conversation_id=conversation_id,
            )
            await self._graph_store.create_node(version)
            created[f"resource_version_{canonical_uri}"] = version.id

            # Create VERSION_OF edge
            version_edge = VersionOf(
                version_id=version.id,
                resource_id=resource.id,
            )
            await self._graph_store.create_edge(version_edge)

            # Create tool relationship
            tool_edge = Relationship(
                source_id=agent_text.id,
                target_id=version.id,
                relationship_type=tool_call.name.upper(),
                conversation_id=conversation_id,
                properties=tool_call.args,
            )
            await self._graph_store.create_edge(tool_edge)

        return created
