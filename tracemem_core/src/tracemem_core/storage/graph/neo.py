import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from neo4j import AsyncGraphDatabase, AsyncDriver

from tracemem_core.models.edges import EdgeBase, Relationship, VersionOf
from tracemem_core.models.nodes import (
    AgentText,
    NodeBase,
    Resource,
    ResourceVersion,
    ToolUseRecord,
    UserText,
)
from tracemem_core.retrieval.results import (
    AgentTextInfo,
    ContextResult,
    ConversationReference,
    ResourceInfo,
    ResourceVersionInfo,
    ToolUse,
    UserTextInfo,
)

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Neo4j implementation of GraphStore."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        namespace: str | None = None,
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._namespace = namespace
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Connect to the Neo4j database."""
        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
        )
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        """Close the connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def initialize_schema(self) -> None:
        """Create constraints and indexes."""
        if not self._driver:
            raise RuntimeError("Not connected")

        async with self._driver.session(database=self._database) as session:
            # Resource URI uniqueness constraint
            await session.run(
                "CREATE CONSTRAINT resource_uri IF NOT EXISTS "
                "FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
            )
            # Index on ResourceVersion content_hash
            await session.run(
                "CREATE INDEX resource_version_hash IF NOT EXISTS "
                "FOR (v:ResourceVersion) ON (v.content_hash)"
            )
            # Index on UserText conversation_id
            await session.run(
                "CREATE INDEX user_text_conversation IF NOT EXISTS "
                "FOR (n:UserText) ON (n.conversation_id)"
            )
            # Index on node id for all node types
            await session.run(
                "CREATE INDEX user_text_id IF NOT EXISTS FOR (n:UserText) ON (n.id)"
            )
            await session.run(
                "CREATE INDEX agent_text_id IF NOT EXISTS FOR (n:AgentText) ON (n.id)"
            )
            await session.run(
                "CREATE INDEX resource_version_id IF NOT EXISTS "
                "FOR (n:ResourceVersion) ON (n.id)"
            )
            await session.run(
                "CREATE INDEX resource_id IF NOT EXISTS FOR (n:Resource) ON (n.id)"
            )
            # Turn index for UserText and AgentText
            await session.run(
                "CREATE INDEX user_text_turn IF NOT EXISTS "
                "FOR (n:UserText) ON (n.turn_index)"
            )
            await session.run(
                "CREATE INDEX agent_text_turn IF NOT EXISTS "
                "FOR (n:AgentText) ON (n.turn_index)"
            )
            # Namespace index for multi-user isolation
            if self._namespace:
                for label in ("UserText", "AgentText", "ResourceVersion", "Resource"):
                    index_name = f"{label.lower()}_namespace"
                    await session.run(
                        f"CREATE INDEX {index_name} IF NOT EXISTS "
                        f"FOR (n:{label}) ON (n.namespace)"
                    )

    @property
    def _ns_filter(self) -> str:
        """Return a Cypher WHERE clause fragment for namespace filtering."""
        if self._namespace:
            return " AND n.namespace = $namespace"
        return ""

    @property
    def _ns_params(self) -> dict[str, str]:
        """Return namespace parameter dict."""
        if self._namespace:
            return {"namespace": self._namespace}
        return {}

    def _ns_set_clause(self, var: str = "n") -> str:
        """Return SET clause for namespace on node creation."""
        if self._namespace:
            return f", {var}.namespace = $namespace"
        return ""

    # =========================================================================
    # Polymorphic node/edge operations
    # =========================================================================

    async def create_node(self, node: NodeBase) -> NodeBase:
        """Create a node. Dispatches based on type."""
        if not self._driver:
            raise RuntimeError("Not connected")

        if isinstance(node, UserText):
            return await self._create_user_text(node)
        elif isinstance(node, AgentText):
            return await self._create_agent_text(node)
        elif isinstance(node, ResourceVersion):
            return await self._create_resource_version(node)
        elif isinstance(node, Resource):
            return await self._create_resource(node)
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    async def create_edge(self, edge: EdgeBase) -> EdgeBase:
        """Create an edge. Dispatches based on type."""
        if not self._driver:
            raise RuntimeError("Not connected")

        if isinstance(edge, Relationship):
            return await self._create_relationship(edge)
        elif isinstance(edge, VersionOf):
            return await self._create_version_of(edge)
        else:
            raise TypeError(f"Unknown edge type: {type(edge)}")

    # =========================================================================
    # Private node creation methods
    # =========================================================================

    async def _create_user_text(self, node: UserText) -> UserText:
        """Create a UserText node."""
        assert self._driver is not None  # Checked by create_node
        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                CREATE (n:UserText {{
                    id: $id,
                    text: $text,
                    conversation_id: $conversation_id,
                    turn_index: $turn_index,
                    created_at: $created_at,
                    last_accessed_at: $last_accessed_at
                }})
                {"SET n.namespace = $namespace" if self._namespace else ""}
                """,
                id=str(node.id),
                text=node.text,
                conversation_id=node.conversation_id,
                turn_index=node.turn_index,
                created_at=node.created_at.isoformat(),
                last_accessed_at=node.last_accessed_at.isoformat(),
                **self._ns_params,
            )
        return node

    async def _create_agent_text(self, node: AgentText) -> AgentText:
        """Create an AgentText node."""
        assert self._driver is not None  # Checked by create_node

        # Serialize tool_uses to JSON for Neo4j storage
        tool_uses_json = json.dumps([tu.model_dump() for tu in node.tool_uses])

        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                CREATE (n:AgentText {{
                    id: $id,
                    text: $text,
                    conversation_id: $conversation_id,
                    turn_index: $turn_index,
                    tool_uses: $tool_uses,
                    created_at: $created_at,
                    last_accessed_at: $last_accessed_at
                }})
                {"SET n.namespace = $namespace" if self._namespace else ""}
                """,
                id=str(node.id),
                text=node.text,
                conversation_id=node.conversation_id,
                turn_index=node.turn_index,
                tool_uses=tool_uses_json,
                created_at=node.created_at.isoformat(),
                last_accessed_at=node.last_accessed_at.isoformat(),
                **self._ns_params,
            )
        return node

    async def _create_resource_version(self, node: ResourceVersion) -> ResourceVersion:
        """Create a ResourceVersion node."""
        assert self._driver is not None  # Checked by create_node
        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                CREATE (n:ResourceVersion {{
                    id: $id,
                    content_hash: $content_hash,
                    uri: $uri,
                    conversation_id: $conversation_id,
                    created_at: $created_at,
                    last_accessed_at: $last_accessed_at
                }})
                {"SET n.namespace = $namespace" if self._namespace else ""}
                """,
                id=str(node.id),
                content_hash=node.content_hash,
                uri=node.uri,
                conversation_id=node.conversation_id,
                created_at=node.created_at.isoformat(),
                last_accessed_at=node.last_accessed_at.isoformat(),
                **self._ns_params,
            )
        return node

    async def _create_resource(self, node: Resource) -> Resource:
        """Create or get a Resource hypernode."""
        assert self._driver is not None  # Checked by create_node
        ns_on_create = ", r.namespace = $namespace" if self._namespace else ""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MERGE (r:Resource {{uri: $uri}})
                ON CREATE SET
                    r.id = $id,
                    r.conversation_id = $conversation_id,
                    r.current_content_hash = $current_content_hash,
                    r.created_at = $created_at,
                    r.last_accessed_at = $last_accessed_at
                    {ns_on_create}
                RETURN r
                """,
                uri=node.uri,
                id=str(node.id),
                conversation_id=node.conversation_id,
                current_content_hash=node.current_content_hash,
                created_at=node.created_at.isoformat(),
                last_accessed_at=node.last_accessed_at.isoformat(),
                **self._ns_params,
            )
            record = await result.single()
            if record:
                r = record["r"]
                return Resource(
                    id=UUID(r["id"]),
                    uri=r["uri"],
                    conversation_id=r["conversation_id"],
                    current_content_hash=r.get("current_content_hash"),
                    created_at=datetime.fromisoformat(r["created_at"]),
                    last_accessed_at=datetime.fromisoformat(r["last_accessed_at"]),
                )
        return node

    async def get_resource_by_uri(self, uri: str) -> Resource | None:
        """Get a Resource by its URI."""
        if not self._driver:
            raise RuntimeError("Not connected")

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (r:Resource {uri: $uri}) RETURN r",
                uri=uri,
            )
            record = await result.single()
            if record:
                r = record["r"]
                return Resource(
                    id=UUID(r["id"]),
                    uri=r["uri"],
                    conversation_id=r["conversation_id"],
                    current_content_hash=r.get("current_content_hash"),
                    created_at=datetime.fromisoformat(r["created_at"]),
                    last_accessed_at=datetime.fromisoformat(r["last_accessed_at"]),
                )
        return None

    async def update_resource_hash(self, uri: str, content_hash: str) -> None:
        """Update the current content hash of a Resource."""
        if not self._driver:
            raise RuntimeError("Not connected")

        async with self._driver.session(database=self._database) as session:
            await session.run(
                """
                MATCH (r:Resource {uri: $uri})
                SET r.current_content_hash = $content_hash,
                    r.last_accessed_at = $last_accessed_at
                """,
                uri=uri,
                content_hash=content_hash,
                last_accessed_at=datetime.now(UTC).isoformat(),
            )

    async def get_resource_version_by_hash(
        self, uri: str, content_hash: str
    ) -> ResourceVersion | None:
        """Get a ResourceVersion by its URI and content hash."""
        if not self._driver:
            raise RuntimeError("Not connected")

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (v:ResourceVersion {uri: $uri, content_hash: $content_hash})
                RETURN v
                LIMIT 1
                """,
                uri=uri,
                content_hash=content_hash,
            )
            record = await result.single()
            if record:
                v = record["v"]
                return ResourceVersion(
                    id=UUID(v["id"]),
                    content_hash=v["content_hash"],
                    uri=v["uri"],
                    conversation_id=v["conversation_id"],
                    created_at=datetime.fromisoformat(v["created_at"]),
                    last_accessed_at=datetime.fromisoformat(v["last_accessed_at"]),
                )
        return None

    # =========================================================================
    # Private edge creation methods
    # =========================================================================

    async def _create_relationship(self, edge: Relationship) -> Relationship:
        """Create a relationship between nodes."""
        assert self._driver is not None  # Checked by create_edge
        # Sanitize relationship type for Cypher
        rel_type = edge.relationship_type.upper().replace(" ", "_")

        async with self._driver.session(database=self._database) as session:
            # Dynamic relationship type requires string interpolation
            # Properties are still parameterized for safety
            # Note: properties dict is serialized to JSON string since Neo4j
            # doesn't support Map types as property values
            query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                CREATE (source)-[r:{rel_type} {{
                    id: $id,
                    conversation_id: $conversation_id,
                    created_at: $created_at,
                    properties: $properties
                }}]->(target)
                RETURN r
                """
            result = await session.run(
                query,
                source_id=str(edge.source_id),
                target_id=str(edge.target_id),
                id=str(edge.id),
                conversation_id=edge.conversation_id,
                created_at=edge.created_at.isoformat(),
                properties=json.dumps(edge.properties),
            )
            # Consume result to commit the auto-commit transaction
            await result.consume()
        return edge

    async def _create_version_of(self, edge: VersionOf) -> VersionOf:
        """Create a VERSION_OF relationship."""
        assert self._driver is not None  # Checked by create_edge
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                """
                MATCH (v:ResourceVersion {id: $version_id})
                MATCH (r:Resource {id: $resource_id})
                CREATE (v)-[:VERSION_OF {
                    id: $id,
                    created_at: $created_at
                }]->(r)
                """,
                version_id=str(edge.version_id),
                resource_id=str(edge.resource_id),
                id=str(edge.id),
                created_at=edge.created_at.isoformat(),
            )
            # Consume result to commit the auto-commit transaction
            await result.consume()
        return edge

    # =========================================================================
    # Basic retrieval operations
    # =========================================================================

    async def get_user_text(self, node_id: UUID) -> UserText | None:
        """Get a UserText node by ID."""
        if not self._driver:
            raise RuntimeError("Not connected")

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                "MATCH (n:UserText {id: $id}) RETURN n",
                id=str(node_id),
            )
            record = await result.single()
            if record:
                n = record["n"]
                return UserText(
                    id=UUID(n["id"]),
                    text=n["text"],
                    conversation_id=n["conversation_id"],
                    turn_index=n.get("turn_index", 0),
                    created_at=datetime.fromisoformat(n["created_at"]),
                    last_accessed_at=datetime.fromisoformat(n["last_accessed_at"]),
                )
        return None

    async def get_last_user_text(self, conversation_id: str) -> UserText | None:
        """Get the most recent UserText node in a conversation by timestamp."""
        if not self._driver:
            raise RuntimeError("Not connected")

        ns_where = "AND u.namespace = $namespace" if self._namespace else ""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (u:UserText {{conversation_id: $conversation_id}})
                WHERE true {ns_where}
                RETURN u
                ORDER BY u.created_at DESC
                LIMIT 1
                """,
                conversation_id=conversation_id,
                **self._ns_params,
            )
            record = await result.single()
            if record:
                u = record["u"]
                return UserText(
                    id=UUID(u["id"]),
                    text=u["text"],
                    conversation_id=u["conversation_id"],
                    turn_index=u.get("turn_index", 0),
                    created_at=datetime.fromisoformat(u["created_at"]),
                    last_accessed_at=datetime.fromisoformat(u["last_accessed_at"]),
                )
        return None

    async def get_last_agent_text(self, conversation_id: str) -> AgentText | None:
        """Get the most recent AgentText node in a conversation by timestamp."""
        if not self._driver:
            raise RuntimeError("Not connected")

        ns_where = "AND a.namespace = $namespace" if self._namespace else ""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (a:AgentText {{conversation_id: $conversation_id}})
                WHERE true {ns_where}
                RETURN a
                ORDER BY a.created_at DESC
                LIMIT 1
                """,
                conversation_id=conversation_id,
                **self._ns_params,
            )
            record = await result.single()
            if record:
                a = record["a"]
                # Deserialize tool_uses from JSON
                tool_uses_data = json.loads(a.get("tool_uses", "[]"))
                tool_uses = [ToolUseRecord(**tu) for tu in tool_uses_data]
                return AgentText(
                    id=UUID(a["id"]),
                    text=a["text"],
                    conversation_id=a["conversation_id"],
                    turn_index=a.get("turn_index", 0),
                    tool_uses=tool_uses,
                    created_at=datetime.fromisoformat(a["created_at"]),
                    last_accessed_at=datetime.fromisoformat(a["last_accessed_at"]),
                )
        return None

    async def get_last_message_node(
        self, conversation_id: str
    ) -> UserText | AgentText | None:
        """Get the most recent message node (UserText or AgentText) by timestamp.

        This is used to maintain conversation continuity when there are multiple
        assistant messages in a row (e.g., with tool usage).
        """
        if not self._driver:
            raise RuntimeError("Not connected")

        ns_filter = "AND n.namespace = $namespace" if self._namespace else ""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (n)
                WHERE (n:UserText OR n:AgentText)
                  AND n.conversation_id = $conversation_id
                  {ns_filter}
                RETURN n, labels(n) as labels
                ORDER BY n.created_at DESC
                LIMIT 1
                """,
                conversation_id=conversation_id,
                **self._ns_params,
            )
            record = await result.single()
            if record:
                n = record["n"]
                labels = record["labels"]
                if "UserText" in labels:
                    return UserText(
                        id=UUID(n["id"]),
                        text=n["text"],
                        conversation_id=n["conversation_id"],
                        turn_index=n.get("turn_index", 0),
                        created_at=datetime.fromisoformat(n["created_at"]),
                        last_accessed_at=datetime.fromisoformat(n["last_accessed_at"]),
                    )
                elif "AgentText" in labels:
                    # Deserialize tool_uses from JSON
                    tool_uses_data = json.loads(n.get("tool_uses", "[]"))
                    tool_uses = [ToolUseRecord(**tu) for tu in tool_uses_data]
                    return AgentText(
                        id=UUID(n["id"]),
                        text=n["text"],
                        conversation_id=n["conversation_id"],
                        turn_index=n.get("turn_index", 0),
                        tool_uses=tool_uses,
                        created_at=datetime.fromisoformat(n["created_at"]),
                        last_accessed_at=datetime.fromisoformat(n["last_accessed_at"]),
                    )
        return None

    async def get_max_turn_index(self, conversation_id: str) -> int:
        """Get the maximum turn index in a conversation.

        Returns -1 if no turns exist (first user message will be turn 0).
        """
        if not self._driver:
            raise RuntimeError("Not connected")

        ns_filter = "AND n.namespace = $namespace" if self._namespace else ""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (n)
                WHERE (n:UserText OR n:AgentText)
                  AND n.conversation_id = $conversation_id
                  {ns_filter}
                RETURN MAX(n.turn_index) as max_turn
                """,
                conversation_id=conversation_id,
                **self._ns_params,
            )
            record = await result.single()
            if record and record["max_turn"] is not None:
                return record["max_turn"]
        return -1

    async def get_last_node_in_turn(
        self, conversation_id: str, turn_index: int
    ) -> UserText | AgentText | None:
        """Get the most recent node in a specific turn by timestamp."""
        if not self._driver:
            raise RuntimeError("Not connected")

        ns_filter = "AND n.namespace = $namespace" if self._namespace else ""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (n)
                WHERE (n:UserText OR n:AgentText)
                  AND n.conversation_id = $conversation_id
                  AND n.turn_index = $turn_index
                  {ns_filter}
                RETURN n, labels(n) as labels
                ORDER BY n.created_at DESC
                LIMIT 1
                """,
                conversation_id=conversation_id,
                turn_index=turn_index,
                **self._ns_params,
            )
            record = await result.single()
            if record:
                n = record["n"]
                labels = record["labels"]
                if "UserText" in labels:
                    return UserText(
                        id=UUID(n["id"]),
                        text=n["text"],
                        conversation_id=n["conversation_id"],
                        turn_index=n.get("turn_index", 0),
                        created_at=datetime.fromisoformat(n["created_at"]),
                        last_accessed_at=datetime.fromisoformat(n["last_accessed_at"]),
                    )
                elif "AgentText" in labels:
                    # Deserialize tool_uses from JSON
                    tool_uses_data = json.loads(n.get("tool_uses", "[]"))
                    tool_uses = [ToolUseRecord(**tu) for tu in tool_uses_data]
                    return AgentText(
                        id=UUID(n["id"]),
                        text=n["text"],
                        conversation_id=n["conversation_id"],
                        turn_index=n.get("turn_index", 0),
                        tool_uses=tool_uses,
                        created_at=datetime.fromisoformat(n["created_at"]),
                        last_accessed_at=datetime.fromisoformat(n["last_accessed_at"]),
                    )
        return None

    async def update_last_accessed(self, node_ids: list[UUID]) -> None:
        """Update last_accessed_at for the given nodes."""
        if not self._driver or not node_ids:
            return

        now = datetime.now(UTC).isoformat()
        str_ids = [str(nid) for nid in node_ids]

        async with self._driver.session(database=self._database) as session:
            await session.run(
                """
                MATCH (n)
                WHERE n.id IN $ids
                SET n.last_accessed_at = $now
                """,
                ids=str_ids,
                now=now,
            )

    # =========================================================================
    # Retrieval query operations
    # =========================================================================

    async def get_node_context(self, node_id: UUID) -> ContextResult:
        """Get full context for a UserText node (user text, agent response, tool uses)."""
        if not self._driver:
            raise RuntimeError("Not connected")

        logger.debug("get_node_context node_id=%s", node_id)
        context = ContextResult()

        async with self._driver.session(database=self._database) as session:
            # Get user text and agent text
            result = await session.run(
                """
                MATCH (u:UserText {id: $id})
                OPTIONAL MATCH (u)-[m:MESSAGE]->(a:AgentText)
                RETURN u, a
                """,
                id=str(node_id),
            )
            record = await result.single()

            if not record:
                logger.debug("get_node_context node_id=%s not found", node_id)
                return context

            if record.get("u"):
                u = record["u"]
                context.user_text = UserTextInfo(
                    id=u["id"],
                    text=u["text"],
                    conversation_id=u["conversation_id"],
                )

            if record.get("a"):
                a = record["a"]
                context.agent_text = AgentTextInfo(
                    id=a["id"],
                    text=a["text"],
                )

            # Get tool uses
            result = await session.run(
                """
                MATCH (u:UserText {id: $id})-[m:MESSAGE]->(a:AgentText)
                MATCH (a)-[r]->(v:ResourceVersion)
                WHERE type(r) <> 'MESSAGE'
                OPTIONAL MATCH (v)-[:VERSION_OF]->(res:Resource)
                RETURN type(r) as tool_name, r.properties as props, v, res
                """,
                id=str(node_id),
            )
            records = await result.data()

            for rec in records:
                props = rec["props"]
                if isinstance(props, str):
                    props = json.loads(props)
                tool_use = ToolUse(
                    tool_name=rec["tool_name"],
                    properties=props or {},
                )
                if rec.get("v"):
                    v = rec["v"]
                    tool_use.resource_version = ResourceVersionInfo(
                        id=v["id"],
                        uri=v["uri"],
                        content_hash=v["content_hash"],
                    )
                if rec.get("res"):
                    res = rec["res"]
                    tool_use.resource = ResourceInfo(
                        id=res["id"],
                        uri=res["uri"],
                    )
                context.tool_uses.append(tool_use)

        logger.debug(
            "get_node_context node_id=%s tool_uses=%d", node_id, len(context.tool_uses)
        )
        return context

    async def get_resource_conversations(
        self,
        uri: str,
        *,
        limit: int = 10,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        exclude_conversation_id: str | None = None,
    ) -> list[ConversationReference]:
        """Find all conversations that accessed a resource by URI."""
        if not self._driver:
            raise RuntimeError("Not connected")

        logger.debug(
            "get_resource_conversations uri=%s limit=%d sort_by=%s sort_order=%s exclude=%s",
            uri,
            limit,
            sort_by,
            sort_order,
            exclude_conversation_id,
        )

        sort_field = "u.created_at" if sort_by == "created_at" else "u.last_accessed_at"
        order = "DESC" if sort_order == "desc" else "ASC"

        query = """
            MATCH (res:Resource {uri: $uri})<-[:VERSION_OF]-(v:ResourceVersion)
            MATCH (v)<-[r]-(a:AgentText)
            WHERE type(r) <> 'VERSION_OF'
            MATCH (u:UserText)-[:MESSAGE*]->(a)
        """
        params: dict[str, str | int] = {"uri": uri}

        if exclude_conversation_id:
            query += " WHERE u.conversation_id <> $exclude_conversation_id"
            params["exclude_conversation_id"] = exclude_conversation_id

        query += f"""
            RETURN DISTINCT u.conversation_id as conversation_id,
                   u.id as user_text_id,
                   u.text as user_text,
                   a.text as agent_text,
                   u.created_at as created_at
            ORDER BY {sort_field} {order}
            LIMIT $limit
        """
        params["limit"] = limit

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, params)
            records = await result.data()

        logger.debug("get_resource_conversations uri=%s results=%d", uri, len(records))

        return [
            ConversationReference(
                conversation_id=rec["conversation_id"],
                user_text_id=rec["user_text_id"],
                user_text=rec["user_text"],
                agent_text=rec.get("agent_text"),
                created_at=(
                    datetime.fromisoformat(rec["created_at"])
                    if rec.get("created_at")
                    else None
                ),
            )
            for rec in records
        ]

    async def get_trajectory_nodes(
        self,
        node_id: UUID,
        *,
        max_depth: int = 100,
    ) -> list[dict[str, Any]]:
        """Get raw nodes reachable from a UserText via MESSAGE edges.

        Returns list of dicts with 'n' (node props) and 'node_labels' (list[str]).
        """
        if not self._driver:
            raise RuntimeError("Not connected")

        logger.debug("get_trajectory_nodes node_id=%s max_depth=%d", node_id, max_depth)

        # Neo4j doesn't support parameterized hop depth, so we string-interpolate
        # the validated int (safe since max_depth is validated by RetrievalConfig).
        query = f"""
            MATCH (start:UserText {{id: $id}})-[:MESSAGE*0..{int(max_depth)}]->(n)
            WHERE n.conversation_id = start.conversation_id
            RETURN n, labels(n) as node_labels
            ORDER BY n.created_at ASC
        """

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, {"id": str(node_id)})
            records = await result.data()

        logger.debug(
            "get_trajectory_nodes node_id=%s results=%d", node_id, len(records)
        )
        return records

    # =========================================================================
    # Raw Cypher execution
    # =========================================================================

    async def execute_cypher(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a raw Cypher query and return results.

        Args:
            query: Cypher query string.
            parameters: Optional query parameters.

        Returns:
            List of result records as dictionaries.

        Example:
            ```python
            results = await store.execute_cypher(
                "MATCH (u:UserText {id: $id})-[:MESSAGE]->(a:AgentText) RETURN u, a",
                {"id": "some-uuid"}
            )
            ```
        """
        if not self._driver:
            raise RuntimeError("Not connected")

        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, parameters or {})
            return await result.data()
