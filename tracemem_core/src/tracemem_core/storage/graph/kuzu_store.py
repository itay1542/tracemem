"""Kùzu embedded graph database implementation of GraphStore."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import kuzu

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


def _result_to_dicts(result: kuzu.QueryResult) -> list[dict[str, Any]]:
    """Convert a Kùzu QueryResult to a list of dicts keyed by column name."""
    columns = result.get_column_names()
    rows = []
    while result.has_next():
        values = result.get_next()
        rows.append(dict(zip(columns, values)))
    return rows


def _single(result: kuzu.QueryResult) -> dict[str, Any] | None:
    """Get a single result row as a dict, or None."""
    columns = result.get_column_names()
    if result.has_next():
        values = result.get_next()
        return dict(zip(columns, values))
    return None


class KuzuGraphStore:
    """Kùzu embedded graph database implementation of GraphStore.

    Uses an embedded Kùzu database that requires no external server.
    All operations are synchronous in Kùzu and wrapped with asyncio.to_thread().
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    async def connect(self) -> None:
        """Connect to the Kùzu database."""
        self._db_path.mkdir(parents=True, exist_ok=True)
        # Kùzu needs a non-existing subpath or existing DB directory
        graph_dir = self._db_path / "kuzu_db"

        def _connect() -> tuple[kuzu.Database, kuzu.Connection]:
            db = kuzu.Database(str(graph_dir))
            conn = kuzu.Connection(db)
            return db, conn

        self._db, self._conn = await asyncio.to_thread(_connect)

    async def close(self) -> None:
        """Close the connection."""
        self._conn = None
        self._db = None

    async def initialize_schema(self) -> None:
        """Create node and relationship tables."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _init_schema(conn: kuzu.Connection) -> None:
            # Node tables
            conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS UserText(
                    id STRING,
                    text STRING,
                    conversation_id STRING,
                    turn_index INT64,
                    created_at STRING,
                    last_accessed_at STRING,
                    PRIMARY KEY(id)
                )
            """)
            conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS AgentText(
                    id STRING,
                    text STRING,
                    conversation_id STRING,
                    turn_index INT64,
                    tool_uses STRING,
                    created_at STRING,
                    last_accessed_at STRING,
                    PRIMARY KEY(id)
                )
            """)
            conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS ResourceVersion(
                    id STRING,
                    content_hash STRING,
                    uri STRING,
                    conversation_id STRING,
                    created_at STRING,
                    last_accessed_at STRING,
                    PRIMARY KEY(id)
                )
            """)
            conn.execute("""
                CREATE NODE TABLE IF NOT EXISTS Resource(
                    id STRING,
                    uri STRING,
                    current_content_hash STRING,
                    conversation_id STRING,
                    created_at STRING,
                    last_accessed_at STRING,
                    PRIMARY KEY(id)
                )
            """)

            # MESSAGE rel table group: supports edges between UserText<->AgentText
            conn.execute("""
                CREATE REL TABLE GROUP IF NOT EXISTS MESSAGE(
                    FROM UserText TO AgentText,
                    FROM AgentText TO UserText,
                    FROM AgentText TO AgentText,
                    id STRING,
                    conversation_id STRING,
                    created_at STRING,
                    properties STRING
                )
            """)

            # VERSION_OF: ResourceVersion -> Resource
            conn.execute("""
                CREATE REL TABLE IF NOT EXISTS VERSION_OF(
                    FROM ResourceVersion TO Resource,
                    id STRING,
                    created_at STRING
                )
            """)

            # TOOL_USE: AgentText -> ResourceVersion (with tool_name property)
            conn.execute("""
                CREATE REL TABLE IF NOT EXISTS TOOL_USE(
                    FROM AgentText TO ResourceVersion,
                    id STRING,
                    tool_name STRING,
                    conversation_id STRING,
                    created_at STRING,
                    properties STRING
                )
            """)

        await asyncio.to_thread(_init_schema, self._conn)

    # =========================================================================
    # Polymorphic node/edge operations
    # =========================================================================

    async def create_node(self, node: NodeBase) -> NodeBase:
        """Create a node. Dispatches based on type."""
        if not self._conn:
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
        if not self._conn:
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
        assert self._conn is not None

        def _create(conn: kuzu.Connection) -> None:
            conn.execute(
                """
                CREATE (n:UserText {
                    id: $id,
                    text: $text,
                    conversation_id: $conversation_id,
                    turn_index: $turn_index,
                    created_at: $created_at,
                    last_accessed_at: $last_accessed_at
                })
                """,
                {
                    "id": str(node.id),
                    "text": node.text,
                    "conversation_id": node.conversation_id,
                    "turn_index": node.turn_index,
                    "created_at": node.created_at.isoformat(),
                    "last_accessed_at": node.last_accessed_at.isoformat(),
                },
            )

        await asyncio.to_thread(_create, self._conn)
        return node

    async def _create_agent_text(self, node: AgentText) -> AgentText:
        """Create an AgentText node."""
        assert self._conn is not None
        tool_uses_json = json.dumps([tu.model_dump() for tu in node.tool_uses])

        def _create(conn: kuzu.Connection) -> None:
            conn.execute(
                """
                CREATE (n:AgentText {
                    id: $id,
                    text: $text,
                    conversation_id: $conversation_id,
                    turn_index: $turn_index,
                    tool_uses: $tool_uses,
                    created_at: $created_at,
                    last_accessed_at: $last_accessed_at
                })
                """,
                {
                    "id": str(node.id),
                    "text": node.text,
                    "conversation_id": node.conversation_id,
                    "turn_index": node.turn_index,
                    "tool_uses": tool_uses_json,
                    "created_at": node.created_at.isoformat(),
                    "last_accessed_at": node.last_accessed_at.isoformat(),
                },
            )

        await asyncio.to_thread(_create, self._conn)
        return node

    async def _create_resource_version(self, node: ResourceVersion) -> ResourceVersion:
        """Create a ResourceVersion node."""
        assert self._conn is not None

        def _create(conn: kuzu.Connection) -> None:
            conn.execute(
                """
                CREATE (n:ResourceVersion {
                    id: $id,
                    content_hash: $content_hash,
                    uri: $uri,
                    conversation_id: $conversation_id,
                    created_at: $created_at,
                    last_accessed_at: $last_accessed_at
                })
                """,
                {
                    "id": str(node.id),
                    "content_hash": node.content_hash,
                    "uri": node.uri,
                    "conversation_id": node.conversation_id,
                    "created_at": node.created_at.isoformat(),
                    "last_accessed_at": node.last_accessed_at.isoformat(),
                },
            )

        await asyncio.to_thread(_create, self._conn)
        return node

    async def _create_resource(self, node: Resource) -> Resource:
        """Create or get a Resource hypernode (MERGE on uri)."""
        assert self._conn is not None

        def _create(conn: kuzu.Connection) -> Resource:
            result = conn.execute(
                "MATCH (r:Resource) WHERE r.uri = $uri RETURN r.id, r.uri, "
                "r.conversation_id, r.current_content_hash, r.created_at, r.last_accessed_at",
                {"uri": node.uri},
            )
            existing = _single(result)
            if existing:
                return Resource(
                    id=UUID(existing["r.id"]),
                    uri=existing["r.uri"],
                    conversation_id=existing["r.conversation_id"],
                    current_content_hash=existing.get("r.current_content_hash"),
                    created_at=datetime.fromisoformat(existing["r.created_at"]),
                    last_accessed_at=datetime.fromisoformat(
                        existing["r.last_accessed_at"]
                    ),
                )

            conn.execute(
                """
                CREATE (r:Resource {
                    id: $id,
                    uri: $uri,
                    conversation_id: $conversation_id,
                    current_content_hash: $current_content_hash,
                    created_at: $created_at,
                    last_accessed_at: $last_accessed_at
                })
                """,
                {
                    "id": str(node.id),
                    "uri": node.uri,
                    "conversation_id": node.conversation_id,
                    "current_content_hash": node.current_content_hash or "",
                    "created_at": node.created_at.isoformat(),
                    "last_accessed_at": node.last_accessed_at.isoformat(),
                },
            )
            return node

        return await asyncio.to_thread(_create, self._conn)

    async def get_resource_by_uri(self, uri: str) -> Resource | None:
        """Get a Resource by its URI."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> Resource | None:
            result = conn.execute(
                "MATCH (r:Resource) WHERE r.uri = $uri RETURN r.id, r.uri, "
                "r.conversation_id, r.current_content_hash, r.created_at, r.last_accessed_at",
                {"uri": uri},
            )
            rec = _single(result)
            if rec:
                return Resource(
                    id=UUID(rec["r.id"]),
                    uri=rec["r.uri"],
                    conversation_id=rec["r.conversation_id"],
                    current_content_hash=rec.get("r.current_content_hash"),
                    created_at=datetime.fromisoformat(rec["r.created_at"]),
                    last_accessed_at=datetime.fromisoformat(rec["r.last_accessed_at"]),
                )
            return None

        return await asyncio.to_thread(_get, self._conn)

    async def update_resource_hash(self, uri: str, content_hash: str) -> None:
        """Update the current content hash of a Resource."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _update(conn: kuzu.Connection) -> None:
            conn.execute(
                """
                MATCH (r:Resource) WHERE r.uri = $uri
                SET r.current_content_hash = $content_hash,
                    r.last_accessed_at = $last_accessed_at
                """,
                {
                    "uri": uri,
                    "content_hash": content_hash,
                    "last_accessed_at": datetime.now(UTC).isoformat(),
                },
            )

        await asyncio.to_thread(_update, self._conn)

    async def get_resource_version_by_hash(
        self, uri: str, content_hash: str
    ) -> ResourceVersion | None:
        """Get a ResourceVersion by its URI and content hash."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> ResourceVersion | None:
            result = conn.execute(
                "MATCH (v:ResourceVersion) WHERE v.uri = $uri AND v.content_hash = $content_hash "
                "RETURN v.id, v.content_hash, v.uri, v.conversation_id, v.created_at, "
                "v.last_accessed_at LIMIT 1",
                {"uri": uri, "content_hash": content_hash},
            )
            rec = _single(result)
            if rec:
                return ResourceVersion(
                    id=UUID(rec["v.id"]),
                    content_hash=rec["v.content_hash"],
                    uri=rec["v.uri"],
                    conversation_id=rec["v.conversation_id"],
                    created_at=datetime.fromisoformat(rec["v.created_at"]),
                    last_accessed_at=datetime.fromisoformat(rec["v.last_accessed_at"]),
                )
            return None

        return await asyncio.to_thread(_get, self._conn)

    # =========================================================================
    # Private edge creation methods
    # =========================================================================

    async def _create_relationship(self, edge: Relationship) -> Relationship:
        """Create a relationship between nodes.

        MESSAGE relationships use the MESSAGE rel table group.
        Tool relationships (READ, WRITE, EDIT, etc.) use the TOOL_USE rel table.
        """
        assert self._conn is not None
        rel_type = edge.relationship_type.upper().replace(" ", "_")

        if rel_type == "MESSAGE":
            return await self._create_message_edge(edge)
        else:
            return await self._create_tool_use_edge(edge, rel_type)

    async def _create_message_edge(self, edge: Relationship) -> Relationship:
        """Create a MESSAGE edge using the MESSAGE rel table group."""
        assert self._conn is not None

        def _create(conn: kuzu.Connection) -> None:
            # Determine source/target types by trying each combination.
            # The MESSAGE group covers: UserText->AgentText, AgentText->UserText,
            # AgentText->AgentText
            combos = [
                ("UserText", "AgentText"),
                ("AgentText", "UserText"),
                ("AgentText", "AgentText"),
            ]
            params = {
                "source_id": str(edge.source_id),
                "target_id": str(edge.target_id),
                "id": str(edge.id),
                "conversation_id": edge.conversation_id,
                "created_at": edge.created_at.isoformat(),
                "properties": json.dumps(edge.properties),
            }
            for src_label, tgt_label in combos:
                # Check if source and target exist with these labels
                check = conn.execute(
                    f"MATCH (s:{src_label}), (t:{tgt_label}) "
                    "WHERE s.id = $source_id AND t.id = $target_id "
                    "RETURN count(*) as cnt",
                    {
                        "source_id": params["source_id"],
                        "target_id": params["target_id"],
                    },
                )
                row = _single(check)
                if row and row["cnt"] > 0:
                    conn.execute(
                        f"MATCH (s:{src_label}), (t:{tgt_label}) "
                        "WHERE s.id = $source_id AND t.id = $target_id "
                        "CREATE (s)-[:MESSAGE {id: $id, conversation_id: $conversation_id, "
                        "created_at: $created_at, properties: $properties}]->(t)",
                        params,
                    )
                    return
            raise ValueError(
                f"Could not find source {edge.source_id} and target {edge.target_id} "
                "with compatible labels for MESSAGE edge"
            )

        await asyncio.to_thread(_create, self._conn)
        return edge

    async def _create_tool_use_edge(
        self, edge: Relationship, tool_name: str
    ) -> Relationship:
        """Create a TOOL_USE edge with tool_name property."""
        assert self._conn is not None

        def _create(conn: kuzu.Connection) -> None:
            conn.execute(
                "MATCH (a:AgentText), (v:ResourceVersion) "
                "WHERE a.id = $source_id AND v.id = $target_id "
                "CREATE (a)-[:TOOL_USE {id: $id, tool_name: $tool_name, "
                "conversation_id: $conversation_id, created_at: $created_at, "
                "properties: $properties}]->(v)",
                {
                    "source_id": str(edge.source_id),
                    "target_id": str(edge.target_id),
                    "id": str(edge.id),
                    "tool_name": tool_name,
                    "conversation_id": edge.conversation_id,
                    "created_at": edge.created_at.isoformat(),
                    "properties": json.dumps(edge.properties),
                },
            )

        await asyncio.to_thread(_create, self._conn)
        return edge

    async def _create_version_of(self, edge: VersionOf) -> VersionOf:
        """Create a VERSION_OF relationship."""
        assert self._conn is not None

        def _create(conn: kuzu.Connection) -> None:
            conn.execute(
                "MATCH (v:ResourceVersion), (r:Resource) "
                "WHERE v.id = $version_id AND r.id = $resource_id "
                "CREATE (v)-[:VERSION_OF {id: $id, created_at: $created_at}]->(r)",
                {
                    "version_id": str(edge.version_id),
                    "resource_id": str(edge.resource_id),
                    "id": str(edge.id),
                    "created_at": edge.created_at.isoformat(),
                },
            )

        await asyncio.to_thread(_create, self._conn)
        return edge

    # =========================================================================
    # Basic retrieval operations
    # =========================================================================

    async def get_user_text(self, node_id: UUID) -> UserText | None:
        """Get a UserText node by ID."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> UserText | None:
            result = conn.execute(
                "MATCH (n:UserText) WHERE n.id = $id "
                "RETURN n.id, n.text, n.conversation_id, n.turn_index, "
                "n.created_at, n.last_accessed_at",
                {"id": str(node_id)},
            )
            rec = _single(result)
            if rec:
                return UserText(
                    id=UUID(rec["n.id"]),
                    text=rec["n.text"],
                    conversation_id=rec["n.conversation_id"],
                    turn_index=rec.get("n.turn_index", 0),
                    created_at=datetime.fromisoformat(rec["n.created_at"]),
                    last_accessed_at=datetime.fromisoformat(rec["n.last_accessed_at"]),
                )
            return None

        return await asyncio.to_thread(_get, self._conn)

    async def get_last_user_text(self, conversation_id: str) -> UserText | None:
        """Get the most recent UserText node in a conversation."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> UserText | None:
            result = conn.execute(
                "MATCH (u:UserText) WHERE u.conversation_id = $conversation_id "
                "RETURN u.id, u.text, u.conversation_id, u.turn_index, "
                "u.created_at, u.last_accessed_at "
                "ORDER BY u.created_at DESC LIMIT 1",
                {"conversation_id": conversation_id},
            )
            rec = _single(result)
            if rec:
                return UserText(
                    id=UUID(rec["u.id"]),
                    text=rec["u.text"],
                    conversation_id=rec["u.conversation_id"],
                    turn_index=rec.get("u.turn_index", 0),
                    created_at=datetime.fromisoformat(rec["u.created_at"]),
                    last_accessed_at=datetime.fromisoformat(rec["u.last_accessed_at"]),
                )
            return None

        return await asyncio.to_thread(_get, self._conn)

    async def get_last_agent_text(self, conversation_id: str) -> AgentText | None:
        """Get the most recent AgentText node in a conversation."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> AgentText | None:
            result = conn.execute(
                "MATCH (a:AgentText) WHERE a.conversation_id = $conversation_id "
                "RETURN a.id, a.text, a.conversation_id, a.turn_index, a.tool_uses, "
                "a.created_at, a.last_accessed_at "
                "ORDER BY a.created_at DESC LIMIT 1",
                {"conversation_id": conversation_id},
            )
            rec = _single(result)
            if rec:
                tool_uses_data = json.loads(rec.get("a.tool_uses", "[]") or "[]")
                tool_uses = [ToolUseRecord(**tu) for tu in tool_uses_data]
                return AgentText(
                    id=UUID(rec["a.id"]),
                    text=rec["a.text"],
                    conversation_id=rec["a.conversation_id"],
                    turn_index=rec.get("a.turn_index", 0),
                    tool_uses=tool_uses,
                    created_at=datetime.fromisoformat(rec["a.created_at"]),
                    last_accessed_at=datetime.fromisoformat(rec["a.last_accessed_at"]),
                )
            return None

        return await asyncio.to_thread(_get, self._conn)

    async def get_last_message_node(
        self, conversation_id: str
    ) -> UserText | AgentText | None:
        """Get the most recent message node (UserText or AgentText) by timestamp."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> UserText | AgentText | None:
            # Use UNION ALL since Kùzu doesn't support (n:UserText OR n:AgentText)
            result = conn.execute(
                "MATCH (n:UserText) WHERE n.conversation_id = $cid "
                "RETURN n.id as id, n.text as text, n.conversation_id as conv, "
                "n.turn_index as turn, n.created_at as created, "
                "n.last_accessed_at as accessed, '' as tool_uses, label(n) as lbl "
                "UNION ALL "
                "MATCH (n:AgentText) WHERE n.conversation_id = $cid "
                "RETURN n.id as id, n.text as text, n.conversation_id as conv, "
                "n.turn_index as turn, n.created_at as created, "
                "n.last_accessed_at as accessed, n.tool_uses as tool_uses, label(n) as lbl",
                {"cid": conversation_id},
            )
            rows = _result_to_dicts(result)
            if not rows:
                return None

            # Sort by created_at DESC, take first
            rows.sort(key=lambda r: r["created"], reverse=True)
            rec = rows[0]

            if rec["lbl"] == "UserText":
                return UserText(
                    id=UUID(rec["id"]),
                    text=rec["text"],
                    conversation_id=rec["conv"],
                    turn_index=rec.get("turn", 0),
                    created_at=datetime.fromisoformat(rec["created"]),
                    last_accessed_at=datetime.fromisoformat(rec["accessed"]),
                )
            else:
                tool_uses_data = json.loads(rec.get("tool_uses") or "[]")
                tool_uses = [ToolUseRecord(**tu) for tu in tool_uses_data]
                return AgentText(
                    id=UUID(rec["id"]),
                    text=rec["text"],
                    conversation_id=rec["conv"],
                    turn_index=rec.get("turn", 0),
                    tool_uses=tool_uses,
                    created_at=datetime.fromisoformat(rec["created"]),
                    last_accessed_at=datetime.fromisoformat(rec["accessed"]),
                )

        return await asyncio.to_thread(_get, self._conn)

    async def get_max_turn_index(self, conversation_id: str) -> int:
        """Get the maximum turn index in a conversation.

        Returns -1 if no turns exist.
        """
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> int:
            # UNION ALL for both node types, then take max in Python
            result = conn.execute(
                "MATCH (n:UserText) WHERE n.conversation_id = $cid "
                "RETURN n.turn_index as turn "
                "UNION ALL "
                "MATCH (n:AgentText) WHERE n.conversation_id = $cid "
                "RETURN n.turn_index as turn",
                {"cid": conversation_id},
            )
            rows = _result_to_dicts(result)
            if not rows:
                return -1
            return max(r["turn"] for r in rows)

        return await asyncio.to_thread(_get, self._conn)

    async def get_last_node_in_turn(
        self, conversation_id: str, turn_index: int
    ) -> UserText | AgentText | None:
        """Get the most recent node in a specific turn by timestamp."""
        if not self._conn:
            raise RuntimeError("Not connected")

        def _get(conn: kuzu.Connection) -> UserText | AgentText | None:
            result = conn.execute(
                "MATCH (n:UserText) WHERE n.conversation_id = $cid AND n.turn_index = $turn "
                "RETURN n.id as id, n.text as text, n.conversation_id as conv, "
                "n.turn_index as turn, n.created_at as created, "
                "n.last_accessed_at as accessed, '' as tool_uses, label(n) as lbl "
                "UNION ALL "
                "MATCH (n:AgentText) WHERE n.conversation_id = $cid AND n.turn_index = $turn "
                "RETURN n.id as id, n.text as text, n.conversation_id as conv, "
                "n.turn_index as turn, n.created_at as created, "
                "n.last_accessed_at as accessed, n.tool_uses as tool_uses, label(n) as lbl",
                {"cid": conversation_id, "turn": turn_index},
            )
            rows = _result_to_dicts(result)
            if not rows:
                return None

            rows.sort(key=lambda r: r["created"], reverse=True)
            rec = rows[0]

            if rec["lbl"] == "UserText":
                return UserText(
                    id=UUID(rec["id"]),
                    text=rec["text"],
                    conversation_id=rec["conv"],
                    turn_index=rec.get("turn", 0),
                    created_at=datetime.fromisoformat(rec["created"]),
                    last_accessed_at=datetime.fromisoformat(rec["accessed"]),
                )
            else:
                tool_uses_data = json.loads(rec.get("tool_uses") or "[]")
                tool_uses = [ToolUseRecord(**tu) for tu in tool_uses_data]
                return AgentText(
                    id=UUID(rec["id"]),
                    text=rec["text"],
                    conversation_id=rec["conv"],
                    turn_index=rec.get("turn", 0),
                    tool_uses=tool_uses,
                    created_at=datetime.fromisoformat(rec["created"]),
                    last_accessed_at=datetime.fromisoformat(rec["accessed"]),
                )

        return await asyncio.to_thread(_get, self._conn)

    async def update_last_accessed(self, node_ids: list[UUID]) -> None:
        """Update last_accessed_at for the given nodes."""
        if not self._conn or not node_ids:
            return

        now = datetime.now(UTC).isoformat()

        def _update(conn: kuzu.Connection) -> None:
            # Update each node type separately since Kùzu requires typed MATCH
            for str_id in [str(nid) for nid in node_ids]:
                for label in ("UserText", "AgentText", "ResourceVersion", "Resource"):
                    conn.execute(
                        f"MATCH (n:{label}) WHERE n.id = $id "
                        "SET n.last_accessed_at = $now",
                        {"id": str_id, "now": now},
                    )

        await asyncio.to_thread(_update, self._conn)

    # =========================================================================
    # Retrieval query operations
    # =========================================================================

    async def get_node_context(self, node_id: UUID) -> ContextResult:
        """Get full context for a UserText node."""
        if not self._conn:
            raise RuntimeError("Not connected")

        logger.debug("get_node_context node_id=%s", node_id)

        def _get(conn: kuzu.Connection) -> ContextResult:
            context = ContextResult()

            # Get user text and agent text
            result = conn.execute(
                "MATCH (u:UserText) WHERE u.id = $id "
                "OPTIONAL MATCH (u)-[:MESSAGE]->(a:AgentText) "
                "RETURN u.id, u.text, u.conversation_id, a.id, a.text",
                {"id": str(node_id)},
            )
            rec = _single(result)
            if not rec:
                return context

            if rec.get("u.id"):
                context.user_text = UserTextInfo(
                    id=rec["u.id"],
                    text=rec["u.text"],
                    conversation_id=rec["u.conversation_id"],
                )

            if rec.get("a.id"):
                context.agent_text = AgentTextInfo(
                    id=rec["a.id"],
                    text=rec["a.text"],
                )

            # Get tool uses via TOOL_USE edges
            result = conn.execute(
                "MATCH (u:UserText)-[:MESSAGE]->(a:AgentText) WHERE u.id = $id "
                "MATCH (a)-[r:TOOL_USE]->(v:ResourceVersion) "
                "OPTIONAL MATCH (v)-[:VERSION_OF]->(res:Resource) "
                "RETURN r.tool_name as tool_name, r.properties as props, "
                "v.id as v_id, v.uri as v_uri, v.content_hash as v_hash, "
                "res.id as res_id, res.uri as res_uri",
                {"id": str(node_id)},
            )
            records = _result_to_dicts(result)

            for rec in records:
                props = rec["props"]
                if isinstance(props, str):
                    props = json.loads(props)
                tool_use = ToolUse(
                    tool_name=rec["tool_name"],
                    properties=props or {},
                )
                if rec.get("v_id"):
                    tool_use.resource_version = ResourceVersionInfo(
                        id=rec["v_id"],
                        uri=rec["v_uri"],
                        content_hash=rec["v_hash"],
                    )
                if rec.get("res_id"):
                    tool_use.resource = ResourceInfo(
                        id=rec["res_id"],
                        uri=rec["res_uri"],
                    )
                context.tool_uses.append(tool_use)

            return context

        result = await asyncio.to_thread(_get, self._conn)
        logger.debug(
            "get_node_context node_id=%s tool_uses=%d", node_id, len(result.tool_uses)
        )
        return result

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
        if not self._conn:
            raise RuntimeError("Not connected")

        logger.debug(
            "get_resource_conversations uri=%s limit=%d sort_by=%s sort_order=%s exclude=%s",
            uri,
            limit,
            sort_by,
            sort_order,
            exclude_conversation_id,
        )

        def _get(conn: kuzu.Connection) -> list[ConversationReference]:
            # Kùzu variable-length paths cap at 30
            query = (
                "MATCH (u:UserText)-[:MESSAGE*1..30]->(a:AgentText)-[r:TOOL_USE]->"
                "(v:ResourceVersion)-[:VERSION_OF]->(res:Resource) "
                "WHERE res.uri = $uri "
            )
            params: dict[str, Any] = {"uri": uri}

            if exclude_conversation_id:
                query += "AND u.conversation_id <> $exclude_cid "
                params["exclude_cid"] = exclude_conversation_id

            # Use alias in ORDER BY (Kùzu requires aliases after RETURN DISTINCT)
            sort_alias = "created_at" if sort_by == "created_at" else "last_accessed_at"
            order = "DESC" if sort_order == "desc" else "ASC"

            if sort_by == "last_accessed_at":
                query += (
                    f"RETURN DISTINCT u.conversation_id as conversation_id, "
                    f"u.id as user_text_id, u.text as user_text, "
                    f"a.text as agent_text, u.created_at as created_at, "
                    f"u.last_accessed_at as last_accessed_at "
                    f"ORDER BY {sort_alias} {order} LIMIT {int(limit)}"
                )
            else:
                query += (
                    f"RETURN DISTINCT u.conversation_id as conversation_id, "
                    f"u.id as user_text_id, u.text as user_text, "
                    f"a.text as agent_text, u.created_at as created_at "
                    f"ORDER BY {sort_alias} {order} LIMIT {int(limit)}"
                )

            result = conn.execute(query, params)
            records = _result_to_dicts(result)

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

        result = await asyncio.to_thread(_get, self._conn)
        logger.debug("get_resource_conversations uri=%s results=%d", uri, len(result))
        return result

    async def get_trajectory_nodes(
        self,
        node_id: UUID,
        *,
        max_depth: int = 100,
    ) -> list[dict[str, Any]]:
        """Get raw nodes reachable from a UserText via MESSAGE edges.

        Returns list of dicts with 'n' (node props dict) and 'node_labels' (list[str]).
        For Kùzu, label() returns a string; we wrap it in a list for compatibility
        with the Neo4j format expected by callers.
        """
        if not self._conn:
            raise RuntimeError("Not connected")

        logger.debug("get_trajectory_nodes node_id=%s max_depth=%d", node_id, max_depth)

        def _get(conn: kuzu.Connection) -> list[dict[str, Any]]:
            # Kùzu caps variable-length paths at 30
            depth = min(int(max_depth), 30)
            result = conn.execute(
                f"MATCH (start:UserText)-[:MESSAGE*0..{depth}]->(n) "
                "WHERE start.id = $id AND n.conversation_id = start.conversation_id "
                "RETURN n.id as id, n.text as text, n.conversation_id as conversation_id, "
                "n.turn_index as turn_index, n.created_at as created_at, "
                "n.last_accessed_at as last_accessed_at, n.tool_uses as tool_uses, "
                "label(n) as node_label "
                "ORDER BY n.created_at ASC",
                {"id": str(node_id)},
            )
            rows = _result_to_dicts(result)

            # Deduplicate by id (variable-length paths can yield duplicates)
            seen: set[str] = set()
            unique_rows: list[dict[str, Any]] = []
            for row in rows:
                if row["id"] not in seen:
                    seen.add(row["id"])
                    unique_rows.append(row)

            # Convert to the format expected by callers (matching Neo4j format)
            records = []
            for row in unique_rows:
                node_props = {
                    "id": row["id"],
                    "text": row["text"],
                    "conversation_id": row["conversation_id"],
                    "turn_index": row.get("turn_index", 0),
                    "created_at": row["created_at"],
                    "last_accessed_at": row["last_accessed_at"],
                }
                if row.get("tool_uses"):
                    node_props["tool_uses"] = row["tool_uses"]
                records.append(
                    {
                        "n": node_props,
                        "node_labels": [row["node_label"]],
                    }
                )
            return records

        result = await asyncio.to_thread(_get, self._conn)
        logger.debug("get_trajectory_nodes node_id=%s results=%d", node_id, len(result))
        return result

    # =========================================================================
    # Raw Cypher execution
    # =========================================================================

    async def execute_cypher(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a raw Cypher query and return results.

        Note: Kùzu uses a subset of Cypher. Callers must use Kùzu-compatible syntax.
        Key differences from Neo4j:
        - label(n) returns string, not list (use label(n) instead of labels(n))
        - No dynamic relationship types; use TOOL_USE with tool_name property
        - UNION ALL instead of (n:LabelA OR n:LabelB)
        """
        if not self._conn:
            raise RuntimeError("Not connected")

        def _execute(conn: kuzu.Connection) -> list[dict[str, Any]]:
            result = conn.execute(query, parameters or {})
            return _result_to_dicts(result)

        return await asyncio.to_thread(_execute, self._conn)
