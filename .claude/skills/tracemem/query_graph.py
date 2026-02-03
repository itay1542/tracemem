#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["tracemem-core>=0.1.0", "pyyaml>=6.0"]
# ///
"""Query the TraceMem knowledge graph directly via Cypher.

Usage:
    uv run .claude/skills/tracemem/query_graph.py "MATCH (n:UserText) RETURN n.text LIMIT 5"
    uv run .claude/skills/tracemem/query_graph.py --stats
    uv run .claude/skills/tracemem/query_graph.py --file-history tracemem_core/src/tracemem_core/tracemem.py

Note: When using graph_store="kuzu" (default), use Kùzu-compatible Cypher:
    - label(n) instead of labels(n) (returns string, not list)
    - TOOL_USE rel with r.tool_name property instead of dynamic relationship types
    - UNION ALL instead of (n:LabelA OR n:LabelB)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add the skill directory to sys.path so tracemem_claude can be imported
sys.path.insert(0, str(Path(__file__).parent))

from tracemem_claude.config import get_hook_config
from tracemem_core import RetrievalConfig, TraceMem
from tracemem_core.config import TraceMemConfig
from tracemem_core.extractors import _canonicalize_file_uri


def _build_config() -> TraceMemConfig:
    hc = get_hook_config()
    home = None if hc.mode == "global" else Path.cwd() / ".tracemem"
    return TraceMemConfig(
        home=home,
        graph_store=hc.graph_store,
        namespace=hc.namespace,
        reranker=hc.reranker,
        neo4j_uri=hc.neo4j_uri,
        neo4j_user=hc.neo4j_user,
        neo4j_password=hc.neo4j_password,
        neo4j_database=hc.neo4j_database,
        embedding_model=hc.embedding_model,
        embedding_dimensions=hc.embedding_dimensions,
        openai_api_key=hc.openai_api_key,
    )


async def run_cypher(query: str, params: dict | None = None) -> list[dict]:
    """Execute a raw Cypher query and return results."""
    async with TraceMem(config=_build_config()) as tm:
        records = await tm._graph_store.execute_cypher(query, params or {})
        return records


async def graph_stats() -> None:
    """Print graph statistics: node counts, sample data."""
    config = _build_config()
    async with TraceMem(config=config) as tm:
        gs = tm._graph_store

        print("=== TraceMem Graph Statistics ===\n")

        if config.graph_store == "kuzu":
            # Kùzu: query each table directly
            for label in ("UserText", "AgentText", "ResourceVersion", "Resource"):
                count = await gs.execute_cypher(
                    f"MATCH (n:{label}) RETURN count(n) as c", {}
                )
                print(f"  {label}: {count[0]['c'] if count else 0}")
        else:
            # Neo4j: use db.labels()
            labels = await gs.execute_cypher(
                "CALL db.labels() YIELD label RETURN label", {}
            )
            print("Node counts:")
            for rec in labels:
                label = rec["label"]
                count = await gs.execute_cypher(
                    f"MATCH (n:{label}) RETURN count(n) as c", {}
                )
                print(f"  {label}: {count[0]['c']}")

            rel_types = await gs.execute_cypher(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
                {},
            )
            print(
                f"\nRelationship types: {', '.join(r['relationshipType'] for r in rel_types)}"
            )

        convs = await gs.execute_cypher(
            "MATCH (u:UserText) RETURN DISTINCT u.conversation_id as cid, count(*) as turns "
            "ORDER BY turns DESC",
            {},
        )
        print(f"\nConversations: {len(convs)}")
        for c in convs:
            print(f"  {c['cid']}: {c['turns']} user turns")

        resources = await gs.execute_cypher(
            "MATCH (r:Resource) RETURN r.uri as uri ORDER BY r.uri",
            {},
        )
        print(f"\nResources ({len(resources)}):")
        for r in resources:
            print(f"  {r['uri']}")


async def file_history(file_path: str, limit: int = 10) -> None:
    """Show conversation history for a specific file."""
    hc = get_hook_config()
    home = None if hc.mode == "global" else Path.cwd() / ".tracemem"
    root = home.parent.resolve() if home else None
    async with TraceMem(config=_build_config()) as tm:
        absolute_path = Path(file_path).resolve()
        uri = _canonicalize_file_uri(f"file://{absolute_path}", root=root)
        config = RetrievalConfig(limit=limit, sort_by="created_at", sort_order="desc")
        refs = await tm.retrieval.get_conversations_for_resource(uri, config=config)

        print(f"=== History for {uri} ({len(refs)} interactions) ===\n")
        for ref in refs:
            print(f"- [{ref.conversation_id[:8]}] {ref.user_text[:120]}")
            if ref.agent_text:
                print(f"  Agent: {ref.agent_text[:150]}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the TraceMem knowledge graph")
    parser.add_argument("query", nargs="?", help="Cypher query to execute")
    parser.add_argument("--stats", action="store_true", help="Show graph statistics")
    parser.add_argument(
        "--file-history", metavar="PATH", help="Show history for a file path"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Limit results (default: 10)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.stats:
        asyncio.run(graph_stats())
    elif args.file_history:
        asyncio.run(file_history(args.file_history, args.limit))
    elif args.query:
        records = asyncio.run(run_cypher(args.query))
        if args.json:
            print(json.dumps(records, indent=2, default=str))
        else:
            for rec in records:
                print(rec)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
