"""Formatters for TraceMem retrieval results."""

from tracemem_core.retrieval.results import (
    ConversationReference,
    RetrievalResult,
    TrajectoryResult,
)


def format_similar_queries(
    results: list[tuple[RetrievalResult, TrajectoryResult]],
) -> str:
    """Format search results with trajectories for UserPromptSubmit stdout.

    Claude receives this as context before the user's current prompt.

    Args:
        results: List of (search_result, trajectory) tuples.

    Returns:
        Formatted string wrapped in tracemem-context tags.
    """
    if not results:
        return ""

    sections: list[str] = []
    for result, trajectory in results:
        lines: list[str] = [f"## Similar Past Query (score: {result.score:.2f}, node_id: {result.node_id})"]
        lines.append(f"**User:** {_truncate(result.text, 200)}")

        # Build trajectory summary: count agent messages, tool uses by name, find follow-up
        agent_msg_count = 0
        tool_counts: dict[str, int] = {}
        follow_up = ""
        for step in trajectory.steps:
            if step.node_type == "AgentText":
                agent_msg_count += 1
                for tu in step.tool_uses:
                    tool_counts[tu.tool_name] = tool_counts.get(tu.tool_name, 0) + 1
            elif step.node_type == "UserText" and step.node_id != str(result.node_id):
                follow_up = step.text

        # Format path summary
        if tool_counts or agent_msg_count:
            parts = []
            total_tools = sum(tool_counts.values())
            if total_tools:
                tool_summary = ", ".join(f"{name} x{count}" for name, count in tool_counts.items())
                parts.append(f"{total_tools} tool uses ({tool_summary})")
            parts.append(f"{agent_msg_count} agent messages")
            lines.append(f"**Path:** [{' | '.join(parts)}]")

        if follow_up:
            lines.append(f"**Next user message:** {_truncate(follow_up, 150)}")

        sections.append("\n".join(lines))

    body = "\n\n".join(sections)
    hint = (
        "<system-reminder>If any of the above TraceMem entries seem relevant to the "
        "current task and you need more detail, use the /tracemem skill to expand on "
        "a specific node_id or search for additional memories.</system-reminder>"
    )
    return f"<tracemem-context>\n{body}\n</tracemem-context>\n{hint}"


def format_resource_history(
    file_path: str,
    refs: list[ConversationReference],
) -> str:
    """Format resource history for PreToolUse additionalContext.

    Args:
        file_path: The file path being accessed.
        refs: List of conversation references for this resource.

    Returns:
        Formatted string describing past interactions with the file.
    """
    if not refs:
        return ""

    lines = [f"TraceMem: Past interactions with {file_path}:"]
    for ref in refs:
        user_part = _truncate(ref.user_text, 80)
        if ref.agent_text:
            agent_part = _truncate(ref.agent_text, 100)
            lines.append(f"- User asked \"{user_part}\" → Agent: {agent_part}")
        else:
            lines.append(f"- User asked \"{user_part}\"")

    lines.append(
        "\nIf any of the above TraceMem entries seem relevant to the current task "
        "and you need more detail, use the /tracemem skill to expand on a specific "
        "entry or search for additional memories."
    )

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
