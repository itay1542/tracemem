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
        lines: list[str] = [f"## Similar Past Query (score: {result.score:.2f})"]
        lines.append(f"**User:** {_truncate(result.text, 200)}")

        # Collect all tool uses and find the last AgentText with actual text
        # (the Stop handler writes the final response as the last AgentText)
        actions = []
        agent_text = ""
        follow_up = ""
        for step in trajectory.steps:
            if step.node_type == "AgentText":
                # Take the last non-empty text (Stop handler's output)
                if step.text:
                    agent_text = step.text
                for tu in step.tool_uses:
                    uri = tu.properties.get("file_path", tu.properties.get("path", ""))
                    if uri:
                        actions.append(f"{tu.tool_name} {uri}")
                    else:
                        actions.append(tu.tool_name)
            elif step.node_type == "UserText" and step.node_id != str(result.node_id):
                follow_up = step.text

        if actions:
            lines.append(f"**Actions:** {' → '.join(actions)}")
        if agent_text:
            lines.append(f"**Response:** {_truncate(agent_text, 300)}")
        if follow_up:
            lines.append(f"**Follow-up:** {_truncate(follow_up, 150)}")

        sections.append("\n".join(lines))

    body = "\n\n".join(sections)
    return f"<tracemem-context>\n{body}\n</tracemem-context>"


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
        user_part = _truncate(ref.user_text, 200)
        if ref.agent_text:
            agent_part = _truncate(ref.agent_text, 100)
            lines.append(f"- User asked \"{user_part}\" → Agent: {agent_part}")
        else:
            lines.append(f"- User asked \"{user_part}\"")

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
