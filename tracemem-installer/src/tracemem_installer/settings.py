"""Merge/remove TraceMem hook entries in .claude/settings.json."""

from typing import Any

TRACEMEM_MARKER = "tracemem"


def build_hook_entries(scope: str) -> dict[str, list[dict]]:
    """Build the 4 hook definitions with scope-aware commands."""
    if scope == "global":
        path = "$HOME/.claude/skills/tracemem"
    else:
        path = ".claude/skills/tracemem"

    return {
        "UserPromptSubmit": [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"uv run {path}/hook.py UserPromptSubmit",
                    }
                ],
            }
        ],
        "PreToolUse": [
            {
                "matcher": "Read|Write|Edit",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"uv run {path}/hook.py PreToolUse",
                    }
                ],
            }
        ],
        "PostToolUse": [
            {
                "matcher": "*",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"uv run {path}/hook.py PostToolUse",
                    }
                ],
            }
        ],
        "Stop": [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": f"uv run {path}/hook.py Stop",
                    }
                ],
            }
        ],
    }


def _is_tracemem_entry(entry: dict) -> bool:
    """Check if a hook entry belongs to TraceMem."""
    for hook in entry.get("hooks", []):
        cmd = hook.get("command", "")
        if TRACEMEM_MARKER in cmd:
            return True
    return False


def merge_hooks(
    settings: dict[str, Any], hook_entries: dict[str, list[dict]]
) -> dict[str, Any]:
    """Merge TraceMem hook entries into settings, replacing stale ones.

    Preserves all non-TraceMem entries. Returns the merged dict.
    """
    hooks = settings.get("hooks", {})

    for event_type, new_entries in hook_entries.items():
        existing = hooks.get(event_type, [])
        # Filter out old TraceMem entries
        kept = [e for e in existing if not _is_tracemem_entry(e)]
        # Append new entries
        kept.extend(new_entries)
        hooks[event_type] = kept

    settings["hooks"] = hooks
    return settings


def remove_hooks(settings: dict[str, Any]) -> dict[str, Any]:
    """Remove all TraceMem entries from settings. Clean up empty keys."""
    hooks = settings.get("hooks", {})

    empty_keys = []
    for event_type, entries in hooks.items():
        hooks[event_type] = [e for e in entries if not _is_tracemem_entry(e)]
        if not hooks[event_type]:
            empty_keys.append(event_type)

    for key in empty_keys:
        del hooks[key]

    if not hooks:
        settings.pop("hooks", None)
    else:
        settings["hooks"] = hooks

    return settings
