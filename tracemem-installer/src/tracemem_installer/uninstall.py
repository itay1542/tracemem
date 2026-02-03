"""Uninstall command — removes TraceMem hooks from .claude/."""

import json
import shutil
from pathlib import Path

from tracemem_installer.settings import remove_hooks


def _resolve_target(scope: str) -> Path:
    if scope == "global":
        return Path.home() / ".claude"
    return Path.cwd() / ".claude"


def run_uninstall(scope: str) -> None:
    claude_dir = _resolve_target(scope)
    skill_dir = claude_dir / "skills" / "tracemem"

    # Remove skill directory
    if skill_dir.exists():
        shutil.rmtree(skill_dir)
        print(f"Removed {skill_dir}")
    else:
        print(f"No TraceMem installation found at {skill_dir}")

    # Clean hooks from settings.json
    settings_path = claude_dir / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError as e:
            print(f"Warning: could not parse {settings_path}: {e}")
            return

        settings = remove_hooks(settings)
        settings_path.write_text(json.dumps(settings, indent=2) + "\n")
        print(f"Cleaned TraceMem entries from {settings_path}")
    else:
        print("No settings.json found — nothing to clean")

    print()
    print("TraceMem hooks uninstalled.")
