"""Init command — installs TraceMem hooks into .claude/."""

import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

from tracemem_installer.settings import build_hook_entries, merge_hooks

GITIGNORE_ENTRY = "skills/tracemem/.env"


def _resolve_target(scope: str) -> Path:
    if scope == "global":
        return Path.home() / ".claude"
    return Path.cwd() / ".claude"


def _copy_templates(skill_dir: Path) -> list[str]:
    """Copy template files into the skill directory.

    Returns list of installed file paths (relative to skill_dir).
    """
    templates = files("tracemem_installer") / "templates"
    installed: list[str] = []

    # Walk the templates resource tree and copy everything
    _copy_resource_tree(templates, skill_dir, installed, skill_dir)

    return installed


def _copy_resource_tree(
    resource: object, dest: Path, installed: list[str], base: Path
) -> None:
    """Recursively copy importlib.resources tree to filesystem."""
    if resource.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for child in resource.iterdir():
            name = child.name
            if name in ("__pycache__",) or name.endswith(".pyc"):
                continue
            _copy_resource_tree(child, dest / name, installed, base)
    elif resource.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = resource.read_text(encoding="utf-8")
        dest.write_text(content, encoding="utf-8")
        # Make .py files executable
        if dest.suffix == ".py":
            os.chmod(dest, 0o755)
        installed.append(str(dest.relative_to(base)))


def _merge_settings(claude_dir: Path, scope: str) -> None:
    """Merge TraceMem hook entries into settings.json."""
    settings_path = claude_dir / "settings.json"

    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError as e:
            print(f"Error: malformed {settings_path}: {e}")
            raise SystemExit(1)

    entries = build_hook_entries(scope)
    settings = merge_hooks(settings, entries)

    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _read_existing_api_key(env_path: Path) -> str | None:
    """Read the OpenAI API key from the .env file."""
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("TRACEMEM_OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def _write_api_key(env_path: Path, api_key: str) -> None:
    """Write the OpenAI API key to the .env file."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(f"TRACEMEM_OPENAI_API_KEY={api_key}\n")
    os.chmod(env_path, 0o600)


def _ensure_gitignore(claude_dir: Path) -> None:
    """Ensure .claude/.gitignore excludes the credentials file."""
    gitignore_path = claude_dir / ".gitignore"

    existing = ""
    if gitignore_path.exists():
        existing = gitignore_path.read_text()

    if GITIGNORE_ENTRY in existing:
        return

    with open(gitignore_path, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"# TraceMem credentials\n{GITIGNORE_ENTRY}\n")


def _prompt_api_key() -> str | None:
    """Prompt the user for their OpenAI API key."""
    print()
    print("TraceMem uses OpenAI embeddings for semantic search.")
    try:
        key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return key or None


def run_init(scope: str, *, force: bool = False) -> None:
    claude_dir = _resolve_target(scope)
    skill_dir = claude_dir / "skills" / "tracemem"
    env_path = skill_dir / ".env"

    if skill_dir.exists() and not force:
        print(f"TraceMem is already installed at {skill_dir}")
        print("Use --force to overwrite the existing installation.")
        # Still merge hooks in case settings.json is out of sync
        _merge_settings(claude_dir, scope)
        return

    # Preserve API key from existing .env before overwriting
    existing_key = _read_existing_api_key(env_path)

    if skill_dir.exists():
        shutil.rmtree(skill_dir)

    # Copy templates
    installed = _copy_templates(skill_dir)

    # Merge hooks into settings.json
    _merge_settings(claude_dir, scope)

    # Ensure .gitignore protects .env
    _ensure_gitignore(claude_dir)

    # Configure API key: restore existing, prompt, or skip
    api_key = existing_key
    if not api_key:
        api_key = _prompt_api_key()

    if api_key:
        _write_api_key(env_path, api_key)

    # Print summary
    print()
    print("TraceMem hooks installed successfully!")
    print()
    print(f"  Location: {skill_dir}")
    print(f"  Files: {len(installed)}")
    print()
    print("Configuration:")
    print(f"  {skill_dir / 'config.yaml'}")
    print()
    if not api_key:
        print("  ** OpenAI API key not configured **")
        print(f"  Add it to {env_path}")
        print("  or export TRACEMEM_OPENAI_API_KEY in your shell.")
        print()
    else:
        print(f"  API key stored in {env_path} (gitignored)")
        print()
    print("Prerequisite:")
    print("  uv — https://docs.astral.sh/uv/")
