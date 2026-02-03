# tracemem-claude

Memory hooks installer for Claude Code. Captures your coding interactions into a knowledge graph for contextual recall.

## Install

```bash
uvx tracemem-claude init          # local mode (per-project)
uvx tracemem-claude init --global # global mode (~/.claude/)
```

## Uninstall

```bash
uvx tracemem-claude uninstall
uvx tracemem-claude uninstall --global
```

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — required for hook execution (PEP 723 script resolution)
- `TRACEMEM_OPENAI_API_KEY` environment variable — OpenAI API key for embeddings

## Configuration

After install, edit `.claude/skills/tracemem/config.yaml` to customize settings.
