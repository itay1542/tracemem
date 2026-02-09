---
name: ops
description: Build, release, and monitor CI/CD for TraceMem packages. Use for version bumping, publishing to PyPI, watching GitHub Actions, and checking release status.
user_invocable: true
context: fork
argument-hint: [build | release | ci | status]
---

# TraceMem Ops

Build, release, and monitor CI/CD for the TraceMem monorepo.

## Repository

- **GitHub**: `itay1542/tracemem`
- **Packages**: `tracemem-core` (core library), `tracemem-claude` (Claude Code hooks installer)
- **PyPI**: https://pypi.org/project/tracemem-core/ / https://pypi.org/project/tracemem-claude/

## Makefile Commands

All ops commands are available via `make`:

```bash
make help              # Show all available commands
make build             # Build both packages locally
make test              # Run unit + kuzu tests (no external deps)
make test-core         # Run all tests (requires Neo4j + OpenAI)
make lint              # Ruff check
make format            # Ruff format + fix

# Release (creates branch + PR; merge to publish)
make release V=0.2.0   # Explicit version
make release-patch     # 0.1.0 -> 0.1.1
make release-minor     # 0.1.0 -> 0.2.0
make release-major     # 0.1.0 -> 1.0.0
```

## Release Workflow

### 1. Pre-release checks

```bash
make lint              # Ensure code passes linting
make test              # Ensure tests pass
make build             # Ensure packages build
```

### 2. Create release PR

```bash
make release-patch     # or release-minor / release-major / release V=x.y.z
```

This:
1. Creates a `release/vX.Y.Z` branch
2. Bumps version in both `tracemem_core/pyproject.toml` and `tracemem-installer/pyproject.toml`
3. Commits, pushes, and opens a PR

### 3. Merge the PR

Merging the `release/v*` PR to main automatically:
1. `release.yml` creates and pushes a `vX.Y.Z` git tag
2. `publish.yml` (triggered by the tag) builds and publishes to PyPI via OIDC

## GitHub Actions Workflows

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Push to main, PRs | Lint (ruff check + format), test, build |
| `release.yml` | PR merge from `release/v*` branch | Auto-creates git tag |
| `publish.yml` | Tags matching `v*` | Build + publish to PyPI (OIDC) |
| `integration.yml` | Manual / weekly Monday 06:00 UTC | Neo4j integration tests via Docker |

## Monitoring CI/CD

### Watch the latest CI run

```bash
gh run list --repo itay1542/tracemem --limit 5
gh run watch --repo itay1542/tracemem <RUN_ID>
```

### View failed job logs

```bash
gh run view <RUN_ID> --repo itay1542/tracemem --log-failed
```

### Check publish workflow after tagging

```bash
gh run list --repo itay1542/tracemem --workflow=publish.yml --limit 3
```

### Trigger integration tests manually

```bash
gh workflow run integration.yml --repo itay1542/tracemem
```

## Checking Release Status

### Current version

```bash
grep -m1 'version = ' tracemem_core/pyproject.toml
```

### Latest on PyPI

```bash
pip index versions tracemem-core
pip index versions tracemem-claude
```

### Git tags

```bash
git tag --sort=-v:refname | head -5
```

## PyPI Trusted Publisher Setup

Both packages use OIDC trusted publishing (no API tokens). Configuration at https://pypi.org/manage/account/publishing/:

| Field | Value |
|-------|-------|
| Owner | `itay1542` |
| Repository | `tracemem` |
| Workflow | `publish.yml` |
| Environment | `pypi` |

Also requires a `pypi` environment in GitHub repo Settings > Environments.

## Troubleshooting

### CI lint fails
```bash
make format            # Auto-fix formatting
make lint              # Verify
```

### CI tests fail
```bash
make test-v            # Run locally with verbose output
```

### Publish fails with "trusted publisher" error
- Verify the trusted publisher is configured on PyPI for both packages
- Verify the `pypi` environment exists in GitHub repo settings
- Check that `publish.yml` has `permissions: id-token: write`

### Neo4j tests fail locally
```bash
docker compose up -d neo4j    # Start Neo4j
make test-core                 # Run all tests including Neo4j
```
