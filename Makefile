.PHONY: install dev test test-core test-integration test-v test-cov lint format typecheck build release release-patch release-minor release-major clean clean-all help

# Default target
help:
	@echo "TraceMem Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install dependencies"
	@echo "  make dev         Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test        Run unit + kuzu tests (no external deps)"
	@echo "  make test-core   Run all core tests (requires Neo4j + OpenAI)"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-v      Run tests with verbose output"
	@echo "  make lint        Run linter (ruff)"
	@echo "  make format      Format code (ruff)"
	@echo "  make typecheck   Run type checker (pyright)"
	@echo ""
	@echo "Build & Release:"
	@echo "  make build       Build all packages"
	@echo "  make release V=0.2.0  Tag and push a release (triggers PyPI publish)"
	@echo "  make release-patch    Bump patch version (0.1.0 -> 0.1.1) and release"
	@echo "  make release-minor    Bump minor version (0.1.0 -> 0.2.0) and release"
	@echo "  make release-major    Bump major version (0.1.0 -> 1.0.0) and release"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       Remove build artifacts and caches"
	@echo "  make clean-all   Remove everything including .venv"

# Setup
install:
	uv sync --all-packages

dev:
	uv sync --all-packages --all-extras

# Development
test:
	uv run pytest tracemem_core/tests/ -m "not neo4j and not openai"

test-core:
	uv run pytest tracemem_core/tests/ -v

test-integration:
	uv run pytest tracemem_core/tests/integration/ -v

test-v:
	uv run pytest tracemem_core/tests/ -m "not neo4j and not openai" -v

test-cov:
	uv run pytest tracemem_core/tests/ -m "not neo4j and not openai" --cov=tracemem_core/src/tracemem_core --cov-report=html

lint:
	uv run ruff check tracemem_core/src/ tracemem_core/tests/

format:
	uv run ruff format tracemem_core/src/ tracemem_core/tests/
	uv run ruff check --fix tracemem_core/src/ tracemem_core/tests/

typecheck:
	uv run mypy tracemem_core/src/

# Build & Release
build:
	uv build --package tracemem-core
	uv build --package tracemem-claude

# Get current version from tracemem_core/pyproject.toml
CURRENT_VERSION := $(shell grep -m1 'version = ' tracemem_core/pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
MAJOR := $(word 1,$(subst ., ,$(CURRENT_VERSION)))
MINOR := $(word 2,$(subst ., ,$(CURRENT_VERSION)))
PATCH := $(word 3,$(subst ., ,$(CURRENT_VERSION)))

release-patch:
	@$(MAKE) release V=$(MAJOR).$(MINOR).$(shell echo $$(($(PATCH)+1)))

release-minor:
	@$(MAKE) release V=$(MAJOR).$(shell echo $$(($(MINOR)+1))).0

release-major:
	@$(MAKE) release V=$(shell echo $$(($(MAJOR)+1))).0.0

release:
ifndef V
	$(error Usage: make release V=x.y.z)
endif
	@echo "Releasing v$(V) (current: $(CURRENT_VERSION))"
	sed -i '' 's/^version = ".*"/version = "$(V)"/' tracemem_core/pyproject.toml
	sed -i '' 's/^version = ".*"/version = "$(V)"/' tracemem-installer/pyproject.toml
	git add tracemem_core/pyproject.toml tracemem-installer/pyproject.toml
	git commit -m "release: v$(V)"
	git tag v$(V)
	@echo ""
	@echo "Tagged v$(V). Push with:"
	@echo "  git push origin main --tags"

# Maintenance
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf .venv/
