.PHONY: install test test-unit test-integration test-graph lint format check clean neo4j-up neo4j-down

# Install dependencies
install:
	uv sync --dev

# Run all tests
test:
	uv run pytest tests/ -v

# Run unit tests only (no API key needed)
test-unit:
	uv run pytest tests/storage/ tests/test_models.py tests/test_tracemem.py -v

# Run integration tests (requires OPENAI_API_KEY in .env)
test-integration:
	uv run pytest tests/integration/ -v -m openai

# Run graph store integration tests (requires Neo4j running)
test-graph:
	uv run pytest tests/storage/graph/ -v

# Start Neo4j for integration tests
neo4j-up:
	docker compose -f ../docker-compose.yml up -d neo4j
	@echo "Waiting for Neo4j to be ready..."
	@sleep 5

# Stop Neo4j
neo4j-down:
	docker compose -f ../docker-compose.yml down

# Run linter
lint:
	uv run ruff check src/ tests/

# Format code
format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# Run all checks (lint + test)
check: lint test

# Clean up cache files
clean:
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
