# Development guide

## Setup

    python -m venv .venv
    pip install -e ".[dev]"

## Checks before opening a pull request

    python -m compileall -q src scripts tests
    pytest

GitHub Actions runs the same checks on every pull request (`.github/workflows/python-tests.yml`).

## Conventions

- Application code goes in `src/smartscholar/`. Import it with absolute package paths, for example `from smartscholar.storage.vector_store import VectorStore`.
- Operational scripts go in `scripts/`. They must not be imported by application code.
- Resource paths come from `smartscholar/config.py`. Do not use relative paths such as `"local/..."` or `"./qdrant_vdb"`.
- Never commit API keys, `.env` files, model weights, `qdrant_vdb/`, or `sessions.db` unless you have decided to track it.
- Commit messages follow the Conventional Commits style, for example `refactor(retrieval): ...`.

## Adding tests

Put tests that need no external services in `tests/unit/`. Put tests that need Qdrant, Ollama or downloaded models in `tests/integration/`, and only add them when they can be executed.
