# Contributing

Thanks for your interest in improving Semantic-Free Code Embedding (SFCE).

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
# Core install (CPU/macOS friendly — enough to import the package and run tests)
uv sync

# With the optional GPU / training / serving extras
uv sync --extra train --extra serve --extra eval
```

## Code quality

We use [ruff](https://docs.astral.sh/ruff/) for both formatting and linting, and
[pre-commit](https://pre-commit.com/) to run it automatically:

```bash
uv run pre-commit install      # one-time setup
uv run ruff format .           # format
uv run ruff check . --fix      # lint + autofix
```

## Tests

```bash
uv run pytest
```

The test suite runs on a core (CPU) install and does not require a GPU, model
weights, or the datasets. Please add tests for new pure/utility code and keep
the import smoke test (`tests/test_imports.py`) green.

## Submitting changes

1. Create a feature branch (do not commit directly to `main`).
2. Make your change with accompanying tests; ensure `ruff` and `pytest` pass.
3. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
   messages (e.g. `fix(eval): ...`, `feat(modeling): ...`).
4. Open a pull/merge request for review.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0 that covers this project.
