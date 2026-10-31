.PHONY: sync lint fmt typecheck test eval check

sync:
	uv sync --locked

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

eval:
	uv run python -m evals

check: lint typecheck test eval
