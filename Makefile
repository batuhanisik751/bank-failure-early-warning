.PHONY: setup fix-venv test lint format ingest panel labels features train evaluate

setup:
	uv sync
	uv run pre-commit install
	$(MAKE) fix-venv

# Python 3.12 skips .pth files flagged hidden on macOS; see scripts/fix_venv.py.
fix-venv:
	uv run --no-sync python scripts/fix_venv.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

ingest:
	uv run bankcanary ingest

panel:
	uv run bankcanary build-panel

labels:
	uv run bankcanary build-labels

features:
	uv run bankcanary build-features

train:
	uv run bankcanary train --model logit

evaluate:
	uv run bankcanary evaluate
