.PHONY: setup fix-venv test lint format ci ingest panel labels features dq train evaluate p1 \
	crosswalk macro features-v2 train-gbdt train-hazard walkforward walkforward-h8 calibrate \
	metrics-report explain promote sensitivity p2 db-up db-down publish publish-dry refresh \
	web-install web-check web-build web-start web-e2e web-lighthouse

# ---- Prototype 1: setup and foundation ----------------------------------------

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

# What .github/workflows/ci.yml runs: Python lint + tests, then the web checks (no database).
ci: lint test web-check

ingest:
	uv run bankcanary ingest

panel:
	uv run bankcanary build-panel

labels:
	uv run bankcanary build-labels

features:
	uv run bankcanary build-features

dq:
	uv run bankcanary dq-report

train:
	uv run bankcanary train --model all

evaluate:
	uv run bankcanary evaluate

p1: ingest panel labels features dq train evaluate

# ---- Prototype 2: depth (each walk-forward year is one call, so no step runs long) ----

YEARS ?= $(shell seq 2008 2024)
YEARS_H8 ?= $(shell seq 2008 2023)

crosswalk:
	uv run bankcanary build-crosswalk

macro:
	uv run bankcanary build-macro

features-v2:
	uv run bankcanary build-features-v2

train-gbdt:
	uv run bankcanary train-gbdt

train-hazard:
	uv run bankcanary train-hazard

walkforward:
	for Y in $(YEARS); do uv run bankcanary walkforward --year $$Y --model all || exit 1; done

walkforward-h8:
	for Y in $(YEARS_H8); do uv run bankcanary walkforward --year $$Y --model logit,gbdt,hazard --horizon 8 || exit 1; done

calibrate:
	uv run bankcanary calibrate --all-years
	uv run bankcanary calibrate --all-years --horizon 8

metrics-report:
	uv run bankcanary metrics-report

explain:
	uv run bankcanary explain --all

promote:
	uv run python scripts/promote_production_models.py

sensitivity:
	uv run bankcanary sensitivity

p2: crosswalk macro features-v2 train-gbdt train-hazard walkforward walkforward-h8 calibrate metrics-report explain promote sensitivity

# ---- Prototype 3: database, publish, refresh and the web app ----------------------
# DATABASE_URL comes from the git-ignored .env (bankcanary.config.load_secrets and
# web/next.config.ts both read it); nothing here needs it on the command line.

db-up:
	docker compose up -d

db-down:
	docker compose down

publish:
	uv run bankcanary publish

publish-dry:
	uv run bankcanary publish --dry-run

refresh:
	uv run bankcanary refresh

web-install:
	cd web && npm ci

# Lint, typecheck and unit tests: the web half of ci.yml, no database needed.
web-check:
	cd web && npm run lint && npm run typecheck && npm test

web-build:
	cd web && npm run build

web-start: web-build
	cd web && npm run start

# Playwright + axe over the built app against the local database.
web-e2e: web-build
	cd web && npm run test:e2e

web-lighthouse: web-build
	cd web && npm run lighthouse
