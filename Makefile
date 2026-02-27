SHELL := /bin/bash
PYTHON := python
VENV_DIR := .venv
ACTIVATE := source $(VENV_DIR)/bin/activate

.PHONY: install install-frontend test run clean docker-build docker-run \
        singularity-build singularity-run singularity-stop pull-slack \
        test-db-init test-db-reset \
        validation functional integration

# Create virtual environment and install all dependencies
install: $(VENV_DIR)/bin/activate install-frontend

$(VENV_DIR)/bin/activate: pyproject.toml
	test -d $(VENV_DIR) || python -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -e ".[dev]"
	touch $(VENV_DIR)/bin/activate

install-frontend:
	@echo "Installing frontend dependencies..."
	@command -v npm >/dev/null 2>&1 || { echo "Error: npm is not installed"; exit 1; }
	cd src/frontend && npm install
	@echo "Frontend dependencies installed."

# =============================================================================
# Testing (assumes .venv is activated in tmux)
# =============================================================================
# Usage: make test [type]
#   make test                 - Run ALL tests (validation + functional + e2e; except integration)
#   make test validation      - Run validation tests only
#   make test functional      - Run functional tests only
#   make test integration     - Run integration tests (auto-switches to test DB)
#
# e2e tests are always included in the default 'make test' run (not a separate target).
# LLM-dependent e2e tests can be skipped for fast runs: SKIP_LLM=1 make test
#
# The integration tests automatically:
#   1. Create paper_curator_test DB if it doesn't exist (via /db/init)
#   2. Switch the backend to paper_curator_test (via /db/switch)
#   3. Run tests against the clean test database
#   4. Switch the backend back to paper_curator (production)
#
# The e2e tests run against the PRODUCTION database:
#   - Read-only tests run freely
#   - Write tests create isolated data and clean up after themselves

BACKEND_URL ?= http://localhost:3100
TEST_DB_NAME := paper_curator_test
TEST_TYPE := $(word 2,$(MAKECMDGOALS))

test:
ifeq ($(TEST_TYPE),validation)
	@echo "=== Running validation tests ==="
	BACKEND_URL=$(BACKEND_URL) pytest tests/validation -v
else ifeq ($(TEST_TYPE),functional)
	@echo "=== Running functional tests ==="
	BACKEND_URL=$(BACKEND_URL) pytest tests/functional -v -s
else ifeq ($(TEST_TYPE),integration)
	@echo "=== Running integration tests (using test DB: $(TEST_DB_NAME)) ==="
	BACKEND_URL=$(BACKEND_URL) TEST_DB_NAME=$(TEST_DB_NAME) pytest tests/integration -v -s
else
	@echo "=== Running all tests: validation + functional + e2e (except integration and deprecated) ==="
	BACKEND_URL=$(BACKEND_URL) SKIP_LLM=$(SKIP_LLM) pytest tests -v -s --ignore=tests/integration --ignore=tests/deprecated
endif

validation functional integration:
	@:

test-db-init:
	@echo "=== Initializing test database via backend API ==="
	@curl -sf -X POST $(BACKEND_URL)/db/init \
		-H "Content-Type: application/json" \
		-d '{"database": "$(TEST_DB_NAME)", "drop_existing": false}' \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print(f'DB: {d[\"database\"]} - {d[\"status\"]}')"

test-db-reset:
	@echo "=== Resetting test database (drop + recreate) ==="
	@curl -sf -X POST $(BACKEND_URL)/db/init \
		-H "Content-Type: application/json" \
		-d '{"database": "$(TEST_DB_NAME)", "drop_existing": true}' \
		| python3 -c "import sys,json; d=json.load(sys.stdin); print(f'DB: {d[\"database\"]} - {d[\"status\"]}')"

# =============================================================================
# Development
# =============================================================================

connect:
	./scripts/connect_endpoint.sh

run:
	docker compose -f src/compose.yml up --build

run-local:
	cd src/backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose -f src/compose.yml build --no-cache

docker-stop:
	docker compose -f src/compose.yml down

# =============================================================================
# Singularity/Apptainer targets
# =============================================================================

CONTAINER_DIR := containers
SIF_BACKEND := $(CONTAINER_DIR)/backend.sif
SIF_FRONTEND := $(CONTAINER_DIR)/frontend.sif
SIF_DB := $(CONTAINER_DIR)/pgvector.sif
CONFIG_FILE := config/config.yaml

singularity-build:
	@mkdir -p $(CONTAINER_DIR)
	singularity pull --force $(SIF_DB) docker://pgvector/pgvector:pg16
	singularity build --force --fakeroot $(SIF_BACKEND) $(CONTAINER_DIR)/backend.def
	singularity build --force --fakeroot $(SIF_FRONTEND) $(CONTAINER_DIR)/frontend.def

singularity-build-backend:
	@mkdir -p $(CONTAINER_DIR)
	singularity build --force --fakeroot $(SIF_BACKEND) $(CONTAINER_DIR)/backend.def

singularity-run:
	./scripts/hpc-services.sh start

singularity-stop:
	./scripts/hpc-services.sh stop

# =============================================================================
# Slack Integration
# =============================================================================

SLACK_TOKEN_FILE := $(HOME)/.ssh/.slack

pull-slack:
	@if [ ! -f "$(SLACK_TOKEN_FILE)" ]; then \
		echo "Error: Slack token not found at $(SLACK_TOKEN_FILE)"; exit 1; \
	fi
	@SLACK_TOKEN=$$(cat "$(SLACK_TOKEN_FILE)") && \
	grep -A 100 "^slack:" "$(CONFIG_FILE)" | grep "^\s*-\s*http" | sed 's/^\s*-\s*//' | while read channel; do \
		echo "=== Processing: $$channel ==="; \
		curl -s -X POST http://localhost:3100/papers/batch-ingest \
			-H "Content-Type: application/json" \
			-d "{\"slack_channel\": \"$$channel\", \"slack_token\": \"$$SLACK_TOKEN\"}" | \
		python3 -c "import sys,json; d=json.load(sys.stdin); \
			print(f'Result: {d.get(\"success\",0)} success, {d.get(\"skipped\",0)} skipped, {d.get(\"errors\",0)} errors')"; \
	done;

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
