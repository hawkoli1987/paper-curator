SHELL := /bin/bash
PYTHON := /Users/hyl/miniconda3/bin/python3.12
VENV_DIR := .venv
ACTIVATE := source $(VENV_DIR)/bin/activate

.PHONY: install test run clean docker-build docker-run

# Create virtual environment and install all dependencies
install: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: pyproject.toml
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -e ".[dev]"
	touch $(VENV_DIR)/bin/activate

# Run pytest tests
test: install
	pytest tests -v

# Start docker-compose stack (frontend + backend)
run:
	docker compose -f src/compose.yml up --build

# Run backend locally with uvicorn (for development)
# Assumes .venv is already activated
run-local:
	cd src/backend && uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Build docker images without starting
docker-build:
	docker compose -f src/compose.yml build

# Stop and remove containers
docker-stop:
	docker compose -f src/compose.yml down

# Clean up virtual environment and cached files
clean:
	rm -rf $(VENV_DIR)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
