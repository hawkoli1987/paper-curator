# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Execution Rules

- **Always use tmux sessions** for running commands:
  - `server2` — start/restart the backend server
  - `paper-curator` — run tests and general commands
- **Never run** `curl`, `make test`, `uvicorn`, `pytest`, or service scripts directly in the shell tool. Always send them to the appropriate tmux session via `tmux send-keys`.
- Python dependencies live in the `uv` env `paper-curator`. If a dependency is missing, install it in-container and update `pyproject.toml`.

## Backend Startup & Restart (Critical)

**The only correct way to start/restart the backend** (send to `server2` tmux session):

```bash
cd /scratch_aisg/SPEC-SF-AISG/yuli/ARF-Training/repos/paper-curator
PYTHONPATH=src/backend .venv/bin/uvicorn app:app --app-dir src/backend --host 0.0.0.0 --port 3100
```

**Why this exact form matters:**
- `cd` to the ARF-Training repo first — `--app-dir src/backend` is resolved relative to CWD, so the correct `app.py` and modules are loaded.
- Use `.venv/bin/uvicorn` (explicit repo-local path) — the `server2` shell may have a different repo's venv on PATH; bare `uvicorn` picks up the wrong one.
- `PYTHONPATH=src/backend` — lets Python resolve intra-backend imports (`import db`, `import config`, etc.).
- **No `--reload`** — the server does NOT auto-reload on code changes. **Every code modification requires a manual restart.**

**After any backend code change, always:**
1. Send `C-c` to `server2` to stop the running server.
2. Re-send the startup command above.
3. Wait for the server to be healthy: `curl http://127.0.0.1:3100/health`
4. Tell the user: **"Server restarted — please test from the frontend."**

Skipping the restart means the old code keeps running and changes have no effect. Always complete the restart before reporting the task as done.

**Verify the right backend is running** (quick smoke test after restart):
```bash
# Should return our custom message, not Pydantic's generic "field required"
curl -s -X POST http://127.0.0.1:3100/summarize/structured \
  -H "Content-Type: application/json" -d '{}' | python3 -c \
  "import sys,json; print(json.load(sys.stdin).get('detail',''))"
# Expected: "Provide pdf_path or arxiv_id (for already-indexed papers)"
```

## Common Commands

```bash
# Install all dependencies
make install

# Run backend locally (send to server2 tmux session) — see "Backend Startup & Restart" above
PYTHONPATH=src/backend .venv/bin/uvicorn app:app --app-dir src/backend --host 0.0.0.0 --port 3100

# Run frontend locally
cd src/frontend && npm run dev

# Docker (local dev)
make run          # docker compose up --build
make docker-stop

# HPC (Singularity)
make singularity-run     # start all services
make singularity-stop    # stop all services
./scripts/hpc-services.sh status

# Slack ingestion
make pull-slack
```

### Testing

```bash
# All tests (validation + functional + e2e) except integration — the standard full run
make test
# Fast run skipping LLM-dependent tests:
SKIP_LLM=1 make test

# By category (for focused debugging only — not a substitute for the full run)
make test validation     # Connectivity + input validation (fast, no LLM)
make test functional     # Single-capability tests (requires backend + LLM)
make test integration    # End-to-end workflows (auto-switches to test DB)

# Single test file
BACKEND_URL=http://localhost:3100 pytest tests/functional/test_paper_operations.py -v -s

# Manage test database
make test-db-init        # Create paper_curator_test DB
make test-db-reset       # Drop and recreate test DB
```

**Testing conventions:**
- `make test` always includes e2e tests — do **not** add separate `make test e2e` or similar targets.
- e2e tests run against the production database; they are the primary regression gate for all endpoints.
- `SKIP_LLM=1` skips LLM-dependent tests for speed; always run without it before marking a task done.

Test categories require different services:
| Category | Backend | LLM | Database |
|---|---|---|---|
| validation | For connectivity tests | No | Production |
| functional | Yes | Yes | Production |
| e2e | Yes | Yes (skippable) | Production |
| integration | Yes | Yes | Test DB (auto-switched) |

## Architecture

```
Frontend (Next.js :3000) → Backend (FastAPI :3100) → PostgreSQL w/ pgvector
                                    ↓
                         LLM endpoint + Embedding endpoint (OpenAI-compatible)
                         External APIs (arXiv, Semantic Scholar, GitHub)
```

### Backend (`src/backend/`)

FastAPI app (`app.py`) with ~60 endpoints. Key modules:

- **`app.py`** — All API routes. Long operations (`/papers/classify`, `/papers/batch-ingest`) are async.
- **`db.py`** — All PostgreSQL operations via psycopg2. Supports runtime DB switching (`switch_database()`) for test isolation. DB config loaded from `config/config.yaml` with env var overrides (`PGHOST`, `PGPORT`, etc.).
- **`rag.py`** — Custom RAG: pymupdf PDF extraction → character chunking → embedding via AsyncOpenAI → pgvector storage → cosine similarity retrieval → LLM answer generation. Replaces PaperQA2.
- **`clustering.py`** — Hierarchical paper classification: L2-normalized embeddings → divisive k-means with silhouette scoring → LLM-generated category names → JSONB tree in DB.
- **`config.py`** — Config loading from `config/config.yaml` with DB setting overrides. Settings can be overridden at runtime via `POST /config` (stored in DB `settings` table).
- **`naming.py`** — LLM calls for generating category names during clustering.
- **`llm_clients.py`** — OpenAI-compatible client factory with ngrok/HPC support.
- **`external_clients.py`** — Shared httpx async HTTP client pool.
- **`prompts/prompts.json`** — All LLM prompt templates (referenced by ID).

### Frontend (`src/frontend/`)

Next.js 14 app with App Router. Key patterns:

- **API routes** (`src/app/api/*/route.ts`) — Proxy requests to the FastAPI backend. Long-running endpoints use `backendPost()` from `src/lib/backend-proxy.ts` (node:http, bypasses undici's 300s timeout).
- **`next.config.js`** rewrites — Simple passthroughs to the backend for short-lived requests. Long-lived endpoints have dedicated API routes instead.
- **`BACKEND_URL` env var** — Backend address. Default: `http://backend:8000` (Docker), override to `http://localhost:3100` for HPC/local.

### Data Flow: Paper Ingestion

1. `POST /papers/batch-ingest` → fetch from Slack or local directory
2. For each arXiv ID: download PDF → extract text → generate embedding → store in `papers` table
3. Chunk PDF text and store in `paper_chunks` table for RAG
4. If `rebuild_on_ingest: true`, trigger `/papers/classify`

### Data Flow: Classification

1. `POST /papers/classify` → load all paper embeddings from DB
2. `clustering.py`: L2-normalize → recursive k-means (silhouette-scored k selection) → build tree
3. `naming.py`: LLM names each cluster based on paper titles in cluster
4. Save tree as JSONB to DB `tree_structure` table

### Data Flow: Topic Query (multi-paper RAG)

1. `POST /topic/search` → embed query → cosine similarity search in `paper_chunks`
2. User selects papers → `POST /topic/{id}/papers`
3. `POST /topic/{id}/query` → retrieve top-k chunks per paper → LLM answer

## Configuration

`config/config.yaml` is the primary config. Settings hierarchy:
1. DB `settings` table (runtime overrides via `POST /config`)
2. `config/config.yaml`
3. Environment variables (`PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`)

Key config sections: `server`, `database`, `endpoints` (LLM + embedding URLs), `rag`, `classification`, `topic_query`, `ui`, `ingestion`, `slack`.

**LLM/Embedding endpoints** must be OpenAI-compatible. `localhost` URLs are auto-converted to `host.docker.internal` in Docker.

## Storage Layout

```
storage/
├── downloads/          # Downloaded PDFs: {arxiv_id}.{title}.pdf
├── schemas/            # Debug output from naming/clustering
└── ...
tests/storage/downloads/  # 10 sample PDFs committed to git (landmark papers)
```

## Deployment

- **Docker**: `src/compose.yml` — backend on :8000, frontend on :3000, PostgreSQL on :5432
- **HPC/Singularity**: `containers/*.sif` — backend on :3100, frontend on :3000, PostgreSQL on :5432
  - Slack token: `~/.ssh/.slack` (chmod 600)
  - Services managed via `scripts/hpc-services.sh`

## Code Conventions

- Use `assert` for preconditions and invariants; avoid try-except for control flow.
- Prefer existing library functions over manual implementations.
- Python scripts should be runnable standalone (accept CLI args where appropriate).
- Never query DB columns that include `embedding` (pgvector returns `numpy.ndarray`, causing Pydantic serialization errors). Use explicit column lists excluding `embedding`.
- File writes must go to `storage/` (bound writable in Singularity), not project-relative paths.

## Git Conventions

- **Never add a `Co-Authored-By` trailer** to commits. Only the user's authorship is recorded.
