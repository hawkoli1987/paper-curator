# Scripts

## Prerequisites
- `paper-curator-service` and GROBID running on `service_base_url` in `config/paperqa.yaml`
- OpenAI-compatible endpoints running on the host for model resolution:
  - `openai_api_base` (LLM) at `https://transmissively-conidial-fredrick.ngrok-free.dev`
  - `openai_api_base2` (VLM) at `http://localhost:8002/v1`
  - `openai_api_base3` (Embeddings) at `http://localhost:8004/v1`

## Start GROBID
```bash
bash scripts/docker_mod.sh
bash scripts/docker_run.sh
```
`docker_run.sh` also adds a `host.docker.internal` entry so the container can reach
local OpenAI-compatible endpoints on your host (ports from `config/paperqa.yaml`).

## Endpoint demos (one endpoint per script)
```bash
# Health check (defaults to config service_base_url)
bash scripts/health.sh

# Resolve metadata for a single arXiv ID
bash scripts/arxiv_resolve.sh --arxiv-id 1706.03762

# Download PDF/LaTeX for a single arXiv ID
bash scripts/arxiv_download.sh --arxiv-id 1706.03762

# Extract PDF text (pass pdf_path from the download response)
bash scripts/pdf_extract.sh --pdf-path storage/downloads/1706.03762v7.Attention_Is_All_You_Need.pdf

# If you need the pdf_path from the download JSON:
python services/scripts/extract_pdf_path.py --download-json storage/outputs/arxiv_download.json

# Summarize (PDF-first; falls back to extract.json if no --pdf-path)
bash scripts/summarize.sh --pdf-path storage/downloads/1706.03762v7.Attention_Is_All_You_Need.pdf

# Embed (uses storage/outputs/extract.json as input)
bash scripts/embed.sh

# QA (PDF-first; context file still supported)
bash scripts/qa.sh --pdf-path storage/downloads/1706.03762v7.Attention_Is_All_You_Need.pdf \
  --question "What is the main contribution?"
```

## What each script does (logic + model/config usage)
- `health.sh`
  - Calls `GET /health` to confirm the FastAPI service is alive behind Nginx.
  - No PaperQA2/GROBID usage; just a service liveness check.

- `arxiv_resolve.sh`
  - Calls `POST /arxiv/resolve`, which uses the `arxiv` Python library to resolve IDs/URLs.
  - Saves metadata to `storage/outputs/arxiv_resolve.json`.

- `arxiv_download.sh`
  - Calls `POST /arxiv/download`, which uses the `arxiv` Python library to download PDF/LaTeX.
  - Writes PDFs under `storage/downloads/` and records paths in `storage/outputs/arxiv_download.json`.

- `pdf_extract.sh`
  - Calls `POST /pdf/extract`.
  - **PaperQA2 path (default):** uses PaperQA2's native PDF parser via `paperqa.readers.read_doc`, with chunking config from `config/paperqa.yaml` (`paperqa_chunk_chars`, `paperqa_chunk_overlap`, `paperqa_use_doc_details`), and `multimodal` set to OFF to avoid image captioning.
  - **GROBID fallback:** if PaperQA2 parsing fails or `force_grobid=true`, calls GROBID at `POST /api/processFulltextDocument` and returns extracted sections.
  - Output is written to `storage/outputs/extract.json` with a `parser` field indicating `paperqa` or `grobid`.

- `summarize.sh`
  - Calls `POST /summarize`.
  - **PaperQA2 flow:** adds the PDF (or text) into `Docs`, then runs retrieval + contextual summaries + answer generation via `Docs.aquery`.
  - **LLM model:** resolved dynamically from `openai_api_base` (`/v1/models`) and passed to PaperQA2 as `openai/<model_id>`.
  - **Embedding model:** resolved from `openai_api_base3` and passed as `openai/<embed_model_id>` with `encoding_format=float`.
  - **Configs applied:** chunking (`paperqa_chunk_chars`, `paperqa_chunk_overlap`), evidence controls (`paperqa_evidence_k`, `paperqa_evidence_summary_length`, `paperqa_evidence_skip_summary`, `paperqa_evidence_relevance_score_cutoff`), and `paperqa_use_doc_details` (with automatic retry that disables metadata lookup on 429s).

- `embed.sh`
  - Calls `POST /embed`.
  - Uses the embedding model resolved from `openai_api_base3` (`/v1/models`) and performs an OpenAI-compatible embedding request.
  - Input text is built from `storage/outputs/extract.json` (prefers PaperQA2 `text`, falls back to GROBID sections).

- `qa.sh`
  - Calls `POST /qa`.
  - **PaperQA2 flow:** adds the PDF (or a provided context file) to `Docs`, then runs `Docs.aquery` for QA.
  - **LLM model:** resolved dynamically from `openai_api_base` (`/v1/models`) as `openai/<model_id>`.
  - **Embedding model:** resolved from `openai_api_base3` as `openai/<embed_model_id>` with `encoding_format=float`.
  - Uses the same PaperQA2 settings as `summarize.sh` (chunking + evidence settings + doc-details retry).
