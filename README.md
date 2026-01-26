# Project Scope

This is a pipeline with interactive graphics to automate the workflow of AI paper reading and curation. The main graph is a tree which classifies the papers into different categories/sub-categories based on their topic. This tree diagram needs to be interactive, such that a new paper can be added into it, and the tree diagram will be update afterwards. It will also allow user to click on each paper to perform additional operations. It will contain the following requirements:

1. ingest_paper
- take an user input of URL from arxiv (alternatively it might be the pdf itself, then you need to identify its source URL from internet) 
- download both the pdf, Latex code (if available) from its arxiv link into structured database

2. parse_pdf
- read the paper from pdf
- compute its embeddings
- provide a concise summary in markdown format include the aspects: 
    - what it does concretely, it needs to be a reasonable logical chain of thoughts, in which the rationale of the previous step justifies the next step, using plain language as much as possible, avoid confusing technical jargons (unless it's specifically named in the paper itself), avoid introducing figurative speech 
    - which area it mainly benefits (model performance / training efficiency / inference throughput / etc.). 
    - the rationale behind this benefit 
    - no duplication in the above content 
    - quantifiable results in the main area it is trying to improve 
    - if the paper contains multiple unique concepts (like DeepSeek-V3 technical report contains MTP, MLA, fine-grain FP8 etc), which are uncorrelated, please futher break the paper into multiple leave nodes (e.g. DeepSeek-v3-MTP, DeepSeek-v3-MLA, etc.) 

3. classify_paper
- adding each new paper as new node to the tree of classification
- using the embedding of this paper, classify the papers by their respective main area of contribution, e.g. primary level of classification can be any of the AI areas (e.g. dataset, evaluation, model architecture, inference, application, vision, speech, linguistic, RL etc). 
- if each category is getting crowded (>10), when add the new paper node, we need to review the classification to determine how shall we further branch out the current category into new sub-classes (using LLM to do this sub class creation and sub-classification), and then migrate the leave nodes under it into the new sub-classes 
- udpate the tree with the latest results of classification

4. git_repo
for each paper, the 1st option (from right-click dropdown menu) will be to obtain its open-source repo (if available), if no official one is available, identify if open-source replication is available as well. 

5. explain_references
for each paper, the 2nd option required is to "explain on key references". Once clicked, it will give me a full list of references in this paper, I can then:
- hover my mouse over each reference, it will trigger a LLM workflow to provide a concise description of this reference, and how is the reference related/contributing to the current paper
- click on the reference, so that it will add the reference into this diagram as a new paper, following all steps above

6. find_similar
for each paper, the 3rd option required is to "find similar papers". Once clicked, it will search in arxiv or google for the most similar 2-5 papers. Again, each will have a concise description of what this reference is about, and how is it similar to the paper 

7. all above info will be persisted in a locally structured database. please help me evaluate if such plan is feasible, and what are the key steps I need to take in n8n to implement such

# Resource

## vllm endpoint that you may need to use
LLM: OPENAI_API_BASE="http://localhost:8001"
VLM: OPENAI_API_BASE2="http://localhost:8002"
Embed: OPENAI_API_BASE3="http://localhost:8004"

## OSS tools

- arxiv.py: Reliable library for metadata + PDF download, can be used in requirement #1

- GROBID: PDF parsing & structured citation extraction, can be used in requirement #2

- PaperQA2 (paper-qa): Summarization + concept extraction + multi-turn QA, Scientifically tuned RAG wrapper, can be used in requirement #2 and #4

- PostgreSQL + pgvector: Database + embeddings + similarity search, can be used in all requirements

- react-d3-tree: Local front-end visualization and curation UI, can be used in all requirements

## Implementation Requirement
modularize the implementation, let's start bottomeup
- first install each of the OSS tools, test if the are working
- if so, then package each as a local MCP server, and then an agent skill, add such agent skill in cursor
- then let the cursor agent run the "Scope" as defined here, with such skills
- standardize runtime in the **single GROBID-based container** (no local uv/venv)

---

# Implementation Plan
This follows your ordering: **(1) package OSS as FastAPI services**, **(2) build internal blocks (UI + missing logic) and only then DB**, **(3) wire via LangGraph**, **(4) pause and reassess**.

### Milestone 1 — package OSS tools as local FastAPI services (with scripts)
- **Goal**: each OSS tool is callable locally via HTTP, with runnable scripts and known failure modes.
- Deliverables (single service):
  - `paper-curator-service`: arXiv resolve + download + GROBID extract + summarize + embed + optional QA
- **Local GROBID setup** (Docker):
  - Pull: `docker pull lfoppiano/grobid:0.8.0`
  - Build (unified container): `bash scripts/docker_mod.sh`
  - Run (unified container): `bash scripts/docker_run.sh`
  - Health: `GET ${service_base_url}/api/isalive`
- **Endpoint list**
  - `paper-curator-service`
    - `GET /health`
    - `POST /arxiv/resolve`
    - `POST /arxiv/download`
    - `POST /pdf/extract` (GROBID-backed)
    - `POST /summarize`
    - `POST /embed`
    - `POST /qa` (optional)
- **Scripts (replace unit tests)**:
  - One script per endpoint under `scripts/` (see `scripts/README.md`)
- **Exit criteria**: services run locally, `/health` passes, scripts succeed end-to-end for 2–3 representative papers.


### PaperQA2 findings (from paper-qa/README + code)
- **What we used naively vs built-in PaperQA2**
  - We manually chunk text and feed `Docs.aadd_texts`; PaperQA2 already provides parsing + chunking (`Docs.aadd` / `Docs.aadd_file`) with configurable chunk size/overlap, metadata validation, and PDF parsing hooks.
  - We run a simple summarize/QA path; PaperQA2’s `Docs.aget_evidence` + `Docs.aquery` implements retrieval, contextual summarization (summary LLM), and LLM re-ranking before answer generation.
  - We skip metadata inference; PaperQA2 can infer/attach citations, title/DOI/authors, and uses metadata in embeddings and ranking.
  - We bypass agentic search and evidence tooling; PaperQA2 provides agentic workflows (search → gather evidence → answer) and a “fake” deterministic path for lower token usage.

- **Capabilities we can still leverage for other scope sections**
  - Paper search + metadata aggregation across providers (Crossref, Semantic Scholar, OpenAlex, Unpaywall, retractions, journal quality).
  - Full-text indexing + reuse (local index, cached answers, `search` over prior answers/documents).
  - Hybrid/sparse+dense embeddings, configurable `evidence_k`, evidence relevance cutoff, and MMR settings.
  - Multimodal parsing/enrichment for figures/tables (media-aware contextual summaries).
  - External vector stores / caching hooks, plus settings presets and CLI profiles for reproducible runs.
  - Code/HTML/Office document ingestion for repository or artifact QA.

### Milestone 2 — internal building blocks (no DB yet)
- **Interactive UI** (React + `react-d3-tree`)
  - Render a local taxonomy state (file/in-memory)
  - Add “ingest paper” input (arXiv URL) and show status/progress
  - Node details panel: summary markdown + references list
- **Non-OSS requirements**
  - Taxonomy maintenance logic (crowding split + migration)
  - Hover caching + debouncing policy for reference explanations
  - Reference resolution + dedupe policy (canonical IDs, fuzzy title match)

### Milestone 3 — LangGraph orchestration (drop n8n for now)
- Build LangGraph graphs for Flows A–F:
  - `ingest_paper_graph`, `parse_and_summarize_graph`, `classify_and_update_tree_graph`
  - UI action graphs: `repo_lookup_graph`, `explain_reference_graph`, `find_similar_graph`
- **Exit criteria**: UI triggers a LangGraph run and receives structured results; failures are surfaced with clear errors + partial outputs where possible.

### Milestone 4 — database (last)
- Introduce PostgreSQL + `pgvector` for durability + search once the behaviors stabilize:
  - Persist papers/artifacts/embeddings
  - Persist taxonomy nodes/edges + rationales
  - Cache hover explanations + similarity results

### Milestone 5 — MCP servers + Cursor agent skills (optional/when ready)
- Wrap the internal backend endpoints (LangGraph + services) as MCP tools and add them as Cursor agent skills.

## Practical notes (to avoid common failure modes)
- **Debounce + cache** hover explanations (otherwise you’ll DDoS your own LLM).
- **Deduplicate papers** aggressively (canonical arXiv ID, DOI, title+authors fuzzy match).
- **Store rationales** (classification + subclass splits) so tree changes are explainable and reversible.
- **Version outputs** (summary, embedding model) to avoid silent drift when you change prompts/models.