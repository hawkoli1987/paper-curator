# Paper Curator — Bug Report

Bugs identified through static code analysis and live endpoint testing (2026-02-27).
Backend: `src/backend/` · Tests run: 53/53 passing (validation + functional).

---

## BUG-001 · Backend startup: `config/config.yaml` not found when run from `src/backend/`

**Severity:** Critical
**Affected endpoints:** All LLM-backed endpoints — `/summarize`, `/qa`, `/qa/structured`, `/summarize/structured`, `/classify`, `/abbreviate`, `/papers/batch-ingest`, `/papers/classify`, `/papers/reabbreviate`, `/papers/reabbreviate-all`, `/summary/merge`, `/summary/dedup`, `/topic/{id}/query`, `/references/explain`

**Symptom:**
HTTP 500 — `"Config file not found: config/config.yaml"` on any endpoint that reads RAG, LLM endpoint, or classification settings.

**Root cause:**
`config.py:46` opens `pathlib.Path("config/config.yaml")` — a path relative to the process working directory. The Makefile and CLAUDE.md both instruct starting the server with `cd src/backend && uvicorn app:app ...`, so the CWD is `src/backend/`. The directory `src/backend/config/` exists but is **empty** — it is a leftover Singularity bind-mount point. The real `config/config.yaml` lives at the repo root and is only made available inside the Singularity container via `--bind ${PROJECT_ROOT}/config:/app/config`. Local/HPC bare-metal runs hit this immediately.

```python
# config.py:44-47
def _load_config() -> dict[str, Any]:
    config_path = pathlib.Path("config/config.yaml")   # CWD-relative → wrong when CWD=src/backend
    if not config_path.exists():
        raise HTTPException(status_code=500, detail="Config file not found: config/config.yaml")
```

**Workaround (in use):**
Start uvicorn from repo root: `PYTHONPATH=src/backend uvicorn app:app --app-dir src/backend --host 0.0.0.0 --port 3100`

**Files:** `src/backend/config.py:44–47`, `Makefile:88`

---

## BUG-002 · Unguarded `response.choices[0]` access across all LLM calls

**Severity:** Critical
**Affected endpoints:** `/summary/merge`, `/summary/dedup`, `/classify`, `/abbreviate`, `/papers/reabbreviate`, `/papers/reabbreviate-all`, `/papers/batch-ingest`, `/references/explain`, `/topic/{id}/query`, `/papers/classify` (naming step)

**Symptom:**
`IndexError: list index out of range` crashing the endpoint with HTTP 500. Happens when the LLM API returns an empty `choices` list — e.g. content filtering triggered, API quota exhausted, model overloaded, or transient error on the vLLM server.

**Root cause:**
Every LLM completion call in the codebase directly indexes `response.choices[0]` with no length guard. Ten confirmed locations:

| File | Line | Endpoint |
|------|------|----------|
| `app.py` | 698 | `POST /summary/merge` |
| `app.py` | 742 | `POST /summary/dedup` |
| `app.py` | 778 | `POST /classify` |
| `app.py` | 803 | `POST /abbreviate` |
| `app.py` | 845 | `POST /papers/reabbreviate` |
| `app.py` | 880 | `POST /papers/reabbreviate-all` (per-paper) |
| `app.py` | 1407 | `POST /papers/batch-ingest` (Slack, per-paper abbreviation) |
| `app.py` | 1578 | `POST /papers/batch-ingest` (directory, per-paper abbreviation) |
| `app.py` | 2130 | `POST /references/explain` |
| `app.py` | 2595 | `POST /topic/{id}/query` (aggregation step) |
| `naming.py` | 446 | `POST /papers/classify` (node naming, per-node) |
| `naming.py` | 732 | `POST /tree/node` (rename via LLM) |

```python
# Example — app.py:803
response = await client.chat.completions.create(...)
abbrev = response.choices[0].message.content.strip()  # crashes if choices=[]
```

**Files:** `src/backend/app.py` (lines above), `src/backend/naming.py:446,732`

---

## BUG-003 · `db.create_paper()` return value used without None-check in batch-ingest

**Severity:** Critical
**Affected endpoints:** `POST /papers/batch-ingest` (both Slack and directory paths), `POST /summarize/structured`

**Symptom:**
`TypeError` or `psycopg2` error mid-ingest, causing that paper's task to fail. Error message is misleading because it references the downstream call (e.g. `update_paper_summary(None, ...)`) rather than the actual failure in `create_paper`.

**Root cause:**
`db.create_paper()` can raise an exception or return `None` (e.g. unique constraint on a race condition, DB connection dropped). The result is immediately used as an integer ID with no check:

```python
# app.py:1412–1432 (Slack path)
db_paper_id = db.create_paper(arxiv_id=arxiv_id, title=title, ...)
# ... several lines later, no check ...
db.update_paper_summary(db_paper_id, summary)   # TypeError if db_paper_id is None

# app.py:1581–1602 (directory path)
db_paper_id = db.create_paper(arxiv_id=paper_id, ...)
db.update_paper_summary(db_paper_id, summary)   # same issue

# app.py:377–378 (/summarize/structured)
_pid = db.create_paper(arxiv_id=payload.arxiv_id, ...)
await rag.index_paper_async(_pid, payload.pdf_path, ...)  # _pid=None → TypeError in index_paper_async
```

**Files:** `src/backend/app.py:1412–1432`, `1581–1602`, `377–378`

---

## BUG-004 · `/references/explain` cache lookup always misses — queries `paper_id=0`

**Severity:** High
**Affected endpoints:** `POST /references/explain`

**Symptom:**
Every call hits the LLM even when the explanation for that reference was already computed and stored. Cache is effectively dead.

**Root cause:**
`app.py:2102` calls `db.get_references(0)` to find a cached explanation. `db.get_references(paper_id)` fetches references **by paper ID**, not by reference ID. Passing `paper_id=0` always returns an empty list because no paper has ID 0. The `next(...)` search over an empty list always returns `None`, so the cache check always misses.

```python
# app.py:2101–2103
refs = db.get_references(0)  # BUG: always returns [] — paper_id=0 does not exist
ref = next((r for r in refs if r.get("id") == payload.reference_id), None) if refs else None
if ref and ref.get("explanation"):
    return {"explanation": ref["explanation"], "from_cache": True}  # never reached
```

The intent was to fetch a single reference row by `reference_id`, but no such function exists in `db.py` — `get_references()` only accepts a `paper_id`.

**Files:** `src/backend/app.py:2101–2105`, `src/backend/db.py` (missing `get_reference_by_id`)

---

## BUG-005 · `StructuredSummarizeRequest.pdf_path` is required but the endpoint handles `None` correctly

**Severity:** High
**Affected endpoints:** `POST /summarize/structured`

**Symptom:**
HTTP 422 — `"pdf_path: field required"` when calling `/summarize/structured` with only `arxiv_id` for a paper that is already fully indexed (chunks in DB). The equivalent endpoint `POST /qa/structured` works fine without `pdf_path`.

**Root cause:**
The Pydantic model declares `pdf_path` as a required `str`, but the endpoint implementation already handles `None` by checking `if payload.pdf_path and payload.arxiv_id`. The model is more restrictive than the logic warrants:

```python
# app.py:82–84  — model
class StructuredSummarizeRequest(BaseModel):
    pdf_path: str = Field(description="Local PDF file path")  # required — wrong
    arxiv_id: Optional[str] = Field(default=None, ...)

# app.py:372–378 — endpoint (handles None fine)
if payload.pdf_path and payload.arxiv_id:
    _paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if _paper and not db.has_paper_chunks(_paper["id"]):
        await rag.index_paper_async(...)
```

Compare with `StructuredQaRequest` (line 103–106) where `pdf_path` is correctly `Optional[str]`.

**Files:** `src/backend/app.py:83`

---

## BUG-006 · `np.mean()` called on potentially empty embeddings list during indexing

**Severity:** High
**Affected endpoints:** `POST /papers/batch-ingest`, `POST /papers/save`, `POST /embed/fulltext` (indirectly via `index_paper_async` and `rag_answer_async`)

**Symptom:**
`ValueError: zero-size array to reduction operation maximum which has no identity` — crashes the indexing step for that paper, leaving it without an embedding or chunks.

**Root cause:**
`rag.py:191` and `rag.py:280` call `np.mean(embeddings, axis=0)` after receiving the batch embedding response. If the embedding API returns an empty list (server error, empty input filtered out, or response truncated), `embeddings` is `[]` and NumPy raises:

```python
# rag.py:179–192 (index_paper_async)
embeddings = await embed_texts(chunk_texts, embed_client, embed_model)
# No check that embeddings is non-empty
for chunk, emb in zip(chunks, embeddings):
    chunk["embedding"] = emb
db.store_paper_chunks(paper_id, chunks)
doc_embedding = np.mean(embeddings, axis=0).tolist()  # ValueError if embeddings=[]

# rag.py:276–281 (rag_answer_async ephemeral path)
doc_emb = np.mean(chunk_embeddings, axis=0).tolist()  # same issue
```

The `if not chunks:` guard at `rag.py:176` only protects against zero text chunks before embedding; it does not guard against the embedding API returning fewer results than expected.

**Files:** `src/backend/rag.py:191`, `src/backend/rag.py:280`

---

## BUG-007 · `reabbreviate-all` runs all papers in a single unbounded `asyncio.gather`

**Severity:** Medium
**Affected endpoints:** `POST /papers/reabbreviate-all`

**Symptom:**
With many papers (100+), `reabbreviate-all` fires all LLM requests simultaneously. This causes rate-limit errors or OOM on the vLLM server, resulting in partial failures or a cascade of BUG-002 (empty choices).

**Root cause:**
`app.py:890` passes every paper to `asyncio.gather` with no concurrency limit:

```python
# app.py:889–890
results = await asyncio.gather(*[abbreviate_one(p) for p in papers])
# No semaphore — fires N concurrent LLM requests where N = len(papers)
```

Compare with `POST /topic/{id}/query` (line 2549) which correctly uses `asyncio.Semaphore(3)`.

**Files:** `src/backend/app.py:889–890`

---

## BUG-008 · `POST /papers/classify` has no None-check on `clustering.build_tree_from_clusters()` return value

**Severity:** Medium
**Affected endpoints:** `POST /papers/classify`, `POST /categories/rebalance`

**Symptom:**
`TypeError: 'NoneType' object is not subscriptable` at `app.py:1705` if `clustering.build_tree_from_clusters()` returns `None` due to an unhandled exception inside the clustering step.

**Root cause:**
`app.py:1703–1705` calls `cluster_result["total_papers"]` immediately after the thread call without checking for `None`:

```python
# app.py:1703–1705
cluster_result = await asyncio.to_thread(clustering.build_tree_from_clusters)
# No None check:
if cluster_result["total_papers"] < 2:   # TypeError if cluster_result is None
```

`_rebuild_tree_async` at line 1124 does have a `.get()` guard (`cluster_result.get("total_papers", 0)`), but the main `classify_papers` endpoint at line 1703 does not.

**Files:** `src/backend/app.py:1703–1705`

---

## BUG-009 · vLLM prefix-cache poisoning makes `enable_thinking=False` unreliable for Qwen3-4B

**Severity:** High
**Affected components:** `POST /papers/classify` (tree naming via `naming.py`), any endpoint calling the SLM (`slm_base_url`)

**Symptom:**
Category node names produced by `/papers/classify` are garbled: truncated mid-sentence (`'Continual'`, `'Merges'`, `'Systemati'`, `'Multi-Sta'`) or look like sentence openings (`'The Paper Proposes'`, `'What It'`). The model is generating thinking tokens (`<think>...`) instead of the final answer — the thinking block hits `max_tokens` before `</think>` closes, so the raw thinking text becomes the "name".

**Root cause — mechanism:**

vLLM stores KV (key-value) attention states keyed on the tokenized prefix of each request (`--enable-prefix-caching`). The `enable_thinking` flag in `chat_template_kwargs` changes how the prompt is **tokenised** (Qwen3 inserts a `<think>` or `</think>` token at a specific position in the chat template). When a request arrives:

1. A request without `enable_thinking=False` — i.e. thinking **ON** — is processed. The prompt tokens (including the thinking-on marker) are stored in the KV cache.
2. A later request for the **same user prompt** but with `enable_thinking=False` shares the identical user-message prefix up to the thinking-toggle token.
3. vLLM re-uses the cached KV states for that shared prefix. The cache was written from a thinking-ON render — the thinking-off marker never appears. The model is already "mid-thinking" and proceeds to generate `<think>...` indefinitely.
4. In the poisoned state, even `max_tokens=500` does not help: the model consumes all tokens with thinking content; `</think>` never appears; `message.content` is a raw thinking fragment.

**Discovery process:**

The issue was intermittent and confusing because the same prompt with the same `enable_thinking=False` flag would sometimes work and sometimes fail:

- `test_thinking_both.py` (simple 62-token prompt): `enable_thinking=False` → 6 tokens, clean output. Worked reliably.
- `test_production_names.py` (real paper summaries, ~300-token prompt): thinking leaked in all 15 calls even at `temperature=0.0`.
- `test_nothink_approaches.py` (same prompt structure, temp=0.1): all 5 approaches including `enable_thinking=False` worked perfectly.
- `test_temp_confirm.py` (re-ran with same prompt, temp 0.0–0.7, N=3 each): all 15 calls failed again.

The contradiction — identical configuration working at one moment and failing at another — was the key clue. The only varying axis was **server-side state between test runs**. Since the user-prompt was identical, the only server-side state affecting generation was the KV cache. Reading `mech_server_llm_q3_4b.pbs` line 61 confirmed `--enable-prefix-caching` was active.

Final confirmation via `test_max500.py`:
- Clean cache: `max_tokens=50 + enable_thinking=False` → `'Efficient Models'` (4 tokens, `finish_reason=stop`)
- Poisoned cache: `max_tokens=500, NO extra_body (thinking ON)` → 500 tokens of `<think>...`, `</think>` **never** appears, `finish_reason=length`

This proved that when the cache carries thinking-ON state, no amount of `max_tokens` budget produces a usable answer.

**Why this is Qwen3-specific (LLM / DeepSeek is immune):**

DeepSeek-V3.2 also runs with `--enable-prefix-caching` enabled. However, DeepSeek does not support a thinking mode at all — it never generates `<think>` tokens regardless of parameters. There is no "thinking ON" state to cache. The `enable_thinking=False` flag is therefore a no-op on DeepSeek and prefix-cache poisoning cannot occur for it.

**Workaround / Fix:**

- **Immediate**: Restart the SLM PBS job (`mech_server_llm_q3_4b`) to clear all KV caches. Re-run `/papers/classify`.
- **Permanent**: Remove `--enable-prefix-caching` from `mech_server_llm_q3_4b.pbs` (line 61). Prefix caching conflicts with per-request thinking-mode toggling for Qwen3-class models.
- **Alternative for naming only**: Use the LLM endpoint (`llm_base_url`) for tree naming. DeepSeek is immune to this issue and produces valid category names, at the cost of slower throughput (LLM is a larger model on 8 GPUs vs SLM on 1 GPU).

**Can we purge the prefix cache without a server restart?**

No. vLLM exposes no REST API endpoint for cache eviction or purging. The prefix cache is an internal GPU-memory data structure. The only levers are: (1) restart the server, (2) remove `--enable-prefix-caching` at next restart (prevents future poisoning), (3) append a unique random suffix to the system prompt on every request (prevents cache hits entirely — a hack, not recommended). There is no `DELETE /v1/cache` or equivalent in the OpenAI-compatible API vLLM exposes.

**Files:** `local_model_server/mech_server_llm_q3_4b.pbs:61` (`--enable-prefix-caching`), `src/backend/naming.py:547` (uses `slm_base_url`)

**Update (2026-03-01):** Empirical investigation showed this issue is REAL for the SLM server, but was NOT the primary cause of garbled category names in production. The actual cause was BUG-010 below. The prefix-cache poisoning remains a valid concern if thinking-ON requests are sent to the SLM server.

---

## BUG-010 · `naming.py::_get_prompt()` relative path search fails from backend CWD — all category names fall back to paper summary fragments

**Severity:** Critical
**Affected endpoints:** `POST /papers/categorize` (tree naming step)
**Status:** FIXED 2026-03-01 — added `Path(__file__).parent / "prompts" / "prompts.json"` as first search path

**Symptom:**
All 1445 category node names are garbled: single verbs (`'Trains'`, `'Analyzes'`, `'Collects'`), fragments (`'Systemati'`, `'Identifie'`, `'Deduplica'`), or forbidden phrases (`'What It Does'` ×135, `'A'` ×98, `'The'` ×67). After the naming run, the deduplication step appends `(node_id[-6:])` suffixes to duplicates, yielding names like `'Systemati (a5055c)'`.

**Root cause:**
`_get_prompt("node_naming", ...)` in `naming.py` searches for the prompt template using three relative paths:
```python
prompt_paths = [
    Path("prompts/prompts.json"),
    Path("../prompts/prompts.json"),
    Path("../../prompts/prompts.json"),
]
```
The backend CWD is `/scratch_aisg/.../paper-curator/`. The actual file is at `src/backend/prompts/prompts.json`. None of the three relative paths resolve correctly from the backend CWD — `_get_prompt` raises `ValueError("Prompt 'node_naming' not found")` on every call.

`_call_llm_for_name` catches ALL exceptions (including this ValueError) in its retry loop:
```python
except Exception as e:
    if attempt < max_retries - 1:
        await asyncio.sleep(1)
    else:
        # Fallback: use first child content words
        words = children_content[0].split()[:3]
        fallback = " ".join(words).title()[:30] or node_id
        fallback = self._sanitize_name(fallback)
        return fallback
```
So on all 3 attempts, the prompt fails to load → ValueError → retry → final fallback. The fallback takes the first 3 words of the first child's paper summary. These summaries are formatted as `"**Concrete Method:** Systematically measures..."` — so:
- `['**Concrete', 'Method:**', 'Systematically']` → `"**Concrete Method:** Systemati"` → `_sanitize_name` strips `**`, then colon-splits `"Concrete Method:"` + `"Systemati"` → **`'Systemati'`**
- `"**What it does:** Medusa..."` → fallback `"What It Does"` (not caught by invalid_starts check on the fallback path)

This bug was mis-attributed to SLM prefix-cache poisoning (BUG-009) until direct testing showed DeepSeek also producing garbled names, and tracing showed 0 LLM calls were actually made during naming (prompt file never loaded).

**Discovery process:**
1. BUG-009 was diagnosed → SLM PBS job was being investigated
2. Switched naming endpoint to LLM (DeepSeek) for a comparison classify — same garbled names appeared
3. Direct call to DeepSeek with the same node's production prompt → perfect names (8/8, 24/24, 10/10 valid in all tests)
4. Traced the contradiction: classify produces `'Systemati'` for a node; direct test of that same node gives `'Hardware and System Optimization'`
5. Discovered `_get_prompt` path resolution: none of the 3 relative paths exist from the backend CWD
6. Confirmed: `ValueError("Prompt 'node_naming' not found")` is raised silently, fallback runs every time

**Fix:**
```python
# naming.py::_get_prompt — add as FIRST search path:
Path(__file__).parent / "prompts" / "prompts.json",  # resolves to src/backend/prompts/prompts.json regardless of CWD
```

**Verification:** Full classify after fix → `'Multilingual Systems'`, `'Quantization Methods'`, `'Reasoning Techniques'`, etc. (1445 nodes, 14 min, all proper noun phrases)

**Note on the fallback:** The fallback mechanism (`first 3 words of first child summary`) is useful for true LLM failures, but is **silent** — it logs a `✓ Named ...` message with the fallback name, indistinguishable from a real LLM response. The fallback also bypasses the `invalid_starts` validation, allowing `'What It Does'` to be stored. Consider adding an explicit WARNING log when the fallback is used.

**Files:** `src/backend/naming.py:50–71` (`_get_prompt`), `src/backend/prompts/prompts.json` (the missing file)

---

## BUG-013 · Single-paper ingest silently aborts at the classify step — paper never saved to DB

**Severity:** Critical
**Affected flow:** Single-paper ingest via UI (`handleUnifiedIngest` in `page.tsx`)
**Status:** FIXED 2026-03-01 — removed stale `/api/classify` call; replaced `addPaperToTree` with delayed tree refresh

**Symptom:**
Paper appears not to exist after ingestion through the "Ingest arXiv paper" UI: it cannot be found via semantic search and clicking "Categorize" reports "No dirty nodes — tree is already up to date". The backend DB contains no record of the paper.

**Root cause:**
In commit `f8825e6` (partial-recategorize), the `/classify` backend endpoint was deleted (along with its `next.config.js` rewrite). Under the new architecture, paper placement into the category tree is done by `place_paper_in_tree()` via embedding cosine similarity as a background task after `/papers/save` — no synchronous classify call is needed at ingest time.

However, `handleUnifiedIngest` in `page.tsx` still called `fetch("/api/classify", ...)` as a **fatal** step 3 in the parallel block:

```typescript
// page.tsx — stale call, endpoint deleted in f8825e6
fetch("/api/classify", {
  method: "POST",
  body: JSON.stringify({ title, abstract, existing_categories: existingCategories }),
})
// …
if (!classifyRes.ok) {     // always true — Next.js returns 404
  updateStep(3, { status: "error", message: `HTTP ${classifyRes.status}` });
  setIsIngesting(false);
  return;   // ← exits before /summarize and /papers/save are ever called
}
```

Since no route or rewrite existed, Next.js returned 404. The fatal error handler exited the function — `/summarize` and `/papers/save` were never called.

**Evidence confirmed:**
- `db.get_paper_by_arxiv_id('2602.22193')` → `None` (paper absent from DB)
- Backend logs: resolve ✓, download ✓, extract ✓, abbreviate ✓ — then nothing. No `/summarize` or `/papers/save` hit.
- `/api/classify` rewrite deleted in f8825e6; no Next.js custom route existed

**Fix (`src/frontend/src/app/page.tsx`, commit `94fbf34`):**
- Removed "Classify (LLM)" step from ingest steps array
- Dropped `/api/classify` fetch and its fatal error handler from the parallel block
- Simplified parallel block: extract + abbreviate only (was extract + classify + abbreviate)
- Renumbered steps: abbreviate→3, summarize→4, save→5
- Removed `category` from `/papers/save` request body (`SavePaperRequest` has no such field)
- Replaced `addPaperToTree` (optimistic local-category guess) with a 1.5 s delayed tree refresh after save — the backend now places the paper in the correct node via embedding similarity descent, so the frontend re-fetches the authoritative tree
- Removed now-unused `existingCategories` useMemo and `addPaperToTree` useCallback
- Fixed duplicate-skip loop count: `i < 7` → `i < 6`
- Rebuilt frontend Singularity SIF (`make singularity-build-frontend`) and restarted service

**Verification:**
Re-ingested `2602.22193` after fix: backend logs show `POST /summarize` ✓, `POST /papers/save` ✓, `[placement] {'placed': True, 'node_id': 'node_ce056cd026bf', 'paper_id': 2316}` ✓. Partial categorize processed 3 dirty nodes. Paper searchable at rank 47/2316 for query "reinforcement learning verifiable rewards RLVR". All 120 tests pass (2 skipped).

**Files:** `src/frontend/src/app/page.tsx` (steps 1082–1285, `existingCategories`, `addPaperToTree`)

