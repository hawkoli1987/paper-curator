from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import re
import uuid
from functools import lru_cache
from typing import Any, Optional

import arxiv
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

import db
import rag
from arxiv_helpers import (
    download_arxiv_pdf_async,
    extract_arxiv_ids_from_text,
    require_identifier,
)
from config import (
    _get_classification_config,
    _get_endpoint_config,
    _get_external_apis_config,
    _get_ingestion_config,
    _get_rag_config,
    _get_topic_query_config,
    _get_ui_config,
    _load_prompt,
    get_prompt,
)
from external_clients import get_http_client
from external_clients import shutdown_http_clients
from llm_clients import (
    get_async_openai_client,
    get_openai_client,
    resolve_model,
)
from slack_helpers import (
    fetch_slack_messages,
    resolve_slack_channel_id,
)

app = FastAPI(title="paper-curator-backend")

# Module-level lock serialising all tree-modifying operations:
# placement, partial recategorize, full rebuild, bootstrap.
# Reads (GET /tree) are NOT locked.
_tree_lock = asyncio.Lock()


@app.on_event("shutdown")
async def shutdown_external_clients():
    """Close all shared HTTP clients on shutdown."""
    await shutdown_http_clients()


# =============================================================================
# Request/Response Models
# =============================================================================

class ArxivResolveRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: Optional[str] = Field(default=None, description="arXiv URL")


class ArxivDownloadRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: Optional[str] = Field(default=None, description="arXiv URL")
    output_dir: Optional[str] = Field(default=None, description="Directory to store downloads")


class PdfExtractRequest(BaseModel):
    pdf_path: str = Field(description="Local PDF file path")


class SummarizeRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Full paper text or extracted sections")
    pdf_path: Optional[str] = Field(default=None, description="Local PDF file path")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID for persisting index")


class StructuredSummarizeRequest(BaseModel):
    pdf_path: Optional[str] = Field(default=None, description="Local PDF file path")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID for persisting index")


class EmbedAbstractRequest(BaseModel):
    text: str = Field(description="Abstract text to embed for similarity search")


class EmbedFulltextRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID of paper")
    pdf_path: str = Field(description="Path to PDF file")


class QaRequest(BaseModel):
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID for cached index lookup")
    context: Optional[str] = Field(default=None, description="Context text to answer from")
    question: str = Field(description="Question to answer")
    pdf_path: Optional[str] = Field(default=None, description="Local PDF file path")


class StructuredQaRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID for cached index lookup (paper must be indexed)")
    pdf_path: Optional[str] = Field(default=None, description="PDF path fallback if index not found")


class SavePaperRequest(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: Optional[str] = None
    summary: Optional[str] = None
    abbreviation: Optional[str] = None  # Short display name (e.g., "mHC", "GPT-4")
    pdf_path: Optional[str] = None
    latex_path: Optional[str] = None
    pdf_url: Optional[str] = None
    published_at: Optional[str] = None


class BatchIngestRequest(BaseModel):
    directory: Optional[str] = Field(default=None, description="Local directory containing PDF files")
    slack_channel: Optional[str] = Field(default=None, description="Slack channel identifier (#channel-name, channel ID, or URL)")
    slack_token: Optional[str] = Field(default=None, description="Slack User OAuth Token (xoxp-...) - not persisted")
    skip_existing: bool = Field(default=True, description="Skip PDFs already in database")
    limit: Optional[int] = Field(default=None, description="Maximum number of papers to ingest (applied after dedup/skip-existing filtering)")


class RepoSearchRequest(BaseModel):
    arxiv_id: str
    title: str


class ReferencesRequest(BaseModel):
    arxiv_id: str


class ExplainReferenceRequest(BaseModel):
    reference_id: int
    source_paper_title: str
    cited_title: str
    citation_context: Optional[str] = None


class SimilarPapersRequest(BaseModel):
    arxiv_id: str


# Topic Query Request Models
class TopicSearchRequest(BaseModel):
    topic: str = Field(description="Topic to search for similar papers")
    offset: int = Field(default=0, description="Pagination offset")
    limit: Optional[int] = Field(default=None, description="Override max_papers_per_batch from config")
    exclude_paper_ids: list[int] = Field(default=[], description="Paper IDs to exclude from results")


class TopicCreateRequest(BaseModel):
    name: str = Field(description="Unique name for the topic (user prefix + topic)")
    topic_query: str = Field(description="Original search topic")


class TopicAddPapersRequest(BaseModel):
    paper_ids: list[int] = Field(description="Paper IDs to add to the topic pool")
    similarity_scores: Optional[list[float]] = Field(default=None, description="Similarity scores for each paper")


class TopicQueryRequest(BaseModel):
    question: str = Field(description="Question to ask across all papers in the topic pool")


# =============================================================================
# Config + prompt helpers are provided by config.py
# =============================================================================
# =============================================================================
# Helper Functions
# =============================================================================

def _require_identifier(arxiv_id: Optional[str], url: Optional[str]) -> str:
    """Extract arXiv ID from provided arxiv_id or URL."""
    return require_identifier(arxiv_id, url)


def _extract_arxiv_ids_from_text(text: str) -> list[str]:
    """Extract all arXiv IDs/URLs from text."""
    return extract_arxiv_ids_from_text(text)


async def _resolve_slack_channel_id(client, channel_input: str) -> str:
    """Resolve Slack channel ID from various input formats."""
    return await resolve_slack_channel_id(client, channel_input)


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Create OpenAI client with ngrok header support if needed."""
    return get_openai_client(base_url, api_key)


def _get_async_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """Create AsyncOpenAI client with ngrok header support if needed."""
    return get_async_openai_client(base_url, api_key)


@lru_cache(maxsize=3)
def _resolve_model(base_url: str, api_key: str) -> str:
    """Resolve model name from endpoint, with ngrok free tier workaround."""
    return resolve_model(base_url, api_key)


async def _download_arxiv_pdf(arxiv_id: str, pdf_url: str) -> str:
    """Download arXiv PDF and return local path with retry logic."""
    return await download_arxiv_pdf_async(arxiv_id)


# =============================================================================
# Core Endpoints
# =============================================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui-config")
def get_ui_config() -> dict[str, Any]:
    """Return UI configuration for the frontend."""
    return _get_ui_config()


@app.post("/arxiv/resolve")
async def arxiv_resolve(payload: ArxivResolveRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[identifier])
        results = await asyncio.to_thread(lambda: list(client.results(search)))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"arXiv API error: {e}")
    if not results:
        raise HTTPException(status_code=404, detail=f"No arXiv result found for: {identifier}")
    result = results[0]
    return {
        "arxiv_id": result.get_short_id(),
        "title": result.title,
        "authors": [author.name for author in result.authors],
        "published": result.published.isoformat() if result.published else None,
        "summary": result.summary,
        "pdf_url": result.pdf_url,
        "entry_id": result.entry_id,
        "comment": result.comment,
    }


@app.post("/arxiv/download")
async def arxiv_download(payload: ArxivDownloadRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    output_dir = payload.output_dir or os.getenv("ARXIV_DOWNLOAD_DIR", "storage/downloads")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[identifier])
        results = await asyncio.to_thread(lambda: list(client.results(search)))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"arXiv API error: {e}")
    if not results:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
    result = results[0]

    pdf_path = await asyncio.to_thread(result.download_pdf, dirpath=output_dir)
    latex_path = None
    if result.source_url():
        latex_path = await asyncio.to_thread(result.download_source, dirpath=output_dir)

    return {
        "arxiv_id": result.get_short_id(),
        "pdf_path": pdf_path,
        "latex_path": latex_path,
    }


@app.post("/pdf/extract")
async def pdf_extract(payload: PdfExtractRequest) -> dict[str, Any]:
    """Extract text from PDF using pymupdf."""
    pdf_path = pathlib.Path(payload.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    text = rag.extract_pdf_text(pdf_path)
    return {"text": text, "parser": "pymupdf"}


@app.post("/summarize")
async def summarize(payload: SummarizeRequest) -> dict[str, Any]:
    """Summarize a paper using custom RAG."""
    prompt_id, prompt_hash, prompt_body = _load_prompt()
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    # Check LLM endpoint
    try:
        model = _resolve_model(base_url, api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM endpoint not available at {base_url}: {e}")
    
    # Check embedding endpoint
    try:
        embed_model = _resolve_model(embed_base_url, api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding endpoint not available at {embed_base_url}. Please start your embedding service on port 8004.")
    
    llm_client = _get_async_openai_client(base_url, api_key)
    embed_client = _get_async_openai_client(embed_base_url, api_key)
    
    summary = await rag.rag_answer_async(
        question=prompt_body,
        text=payload.text,
        pdf_path=payload.pdf_path,
        arxiv_id=payload.arxiv_id,
        llm_client=llm_client,
        llm_model=model,
        embed_client=embed_client,
        embed_model=embed_model,
    )
    return {
        "summary": summary.strip(),
        "prompt_id": prompt_id,
        "prompt_hash": prompt_hash,
        "model": f"openai/{model}",
        "indexed": payload.arxiv_id is not None,
    }


@app.post("/summarize/structured")
async def summarize_structured(payload: StructuredSummarizeRequest) -> dict[str, Any]:
    """Generate a structured summary with components and detailed analysis.
    
    Flow:
    1. Extract key components/concepts from the paper
    2. For each component, query 4 aspects in parallel
    3. Return structured sections
    """
    # Input validation first — before any network calls
    if not payload.pdf_path and not payload.arxiv_id:
        raise HTTPException(status_code=422, detail="Provide pdf_path or arxiv_id (for already-indexed papers)")

    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]

    model = _resolve_model(base_url, api_key)
    embed_model = _resolve_model(embed_base_url, api_key)

    llm_client = _get_async_openai_client(base_url, api_key)
    embed_client = _get_async_openai_client(embed_base_url, api_key)

    # Pre-index: ensure chunks are stored so all queries use fast DB-backed retrieval
    if payload.pdf_path and payload.arxiv_id:
        _paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
        if _paper and not db.has_paper_chunks(_paper["id"]):
            await rag.index_paper_async(_paper["id"], payload.pdf_path, embed_client, embed_model)
        elif not _paper:
            _pid = db.create_paper(arxiv_id=payload.arxiv_id, title=f"Paper {payload.arxiv_id}", authors=[], pdf_path=payload.pdf_path)
            if _pid is None:
                raise HTTPException(status_code=500, detail=f"Failed to create DB record for {payload.arxiv_id}")
            await rag.index_paper_async(_pid, payload.pdf_path, embed_client, embed_model)
    
    # Step 1: Extract components using RAG
    extract_prompt = get_prompt("extract_components")
    components_raw = await rag.rag_answer_async(
        question=extract_prompt,
        pdf_path=payload.pdf_path,
        arxiv_id=payload.arxiv_id,
        llm_client=llm_client,
        llm_model=model,
        embed_client=embed_client,
        embed_model=embed_model,
    )
    
    # Parse components JSON - extract array from response
    match = re.search(r'\[.*\]', components_raw, re.DOTALL)
    components = json.loads(match.group()) if match else [components_raw.strip().strip('"\'')]
    components = components or ["Main Contribution"]
    
    # Step 2: For each component, query 4 aspects sequentially
    async def analyze_component(component: str) -> dict[str, Any]:
        """Analyze a single component with 4 sequential queries."""
        result = {"component": component}
        
        aspects = [
            ("steps", get_prompt("component_steps", component=component)),
            ("benefits", get_prompt("component_benefits", component=component)),
            ("rationale", get_prompt("component_rationale", component=component)),
            ("results", get_prompt("component_results", component=component)),
        ]
        
        for aspect, prompt in aspects:
            answer = await rag.rag_answer_async(
                question=prompt,
                pdf_path=payload.pdf_path,
                arxiv_id=payload.arxiv_id,
                llm_client=llm_client,
                llm_model=model,
                embed_client=embed_client,
                embed_model=embed_model,
            )
            result[aspect] = answer.strip()
        
        return result
    
    sections = []
    for component in components:
        section = await analyze_component(component)
        sections.append(section)
    
    return {
        "components": list(components),
        "sections": list(sections),
        "model": f"openai/{model}",
        "indexed": payload.arxiv_id is not None,
    }


@app.post("/embed/abstract")
async def embed_abstract(payload: EmbedAbstractRequest) -> dict[str, Any]:
    """Embed abstract text for pgvector similarity search."""
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    response = await client.embeddings.create(model=model, input=payload.text)
    vector = response.data[0].embedding
    return {"embedding": vector, "model": model}


@app.post("/embed_query")
async def embed_query(payload: EmbedAbstractRequest) -> dict[str, Any]:
    """Embed query text for similarity search (replaces /embed/abstract)."""
    return await embed_abstract(payload)


@app.post("/embed/fulltext")
async def embed_fulltext(payload: EmbedFulltextRequest) -> dict[str, Any]:
    """Index full PDF: chunk, embed, store in paper_chunks table."""
    endpoint_config = _get_endpoint_config()
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    try:
        embed_model = _resolve_model(embed_base_url, api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding endpoint not available: {e}")
    
    embed_client = _get_async_openai_client(embed_base_url, api_key)
    
    # Ensure paper exists in DB (create minimal entry if not)
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        paper_id = db.create_paper(
            arxiv_id=payload.arxiv_id,
            title=f"Paper {payload.arxiv_id}",
            authors=[],
            pdf_path=payload.pdf_path,
        )
    else:
        paper_id = paper["id"]
    
    result = await rag.index_paper_async(paper_id, payload.pdf_path, embed_client, embed_model)
    return {"indexed": result["indexed"], "cached": result["cached"], "arxiv_id": payload.arxiv_id}




@app.post("/qa")
async def qa(payload: QaRequest) -> dict[str, Any]:
    """Answer a question about a paper using custom RAG.
    
    Persists the query and answer to the database for later retrieval.
    """
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    embed_model = _resolve_model(embed_base_url, api_key)
    
    llm_client = _get_async_openai_client(base_url, api_key)
    embed_client = _get_async_openai_client(embed_base_url, api_key)
    
    # Check if we have indexed chunks
    has_cache = False
    if payload.arxiv_id:
        paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
        if paper:
            has_cache = db.has_paper_chunks(paper["id"])
    
    answer = await rag.rag_answer_async(
        question=payload.question,
        text=payload.context,
        pdf_path=payload.pdf_path,
        arxiv_id=payload.arxiv_id,
        llm_client=llm_client,
        llm_model=model,
        embed_client=embed_client,
        embed_model=embed_model,
    )
    
    # Persist query to database if arxiv_id provided
    if payload.arxiv_id:
        paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
        if paper:
            db.add_query(paper["id"], payload.question, answer, f"openai/{model}")
    
    return {"answer": answer, "used_cache": has_cache}


@app.post("/qa/structured")
async def qa_structured(payload: StructuredQaRequest) -> dict[str, Any]:
    """Run structured analysis on a paper using custom RAG.
    
    Flow:
    1. Extract key components from paper (sequential - need results first)
    2. For ALL components, query ALL 4 aspects in parallel
    3. Return structured sections
    """
    # Check for indexed chunks or PDF
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id) if payload.arxiv_id else None
    has_chunks = paper and db.has_paper_chunks(paper["id"]) if paper else False
    if not has_chunks and not payload.pdf_path:
        raise HTTPException(
            status_code=404, 
            detail=f"No indexed chunks for {payload.arxiv_id}. Ingest the paper first."
        )
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    embed_model = _resolve_model(embed_base_url, api_key)
    
    llm_client = _get_async_openai_client(base_url, api_key)
    embed_client = _get_async_openai_client(embed_base_url, api_key)
    
    # Pre-index: ensure chunks are stored so all queries use fast DB-backed retrieval
    if payload.pdf_path and payload.arxiv_id and not has_chunks:
        paper_obj = paper
        if not paper_obj:
            paper_id_new = db.create_paper(
                arxiv_id=payload.arxiv_id,
                title=f"Paper {payload.arxiv_id}",
                authors=[],
                pdf_path=payload.pdf_path,
            )
        else:
            paper_id_new = paper_obj["id"]
        await rag.index_paper_async(paper_id_new, payload.pdf_path, embed_client, embed_model)
    
    async def run_query_async(question: str) -> str:
        """Run a single RAG query asynchronously."""
        return await rag.rag_answer_async(
            question=question,
            pdf_path=payload.pdf_path,
            arxiv_id=payload.arxiv_id,
            llm_client=llm_client,
            llm_model=model,
            embed_client=embed_client,
            embed_model=embed_model,
        )
    
    # Step 1: Extract components (sequential - need results before step 2)
    extract_prompt = get_prompt("extract_components")
    components_raw = await run_query_async(extract_prompt)
    
    # Parse components JSON
    try:
        match = re.search(r'\[.*\]', components_raw, re.DOTALL)
        if match:
            components = json.loads(match.group())
        else:
            components = [components_raw.strip().strip('"\'')]
    except json.JSONDecodeError:
        components = [components_raw.strip().strip('"\'')]
    
    if not components:
        components = ["Main Contribution"]
    
    # Limit components to prevent excessive runtime
    components = components[:5]
    
    # Step 2: Build all queries and run them in parallel
    query_tasks = []
    query_metadata = []
    
    for idx, component in enumerate(components):
        for aspect, prompt_name in [
            ("steps", "component_steps"),
            ("benefits", "component_benefits"),
            ("rationale", "component_rationale"),
            ("results", "component_results"),
        ]:
            question = get_prompt(prompt_name, component=component)
            query_tasks.append(run_query_async(question))
            query_metadata.append((idx, aspect))
    
    # Run ALL queries in parallel
    results = await asyncio.gather(*query_tasks, return_exceptions=True)
    
    # Reassemble results into sections
    sections = [{"component": comp} for comp in components]
    for (idx, aspect), result in zip(query_metadata, results):
        if isinstance(result, Exception):
            sections[idx][aspect] = f"Error: {result}"
        else:
            sections[idx][aspect] = result
    
    # Persist structured analysis to database
    structured_result = {
        "components": components,
        "sections": sections,
        "model": f"openai/{model}",
    }
    
    if payload.arxiv_id:
        paper_obj = db.get_paper_by_arxiv_id(payload.arxiv_id)
        if paper_obj:
            db.update_paper_structured_summary(paper_obj["id"], structured_result)
    
    return structured_result


class MergeSummaryRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID of the paper")
    selected_qa: list[dict[str, str]] = Field(description="List of {question, answer} pairs to merge")


class DedupSummaryRequest(BaseModel):
    arxiv_id: str = Field(description="arXiv ID of the paper")


@app.post("/summary/merge")
async def merge_qa_to_summary(payload: MergeSummaryRequest) -> dict[str, Any]:
    """Merge selected Q&A content into the paper's summary.
    
    Uses LLM to intelligently integrate Q&A content into appropriate
    sections of the existing summary.
    """
    # Get paper from DB
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    current_summary = paper.get("summary", "")
    if not current_summary:
        raise HTTPException(status_code=400, detail="Paper has no summary to merge into")
    
    if not payload.selected_qa:
        raise HTTPException(status_code=400, detail="No Q&A pairs provided")
    
    # Format Q&A content
    qa_content = "\n\n".join([
        f"Q: {qa['question']}\nA: {qa['answer']}"
        for qa in payload.selected_qa
    ])
    
    # Get LLM config
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    # Get merge prompt
    prompt = get_prompt(
        "merge_qa_to_summary",
        current_summary=current_summary,
        qa_content=qa_content,
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
    )
    if not response.choices:
        raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
    updated_summary = response.choices[0].message.content.strip()

    # Update summary in database
    db.update_paper_summary(paper["id"], updated_summary)
    
    return {
        "summary": updated_summary,
        "merged_count": len(payload.selected_qa),
        "model": model,
    }


@app.post("/summary/dedup")
async def dedup_summary(payload: DedupSummaryRequest) -> dict[str, Any]:
    """Remove duplicated content from a paper's summary.
    
    Uses LLM to identify and remove redundant information while
    preserving the most detailed version.
    """
    # Get paper from DB
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    current_summary = paper.get("summary", "")
    if not current_summary:
        raise HTTPException(status_code=400, detail="Paper has no summary")
    
    # Get LLM config
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    # Get dedup prompt
    prompt = get_prompt("dedup_summary", summary=current_summary)
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.1,
    )
    if not response.choices:
        raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
    deduped_summary = response.choices[0].message.content.strip()

    # Update summary in database
    db.update_paper_summary(paper["id"], deduped_summary)
    
    return {
        "summary": deduped_summary,
        "original_length": len(current_summary),
        "new_length": len(deduped_summary),
        "model": model,
    }




class AbbreviateRequest(BaseModel):
    title: str


@app.post("/abbreviate")
async def abbreviate(payload: AbbreviateRequest) -> dict[str, Any]:
    """Generate a short abbreviation for a paper title."""
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    prompt = get_prompt("abbreviate", title=payload.title)
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.1,
    )
    if not response.choices:
        raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
    abbrev = response.choices[0].message.content.strip()
    # Clean up any quotes or extra formatting
    abbrev = abbrev.strip('"\'').strip()
    return {"abbreviation": abbrev, "model": model}


class ReabbreviateRequest(BaseModel):
    arxiv_id: str
    custom_name: Optional[str] = None  # If provided, use this instead of LLM


@app.post("/papers/reabbreviate")
async def reabbreviate_paper(payload: ReabbreviateRequest) -> dict[str, Any]:
    """Re-generate abbreviation for an existing paper and update tree node.
    
    If custom_name is provided, use it directly.
    Otherwise, generate using LLM with temperature=0.7 for variety.
    """
    # Get paper from DB
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    if payload.custom_name:
        # Use custom name provided by user
        abbrev = payload.custom_name.strip()
    else:
        # Generate new abbreviation using LLM
        endpoint_config = _get_endpoint_config()
        base_url = endpoint_config["llm_base_url"]
        api_key = endpoint_config["api_key"]
        model = _resolve_model(base_url, api_key)
        client = _get_async_openai_client(base_url, api_key)
        
        prompt = get_prompt("abbreviate", title=paper["title"])
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.7,  # Higher temperature for variety on re-runs
        )
        if not response.choices:
            raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
        abbrev = response.choices[0].message.content.strip().strip('"\'').strip()
    
    # Store abbreviation in papers table
    db.update_paper_abbreviation(paper["id"], abbrev)
    
    # Update tree node name
    node_id = db.find_paper_node_id(paper["id"])
    if node_id:
        db.update_tree_node_name(node_id, abbrev)
    
    return {"abbreviation": abbrev, "node_id": node_id, "paper_id": paper["id"]}


@app.post("/papers/reabbreviate-all")
async def reabbreviate_all_papers() -> dict[str, Any]:
    """Re-generate abbreviations for ALL papers in parallel."""
    # Get all papers
    papers = db.get_all_papers()
    if not papers:
        return {"updated": 0, "results": []}
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    sem = asyncio.Semaphore(3)

    async def abbreviate_one(paper: dict) -> dict[str, Any]:
        async with sem:
            prompt = get_prompt("abbreviate", title=paper["title"])
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1,
            )
            if not response.choices:
                raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
            abbrev = response.choices[0].message.content.strip().strip('"\'').strip()
            # Store abbreviation in papers table
            db.update_paper_abbreviation(paper["id"], abbrev)
            # Update tree node name
            node_id = db.find_paper_node_id(paper["id"])
            if node_id:
                db.update_tree_node_name(node_id, abbrev)
            return {"arxiv_id": paper["arxiv_id"], "abbreviation": abbrev}

    # Run with bounded concurrency (max 3 simultaneous LLM requests)
    results = await asyncio.gather(*[abbreviate_one(p) for p in papers])
    
    return {"updated": len(results), "results": results}


class RenameCategoryRequest(BaseModel):
    node_id: str
    custom_name: Optional[str] = None  # If provided, use this instead of LLM


@app.post("/categories/rename")
async def rename_category(payload: RenameCategoryRequest) -> dict[str, Any]:
    """Rename a category node using LLM or custom name.
    
    Uses contrastive naming logic (considers siblings and their children).
    If custom_name is provided, uses it directly.
    Otherwise, generates using LLM with temperature=0.7 for variety.
    """
    import naming
    
    try:
        result = await naming.rename_single_category(
            node_id=payload.node_id,
            custom_name=payload.custom_name,
            temperature=0.7,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Database Switching Endpoints
# =============================================================================

class DbSwitchRequest(BaseModel):
    database: str = Field(..., description="Database name to switch to (e.g., 'paper_curator_test')")

class DbInitRequest(BaseModel):
    database: str = Field(..., description="Database name to create and initialize")
    drop_existing: bool = Field(default=False, description="Drop existing database before creating")


@app.get("/db/status")
def db_status() -> dict[str, Any]:
    """Return the currently active database name and connection info."""
    active = db.get_active_database()
    # Quick connectivity check
    try:
        with db.get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), COUNT(*) FROM papers")
                row = cur.fetchone()
                return {
                    "database": active,
                    "connected_to": row[0],
                    "paper_count": row[1],
                    "status": "ok",
                }
    except Exception as e:
        return {
            "database": active,
            "connected_to": None,
            "paper_count": None,
            "status": f"error: {e}",
        }


@app.post("/db/switch")
def db_switch(payload: DbSwitchRequest) -> dict[str, Any]:
    """Switch the active database at runtime.

    All subsequent DB operations will use the new database.
    The previous database name is returned so callers can switch back.
    """
    previous = db.switch_database(payload.database)
    # Verify connectivity
    try:
        with db.get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database(), COUNT(*) FROM papers")
                row = cur.fetchone()
                return {
                    "previous_database": previous,
                    "current_database": row[0],
                    "paper_count": row[1],
                    "status": "ok",
                }
    except Exception as e:
        # Roll back to previous database on failure
        db.switch_database(previous)
        raise HTTPException(
            status_code=400,
            detail=f"Cannot connect to database '{payload.database}': {e}. Reverted to '{previous}'.",
        )


@app.post("/db/init")
def db_init(payload: DbInitRequest) -> dict[str, Any]:
    """Create and initialize a database with the paper-curator schema.

    This is used to set up a fresh test database. The backend stays connected
    to whatever database was active before this call.
    """
    target = payload.database
    # Safety: never allow dropping the production database
    if payload.drop_existing and target == "paper_curator":
        raise HTTPException(status_code=400, detail="Refusing to drop the production database")

    try:
        admin_conn = db.get_admin_connection("postgres")
        cur = admin_conn.cursor()

        if payload.drop_existing:
            # Terminate existing connections first
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (target,),
            )
            cur.execute(f'DROP DATABASE IF EXISTS "{target}"')

        # Create database
        cur.execute(f'CREATE DATABASE "{target}" OWNER curator')
        cur.close()
        admin_conn.close()

        # Run init.sql on the new database
        init_sql_paths = [
            pathlib.Path("src/backend/init.sql"),
            pathlib.Path("init.sql"),
            pathlib.Path("../backend/init.sql"),
        ]
        init_sql = None
        for p in init_sql_paths:
            if p.exists():
                init_sql = p.read_text()
                break

        if init_sql:
            conn = db.get_admin_connection(dbname=target)
            with conn.cursor() as c:
                c.execute(init_sql)
            conn.close()

        return {
            "database": target,
            "status": "created",
            "schema_initialized": init_sql is not None,
        }

    except Exception as dup_err:
        if "already exists" in str(dup_err):
            return {
                "database": target,
                "status": "already_exists",
                "schema_initialized": False,
            }
        raise HTTPException(status_code=500, detail=f"Failed to init database '{target}': {dup_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init database '{target}': {e}")


# =============================================================================
# Database & Tree Endpoints
# =============================================================================

@app.post("/papers/save")
async def save_paper(payload: SavePaperRequest) -> dict[str, Any]:
    """Save a paper to the database.
    
    Note: Paper is saved but not added to tree immediately.
    Tree will be rebuilt using embedding-based clustering when user clicks "Re-classify"
    or automatically if rebuild_on_ingest is enabled.
    """
    # Create paper in DB
    paper_id = db.create_paper(
        arxiv_id=payload.arxiv_id,
        title=payload.title,
        authors=payload.authors,
        abstract=payload.abstract,
        summary=payload.summary,
        abbreviation=payload.abbreviation,
        pdf_path=payload.pdf_path,
        latex_path=payload.latex_path,
        pdf_url=payload.pdf_url,
        published_at=payload.published_at,
    )

    # Index the paper (chunk + embed) so it has an embedding for classification.
    # The summarize step runs *before* /papers/save, so rag_answer_async can't
    # persist chunks (paper doesn't exist in DB yet).  We fix that here.
    indexed = False
    if payload.pdf_path:
        try:
            endpoint_config = _get_endpoint_config()
            embed_base_url = endpoint_config["embedding_base_url"]
            api_key = endpoint_config["api_key"]
            embed_model = _resolve_model(embed_base_url, api_key)
            embed_client = _get_async_openai_client(embed_base_url, api_key)
            result = await rag.index_paper_async(
                paper_id, payload.pdf_path, embed_client, embed_model
            )
            indexed = result.get("indexed", False)
        except Exception as e:
            # Non-fatal: paper is saved, embedding can be computed later
            print(f"[save_paper] Warning: failed to index paper {paper_id}: {e}")

    # Place paper into tree (or bootstrap if tree is empty)
    placement_result: dict[str, Any] = {}
    if indexed:
        asyncio.create_task(_place_or_bootstrap_async(paper_id))
        placement_result = {"placement_scheduled": True}

    return {
        "paper_id": paper_id,
        "indexed": indexed,
        "message": "Paper saved and queued for tree placement.",
        **placement_result,
    }


async def _bootstrap_tree_async() -> None:
    """Full tree build used when the tree is empty (bootstrap on first paper ingest)."""
    import clustering
    import naming
    cluster_result = await asyncio.to_thread(clustering.build_tree_from_clusters)
    if cluster_result.get("total_papers", 0) < 2:
        print(
            f"Bootstrap skipped: {cluster_result.get('message', 'Not enough papers')}",
            flush=True,
        )
        return
    tree_structure = {
        "name": cluster_result.get("name", "AI Papers"),
        "children": cluster_result.get("children", []),
    }
    await asyncio.to_thread(clustering.write_tree_to_database, tree_structure)
    naming_result = await naming.name_tree_nodes()
    print(
        f"Bootstrap complete: papers={cluster_result.get('total_papers', 0)}, "
        f"clusters={cluster_result.get('total_clusters', 0)}, "
        f"nodes_named={naming_result.get('nodes_named', 0)}",
        flush=True,
    )


async def _place_or_bootstrap_async(paper_id: int) -> None:
    """Background task: place paper in tree, or bootstrap if tree is empty.

    Acquires _tree_lock so concurrent ingestions are serialised.
    """
    import clustering
    async with _tree_lock:
        tree = db.get_tree()
        if not tree.get("children"):
            # Tree is empty — bootstrap
            print(f"[placement] Tree empty, bootstrapping for paper {paper_id}", flush=True)
            await _bootstrap_tree_async()
        else:
            result = await asyncio.to_thread(clustering.place_paper_in_tree, paper_id)
            print(f"[placement] {result}", flush=True)


async def _fetch_slack_messages(client, channel_id: str) -> list[dict[str, Any]]:
    """Fetch all messages from a Slack channel with pagination.
    
    Returns list of message objects containing 'text' field.
    Uses asyncio.to_thread to avoid blocking the event loop.
    """
    return await fetch_slack_messages(client, channel_id)


@app.post("/papers/batch-ingest")
async def batch_ingest(payload: BatchIngestRequest) -> dict[str, Any]:
    """Batch ingest PDFs from a local directory or arXiv papers from a Slack channel.
    
    For directory:
    - For each PDF:
      1. Extract text to get title (from first page)
      2. Classify into category
      3. Generate abbreviation
      4. Summarize
      5. Save to database
    
    For Slack channel:
    - Fetch all messages from channel
    - Extract arXiv IDs/URLs from messages
    - Ingest each arXiv paper found
    """
    import glob
    import shutil
    
    results = []
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    # Get skip_existing from config (payload.skip_existing is ignored now - config.yaml controls this)
    ingestion_config = _get_ingestion_config()
    skip_existing = ingestion_config["skip_existing"]
    
    # Verify endpoints before starting
    try:
        model = _resolve_model(base_url, api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM endpoint not available: {e}")
    
    try:
        embed_model = _resolve_model(embed_base_url, api_key)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding endpoint not available: {e}")
    
    llm_client = _get_async_openai_client(base_url, api_key)
    embed_client = _get_async_openai_client(embed_base_url, api_key)
    
    # Progress logging for Slack ingestion
    progress_log = []
    
    def log_progress(message: str):
        """Log progress message (for debugging and user feedback)."""
        import time as _time
        ts = _time.strftime("%H:%M:%S")
        progress_log.append(message)
        print(f"[Slack Ingest {ts}] {message}")  # Also print to Docker logs
    
    # Determine source type
    if payload.slack_channel:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
        # Slack channel ingestion - wrap entire block to capture progress_log in errors
        try:
            log_progress("Starting Slack channel ingestion...")
            if not payload.slack_token:
                raise HTTPException(status_code=400, detail="Slack token is required for Slack channel ingestion")
            
            # Initialize Slack client
            log_progress("Initializing Slack client...")
            slack_client = WebClient(token=payload.slack_token)
            
            # Test token validity first
            log_progress("Testing Slack token validity...")
            try:
                auth_test = await asyncio.to_thread(slack_client.auth_test)
                if not auth_test["ok"]:
                    raise HTTPException(status_code=401, detail=f"Invalid Slack token: {auth_test.get('error', 'Unknown error')}")
                log_progress(f"✓ Token valid (workspace: {auth_test.get('team', 'Unknown')})")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=401, detail=f"Failed to authenticate with Slack: {e}")
            
            # Resolve channel ID (wrap blocking call)
            log_progress(f"Resolving channel: {payload.slack_channel}")
            try:
                channel_id = await _resolve_slack_channel_id(
                    slack_client,
                    payload.slack_channel
                )
                log_progress(f"✓ Channel resolved: {channel_id}")
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error resolving Slack channel: {e}")
            
            # Fetch all messages
            log_progress("Fetching messages from Slack channel (this may take a while for large channels)...")
            try:
                messages = await _fetch_slack_messages(slack_client, channel_id)
                log_progress(f"✓ Fetched {len(messages)} messages from channel")
                if not messages:
                    return {
                        "total": 0,
                        "success": 0,
                        "skipped": 0,
                        "errors": 0,
                        "results": [],
                        "progress_log": progress_log,
                        "message": "No messages found in channel",
                    }
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                error_detail = f"Error fetching Slack messages: {str(e)}\n{traceback.format_exc()}"
                log_progress(f"✗ Error fetching messages: {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)
            
            # Extract arXiv IDs from all messages
            log_progress("Extracting arXiv IDs from messages...")
            all_arxiv_ids = set()
            debug_sample_logged = False
            try:
                for i, message in enumerate(messages):
                    if (i + 1) % 100 == 0:
                        log_progress(f"  Processing message {i+1}/{len(messages)}...")
                    
                    # Debug: log first message with attachments or blocks to understand structure
                    if not debug_sample_logged and (message.get("attachments") or message.get("blocks")):
                        import json
                        log_progress(f"  [DEBUG] Sample message structure: {json.dumps(message, indent=2)[:2000]}")
                        debug_sample_logged = True
                    
                    # Extract from main text
                    text = message.get("text", "")
                    arxiv_ids = _extract_arxiv_ids_from_text(text)
                    all_arxiv_ids.update(arxiv_ids)
                    
                    # Also check attachments (unfurled URLs)
                    for attachment in message.get("attachments", []):
                        # Check title_link, original_url, and text in attachments
                        for field in ["title_link", "original_url", "from_url", "text", "fallback"]:
                            if field in attachment:
                                arxiv_ids = _extract_arxiv_ids_from_text(str(attachment[field]))
                                all_arxiv_ids.update(arxiv_ids)
                    
                    # Also check blocks (rich text format)
                    for block in message.get("blocks", []):
                        # Extract URLs from section blocks
                        if block.get("type") == "section":
                            block_text = block.get("text", {})
                            if isinstance(block_text, dict):
                                arxiv_ids = _extract_arxiv_ids_from_text(block_text.get("text", ""))
                                all_arxiv_ids.update(arxiv_ids)
                        # Extract from rich_text blocks
                        if block.get("type") == "rich_text":
                            for element in block.get("elements", []):
                                for sub_element in element.get("elements", []):
                                    if sub_element.get("type") == "link":
                                        url = sub_element.get("url", "")
                                        arxiv_ids = _extract_arxiv_ids_from_text(url)
                                        all_arxiv_ids.update(arxiv_ids)
                                    elif sub_element.get("type") == "text":
                                        arxiv_ids = _extract_arxiv_ids_from_text(sub_element.get("text", ""))
                                        all_arxiv_ids.update(arxiv_ids)
                log_progress(f"✓ Found {len(all_arxiv_ids)} unique arXiv papers")
            except Exception as e:
                import traceback
                error_detail = f"Error extracting arXiv IDs from messages: {str(e)}\n{traceback.format_exc()}"
                log_progress(f"✗ Error extracting arXiv IDs: {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)
            
            if not all_arxiv_ids:
                return {
                    "total": 0,
                    "success": 0,
                    "skipped": 0,
                    "errors": 0,
                    "results": [],
                    "progress_log": progress_log,
                }
            
            # Phase 1: Batch DB check - filter existing papers in one query
            arxiv_ids_list = sorted(list(all_arxiv_ids))
            if skip_existing:
                existing_ids = db.get_existing_arxiv_ids(arxiv_ids_list)
                new_ids = [aid for aid in arxiv_ids_list if aid not in existing_ids]
                for aid in arxiv_ids_list:
                    if aid in existing_ids:
                        results.append({"file": aid, "status": "skipped", "reason": "already exists"})
                log_progress(f"✓ {len(existing_ids)} already exist, {len(new_ids)} new papers to ingest")
            else:
                new_ids = arxiv_ids_list
                log_progress(f"Skipping disabled, will process all {len(new_ids)} papers")
            
            # Apply limit if specified
            if payload.limit and len(new_ids) > payload.limit:
                log_progress(f"Limiting ingestion from {len(new_ids)} to {payload.limit} papers")
                new_ids = new_ids[:payload.limit]
            
            if not new_ids:
                log_progress("No new papers to ingest")
            else:
                # Phase 2: Parallel ingestion
                max_concurrent = ingestion_config.get("max_concurrent", 5)
                semaphore = asyncio.Semaphore(max_concurrent)
                log_progress(f"Starting parallel ingestion of {len(new_ids)} papers (concurrency={max_concurrent})...")
                
                # Counters for progress tracking (thread-safe via asyncio single-thread model)
                completed_count = [0]
                
                async def ingest_single_paper(arxiv_id: str) -> dict[str, Any]:
                    """Ingest a single arXiv paper. Runs within semaphore."""
                    import time as _time
                    t_wait_start = _time.monotonic()
                    async with semaphore:
                        t_start = _time.monotonic()
                        wait_s = t_start - t_wait_start
                        try:
                            # 1. Fetch arXiv metadata
                            log_progress(f"  [{arxiv_id}] Fetching metadata from arXiv...")
                            arxiv_client = arxiv.Client()
                            search = arxiv.Search(id_list=[arxiv_id])
                            arxiv_results_list = await asyncio.to_thread(
                                lambda: list(arxiv_client.results(search))
                            )
                            
                            if not arxiv_results_list:
                                log_progress(f"  [{arxiv_id}] ✗ arXiv paper not found")
                                return {"file": arxiv_id, "status": "error", "reason": f"arXiv paper not found: {arxiv_id}"}
                            
                            arxiv_result = arxiv_results_list[0]
                            title = arxiv_result.title
                            authors = [author.name for author in arxiv_result.authors]
                            abstract = arxiv_result.summary
                            pdf_url = arxiv_result.pdf_url
                            t_arxiv = _time.monotonic() - t_start
                            log_progress(f"  [{arxiv_id}] ✓ Found: {title} (arxiv={t_arxiv:.1f}s, wait={wait_s:.1f}s)")
                            
                            # 2. Download PDF
                            log_progress(f"  [{arxiv_id}] Downloading PDF...")
                            pdf_path = await _download_arxiv_pdf(arxiv_id, pdf_url)
                            t_pdf = _time.monotonic() - t_start
                            log_progress(f"  [{arxiv_id}] ✓ PDF downloaded ({t_pdf:.1f}s)")
                            
                            # 3. Abbreviate
                            log_progress(f"  [{arxiv_id}] Generating abbreviation...")
                            abbrev_prompt = get_prompt("abbreviate", title=title)
                            abbrev_resp = await llm_client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": abbrev_prompt}],
                                max_tokens=20,
                                temperature=0.1,
                            )
                            if not abbrev_resp.choices:
                                raise RuntimeError(f"LLM returned empty response for abbreviation of {arxiv_id}")
                            abbreviation = abbrev_resp.choices[0].message.content.strip().strip('"\'')
                            log_progress(f"  [{arxiv_id}] ✓ Abbreviation: {abbreviation}")
                            
                            # 4. Save to database first (without summary)
                            log_progress(f"  [{arxiv_id}] Saving to database...")
                            db_paper_id = db.create_paper(
                                arxiv_id=arxiv_id,
                                title=title,
                                authors=authors,
                                abstract=abstract[:1000],
                                pdf_path=pdf_path,
                            )
                            if db_paper_id is None:
                                raise RuntimeError(f"Failed to create DB record for {arxiv_id}")
                            
                            # 5. Summarize using RAG (auto-indexes: chunks + embedding)
                            log_progress(f"  [{arxiv_id}] Summarizing paper...")
                            prompt_id, prompt_hash, prompt_body = _load_prompt()
                            summary = await rag.rag_answer_async(
                                question=prompt_body,
                                pdf_path=pdf_path,
                                arxiv_id=arxiv_id,
                                llm_client=llm_client,
                                llm_model=model,
                                embed_client=embed_client,
                                embed_model=embed_model,
                            )
                            db.update_paper_summary(db_paper_id, summary)
                            t_summary = _time.monotonic() - t_start
                            log_progress(f"  [{arxiv_id}] ✓ Summary generated ({t_summary:.1f}s)")
                            
                            # Schedule background tree placement
                            asyncio.create_task(_place_or_bootstrap_async(db_paper_id))

                            t_total = _time.monotonic() - t_start
                            completed_count[0] += 1
                            log_progress(f"  [{arxiv_id}] ✓ Ingested ({completed_count[0]}/{len(new_ids)}) total={t_total:.1f}s")
                            return {
                                "file": arxiv_id,
                                "status": "success",
                                "paper_id": arxiv_id,
                                "title": title,
                                "abbreviation": abbreviation,
                            }
                        except Exception as e:
                            import traceback
                            error_msg = str(e)
                            log_progress(f"  [{arxiv_id}] ✗ Error: {error_msg}")
                            tb = traceback.format_exc()
                            if len(tb) > 500:
                                tb = tb[:500] + "... (truncated)"
                            completed_count[0] += 1
                            return {"file": arxiv_id, "status": "error", "reason": f"{error_msg}\n{tb}"}
                
                # Run all tasks in parallel (bounded by semaphore)
                parallel_results = await asyncio.gather(
                    *[ingest_single_paper(aid) for aid in new_ids],
                    return_exceptions=False,
                )
                results.extend(parallel_results)
                
                # Phase 3: Queue successfully ingested IDs for background Semantic Scholar metadata fetch
                successful_ids = [r["file"] for r in parallel_results if r.get("status") == "success"]
                if successful_ids:
                    _append_pending_metadata(successful_ids)
                    log_progress(f"✓ Added {len(successful_ids)} papers to pending metadata queue")
        except HTTPException as e:
            # Include progress_log in HTTPException response
            import json
            detail = e.detail
            if isinstance(detail, str):
                detail = {"message": detail, "progress_log": progress_log}
            elif isinstance(detail, dict):
                detail["progress_log"] = progress_log
            else:
                detail = {"message": str(detail), "progress_log": progress_log}
            raise HTTPException(status_code=e.status_code, detail=detail)
        except Exception as e:
            # Catch any unexpected errors and include progress_log
            import traceback
            error_detail = f"Unexpected error during Slack ingestion: {str(e)}\n{traceback.format_exc()}"
            log_progress(f"✗ Fatal error: {error_detail}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": error_detail,
                    "progress_log": progress_log
                }
            )
    
    elif payload.directory:
        # Directory ingestion (existing logic)
        # Handle paths: /Users/hyl/xxx becomes /host_home/xxx in Docker
        dir_path = payload.directory
        home_dir = os.environ.get("HOME", "/root")
        
        # Check if running in Docker (home is /root) and path starts with typical macOS path
        if home_dir == "/root" and dir_path.startswith("/Users/"):
            # Extract the path after /Users/username/
            parts = dir_path.split("/")
            if len(parts) > 3:
                # /Users/username/rest -> /host_home/rest
                docker_path = "/host_home/" + "/".join(parts[3:])
                directory = pathlib.Path(docker_path)
            else:
                directory = pathlib.Path(dir_path)
        else:
            directory = pathlib.Path(dir_path)
        
        if not directory.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Directory not found: {payload.directory} (mapped to {directory})"
            )
        
        # Find all PDFs
        pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
        if not pdf_files:
            raise HTTPException(status_code=400, detail=f"No PDF files found in: {payload.directory}")
        
        # Batch check for existing papers
        paper_ids_map = {}  # paper_id -> pdf_file
        for pdf_file in pdf_files:
            filename = pdf_file.stem
            paper_id = f"local_{hashlib.md5(filename.encode()).hexdigest()[:12]}"
            paper_ids_map[paper_id] = pdf_file
        
        if skip_existing:
            existing_ids = db.get_existing_arxiv_ids(list(paper_ids_map.keys()))
            for pid, pf in paper_ids_map.items():
                if pid in existing_ids:
                    results.append({"file": pf.name, "status": "skipped", "reason": "already exists"})
            new_files = {pid: pf for pid, pf in paper_ids_map.items() if pid not in existing_ids}
        else:
            new_files = paper_ids_map
        
        # Apply limit if specified
        if payload.limit and len(new_files) > payload.limit:
            # Keep only the first N files
            limited_keys = list(new_files.keys())[:payload.limit]
            new_files = {k: new_files[k] for k in limited_keys}
        
        if new_files:
            max_concurrent = ingestion_config.get("max_concurrent", 5)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def ingest_local_pdf(paper_id: str, pdf_file: pathlib.Path) -> dict[str, Any]:
                async with semaphore:
                    try:
                        # Copy PDF to storage
                        storage_dir = pathlib.Path("storage/downloads")
                        storage_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = storage_dir / f"{paper_id}.pdf"
                        shutil.copy2(pdf_file, dest_path)
                        pdf_path = str(dest_path)
                        
                        # Extract text using pymupdf
                        try:
                            full_text = rag.extract_pdf_text(pdf_file)
                            first_chunk = full_text[:2000]
                        except Exception as ex:
                            return {"file": pdf_file.name, "status": "error", "reason": f"PDF extract failed: {ex}"}
                        
                        if not first_chunk:
                            return {"file": pdf_file.name, "status": "error", "reason": "PDF extraction returned empty text"}
                        
                        title = pdf_file.stem.replace("_", " ").replace("-", " ").strip()
                        
                        # Abbreviate
                        abbrev_prompt = get_prompt("abbreviate", title=title)
                        abbrev_resp = await llm_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": abbrev_prompt}],
                            max_tokens=20,
                            temperature=0.1,
                        )
                        if not abbrev_resp.choices:
                            raise RuntimeError(f"LLM returned empty response for abbreviation of {paper_id}")
                        abbreviation = abbrev_resp.choices[0].message.content.strip().strip('"\'')

                        # Create paper in DB first (without summary)
                        db_paper_id = db.create_paper(
                            arxiv_id=paper_id,
                            title=title,
                            authors=[],
                            abstract=first_chunk[:1000],
                            pdf_path=pdf_path,
                        )
                        if db_paper_id is None:
                            raise RuntimeError(f"Failed to create DB record for {paper_id}")

                        # Summarize using RAG (will auto-index chunks)
                        prompt_id, prompt_hash, prompt_body = _load_prompt()
                        summary = await rag.rag_answer_async(
                            question=prompt_body,
                            pdf_path=pdf_path,
                            arxiv_id=paper_id,
                            llm_client=llm_client,
                            llm_model=model,
                            embed_client=embed_client,
                            embed_model=embed_model,
                        )
                        
                        # Update paper with summary
                        db.update_paper_summary(db_paper_id, summary)

                        # Schedule background tree placement
                        asyncio.create_task(_place_or_bootstrap_async(db_paper_id))

                        return {
                            "file": pdf_file.name,
                            "status": "success",
                            "paper_id": paper_id,
                            "title": title,
                            "abbreviation": abbreviation,
                        }
                    except Exception as e:
                        return {"file": pdf_file.name, "status": "error", "reason": str(e)}
            
            dir_results = await asyncio.gather(
                *[ingest_local_pdf(pid, pf) for pid, pf in new_files.items()],
                return_exceptions=False,
            )
            results.extend(dir_results)
    else:
        # Neither slack_channel nor directory provided
        raise HTTPException(status_code=400, detail="Either 'directory' or 'slack_channel' must be provided")
    
    # Calculate summary statistics
    success_count = sum(1 for r in results if r["status"] == "success")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    # Determine total count based on source
    if payload.slack_channel:
        # For Slack, total is the number of unique arXiv IDs found
        total = len(results)  # All unique arXiv IDs (success + skipped + errors)
    elif payload.directory:
        total = len(pdf_files)
    else:
        total = 0
    
    # Add final summary to progress log
    if payload.slack_channel:
        log_progress(f"✓ Completed: {success_count} success, {skip_count} skipped, {error_count} errors")
    
    # Trigger background Semantic Scholar metadata fetch
    if success_count > 0:
        asyncio.create_task(_run_background_metadata_fetch())
        if payload.slack_channel:
            log_progress("Background Semantic Scholar metadata fetch started")
    
    return {
        "total": total,
        "success": success_count,
        "skipped": skip_count,
        "errors": error_count,
        "results": results,
        "progress_log": progress_log if payload.slack_channel else [],
    }


@app.post("/papers/categorize")
async def categorize_papers(full: bool = False) -> dict[str, Any]:
    """Recategorize papers.

    - Default (``full=false``): partial recategorize — re-clusters only dirty nodes,
      then names the new child nodes.
    - ``?full=true``: full rebuild — clusters all papers from scratch and names
      all nodes. Slow (may take 30+ minutes for large libraries).

    Acquires the tree lock so concurrent placements are blocked until done.
    """
    import clustering
    import naming

    # Pre-check: LLM endpoint must be reachable (naming step requires it)
    endpoint_config = _get_endpoint_config()
    try:
        _resolve_model(endpoint_config["llm_base_url"], endpoint_config["api_key"])
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"LLM endpoint unreachable — categorize aborted: {e}",
        )

    async with _tree_lock:
        if full:
            # --- Full rebuild ---
            # Ensure all papers have embeddings
            papers_without_embeddings = db.get_papers_missing_embeddings()
            if papers_without_embeddings:
                embed_base_url = endpoint_config["embedding_base_url"]
                api_key = endpoint_config["api_key"]
                try:
                    embed_model = _resolve_model(embed_base_url, api_key)
                    embed_client = _get_async_openai_client(embed_base_url, api_key)
                except Exception:
                    embed_client = None
                    embed_model = None
                for arxiv_id in papers_without_embeddings:
                    await rag.ensure_paper_embedding(arxiv_id, embed_client, embed_model)

            cluster_result = await asyncio.to_thread(clustering.build_tree_from_clusters)
            if cluster_result is None:
                raise HTTPException(status_code=500, detail="Clustering failed")
            if cluster_result["total_papers"] < 2:
                return {
                    "message": cluster_result.get("message", "Not enough papers with embeddings"),
                    "mode": "full",
                    "papers_classified": cluster_result["total_papers"],
                    "clusters_created": 0,
                    "nodes_named": 0,
                }
            tree_structure = {
                "name": cluster_result.get("name", "AI Papers"),
                "children": cluster_result.get("children", []),
            }
            await asyncio.to_thread(clustering.write_tree_to_database, tree_structure)
            # Clear all dirty flags — full rebuild makes them irrelevant
            db.clear_dirty_tree_nodes(db.get_dirty_tree_nodes())
            naming_result = await naming.name_tree_nodes()
            return {
                "message": "Full rebuild completed and tree saved",
                "mode": "full",
                "papers_classified": cluster_result["total_papers"],
                "clusters_created": cluster_result["total_clusters"],
                "nodes_named": naming_result.get("nodes_named", 0),
            }
        else:
            # --- Partial recategorize ---
            dirty_node_ids = db.get_dirty_tree_nodes()
            if not dirty_node_ids:
                return {
                    "message": "No dirty nodes — tree is already up to date",
                    "mode": "partial",
                    "recategorized": 0,
                    "nodes_named": 0,
                }
            recategorize_result = await asyncio.to_thread(
                clustering.partial_recategorize, dirty_node_ids
            )
            new_child_ids = recategorize_result.get("new_child_nodes", [])
            naming_result = await naming.name_tree_nodes(node_ids=new_child_ids if new_child_ids else None)
            return {
                "message": f"Partial recategorize complete: {recategorize_result['recategorized']} nodes rebuilt",
                "mode": "partial",
                "recategorized": recategorize_result["recategorized"],
                "nodes": recategorize_result["nodes"],
                "nodes_named": naming_result.get("nodes_named", 0),
            }


@app.get("/papers/{arxiv_id}/cached-data")
def get_paper_cached_data(arxiv_id: str, require_embedding: bool = True) -> dict[str, Any]:
    """Get all cached data for a paper (repos, refs, similar, queries, structured_summary).
    
    Used to restore state when selecting a paper in the GUI.
    
    Args:
        arxiv_id: The arXiv ID of the paper
        require_embedding: If True (default), returns 404 for papers without embeddings.
                          This ensures only "fully ingested" papers are returned.
    """
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
    
    # Treat papers without embeddings as "incomplete" - they need re-ingestion
    if require_embedding and paper.get("embedding") is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper {arxiv_id} exists but has no embedding - needs re-ingestion"
        )
    
    cached_data = db.get_paper_cached_data(paper["id"])
    structured_summary = db.get_paper_structured_summary(paper["id"])
    
    return {
        "arxiv_id": arxiv_id,
        "paper_id": paper["id"],
        "structured_summary": structured_summary,
        **cached_data,
    }


@app.get("/tree")
def get_tree() -> dict[str, Any]:
    """Get the full tree structure (already in frontend format with paper metadata)."""
    return db.get_tree()



@app.delete("/papers/{arxiv_id}")
def delete_paper(arxiv_id: str) -> dict[str, Any]:
    """Delete a paper from database.
    
    Removes the paper from the database (which cascades to delete
    all associated data like references, similar papers, queries, repos).
    
    Note: Tree will need to be rebuilt after deletion to reflect changes.
    """
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")
    
    # Delete paper (cascades to all related data)
    db.delete_paper(paper["id"])
    
    # Note: Tree structure is stored as JSONB, so it will contain stale references
    # User should rebuild tree after deleting papers
    
    return {
        "status": "ok",
        "deleted_arxiv_id": arxiv_id,
        "deleted_paper_id": paper["id"],
        "message": "Paper deleted. Rebuild tree to update classification.",
    }


class PrefetchRequest(BaseModel):
    arxiv_id: str
    title: str


@app.post("/papers/prefetch")
async def prefetch_paper_data(payload: PrefetchRequest) -> dict[str, Any]:
    """Prefetch repos, references, and similar papers in parallel.
    
    Call this after paper ingest to cache auxiliary data for faster UI loading.
    """
    apis_config = _get_external_apis_config()
    api_key = apis_config.get("semantic_scholar_api_key")
    
    # Run all three fetches in parallel
    repos_task = _prefetch_repos(payload.arxiv_id, payload.title, apis_config)
    refs_task = _get_semantic_scholar_references(payload.arxiv_id, api_key)
    similar_task = _get_semantic_scholar_recommendations(payload.arxiv_id, 10, api_key)
    
    repos, refs, similar = await asyncio.gather(
        repos_task, refs_task, similar_task,
        return_exceptions=True
    )
    
    return {
        "repos": repos if not isinstance(repos, Exception) else [],
        "references": refs if not isinstance(refs, Exception) else [],
        "similar": similar if not isinstance(similar, Exception) else [],
        "repos_error": str(repos) if isinstance(repos, Exception) else None,
        "refs_error": str(refs) if isinstance(refs, Exception) else None,
        "similar_error": str(similar) if isinstance(similar, Exception) else None,
    }


async def _prefetch_repos(arxiv_id: str, title: str, apis_config: dict) -> list[dict]:
    """Helper to prefetch repos for prefetch endpoint."""
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if paper:
        cached = db.get_cached_repos(paper["id"])
        if cached:
            return cached
    
    repos = []
    if apis_config["papers_with_code_enabled"]:
        pwc_repos = await _search_papers_with_code(title)
        repos.extend(pwc_repos)
    
    if apis_config["github_search_enabled"]:
        has_official = any(r.get("is_official") for r in repos)
        if not has_official:
            github_repos = await _search_github(title, apis_config.get("github_token"))
            repos.extend(github_repos)
    
    # Cache results
    if paper and repos:
        for repo in repos:
            db.cache_repo(
                paper_id=paper["id"],
                source=repo.get("source", "unknown"),
                repo_url=repo.get("url"),
                repo_name=repo.get("name"),
                stars=repo.get("stars"),
                is_official=repo.get("is_official", False),
            )
    
    return repos


# =============================================================================
# Repository Search (Feature 4)
# =============================================================================

async def _search_papers_with_code(title: str) -> list[dict[str, Any]]:
    """Search Papers With Code for repositories."""
    client = get_http_client("paperswithcode", timeout=10.0)
    # Search for paper
    search_url = "https://paperswithcode.com/api/v1/papers/"
    response = await client.get(search_url, params={"q": title[:100]})
    if response.status_code != 200:
        return []
    
    data = response.json()
    results = data.get("results", [])
    if not results:
        return []
    
    repos = []
    for paper in results[:3]:  # Check top 3 matches
        paper_id = paper.get("id")
        if not paper_id:
            continue
        
        # Get repos for this paper
        repos_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
        repos_response = await client.get(repos_url)
        if repos_response.status_code == 200:
            repos_data = repos_response.json()
            for repo in repos_data.get("results", []):
                repos.append({
                    "source": "paperswithcode",
                    "repo_url": repo.get("url"),
                    "repo_name": repo.get("url", "").split("/")[-1] if repo.get("url") else None,
                    "stars": repo.get("stars"),
                    "is_official": repo.get("is_official", False),
                })
    return repos


async def _search_github(title: str, github_token: Optional[str] = None) -> list[dict[str, Any]]:
    """Search GitHub for repositories by paper title."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    client = get_http_client("github", timeout=10.0, headers={"Accept": "application/vnd.github.v3+json"})
    # Clean title for search
    clean_title = re.sub(r'[^\w\s]', '', title)[:50]
    search_url = "https://api.github.com/search/repositories"
    response = await client.get(
        search_url,
        params={"q": clean_title, "sort": "stars", "order": "desc", "per_page": 5},
        headers=headers,  # Override with auth if provided
    )
    if response.status_code != 200:
        return []
    
    data = response.json()
    repos = []
    for item in data.get("items", []):
        repos.append({
            "source": "github",
            "repo_url": item.get("html_url"),
            "repo_name": item.get("full_name"),
            "stars": item.get("stargazers_count"),
            "is_official": False,
        })
    return repos


@app.post("/repos/search")
async def search_repos(payload: RepoSearchRequest) -> dict[str, Any]:
    """Search for GitHub repositories associated with a paper."""
    apis_config = _get_external_apis_config()
    
    # Check cache first
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if paper:
        cached = db.get_cached_repos(paper["id"])
        if cached:
            return {"repos": cached, "from_cache": True}
    
    repos = []
    
    # 1. Try Papers With Code first
    if apis_config["papers_with_code_enabled"]:
        pwc_repos = await _search_papers_with_code(payload.title)
        repos.extend(pwc_repos)
    
    # 2. Fall back to GitHub search if no official repos found
    if apis_config["github_search_enabled"]:
        has_official = any(r.get("is_official") for r in repos)
        if not has_official:
            github_repos = await _search_github(payload.title, apis_config.get("github_token"))
            repos.extend(github_repos)
    
    # Cache results
    if paper and repos:
        for repo in repos:
            db.cache_repo(
                paper_id=paper["id"],
                source=repo["source"],
                repo_url=repo.get("repo_url"),
                repo_name=repo.get("repo_name"),
                stars=repo.get("stars"),
                is_official=repo.get("is_official", False),
            )
    
    return {"repos": repos, "from_cache": False}


# =============================================================================
# References (Feature 5)
# =============================================================================

async def _get_semantic_scholar_references(arxiv_id: str, api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """Get references from Semantic Scholar API."""
    # Clean arXiv ID (remove version suffix)
    clean_id = re.sub(r'v\d+$', '', arxiv_id)
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    client = get_http_client("semantic_scholar", timeout=15.0)
    # First get the paper ID
    paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}"
    params = {"fields": "references.title,references.authors,references.year,references.externalIds"}
    response = await client.get(paper_url, params=params, headers=headers)
    
    if response.status_code == 429:
        print("Semantic Scholar rate limited. Consider adding an API key to config/config.yaml")
        return []
    
    if response.status_code != 200:
        print(f"Semantic Scholar returned {response.status_code}: {response.text[:200]}")
        return []
    
    data = response.json()
    references = []
    for ref in data.get("references", []):
        arxiv_ref_id = None
        external_ids = ref.get("externalIds", {})
        if external_ids and "ArXiv" in external_ids:
            arxiv_ref_id = external_ids["ArXiv"]
        
        authors = [a.get("name", "") for a in ref.get("authors", [])]
        references.append({
            "cited_title": ref.get("title", "Unknown"),
            "cited_arxiv_id": arxiv_ref_id,
            "cited_authors": authors,
            "cited_year": ref.get("year"),
        })
    return references


@app.post("/references/fetch")
async def fetch_references(payload: ReferencesRequest) -> dict[str, Any]:
    """Fetch references for a paper."""
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    
    # Check cache first
    if paper:
        cached = db.get_references(paper["id"])
        if cached:
            return {"references": cached, "from_cache": True}
    
    apis_config = _get_external_apis_config()
    references = []
    
    # Try Semantic Scholar first
    if apis_config["semantic_scholar_enabled"]:
        api_key = apis_config.get("semantic_scholar_api_key")
        references = await _get_semantic_scholar_references(payload.arxiv_id, api_key)
    
    # TODO: Add LaTeX parsing fallback
    # TODO: Add PDF text parsing fallback
    
    # Cache references
    if paper and references:
        for ref in references:
            db.add_reference(
                source_paper_id=paper["id"],
                cited_title=ref["cited_title"],
                cited_arxiv_id=ref.get("cited_arxiv_id"),
                cited_authors=ref.get("cited_authors"),
                cited_year=ref.get("cited_year"),
            )
        # Refresh from DB to get IDs
        references = db.get_references(paper["id"])
    
    return {"references": references, "from_cache": False}


@app.post("/references/explain")
async def explain_reference(payload: ExplainReferenceRequest) -> dict[str, Any]:
    """Generate an explanation for a reference using LLM."""
    # Check cache
    ref = db.get_reference_by_id(payload.reference_id) if payload.reference_id else None
    if ref and ref.get("explanation"):
        return {"explanation": ref["explanation"], "from_cache": True}
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_async_openai_client(base_url, api_key)
    
    context_section = ""
    if payload.citation_context:
        context_section = f"\nCitation context: \"{payload.citation_context}\""
    
    prompt = get_prompt(
        "explain_reference",
        source_title=payload.source_paper_title,
        cited_title=payload.cited_title,
        context_section=context_section,
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    if not response.choices:
        raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
    explanation = response.choices[0].message.content.strip()

    # Cache the explanation
    if payload.reference_id:
        db.update_reference_explanation(payload.reference_id, explanation)
    
    return {"explanation": explanation, "from_cache": False}


# =============================================================================
# Similar Papers (Feature 6) - Uses Semantic Scholar Recommendations API
# =============================================================================

async def _get_semantic_scholar_recommendations(arxiv_id: str, limit: int = 10, api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """Get paper recommendations from Semantic Scholar API (searches the internet)."""
    # Clean arXiv ID (remove version suffix like v1, v2)
    clean_id = re.sub(r'v\d+$', '', arxiv_id)
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    
    client = get_http_client("semantic_scholar", timeout=15.0)
    # Use the Recommendations API to find similar papers from Semantic Scholar's 200M+ paper database
    rec_url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/arXiv:{clean_id}"
    params = {
        "fields": "title,abstract,authors,year,externalIds,citationCount,url",
        "limit": limit,
        "from": "all-cs",  # Search all CS papers, use "recent" for recent papers only
    }
    
    response = await client.get(rec_url, params=params, headers=headers)
    
    if response.status_code != 200:
        print(f"Semantic Scholar Recommendations API returned {response.status_code}: {response.text}")
        return []
    
    data = response.json()
    recommendations = []
    
    for paper in data.get("recommendedPapers", []):
        arxiv_ref_id = None
        external_ids = paper.get("externalIds", {})
        if external_ids and "ArXiv" in external_ids:
            arxiv_ref_id = external_ids["ArXiv"]
        
        authors = [a.get("name", "") for a in paper.get("authors", [])]
        recommendations.append({
            "arxiv_id": arxiv_ref_id,
            "title": paper.get("title", "Unknown"),
            "abstract": paper.get("abstract", ""),
            "authors": authors,
            "year": paper.get("year"),
            "citation_count": paper.get("citationCount", 0),
            "url": paper.get("url", ""),
            "source": "semantic_scholar",
        })
    
    return recommendations


@app.post("/papers/similar")
async def find_similar_papers(payload: SimilarPapersRequest) -> dict[str, Any]:
    """Find similar papers using Semantic Scholar Recommendations API (searches internet)."""
    # Get UI config for limit
    ui_config = _get_ui_config()
    limit = ui_config.get("max_similar_papers", 10)
    
    # Check cache first (optional - paper may not be in local DB)
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if paper:
        cached = db.get_cached_similar_papers(paper["id"])
        if cached:
            return {"similar_papers": cached, "from_cache": True}
    
    # Query Semantic Scholar Recommendations API (searches the internet)
    apis_config = _get_external_apis_config()
    api_key = apis_config.get("semantic_scholar_api_key")
    similar = await _get_semantic_scholar_recommendations(payload.arxiv_id, limit, api_key)
    
    # Cache results if paper is in our DB
    if paper and similar:
        for s in similar:
            db.cache_similar_paper(
                paper_id=paper["id"],
                similar_arxiv_id=s.get("arxiv_id"),
                similar_title=s["title"],
                similarity_score=None,  # No similarity score from API
            )
    
    return {"similar_papers": similar, "from_cache": False}


# =============================================================================
# Topic Query Endpoints (Multi-Paper RAG)
# =============================================================================

@app.post("/topic/search")
async def topic_search(payload: TopicSearchRequest) -> dict[str, Any]:
    """Search for papers similar to a topic using embedding similarity.
    
    Returns paginated results ordered by similarity score.
    """
    topic_config = _get_topic_query_config()
    limit = payload.limit or topic_config["max_papers_per_batch"]
    min_similarity = topic_config["similarity_threshold"]
    
    # Get embedding endpoint config
    endpoint_config = _get_endpoint_config()
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    # Embed the topic
    client = _get_async_openai_client(embed_base_url, api_key)
    embed_model = resolve_model(embed_base_url, api_key)
    
    response = await client.embeddings.create(
        model=embed_model,
        input=payload.topic,
    )
    topic_embedding = response.data[0].embedding
    
    # Search papers by embedding
    papers, has_more = db.search_papers_by_embedding(
        embedding=topic_embedding,
        limit=limit,
        offset=payload.offset,
        min_similarity=min_similarity,
        exclude_paper_ids=payload.exclude_paper_ids if payload.exclude_paper_ids else None,
    )
    
    return {
        "papers": papers,
        "has_more": has_more,
        "topic_embedding": topic_embedding,  # Return for creating topic
    }


@app.get("/topic/list")
async def topic_list() -> dict[str, Any]:
    """Get all topics ordered by recency."""
    topics = db.get_all_topics()
    return {"topics": topics}


@app.post("/topic/create")
async def topic_create(payload: TopicCreateRequest) -> dict[str, Any]:
    """Create a new topic for paper pool."""
    # Check if topic name already exists
    existing = db.get_topic_by_name(payload.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Topic with name '{payload.name}' already exists")
    
    # Get embedding for the topic
    endpoint_config = _get_endpoint_config()
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    
    client = _get_async_openai_client(embed_base_url, api_key)
    embed_model = resolve_model(embed_base_url, api_key)
    
    response = await client.embeddings.create(
        model=embed_model,
        input=payload.topic_query,
    )
    topic_embedding = response.data[0].embedding
    
    # Create the topic
    topic_id = db.create_topic(
        name=payload.name,
        topic_query=payload.topic_query,
        embedding=topic_embedding,
    )
    
    return {"topic_id": topic_id, "name": payload.name}


@app.get("/topic/check")
async def topic_check(topic_query: str) -> dict[str, Any]:
    """Check if topics exist for a given topic query string."""
    existing = db.get_topics_by_query(topic_query)
    return {
        "exists": len(existing) > 0,
        "topics": existing,
    }


@app.get("/topic/{topic_id}")
async def topic_get(topic_id: int) -> dict[str, Any]:
    """Get full topic details with papers and queries."""
    topic = db.get_topic_full(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return topic


@app.post("/topic/{topic_id}/papers")
async def topic_add_papers(topic_id: int, payload: TopicAddPapersRequest) -> dict[str, Any]:
    """Add papers to a topic pool."""
    # Verify topic exists
    topic = db.get_topic_by_id(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    added = db.add_papers_to_topic(
        topic_id=topic_id,
        paper_ids=payload.paper_ids,
        similarity_scores=payload.similarity_scores,
    )
    
    return {"added": added, "topic_id": topic_id}


@app.delete("/topic/{topic_id}/papers/{paper_id}")
async def topic_remove_paper(topic_id: int, paper_id: int) -> dict[str, Any]:
    """Remove a paper from a topic pool."""
    removed = db.remove_paper_from_topic(topic_id, paper_id)
    return {"removed": removed}


@app.delete("/topic/{topic_id}")
async def topic_delete(topic_id: int) -> dict[str, Any]:
    """Delete a topic and all its associated data."""
    deleted = db.delete_topic(topic_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Topic not found")
    return {"deleted": True}


@app.post("/topic/{topic_id}/query")
async def topic_query(topic_id: int, payload: TopicQueryRequest) -> dict[str, Any]:
    """Query across all papers in a topic pool using hierarchical RAG.
    
    1. Get all papers in the pool
    2. Query each paper with PaperQA
    3. Aggregate responses into final summary
    """
    topic_config = _get_topic_query_config()
    debug_mode = topic_config["debug_mode"]
    
    # Get topic with papers
    topic = db.get_topic_full(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    papers = topic.get("papers", [])
    if not papers:
        raise HTTPException(status_code=400, detail="Topic has no papers in pool")
    
    # Get LLM config
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    raw_model = resolve_model(base_url, api_key)
    llm_client = _get_async_openai_client(base_url, api_key)
    client = llm_client  # backward-compat alias for aggregation step
    
    # Get embed config once
    embed_base_url = endpoint_config["embedding_base_url"]
    embed_model_name = resolve_model(embed_base_url, api_key)
    embed_client = _get_async_openai_client(embed_base_url, api_key)
    
    # Get question embedding for debug output
    question_embedding = None
    if debug_mode:
        embed_response = await embed_client.embeddings.create(
            model=embed_model_name,
            input=payload.question,
        )
        question_embedding = embed_response.data[0].embedding
    
    # Query each paper using RAG
    paper_responses = []
    
    # Per-paper prompt template (for debug output)
    per_paper_prompt_template = f"""Question: {payload.question}

[RAG retrieves relevant chunks from this paper and prompts the LLM to answer based on those chunks]"""

    def detect_text_quality(text: str) -> dict:
        """Detect text quality issues like font encoding problems."""
        if not text:
            return {"quality": "empty", "corrupted_ratio": 1.0, "uni_sequences": 0, "readable_ratio": 0.0}
        
        import re
        
        # Count /uni escape sequences (indicates font encoding issues)
        uni_pattern = r'/uni[0-9a-fA-F]{8}'
        uni_matches = re.findall(uni_pattern, text)
        uni_count = len(uni_matches)
        
        # Detect gibberish patterns (decoded but still wrong characters)
        # Count sequences of unusual character combinations
        gibberish_patterns = [
            r'[A-Z]{5,}',  # Long uppercase sequences like "WMKQSMH"
            r'[0-9]{8,}',  # Long number sequences (leftover from /uni decoding)
            r'\\[a-zA-Z]',  # Escaped characters
            r'[^\x00-\x7F]+',  # Non-ASCII sequences (might be corrupted)
        ]
        gibberish_count = 0
        for pattern in gibberish_patterns:
            gibberish_count += len(re.findall(pattern, text))
        
        # Count readable English words (common words as sanity check)
        common_words = ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'are', 'was', 'were', 
                        'have', 'been', 'model', 'data', 'learning', 'training', 'results', 'paper']
        readable_words = sum(1 for word in common_words if word.lower() in text.lower())
        
        # Calculate quality metrics
        total_chars = len(text)
        uni_corrupted_chars = uni_count * 12
        
        # Estimate readability based on word presence and gibberish ratio
        words_in_text = len(text.split())
        gibberish_ratio = min(1.0, gibberish_count / max(1, words_in_text))
        readable_ratio = min(1.0, readable_words / max(1, min(10, words_in_text)))
        
        # Combined corruption estimate
        if uni_count > 0:
            corrupted_ratio = min(1.0, uni_corrupted_chars / max(1, total_chars))
        else:
            corrupted_ratio = gibberish_ratio * 0.5  # Gibberish is a softer indicator
        
        # Determine quality level
        if uni_count > total_chars / 20:  # More than 5% are /uni sequences
            quality = "poor"
        elif gibberish_ratio > 0.5 and readable_ratio < 0.3:
            quality = "poor"
        elif gibberish_ratio > 0.3 and readable_ratio < 0.5:
            quality = "degraded"  
        elif uni_count > 0 or gibberish_ratio > 0.2:
            quality = "minor_issues"
        else:
            quality = "good"
        
        return {
            "quality": quality,
            "corrupted_ratio": round(corrupted_ratio, 3),
            "uni_sequences": uni_count,
            "readable_ratio": round(readable_ratio, 3),
            "gibberish_ratio": round(gibberish_ratio, 3),
        }
    
    async def query_single_paper(paper: dict) -> dict[str, Any]:
        """Query a single paper and return response."""
        arxiv_id = paper.get("arxiv_id")
        pdf_path = paper.get("pdf_path")
        title = paper.get("title", "Unknown")
        
        try:
            result = await rag.rag_answer_async(
                question=payload.question,
                pdf_path=pdf_path,
                arxiv_id=arxiv_id,
                llm_client=llm_client,
                llm_model=raw_model,
                embed_client=embed_client,
                embed_model=embed_model_name,
                return_details=debug_mode,
            )
            
            if debug_mode and isinstance(result, dict):
                # Analyze text quality of retrieved chunks using RAW text (before cleaning)
                chunks = result.get("chunks", [])
                
                # Use raw text if available for quality detection
                raw_texts = []
                for c in chunks:
                    raw = c.get("text_raw_sample") or c.get("text", "")
                    raw_texts.append(raw)
                
                total_raw_text = " ".join(raw_texts)
                text_quality = detect_text_quality(total_raw_text)
                
                # Add quality info to each chunk
                chunks_with_quality = []
                for c in chunks:
                    raw_text = c.get("text_raw_sample") or c.get("text", "")
                    chunk_quality = detect_text_quality(raw_text)
                    chunks_with_quality.append({
                        **c,
                        "text_quality": chunk_quality["quality"],
                    })
                
                return {
                    "paper_id": paper["paper_id"],
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "similarity_score": paper.get("similarity_score"),
                    "response": result["answer"],
                    "chunks_retrieved": result.get("chunks_retrieved", 0),
                    "chunks": chunks_with_quality,
                    "overall_text_quality": text_quality,
                    "per_paper_prompt": per_paper_prompt_template,
                    "success": True,
                }
            else:
                return {
                    "paper_id": paper["paper_id"],
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "similarity_score": paper.get("similarity_score"),
                    "response": result,
                    "success": True,
                }
        except Exception as e:
            import traceback
            print(f"[TOPIC QUERY] Error querying paper {arxiv_id}: {e}")
            traceback.print_exc()
            return {
                "paper_id": paper["paper_id"],
                "arxiv_id": arxiv_id,
                "title": title,
                "similarity_score": paper.get("similarity_score"),
                "response": f"Error: {str(e)}",
                "success": False,
            }
    
    # Query papers in parallel (with semaphore to limit concurrency)
    semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent queries
    
    async def query_with_semaphore(paper: dict) -> dict[str, Any]:
        async with semaphore:
            return await query_single_paper(paper)
    
    paper_responses = await asyncio.gather(*[query_with_semaphore(p) for p in papers])
    
    # Filter successful responses
    successful_responses = [r for r in paper_responses if r["success"]]
    
    if not successful_responses:
        raise HTTPException(status_code=500, detail="All paper queries failed")
    
    # Aggregate responses into final summary
    aggregation_prompt = f"""You are summarizing answers from multiple research papers about a topic.

Question: {payload.question}

Here are the responses from individual papers:

"""
    for i, resp in enumerate(successful_responses, 1):
        aggregation_prompt += f"""
Paper {i}: {resp['title']}
Response: {resp['response']}

---
"""
    
    aggregation_prompt += """
Please synthesize these responses into a coherent, comprehensive answer. 
- Cite which papers support each point using [Paper N] format
- Note any disagreements or different perspectives between papers
- Prioritize findings that appear across multiple papers
- Be concise but thorough
"""
    
    # Generate final summary
    response = await client.chat.completions.create(
        model=raw_model,
        messages=[{"role": "user", "content": aggregation_prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    if not response.choices:
        raise HTTPException(status_code=502, detail="LLM returned empty response (choices=[])")
    final_answer = response.choices[0].message.content.strip()
    
    # Save to database
    db.add_topic_query(
        topic_id=topic_id,
        question=payload.question,
        answer=final_answer,
        paper_responses=paper_responses if debug_mode else None,
        model=raw_model,
    )
    
    # Save debug output if enabled
    if debug_mode:
        import os
        os.makedirs("storage/schemas", exist_ok=True)
        
        # Build comprehensive debug data
        papers_debug = []
        for paper in papers:
            paper_debug = {
                "paper_id": paper["paper_id"],
                "arxiv_id": paper.get("arxiv_id"),
                "title": paper.get("title"),
                "similarity_score": paper.get("similarity_score"),
            }
            # Get paper embedding from DB if available
            full_paper = db.get_paper_by_id(paper["paper_id"])
            if full_paper:
                emb = full_paper.get("embedding")
                if emb is not None and len(emb) > 0:
                    # Convert to list for JSON serialization (truncate for readability)
                    paper_debug["embedding_dim"] = len(emb)
                    paper_debug["embedding_sample"] = list(emb[:5])
            papers_debug.append(paper_debug)
        
        # Handle embeddings that might be numpy arrays
        topic_emb = topic.get("embedding")
        topic_emb_dim = len(topic_emb) if topic_emb is not None and hasattr(topic_emb, '__len__') else 0
        topic_emb_sample = list(topic_emb[:5]) if topic_emb is not None and len(topic_emb) >= 5 else None
        
        q_emb_dim = len(question_embedding) if question_embedding is not None else 0
        q_emb_sample = list(question_embedding[:5]) if question_embedding is not None else None
        
        # Calculate text quality summary across all papers
        quality_summary = {
            "total_papers": len(paper_responses),
            "good_quality": 0,
            "minor_issues": 0,
            "degraded": 0,
            "poor_quality": 0,
            "errors": 0,
        }
        for pr in paper_responses:
            if not pr.get("success"):
                quality_summary["errors"] += 1
            elif "overall_text_quality" in pr:
                q = pr["overall_text_quality"]["quality"]
                if q == "good":
                    quality_summary["good_quality"] += 1
                elif q == "minor_issues":
                    quality_summary["minor_issues"] += 1
                elif q == "degraded":
                    quality_summary["degraded"] += 1
                elif q == "poor":
                    quality_summary["poor_quality"] += 1
        
        debug_data = {
            "topic_id": topic_id,
            "topic_name": topic["name"],
            "topic_embedding_dim": topic_emb_dim,
            "topic_embedding_sample": topic_emb_sample,
            "question": payload.question,
            "question_embedding_dim": q_emb_dim,
            "question_embedding_sample": q_emb_sample,
            "papers_in_pool": papers_debug,
            "text_quality_summary": quality_summary,
            "paper_responses": paper_responses,
            "aggregation_prompt": aggregation_prompt,
            "final_answer": final_answer,
            "model": raw_model,
        }
        
        # Custom JSON encoder for numpy types
        import numpy as np
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return super().default(obj)
        
        # Save to topic-specific file
        safe_topic_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in topic["name"][:50])
        debug_file = f"storage/schemas/topic_query_{safe_topic_name}.json"
        with open(debug_file, "w") as f:
            json.dump(debug_data, f, indent=2, cls=NumpyEncoder)
        
        # Also save to generic file for backwards compatibility
        with open("storage/schemas/topic_query.json", "w") as f:
            json.dump(debug_data, f, indent=2, cls=NumpyEncoder)
    
    result = {
        "answer": final_answer,
        "papers_queried": len(papers),
        "successful_queries": len(successful_responses),
    }
    
    if debug_mode:
        result["paper_responses"] = paper_responses
    
    return result


# =============================================================================
# Background Semantic Scholar Metadata Fetch
# =============================================================================

PENDING_METADATA_PATH = pathlib.Path("storage/pending_metadata.json")


def _append_pending_metadata(arxiv_ids: list[str]) -> None:
    """Append arXiv IDs to pending_metadata.json for background Semantic Scholar fetch.
    Uses file locking for concurrency safety.
    """
    import fcntl
    import json
    
    PENDING_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Use a lock file alongside the json file
    lock_path = PENDING_METADATA_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if PENDING_METADATA_PATH.exists():
                existing = json.loads(PENDING_METADATA_PATH.read_text())
            else:
                existing = []
            # Add new IDs, avoiding duplicates
            existing_set = set(existing)
            for aid in arxiv_ids:
                if aid not in existing_set:
                    existing.append(aid)
                    existing_set.add(aid)
            PENDING_METADATA_PATH.write_text(json.dumps(existing, indent=2))
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _pop_pending_metadata() -> list[str]:
    """Atomically read and clear pending_metadata.json. Returns list of arXiv IDs."""
    import fcntl
    import json
    
    if not PENDING_METADATA_PATH.exists():
        return []
    
    lock_path = PENDING_METADATA_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if PENDING_METADATA_PATH.exists():
                ids = json.loads(PENDING_METADATA_PATH.read_text())
                PENDING_METADATA_PATH.write_text("[]")
                return ids
            return []
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


async def _fetch_metadata_for_paper(arxiv_id: str) -> dict[str, str]:
    """Fetch and cache Semantic Scholar references + similar papers for a single paper.
    Returns a status dict.
    """
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        return {"arxiv_id": arxiv_id, "status": "skipped", "reason": "not in DB"}
    
    apis_config = _get_external_apis_config()
    api_key = apis_config.get("semantic_scholar_api_key")
    status_parts = []
    
    # References
    if not db.get_references(paper["id"]):
        try:
            refs = await _get_semantic_scholar_references(arxiv_id, api_key)
            if refs:
                for ref in refs:
                    db.add_reference(
                        source_paper_id=paper["id"],
                        cited_title=ref["cited_title"],
                        cited_arxiv_id=ref.get("cited_arxiv_id"),
                        cited_authors=ref.get("cited_authors"),
                        cited_year=ref.get("cited_year"),
                    )
                status_parts.append(f"refs={len(refs)}")
            else:
                status_parts.append("refs=0")
        except Exception as e:
            status_parts.append(f"refs_err={e}")
    else:
        status_parts.append("refs=cached")
    
    # Similar papers
    if not db.get_cached_similar_papers(paper["id"]):
        try:
            similar = await _get_semantic_scholar_recommendations(arxiv_id, 10, api_key)
            if similar:
                for s in similar:
                    db.cache_similar_paper(
                        paper_id=paper["id"],
                        similar_arxiv_id=s.get("arxiv_id"),
                        similar_title=s["title"],
                        similarity_score=None,
                    )
                status_parts.append(f"similar={len(similar)}")
            else:
                status_parts.append("similar=0")
        except Exception as e:
            status_parts.append(f"similar_err={e}")
    else:
        status_parts.append("similar=cached")
    
    return {"arxiv_id": arxiv_id, "status": "ok", "detail": ", ".join(status_parts)}


async def _run_background_metadata_fetch() -> None:
    """Background task: fetch Semantic Scholar metadata for all pending papers.
    Logs to logs/backend_auxiliary.log. Respects rate limits with 1s delay between papers.
    """
    import logging
    
    # Set up auxiliary logger
    aux_logger = logging.getLogger("background_metadata")
    aux_logger.setLevel(logging.INFO)
    if not aux_logger.handlers:
        log_dir = pathlib.Path("logs")
        if not log_dir.exists():
            log_dir = pathlib.Path("storage")  # fallback inside container
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_dir / "backend_auxiliary.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        aux_logger.addHandler(handler)
    
    pending = _pop_pending_metadata()
    if not pending:
        return
    
    apis_config = _get_external_apis_config()
    if not apis_config.get("semantic_scholar_enabled", True):
        aux_logger.info(f"Semantic Scholar disabled, skipping {len(pending)} pending papers")
        return
    
    aux_logger.info(f"Starting background metadata fetch for {len(pending)} papers")
    
    for i, arxiv_id in enumerate(pending):
        try:
            result = await _fetch_metadata_for_paper(arxiv_id)
            aux_logger.info(f"  [{i+1}/{len(pending)}] {arxiv_id}: {result.get('detail', result.get('status'))}")
        except Exception as e:
            aux_logger.error(f"  [{i+1}/{len(pending)}] {arxiv_id}: error - {e}")
        
        # Rate limit: 1 request per second to avoid Semantic Scholar rate limits
        # (each paper makes 2 API calls: references + recommendations)
        await asyncio.sleep(1.0)
    
    aux_logger.info(f"Background metadata fetch complete for {len(pending)} papers")


# =============================================================================
# Settings/Config Endpoints
# =============================================================================

# Define the settings schema with metadata
SETTINGS_SCHEMA = {
    # LLM Settings
    "llm_base_url": {"category": "llm", "type": "string", "label": "LLM Base URL", "readonly": False},
    "embedding_base_url": {"category": "llm", "type": "string", "label": "Embedding Base URL", "readonly": False},
    # api_key is intentionally excluded (not shown in GUI per user requirement)
    
    # Ingestion Settings
    "skip_existing": {"category": "ingestion", "type": "boolean", "label": "Skip Existing Papers", "readonly": False},
    
    # Categorization Settings
    "branching_factor": {"category": "categorization", "type": "integer", "label": "Branching Factor", "readonly": False},
    
    # RAG Settings
    "chunk_chars": {"category": "rag", "type": "integer", "label": "Chunk Size (chars)", "readonly": False},
    "chunk_overlap": {"category": "rag", "type": "integer", "label": "Chunk Overlap (chars)", "readonly": False},
    "evidence_k": {"category": "rag", "type": "integer", "label": "Evidence K", "readonly": False},
    "evidence_summary_length": {"category": "rag", "type": "string", "label": "Evidence Summary Length", "readonly": False},
    
    # UI Settings
    "max_similar_papers": {"category": "ui", "type": "integer", "label": "Max Similar Papers", "readonly": False},
    "tree_auto_save_interval_ms": {"category": "ui", "type": "integer", "label": "Tree Auto-save Interval (ms)", "readonly": False},
    
    # Topic Query Settings
    "max_papers_per_batch": {"category": "topic_query", "type": "integer", "label": "Papers Per Batch", "readonly": False},
    "similarity_threshold": {"category": "topic_query", "type": "float", "label": "Similarity Threshold", "readonly": False},
    "chunks_per_paper": {"category": "topic_query", "type": "integer", "label": "Chunks Per Paper", "readonly": False},
    "topic_debug_mode": {"category": "topic_query", "type": "boolean", "label": "Debug Mode", "readonly": False},
    
    # Database Settings
    "database_name": {"category": "database", "type": "string", "label": "Active Database", "readonly": False},

    # Port Settings (read-only, require restart)
    "frontend_port": {"category": "ports", "type": "integer", "label": "Frontend Port", "readonly": True},
    "backend_port": {"category": "ports", "type": "integer", "label": "Backend Port", "readonly": True},
    "database_port": {"category": "ports", "type": "integer", "label": "Database Port", "readonly": True},
}


def _load_config_yaml() -> dict[str, Any]:
    """Load config.yaml as defaults."""
    import yaml
    config_paths = [
        pathlib.Path("config/config.yaml"),
        pathlib.Path("../config/config.yaml"),
        pathlib.Path("../../config/config.yaml"),
    ]
    for path in config_paths:
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


def _get_effective_config() -> dict[str, Any]:
    """Get effective config: DB settings override config.yaml defaults."""
    # Load defaults from config.yaml
    yaml_config = _load_config_yaml()
    
    # Flatten nested config for easy access
    # NOTE: llm_base_url and embedding_base_url live under `endpoints:` in config.yaml
    _endpoints = yaml_config.get("endpoints", {})
    defaults = {
        "llm_base_url": _endpoints.get("llm_base_url", "http://localhost:8001/v1"),
        "embedding_base_url": _endpoints.get("embedding_base_url", "http://localhost:8004/v1"),
        "skip_existing": yaml_config.get("ingestion", {}).get("skip_existing", False),
        "branching_factor": yaml_config.get("classification", {}).get("branching_factor", 5),
        "chunk_chars": yaml_config.get("rag", yaml_config.get("paperqa", {})).get("chunk_chars", 3000),
        "chunk_overlap": yaml_config.get("rag", yaml_config.get("paperqa", {})).get("chunk_overlap", 100),
        "evidence_k": yaml_config.get("rag", yaml_config.get("paperqa", {})).get("evidence_k", 10),
        "evidence_summary_length": yaml_config.get("rag", yaml_config.get("paperqa", {})).get("evidence_summary_length", "about 100 words"),
        "max_similar_papers": yaml_config.get("ui", {}).get("max_similar_papers", 5),
        "tree_auto_save_interval_ms": yaml_config.get("ui", {}).get("tree_auto_save_interval_ms", 30000),
        "max_papers_per_batch": yaml_config.get("topic_query", {}).get("max_papers_per_batch", 10),
        "similarity_threshold": yaml_config.get("topic_query", {}).get("similarity_threshold", 0.5),
        "chunks_per_paper": yaml_config.get("topic_query", {}).get("chunks_per_paper", 5),
        "topic_debug_mode": yaml_config.get("topic_query", {}).get("debug_mode", False),
        "database_name": db.get_active_database(),
        "frontend_port": yaml_config.get("frontend_port", 3000),
        "backend_port": yaml_config.get("backend_port", 3100),
        "database_port": yaml_config.get("database_port", 5432),
    }
    
    # Load DB overrides
    db_settings = db.get_all_settings()
    
    # Merge: DB takes precedence
    effective = {}
    for key, schema in SETTINGS_SCHEMA.items():
        if key in db_settings:
            # Parse value based on type
            raw_value = db_settings[key]["value"]
            if schema["type"] == "boolean":
                effective[key] = raw_value.lower() in ("true", "1", "yes")
            elif schema["type"] == "integer":
                effective[key] = int(raw_value)
            elif schema["type"] == "float":
                effective[key] = float(raw_value)
            else:
                effective[key] = raw_value
        else:
            effective[key] = defaults.get(key)
    
    return effective


def _validate_setting(key: str, value: Any) -> tuple[bool, Optional[str]]:
    """Validate a setting value. Returns (is_valid, warning_message)."""
    schema = SETTINGS_SCHEMA.get(key)
    if not schema:
        return False, f"Unknown setting: {key}"
    
    if schema["readonly"]:
        return False, f"Setting '{key}' is read-only"
    
    # Type validation
    try:
        if schema["type"] == "boolean":
            if not isinstance(value, bool):
                value = str(value).lower() in ("true", "1", "yes")
        elif schema["type"] == "integer":
            int(value)
        elif schema["type"] == "float":
            float(value)
    except (ValueError, TypeError):
        return False, f"Invalid type for {key}: expected {schema['type']}"
    
    # URL validation for endpoints
    if key in ("llm_base_url", "embedding_base_url"):
        if not str(value).startswith(("http://", "https://")):
            return True, f"Warning: {key} should start with http:// or https://"
    
    return True, None


class ConfigUpdateRequest(BaseModel):
    settings: dict[str, Any]


@app.get("/config")
def get_config() -> dict[str, Any]:
    """Get current configuration with schema metadata."""
    effective = _get_effective_config()
    db_settings = db.get_all_settings()
    
    # Build response with schema info
    result = {
        "settings": {},
        "categories": {
            "llm": {"label": "LLM Endpoints", "description": "Language model and embedding service URLs"},
            "ingestion": {"label": "Ingestion", "description": "Paper ingestion behavior"},
            "classification": {"label": "Classification", "description": "Tree clustering and auto-rebuild settings"},
            "rag": {"label": "RAG", "description": "Document chunking and retrieval settings"},
            "ui": {"label": "UI", "description": "User interface behavior settings"},
            "topic_query": {"label": "Topic Query", "description": "Multi-paper RAG query settings"},
            "database": {"label": "Database", "description": "Active database (switch between production and test)"},
            "ports": {"label": "Ports", "description": "Service ports (read-only, change in server config)"},
        }
    }
    
    for key, schema in SETTINGS_SCHEMA.items():
        is_overridden = key in db_settings
        result["settings"][key] = {
            "value": effective.get(key),
            "category": schema["category"],
            "type": schema["type"],
            "label": schema["label"],
            "readonly": schema["readonly"],
            "is_overridden": is_overridden,
            "updated_at": db_settings[key]["updated_at"].isoformat() if is_overridden else None,
        }
    
    return result


@app.post("/config")
def update_config(payload: ConfigUpdateRequest) -> dict[str, Any]:
    """Update configuration settings."""
    warnings = []
    errors = []
    updated = []
    
    for key, value in payload.settings.items():
        # Special handling: database_name switches the active DB
        if key == "database_name":
            try:
                prev = db.switch_database(str(value))
                updated.append(key)
                # Verify connectivity
                with db.get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT current_database()")
                continue
            except Exception as e:
                db.switch_database(prev)
                errors.append(f"Cannot switch to database '{value}': {e}")
                continue

        is_valid, message = _validate_setting(key, value)
        
        if not is_valid:
            errors.append(message)
            continue
        
        if message:  # Warning
            warnings.append(message)
        
        # Convert to string for storage
        schema = SETTINGS_SCHEMA[key]
        str_value = str(value).lower() if schema["type"] == "boolean" else str(value)
        
        db.set_setting(key, str_value, schema["category"])
        updated.append(key)
    
    return {
        "success": len(errors) == 0,
        "updated": updated,
        "warnings": warnings,
        "errors": errors,
        "config": _get_effective_config(),
    }


@app.post("/config/reset")
def reset_config() -> dict[str, Any]:
    """Reset all settings to defaults from config.yaml."""
    deleted_count = db.clear_all_settings()
    
    return {
        "success": True,
        "message": f"Reset {deleted_count} settings to defaults",
        "config": _get_effective_config(),
    }
