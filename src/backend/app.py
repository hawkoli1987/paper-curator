from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import re
import uuid
from functools import lru_cache
from typing import Any, Optional

import arxiv
import httpx
import yaml
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from paperqa import Docs
from paperqa.readers import read_doc
from paperqa.settings import (
    AnswerSettings,
    MultimodalOptions,
    ParsingSettings,
    Settings,
    make_default_litellm_model_list_settings,
)
from paperqa.types import Doc
from pydantic import BaseModel, Field

import db

app = FastAPI(title="paper-curator-backend")


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


class EmbedRequest(BaseModel):
    text: str = Field(description="Text to embed")


class QaRequest(BaseModel):
    context: Optional[str] = Field(default=None, description="Context text to answer from")
    question: str = Field(description="Question to answer")
    pdf_path: Optional[str] = Field(default=None, description="Local PDF file path")


class ClassifyRequest(BaseModel):
    title: str = Field(description="Paper title")
    abstract: str = Field(description="Paper abstract or summary")
    existing_categories: list[str] = Field(default=[], description="Existing categories in the tree")


class SavePaperRequest(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: Optional[str] = None
    summary: Optional[str] = None
    pdf_path: Optional[str] = None
    latex_path: Optional[str] = None
    pdf_url: Optional[str] = None
    published_at: Optional[str] = None
    category: str


class TreeNodeRequest(BaseModel):
    node_id: str
    name: str
    node_type: str  # 'category' or 'paper'
    parent_id: Optional[str] = None
    paper_id: Optional[int] = None
    position: int = 0


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


# =============================================================================
# Config Loading
# =============================================================================

@lru_cache(maxsize=4)
def _load_config_cached(config_mtime: float) -> dict[str, Any]:
    config_path = pathlib.Path("config/paperqa.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config


def _load_config() -> dict[str, Any]:
    config_path = pathlib.Path("config/paperqa.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=500, detail="Config file not found: config/paperqa.yaml")
    return _load_config_cached(config_path.stat().st_mtime)


def _get_endpoint_config() -> dict[str, str]:
    """Get endpoint configuration."""
    config = _load_config()
    endpoints = config.get("endpoints", {})
    # Support both new and legacy config format
    return {
        "llm_base_url": endpoints.get("llm_base_url", config.get("openai_api_base", "")),
        "embedding_base_url": endpoints.get("embedding_base_url", config.get("openai_api_base3", "")),
        "api_key": endpoints.get("api_key", config.get("openai_api_key", "local-key")),
    }


def _get_paperqa_config() -> dict[str, Any]:
    """Get PaperQA2 configuration."""
    config = _load_config()
    pqa = config.get("paperqa", {})
    return {
        "chunk_chars": int(pqa.get("chunk_chars", config.get("paperqa_chunk_chars", 5000))),
        "chunk_overlap": int(pqa.get("chunk_overlap", config.get("paperqa_chunk_overlap", 250))),
        "use_doc_details": bool(pqa.get("use_doc_details", config.get("paperqa_use_doc_details", True))),
        "evidence_k": int(pqa.get("evidence_k", config.get("paperqa_evidence_k", 10))),
        "evidence_summary_length": str(pqa.get("evidence_summary_length", config.get("paperqa_evidence_summary_length", "about 100 words"))),
        "evidence_skip_summary": bool(pqa.get("evidence_skip_summary", config.get("paperqa_evidence_skip_summary", False))),
        "evidence_relevance_score_cutoff": float(pqa.get("evidence_relevance_score_cutoff", config.get("paperqa_evidence_relevance_score_cutoff", 1))),
    }


def _get_ui_config() -> dict[str, Any]:
    """Get UI configuration."""
    config = _load_config()
    ui = config.get("ui", {})
    return {
        "hover_debounce_ms": int(ui.get("hover_debounce_ms", 500)),
        "max_similar_papers": int(ui.get("max_similar_papers", 5)),
        "tree_auto_save_interval_ms": int(ui.get("tree_auto_save_interval_ms", 30000)),
    }


def _get_external_apis_config() -> dict[str, Any]:
    """Get external APIs configuration."""
    config = _load_config()
    apis = config.get("external_apis", {})
    return {
        "papers_with_code_enabled": bool(apis.get("papers_with_code_enabled", True)),
        "github_search_enabled": bool(apis.get("github_search_enabled", True)),
        "semantic_scholar_enabled": bool(apis.get("semantic_scholar_enabled", True)),
        "github_token": apis.get("github_token"),
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _require_identifier(arxiv_id: Optional[str], url: Optional[str]) -> str:
    """Extract arXiv ID from provided arxiv_id or URL."""
    if arxiv_id:
        return arxiv_id
    if url:
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/\s?]+)", url)
        if match:
            return match.group(1).replace(".pdf", "")
        return url
    raise HTTPException(status_code=400, detail="Provide arxiv_id or url.")


def _load_prompt() -> tuple[str, str, str]:
    prompt_path = pathlib.Path("prompts/paper_summary.md")
    if not prompt_path.exists():
        raise HTTPException(status_code=500, detail="Prompt file not found.")
    content = prompt_path.read_text(encoding="utf-8").strip()
    first_line, _, rest = content.partition("\n")
    if not first_line.startswith("ID: "):
        raise HTTPException(status_code=500, detail="Prompt ID missing.")
    prompt_id = first_line.replace("ID: ", "").strip()
    prompt_body = rest.strip()
    prompt_hash = hashlib.sha256(prompt_body.encode("utf-8")).hexdigest()
    return prompt_id, prompt_hash, prompt_body


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=3)
def _resolve_model(base_url: str, api_key: str) -> str:
    client = _get_openai_client(base_url, api_key)
    models = client.models.list()
    model_ids = sorted([model.id for model in models.data if getattr(model, "id", None)])
    assert model_ids, "No models returned by OpenAI-compatible endpoint."
    return model_ids[0]


def _paperqa_answer(
    text: Optional[str],
    pdf_path: Optional[str],
    question: str,
    llm_base_url: str,
    embed_base_url: str,
    api_key: str,
    llm_model: str,
    embed_model: str,
) -> str:
    assert text or pdf_path, "Provide text or pdf_path."
    os.environ["OPENAI_API_BASE"] = llm_base_url
    os.environ["OPENAI_API_KEY"] = api_key

    async def _run() -> Any:
        docs = Docs()
        content_path: Optional[pathlib.Path] = None
        if pdf_path:
            content_path = pathlib.Path(pdf_path)
        elif text is not None:
            inputs_dir = pathlib.Path("storage/inputs")
            inputs_dir.mkdir(parents=True, exist_ok=True)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            content_path = inputs_dir / f"paperqa_{text_hash}.txt"
            content_path.write_text(text, encoding="utf-8")
        if content_path is None or not content_path.exists():
            raise HTTPException(status_code=404, detail="Content path not found.")
        
        llm_config = make_default_litellm_model_list_settings(llm_model)
        llm_config["model_list"][0]["litellm_params"].update(
            {"api_base": llm_base_url, "api_key": api_key}
        )
        summary_llm_config = make_default_litellm_model_list_settings(llm_model)
        summary_llm_config["model_list"][0]["litellm_params"].update(
            {"api_base": llm_base_url, "api_key": api_key}
        )
        
        pqa_config = _get_paperqa_config()
        reader_config = {
            "chunk_chars": pqa_config["chunk_chars"],
            "overlap": pqa_config["chunk_overlap"],
        }

        def _build_settings(use_doc_details: bool) -> Settings:
            parsing_settings = ParsingSettings(
                reader_config=reader_config,
                use_doc_details=use_doc_details,
                multimodal=MultimodalOptions.OFF,
            )
            answer_settings = AnswerSettings(
                evidence_k=pqa_config["evidence_k"],
                evidence_summary_length=pqa_config["evidence_summary_length"],
                evidence_skip_summary=pqa_config["evidence_skip_summary"],
                evidence_relevance_score_cutoff=pqa_config["evidence_relevance_score_cutoff"],
            )
            return Settings(
                llm=llm_model,
                llm_config=llm_config,
                summary_llm=llm_model,
                summary_llm_config=summary_llm_config,
                embedding=embed_model,
                embedding_config={
                    "kwargs": {
                        "api_base": embed_base_url,
                        "api_key": api_key,
                        "encoding_format": "float",
                    },
                },
                parsing=parsing_settings,
                answer=answer_settings,
            )

        settings = _build_settings(pqa_config["use_doc_details"])
        await docs.aadd(str(content_path), settings=settings)
        return await docs.aquery(question, settings=settings)

    result = asyncio.run(_run())
    return str(result.answer) if hasattr(result, "answer") else str(result)


def _paperqa_extract_pdf(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Extract text from PDF using PaperQA2's native parser."""
    pqa_config = _get_paperqa_config()
    reader_config = {
        "chunk_chars": pqa_config["chunk_chars"],
        "overlap": pqa_config["chunk_overlap"],
    }
    parsing_settings = ParsingSettings(
        reader_config=reader_config,
        use_doc_details=pqa_config["use_doc_details"],
        multimodal=MultimodalOptions.OFF,
    )
    settings = Settings(parsing=parsing_settings)

    async def _run() -> dict[str, Any]:
        doc = Doc(docname=pdf_path.stem, dockey=pdf_path.stem, citation="Local PDF")
        parsed_text = await read_doc(
            str(pdf_path),
            doc,
            parsed_text_only=True,
            parse_pdf=settings.parsing.parse_pdf,
            **settings.parsing.reader_config,
        )
        text = parsed_text.reduce_content()
        return {"text": text, "parser": "paperqa"}

    return asyncio.run(_run())


# =============================================================================
# Core Endpoints
# =============================================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def get_config() -> dict[str, Any]:
    """Return UI configuration for the frontend."""
    return _get_ui_config()


@app.post("/arxiv/resolve")
def arxiv_resolve(payload: ArxivResolveRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
    assert results, f"No arXiv result found for: {identifier}"
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
def arxiv_download(payload: ArxivDownloadRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    output_dir = payload.output_dir or os.getenv("ARXIV_DOWNLOAD_DIR", "storage/downloads")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
    if not results:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
    result = results[0]

    pdf_path = result.download_pdf(dirpath=output_dir)
    latex_path = result.download_source(dirpath=output_dir) if result.source_url() else None

    return {
        "arxiv_id": result.get_short_id(),
        "pdf_path": pdf_path,
        "latex_path": latex_path,
    }


@app.post("/pdf/extract")
def pdf_extract(payload: PdfExtractRequest) -> dict[str, Any]:
    """Extract text from PDF using PaperQA2's native parser."""
    pdf_path = pathlib.Path(payload.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    return _paperqa_extract_pdf(pdf_path)


@app.post("/summarize")
def summarize(payload: SummarizeRequest) -> dict[str, Any]:
    prompt_id, prompt_hash, prompt_body = _load_prompt()
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = f"openai/{_resolve_model(base_url, api_key)}"
    embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    summary = _paperqa_answer(
        payload.text,
        payload.pdf_path,
        prompt_body,
        base_url,
        embed_base_url,
        api_key,
        model,
        embed_model,
    )
    return {
        "summary": summary.strip(),
        "prompt_id": prompt_id,
        "prompt_hash": prompt_hash,
        "model": model,
    }


@app.post("/embed")
def embed(payload: EmbedRequest) -> dict[str, Any]:
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_openai_client(base_url, api_key)
    response = client.embeddings.create(model=model, input=payload.text)
    vector = response.data[0].embedding
    return {"embedding": vector, "model": model}


@app.post("/qa")
def qa(payload: QaRequest) -> dict[str, Any]:
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    embed_base_url = endpoint_config["embedding_base_url"]
    api_key = endpoint_config["api_key"]
    model = f"openai/{_resolve_model(base_url, api_key)}"
    embed_model = f"openai/{_resolve_model(embed_base_url, api_key)}"
    answer = _paperqa_answer(
        payload.context,
        payload.pdf_path,
        payload.question,
        base_url,
        embed_base_url,
        api_key,
        model,
        embed_model,
    )
    return {"answer": answer}


CLASSIFY_PROMPT = """You are a research paper classifier. Given a paper's title and abstract, determine the most appropriate category for it.

Primary AI research categories:
- Model Architecture (transformer variants, attention mechanisms, novel neural network designs)
- Training Efficiency (optimization, distributed training, memory efficiency)
- Inference Optimization (quantization, pruning, distillation, serving)
- Reinforcement Learning (RL algorithms, RLHF, reward modeling)
- Vision (image classification, object detection, segmentation, generation)
- Speech (ASR, TTS, audio processing)
- Natural Language Processing (text understanding, generation, translation)
- Multimodal (vision-language, audio-visual, cross-modal)
- Datasets & Benchmarks (new datasets, evaluation frameworks)
- Applications (specific use cases, deployments)

Existing categories in the tree: {existing_categories}

Instructions:
1. If the paper fits well into an existing category, use that exact category name.
2. If it doesn't fit existing categories, suggest the most appropriate category from the primary list.
3. If it's a very specific sub-topic and there are already many papers in a category, suggest a more specific sub-category.

Paper Title: {title}

Paper Abstract: {abstract}

Respond with ONLY the category name, nothing else. Do not include explanations or punctuation."""


@app.post("/classify")
def classify(payload: ClassifyRequest) -> dict[str, Any]:
    """Classify a paper into a category using LLM."""
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_openai_client(base_url, api_key)

    existing_str = ", ".join(payload.existing_categories) if payload.existing_categories else "None yet"
    prompt = CLASSIFY_PROMPT.format(
        existing_categories=existing_str,
        title=payload.title,
        abstract=payload.abstract[:2000],
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.1,
    )
    category = response.choices[0].message.content.strip()
    return {"category": category, "model": model}


# =============================================================================
# Database & Tree Endpoints
# =============================================================================

@app.post("/papers/save")
def save_paper(payload: SavePaperRequest) -> dict[str, Any]:
    """Save a paper to the database and add it to the tree."""
    # Create paper in DB
    paper_id = db.create_paper(
        arxiv_id=payload.arxiv_id,
        title=payload.title,
        authors=payload.authors,
        abstract=payload.abstract,
        summary=payload.summary,
        pdf_path=payload.pdf_path,
        latex_path=payload.latex_path,
        pdf_url=payload.pdf_url,
        published_at=payload.published_at,
    )
    
    # Check if category exists, if not create it
    tree = db.get_tree()
    category_exists = any(n["name"] == payload.category and n["node_type"] == "category" for n in tree)
    
    if not category_exists:
        category_node_id = f"cat_{uuid.uuid4().hex[:8]}"
        db.add_tree_node(
            node_id=category_node_id,
            name=payload.category,
            node_type="category",
            parent_id="root",
        )
    else:
        category_node_id = next(n["node_id"] for n in tree if n["name"] == payload.category and n["node_type"] == "category")
    
    # Add paper node
    paper_node_id = f"paper_{payload.arxiv_id.replace('.', '_')}"
    short_name = payload.title[:40] + "..." if len(payload.title) > 40 else payload.title
    db.add_tree_node(
        node_id=paper_node_id,
        name=short_name,
        node_type="paper",
        parent_id=category_node_id,
        paper_id=paper_id,
    )
    
    return {"paper_id": paper_id, "node_id": paper_node_id}


@app.get("/tree")
def get_tree() -> dict[str, Any]:
    """Get the full tree structure."""
    nodes = db.get_tree()
    
    # Build tree structure
    def build_tree(parent_id: Optional[str]) -> list[dict[str, Any]]:
        children = []
        for node in nodes:
            if node["parent_id"] == parent_id:
                child = {
                    "node_id": node["node_id"],
                    "name": node["name"],
                    "node_type": node["node_type"],
                }
                if node["node_type"] == "paper" and node["paper_id"]:
                    child["attributes"] = {
                        "arxivId": node.get("arxiv_id"),
                        "title": node.get("paper_title"),
                        "authors": node.get("authors") or [],
                        "summary": node.get("summary"),
                    }
                grandchildren = build_tree(node["node_id"])
                if grandchildren:
                    child["children"] = grandchildren
                children.append(child)
        return children
    
    # Find root and build from there
    root = next((n for n in nodes if n["node_type"] == "root"), None)
    if not root:
        return {"name": "AI Papers", "children": []}
    
    return {
        "name": root["name"],
        "children": build_tree(root["node_id"]),
    }


@app.post("/tree/node")
def add_tree_node(payload: TreeNodeRequest) -> dict[str, str]:
    """Add a node to the tree."""
    db.add_tree_node(
        node_id=payload.node_id,
        name=payload.name,
        node_type=payload.node_type,
        parent_id=payload.parent_id,
        paper_id=payload.paper_id,
        position=payload.position,
    )
    return {"status": "ok"}


@app.delete("/tree/node/{node_id}")
def delete_tree_node(node_id: str) -> dict[str, str]:
    """Delete a node from the tree."""
    db.delete_tree_node(node_id)
    return {"status": "ok"}


# =============================================================================
# Repository Search (Feature 4)
# =============================================================================

async def _search_papers_with_code(title: str) -> list[dict[str, Any]]:
    """Search Papers With Code for repositories."""
    async with httpx.AsyncClient(timeout=10.0) as client:
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
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Clean title for search
        clean_title = re.sub(r'[^\w\s]', '', title)[:50]
        search_url = "https://api.github.com/search/repositories"
        response = await client.get(
            search_url,
            params={"q": clean_title, "sort": "stars", "order": "desc", "per_page": 5},
            headers=headers,
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

async def _get_semantic_scholar_references(arxiv_id: str) -> list[dict[str, Any]]:
    """Get references from Semantic Scholar API."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        # First get the paper ID
        paper_url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}"
        params = {"fields": "references.title,references.authors,references.year,references.externalIds"}
        response = await client.get(paper_url, params=params)
        
        if response.status_code != 200:
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
        references = await _get_semantic_scholar_references(payload.arxiv_id)
    
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


EXPLAIN_REFERENCE_PROMPT = """You are a research paper analyst. Explain how a cited paper relates to the source paper.

Source paper: {source_title}
Cited paper: {cited_title}
{context_section}

Provide a concise 2-3 sentence explanation of:
1. What the cited paper is about
2. How it relates to or contributes to the source paper

Be specific and technical but accessible."""


@app.post("/references/explain")
def explain_reference(payload: ExplainReferenceRequest) -> dict[str, Any]:
    """Generate an explanation for a reference using LLM."""
    # Check cache
    refs = db.get_references(0)  # We need to query by ref ID
    ref = next((r for r in refs if r.get("id") == payload.reference_id), None) if refs else None
    if ref and ref.get("explanation"):
        return {"explanation": ref["explanation"], "from_cache": True}
    
    endpoint_config = _get_endpoint_config()
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_openai_client(base_url, api_key)
    
    context_section = ""
    if payload.citation_context:
        context_section = f"\nCitation context: \"{payload.citation_context}\""
    
    prompt = EXPLAIN_REFERENCE_PROMPT.format(
        source_title=payload.source_paper_title,
        cited_title=payload.cited_title,
        context_section=context_section,
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    explanation = response.choices[0].message.content.strip()
    
    # Cache the explanation
    if payload.reference_id:
        db.update_reference_explanation(payload.reference_id, explanation)
    
    return {"explanation": explanation, "from_cache": False}


# =============================================================================
# Similar Papers (Feature 6)
# =============================================================================

@app.post("/papers/similar")
def find_similar_papers(payload: SimilarPapersRequest) -> dict[str, Any]:
    """Find similar papers using embedding similarity."""
    paper = db.get_paper_by_arxiv_id(payload.arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found in database")
    
    # Check cache
    cached = db.get_cached_similar_papers(paper["id"])
    if cached:
        return {"similar_papers": cached, "from_cache": True}
    
    # Get or compute embedding
    embedding = paper.get("embedding")
    if embedding is None:
        # Compute embedding from abstract + title
        text = f"{paper['title']}\n\n{paper.get('abstract', '')}"
        endpoint_config = _get_endpoint_config()
        base_url = endpoint_config["embedding_base_url"]
        api_key = endpoint_config["api_key"]
        model = _resolve_model(base_url, api_key)
        client = _get_openai_client(base_url, api_key)
        
        response = client.embeddings.create(model=model, input=text)
        embedding = response.data[0].embedding
        
        # Save embedding
        db.update_paper_embedding(paper["id"], embedding)
    
    # Find similar papers
    ui_config = _get_ui_config()
    similar = db.find_similar_papers(
        embedding=embedding,
        limit=ui_config["max_similar_papers"],
        exclude_id=paper["id"],
    )
    
    # Cache results
    for s in similar:
        db.cache_similar_paper(
            paper_id=paper["id"],
            similar_arxiv_id=s.get("arxiv_id"),
            similar_title=s["title"],
            similarity_score=s.get("similarity"),
        )
    
    return {"similar_papers": similar, "from_cache": False}
