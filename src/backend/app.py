from __future__ import annotations

import asyncio
import hashlib
import os
import pathlib
import re
from functools import lru_cache
from typing import Any, Optional

import arxiv
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

app = FastAPI(title="paper-curator-backend")


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


def _require_identifier(arxiv_id: Optional[str], url: Optional[str]) -> str:
    """Extract arXiv ID from provided arxiv_id or URL."""
    if arxiv_id:
        return arxiv_id
    if url:
        # Parse arXiv ID from URL (e.g., https://arxiv.org/abs/1706.03762)
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


@lru_cache(maxsize=4)
def _load_config_cached(config_mtime: float) -> dict[str, Any]:
    config_path = pathlib.Path("config/paperqa.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = (
        "openai_api_base",
        "openai_api_base3",
        "openai_api_key",
    )
    for key in required:
        if key not in config:
            raise HTTPException(status_code=500, detail=f"Missing '{key}' in config.")
    return config


def _load_config() -> dict[str, Any]:
    config_path = pathlib.Path("config/paperqa.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=500, detail="Config file not found: config/paperqa.yaml")
    return _load_config_cached(config_path.stat().st_mtime)


def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=3)
def _resolve_model(base_url: str, api_key: str) -> str:
    client = _get_openai_client(base_url, api_key)
    try:
        models = client.models.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Model list failed: {exc}") from exc
    model_ids = sorted([model.id for model in models.data if getattr(model, "id", None)])
    if not model_ids:
        raise HTTPException(status_code=502, detail="No models returned by OpenAI-compatible endpoint.")
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
        config = _load_config()
        reader_config = {
            "chunk_chars": int(config.get("paperqa_chunk_chars", 5000)),
            "overlap": int(config.get("paperqa_chunk_overlap", 250)),
        }

        def _build_settings(use_doc_details: bool) -> Settings:
            parsing_settings = ParsingSettings(
                reader_config=reader_config,
                use_doc_details=use_doc_details,
                multimodal=MultimodalOptions.OFF,
            )
            answer_settings = AnswerSettings(
                evidence_k=int(config.get("paperqa_evidence_k", 10)),
                evidence_summary_length=str(
                    config.get("paperqa_evidence_summary_length", "about 100 words")
                ),
                evidence_skip_summary=bool(
                    config.get("paperqa_evidence_skip_summary", False)
                ),
                evidence_relevance_score_cutoff=float(
                    config.get("paperqa_evidence_relevance_score_cutoff", 1)
                ),
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

        use_doc_details = bool(config.get("paperqa_use_doc_details", True))
        settings = _build_settings(use_doc_details)
        try:
            await docs.aadd(str(content_path), settings=settings)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if use_doc_details and "semanticscholar" in message.lower() and "429" in message:
                print("WARNING: Semantic Scholar rate limit hit; retrying without doc details.")
                settings = _build_settings(False)
                docs = Docs()
                await docs.aadd(str(content_path), settings=settings)
            else:
                raise
        return await docs.aquery(question, settings=settings)

    try:
        result = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"PaperQA request failed: {exc}") from exc

    if hasattr(result, "answer"):
        return str(result.answer)
    return str(result)


def _paperqa_extract_pdf(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Extract text from PDF using PaperQA2's native parser."""
    config = _load_config()
    reader_config = {
        "chunk_chars": int(config.get("paperqa_chunk_chars", 5000)),
        "overlap": int(config.get("paperqa_chunk_overlap", 250)),
    }
    parsing_settings = ParsingSettings(
        reader_config=reader_config,
        use_doc_details=bool(config.get("paperqa_use_doc_details", True)),
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
        return {
            "text": text,
            "parser": "paperqa",
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"PaperQA2 parse failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/arxiv/resolve")
def arxiv_resolve(payload: ArxivResolveRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    try:
        results = list(client.results(search))
    except arxiv.HTTPError:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
    if not results:
        raise HTTPException(status_code=404, detail="No arXiv result found.")
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
    latex_path = None
    source_error = None
    try:
        latex_path = result.download_source(dirpath=output_dir)
    except Exception as exc:  # noqa: BLE001
        source_error = str(exc)

    return {
        "arxiv_id": result.get_short_id(),
        "pdf_path": pdf_path,
        "latex_path": latex_path,
        "latex_error": source_error,
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
    config = _load_config()
    base_url = config["openai_api_base"]
    embed_base_url = config["openai_api_base3"]
    api_key = config["openai_api_key"]
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
    config = _load_config()
    base_url = config["openai_api_base3"]
    api_key = config["openai_api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_openai_client(base_url, api_key)
    try:
        response = client.embeddings.create(model=model, input=payload.text)
        vector = response.data[0].embedding
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc
    return {"embedding": vector, "model": model}


@app.post("/qa")
def qa(payload: QaRequest) -> dict[str, Any]:
    config = _load_config()
    base_url = config["openai_api_base"]
    embed_base_url = config["openai_api_base3"]
    api_key = config["openai_api_key"]
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
