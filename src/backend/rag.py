"""Custom RAG (Retrieval-Augmented Generation) implementation.

Replaces PaperQA2 with a simpler, direct implementation:
- PDF text extraction via pymupdf (fitz)
- Character-based chunking with overlap
- Embedding via AsyncOpenAI API
- Chunk storage in paper_chunks table (pgvector)
- Retrieval via cosine similarity
- LLM answer generation via AsyncOpenAI
"""
from __future__ import annotations

import pathlib
import re
from typing import Any, Optional

import fitz  # pymupdf
import numpy as np
from fastapi import HTTPException
from openai import AsyncOpenAI

import db


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: pathlib.Path) -> str:
    """Extract full text from a PDF using pymupdf.

    Returns concatenated text from all pages.
    """
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    text = "\n".join(pages)
    # PostgreSQL text columns cannot store NUL (0x00) characters
    return text.replace("\x00", "")


# ---------------------------------------------------------------------------
# Text cleaning (ported from paperqa_helpers.py)
# ---------------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """Clean text extracted from PDFs, decoding font escape sequences."""
    if not text:
        return text

    def decode_uni_sequence(match: re.Match) -> str:
        hex_str = match.group(1)
        try:
            code_point = int(hex_str, 16)
            if 0x20 <= code_point <= 0x10FFFF:
                return chr(code_point)
            return ""
        except (ValueError, OverflowError):
            return ""

    # /uniXXXXXXXX sequences
    cleaned = re.sub(r"/uni([0-9a-fA-F]{8})", decode_uni_sequence, text)

    # /uXXXX sequences
    def decode_u_sequence(match: re.Match) -> str:
        hex_str = match.group(1)
        try:
            code_point = int(hex_str, 16)
            if 0x20 <= code_point <= 0x10FFFF:
                return chr(code_point)
            return ""
        except (ValueError, OverflowError):
            return ""

    cleaned = re.sub(r"/u([0-9a-fA-F]{4})", decode_u_sequence, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_chars: int = 5000,
    overlap: int = 250,
) -> list[dict[str, Any]]:
    """Split *text* into overlapping character-based chunks.

    Returns a list of dicts with keys: text, chunk_index, char_start, char_end.
    """
    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(
            {
                "text": text[start:end],
                "chunk_index": idx,
                "char_start": start,
                "char_end": end,
            }
        )
        if end >= len(text):
            break
        start = end - overlap
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

async def embed_texts(
    texts: list[str],
    client: AsyncOpenAI,
    model: str,
    batch_size: int = 20,
) -> list[list[float]]:
    """Embed a list of texts using the OpenAI-compatible embedding API.

    Handles batching automatically.
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        batch_embs = sorted(response.data, key=lambda d: d.index)
        for item in batch_embs:
            all_embeddings.append(item.embedding)
    return all_embeddings


async def _embed_single(
    text: str,
    client: AsyncOpenAI,
    model: str,
) -> list[float]:
    """Embed a single text string and return its vector."""
    response = await client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Indexing (chunk + embed + store)
# ---------------------------------------------------------------------------

async def index_paper_async(
    paper_id: int,
    pdf_path: str,
    embed_client: AsyncOpenAI,
    embed_model: str,
) -> dict[str, Any]:
    """Index a paper: extract text, chunk, embed, store in paper_chunks table.

    Also computes the document-level embedding (mean pooling of chunks) and
    persists it to papers.embedding.

    Returns {"indexed": True, "cached": bool, "chunks": int}.
    """
    from config import _get_rag_config

    # Already indexed?
    if db.has_paper_chunks(paper_id):
        return {"indexed": True, "cached": True, "chunks": 0}

    config = _get_rag_config()
    full_text = extract_pdf_text(pathlib.Path(pdf_path))
    chunks = chunk_text(full_text, config["chunk_chars"], config["chunk_overlap"])

    if not chunks:
        return {"indexed": False, "cached": False, "chunks": 0}

    # Embed all chunks
    chunk_texts = [c["text"] for c in chunks]
    embeddings = await embed_texts(chunk_texts, embed_client, embed_model)

    # Attach embeddings to chunk dicts
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    # Store in DB
    db.store_paper_chunks(paper_id, chunks)

    # Compute and persist document-level embedding (mean pooling)
    if not embeddings:
        return {"indexed": False, "cached": False, "chunks": 0}
    doc_embedding = np.mean(embeddings, axis=0).tolist()
    db.update_paper_embedding(paper_id, doc_embedding)

    return {"indexed": True, "cached": False, "chunks": len(chunks)}


# ---------------------------------------------------------------------------
# RAG answer
# ---------------------------------------------------------------------------

async def rag_answer_async(
    question: str,
    text: Optional[str] = None,
    pdf_path: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    llm_client: Optional[AsyncOpenAI] = None,
    llm_model: Optional[str] = None,
    embed_client: Optional[AsyncOpenAI] = None,
    embed_model: Optional[str] = None,
    return_details: bool = False,
) -> "str | dict[str, Any]":
    """Full RAG pipeline: chunk -> embed -> retrieve -> answer.

    Two modes:
    * DB-backed -- paper exists in DB with indexed chunks.  We embed the
      question, retrieve top-k chunks via pgvector, and generate an answer.
    * Ephemeral -- no indexed chunks.  We extract text (from pdf_path or
      text), chunk + embed in-memory, retrieve via numpy cosine similarity,
      then generate an answer.  If the paper exists in DB we also persist
      chunks for future calls.
    """
    from config import _get_rag_config

    assert llm_client is not None, "llm_client required"
    assert embed_client is not None, "embed_client required"

    config = _get_rag_config()
    evidence_k: int = config["evidence_k"]

    # Try to find paper in DB
    paper_id: Optional[int] = None
    if arxiv_id:
        paper = db.get_paper_by_arxiv_id(arxiv_id)
        if paper:
            paper_id = paper["id"]

    # ---- DB-backed retrieval ----
    if paper_id and db.has_paper_chunks(paper_id):
        q_emb = await _embed_single(question, embed_client, embed_model)
        chunks_rows = db.search_paper_chunks(paper_id, q_emb, top_k=evidence_k)
        retrieved = [
            {"text": r["text"], "score": r.get("similarity")}
            for r in chunks_rows
        ]
    else:
        # ---- Ephemeral mode ----
        if pdf_path:
            full_text = extract_pdf_text(pathlib.Path(pdf_path))
        elif text:
            full_text = text
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide text or pdf_path for RAG query.",
            )

        text_chunks = chunk_text(
            full_text, config["chunk_chars"], config["chunk_overlap"]
        )
        chunk_texts = [c["text"] for c in text_chunks]

        # Embed chunks + question in a single batch
        all_embeddings = await embed_texts(
            chunk_texts + [question], embed_client, embed_model
        )
        chunk_embeddings = all_embeddings[: len(chunk_texts)]
        q_emb = all_embeddings[-1]

        # In-memory cosine similarity
        retrieved = _in_memory_search(
            chunk_texts, chunk_embeddings, q_emb, evidence_k
        )

        # Persist to DB if paper exists
        if paper_id and chunk_embeddings:
            for chunk, emb in zip(text_chunks, chunk_embeddings):
                chunk["embedding"] = emb
            db.store_paper_chunks(paper_id, text_chunks)
            # Also persist doc-level embedding
            doc_emb = np.mean(chunk_embeddings, axis=0).tolist()
            db.update_paper_embedding(paper_id, doc_emb)

    # ---- Generate answer ----
    context = "\n\n---\n\n".join(c["text"] for c in retrieved)
    answer = await _generate_answer(question, context, llm_client, llm_model)

    if return_details:
        return {
            "answer": answer,
            "chunks_retrieved": len(retrieved),
            "chunks": [
                {
                    "text": clean_pdf_text(c["text"][:500]),
                    "score": c.get("score"),
                }
                for c in retrieved
            ],
        }
    return answer


# ---------------------------------------------------------------------------
# Ensure paper has a document-level embedding
# ---------------------------------------------------------------------------

async def ensure_paper_embedding(
    arxiv_id: str,
    embed_client: Optional[AsyncOpenAI] = None,
    embed_model: Optional[str] = None,
) -> bool:
    """Make sure papers.embedding is populated.

    1. Already has embedding -> True
    2. Has chunks in paper_chunks -> compute mean, store -> True
    3. Has PDF on disk -> index (create chunks + embedding) -> True
    4. Otherwise -> False
    """
    paper = db.get_paper_by_arxiv_id(arxiv_id)
    if not paper:
        return False

    if paper.get("embedding") is not None:
        return True

    paper_id = paper["id"]

    # Compute from existing chunks
    if db.has_paper_chunks(paper_id):
        chunks = db.get_paper_chunks(paper_id)
        embeddings = [c["embedding"] for c in chunks if c.get("embedding") is not None]
        if embeddings:
            doc_emb = np.mean(embeddings, axis=0).tolist()
            db.update_paper_embedding(paper_id, doc_emb)
            return True

    # Try to index from PDF
    pdf_path = paper.get("pdf_path")
    if pdf_path and pathlib.Path(pdf_path).exists() and embed_client and embed_model:
        result = await index_paper_async(paper_id, pdf_path, embed_client, embed_model)
        return result.get("indexed", False)

    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _in_memory_search(
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
    query_embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return top-k chunks by cosine similarity (numpy, in-memory)."""
    if not chunk_embeddings:
        return []

    emb_array = np.array(chunk_embeddings, dtype=np.float32)
    q_array = np.array(query_embedding, dtype=np.float32)

    # Cosine similarity
    norms = np.linalg.norm(emb_array, axis=1) * np.linalg.norm(q_array)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    sims = np.dot(emb_array, q_array) / norms

    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [
        {"text": chunk_texts[i], "score": float(sims[i])}
        for i in top_indices
    ]


async def _generate_answer(
    question: str,
    context: str,
    llm_client: AsyncOpenAI,
    llm_model: str,
    max_context_chars: int = 16000,
) -> str:
    """Call the LLM to answer *question* given retrieved *context*.

    Truncates *context* if it would push the prompt beyond the model's
    context window.  A rough 4-chars-per-token heuristic is used.
    """
    # Truncate context to fit within model limits
    # Reserve ~1500 tokens for the answer and prompt overhead
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n\n[Context truncated for length]"

    prompt = (
        "Answer the following question based ONLY on the provided context "
        "from a research paper. If the context does not contain enough "
        "information, say so.\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + question + "\n\n"
        "Answer:"
    )

    # Estimate tokens: ~4 chars per token for English text
    prompt_tokens_est = len(prompt) // 4
    # Leave room for the answer; clamp max_tokens to at least 256
    max_tokens = max(256, 8192 - prompt_tokens_est - 100)
    max_tokens = min(max_tokens, 2000)  # cap at 2000

    response = await llm_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()