from __future__ import annotations

import os
import pathlib
from typing import Any

import arxiv
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="ingest-parse-service")


class ArxivResolveRequest(BaseModel):
    arxiv_id: str | None = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: str | None = Field(default=None, description="arXiv URL")


class ArxivDownloadRequest(BaseModel):
    arxiv_id: str | None = Field(default=None, description="arXiv ID, e.g. 1706.03762")
    url: str | None = Field(default=None, description="arXiv URL")
    output_dir: str | None = Field(default=None, description="Directory to store downloads")


class PdfExtractRequest(BaseModel):
    pdf_path: str = Field(description="Local PDF file path")
    grobid_url: str | None = Field(default=None, description="Override GROBID URL")


def _require_identifier(arxiv_id: str | None, url: str | None) -> str:
    if arxiv_id:
        return arxiv_id
    if url:
        return url
    raise HTTPException(status_code=400, detail="Provide arxiv_id or url.")


def _get_grobid_url(override: str | None) -> str:
    return override or os.getenv("GROBID_URL", "http://localhost:9070")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/arxiv/resolve")
def arxiv_resolve(payload: ArxivResolveRequest) -> dict[str, Any]:
    identifier = _require_identifier(payload.arxiv_id, payload.url)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[identifier])
    results = list(client.results(search))
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
    except Exception as exc:  # noqa: BLE001 - propagate as response field
        source_error = str(exc)

    return {
        "arxiv_id": result.get_short_id(),
        "pdf_path": pdf_path,
        "latex_path": latex_path,
        "latex_error": source_error,
    }


@app.post("/pdf/extract")
def pdf_extract(payload: PdfExtractRequest) -> dict[str, Any]:
    pdf_path = pathlib.Path(payload.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")

    grobid_url = _get_grobid_url(payload.grobid_url)
    endpoint = f"{grobid_url.rstrip('/')}/api/processFulltextDocument"

    with pdf_path.open("rb") as handle:
        files = {"input": (pdf_path.name, handle, "application/pdf")}
        response = requests.post(endpoint, files=files, timeout=120)

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"GROBID error: {response.status_code} {response.text[:200]}",
        )

    tei_xml = response.text
    parsed = _parse_tei(tei_xml)
    return {"tei_xml": tei_xml, **parsed}


def _parse_tei(tei_xml: str) -> dict[str, Any]:
    import xml.etree.ElementTree as ET

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    root = ET.fromstring(tei_xml)

    title_el = root.find(".//tei:titleStmt/tei:title", ns)
    abstract_el = root.find(".//tei:abstract", ns)

    sections = []
    for div in root.findall(".//tei:text/tei:body/tei:div", ns):
        head = div.find("tei:head", ns)
        paragraphs = [p.text for p in div.findall("tei:p", ns) if p.text]
        if paragraphs:
            sections.append(
                {
                    "title": head.text if head is not None else None,
                    "text": "\n".join(paragraphs),
                }
            )

    references = []
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", ns):
        title = bibl.find(".//tei:title", ns)
        authors = [a.text for a in bibl.findall(".//tei:author/tei:persName/tei:surname", ns) if a.text]
        references.append(
            {
                "title": title.text if title is not None else None,
                "authors": authors,
            }
        )

    return {
        "title": title_el.text if title_el is not None else None,
        "abstract": "".join(abstract_el.itertext()).strip() if abstract_el is not None else None,
        "sections": sections,
        "references": references,
    }
