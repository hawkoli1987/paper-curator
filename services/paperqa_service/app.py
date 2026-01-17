from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field
import yaml

try:
    from paperqa import Docs
except Exception:  # noqa: BLE001
    Docs = None


app = FastAPI(title="paperqa-service")


class SummarizeRequest(BaseModel):
    text: str = Field(description="Full paper text or extracted sections")


class EmbedRequest(BaseModel):
    text: str = Field(description="Text to embed")


class QaRequest(BaseModel):
    context: str = Field(description="Context text to answer from")
    question: str = Field(description="Question to answer")


def _load_prompt() -> tuple[str, str, str]:
    prompt_path = Path("prompts/paper_summary.md")
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


@lru_cache(maxsize=1)
def _load_config() -> dict[str, Any]:
    config_path = Path("config/paperqa.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=500, detail="Config file not found: config/paperqa.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = (
        "openai_api_base",
        "openai_api_base2",
        "openai_api_base3",
        "openai_api_key",
    )
    for key in required:
        if key not in config:
            raise HTTPException(status_code=500, detail=f"Missing '{key}' in config.")
    return config


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize")
def summarize(payload: SummarizeRequest) -> dict[str, Any]:
    prompt_id, prompt_hash, prompt_body = _load_prompt()
    config = _load_config()
    base_url = config["openai_api_base"]
    api_key = config["openai_api_key"]
    model = _resolve_model(base_url, api_key)
    client = _get_openai_client(base_url, api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_body},
                {"role": "user", "content": payload.text},
            ],
            temperature=0.2,
        )
        summary = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc
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
    if Docs is None:
        raise HTTPException(status_code=500, detail="paper-qa is not installed.")
    config = _load_config()
    os.environ["OPENAI_API_BASE"] = config["openai_api_base"]
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
    docs = Docs()
    docs.add(payload.context, docname="context")
    try:
        result = docs.query(payload.question)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"QA request failed: {exc}") from exc
    return {"answer": str(result)}
