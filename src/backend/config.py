"""Configuration and prompt loading helpers."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from functools import lru_cache
from typing import Any, Optional

import yaml
from fastapi import HTTPException

_prompts_cache: dict[str, Any] | None = None
_prompts_mtime: float = 0


def _get_db_setting(key: str, default: Any = None, value_type: str = "string") -> Any:
    """Get a setting from the database, with type conversion and fallback to default.
    
    This function is used to check DB overrides before falling back to config.yaml values.
    """
    try:
        import db
        value = db.get_setting(key)
        if value is None:
            return default
        
        # Type conversion
        if value_type == "boolean":
            return value.lower() in ("true", "1", "yes")
        elif value_type == "integer":
            return int(value)
        elif value_type == "float":
            return float(value)
        else:
            return value
    except Exception:
        # If DB is not available (e.g., during startup), return default
        return default


@lru_cache(maxsize=4)
def _load_config_cached(config_mtime: float, config_path_str: str) -> dict[str, Any]:
    config_path = pathlib.Path(config_path_str)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return config


def _load_config() -> dict[str, Any]:
    candidates = [
        pathlib.Path("config/config.yaml"),                          # repo-root CWD
        pathlib.Path(__file__).parents[2] / "config" / "config.yaml",  # always correct
    ]
    for config_path in candidates:
        if config_path.exists():
            mtime = config_path.stat().st_mtime
            return _load_config_cached(mtime, str(config_path))
    raise HTTPException(status_code=500, detail="Config file not found: config/config.yaml")


def _load_prompts() -> dict[str, Any]:
    """Load prompts from JSON file with caching based on file modification time."""
    global _prompts_cache, _prompts_mtime

    prompts_path = pathlib.Path(__file__).parent / "prompts" / "prompts.json"
    if not prompts_path.exists():
        raise HTTPException(status_code=500, detail="Prompts file not found: prompts/prompts.json")

    current_mtime = prompts_path.stat().st_mtime
    if _prompts_cache is None or current_mtime > _prompts_mtime:
        _prompts_cache = json.loads(prompts_path.read_text(encoding="utf-8"))
        _prompts_mtime = current_mtime

    return _prompts_cache


def get_prompt(prompt_id: str, **kwargs: Any) -> str:
    """Get a prompt by ID and format it with the provided variables."""
    prompts = _load_prompts()
    if prompt_id not in prompts:
        raise HTTPException(status_code=500, detail=f"Prompt not found: {prompt_id}")

    template = prompts[prompt_id]["template"]
    return template.format(**kwargs)


def _load_prompt() -> tuple[str, str, str]:
    """Load paper_summary prompt from JSON prompts file."""
    prompts = _load_prompts()
    if "paper_summary" not in prompts:
        raise HTTPException(status_code=500, detail="Prompt 'paper_summary' not found in prompts.json")

    prompt_data = prompts["paper_summary"]
    prompt_id = prompt_data.get("id", "paper_summary_v1")
    prompt_body = prompt_data["template"]
    prompt_hash = hashlib.sha256(prompt_body.encode("utf-8")).hexdigest()
    return prompt_id, prompt_hash, prompt_body


def _convert_localhost_for_docker(url: str) -> str:
    """Convert localhost URLs to host.docker.internal when running in Docker.
    
    Note: This conversion is only needed for Docker. In Singularity/Apptainer,
    localhost works directly since containers share the host network.
    """
    if not url:
        return url
    
    # Skip conversion if running in Singularity/Apptainer (these env vars are set by the runtime)
    if os.environ.get("SINGULARITY_CONTAINER") or os.environ.get("APPTAINER_CONTAINER"):
        return url
    
    # Only convert for Docker (detected by postgresql:// in DATABASE_URL)
    in_docker = os.environ.get("DATABASE_URL", "").startswith("postgresql://")
    if in_docker:
        url = url.replace("://localhost:", "://host.docker.internal:")
        url = url.replace("://127.0.0.1:", "://host.docker.internal:")
    return url


def _get_endpoint_config() -> dict[str, str]:
    """Get endpoint configuration. DB settings override config.yaml."""
    config = _load_config()
    endpoints = config.get("endpoints", {})
    
    # Get defaults from config.yaml
    llm_url_default = endpoints.get("llm_base_url", config.get("openai_api_base", ""))
    embed_url_default = endpoints.get("embedding_base_url", config.get("openai_api_base3", ""))
    
    # Check DB overrides
    llm_url = _get_db_setting("llm_base_url", llm_url_default, "string")
    embed_url = _get_db_setting("embedding_base_url", embed_url_default, "string")
    
    return {
        "llm_base_url": _convert_localhost_for_docker(llm_url),
        "embedding_base_url": _convert_localhost_for_docker(embed_url),
        "api_key": endpoints.get("api_key", config.get("openai_api_key", "local-key")),
    }


def _get_rag_config() -> dict[str, Any]:
    """Get RAG configuration (chunking, retrieval). DB settings override config.yaml."""
    config = _load_config()
    # Support both 'rag' and legacy 'paperqa' section names
    pqa = config.get("rag", config.get("paperqa", {}))

    
    # Get defaults from config.yaml
    chunk_chars_default = int(pqa.get("chunk_chars", config.get("paperqa_chunk_chars", 5000)))
    chunk_overlap_default = int(pqa.get("chunk_overlap", config.get("paperqa_chunk_overlap", 250)))
    evidence_k_default = int(pqa.get("evidence_k", config.get("paperqa_evidence_k", 10)))
    evidence_summary_length_default = str(pqa.get("evidence_summary_length", config.get("paperqa_evidence_summary_length", "about 100 words")))
    
    return {
        "chunk_chars": _get_db_setting("chunk_chars", chunk_chars_default, "integer"),
        "chunk_overlap": _get_db_setting("chunk_overlap", chunk_overlap_default, "integer"),
        "use_doc_details": bool(pqa.get("use_doc_details", config.get("paperqa_use_doc_details", True))),
        "evidence_k": _get_db_setting("evidence_k", evidence_k_default, "integer"),
        "evidence_summary_length": _get_db_setting("evidence_summary_length", evidence_summary_length_default, "string"),
        "evidence_skip_summary": bool(pqa.get("evidence_skip_summary", config.get("paperqa_evidence_skip_summary", False))),
        "evidence_relevance_score_cutoff": float(pqa.get("evidence_relevance_score_cutoff", config.get("paperqa_evidence_relevance_score_cutoff", 1))),
    }


def _get_ui_config() -> dict[str, Any]:
    """Get UI configuration. DB settings override config.yaml."""
    config = _load_config()
    ui = config.get("ui", {})
    
    # Get defaults from config.yaml
    max_similar_default = int(ui.get("max_similar_papers", 5))
    auto_save_default = int(ui.get("tree_auto_save_interval_ms", 30000))
    tree_category_font_default = int(ui.get("tree_category_font_size", 20))
    tree_paper_font_default = int(ui.get("tree_paper_font_size", 20))
    
    return {
        "hover_debounce_ms": int(ui.get("hover_debounce_ms", 500)),
        "max_similar_papers": _get_db_setting("max_similar_papers", max_similar_default, "integer"),
        "tree_auto_save_interval_ms": _get_db_setting("tree_auto_save_interval_ms", auto_save_default, "integer"),
        "tree_category_font_size": _get_db_setting("tree_category_font_size", tree_category_font_default, "integer"),
        "tree_paper_font_size": _get_db_setting("tree_paper_font_size", tree_paper_font_default, "integer"),
    }


def _get_classification_config() -> dict[str, Any]:
    """Get classification configuration. DB settings override config.yaml."""
    config = _load_config()
    classification = config.get("classification", {})
    
    # Get defaults from config.yaml
    branching_default = int(classification.get("branching_factor", 5))
    rebuild_default = bool(classification.get("rebuild_on_ingest", True))
    
    return {
        "branching_factor": _get_db_setting("branching_factor", branching_default, "integer"),
        "rebuild_on_ingest": _get_db_setting("rebuild_on_ingest", rebuild_default, "boolean"),
    }


def _get_ingestion_config() -> dict[str, Any]:
    """Get ingestion configuration. DB settings override config.yaml."""
    config = _load_config()
    ingestion = config.get("ingestion", {})
    
    skip_existing_default = bool(ingestion.get("skip_existing", True))
    
    max_concurrent_default = int(ingestion.get("max_concurrent", 5))
    
    return {
        "skip_existing": _get_db_setting("skip_existing", skip_existing_default, "boolean"),
        "max_concurrent": max(1, _get_db_setting("max_concurrent", max_concurrent_default, "integer")),
    }


def _get_external_apis_config() -> dict[str, Any]:
    """Get external API configuration."""
    config = _load_config()
    apis = config.get("external_apis", {})
    return {
        "papers_with_code_enabled": bool(apis.get("papers_with_code_enabled", True)),
        "github_search_enabled": bool(apis.get("github_search_enabled", True)),
        "semantic_scholar_enabled": bool(apis.get("semantic_scholar_enabled", True)),
        "github_token": apis.get("github_token"),
        "semantic_scholar_api_key": apis.get("semantic_scholar_api_key"),
    }


def _get_topic_query_config() -> dict[str, Any]:
    """Get topic query configuration for multi-paper RAG. DB settings override config.yaml."""
    config = _load_config()
    topic = config.get("topic_query", {})
    
    # Get defaults from config.yaml
    max_papers_default = int(topic.get("max_papers_per_batch", 10))
    similarity_default = float(topic.get("similarity_threshold", 0.5))
    chunks_default = int(topic.get("chunks_per_paper", 5))
    debug_default = bool(topic.get("debug_mode", False))
    
    return {
        "max_papers_per_batch": _get_db_setting("max_papers_per_batch", max_papers_default, "integer"),
        "similarity_threshold": _get_db_setting("similarity_threshold", similarity_default, "float"),
        "chunks_per_paper": _get_db_setting("chunks_per_paper", chunks_default, "integer"),
        "debug_mode": _get_db_setting("topic_debug_mode", debug_default, "boolean"),
    }


# Backward-compat alias
_get_paperqa_config = _get_rag_config
