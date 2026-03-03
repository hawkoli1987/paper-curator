"""LLM client helpers and model resolution."""
from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI, OpenAI


def get_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Create OpenAI client with ngrok header support if needed."""
    if "ngrok" in base_url.lower():
        import httpx
        http_client = httpx.Client(
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=600.0,
            follow_redirects=True,
        )
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            default_headers={"ngrok-skip-browser-warning": "true"},
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def get_async_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """Create AsyncOpenAI client with ngrok header support if needed."""
    if "ngrok" in base_url.lower():
        import httpx
        http_client = httpx.AsyncClient(
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=600.0,
            follow_redirects=True,
        )
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            default_headers={"ngrok-skip-browser-warning": "true"},
        )
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


@lru_cache(maxsize=3)
def resolve_model(base_url: str, api_key: str) -> str:
    """Resolve model name from endpoint, with ngrok free tier workaround."""
    client = get_openai_client(base_url, api_key)
    try:
        models = client.models.list()
        model_ids = sorted([model.id for model in models.data if getattr(model, "id", None)])
        assert model_ids, "No models returned by OpenAI-compatible endpoint."
        return model_ids[0]
    except Exception as e:
        error_msg = str(e)
        if "ngrok" in base_url.lower() and (
            "ERR_NGROK_3004" in error_msg
            or "gateway error" in error_msg.lower()
            or "invalid or incomplete HTTP response" in error_msg.lower()
            or "ngrok gateway error" in error_msg.lower()
        ):
            help_msg = (
                "ngrok ERR_NGROK_3004: Browser warning page blocking programmatic access.\n\n"
                "SOLUTIONS (choose one):\n\n"
                "1. Configure ngrok to skip browser warning (RECOMMENDED):\n"
                "   Restart ngrok with:\n"
                "   ngrok http 8001 --request-header-add 'ngrok-skip-browser-warning: true'\n\n"
                "   OR add to ~/.ngrok2/ngrok.yml:\n"
                "   tunnels:\n"
                "     llm:\n"
                "       addr: 8001\n"
                "       request_header:\n"
                "         add: ['ngrok-skip-browser-warning: true']\n\n"
                "2. Upgrade to ngrok paid plan (has Edge request headers)\n\n"
                "3. Use alternative tunneling:\n"
                "   - Cloudflare Tunnel (free, no browser warning)\n"
                "   - localtunnel (free, simple)\n"
                "   - serveo (free, SSH-based)\n\n"
                f"Current endpoint: {base_url}\n"
                f"Original error: {error_msg}"
            )
            raise Exception(help_msg)
        raise


def strip_thinking(content: str) -> str:
    """Remove <think>...</think> blocks from LLM response content.

    Applied defensively to all LLM responses. When vllm's reasoning parser is
    active the thinking content should already be in message.reasoning_content,
    but parser leakage or models that echo the tag can still produce <think>
    blocks in message.content. This strips them unconditionally.
    """
    import re
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def get_llm_config() -> dict[str, str]:
    """Load LLM configuration from config (with DB overrides).
    
    Returns:
        dict with 'base_url', 'api_key', 'model'
    """
    from config import _get_endpoint_config
    
    # Get config from centralized config module (supports DB overrides)
    endpoint_config = _get_endpoint_config()
    
    base_url = endpoint_config["llm_base_url"]
    api_key = endpoint_config["api_key"]
    model = resolve_model(base_url, api_key)
    
    return {"base_url": base_url, "api_key": api_key, "model": model}
