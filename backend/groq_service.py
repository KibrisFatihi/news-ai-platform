"""
Groq AI Service Module - Compatibility Layer
Re-exports from summarizer.py and config.py for backward compatibility.
All actual logic consolidated in backend/summarizer.py.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.summarizer import ResilientSummarizer, SummarizeResult, create_resilient_summarizer, summarize_with_groq
from backend.rate_limiter import get_groq_rate_limiter
from backend.circuit_breaker import get_circuit_breaker
from backend.logger import get_logger
from config import GroqConfig, get_config

logger = get_logger("groq_service")


# --- Re-exported constants from config ---
GROQ_API_URL = get_config().groq.api_url
DEFAULT_MODEL = get_config().groq.default_model
FAST_MODEL = get_config().groq.fast_model


def get_groq_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Groq API key from override or environment (delegates to config)."""
    return get_config().get_groq_api_key(override_key)


def get_groq_status() -> dict[str, Any]:
    """Get Groq service status for monitoring (delegates to rate_limiter + circuit_breaker)."""
    rate_limiter = get_groq_rate_limiter()
    circuit_breaker = get_circuit_breaker("groq_api")

    return {
        "rate_limiter": rate_limiter.get_status(),
        "circuit_breaker": circuit_breaker.get_status(),
        "api_key_configured": bool(get_groq_api_key()),
        "model_rotation": {
            "enabled": get_config().groq.model_rotation_enabled,
            "models": list(get_config().groq.model_rotation),
        },
    }


# summarize_with_groq is re-exported from summarizer (backwards compatible)
# ResilientSummarizer, SummarizeResult, create_resilient_summarizer are re-exported from summarizer