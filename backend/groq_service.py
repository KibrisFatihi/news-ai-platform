"""
Groq AI Service Module
Provides free, high-speed LLM summarization using Groq Cloud API (LLaMA 3.3 / LLaMA 3.1).
Gracefully falls back to local NLP engine when offline or if no API key is provided.

Refactored to use ResilientSummarizer - consolidates 5 fallback blocks into 1.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.summarizer import ResilientSummarizer, SummarizeResult, create_resilient_summarizer
from backend.nlp_engine import extractive_summarize
from backend.rate_limiter import get_groq_rate_limiter
from backend.circuit_breaker import get_circuit_breaker
from backend.logger import get_logger
from config import GroqConfig, get_config

logger = get_logger("groq_service")


def get_groq_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """
    Retrieve Groq API key from override, environment variable, or config.

    Args:
        override_key: Optional key passed directly (highest priority)

    Returns:
        API key string or None
    """
    return get_config().get_groq_api_key(override_key)


# --- System Prompt (moved to summarizer module, kept for backwards compat) ---
SYSTEM_PROMPT = (
    "Sen tarafsız ve profesyonel bir Türkçe haber editörüsün. "
    "Verilen haber başlığı ve metnini analiz et. "
    "Tam olarak 2 akıcı, net ve anlaşılır cümleyle Türkçe bir özet çıkar. "
    "Ekstra giriş/çıkış cümlesi (örneğin 'İşte özet:' gibi) yazma, sadece özeti ver."
)


def summarize_with_groq(
    title: str,
    content: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    groq_config: Optional[GroqConfig] = None
) -> Dict[str, Any]:
    """
    Summarizes news content using Groq Free API.
    If no key is present or request fails, falls back to extractive NLP.

    Features (now via ResilientSummarizer):
    - Rate limiting (respects Groq free tier limits)
    - Circuit breaker (prevents cascade failures)
    - Structured logging
    - Graceful fallback to local NLP

    Args:
        title: Article title
        content: Article content
        api_key: Optional API key override
        model: Optional model override (uses config default if None)
        groq_config: Optional Groq config override

    Returns:
        {
            "summary": str,
            "engine": str ("Groq LLaMA 3.3" or "Yerel NLP (Sıfır Token)"),
            "success": bool,
            "model": str
        }
    """
    summarizer = create_resilient_summarizer(api_key=api_key, model=model)
    result: SummarizeResult = summarizer.summarize(title, content)
    return result.model_dump()


def get_groq_status() -> dict[str, Any]:
    """Get Groq service status for monitoring."""
    from backend.rate_limiter import get_groq_rate_limiter
    from backend.circuit_breaker import get_circuit_breaker

    rate_limiter = get_groq_rate_limiter()
    circuit_breaker = get_circuit_breaker("groq_api")

    return {
        "rate_limiter": rate_limiter.get_status(),
        "circuit_breaker": circuit_breaker.get_status(),
        "api_key_configured": bool(get_groq_api_key())
    }


# For backwards compatibility
GROQ_API_URL = get_config().groq.api_url
DEFAULT_MODEL = get_config().groq.default_model
FAST_MODEL = get_config().groq.fast_model