"""
Summarizer Protocol & Resilient Summarizer
Consolidates 5 duplicate fallback blocks into a single, testable component.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import requests

from backend.circuit_breaker import CircuitOpenError, get_circuit_breaker
from backend.logger import get_logger, log_groq_call
from backend.nlp_engine import extractive_summarize
from backend.rate_limiter import get_groq_rate_limiter
from backend.models import SummarizeResult
from config import GroqConfig, get_config

logger = get_logger("summarizer")


# --- Protocol Definition ---

class SummarizerProtocol(Protocol):
    """Protocol for any summarizer implementation."""

    def summarize(self, title: str, content: str) -> SummarizeResult:
        """Summarize the given title and content."""
        ...


class ModelUnavailableError(Exception):
    """Raised when the Groq model is permanently unavailable (e.g. 404 model_not_found).

    Retrying is wasteful; the summarizer should fall back to local NLP instead.
    """


# --- Fallback Strategy ---

@dataclass(frozen=True)
class FallbackStrategy:
    """Configuration for a fallback strategy."""
    name: str
    engine_label: str
    model_label: str
    success: bool = False


FALLBACK_CHAIN: tuple[FallbackStrategy, ...] = (
    FallbackStrategy(
        name="rate_limit_timeout",
        engine_label="Yerel NLP (Yedek Mod - Rate Limit)",
        model_label="Extractive TextRank",
    ),
    FallbackStrategy(
        name="circuit_open",
        engine_label="Yerel NLP (Yedek Mod - Circuit Breaker)",
        model_label="Extractive TextRank",
    ),
    FallbackStrategy(
        name="timeout",
        engine_label="Yerel NLP (Yedek Mod - Timeout)",
        model_label="Extractive TextRank",
    ),
    FallbackStrategy(
        name="http_error",
        engine_label="Yerel NLP (Yedek Mod - HTTP Error)",
        model_label="Extractive TextRank",
    ),
    FallbackStrategy(
        name="unexpected_error",
        engine_label="Yerel NLP (Yedek Mod - Error)",
        model_label="Extractive TextRank",
    ),
)


# --- Core Summarizer Implementations ---

class LocalNLPSummarizer:
    """Local NLP extractive summarizer (always available, zero cost)."""

    def __init__(self, num_sentences: int = 2):
        self.num_sentences = num_sentences

    def summarize(self, title: str, content: str) -> SummarizeResult:
        summary = extractive_summarize(content, num_sentences=self.num_sentences)
        return SummarizeResult(
            summary=summary,
            engine="Yerel NLP (Sıfır Token)",
            success=True,
            model="Extractive TextRank",
        )


class GroqSummarizer:
    """Groq API summarizer with rate limiting and circuit breaker."""

    def __init__(
        self,
        api_key: str,
        model: str,
        groq_config: GroqConfig,
        rate_limiter: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.groq_config = groq_config
        self.rate_limiter = rate_limiter or get_groq_rate_limiter()
        self.circuit_breaker = circuit_breaker or get_circuit_breaker(
            "groq_api", groq_config=groq_config
        )
        self.system_prompt = (
            "Sen tarafsız ve profesyonel bir Türkçe haber editörüsün. "
            "Verilen haber başlığı ve metnini analiz et. "
            "Tam olarak 2 akıcı, net ve anlaşılır cümleyle Türkçe bir özet çıkar. "
            "Ekstra giriş/çıkış cümlesi (örneğin 'İşte özet:' gibi) yazma, sadece özeti ver."
        )

    def _make_request(self, title: str, content: str) -> SummarizeResult:
        """Make the actual Groq API request."""
        user_prompt = f"Başlık: {title}\nİçerik: {content}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.groq_config.temperature,
            "max_tokens": self.groq_config.max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.groq_config.api_url,
            headers=headers,
            json=payload,
            timeout=self.groq_config.request_timeout,
        )

        if response.status_code == 200:
            data = response.json()
            summary_text = data["choices"][0]["message"]["content"].strip()
            return SummarizeResult(
                summary=summary_text,
                engine=f"Groq ({self.model})",
                success=True,
                model=self.model,
            )
        else:
            error_msg = f"Groq API status {response.status_code}: {response.text}"
            if response.status_code == 429:
                error_msg += " (Rate limited)"
            elif response.status_code == 401:
                error_msg += " (Invalid API key)"
            # A 404 model_not_found is permanent for this model/account — no
            # amount of retrying will make it succeed. Signal the resilient
            # summarizer to fall back to local NLP for the rest of this run.
            if response.status_code == 404 and "model_not_found" in response.text:
                raise ModelUnavailableError(error_msg, response=response)
            raise requests.HTTPError(error_msg, response=response)

    def summarize(self, title: str, content: str) -> SummarizeResult:
        """Summarize with Groq API, with full resilience."""
        start_time = time.time()

        # Check circuit breaker first: if OPEN there is no point in waiting on
        # the rate limiter just to fall back immediately.
        from backend.circuit_breaker import CircuitState
        if self.circuit_breaker.state == CircuitState.OPEN:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(
                "Circuit breaker OPEN, falling back to local NLP",
                extra={"extra_fields": {
                    "event": "groq_circuit_open",
                    "model": self.model,
                    "duration_ms": duration_ms,
                }},
            )
            raise CircuitOpenError(
                f"Circuit breaker '{self.circuit_breaker.name}' is OPEN."
            )

        # Acquire rate limit slot
        try:
            acquired = self.rate_limiter.acquire(count=1, blocking=True, timeout=30.0)
            if not acquired:
                raise TimeoutError("Rate limit timeout")
        except Exception as e:
            logger.warning(
                f"Rate limiter error: {e}, falling back",
                extra={"extra_fields": {"event": "groq_rate_limiter_error", "error": str(e)}}
            )
            raise

        # Execute with circuit breaker protection
        try:
            result = self.circuit_breaker.call(self._make_request, title, content)
            duration_ms = (time.time() - start_time) * 1000
            log_groq_call(logger, self.model, success=True, duration_ms=duration_ms)

            logger.info(
                f"Groq summarization successful: {self.model}",
                extra={"extra_fields": {
                    "event": "groq_success",
                    "model": self.model,
                    "duration_ms": duration_ms,
                    "summary_length": len(result.summary)
                }}
            )
            return result

        except CircuitOpenError:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(
                "Circuit breaker OPEN, falling back to local NLP",
                extra={"extra_fields": {
                    "event": "groq_circuit_open",
                    "model": self.model,
                    "duration_ms": duration_ms
                }}
            )
            raise

        except requests.Timeout:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"Groq API timeout ({self.groq_config.request_timeout}s), falling back",
                extra={"extra_fields": {
                    "event": "groq_timeout",
                    "model": self.model,
                    "duration_ms": duration_ms
                }}
            )
            raise

        except requests.HTTPError as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"Groq API HTTP error: {e}, falling back",
                extra={"extra_fields": {
                    "event": "groq_http_error",
                    "model": self.model,
                    "error": str(e),
                    "duration_ms": duration_ms
                }}
            )
            raise

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Groq API unexpected error: {e}, falling back",
                extra={"extra_fields": {
                    "event": "groq_unexpected_error",
                    "model": self.model,
                    "error": str(e),
                    "duration_ms": duration_ms
                }}
            )
            raise


# --- Resilient Summarizer (Main Entry Point) ---

class ResilientSummarizer:
    """
    Resilient summarizer that tries Groq first, then falls back through
    a chain of strategies, finally using local NLP.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: Optional[str] = None,
        groq_config: Optional[GroqConfig] = None,
        local_summarizer: Optional[LocalNLPSummarizer] = None,
        groq_summarizer_factory: Optional[Callable[[str, str, GroqConfig], GroqSummarizer]] = None,
    ):
        cfg = groq_config or get_config().groq
        self.groq_api_key = groq_api_key or get_config().get_groq_api_key()
        self.model = model or cfg.default_model
        self.groq_config = cfg
        self.local_summarizer = local_summarizer or LocalNLPSummarizer()
        self.groq_summarizer_factory = groq_summarizer_factory or GroqSummarizer
        # Once Groq proves permanently unavailable (e.g. model_not_found) in
        # this run, skip it for every remaining article instead of retrying.
        self._groq_permanently_unavailable: bool = False

    def summarize(self, title: str, content: str) -> SummarizeResult:
        """
        Summarize with full resilience: Groq -> fallback chain -> local NLP.
        """
        # No API key? Use local directly.
        if not self.groq_api_key:
            logger.info(
                "No Groq API key, using local NLP",
                extra={"extra_fields": {"event": "groq_fallback_no_key", "model": self.model}}
            )
            return self.local_summarizer.summarize(title, content)

        # Groq already proven unavailable in this run? Skip straight to local.
        if self._groq_permanently_unavailable:
            return self._fallback("circuit_open", title, content)

        # Try Groq
        groq_summarizer = self.groq_summarizer_factory(
            self.groq_api_key, self.model, self.groq_config
        )

        try:
            return groq_summarizer.summarize(title, content)
        except ModelUnavailableError as e:
            # Mark Groq unavailable for the rest of this run so every
            # remaining article goes straight to local NLP (no more 404s,
            # no rate-limiter blocking, no circuit-breaker thrashing).
            self._groq_permanently_unavailable = True
            logger.warning(
                f"Groq model unavailable ({self.model}), switching to local NLP for remaining articles",
                extra={"extra_fields": {
                    "event": "groq_model_unavailable",
                    "model": self.model,
                    "error": str(e),
                }},
            )
            return self._fallback("http_error", title, content)
        except TimeoutError:
            return self._fallback("rate_limit_timeout", title, content)
        except CircuitOpenError:
            return self._fallback("circuit_open", title, content)
        except requests.Timeout:
            return self._fallback("timeout", title, content)
        except requests.HTTPError:
            return self._fallback("http_error", title, content)
        except Exception:
            return self._fallback("unexpected_error", title, content)

    def _fallback(self, strategy_name: str, title: str, content: str) -> SummarizeResult:
        """Execute a fallback strategy."""
        strategy = next(s for s in FALLBACK_CHAIN if s.name == strategy_name)
        local_result = self.local_summarizer.summarize(title, content)
        return SummarizeResult(
            summary=local_result.summary,
            engine=strategy.engine_label,
            success=strategy.success,
            model=strategy.model_label,
        )


# --- Factory Function ---

def create_resilient_summarizer(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ResilientSummarizer:
    """Create a resilient summarizer with default configuration."""
    return ResilientSummarizer(groq_api_key=api_key, model=model)


# --- Backwards Compatibility ---

def summarize_with_groq(
    title: str,
    content: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    groq_config: Optional[GroqConfig] = None,
) -> dict[str, Any]:
    """
    Backwards compatible function that uses the new ResilientSummarizer.
    """
    summarizer = create_resilient_summarizer(api_key=api_key, model=model)
    result = summarizer.summarize(title, content)
    return result.model_dump()