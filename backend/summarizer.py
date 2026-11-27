"""
Summarizer - Resilient Groq + Local NLP Summarization
Consolidates Groq API calls with rate limiting, circuit breaker, model rotation, and local fallback.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import requests

from backend.circuit_breaker import CircuitOpenError, get_circuit_breaker
from backend.logger import get_logger, log_groq_call
from backend.nlp_engine import extractive_summarize
from backend.rate_limiter import get_groq_rate_limiter
from backend.models import SummarizeResult
from config import GroqConfig, NLPConfig, get_config

logger = get_logger("summarizer")


class _EmptySummaryError(ValueError):
    """Raised when Groq returns HTTP 200 but an empty/unusable summary."""


def _clean_groq_summary(raw_content: Optional[str]) -> str:
    """Strip inline reasoning tags (<think>...</think>) and normalize whitespace."""
    if not raw_content:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


# --- Fallback Strategy ---

@dataclass(frozen=True)
class FallbackStrategy:
    """Configuration for a fallback strategy."""
    name: str
    engine_label: str
    model_label: str
    success: bool = False


FALLBACK_CHAIN: tuple[FallbackStrategy, ...] = (
    FallbackStrategy("rate_limit_timeout", "Yerel NLP (Yedek Mod - Rate Limit)", "Extractive TextRank"),
    FallbackStrategy("circuit_open", "Yerel NLP (Yedek Mod - Circuit Breaker)", "Extractive TextRank"),
    FallbackStrategy("timeout", "Yerel NLP (Yedek Mod - Timeout)", "Extractive TextRank"),
    FallbackStrategy("http_error", "Yerel NLP (Yedek Mod - HTTP Error)", "Extractive TextRank"),
    FallbackStrategy("unexpected_error", "Yerel NLP (Yedek Mod - Error)", "Extractive TextRank"),
)


# --- Model Rotation ---

class ModelRotator:
    """Thread-safe round-robin rotator over a list of model names."""

    __slots__ = ("_models", "_index", "_lock")

    def __init__(self, models: List[str]):
        if not models:
            raise ValueError("ModelRotator requires at least one model")
        self._models = list(models)
        self._index = 0
        self._lock = threading.RLock()

    def current_model(self) -> str:
        with self._lock:
            return self._models[self._index % len(self._models)]

    def next_model(self) -> str:
        with self._lock:
            self._index = (self._index + 1) % len(self._models)
            return self._models[self._index % len(self._models)]


# --- Local NLP Summarizer ---

class LocalNLPSummarizer:
    """Local NLP extractive summarizer (always available, zero cost)."""

    __slots__ = ("num_sentences",)

    def __init__(self, num_sentences: int = 2):
        self.num_sentences = num_sentences

    def summarize(self, title: str, content: str) -> SummarizeResult:
        summary = extractive_summarize(content, num_sentences=self.num_sentences, title=title)
        return SummarizeResult(
            summary=summary,
            engine="Yerel NLP (Sıfır Token)",
            success=True,
            model="Extractive TextRank",
        )


# --- Resilient Summarizer (Main Entry Point) ---

class ResilientSummarizer:
    """
    Resilient summarizer: Groq (with model rotation) -> fallback chain -> local NLP.

    When rotation enabled, articles round-robin across models. On 429, next model tried.
    """

    __slots__ = (
        "groq_api_key", "groq_config", "local_summarizer", "_groq_permanently_unavailable",
        "model", "_rotation_models", "_rotator", "_system_prompt"
    )

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: Optional[str] = None,
        groq_config: Optional[GroqConfig] = None,
        local_summarizer: Optional[LocalNLPSummarizer] = None,
        model_rotation: Optional[List[str]] = None,
        model_rotation_enabled: Optional[bool] = None,
    ):
        cfg = groq_config or get_config().groq
        nlp_cfg = get_config().nlp
        self.groq_api_key = groq_api_key or get_config().get_groq_api_key()
        self.groq_config = cfg
        self.local_summarizer = local_summarizer or LocalNLPSummarizer(num_sentences=nlp_cfg.summary_sentences)
        self._groq_permanently_unavailable = False

        # System prompt (single source of truth)
        self._system_prompt = (
            "Sen tarafsız ve profesyonel bir Türkçe haber editörüsün. "
            "Verilen haber başlığı ve metnini analiz et. "
            "Tam olarak 2 akıcı, net ve anlaşılır cümleyle Türkçe bir özet çıkar. "
            "Özet, haberdaki ana olayı ve önemli detayı içermeli. "
            "Ekstra giriş/çıkış cümlesi (örneğin 'İşte özet:', 'Özet:') yazma, sadece özeti ver. "
            "Başlıkla alakasız bilgiler ekleme, metinde olmayan şeyleri uydurma. "
            "Cümleler kısa ve net olsun, gereksiz sıfatlardan kaçın."
        )

        # Model rotation setup
        rotation_enabled = (
            cfg.model_rotation_enabled if model_rotation_enabled is None else model_rotation_enabled
        )
        rotation_models = list(model_rotation) if model_rotation is not None else list(cfg.model_rotation)

        if model:
            self.model = model
            self._rotation_models = [model]
            self._rotator = None
        elif rotation_enabled and len(rotation_models) > 1:
            self.model = rotation_models[0]
            self._rotation_models = rotation_models
            self._rotator = ModelRotator(rotation_models)
        else:
            self.model = rotation_models[0] if rotation_models else cfg.default_model
            self._rotation_models = [self.model]
            self._rotator = None

    def summarize(self, title: str, content: str) -> SummarizeResult:
        """Summarize with full resilience: Groq (rotation) -> fallback -> local NLP."""
        if not self.groq_api_key:
            logger.info("No Groq API key, using local NLP", extra={"extra_fields": {"event": "groq_fallback_no_key"}})
            return self.local_summarizer.summarize(title, content)

        if self._groq_permanently_unavailable:
            return self._fallback("circuit_open", title, content)

        for attempt in range(len(self._rotation_models)):
            model = self._rotator.current_model() if self._rotator else self._rotation_models[0]

            # Per-model rate limiter & circuit breaker
            rate_limiter = get_groq_rate_limiter(model)
            circuit_breaker = get_circuit_breaker(f"groq_api:{model}", groq_config=self.groq_config)

            # Check circuit breaker first
            if circuit_breaker.state == "open":
                return self._fallback("circuit_open", title, content)

            # Acquire rate limit slot (non-blocking)
            try:
                if not rate_limiter.acquire(count=1, blocking=False):
                    raise Exception("RateLimitExceeded")
            except Exception:
                logger.warning(
                    f"Groq 429 on model {model}, failing over to next model",
                    extra={"extra_fields": {"event": "groq_rate_limit_failover", "model": model, "attempt": attempt + 1}}
                )
                if self._rotator:
                    self._rotator.next_model()
                if attempt < len(self._rotation_models) - 1:
                    continue
                return self._fallback("rate_limit_timeout", title, content)

            # Make Groq API request
            try:
                result = self._make_groq_request(title, content, model, circuit_breaker)
                if self._rotator:
                    self._rotator.next_model()
                return result
            except requests.Timeout:
                return self._fallback("timeout", title, content)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    # Shouldn't reach here (caught by rate limiter), but safety
                    logger.warning(f"Groq 429 on model {model} (HTTP)", extra={"extra_fields": {"event": "groq_429_http", "model": model}})
                    if self._rotator:
                        self._rotator.next_model()
                    if attempt < len(self._rotation_models) - 1:
                        continue
                    return self._fallback("rate_limit_timeout", title, content)
                if e.response is not None and e.response.status_code == 404 and "model_not_found" in e.response.text:
                    logger.warning(
                        f"Groq model unavailable ({model}), failing over",
                        extra={"extra_fields": {"event": "groq_model_unavailable", "model": model}},
                    )
                    if self._rotator:
                        self._rotator.next_model()
                    if attempt < len(self._rotation_models) - 1:
                        continue
                    self._groq_permanently_unavailable = True
                    return self._fallback("http_error", title, content)
                return self._fallback("http_error", title, content)
            except CircuitOpenError:
                return self._fallback("circuit_open", title, content)
            except _EmptySummaryError as e:
                logger.warning(
                    f"Groq returned empty summary on model {model}: {e}",
                    extra={"extra_fields": {"event": "groq_empty_summary", "model": model, "attempt": attempt + 1}},
                )
                if self._rotator:
                    self._rotator.next_model()
                if attempt < len(self._rotation_models) - 1:
                    continue
                return self._fallback("unexpected_error", title, content)
            except Exception:
                return self._fallback("unexpected_error", title, content)

        return self._fallback("rate_limit_timeout", title, content)

    def _make_groq_request(self, title: str, content: str, model: str, circuit_breaker) -> SummarizeResult:
        """Execute Groq API request with circuit breaker protection."""
        start_time = time.time()

        def _do_request():
            user_prompt = f"Başlık: {title}\nİçerik: {content}"
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.groq_config.temperature,
                "max_tokens": self.groq_config.max_tokens,
            }
            if model.startswith("qwen/"):
                # Disable Qwen thinking tokens so completion budget goes to the summary.
                payload["reasoning_effort"] = "none"
            elif model.startswith("openai/gpt-oss"):
                # Minimize GPT-OSS reasoning budget and omit raw reasoning from response.
                payload["reasoning_effort"] = "low"
                payload["include_reasoning"] = False

            headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}

            response = requests.post(
                self.groq_config.api_url,
                headers=headers,
                json=payload,
                timeout=self.groq_config.request_timeout,
            )

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise _EmptySummaryError(f"Groq model {model} returned no choices")
                message = choices[0].get("message") or {}
                summary_text = _clean_groq_summary(message.get("content"))
                if not summary_text:
                    raise _EmptySummaryError(f"Groq model {model} returned empty content")
                return SummarizeResult(
                    summary=summary_text,
                    engine=f"Groq ({model})",
                    success=True,
                    model=model,
                )
            elif response.status_code == 429:
                raise requests.HTTPError(f"Rate limited: {response.text}", response=response)
            elif response.status_code == 401:
                raise requests.HTTPError(f"Invalid API key: {response.text}", response=response)
            elif response.status_code == 404 and "model_not_found" in response.text:
                raise requests.HTTPError(f"Model not found: {response.text}", response=response)
            else:
                raise requests.HTTPError(f"Groq API {response.status_code}: {response.text}", response=response)

        result = circuit_breaker.call(_do_request)
        duration_ms = (time.time() - start_time) * 1000
        log_groq_call(logger, model, success=True, duration_ms=duration_ms)
        return result

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
    model_rotation: Optional[List[str]] = None,
    model_rotation_enabled: Optional[bool] = None,
) -> ResilientSummarizer:
    """Create a resilient summarizer with default configuration."""
    return ResilientSummarizer(
        groq_api_key=api_key,
        model=model,
        model_rotation=model_rotation,
        model_rotation_enabled=model_rotation_enabled,
    )


# --- Backwards Compatibility ---

def summarize_with_groq(
    title: str,
    content: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    groq_config: Optional[GroqConfig] = None,
) -> dict[str, Any]:
    """Backwards compatible function that uses the new ResilientSummarizer."""
    summarizer = create_resilient_summarizer(api_key=api_key, model=model)
    result = summarizer.summarize(title, content)
    return result.model_dump()