"""
Summarizer - Resilient Groq + Local NLP Summarization
Consolidates Groq API calls with rate limiting, circuit breaker, model rotation, and local fallback.
"""
from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import requests

from backend.circuit_breaker import CircuitOpenError, get_circuit_breaker
from backend.logger import get_logger, log_groq_call
from backend.models import SummarizeResult
from backend.nlp_engine import extractive_summarize
from backend.rate_limiter import get_groq_rate_limiter
from config import GroqConfig, get_config

logger = get_logger("summarizer")

MAX_CONTENT_CHARS = 800  # TPM limitini korumak için haber başına maks karakter


class _EmptySummaryError(ValueError):
    """Raised when Groq returns HTTP 200 but an empty/unusable summary."""


class _RateLimitError(Exception):
    """Raised on HTTP 429 so circuit breaker doesn't trip on quota limits."""


def _clean_groq_summary(raw_content: Optional[str]) -> str:
    """Strip inline reasoning tags (<think>...</think>) and normalize whitespace."""
    if not raw_content:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _truncate_content(content: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """Truncate article content cleanly at word boundary to stay within TPM limits."""
    if not content or len(content) <= max_chars:
        return content or ""
    truncated = content[:max_chars]
    last_space = truncated.rfind(" ")
    return (truncated[:last_space] if last_space > max_chars * 0.8 else truncated) + "..."


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


# --- Model Rotation with Cooldown ---

class ModelRotator:
    """Thread-safe round-robin rotator with per-model 429 cooldown support."""

    __slots__ = ("_models", "_index", "_lock", "_cooldowns")

    def __init__(self, models: List[str]):
        if not models:
            raise ValueError("ModelRotator requires at least one model")
        self._models = list(models)
        self._index = 0
        self._lock = threading.RLock()
        self._cooldowns: dict[str, float] = {}

    def mark_rate_limited(self, model: str, cooldown_seconds: float = 1.5) -> None:
        """Put a rate-limited model on temporary cooldown so workers skip it."""
        with self._lock:
            self._cooldowns[model] = time.time() + cooldown_seconds

    def is_available(self, model: str) -> bool:
        with self._lock:
            until = self._cooldowns.get(model, 0.0)
            return time.time() >= until

    def min_wait_seconds(self) -> float:
        """Return seconds until the earliest model becomes available (0 if one is ready)."""
        with self._lock:
            now = time.time()
            waits = [max(0.0, self._cooldowns.get(m, 0.0) - now) for m in self._models]
            return min(waits) if waits else 0.0

    def current_model(self) -> str:
        with self._lock:
            now = time.time()
            n = len(self._models)
            for offset in range(n):
                candidate = self._models[(self._index + offset) % n]
                if now >= self._cooldowns.get(candidate, 0.0):
                    self._index = (self._index + offset) % n
                    return candidate
            return self._models[self._index % n]

    def next_model(self) -> str:
        with self._lock:
            self._index = (self._index + 1) % len(self._models)
            return self.current_model()


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

        self._system_prompt = (
            "Sen tarafsız ve profesyonel bir Türkçe haber editörüsün. "
            "Verilen haber başlığı ve metnini analiz et. "
            "Tam olarak 2 akıcı, net ve anlaşılır cümleyle Türkçe bir özet çıkar. "
            "Özet, haberdaki ana olayı ve önemli detayı içermeli. "
            "Ekstra giriş/çıkış cümlesi (örneğin 'İşte özet:', 'Özet:') yazma, sadece özeti ver. "
            "Başlıkla alakasız bilgiler ekleme, metinde olmayan şeyleri uydurma. "
            "Cümleler kısa ve net olsun, gereksiz sıfatlardan kaçın."
        )

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
        """Summarize a single article with full resilience and short cooldown recovery."""
        if not self.groq_api_key:
            return self.local_summarizer.summarize(title, content)

        if self._groq_permanently_unavailable:
            return self._fallback("circuit_open", title, content)

        safe_content = _truncate_content(content)

        for _round in range(2):
            if self._rotator:
                wait = self._rotator.min_wait_seconds()
                if 0 < wait <= 1.6:
                    time.sleep(wait + random.uniform(0.05, 0.25))

            for attempt in range(len(self._rotation_models)):
                model = self._rotator.current_model() if self._rotator else self._rotation_models[0]

                if self._rotator and not self._rotator.is_available(model):
                    self._rotator.next_model()
                    continue

                rate_limiter = get_groq_rate_limiter(model)
                circuit_breaker = get_circuit_breaker(f"groq_api:{model}", groq_config=self.groq_config)

                if circuit_breaker.state == "open":
                    if self._rotator:
                        self._rotator.next_model()
                    continue

                if not rate_limiter.acquire(count=1, blocking=False):
                    logger.warning(
                        f"Groq 429 on model {model}, failing over to next model",
                        extra={"extra_fields": {"event": "groq_rate_limit_failover", "model": model, "attempt": attempt + 1}},
                    )
                    if self._rotator:
                        self._rotator.mark_rate_limited(model, 1.5)
                        self._rotator.next_model()
                    continue

                try:
                    user_prompt = f"Başlık: {title}\nİçerik: {safe_content}"
                    raw_text = self._call_groq_api(
                        system_prompt=self._system_prompt,
                        user_prompt=user_prompt,
                        model=model,
                        circuit_breaker=circuit_breaker,
                        max_tokens=self.groq_config.max_tokens,
                    )
                    summary_text = _clean_groq_summary(raw_text)
                    if not summary_text:
                        raise _EmptySummaryError(f"Groq model {model} returned empty content")

                    if self._rotator:
                        self._rotator.next_model()
                    return SummarizeResult(
                        summary=summary_text,
                        engine=f"Groq ({model})",
                        success=True,
                        model=model,
                    )
                except _RateLimitError:
                    logger.warning(
                        f"Groq 429 on model {model} (HTTP)",
                        extra={"extra_fields": {"event": "groq_429_http", "model": model}},
                    )
                    if self._rotator:
                        self._rotator.mark_rate_limited(model, 1.5)
                        self._rotator.next_model()
                    continue
                except requests.Timeout:
                    return self._fallback("timeout", title, content)
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 404 and "model_not_found" in e.response.text:
                        logger.warning(f"Groq model unavailable ({model}), failing over")
                        if self._rotator:
                            self._rotator.mark_rate_limited(model, 3600.0)
                            self._rotator.next_model()
                        continue
                    return self._fallback("http_error", title, content)
                except CircuitOpenError:
                    return self._fallback("circuit_open", title, content)
                except _EmptySummaryError as e:
                    logger.warning(f"Groq returned empty summary on model {model}: {e}")
                    if self._rotator:
                        self._rotator.next_model()
                    continue
                except Exception:
                    return self._fallback("unexpected_error", title, content)

        return self._fallback("rate_limit_timeout", title, content)

    def _call_groq_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        circuit_breaker,
        max_tokens: int,
    ) -> str:
        """Low-level Groq HTTP call with circuit breaker and logging."""
        start_time = time.time()

        def _do_request() -> Optional[str]:
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.groq_config.temperature,
                "max_tokens": max_tokens,
            }

            if model.startswith("qwen/"):
                payload["reasoning_effort"] = "none"
            elif model.startswith("openai/gpt-oss"):
                payload["reasoning_effort"] = "low"
                payload["include_reasoning"] = False

            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
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
                choices = data.get("choices") or []
                if not choices:
                    raise _EmptySummaryError(f"Groq model {model} returned no choices")
                message = choices[0].get("message") or {}
                content = message.get("content") or ""
                if not content.strip():
                    raise _EmptySummaryError(f"Groq model {model} returned empty content")
                return content
            elif response.status_code == 429:
                return None  # Circuit breaker'a failure yazdırma
            elif response.status_code == 401:
                raise requests.HTTPError(f"Invalid API key: {response.text}", response=response)
            elif response.status_code == 404 and "model_not_found" in response.text:
                raise requests.HTTPError(f"Model not found: {response.text}", response=response)
            else:
                raise requests.HTTPError(f"Groq API {response.status_code}: {response.text}", response=response)

        raw_content = circuit_breaker.call(_do_request)
        if raw_content is None:
            raise _RateLimitError("Rate limited (HTTP 429)")

        duration_ms = (time.time() - start_time) * 1000
        log_groq_call(logger, model, success=True, duration_ms=duration_ms)
        return raw_content

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