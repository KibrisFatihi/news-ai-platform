"""
Tests for the Summarizer module (ResilientSummarizer).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.circuit_breaker import CircuitOpenError
from backend.summarizer import (
    FALLBACK_CHAIN,
    FallbackStrategy,
    LocalNLPSummarizer,
    ResilientSummarizer,
    SummarizeResult,
    create_resilient_summarizer,
)


class TestLocalNLPSummarizer:
    """Tests for LocalNLPSummarizer."""

    def test_summarize_basic(self):
        """Test basic local summarization."""
        summarizer = LocalNLPSummarizer(num_sentences=2)
        content = "İlk cümle. İkinci cümle. Üçüncü cümle. Dördüncü cümle."

        result = summarizer.summarize("Test Title", content)

        assert isinstance(result, SummarizeResult)
        assert result.success is True
        assert "Yerel NLP" in result.engine
        assert result.model == "Extractive TextRank"
        assert len(result.summary) > 0

    def test_summarize_empty_content(self):
        """Test with empty content."""
        summarizer = LocalNLPSummarizer()
        result = summarizer.summarize("Title", "")
        assert result.summary == ""


class TestResilientSummarizer:
    """Tests for ResilientSummarizer."""

    def test_summarize_no_api_key(self):
        """Test fallback to local NLP when no API key."""
        summarizer = create_resilient_summarizer(api_key=None)
        result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

        assert result.success is True
        assert "Yerel NLP (Sıfır Token)" in result.engine

    def test_summarize_groq_success(self):
        """Test successful Groq summarization."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Groq summary"}}]
            }
            mock_post.return_value = mock_response

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is True
            assert "Groq" in result.engine

    def test_summarize_rate_limit_fallback(self):
        """Test fallback when rate limit times out."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = False  # Rate limit timeout
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb.return_value = mock_cb_instance

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is False
            assert "Rate Limit" in result.engine

    def test_summarize_circuit_open_fallback(self):
        """Test fallback when circuit breaker is open."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = CircuitOpenError("Circuit open")
            mock_cb.return_value = mock_cb_instance

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is False
            assert "Circuit Breaker" in result.engine

    def test_summarize_timeout_fallback(self):
        """Test fallback on timeout."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            mock_post.side_effect = requests.Timeout("Request timed out")

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is False
            assert "Timeout" in result.engine

    def test_summarize_http_error_fallback(self):
        """Test fallback on HTTP error."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_post.return_value = mock_response

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is False
            assert "HTTP Error" in result.engine

    def test_summarize_unexpected_error_fallback(self):
        """Test fallback on unexpected error."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            mock_post.side_effect = Exception("Unexpected error")

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is False
            assert "Error" in result.engine


class TestModelRotation:
    """Tests for Groq model rotation (round-robin + 429 failover)."""

    def _make_response(self, status_code: int, text: str = "") -> MagicMock:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = text
        if status_code == 200:
            mock_response.json.return_value = {
                "choices": [{"message": {"content": f"summary via {text}"}}]
            }
        return mock_response

    def _mock_groq(self, mock_post, responses):
        """Queue canned responses for requests.post (status_code, text)."""
        queue = [self._make_response(status, text) for status, text in responses]
        mock_post.side_effect = queue

    def _used_models(self, mock_post) -> list[str]:
        return [call.kwargs["json"]["model"] for call in mock_post.call_args_list]

    def test_rotation_uses_configured_models(self):
        """Rotation models come from config and rotator is wired up."""
        summarizer = create_resilient_summarizer(api_key="gsk_test_key")
        assert summarizer._rotator is not None
        assert summarizer._rotation_models == [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]

    def test_round_robin_order_across_articles(self):
        """Consecutive articles hit consecutive models in rotation order."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            self._mock_groq(mock_post, [(200, "a"), (200, "b"), (200, "c")])

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            results = [
                summarizer.summarize("Title", f"Content {i}.")
                for i in range(3)
            ]

            assert [r.model for r in results] == summarizer._rotation_models
            assert self._used_models(mock_post) == [r.model for r in results]
            assert all(r.success for r in results)
            assert all("Groq" in r.engine for r in results)

    def test_429_fails_over_to_next_model(self):
        """429 on the first model is retried on the next; local NLP not used."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            self._mock_groq(mock_post, [
                (429, "rate limit exceeded"),
                (200, "openai/gpt-oss-20b"),
            ])

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is True
            assert result.model == "openai/gpt-oss-20b"
            assert "Groq" in result.engine
            assert self._used_models(mock_post) == [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ]

    def test_all_models_429_falls_back_to_local_nlp(self):
        """All models rate-limited -> local NLP fallback, every model tried."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            self._mock_groq(mock_post, [
                (429, "rate limit exceeded"),
                (429, "rate limit exceeded"),
                (429, "rate limit exceeded"),
            ])

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is False
            assert "Rate Limit" in result.engine
            assert "Yerel NLP" in result.engine
            assert self._used_models(mock_post) == list(summarizer._rotation_models)

    def test_explicit_model_bypasses_rotation(self):
        """``model`` parameter disables rotation; 429 does not try other models."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            self._mock_groq(mock_post, [(429, "rate limit exceeded")])

            summarizer = create_resilient_summarizer(
                api_key="gsk_test_key", model="qwen/qwen3.8-27b"
            )
            assert summarizer._rotator is None
            assert summarizer._rotation_models == ["qwen/qwen3.8-27b"]

            result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

            assert result.success is False
            assert "Rate Limit" in result.engine
            assert self._used_models(mock_post) == ["qwen/qwen3.8-27b"]

    def test_rotation_disabled_uses_default_model(self):
        """model_rotation_enabled=False uses a single model, no rotator."""
        summarizer = create_resilient_summarizer(
            api_key="gsk_test_key", model_rotation_enabled=False
        )
        assert summarizer._rotator is None
        assert summarizer.model == "openai/gpt-oss-120b"
        assert summarizer._rotation_models == ["openai/gpt-oss-120b"]
        # 429 on the only model -> fallback, no other model tried
        with patch("backend.summarizer.requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.text = "rate limit exceeded"
            mock_post.return_value = mock_response
            with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
                 patch("backend.summarizer.get_circuit_breaker") as mock_cb:
                mock_rl_instance = MagicMock()
                mock_rl_instance.acquire.return_value = True
                mock_rl.return_value = mock_rl_instance
                mock_cb_instance = MagicMock()
                mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
                mock_cb.return_value = mock_cb_instance
                result = summarizer.summarize("Test Title", "Test content here. Another sentence.")
                assert result.success is False
                assert "Rate Limit" in result.engine
                assert mock_post.call_count == 1


class TestFallbackChain:
    """Tests for fallback chain configuration."""

    def test_fallback_chain_order(self):
        """Test that fallback chain has correct order."""
        names = [s.name for s in FALLBACK_CHAIN]
        expected = [
            "rate_limit_timeout",
            "circuit_open",
            "timeout",
            "http_error",
            "unexpected_error"
        ]
        assert names == expected

    def test_fallback_strategy_labels(self):
        """Test that fallback strategies have correct labels."""
        for strategy in FALLBACK_CHAIN:
            assert isinstance(strategy, FallbackStrategy)
            assert strategy.name
            assert strategy.engine_label
            assert strategy.model_label
            assert strategy.success is False  # All fallbacks are failures


class TestSummarizeWithGroqBackwardsCompat:
    """Tests for backwards compatible summarize_with_groq function."""

    def test_summarize_with_groq_no_key(self):
        """Test backwards compatible function with no key."""
        from backend.groq_service import summarize_with_groq

        with patch("backend.groq_service.get_groq_api_key") as mock_get_key, \
             patch("backend.summarizer.extractive_summarize") as mock_extractive:

            mock_get_key.return_value = None
            mock_extractive.return_value = "Local summary"

            result = summarize_with_groq("Test Title", "Test content")

            assert result["success"] is True
            assert "Yerel NLP" in result["engine"]
            assert result["summary"] == "Local summary"


class TestReasoningModelHandling:
    """Tests for reasoning model parameters, <think> stripping, and empty-summary failover."""

    def test_qwen_payload_disables_reasoning(self):
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl.return_value.acquire.return_value = True
            mock_cb.return_value.call.side_effect = lambda f, *a, **kw: f(*a, **kw)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Temiz haber özeti."}}]
            }
            mock_post.return_value = mock_response

            summarizer = create_resilient_summarizer(api_key="gsk_test_key", model="qwen/qwen3.8-27b")
            result = summarizer.summarize("Başlık", "Haber içeriği burada.")

            assert result.success is True
            assert result.summary == "Temiz haber özeti."
            sent_payload = mock_post.call_args.kwargs["json"]
            assert sent_payload["reasoning_effort"] == "none"
            assert sent_payload["max_tokens"] >= 512

    def test_gpt_oss_payload_sets_low_reasoning_and_hides_reasoning(self):
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl.return_value.acquire.return_value = True
            mock_cb.return_value.call.side_effect = lambda f, *a, **kw: f(*a, **kw)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Temiz GPT-OSS özeti."}}]
            }
            mock_post.return_value = mock_response

            summarizer = create_resilient_summarizer(api_key="gsk_test_key", model="openai/gpt-oss-120b")
            result = summarizer.summarize("Başlık", "Haber içeriği burada.")

            assert result.success is True
            sent_payload = mock_post.call_args.kwargs["json"]
            assert sent_payload["reasoning_effort"] == "low"
            assert sent_payload["include_reasoning"] is False
            assert sent_payload["max_tokens"] >= 512

    def test_strips_think_tags_from_summary(self):
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl.return_value.acquire.return_value = True
            mock_cb.return_value.call.side_effect = lambda f, *a, **kw: f(*a, **kw)

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "<think>İçsel düşünme...</think>\n\nAsıl özet cümlesi."}}]
            }
            mock_post.return_value = mock_response

            summarizer = create_resilient_summarizer(api_key="gsk_test_key", model="qwen/qwen3.8-27b")
            result = summarizer.summarize("Başlık", "Haber içeriği burada.")

            assert result.success is True
            assert result.summary == "Asıl özet cümlesi."

    def test_empty_or_think_only_content_fails_over_to_next_model(self):
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl.return_value.acquire.return_value = True
            mock_cb.return_value.call.side_effect = lambda f, *a, **kw: f(*a, **kw)

            empty_resp = MagicMock()
            empty_resp.status_code = 200
            empty_resp.json.return_value = {"choices": [{"message": {"content": None}}]}

            think_only_resp = MagicMock()
            think_only_resp.status_code = 200
            think_only_resp.json.return_value = {
                "choices": [{"message": {"content": "<think>Yarım kalan düşünce"}}]
            }

            good_resp = MagicMock()
            good_resp.status_code = 200
            good_resp.json.return_value = {
                "choices": [{"message": {"content": "Üçüncü modelden gelen geçerli özet."}}]
            }

            mock_post.side_effect = [empty_resp, think_only_resp, good_resp]

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Başlık", "Haber içeriği burada.")

            assert result.success is True
            assert result.model == "qwen/qwen3.8-27b"
            assert result.summary == "Üçüncü modelden gelen geçerli özet."

    def test_all_empty_responses_fall_back_to_local_nlp(self):
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb, \
             patch("backend.summarizer.requests.post") as mock_post:

            mock_rl.return_value.acquire.return_value = True
            mock_cb.return_value.call.side_effect = lambda f, *a, **kw: f(*a, **kw)

            empty_resp = MagicMock()
            empty_resp.status_code = 200
            empty_resp.json.return_value = {"choices": [{"message": {"content": "   "}}]}
            mock_post.return_value = empty_resp

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Başlık", "Birinci cümle. İkinci cümle.")

            assert result.success is False
            assert "Yerel NLP" in result.engine
            assert len(result.summary) > 0