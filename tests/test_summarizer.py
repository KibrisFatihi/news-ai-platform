"""
Tests for the Summarizer module (ResilientSummarizer).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from backend.circuit_breaker import CircuitOpenError
from backend.summarizer import (
    ResilientSummarizer,
    create_resilient_summarizer,
    FallbackStrategy,
    FALLBACK_CHAIN,
    GroqSummarizer,
    LocalNLPSummarizer,
    SummarizeResult,
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


class TestGroqSummarizer:
    """Tests for GroqSummarizer."""

    @patch("backend.summarizer.requests.post")
    def test_summarize_success(self, mock_post):
        """Test successful Groq API call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Groq summary here"}}]
        }
        mock_post.return_value = mock_response

        # Mock rate limiter and circuit breaker
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            summarizer = GroqSummarizer(
                api_key="gsk_test_key",
                model="llama-3.3-70b-versatile",
                groq_config=MagicMock()
            )
            result = summarizer.summarize("Test Title", "Test content")

            assert result.success is True
            assert "Groq" in result.engine
            assert result.summary == "Groq summary here"

    @patch("backend.summarizer.requests.post")
    def test_summarize_http_error(self, mock_post):
        """Test HTTP error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            summarizer = GroqSummarizer(
                api_key="gsk_test_key",
                model="llama-3.3-70b-versatile",
                groq_config=MagicMock()
            )

            with pytest.raises(requests.HTTPError):
                summarizer.summarize("Test Title", "Test content")

    @patch("backend.summarizer.requests.post")
    def test_summarize_timeout(self, mock_post):
        """Test timeout handling."""
        mock_post.side_effect = requests.Timeout("Request timed out")

        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            summarizer = GroqSummarizer(
                api_key="gsk_test_key",
                model="llama-3.3-70b-versatile",
                groq_config=MagicMock()
            )

            with pytest.raises(requests.Timeout):
                summarizer.summarize("Test Title", "Test content")


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