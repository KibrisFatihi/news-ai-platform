"""
Tests for the Groq service module.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests

from backend.circuit_breaker import CircuitOpenError
from backend.groq_service import (
    DEFAULT_MODEL,
    FAST_MODEL,
    GROQ_API_URL,
    get_groq_api_key,
    get_groq_status,
    summarize_with_groq,
)
from backend.summarizer import ResilientSummarizer, create_resilient_summarizer


class TestGetGroqApiKey:
    """Tests for get_groq_api_key function."""

    def test_get_groq_api_key_override(self):
        """Test override key takes precedence."""
        key = get_groq_api_key(override_key="test_override_key")
        assert key == "test_override_key"

    def test_get_groq_api_key_none(self):
        """Test returns None when no key provided."""
        # Temporarily remove env var
        import os
        old_key = os.environ.pop("GROQ_API_KEY", None)
        try:
            key = get_groq_api_key()
            assert key is None
        finally:
            if old_key:
                os.environ["GROQ_API_KEY"] = old_key

    def test_get_groq_api_key_env(self, monkeypatch):
        """Test reading from environment variable."""
        monkeypatch.setenv("GROQ_API_KEY", "gsk_env_key")
        key = get_groq_api_key()
        assert key == "gsk_env_key"

    def test_get_groq_api_key_strips_whitespace(self, monkeypatch):
        """Test that whitespace is stripped."""
        monkeypatch.setenv("GROQ_API_KEY", "  gsk_spaced_key  ")
        key = get_groq_api_key()
        assert key == "gsk_spaced_key"


class TestSummarizeWithGroq:
    """Tests for summarize_with_groq function."""

    @patch("backend.groq_service.get_groq_api_key")
    @patch("backend.summarizer.extractive_summarize")
    def test_summarize_no_api_key(self, mock_extractive, mock_get_key):
        """Test fallback when no API key."""
        mock_get_key.return_value = None
        mock_extractive.return_value = "Local summary"

        result = summarize_with_groq("Test Title", "Test content")

        assert result["success"] is True
        assert "Yerel NLP" in result["engine"]
        assert result["summary"] == "Local summary"
        mock_extractive.assert_called_once()

    @patch("backend.groq_service.get_groq_api_key")
    @patch("backend.summarizer.requests.post")
    @patch("backend.summarizer.get_groq_rate_limiter")
    @patch("backend.summarizer.get_circuit_breaker")
    def test_summarize_success(self, mock_cb, mock_rl, mock_post, mock_get_key):
        """Test successful Groq API call."""
        mock_get_key.return_value = "gsk_test_key"

        # Mock rate limiter
        mock_rl_instance = MagicMock()
        mock_rl_instance.acquire.return_value = True
        mock_rl.return_value = mock_rl_instance

        # Mock circuit breaker
        mock_cb_instance = MagicMock()
        mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
        mock_cb.return_value = mock_cb_instance

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Groq summary here"}}]
        }
        mock_post.return_value = mock_response

        result = summarize_with_groq("Test Title", "Test content", api_key="gsk_test_key")

        assert result["success"] is True
        assert "Groq" in result["engine"]
        assert result["summary"] == "Groq summary here"
        assert result["model"] == DEFAULT_MODEL

    @patch("backend.groq_service.get_groq_api_key")
    @patch("backend.summarizer.requests.post")
    @patch("backend.summarizer.get_groq_rate_limiter")
    @patch("backend.summarizer.get_circuit_breaker")
    @patch("backend.summarizer.extractive_summarize")
    def test_summarize_http_error_fallback(self, mock_extractive, mock_cb, mock_rl, mock_post, mock_get_key):
        """Test fallback on HTTP error."""
        mock_get_key.return_value = "gsk_test_key"

        mock_rl_instance = MagicMock()
        mock_rl_instance.acquire.return_value = True
        mock_rl.return_value = mock_rl_instance

        mock_cb_instance = MagicMock()
        mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
        mock_cb.return_value = mock_cb_instance

        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        mock_extractive.return_value = "Fallback summary"

        result = summarize_with_groq("Test Title", "Test content", api_key="gsk_test_key")

        assert result["success"] is False
        assert "Yedek Mod" in result["engine"]
        assert result["summary"] == "Fallback summary"

    @patch("backend.groq_service.get_groq_api_key")
    @patch("backend.summarizer.requests.post")
    @patch("backend.summarizer.get_groq_rate_limiter")
    @patch("backend.summarizer.get_circuit_breaker")
    @patch("backend.summarizer.extractive_summarize")
    def test_summarize_timeout_fallback(self, mock_extractive, mock_cb, mock_rl, mock_post, mock_get_key):
        """Test fallback on timeout."""
        mock_get_key.return_value = "gsk_test_key"

        mock_rl_instance = MagicMock()
        mock_rl_instance.acquire.return_value = True
        mock_rl.return_value = mock_rl_instance

        mock_cb_instance = MagicMock()
        mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
        mock_cb.return_value = mock_cb_instance

        # Mock timeout
        mock_post.side_effect = requests.Timeout("Request timed out")
        mock_extractive.return_value = "Timeout fallback"

        result = summarize_with_groq("Test Title", "Test content", api_key="gsk_test_key")

        assert result["success"] is False
        assert "Timeout" in result["engine"]
        assert result["summary"] == "Timeout fallback"

    @patch("backend.groq_service.get_groq_api_key")
    @patch("backend.summarizer.get_groq_rate_limiter")
    @patch("backend.summarizer.get_circuit_breaker")
    @patch("backend.summarizer.extractive_summarize")
    def test_summarize_circuit_open_fallback(self, mock_extractive, mock_cb, mock_rl, mock_get_key):
        """Test fallback when circuit breaker is open."""
        mock_get_key.return_value = "gsk_test_key"

        mock_rl_instance = MagicMock()
        mock_rl_instance.acquire.return_value = True
        mock_rl.return_value = mock_rl_instance

        # Circuit breaker raises CircuitOpenError
        mock_cb_instance = MagicMock()
        mock_cb_instance.call.side_effect = CircuitOpenError("Circuit open")
        mock_cb.return_value = mock_cb_instance

        mock_extractive.return_value = "Circuit breaker fallback"

        result = summarize_with_groq("Test Title", "Test content", api_key="gsk_test_key")

        assert result["success"] is False
        assert "Circuit Breaker" in result["engine"]
        assert result["summary"] == "Circuit breaker fallback"

    @patch("backend.groq_service.get_groq_api_key")
    @patch("backend.summarizer.get_groq_rate_limiter")
    @patch("backend.summarizer.extractive_summarize")
    def test_summarize_rate_limit_fallback(self, mock_extractive, mock_rl, mock_get_key):
        """Test fallback when rate limiter times out."""
        mock_get_key.return_value = "gsk_test_key"

        # Rate limiter returns False (timeout)
        mock_rl_instance = MagicMock()
        mock_rl_instance.acquire.return_value = False
        mock_rl.return_value = mock_rl_instance

        mock_extractive.return_value = "Rate limit fallback"

        result = summarize_with_groq("Test Title", "Test content", api_key="gsk_test_key")

        assert result["success"] is False
        assert "Rate Limit" in result["engine"]
        assert result["summary"] == "Rate limit fallback"

    def test_summarize_custom_model(self):
        """Test using custom model parameter."""
        with patch("backend.groq_service.get_groq_api_key") as mock_get_key, \
             patch("backend.summarizer.requests.post") as mock_post, \
             patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_get_key.return_value = "gsk_test_key"
            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Fast model summary"}}]
            }
            mock_post.return_value = mock_response

            # Pass explicit api_key to avoid config lookup
            result = summarize_with_groq("Title", "Content", api_key="gsk_test_key", model=FAST_MODEL)

            assert result["model"] == FAST_MODEL
            # Verify the model was passed to API
            call_args = mock_post.call_args
            assert call_args[1]["json"]["model"] == FAST_MODEL


class TestResilientSummarizer:
    """Tests for the new ResilientSummarizer class."""

    def test_summarize_no_api_key(self):
        """Test summarizer with no API key uses local NLP."""
        summarizer = create_resilient_summarizer(api_key=None)
        result = summarizer.summarize("Test Title", "Test content here. Another sentence.")

        assert result.success is True
        assert "Yerel NLP" in result.engine
        assert len(result.summary) > 0

    def test_summarize_with_mock_groq(self):
        """Test summarizer with mocked Groq."""
        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            with patch("backend.summarizer.requests.post") as mock_post:
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
                assert result.summary == "Groq summary"

    def test_summarize_fallback_chain(self):
        """Test fallback chain: rate_limit -> circuit -> timeout -> http -> unexpected."""
        from backend.summarizer import FALLBACK_CHAIN

        # Verify fallback chain order
        names = [s.name for s in FALLBACK_CHAIN]
        expected = [
            "rate_limit_timeout",
            "circuit_open",
            "timeout",
            "http_error",
            "unexpected_error"
        ]
        assert names == expected


class TestGetGroqStatus:
    """Tests for get_groq_status function."""

    def test_get_groq_status_structure(self):
        """Test that status has expected structure."""
        status = get_groq_status()

        assert "rate_limiter" in status
        assert "circuit_breaker" in status
        assert "api_key_configured" in status
        assert isinstance(status["api_key_configured"], bool)


class TestConstants:
    """Test exported constants."""

    def test_groq_api_url(self):
        assert GROQ_API_URL == "https://api.groq.com/openai/v1/chat/completions"

    def test_default_model(self):
        assert DEFAULT_MODEL == "llama-3.3-70b-versatile"

    def test_fast_model(self):
        assert FAST_MODEL == "llama-3.1-8b-instant"