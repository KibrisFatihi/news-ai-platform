"""
Tests for the Rate Limiter module.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from backend.rate_limiter import RateLimitWindow, RateLimiter, get_groq_rate_limiter, reset_groq_rate_limiter


# pytest-asyncio is no longer needed since AsyncRateLimiter was removed
# pytest_plugins = ["pytest_asyncio"]


class TestRateLimitWindow:
    """Tests for RateLimitWindow class."""

    def test_acquire_success(self):
        """Test successful slot acquisition."""
        window = RateLimitWindow(max_requests=5, window_seconds=60)
        assert window.acquire(1) is True
        assert window.acquire(2) is True
        assert len(window.requests) == 3

    def test_acquire_failure_when_full(self):
        """Test failure when window is full."""
        window = RateLimitWindow(max_requests=2, window_seconds=60)
        assert window.acquire(1) is True
        assert window.acquire(1) is True
        assert window.acquire(1) is False  # Window full

    def test_acquire_cleans_expired(self):
        """Test that expired entries are cleaned on acquire."""
        window = RateLimitWindow(max_requests=5, window_seconds=1)
        # Add old entry
        window.requests.append(time.time() - 10)
        assert window.acquire(1) is True
        assert len(window.requests) == 1  # Old entry cleaned

    def test_get_remaining(self):
        """Test getting remaining slots."""
        window = RateLimitWindow(max_requests=5, window_seconds=60)
        window.acquire(2)
        assert window.get_remaining() == 3

    def test_get_reset_time(self):
        """Test getting reset time."""
        window = RateLimitWindow(max_requests=5, window_seconds=60)
        assert window.get_reset_time() is None

        window.acquire(1)
        reset_time = window.get_reset_time()
        assert reset_time is not None
        assert reset_time > time.time()

    def test_wait_time(self):
        """Test wait time calculation."""
        window = RateLimitWindow(max_requests=1, window_seconds=60)
        assert window.wait_time() == 0.0

        window.acquire(1)
        wait = window.wait_time()
        assert wait > 0
        assert wait <= 60


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_acquire_non_blocking(self):
        """Test non-blocking acquire."""
        limiter = RateLimiter(rpm=2, rpd=1000)
        assert limiter.acquire(1, blocking=False) is True
        assert limiter.acquire(1, blocking=False) is True
        assert limiter.acquire(1, blocking=False) is False  # RPM exceeded

    def test_acquire_blocking_success(self):
        """Test blocking acquire succeeds when slot available."""
        limiter = RateLimiter(rpm=5, rpd=1000)
        assert limiter.acquire(1, blocking=True, timeout=1.0) is True

    def test_acquire_blocking_timeout(self):
        """Test blocking acquire times out."""
        limiter = RateLimiter(rpm=1, rpd=1000)
        limiter.acquire(1, blocking=False)  # Use the only slot

        # Should timeout waiting for slot
        result = limiter.acquire(1, blocking=True, timeout=0.1)
        assert result is False

    def test_get_status(self):
        """Test status reporting."""
        limiter = RateLimiter(rpm=30, rpd=100)
        limiter.acquire(5, blocking=False)
        limiter.acquire(3, blocking=False)

        status = limiter.get_status()
        assert "windows" in status
        assert len(status["windows"]) == 2
        assert status["total_requests"] == 8

    def test_reset(self):
        """Test reset clears all windows."""
        limiter = RateLimiter(rpm=5, rpd=100)
        limiter.acquire(3, blocking=False)

        limiter.reset()

        status = limiter.get_status()
        assert status["windows"][0]["current_usage"] == 0
        assert status["windows"][1]["current_usage"] == 0
        assert status["total_requests"] == 0

    def test_get_retry_after(self):
        """Test Retry-After header value."""
        limiter = RateLimiter(rpm=1, rpd=1000)
        limiter.acquire(1, blocking=False)

        retry_after = limiter.get_retry_after()
        assert retry_after >= 1
        assert retry_after <= 61  # Window + 1 second buffer


class TestRateLimiterGlobal:
    """Tests for global rate limiter instance."""

    def test_get_groq_rate_limiter_singleton(self):
        """Test that get_groq_rate_limiter returns singleton."""
        reset_groq_rate_limiter()
        limiter1 = get_groq_rate_limiter()
        limiter2 = get_groq_rate_limiter()
        assert limiter1 is limiter2

    def test_reset_groq_rate_limiter(self):
        """Test reset creates new instance."""
        reset_groq_rate_limiter()
        limiter1 = get_groq_rate_limiter()
        reset_groq_rate_limiter()
        limiter2 = get_groq_rate_limiter()
        assert limiter1 is not limiter2


class TestRateLimiterIntegration:
    """Integration tests for rate limiter with Groq service."""

    @patch("backend.summarizer.requests.post")
    def test_rate_limiter_integration_success(self, mock_post):
        """Test rate limiter works with summarizer."""
        from backend.summarizer import create_resilient_summarizer

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Groq summary"}}]
        }
        mock_post.return_value = mock_response

        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = True
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb_instance.call.side_effect = lambda f, *a, **kw: f(*a, **kw)
            mock_cb.return_value = mock_cb_instance

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test", "Content here. More content.")

            assert result.success is True
            mock_rl_instance.acquire.assert_called_once()

    @patch("backend.summarizer.requests.post")
    def test_rate_limiter_integration_timeout(self, mock_post):
        """Test rate limiter timeout triggers fallback."""
        from backend.summarizer import create_resilient_summarizer

        with patch("backend.summarizer.get_groq_rate_limiter") as mock_rl, \
             patch("backend.summarizer.get_circuit_breaker") as mock_cb:

            mock_rl_instance = MagicMock()
            mock_rl_instance.acquire.return_value = False  # Timeout
            mock_rl.return_value = mock_rl_instance

            mock_cb_instance = MagicMock()
            mock_cb.return_value = mock_cb_instance

            summarizer = create_resilient_summarizer(api_key="gsk_test_key")
            result = summarizer.summarize("Test", "Content here. More content.")

            assert result.success is False
            assert "Rate Limit" in result.engine


