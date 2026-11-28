"""
Tests for the API handlers module.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import time
import pytest

from backend.handlers import (
    handle_get_health,
    handle_get_news,
    handle_get_status,
    handle_not_found,
    handle_options,
    handle_post_refresh,
)
from backend.middleware import generate_request_id
from backend.models import Article, HealthResponse, NewsResponse, RefreshResponse, StatusResponse


class TestHandleGetNews:
    """Tests for handle_get_news handler."""

    def test_handle_get_news_empty(self):
        """Test handler with empty news cache."""
        request_id = generate_request_id()
        headers = {"Origin": "https://example.com"}

        with patch("backend.handlers.common.get_news_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.load_or_fallback.return_value = []
            mock_repo_instance.get_last_modified.return_value = None
            mock_repo.return_value = mock_repo_instance

            status, resp_headers, body = handle_get_news(headers, request_id)

            assert status == 200
            assert body["success"] is True
            assert body["count"] == 0
            assert body["articles"] == []
            # Check case-insensitive
            assert any(k.lower() == "access-control-allow-origin" for k in resp_headers)

    def test_handle_get_news_with_articles(self):
        """Test handler with articles."""
        request_id = generate_request_id()
        headers = {"Origin": "https://example.com"}

        articles = [
            {
                "id": "1",
                "title": "Test Haber",
                "source": "Test Source",
                "category": "Genel Gündem",
                "summary": "Özet",
                "content": "İçerik",
                "original_content": "İçerik",
                "link": "https://example.com",
                "ai_engine": "Yerel NLP",
                "ai_model": "TextRank",
                "published": "2026-01-01",
                "updated_at": "2026-01-01T12:00:00",
                "related_count": 0,
                "related_sources": []
            }
        ]

        with patch("backend.handlers.common.get_news_repository") as mock_repo, \
             patch("backend.handlers.common.cfg") as mock_cfg:

            mock_repo_instance = MagicMock()
            mock_repo_instance.load_or_fallback.return_value = articles
            mock_repo_instance.get_last_modified.return_value = 1767268800
            mock_repo.return_value = mock_repo_instance

            mock_cfg.get_groq_api_key.return_value = None

            status, resp_headers, body = handle_get_news(headers, request_id)

            assert status == 200
            assert body["success"] is True
            assert body["count"] == 1
            assert len(body["articles"]) == 1
            assert body["articles"][0]["title"] == "Test Haber"

    def test_handle_get_news_conditional_304(self):
        """Test 304 Not Modified response."""
        request_id = generate_request_id()
        headers = {
            "Origin": "https://example.com",
            "If-None-Match": '"abc123"'
        }

        articles = [{"title": "Test"}]

        with patch("backend.handlers.common.get_news_repository") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.load_or_fallback.return_value = articles
            mock_repo_instance.get_last_modified.return_value = 1767268800
            mock_repo.return_value = mock_repo_instance

            with patch("backend.middleware.compute_etag", return_value="abc123"):
                status, resp_headers, body = handle_get_news(headers, request_id)

                assert status == 304
                assert body == {}


class TestHandleGetStatus:
    """Tests for handle_get_status handler."""

    def test_handle_get_status(self):
        """Test status handler."""
        request_id = generate_request_id()
        headers = {"Origin": "https://example.com"}

        with patch("backend.handlers.common.cfg") as mock_cfg:
            mock_cfg.get_groq_api_key.return_value = "gsk_test"
            mock_cfg.groq.default_model = "llama-3.3-70b-versatile"

            with patch("backend.handlers.common._server_start_time", 1767268800):
                with patch("time.time", return_value=1767268900):
                    status, resp_headers, body = handle_get_status(headers, request_id)

                    assert status == 200
                    assert body["status"] == "online"
                    assert body["groq_connected"] is True
                    assert body["model"] == "llama-3.3-70b-versatile"


class TestHandleGetHealth:
    """Tests for handle_get_health handler."""

    def test_handle_get_health_healthy(self):
        """Test health handler when healthy."""
        request_id = generate_request_id()
        headers = {"Origin": "https://example.com"}

        with patch("backend.handlers.common.get_news_repository") as mock_repo, \
             patch("backend.groq_service.get_groq_status") as mock_groq_status, \
             patch("backend.circuit_breaker.get_all_circuit_breakers") as mock_cb:

            mock_repo_instance = MagicMock()
            mock_repo_instance.load_or_fallback.return_value = [{"title": "Test"}]
            mock_repo_instance.get_last_modified.return_value = 1767268800
            mock_repo.return_value = mock_repo_instance

            mock_groq_status.return_value = {
                "rate_limiter": {"windows": []},
                "circuit_breaker": {"state": "closed"},
                "api_key_configured": True
            }

            mock_cb.return_value = {
                "groq_api": {"state": "closed"}
            }

            with patch("backend.handlers.common._server_start_time", 1767268800):
                with patch("time.time", return_value=1767268900):
                    status, resp_headers, body = handle_get_health(headers, request_id)

                    assert status == 200
                    assert body["status"] == "healthy"
                    assert "components" in body


class TestHandlePostRefresh:
    """Tests for handle_post_refresh handler."""

    def test_handle_post_refresh_success(self):
        """Test successful refresh — pipeline runs in background, returns 202."""
        request_id = generate_request_id()
        headers = {
            "Origin": "https://example.com",
            "X-Refresh-Token": "valid_token",
            "Content-Type": "application/json"
        }
        body = b'{"api_key": "gsk_test"}'
        client_ip = "127.0.0.1"

        with patch("backend.middleware.verify_refresh_token", return_value=True), \
             patch("backend.middleware.check_refresh_rate_limit", return_value=(True, 0)), \
             patch("main.run_pipeline") as mock_pipeline, \
             patch("backend.handlers._refresh_lock", MagicMock()):

            mock_pipeline.return_value = []

            status, resp_headers, resp_body = handle_post_refresh(headers, body, client_ip, request_id)

            assert status == 202
            assert resp_body["success"] is True
            assert resp_body["status"] == "running"

            # Background thread needs a moment to start and call run_pipeline
            time.sleep(0.5)
            mock_pipeline.assert_called_once_with(api_key="gsk_test")

    def test_handle_post_refresh_invalid_token(self):
        """Test refresh with invalid token."""
        request_id = generate_request_id()
        headers = {
            "Origin": "https://example.com",
            "X-Refresh-Token": "invalid_token"
        }
        body = b"{}"
        client_ip = "127.0.0.1"

        with patch("backend.middleware.verify_refresh_token", return_value=False):
            status, resp_headers, resp_body = handle_post_refresh(headers, body, client_ip, request_id)

            assert status == 401
            assert resp_body["success"] is False

    def test_handle_post_refresh_rate_limited(self):
        """Test refresh rate limited."""
        request_id = generate_request_id()
        headers = {
            "X-Refresh-Token": "valid_token"
        }
        body = b"{}"
        client_ip = "127.0.0.1"

        with patch("backend.middleware.verify_refresh_token", return_value=True), \
             patch("backend.middleware.check_refresh_rate_limit", return_value=(False, 60)):
            status, resp_headers, resp_body = handle_post_refresh(headers, body, client_ip, request_id)

            assert status == 429
            assert resp_body["success"] is False
            assert resp_headers.get("Retry-After") == "60"

    def test_handle_post_refresh_concurrent_blocked(self):
        """Test concurrent refresh blocked."""
        request_id = generate_request_id()
        headers = {
            "Origin": "https://example.com",
            "X-Refresh-Token": "valid_token"
        }
        body = b"{}"
        client_ip = "127.0.0.1"

        mock_lock = MagicMock()
        mock_lock.acquire.return_value = False

        with patch("backend.middleware.verify_refresh_token", return_value=True), \
             patch("backend.middleware.check_refresh_rate_limit", return_value=(True, 0)), \
             patch("backend.handlers.refresh._refresh_lock", mock_lock):
            status, resp_headers, resp_body = handle_post_refresh(headers, body, client_ip, request_id)

            assert status == 409
            assert resp_body["success"] is False


class TestHandleOptions:
    """Tests for handle_options handler."""

    def test_handle_options(self):
        """Test OPTIONS preflight handler."""
        headers = {"Origin": "https://example.com"}

        status, resp_headers, body = handle_options(headers)

        assert status == 200
        # Check case-insensitive
        assert any(k.lower() == "access-control-allow-origin" for k in resp_headers)
        assert any(k.lower() == "access-control-allow-methods" for k in resp_headers)
        assert any(k.lower() == "access-control-allow-headers" for k in resp_headers)
        assert body == {}


class TestHandleNotFound:
    """Tests for handle_not_found handler."""

    def test_handle_not_found(self):
        """Test 404 handler."""
        request_id = generate_request_id()
        headers = {"Origin": "https://example.com"}

        status, resp_headers, body = handle_not_found(headers, request_id)

        assert status == 404
        assert body["success"] is False
        assert body["error"] == "Endpoint not found"
        assert "request_id" in body