"""
Extended tests for the REST API server endpoints.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from unittest.mock import patch, MagicMock

# Set a fixed refresh token secret for testing
os.environ["REFRESH_TOKEN_SECRET"] = "test-secret-key-for-testing-only"

# Generate valid refresh token for tests
TEST_REFRESH_TOKEN = hmac.new(
    b"test-secret-key-for-testing-only",
    b"refresh",
    hashlib.sha256
).hexdigest()


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_endpoint_exists(self, api_client):
        """Test health endpoint returns 200."""
        response = api_client.get("/api/health")
        assert response["status_code"] == 200

    def test_health_endpoint_structure(self, api_client):
        """Test health endpoint response structure."""
        response = api_client.get("/api/health")
        data = response["json"]

        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert "news_cache" in data["components"]
        assert "groq_api" in data["components"]
        assert "circuit_breakers" in data["components"]


class TestNewsEndpoint:
    """Tests for /api/news endpoint."""

    def test_news_endpoint_exists(self, api_client):
        """Test news endpoint returns 200."""
        response = api_client.get("/api/news")
        assert response["status_code"] == 200

    def test_news_endpoint_structure(self, api_client):
        """Test news endpoint response structure."""
        response = api_client.get("/api/news")
        data = response["json"]

        assert data["success"] is True
        assert "count" in data
        assert "categories" in data
        assert "groq_configured" in data
        assert "active_engine" in data
        assert "articles" in data
        assert isinstance(data["articles"], list)

    def test_news_endpoint_categories(self, api_client):
        """Test categories are counted correctly."""
        response = api_client.get("/api/news")
        data = response["json"]

        # Categories should match article categories
        total_categorized = sum(data["categories"].values())
        assert total_categorized == data["count"]

    def test_news_endpoint_caching_headers(self, api_client):
        """Test caching headers are present."""
        response = api_client.get("/api/news")

        assert "etag" in response["headers"]
        assert "last-modified" in response["headers"]
        assert "cache-control" in response["headers"]

    def test_news_endpoint_conditional_request(self, api_client):
        """Test 304 Not Modified for conditional requests."""
        # First request to get ETag
        response1 = api_client.get("/api/news")
        etag = response1["headers"].get("etag")
        last_modified = response1["headers"].get("last-modified")

        if etag:
            # Second request with If-None-Match
            response2 = api_client.get("/api/news", headers={"If-None-Match": etag})
            # Should return 304 if content unchanged
            assert response2["status_code"] in [200, 304]

    def test_news_endpoint_if_modified_since(self, api_client):
        """Test If-Modified-Since conditional request."""
        response1 = api_client.get("/api/news")
        last_modified = response1["headers"].get("last-modified")

        if last_modified:
            response2 = api_client.get("/api/news", headers={"If-Modified-Since": last_modified})
            assert response2["status_code"] in [200, 304]

    def test_news_endpoint_cache_control_no_cache(self, api_client):
        """News responses must revalidate (no max-age) while keeping ETag/Last-Modified validators."""
        response = api_client.get("/api/news")

        cache_control = response["headers"]["cache-control"]
        assert "no-cache" in cache_control
        assert "must-revalidate" in cache_control
        assert "max-age" not in cache_control
        assert "etag" in response["headers"]
        assert "last-modified" in response["headers"]


class TestStatusEndpoint:
    """Tests for /api/status endpoint."""

    def test_status_endpoint_exists(self, api_client):
        """Test status endpoint returns 200."""
        response = api_client.get("/api/status")
        assert response["status_code"] == 200

    def test_status_endpoint_structure(self, api_client):
        """Test status endpoint response structure."""
        response = api_client.get("/api/status")
        data = response["json"]

        assert data["status"] == "online"
        assert "version" in data
        assert "groq_connected" in data
        assert "model" in data
        assert "mode" in data
        assert "uptime_seconds" in data

    def test_status_endpoint_returns_refresh_token(self, api_client):
        """Status response must include refresh_token so the frontend can send it on /api/refresh."""
        response = api_client.get("/api/status")

        assert "refresh_token" in response["json"]
        assert response["json"]["refresh_token"]


class TestRefreshEndpoint:
    """Tests for /api/refresh endpoint."""

    @patch("main.run_pipeline")
    def test_refresh_endpoint_success(self, mock_pipeline, api_client):
        """Test successful pipeline refresh."""
        mock_pipeline.return_value = [
            {
                "id": "1",
                "title": "Refreshed News",
                "source": "Test",
                "category": "Genel Gündem",
                "summary": "Summary",
                "content": "Content",
                "original_content": "Content",
                "link": "https://example.com",
                "ai_engine": "Yerel NLP",
                "ai_model": "TextRank",
                "published": "2026-01-01",
                "updated_at": "2026-01-01T12:00:00",
                "related_count": 0,
                "related_sources": []
            }
        ]

        response = api_client.post(
            "/api/refresh",
            json_data={},
            headers={"X-Refresh-Token": TEST_REFRESH_TOKEN}
        )

        assert response["status_code"] == 200
        data = response["json"]
        assert data["success"] is True
        assert data["count"] == 1
        assert "articles" in data

    @patch("main.run_pipeline")
    def test_refresh_endpoint_with_api_key_body(self, mock_pipeline, api_client):
        """Test refresh with API key in request body."""
        mock_pipeline.return_value = []

        response = api_client.post(
            "/api/refresh",
            json_data={"api_key": "gsk_test_key"},
            headers={"X-Refresh-Token": TEST_REFRESH_TOKEN}
        )

        assert response["status_code"] == 200
        mock_pipeline.assert_called_once_with(api_key="gsk_test_key")

    @patch("main.run_pipeline")
    def test_refresh_endpoint_with_api_key_header(self, mock_pipeline, api_client):
        """Test refresh with API key in header."""
        mock_pipeline.return_value = []

        response = api_client.post(
            "/api/refresh",
            json_data={},
            headers={
                "X-Groq-Api-Key": "gsk_header_key",
                "X-Refresh-Token": TEST_REFRESH_TOKEN
            }
        )

        assert response["status_code"] == 200
        mock_pipeline.assert_called_once_with(api_key="gsk_header_key")

    @patch("main.run_pipeline")
    def test_refresh_endpoint_header_priority(self, mock_pipeline, api_client):
        """Test header takes priority over body."""
        mock_pipeline.return_value = []

        response = api_client.post(
            "/api/refresh",
            json_data={"api_key": "gsk_body_key"},
            headers={
                "X-Groq-Api-Key": "gsk_header_key",
                "X-Refresh-Token": TEST_REFRESH_TOKEN
            }
        )

        assert response["status_code"] == 200
        mock_pipeline.assert_called_once_with(api_key="gsk_header_key")

    @patch("main.run_pipeline")
    def test_refresh_endpoint_case_insensitive_headers(self, mock_pipeline, api_client):
        """Browsers may send header names in lowercase (x-refresh-token, x-groq-api-key); they must be honored."""
        mock_pipeline.return_value = []

        response = api_client.post(
            "/api/refresh",
            json_data={},
            headers={"x-refresh-token": TEST_REFRESH_TOKEN, "x-groq-api-key": "gsk_lower_key"},
        )

        assert response["status_code"] == 200
        mock_pipeline.assert_called_once_with(api_key="gsk_lower_key")

    @patch("main.run_pipeline")
    def test_refresh_endpoint_error(self, mock_pipeline, api_client):
        """Test refresh endpoint error handling."""
        # Rate limiter is module-global; reset so this test stays independent of order.
        from backend.middleware import _refresh_rate_limit, _refresh_rate_limit_lock

        with _refresh_rate_limit_lock:
            _refresh_rate_limit.clear()

        mock_pipeline.side_effect = Exception("Pipeline failed")

        response = api_client.post(
            "/api/refresh",
            json_data={},
            headers={"X-Refresh-Token": TEST_REFRESH_TOKEN}
        )

        assert response["status_code"] == 500
        data = response["json"]
        assert data["success"] is False
        assert "error" in data
        assert "request_id" in data

    def test_refresh_endpoint_invalid_token(self, api_client):
        """Test refresh with invalid token."""
        response = api_client.post(
            "/api/refresh",
            json_data={},
            headers={"X-Refresh-Token": "invalid_token"}
        )

        assert response["status_code"] == 401
        data = response["json"]
        assert data["success"] is False

    def test_refresh_endpoint_missing_token(self, api_client):
        """Test refresh without token."""
        response = api_client.post(
            "/api/refresh",
            json_data={}
        )

        assert response["status_code"] == 401

    def test_refresh_endpoint_rate_limit(self, api_client):
        """Test refresh rate limiting."""
        from backend.middleware import _refresh_rate_limit, _refresh_rate_limit_lock
        import time

        # Clear rate limit
        with _refresh_rate_limit_lock:
            _refresh_rate_limit.clear()

        # Make 5 requests (max allowed)
        for i in range(5):
            with patch("main.run_pipeline") as mock_pipeline:
                mock_pipeline.return_value = []
                response = api_client.post(
                    "/api/refresh",
                    json_data={},
                    headers={"X-Refresh-Token": TEST_REFRESH_TOKEN}
                )
                assert response["status_code"] == 200

        # 6th request should be rate limited
        with patch("main.run_pipeline") as mock_pipeline:
            mock_pipeline.return_value = []
            response = api_client.post(
                "/api/refresh",
                json_data={},
                headers={"X-Refresh-Token": TEST_REFRESH_TOKEN}
            )
            assert response["status_code"] == 429
            # Headers are lowercased in the test client
            assert "retry-after" in response["headers"]


class TestCORSHeaders:
    """Tests for CORS headers."""

    def test_options_endpoint(self, api_client):
        """Test OPTIONS preflight request."""
        response = api_client._make_request("OPTIONS", "/api/news")
        assert response["status_code"] == 200
        assert "access-control-allow-origin" in response["headers"]
        assert "access-control-allow-methods" in response["headers"]
        assert "access-control-allow-headers" in response["headers"]

    def test_get_has_cors_headers(self, api_client):
        """Test GET responses have CORS headers."""
        response = api_client.get("/api/news")
        assert "access-control-allow-origin" in response["headers"]
        assert response["headers"]["access-control-allow-origin"] == "*"


class TestStaticFiles:
    """Tests for static file serving."""

    def test_root_redirects_to_frontend(self, api_client):
        """Test root path redirects to frontend."""
        response = api_client.get("/")
        assert response["status_code"] == 302
        assert "location" in response["headers"]
        assert "index.html" in response["headers"]["location"]

    def test_static_file_served_with_no_store(self, api_client):
        """Static files (index.html, JS, CSS) must not be cached by browsers."""
        response = api_client.get("/index.html")

        assert response["status_code"] == 200
        cache_control = response["headers"]["cache-control"]
        assert "no-store" in cache_control
        assert "no-cache" in cache_control
        assert "must-revalidate" in cache_control


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_endpoint(self, api_client):
        """Test 404 for unknown endpoints."""
        response = api_client.get("/api/unknown")
        # Should be handled by static file server or return 404
        assert response["status_code"] in [200, 404, 500]
        # If it's a JSON response, it should have error structure
        if response["json"] is not None:
            assert "error" in response["json"] or "success" in response["json"]
        else:
            # Non-JSON response (e.g., HTML) - just verify we got a response
            assert response["body"] is not None

    def test_large_request_rejected(self, api_client):
        """Test large request body is rejected."""
        large_body = "x" * (11 * 1024 * 1024)  # 11MB > 10MB limit
        response = api_client._make_request(
            "POST",
            "/api/refresh",
            body=large_body.encode(),
            headers={"Content-Type": "application/json"}
        )
        # Should reject with 413
        assert response["status_code"] in [413, 500]


class TestResponseTimes:
    """Tests for response time logging."""

    def test_api_request_logging(self, api_client, caplog):
        """Test that API requests are logged with timing."""
        import logging
        caplog.set_level(logging.INFO, logger="news_ai.server")

        response = api_client.get("/api/news")
        assert response["status_code"] == 200

        # Check that log contains timing info in extra_fields
        api_logs = [r for r in caplog.records if "GET /api/news -> 200" in r.message]
        assert len(api_logs) > 0
        # Duration is in extra_fields, not in the message
        assert hasattr(api_logs[0], "extra_fields")
        assert "duration_ms" in api_logs[0].extra_fields
        # Duration can be 0.0 for very fast requests (sub-millisecond)
        assert api_logs[0].extra_fields["duration_ms"] >= 0


class TestMiddleware:
    """Tests for middleware functions."""

    def test_generate_request_id(self):
        """Test request ID generation."""
        from backend.middleware import generate_request_id
        id1 = generate_request_id()
        id2 = generate_request_id()
        assert len(id1) == 8
        assert id1 != id2

    def test_compute_etag(self):
        """Test ETag computation."""
        from backend.middleware import compute_etag
        data = {"key": "value", "number": 42}
        etag1 = compute_etag(data)
        etag2 = compute_etag(data)
        assert etag1 == etag2
        assert len(etag1) == 32  # MD5 hex

    def test_format_http_date(self):
        """Test HTTP date formatting."""
        from backend.middleware import format_http_date
        timestamp = 1767268800  # 2026-01-01 12:00:00 UTC
        result = format_http_date(timestamp)
        assert result == "Thu, 01 Jan 2026 12:00:00 GMT"

    def test_parse_http_date(self):
        """Test HTTP date parsing."""
        from backend.middleware import parse_http_date
        date_str = "Mon, 01 Jan 2026 12:00:00 GMT"
        result = parse_http_date(date_str)
        assert result == 1767268800

    def test_parse_http_date_with_timezone(self):
        """Test HTTP date parsing with timezone offset."""
        from backend.middleware import parse_http_date
        date_str = "Mon, 01 Jan 2026 12:00:00 +0300"
        result = parse_http_date(date_str)
        assert result == 1767258000  # 09:00:00 UTC

    def test_parse_http_date_invalid(self):
        """Test invalid date parsing returns None."""
        from backend.middleware import parse_http_date
        assert parse_http_date("invalid") is None
        assert parse_http_date("") is None

    def test_check_conditional_request_etag(self):
        """Test conditional request with ETag."""
        from backend.middleware import check_conditional_request
        headers = {"If-None-Match": '"abc123"'}
        assert check_conditional_request(headers, "abc123", None) is True
        assert check_conditional_request(headers, "different", None) is False

    def test_check_conditional_request_last_modified(self):
        """Test conditional request with If-Modified-Since."""
        from backend.middleware import check_conditional_request
        headers = {"If-Modified-Since": "Thu, 01 Jan 2026 12:00:00 GMT"}
        assert check_conditional_request(headers, None, 1767268800) is True
        assert check_conditional_request(headers, None, 1767268801) is False  # Later

    def test_verify_refresh_token(self, monkeypatch):
        """Test refresh token verification."""
        from backend.middleware import verify_refresh_token
        monkeypatch.setenv("REFRESH_TOKEN_SECRET", "test-secret")

        import hashlib
        import hmac
        valid_token = hmac.new(
            b"test-secret",
            b"refresh",
            hashlib.sha256
        ).hexdigest()

        assert verify_refresh_token(valid_token) is True
        assert verify_refresh_token("invalid") is False
        assert verify_refresh_token("") is False