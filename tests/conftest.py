"""
Pytest configuration and fixtures for News AI Platform tests.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_article() -> dict[str, Any]:
    """Return a sample article for testing."""
    return {
        "id": "test_1",
        "source": "Test Source",
        "title": "Test Haber Başlığı",
        "content": "Bu bir test haberidir. İçerik burada yer alır.",
        "link": "https://example.com/test",
        "published": "2026-01-01"
    }


@pytest.fixture
def sample_articles() -> list:
    """Return a list of sample articles for testing."""
    return [
        {
            "id": "test_1",
            "source": "BBC Türkçe",
            "title": "Yapay Zeka Gelişmeleri Hızla Devam Ediyor",
            "content": "Yapay zeka teknolojileri hızla gelişiyor. Yeni modeller piyasaya sürülüyor.",
            "link": "https://bbc.com/test1",
            "published": "2026-01-01"
        },
        {
            "id": "test_2",
            "source": "TRT Haber",
            "title": "Ekonomi Bakanı Faiz Kararını Açıkladı",
            "content": "Merkez Bankası faiz oranlarını değiştirmedi. Ekonomi bakanı açıklama yaptı.",
            "link": "https://trt.com/test2",
            "published": "2026-01-01"
        },
        {
            "id": "test_3",
            "source": "NTV",
            "title": "Savunma Sanayii Yeni Projeyi Başlattı",
            "content": "Yeni bir siha projesi başlatıldı. Savunma sanayii gelişiyor.",
            "link": "https://ntv.com/test3",
            "published": "2026-01-01"
        }
    ]


@pytest.fixture
def mock_rss_feed() -> dict[str, Any]:
    """Return a mock RSS feed response."""
    return {
        "entries": [
            {
                "title": "Test Haber 1",
                "summary": "Bu test haberinin özeti.",
                "link": "https://example.com/1",
                "published": "Mon, 01 Jan 2026 12:00:00 GMT"
            },
            {
                "title": "Test Haber 2",
                "summary": "İkinci test haberi.",
                "link": "https://example.com/2",
                "published": "Mon, 01 Jan 2026 13:00:00 GMT"
            }
        ]
    }


@pytest.fixture
def temp_news_feed(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary news_feed.json file for testing."""
    feed_data = [
        {
            "title": "Cached Haber",
            "source": "Test Source",
            "category": "Test",
            "summary": "Cached summary",
            "original_content": "Cached content",
            "link": "https://example.com/cached",
            "ai_engine": "Yerel NLP",
            "ai_model": "TextRank",
            "published": "2026-01-01",
            "updated_at": "12:00:00",
            "related_count": 0,
            "related_sources": []
        }
    ]
    feed_file = tmp_path / "news_feed.json"
    with open(feed_file, "w", encoding="utf-8") as f:
        json.dump(feed_data, f)
    yield feed_file


@pytest.fixture
def mock_groq_response() -> dict[str, Any]:
    """Return a mock Groq API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Bu bir test özetidir. İkinci cümle de buradadır."
                }
            }
        ]
    }


@pytest.fixture
def api_client():
    """Create a test client for the API server."""
    import io

    from server import NewsAPIRequestHandler

    class TestClient:
        def __init__(self):
            self.handler_class = NewsAPIRequestHandler

        def _make_request(self, method: str, path: str, body: bytes = b"", headers: dict = None):
            """Simulate an HTTP request."""
            headers = headers or {}
            request_line = f"{method} {path} HTTP/1.1"
            request_headers = "\r\n".join([f"{k}: {v}" for k, v in headers.items()])
            if body:
                request_headers += f"\r\nContent-Length: {len(body)}"
            request_data = f"{request_line}\r\n{request_headers}\r\n\r\n".encode() + body

            # Create mock socket
            sock = MagicMock()
            sock.makefile.return_value = io.BytesIO(request_data)
            sent_data = []

            def capture_sendall(data):
                sent_data.append(data)

            sock.sendall = capture_sendall

            # Create handler
            handler = self.handler_class(sock, ("127.0.0.1", 8000), None)
            handler.setup()
            try:
                handler.handle()
            except Exception:
                pass

            # Parse response
            response_bytes = b"".join(sent_data)
            return self._parse_response(response_bytes)

        def _parse_response(self, data: bytes):
            """Parse HTTP response."""
            lines = data.split(b"\r\n")
            status_line = lines[0].decode()
            status_code = int(status_line.split()[1])

            # Find body
            body_start = 0
            for i, line in enumerate(lines):
                if line == b"":
                    body_start = i + 1
                    break

            headers = {}
            for line in lines[1:body_start]:
                if b":" in line:
                    k, v = line.split(b":", 1)
                    headers[k.decode().lower()] = v.decode().strip()

            body = b"\r\n".join(lines[body_start:])
            # Try to parse as JSON, but don't fail if it's not JSON
            try:
                json_data = json.loads(body) if body else None
            except json.JSONDecodeError:
                json_data = None
            return {
                "status_code": status_code,
                "headers": headers,
                "body": body,
                "json": json_data
            }

        def get(self, path: str, headers: dict = None):
            return self._make_request("GET", path, headers=headers)

        def post(self, path: str, json_data: dict = None, headers: dict = None):
            body = json.dumps(json_data or {}).encode() if json_data else b""
            h = headers or {}
            h["Content-Type"] = "application/json"
            return self._make_request("POST", path, body=body, headers=h)

    return TestClient()


@pytest.fixture
def temp_repo_path(tmp_path: Path) -> Path:
    """Create a temporary path for repository testing."""
    return tmp_path / "test_news_feed.json"


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the global rate limiter before and after every test."""
    from backend.rate_limiter import reset_groq_rate_limiter
    reset_groq_rate_limiter()
    yield
    reset_groq_rate_limiter()


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset all circuit breakers before and after every test."""
    from backend.circuit_breaker import reset_all_circuit_breakers
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


@pytest.fixture(autouse=True)
def reset_refresh_state():
    """Reset refresh endpoint rate limit, lock, and token secret state before and after every test."""
    import os

    import backend.handlers as handlers_module
    from backend.middleware import _refresh_rate_limit, _refresh_rate_limit_lock

    with _refresh_rate_limit_lock:
        _refresh_rate_limit.clear()

    if handlers_module._refresh_lock.locked():
        handlers_module._refresh_lock.release()
    handlers_module._refresh_in_progress = False

    # Restore the test refresh token secret so token-computation tests stay
    # isolated from any test that mutates os.environ["REFRESH_TOKEN_SECRET"]
    # without reverting it (defense in depth).
    os.environ["REFRESH_TOKEN_SECRET"] = "test-secret-key-for-testing-only"

    yield

    with _refresh_rate_limit_lock:
        _refresh_rate_limit.clear()

    if handlers_module._refresh_lock.locked():
        handlers_module._refresh_lock.release()
    handlers_module._refresh_in_progress = False

    os.environ["REFRESH_TOKEN_SECRET"] = "test-secret-key-for-testing-only"


@pytest.fixture
def reset_vector_cache():
    """Clear the vector cache before and after test."""
    from backend.nlp_engine import clear_vector_cache
    clear_vector_cache()
    yield
    clear_vector_cache()


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]