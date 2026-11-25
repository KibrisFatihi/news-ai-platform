"""
End-to-end integration tests for News AI Platform.
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
import requests


@pytest.mark.integration
class TestFullPipelineE2E:
    """End-to-end tests for the full pipeline."""

    def test_pipeline_run_without_api_key(self):
        """Test complete pipeline execution without API key."""
        from main import run_pipeline

        results = run_pipeline(api_key=None, max_per_source=1)

        assert len(results) > 0
        for article in results:
            assert "title" in article
            assert "summary" in article
            assert "category" in article
            assert "ai_engine" in article
            assert "Yerel NLP" in article["ai_engine"]
            assert "source" in article
            assert "link" in article

    def test_pipeline_run_with_invalid_api_key(self):
        """Test pipeline with invalid API key falls back to local."""
        from main import run_pipeline

        results = run_pipeline(api_key="invalid_key", max_per_source=1)

        assert len(results) > 0
        for article in results:
            assert "Yerel NLP" in article["ai_engine"] or "Yedek Mod" in article["ai_engine"]

    def test_pipeline_output_saved_to_json(self, tmp_path, monkeypatch):
        """Test pipeline output is saved to news_feed.json."""
        from config import AppConfig, CacheConfig
        from main import run_pipeline
        from backend.repository import reset_news_repository

        # Temporarily redirect output
        test_path = tmp_path / "test_news_feed.json"

        # Create a new config with custom cache path (config is frozen, so recreate)
        import config as config_module
        original_config = config_module.get_config()
        custom_cache = CacheConfig(
            news_feed_path=str(test_path),
            max_age_seconds=original_config.cache.max_age_seconds,
            cleanup_on_startup=original_config.cache.cleanup_on_startup
        )
        custom_config = AppConfig(
            rss=original_config.rss,
            nlp=original_config.nlp,
            groq=original_config.groq,
            server=original_config.server,
            logging=original_config.logging,
            cache=custom_cache,
            base_dir=original_config.base_dir
        )

        # Patch the global config instance and get_config function
        monkeypatch.setattr(config_module, "config", custom_config)
        monkeypatch.setattr(config_module, "get_config", lambda: custom_config)

        # Reset repository so it picks up the new config
        reset_news_repository()

        results = run_pipeline(api_key=None, max_per_source=1)
        assert test_path.exists()

        with open(test_path, encoding="utf-8") as f:
            saved_data = json.load(f)

        assert len(saved_data) == len(results)

    @patch("backend.crawler.feedparser.parse")
    def test_pipeline_with_mocked_rss(self, mock_parse):
        """Test pipeline with mocked RSS feeds."""
        from main import run_pipeline

        # Mock feedparser
        mock_feed = MagicMock()
        mock_feed.entries = [
            MagicMock(
                title="E2E Test Haber",
                summary="Bu bir E2E test haberi özeti.",
                link="https://example.com/e2e",
                published="Mon, 01 Jan 2026 12:00:00 GMT"
            )
        ]
        mock_parse.return_value = mock_feed

        results = run_pipeline(api_key=None, max_per_source=1)

        assert len(results) == 1
        assert results[0]["title"] == "E2E Test Haber"


@pytest.mark.integration
class TestAPIServerE2E:
    """End-to-end tests with actual HTTP server."""

    @pytest.fixture(scope="class")
    def server_url(self, project_root):
        """Start server in background for testing."""
        import hashlib
        import hmac
        import platform
        import subprocess

        # Kill any existing process on port 8000 before starting
        def kill_port_8000():
            try:
                import psutil
                for conn in psutil.net_connections():
                    if conn.laddr.port == 8000 and conn.pid:
                        try:
                            psutil.Process(conn.pid).kill()
                        except Exception:
                            pass
            except ImportError:
                if platform.system() == "Windows":
                    subprocess.run(
                        ["cmd", "/c", "for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :8000') do taskkill /F /PID %a 2>nul"],
                        shell=True, capture_output=True
                    )
                else:
                    subprocess.run(["fuser", "-k", "8000/tcp"], capture_output=True)

        # Clean up any stale processes on port 8000
        kill_port_8000()
        time.sleep(1)  # Wait for port to be released

        # Set fixed token secret for testing
        test_secret = "test_refresh_token_secret_32bytes!!"
        os.environ["REFRESH_TOKEN_SECRET"] = test_secret

        # Pre-compute the valid token
        expected_token = hmac.new(
            test_secret.encode(),
            b"refresh",
            hashlib.sha256
        ).hexdigest()

        # Start server
        env = {
            **os.environ,
            "REFRESH_TOKEN_SECRET": test_secret,
            "PYTHONPATH": str(project_root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }

        proc = subprocess.Popen(
            [sys.executable, "-B", "server.py"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Wait for server to start
        time.sleep(2)

        # Check if server is responding
        for _ in range(10):
            try:
                requests.get("http://localhost:8000/api/status", timeout=1)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)

        yield "http://localhost:8000", expected_token

        # Cleanup - wait a bit to capture debug output
        time.sleep(1)
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            print(f"[SERVER STDOUT]:\n{stdout.decode('utf-8', errors='replace')}")
            print(f"[SERVER STDERR]:\n{stderr.decode('utf-8', errors='replace')}")
        except Exception as e:
            print(f"[SERVER] Could not capture output: {e}")
        kill_port_8000()

    def test_status_endpoint(self, server_url):
        """Test /api/status endpoint."""
        url, _ = server_url
        response = requests.get(f"{url}/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "version" in data

    def test_news_endpoint(self, server_url):
        """Test /api/news endpoint."""
        url, _ = server_url
        response = requests.get(f"{url}/api/news")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "articles" in data

    def test_health_endpoint(self, server_url):
        """Test /api/health endpoint."""
        url, _ = server_url
        response = requests.get(f"{url}/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]

    def test_refresh_endpoint(self, server_url):
        """Test /api/refresh endpoint (may take time)."""
        url, token = server_url
        response = requests.post(
            f"{url}/api/refresh",
            json={},
            headers={"X-Refresh-Token": token},
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "count" in data

    def test_cors_headers(self, server_url):
        """Test CORS headers on API endpoints."""
        url, _ = server_url
        response = requests.options(f"{url}/api/news")
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == "*"

    def test_frontend_served(self, server_url):
        """Test frontend is served."""
        url, _ = server_url
        response = requests.get(f"{url}/", allow_redirects=True)
        # Should redirect and serve frontend
        assert response.status_code == 200


@pytest.mark.integration
class TestGroqIntegration:
    """Integration tests for Groq API (requires actual API key)."""

    @pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set"
    )
    def test_groq_summarization(self):
        """Test actual Groq API summarization."""
        from backend.groq_service import summarize_with_groq

        result = summarize_with_groq(
            "Test Başlık",
            "Bu bir test içeriğidir. Groq API özetlemesi bekleniyor."
        )

        assert result["success"] is True
        assert "Groq" in result["engine"]
        assert len(result["summary"]) > 0

    @pytest.mark.skipif(
        not os.environ.get("GROQ_API_KEY"),
        reason="GROQ_API_KEY not set"
    )
    def test_groq_fast_model(self):
        """Test Groq with fast model."""
        from backend.groq_service import FAST_MODEL, summarize_with_groq

        result = summarize_with_groq(
            "Test Başlık",
            "Bu bir test içeriğidir.",
            model=FAST_MODEL
        )

        assert result["success"] is True
        assert result["model"] == FAST_MODEL


import json
