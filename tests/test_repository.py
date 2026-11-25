"""
Tests for the News Repository module.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import threading
from pathlib import Path

import pytest

from backend.repository import NewsRepository, get_news_repository, reset_news_repository

IS_WINDOWS = sys.platform == "win32"


class TestNewsRepository:
    """Tests for NewsRepository class."""

    def test_save_and_load(self, tmp_path):
        """Test basic save and load."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        articles = [
            {
                "id": "1",
                "title": "Test Haber",
                "source": "Test Source",
                "category": "Test",
                "summary": "Özet",
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

        repo.save(articles)
        loaded = repo.load()

        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test Haber"

    def test_load_missing_file_returns_none(self, tmp_path):
        """Test loading non-existent file returns None."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "nonexistent.json"

        result = repo.load()
        assert result is None

    def test_load_or_fallback_returns_empty_list(self, tmp_path):
        """Test load_or_fallback returns empty list when no cache."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "nonexistent.json"

        result = repo.load_or_fallback()
        assert result == []

    def test_atomic_write(self, tmp_path):
        """Test atomic write prevents partial reads."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        articles = [{"title": "Test", "data": "x" * 10000}]
        repo.save(articles)

        # Verify file is valid JSON
        loaded = repo.load()
        assert loaded is not None
        assert loaded[0]["title"] == "Test"

    @pytest.mark.skipif(IS_WINDOWS, reason="Windows file permissions work differently, chmod doesn't prevent file creation")
    def test_atomic_write_temp_file_cleanup(self, tmp_path):
        """Test temp file is cleaned up on error."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        # Make directory read-only
        os.chmod(tmp_path, 0o555)
        try:
            with pytest.raises((OSError, PermissionError)):
                repo.save([{"title": "Test"}])
        finally:
            os.chmod(tmp_path, 0o755)

        # No temp files should remain
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

    @pytest.mark.skipif(IS_WINDOWS, reason="Windows os.replace fails when file is open by another process")
    def test_concurrent_readers_no_partial_reads(self, tmp_path):
        """Test concurrent readers don't see partial writes."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        results = []
        errors = []

        def writer():
            for i in range(20):
                try:
                    repo.save([{"title": f"Test {i}", "data": "x" * 5000}])
                except Exception as e:
                    errors.append(e)
                time.sleep(0.005)

        def reader():
            for _ in range(40):
                try:
                    data = repo.load()
                    if data:
                        results.append(data)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.002)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All reads should be valid
        for data in results:
            assert isinstance(data, list)
            assert "title" in data[0]

    def test_is_stale(self, tmp_path):
        """Test staleness detection."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo.max_age_seconds = 1  # 1 second

        repo.save([{"title": "Test"}])
        assert not repo.is_stale()

        time.sleep(1.1)
        assert repo.is_stale()

    def test_get_last_modified(self, tmp_path):
        """Test getting last modified timestamp."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        assert repo.get_last_modified() is None

        repo.save([{"title": "Test"}])
        mtime = repo.get_last_modified()
        assert mtime is not None
        assert mtime <= time.time()

    def test_cleanup(self, tmp_path):
        """Test cleanup removes file."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        repo.save([{"title": "Test"}])
        assert repo.feed_path.exists()

        repo.cleanup()
        assert not repo.feed_path.exists()

    def test_transaction_context_manager(self, tmp_path):
        """Test transaction context manager."""
        repo = NewsRepository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        # Initial save
        repo.save([{"title": "Initial"}])

        # Transaction that succeeds
        with repo.transaction() as articles:
            articles.append({"title": "Added"})

        loaded = repo.load()
        assert len(loaded) == 2

        # Transaction that fails (should not save)
        try:
            with repo.transaction() as articles:
                articles.append({"title": "Should not save"})
                raise ValueError("Intentional error")
        except ValueError:
            pass

        loaded = repo.load()
        assert len(loaded) == 2  # Still 2, failed transaction rolled back


class TestGlobalRepository:
    """Tests for global repository functions."""

    def test_get_news_repository_singleton(self):
        """Test global repository is singleton."""
        reset_news_repository()
        repo1 = get_news_repository()
        repo2 = get_news_repository()
        assert repo1 is repo2

    def test_reset_news_repository(self):
        """Test reset creates new instance."""
        reset_news_repository()
        repo1 = get_news_repository()
        reset_news_repository()
        repo2 = get_news_repository()
        assert repo1 is not repo2

    def test_backwards_compat_functions(self, tmp_path):
        """Test backwards compatible save/load functions."""
        from backend.repository import save_news_feed, load_news_feed

        reset_news_repository()
        repo = get_news_repository()
        repo.feed_path = tmp_path / "news_feed.json"
        repo._ensure_parent_dir()

        articles = [{"title": "Test"}]
        save_news_feed(articles)
        loaded = load_news_feed()

        assert len(loaded) == 1
        assert loaded[0]["title"] == "Test"