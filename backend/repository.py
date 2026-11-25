"""
News Repository
Handles atomic persistence, fallback loading, and cache management.
Single point of truth for news feed storage.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from backend.logger import get_logger
from backend.models import Article, NewsFeed
from config import CacheConfig, get_config

logger = get_logger("repository")


class NewsRepository:
    """
    Repository for news feed persistence.
    Provides atomic writes, fallback loading, and cache invalidation.
    """

    def __init__(self, cache_config: Optional[CacheConfig] = None):
        cfg = cache_config or get_config().cache
        self.feed_path = Path(cfg.news_feed_path)
        self.max_age_seconds = cfg.max_age_seconds
        self.cleanup_on_startup = cfg.cleanup_on_startup
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Ensure the parent directory exists."""
        self.feed_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, articles: NewsFeed) -> None:
        """
        Atomically save articles to JSON file.
        Uses temporary file + atomic replace to prevent corruption.
        """
        start_time = time.time()

        # Write to temporary file first
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.feed_path.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp_file:
            json.dump(articles, tmp_file, ensure_ascii=False, indent=2)
            tmp_path = tmp_file.name

        try:
            # Atomic replace (os.replace is atomic on POSIX, near-atomic on Windows)
            os.replace(tmp_path, self.feed_path)
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"News feed saved atomically: {len(articles)} articles",
                extra={"extra_fields": {
                    "event": "news_feed_save",
                    "count": len(articles),
                    "duration_ms": duration_ms,
                    "path": str(self.feed_path)
                }}
            )
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> Optional[NewsFeed]:
        """
        Load articles from JSON file.
        Returns None if file doesn't exist or is invalid.
        """
        if not self.feed_path.exists():
            logger.debug("News feed file does not exist", extra={"extra_fields": {"event": "news_feed_load_missing"}})
            return None

        try:
            with open(self.feed_path, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.warning("News feed is not a list, ignoring", extra={"extra_fields": {"event": "news_feed_invalid_format"}})
                return None

            logger.info(
                f"News feed loaded: {len(data)} articles",
                extra={"extra_fields": {"event": "news_feed_load", "count": len(data)}}
            )
            return data

        except json.JSONDecodeError as e:
            logger.error(
                f"News feed JSON decode error: {e}",
                extra={"extra_fields": {"event": "news_feed_json_error", "error": str(e)}}
            )
            return None
        except Exception as e:
            logger.error(
                f"News feed load error: {e}",
                extra={"extra_fields": {"event": "news_feed_load_error", "error": str(e)}}
            )
            return None

    def load_or_fallback(self) -> NewsFeed:
        """
        Load articles, or return empty list if unavailable.
        This is the single fallback point - no duplicate logic elsewhere.
        """
        data = self.load()
        if data is not None:
            return data

        logger.warning(
            "No news feed available, returning empty list",
            extra={"extra_fields": {"event": "news_feed_empty_fallback"}}
        )
        return []

    def is_stale(self) -> bool:
        """Check if the cached feed is older than max_age_seconds."""
        if not self.feed_path.exists():
            return True
        mtime = self.feed_path.stat().st_mtime
        return (time.time() - mtime) > self.max_age_seconds

    def get_last_modified(self) -> Optional[float]:
        """Get the last modified timestamp of the feed file."""
        if self.feed_path.exists():
            return self.feed_path.stat().st_mtime
        return None

    def cleanup(self) -> None:
        """Remove the news feed file (for testing or reset)."""
        if self.feed_path.exists():
            self.feed_path.unlink()
            logger.info("News feed file removed", extra={"extra_fields": {"event": "news_feed_cleanup"}})

    @contextmanager
    def transaction(self):
        """
        Context manager for transactional updates.
        Loads current data, yields it for modification, then saves atomically.
        """
        articles = self.load_or_fallback()
        try:
            yield articles
            self.save(articles)
        except Exception:
            logger.error("Transaction failed, changes not saved", extra={"extra_fields": {"event": "repository_transaction_failed"}})
            raise


# --- Global Repository Instance ---

_repository: Optional[NewsRepository] = None


def get_news_repository() -> NewsRepository:
    """Get or create the global news repository instance."""
    global _repository
    if _repository is None:
        _repository = NewsRepository()
    return _repository


def reset_news_repository() -> None:
    """Reset the global repository (for testing)."""
    global _repository
    _repository = None


# --- Backwards Compatibility ---

def save_news_feed(articles: NewsFeed, output_path: Optional[Path] = None) -> None:
    """Backwards compatible save function."""
    repo = get_news_repository()
    if output_path:
        # Temporarily use different path
        old_path = repo.feed_path
        repo.feed_path = output_path
        repo._ensure_parent_dir()
        repo.save(articles)
        repo.feed_path = old_path
    else:
        repo.save(articles)


def load_news_feed() -> NewsFeed:
    """Backwards compatible load function."""
    return get_news_repository().load_or_fallback()