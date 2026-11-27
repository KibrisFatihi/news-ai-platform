"""
News Repository
Handles atomic persistence, fallback loading, and cache management.
Single point of truth for news feed storage.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from backend.logger import get_logger
from backend.models import Article, NewsFeed
from config import CacheConfig, get_config

logger = get_logger("repository")


def _compute_content_hash(title: str, content: str) -> str:
    """Compute SHA256 hash of title + content for delta detection."""
    combined = f"{title}|{content}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _get_existing_hashes(articles: NewsFeed) -> Dict[str, str]:
    """Get mapping of URL -> content_hash from existing articles."""
    return {a.get("link", ""): a.get("content_hash", "") for a in articles if a.get("link")}


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

    def save_progressive(self, articles: NewsFeed) -> NewsFeed:
        """
        Progressive save: merge new articles with existing, preserve completed summaries.

        Args:
            articles: New articles from pipeline (with summary_status="pending" or "completed")

        Returns:
            Merged article list saved to disk
        """
        existing = self.load_or_fallback()
        existing_hashes = _get_existing_hashes(existing)
        existing_by_url = {a.get("link", ""): a for a in existing if a.get("link")}

        merged: NewsFeed = []
        new_count = 0
        preserved_count = 0
        updated_count = 0

        for article in articles:
            url = article.get("link", "")
            content_hash = _compute_content_hash(article.get("title", ""), article.get("original_content", ""))
            article["content_hash"] = content_hash

            if url in existing_by_url:
                existing_article = existing_by_url[url]
                existing_hash = existing_hashes.get(url, "")

                # Check if existing summary was AI-generated (Groq) vs local NLP
                ai_engine = existing_article.get("ai_engine", "")
                is_ai_generated = ai_engine.startswith("Groq") or "Groq" in ai_engine

                if existing_hash == content_hash and existing_article.get("summary_status") == "completed" and is_ai_generated:
                    # Content unchanged and already summarized with AI - preserve it
                    merged.append(existing_article)
                    preserved_count += 1
                else:
                    # Content changed, not yet summarized, or was local NLP - use new article (may have pending summary)
                    merged.append(article)
                    updated_count += 1
            else:
                # Brand new article
                merged.append(article)
                new_count += 1

        # Sort by published date descending (newest first)
        merged.sort(key=lambda x: x.get("published", ""), reverse=True)

        self.save(merged)

        logger.info(
            f"Progressive save: {new_count} new, {updated_count} updated, {preserved_count} preserved (AI only)",
            extra={"extra_fields": {
                "event": "news_feed_progressive_save",
                "new": new_count,
                "updated": updated_count,
                "preserved": preserved_count,
                "total": len(merged)
            }}
        )

        return merged

    def update_article_summaries(self, summaries: List[Dict]) -> int:
        """
        Update specific articles with completed summaries.

        Args:
            summaries: List of dicts with keys: link, summary, ai_engine, ai_model,
                       summary_status, updated_at

        Returns:
            Number of articles updated
        """
        with self.transaction() as articles:
            article_by_url = {a.get("link", ""): a for a in articles if a.get("link")}
            updated = 0
            for summary in summaries:
                url = summary.get("link")
                if url in article_by_url:
                    article = article_by_url[url]
                    article["summary"] = summary.get("summary", article.get("summary", ""))
                    article["ai_engine"] = summary.get("ai_engine", article.get("ai_engine", ""))
                    article["ai_model"] = summary.get("ai_model", article.get("ai_model", ""))
                    article["summary_status"] = summary.get("summary_status", "completed")
                    article["updated_at"] = summary.get("updated_at", article.get("updated_at", ""))
                    updated += 1
            return updated

    def get_pending_articles(self) -> List[Dict]:
        """Get articles that still need summarization."""
        articles = self.load_or_fallback()
        return [a for a in articles if a.get("summary_status") == "pending"]


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