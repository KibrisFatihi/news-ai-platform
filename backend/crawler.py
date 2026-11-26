"""
Crawler Module
Fetches news from various RSS feeds concurrently with browser User-Agent and timeout.
Uses NewsRepository for fallback cache support.
"""
from __future__ import annotations

import html
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import feedparser

from backend.logger import get_logger, log_rss_fetch
from backend.repository import get_news_repository
from config import RSSFeedConfig, get_config

logger = get_logger("crawler")

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def clean_html(raw_html: str) -> str:
    """Removes HTML tags, decodes HTML entities, and cleans up whitespace."""
    if not raw_html:
        return ""
    clean_text = html.unescape(raw_html)
    clean_text = re.sub(r'<.*?>', '', clean_text)
    clean_text = html.unescape(clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def _fetch_single_feed(
    source_name: str,
    feed_url: str,
    max_items: int,
    timeout: int
) -> Tuple[str, List[dict[str, str]]]:
    """Fetch and parse a single RSS feed with timeout and browser headers."""
    source_start = time.time()
    fetched: List[dict[str, str]] = []

    try:
        req = urllib.request.Request(
            feed_url,
            headers={
                "User-Agent": BROWSER_USER_AGENT,
                "Accept": "application/rss+xml, application/xml, text/xml, */*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read()

        feed = feedparser.parse(raw_bytes)
        count = 0
        for entry in feed.entries:
            if count >= max_items:
                break

            title = clean_html(getattr(entry, "title", ""))
            summary = clean_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            link = getattr(entry, "link", "").strip()
            published = getattr(entry, "published", "") or getattr(entry, "updated", "")

            content = summary if len(summary) > 20 else title

            if title:
                fetched.append({
                    "id": f"{source_name}_{count+1}",
                    "source": source_name,
                    "title": title,
                    "content": content,
                    "link": link,
                    "published": published,
                })
                count += 1

        duration_ms = (time.time() - source_start) * 1000
        log_rss_fetch(logger, source_name, count, success=(count > 0))
        logger.debug(
            f"RSS fetch completed: {source_name} ({count} items)",
            extra={"extra_fields": {
                "event": "rss_fetch_complete",
                "source": source_name,
                "count": count,
                "duration_ms": duration_ms,
            }},
        )
        return source_name, fetched

    except Exception as e:
        duration_ms = (time.time() - source_start) * 1000
        log_rss_fetch(logger, source_name, 0, success=False, error=str(e))
        logger.warning(
            f"Error fetching from {source_name}: {e}",
            extra={"extra_fields": {
                "event": "rss_fetch_error",
                "source": source_name,
                "error": str(e),
                "duration_ms": duration_ms,
            }},
        )
        return source_name, []


def fetch_news(
    max_per_source: Optional[int] = None,
    rss_config: Optional[RSSFeedConfig] = None
) -> List[dict[str, str]]:
    """
    Fetches latest news concurrently from configured RSS feeds.
    Falls back to cached news feed via NewsRepository if network unavailable.
    """
    cfg = rss_config or get_config().rss
    max_items = max_per_source or cfg.max_per_source
    repo = get_news_repository()

    articles: List[dict[str, str]] = []
    start_time = time.time()

    logger.info(
        "Starting concurrent RSS fetch",
        extra={"extra_fields": {
            "event": "rss_fetch_start",
            "sources": list(cfg.feeds.keys()),
            "max_per_source": max_items,
        }},
    )

    max_workers = min(12, max(1, len(cfg.feeds)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_feed, source_name, feed_url, max_items, cfg.request_timeout): source_name
            for source_name, feed_url in cfg.feeds.items()
        }
        for future in as_completed(futures):
            _, source_articles = future.result()
            articles.extend(source_articles)

    total_duration_ms = (time.time() - start_time) * 1000

    if not articles:
        logger.warning(
            "No articles fetched from RSS, attempting fallback cache",
            extra={"extra_fields": {"event": "rss_fallback_attempt"}},
        )
        articles = repo.load_or_fallback() or []
        if articles:
            # Cached feeds are stored in the pipeline-output schema
            # ("original_content", "summary") while the crawler contract and the
            # downstream NLP pipeline expect crawler-schema articles with a
            # "content" field. Normalize cached articles so every article has
            # usable content (prevents KeyError: 'content' when RSS is
            # unreachable and the fallback cache is used).
            for article in articles:
                if not article.get("content"):
                    article["content"] = (
                        article.get("original_content")
                        or article.get("summary")
                        or article.get("title")
                        or ""
                    )
            logger.info(
                f"Fallback cache loaded: {len(articles)} articles",
                extra={"extra_fields": {
                    "event": "rss_fallback_success",
                    "count": len(articles),
                }},
            )
        else:
            logger.warning(
                "No fallback cache available",
                extra={"extra_fields": {"event": "rss_fallback_no_cache"}},
            )

    logger.info(
        f"RSS fetch completed: {len(articles)} total articles in {total_duration_ms:.0f}ms",
        extra={"extra_fields": {
            "event": "rss_fetch_total_complete",
            "total_articles": len(articles),
            "duration_ms": total_duration_ms,
        }},
    )

    return articles


if __name__ == "__main__":
    from backend.logger import setup_logging
    setup_logging(level="DEBUG")

    news = fetch_news(max_per_source=2)
    logger.info(f"Toplam çekilen haber: {len(news)}")
    for item in news[:5]:
        logger.info(f"[{item['source']}] {item['title']}")