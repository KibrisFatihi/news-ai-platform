"""
Common utilities for API handlers.
"""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.logger import get_logger
from backend import middleware as middleware_h
from backend.models import Article
from backend.repository import get_news_repository
from config import get_config

logger = get_logger("handlers.common")

cfg = get_config()
server_cfg = cfg.server

# Server start time for uptime
_server_start_time = __import__("time").time()


def _normalize_article(item: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure article has all required fields for Article model (handles None values safely)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    summary_val = item.get("summary") or ""
    content_val = item.get("content") or summary_val or item.get("original_content") or ""

    return {
        "id": str(item.get("id") or uuid.uuid4())[:8],
        "title": item.get("title") or "Başlıksız Haber",
        "source": item.get("source") or "Unknown",
        "category": item.get("category") or "Genel Gündem",
        "summary": summary_val,
        "content": content_val,
        "original_content": item.get("original_content") or content_val,
        "link": item.get("link") or "",
        "image_url": item.get("image_url") or None,
        "ai_engine": item.get("ai_engine") or "Yerel NLP",
        "ai_model": item.get("ai_model") or "TextRank",
        "published": str(item.get("published") or now_iso),
        "updated_at": str(item.get("updated_at") or now_iso),
        "related_count": item.get("related_count") if item.get("related_count") is not None else 0,
        "related_sources": item.get("related_sources") or [],
    }


def _get_header(headers: Dict[str, str], name: str) -> Optional[str]:
    """Case-insensitive header lookup.

    Browsers may send header names in lowercase (e.g. ``x-refresh-token``,
    ``x-groq-api-key``), but a plain ``dict`` lookup is case-sensitive. Check
    the exact name first, then fall back to a case-insensitive scan so both
    callers (HTTP server and direct unit tests) work regardless of casing.
    """
    if name in headers:
        return headers[name]
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


def _build_cors_headers(request_headers: Dict[str, str]) -> Dict[str, str]:
    """Build CORS headers for response."""
    return middleware_h.get_cors_headers(_get_header(request_headers, "Origin"))


def _handle_conditional(
    request_headers: Dict[str, str],
    request_id: str,
    articles: list,
    repo,
) -> tuple[bool, Dict[str, str], Dict[str, Any]]:
    """Handle conditional request (304 Not Modified). Returns (handled, headers, body)."""
    cache_etag = middleware_h.compute_etag(articles)
    cache_last_modified = repo.get_last_modified()

    if middleware_h.check_conditional_request(request_headers, cache_etag, cache_last_modified):
        headers = _build_cors_headers(request_headers)
        middleware_h.add_cache_headers(headers, cache_etag, cache_last_modified)
        middleware_h.log_api_request(logger, "GET", "/api/news", 304, 0, request_id)
        return True, headers, {}
    return False, {}, {}


def _finalize_response(
    request_headers: Dict[str, str],
    request_id: str,
    method: str,
    path: str,
    status: int,
    body: Dict[str, Any],
    start_time: float,
    etag: str = None,
    last_modified: float = None,
    cache_control: str = "no-cache, must-revalidate",
) -> tuple[int, Dict[str, str], Dict[str, Any]]:
    """Finalize response with headers and logging."""
    headers = _build_cors_headers(request_headers)
    if etag:
        middleware_h.add_cache_headers(headers, etag, last_modified, cache_control)
    middleware_h.log_api_request(logger, method, path, status, (time.time() - start_time) * 1000, request_id)
    return status, headers, body