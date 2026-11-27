"""
GET /api/news handler.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from backend.models import Article, NewsResponse
from backend.handlers import common as common_h
from backend import middleware as middleware_h


def handle_get_news(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """
    Handle GET /api/news request.
    Returns (status_code, headers, body).
    """
    start_time = time.time()
    repo = common_h.get_news_repository()

    # Load cached data
    articles = repo.load_or_fallback() or []

    # Check conditional request (304 Not Modified)
    handled, headers, body = common_h._handle_conditional(request_headers, request_id, articles, repo)
    if handled:
        return 304, headers, body

    # Build response
    categories: Dict[str, int] = {}
    for item in articles:
        cat = item.get("category") or "Genel"
        categories[cat] = categories.get(cat, 0) + 1

    groq_configured = bool(common_h.cfg.get_groq_api_key())

    try:
        response_data = NewsResponse(
            success=True,
            count=len(articles),
            categories=categories,
            groq_configured=groq_configured,
            active_engine="Groq LLaMA 3.3" if groq_configured else "Yerel NLP (Sıfır Token)",
            articles=[Article(**common_h._normalize_article(a)) for a in articles]
        ).model_dump()
    except Exception as e:
        common_h.logger.exception(f"NewsResponse validation failed [{request_id}]: {e}")
        raise

    return common_h._finalize_response(
        request_headers, request_id, "GET", "/api/news", 200, response_data, start_time,
        etag=middleware_h.compute_etag(articles),
        last_modified=repo.get_last_modified(),
    )