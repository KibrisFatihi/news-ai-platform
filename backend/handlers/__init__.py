"""
API Request Handlers Package
Business logic for each endpoint - separated from HTTP concerns for testability.

Exports:
- handle_get_news: GET /api/news
- handle_get_status: GET /api/status
- handle_get_health: GET /api/health
- handle_post_refresh: POST /api/refresh
- handle_options: OPTIONS preflight
- handle_not_found: 404 handler
- _refresh_lock: threading.Lock for refresh concurrency control
- get_news_repository: repository factory
- cfg: config instance
- compute_etag: etag computation
- verify_refresh_token: token verification
- check_refresh_rate_limit: rate limit check
- _server_start_time: server uptime reference
- logger: module logger
"""
from __future__ import annotations

import threading

from backend.handlers.news import handle_get_news
from backend.handlers.status import handle_get_status
from backend.handlers.health import handle_get_health
from backend.handlers.weather import handle_get_weather
from backend.handlers.refresh import (
    handle_post_refresh,
    _refresh_lock,
)
from backend.handlers.others import handle_options, handle_not_found
from backend.handlers.common import (
    get_news_repository,
    cfg,
    _server_start_time,
    logger,
    _normalize_article,
    _build_cors_headers,
    _finalize_response,
    _get_header,
)
from backend.middleware import check_refresh_rate_limit, compute_etag, verify_refresh_token

# Re-export time module for backward compatibility with tests
import time as _time
time = _time

# Re-export for backward compatibility with tests
_refresh_lock = _refresh_lock

__all__ = [
    "handle_get_news",
    "handle_get_status",
    "handle_get_health",
    "handle_get_weather",
    "handle_post_refresh",
    "handle_options",
    "handle_not_found",
    "_refresh_lock",
    "get_news_repository",
    "cfg",
    "compute_etag",
    "verify_refresh_token",
    "check_refresh_rate_limit",
    "_server_start_time",
    "logger",
    "_normalize_article",
    "_build_cors_headers",
    "_finalize_response",
    "_get_header",
]