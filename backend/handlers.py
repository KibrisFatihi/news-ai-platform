"""
API Request Handlers
Business logic for each endpoint - separated from HTTP concerns for testability.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.logger import get_logger
from backend.middleware import (
    add_cache_headers,
    check_conditional_request,
    check_refresh_rate_limit,
    compute_etag,
    create_error_response,
    create_success_response,
    format_http_date,
    generate_request_id,
    get_cors_headers,
    handle_options_request,
    log_api_request,
    verify_refresh_token,
)
from backend.models import Article, HealthResponse, NewsResponse, RefreshResponse, StatusResponse
from backend.repository import get_news_repository
from config import get_config
# run_pipeline imported locally in handle_post_refresh to avoid circular import

logger = get_logger("handlers")

cfg = get_config()
server_cfg = cfg.server

# Server start time for uptime
_server_start_time = time.time()

# Refresh protection lock
_refresh_lock = threading.Lock()
_refresh_in_progress = False


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
        "ai_engine": item.get("ai_engine") or "Yerel NLP",
        "ai_model": item.get("ai_model") or "TextRank",
        "published": str(item.get("published") or now_iso),
        "updated_at": str(item.get("updated_at") or now_iso),
        "related_count": item.get("related_count") if item.get("related_count") is not None else 0,
        "related_sources": item.get("related_sources") or [],
    }


def handle_get_news(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """
    Handle GET /api/news request.
    Returns (status_code, headers, body).
    """
    start_time = time.time()
    repo = get_news_repository()

    # Load cached data
    articles = repo.load_or_fallback() or []
    cache_etag = compute_etag(articles)
    cache_last_modified = repo.get_last_modified()

    # Check conditional request
    if check_conditional_request(request_headers, cache_etag, cache_last_modified):
        headers = get_cors_headers(request_headers.get("Origin", ""))
        add_cache_headers(headers, cache_etag, cache_last_modified)
        log_api_request(logger, "GET", "/api/news", 304, (time.time() - start_time) * 1000, request_id)
        return 304, headers, {}

    # Build response
    categories: Dict[str, int] = {}
    for item in articles:
        cat = item.get("category") or "Genel"
        categories[cat] = categories.get(cat, 0) + 1

    groq_configured = bool(cfg.get_groq_api_key())

    try:
        response_data = NewsResponse(
            success=True,
            count=len(articles),
            categories=categories,
            groq_configured=groq_configured,
            active_engine="Groq LLaMA 3.3" if groq_configured else "Yerel NLP (Sıfır Token)",
            articles=[Article(**_normalize_article(a)) for a in articles]
        ).model_dump()
    except Exception as e:
        logger.exception(f"NewsResponse validation failed [{request_id}]: {e}")
        raise

    headers = get_cors_headers(request_headers.get("Origin", ""))
    add_cache_headers(
        headers,
        etag=cache_etag,
        last_modified=cache_last_modified,
        cache_control=f"max-age={server_cfg.cache_ttl_seconds}"
    )

    log_api_request(logger, "GET", "/api/news", 200, (time.time() - start_time) * 1000, request_id)
    return 200, headers, response_data


def handle_get_status(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle GET /api/status request."""
    start_time = time.time()

    groq_key = cfg.get_groq_api_key()

    response_data = StatusResponse(
        status="online",
        version="2.0.0",
        groq_connected=bool(groq_key),
        model=cfg.groq.default_model,
        mode="hybrid",
        uptime_seconds=time.time() - _server_start_time
    ).model_dump()

    headers = get_cors_headers(request_headers.get("Origin", ""))
    log_api_request(logger, "GET", "/api/status", 200, (time.time() - start_time) * 1000, request_id)
    return 200, headers, response_data


def handle_get_health(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle GET /api/health request."""
    start_time = time.time()

    repo = get_news_repository()
    articles = repo.load_or_fallback()

    from backend.groq_service import get_groq_status
    from backend.circuit_breaker import get_all_circuit_breakers

    groq_status = get_groq_status()
    circuit_breakers = get_all_circuit_breakers()

    response_data = HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="2.0.0",
        uptime_seconds=time.time() - _server_start_time,
        components={
            "news_cache": {
                "loaded": articles is not None,
                "article_count": len(articles) if articles else 0,
                "last_modified": format_http_date(repo.get_last_modified()) if repo.get_last_modified() else None
            },
            "groq_api": groq_status,
            "circuit_breakers": circuit_breakers
        }
    )

    # Check for degraded status
    cb_states = [cb.get("state", "closed") for cb in circuit_breakers.values()]
    if any(state == "open" for state in cb_states):
        response_data.status = "degraded"
        response_data.warnings = ["One or more circuit breakers are OPEN"]

    headers = get_cors_headers(request_headers.get("Origin", ""))
    log_api_request(logger, "GET", "/api/health", 200, (time.time() - start_time) * 1000, request_id)
    return 200, headers, response_data.model_dump()


def handle_post_refresh(
    request_headers: Dict[str, str],
    request_body: bytes,
    client_ip: str,
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """
    Handle POST /api/refresh request.
    Returns (status_code, headers, body).
    """
    start_time = time.time()

    # Check request size
    content_length = len(request_body)
    if content_length > server_cfg.max_request_size_mb * 1024 * 1024:
        error_resp = create_error_response("Request too large", 413, request_id)
        headers = get_cors_headers(request_headers.get("Origin", ""))
        log_api_request(logger, "POST", "/api/refresh", 413, (time.time() - start_time) * 1000, request_id)
        return 413, headers, error_resp

    # Verify refresh token
    refresh_token = request_headers.get("X-Refresh-Token")
    if not verify_refresh_token(refresh_token):
        logger.warning(
            "Invalid or missing X-Refresh-Token",
            extra={"extra_fields": {"event": "refresh_token_invalid"}}
        )
        error_resp = create_error_response("Unauthorized: Invalid refresh token", 401, request_id)
        headers = get_cors_headers(request_headers.get("Origin", ""))
        log_api_request(logger, "POST", "/api/refresh", 401, (time.time() - start_time) * 1000, request_id)
        return 401, headers, error_resp

    # IP-based rate limiting
    allowed, retry_after = check_refresh_rate_limit(client_ip)
    if not allowed:
        logger.warning(
            f"Rate limit exceeded for /api/refresh from {client_ip}",
            extra={"extra_fields": {"event": "refresh_rate_limit_exceeded", "ip": client_ip}}
        )
        error_resp = create_error_response("Rate limit exceeded. Try again later.", 429, request_id)
        headers = get_cors_headers(request_headers.get("Origin", ""))
        headers["Retry-After"] = str(retry_after)
        log_api_request(logger, "POST", "/api/refresh", 429, (time.time() - start_time) * 1000, request_id)
        return 429, headers, error_resp

    # Prevent concurrent refreshes
    global _refresh_in_progress
    if not _refresh_lock.acquire(blocking=False):
        logger.warning(
            "Refresh already in progress",
            extra={"extra_fields": {"event": "refresh_concurrent_blocked"}}
        )
        error_resp = create_error_response("Refresh already in progress", 409, request_id)
        headers = get_cors_headers(request_headers.get("Origin", ""))
        log_api_request(logger, "POST", "/api/refresh", 409, (time.time() - start_time) * 1000, request_id)
        return 409, headers, error_resp

    try:
        # Parse request body
        api_key = None
        if request_body:
            try:
                payload = json.loads(request_body.decode("utf-8"))
                api_key = payload.get("api_key")
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Invalid JSON in request body: {e}",
                    extra={"extra_fields": {"event": "api_invalid_json", "error": str(e)}}
                )

        # Header takes precedence over body
        header_key = request_headers.get("X-Groq-Api-Key")
        if header_key:
            api_key = header_key

        # Validate API key format if provided
        if api_key and not api_key.startswith("gsk_"):
            logger.warning(
                "Invalid API key format (should start with gsk_)",
                extra={"extra_fields": {"event": "api_invalid_key_format"}}
            )

        logger.info(
            "Pipeline refresh triggered",
            extra={"extra_fields": {
                "event": "pipeline_refresh_triggered",
                "has_api_key": bool(api_key)
            }}
        )

        # Import here to avoid circular import
        from main import run_pipeline

        try:
            articles = run_pipeline(api_key=api_key) or []

            # Normalize articles to ensure all required fields for Article model
            normalized_articles = [_normalize_article(a) for a in articles]

            response_data = RefreshResponse(
                success=True,
                message="Haber akışı ve yapay zeka özetleri başarıyla güncellendi.",
                count=len(normalized_articles),
                articles=[Article(**a) for a in normalized_articles]
            ).model_dump()

            headers = get_cors_headers(request_headers.get("Origin", ""))
            log_api_request(logger, "POST", "/api/refresh", 200, (time.time() - start_time) * 1000, request_id)
            return 200, headers, response_data

        except Exception as e:
            logger.exception(
                f"Pipeline execution failed [{request_id}]: {e}",
                extra={"extra_fields": {"event": "pipeline_error", "error": str(e), "request_id": request_id}}
            )
            error_resp = create_error_response("Internal server error", 500, request_id)
            headers = get_cors_headers(request_headers.get("Origin", ""))
            log_api_request(logger, "POST", "/api/refresh", 500, (time.time() - start_time) * 1000, request_id)
            return 500, headers, error_resp

    finally:
        _refresh_lock.release()


def handle_options(
    request_headers: Dict[str, str]
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle OPTIONS preflight request."""
    origin = request_headers.get("Origin", "")
    status, headers = handle_options_request(origin)
    return status, headers, {}


def handle_not_found(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle 404 for unknown endpoints."""
    error_resp = create_error_response("Endpoint not found", 404, request_id)
    headers = get_cors_headers(request_headers.get("Origin", ""))
    return 404, headers, error_resp