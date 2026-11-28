"""
POST /api/refresh handler.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

from backend.handlers import common as common_h
from backend import middleware as middleware_h

# Refresh protection lock
_refresh_lock = threading.Lock()


def _run_pipeline_in_background(api_key: Optional[str], lock: threading.Lock, request_id: str) -> None:
    """Pipeline'ı arka plan thread'inde çalıştırır; bitince kilidi serbest bırakır."""
    try:
        # Import here to avoid circular import
        from main import run_pipeline

        articles = run_pipeline(api_key=api_key) or []

        common_h.logger.info(
            "Background pipeline completed",
            extra={"extra_fields": {
                "event": "background_pipeline_completed",
                "article_count": len(articles),
                "request_id": request_id
            }}
        )
    except Exception as e:
        common_h.logger.exception(
            f"Background pipeline failed [{request_id}]: {e}",
            extra={"extra_fields": {"event": "background_pipeline_failed", "error": str(e), "request_id": request_id}}
        )
    finally:
        lock.release()


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
    if content_length > common_h.server_cfg.max_request_size_mb * 1024 * 1024:
        error_resp = middleware_h.create_error_response("Request too large", 413, request_id)
        headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
        middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 413, (time.time() - start_time) * 1000, request_id)
        return 413, headers, error_resp

    # Verify refresh token (header okuması büyük/küçük harfe duyarsız)
    refresh_token = common_h._get_header(request_headers, "X-Refresh-Token")
    if not middleware_h.verify_refresh_token(refresh_token):
        common_h.logger.warning(
            "Invalid or missing X-Refresh-Token",
            extra={"extra_fields": {"event": "refresh_token_invalid"}}
        )
        error_resp = middleware_h.create_error_response("Unauthorized: Invalid refresh token", 401, request_id)
        headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
        middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 401, (time.time() - start_time) * 1000, request_id)
        return 401, headers, error_resp

    # Origin kısıtlaması — sadece izin verilen originlerden /api/refresh isteği kabul edilir
    origin = common_h._get_header(request_headers, "Origin")
    if origin and "*" not in common_h.server_cfg.cors_allowed_origins and origin not in common_h.server_cfg.cors_allowed_origins:
        common_h.logger.warning(
            f"Origin not allowed for /api/refresh: {origin}",
            extra={"extra_fields": {"event": "refresh_origin_not_allowed", "origin": origin}}
        )
        error_resp = middleware_h.create_error_response("Forbidden: Origin not allowed", 403, request_id)
        headers = middleware_h.get_cors_headers(origin)
        middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 403, (time.time() - start_time) * 1000, request_id)
        return 403, headers, error_resp

    # IP-based rate limiting
    allowed, retry_after = middleware_h.check_refresh_rate_limit(client_ip)
    if not allowed:
        common_h.logger.warning(
            f"Rate limit exceeded for /api/refresh from {client_ip}",
            extra={"extra_fields": {"event": "refresh_rate_limit_exceeded", "ip": client_ip}}
        )
        error_resp = middleware_h.create_error_response("Rate limit exceeded. Try again later.", 429, request_id)
        headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
        headers["Retry-After"] = str(retry_after)
        middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 429, (time.time() - start_time) * 1000, request_id)
        return 429, headers, error_resp

    # Prevent concurrent refreshes
    if not _refresh_lock.acquire(blocking=False):
        common_h.logger.warning(
            "Refresh already in progress",
            extra={"extra_fields": {"event": "refresh_concurrent_blocked"}}
        )
        error_resp = middleware_h.create_error_response("Refresh already in progress", 409, request_id)
        headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
        middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 409, (time.time() - start_time) * 1000, request_id)
        return 409, headers, error_resp

    # Parse request body
    api_key = None
    if request_body:
        try:
            payload = json.loads(request_body.decode("utf-8"))
            api_key = payload.get("api_key")
        except json.JSONDecodeError as e:
            common_h.logger.warning(
                f"Invalid JSON in request body: {e}",
                extra={"extra_fields": {"event": "api_invalid_json", "error": str(e)}}
            )

    # Header takes precedence over body
    header_key = common_h._get_header(request_headers, "X-Groq-Api-Key")
    if header_key:
        api_key = header_key

    # Validate API key format if provided
    if api_key and not api_key.startswith("gsk_"):
        common_h.logger.warning(
            "Invalid API key format (should start with gsk_)",
            extra={"extra_fields": {"event": "api_invalid_key_format"}}
        )

    common_h.logger.info(
        "Pipeline refresh triggered (background)",
        extra={"extra_fields": {
            "event": "pipeline_refresh_triggered",
            "has_api_key": bool(api_key)
        }}
    )

    # ------------------------------------------------------------------
    # TEST / CI MODE: MOCK_PIPELINE=1 çalışır senkron, stub döner.
    # Gerçek pipeline arka plan thread'inde çalışır, HTTP yanıtı anında 202.
    # ------------------------------------------------------------------
    if os.getenv("MOCK_PIPELINE") == "1":
        common_h.logger.info(
            "MOCK_PIPELINE mode: returning stub response",
            extra={"extra_fields": {"event": "pipeline_mock"}}
        )
        mock_resp = {"success": True, "message": "Mock pipeline completed.", "count": 1, "articles": []}
        _refresh_lock.release()
        return common_h._finalize_response(
            request_headers, request_id, "POST", "/api/refresh", 200, mock_resp, start_time
        )

    # Gerçek pipeline: arka plan thread'inde başlat, kilidi buradan serbest bırakma
    # (background thread finally bloğunda bırakır). Render 60s HTTP timeout'u
    # içinde yanıt dönmüş olur; pipeline arka planda çalışmaya devam eder.
    background_thread = threading.Thread(
        target=_run_pipeline_in_background,
        args=(api_key, _refresh_lock, request_id),
        daemon=True,
        name=f"RefreshPipeline-{request_id}"
    )
    background_thread.start()

    headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
    middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 202, (time.time() - start_time) * 1000, request_id)
    return 202, headers, {
        "success": True,
        "message": "Refresh başlatıldı, arka planda çalışıyor. /api/news endpoint'inden takip edebilirsiniz.",
        "status": "running",
        "request_id": request_id
    }