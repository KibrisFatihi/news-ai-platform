"""
POST /api/refresh handler.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Dict, Tuple

from backend.models import Article, RefreshResponse
from backend.handlers import common as common_h
from backend import middleware as middleware_h

# Refresh protection lock
_refresh_lock = threading.Lock()
_refresh_in_progress = False


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
    global _refresh_in_progress
    if not _refresh_lock.acquire(blocking=False):
        common_h.logger.warning(
            "Refresh already in progress",
            extra={"extra_fields": {"event": "refresh_concurrent_blocked"}}
        )
        error_resp = middleware_h.create_error_response("Refresh already in progress", 409, request_id)
        headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
        middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 409, (time.time() - start_time) * 1000, request_id)
        return 409, headers, error_resp

    try:
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
            "Pipeline refresh triggered",
            extra={"extra_fields": {
                "event": "pipeline_refresh_triggered",
                "has_api_key": bool(api_key)
            }}
        )

        # ------------------------------------------------------------------
        # TEST / CI MODE: MOCK_PIPELINE=1 skips the real pipeline so E2E
        # tests can verify endpoint auth, routing and response format without
        # waiting for the full NLP + summarization pipeline (~90-120 s).
        # ------------------------------------------------------------------
        if os.getenv("MOCK_PIPELINE") == "1":
            common_h.logger.info(
                "MOCK_PIPELINE mode: returning stub response",
                extra={"extra_fields": {"event": "pipeline_mock"}}
            )
            mock_resp = {"success": True, "message": "Mock pipeline completed.", "count": 1, "articles": []}
            return common_h._finalize_response(
                request_headers, request_id, "POST", "/api/refresh", 200, mock_resp, start_time
            )

        # Import here to avoid circular import
        from main import run_pipeline
        from config import get_config

        # Run the pipeline with a hard timeout so a hung RSS feed or Groq call
        # cannot hold the refresh lock (and the browser "Taranıyor..." state)
        # indefinitely.
        pipeline_timeout_seconds = get_config().server.pipeline_timeout_seconds
        try:
            with ThreadPoolExecutor(max_workers=1) as pipeline_executor:
                future = pipeline_executor.submit(run_pipeline, api_key=api_key)
                articles = future.result(timeout=pipeline_timeout_seconds) or []

            common_h.logger.info(
                "Pipeline refresh completed",
                extra={"extra_fields": {
                    "event": "pipeline_refresh_completed",
                    "article_count": len(articles)
                }}
            )
        except FutureTimeoutError:
            common_h.logger.error(
                f"Pipeline refresh timed out after {pipeline_timeout_seconds} seconds",
                extra={"extra_fields": {"event": "pipeline_refresh_timeout"}}
            )
            error_resp = middleware_h.create_error_response(
                "Refresh timed out. Sunucu yanıt vermedi, lütfen tekrar deneyin.",
                504,
                request_id
            )
            headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
            middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 504, (time.time() - start_time) * 1000, request_id)
            return 504, headers, error_resp

        try:
            # Normalize articles to ensure all required fields for Article model
            normalized_articles = [common_h._normalize_article(a) for a in articles]

            response_data = RefreshResponse(
                success=True,
                message="Haber akışı ve yapay zeka özetleri başarıyla güncellendi.",
                count=len(normalized_articles),
                articles=[Article(**a) for a in normalized_articles]
            ).model_dump()

            return common_h._finalize_response(
                request_headers, request_id, "POST", "/api/refresh", 200, response_data, start_time
            )
        except Exception as e:
            common_h.logger.exception(
                f"Pipeline execution failed [{request_id}]: {e}",
                extra={"extra_fields": {"event": "pipeline_error", "error": str(e), "request_id": request_id}}
            )
            error_resp = middleware_h.create_error_response("Internal server error", 500, request_id)
            headers = middleware_h.get_cors_headers(common_h._get_header(request_headers, "Origin"))
            middleware_h.log_api_request(common_h.logger, "POST", "/api/refresh", 500, (time.time() - start_time) * 1000, request_id)
            return 500, headers, error_resp

    finally:
        _refresh_lock.release()