"""
Middleware for HTTP Server
Handles: CORS, Request-ID, Error Handling, Rate Limiting, Conditional Requests
"""
from __future__ import annotations

import email.utils
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import get_config

cfg = get_config()
server_cfg = cfg.server


# --- Request ID Generation ---

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex[:8]


# --- CORS Handling ---

def get_cors_headers(origin: str) -> Dict[str, str]:
    """Get CORS headers for a given origin."""
    allowed_origins = server_cfg.cors_allowed_origins
    headers = {}

    if "*" in allowed_origins or origin in allowed_origins:
        headers["Access-Control-Allow-Origin"] = origin if origin else "*"
    else:
        # Origin not allowed - still return headers but without ACAO
        pass

    headers["Access-Control-Allow-Methods"] = ", ".join(server_cfg.cors_allowed_methods)
    headers["Access-Control-Allow-Headers"] = ", ".join(server_cfg.cors_allowed_headers)
    headers["Access-Control-Max-Age"] = "86400"  # 24 hours

    return headers


def handle_options_request(origin: str) -> Tuple[int, Dict[str, str]]:
    """Handle OPTIONS preflight request."""
    headers = get_cors_headers(origin)
    return 200, headers


# --- Conditional Request Helpers ---

def compute_etag(data: Any) -> str:
    """Compute ETag from JSON data."""
    json_str = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()


def format_http_date(timestamp: float) -> str:
    """Format timestamp as HTTP date (RFC 7231) - UTC."""
    return datetime.utcfromtimestamp(timestamp).strftime("%a, %d %b %Y %H:%M:%S GMT")


def parse_http_date(date_str: str) -> Optional[float]:
    """
    Parse HTTP date string (RFC 7231) using email.utils for timezone awareness.
    Returns timestamp in UTC or None if parsing fails.
    """
    try:
        # email.utils.parsedate_to_datetime handles timezone-aware parsing
        dt = email.utils.parsedate_to_datetime(date_str)
        if dt.tzinfo is not None:
            # Use calendar.timegm for UTC timetuple (not time.mktime which assumes local time)
            import calendar
            return calendar.timegm(dt.utctimetuple())
        return dt.timestamp()
    except Exception:
        return None


def check_conditional_request(
    headers: Dict[str, str],
    etag: Optional[str],
    last_modified: Optional[float]
) -> bool:
    """
    Check If-None-Match and If-Modified-Since headers.
    Returns True if 304 Not Modified should be sent.
    """
    # If-None-Match (ETag)
    if_none_match = headers.get("If-None-Match")
    if if_none_match and etag:
        client_etag = if_none_match.strip('"')
        if client_etag == etag:
            return True

    # If-Modified-Since
    if_modified_since = headers.get("If-Modified-Since")
    if if_modified_since and last_modified:
        client_time = parse_http_date(if_modified_since)
        if client_time is not None and client_time >= last_modified:
            return True

    return False


def add_cache_headers(
    headers: Dict[str, str],
    etag: Optional[str] = None,
    last_modified: Optional[float] = None,
    cache_control: Optional[str] = None
) -> None:
    """Add caching headers to response."""
    if server_cfg.etag_enabled and etag:
        headers["ETag"] = f'"{etag}"'
    if last_modified:
        headers["Last-Modified"] = format_http_date(last_modified)
    if cache_control:
        headers["Cache-Control"] = cache_control
    else:
        headers["Cache-Control"] = "no-cache, no-store, must-revalidate"


# --- Rate Limiting for /api/refresh ---

_refresh_rate_limit: defaultdict[str, List[float]] = defaultdict(list)
_refresh_rate_limit_lock = threading.Lock()
REFRESH_RATE_LIMIT_WINDOW = server_cfg.refresh_rate_limit_window
REFRESH_RATE_LIMIT_MAX = server_cfg.refresh_rate_limit_max


def check_refresh_rate_limit(client_ip: str) -> Tuple[bool, int]:
    """
    Check IP-based rate limit for /api/refresh.
    Returns (allowed, retry_after_seconds).
    """
    now = time.time()
    with _refresh_rate_limit_lock:
        # Clean old entries
        cutoff = now - REFRESH_RATE_LIMIT_WINDOW
        _refresh_rate_limit[client_ip] = [
            t for t in _refresh_rate_limit[client_ip] if t > cutoff
        ]

        if len(_refresh_rate_limit[client_ip]) >= REFRESH_RATE_LIMIT_MAX:
            # Calculate retry-after
            oldest = _refresh_rate_limit[client_ip][0]
            retry_after = int(oldest + REFRESH_RATE_LIMIT_WINDOW - now) + 1
            return False, max(1, retry_after)

        _refresh_rate_limit[client_ip].append(now)
        return True, 0


# --- Refresh Token Verification ---

_DEFAULT_REFRESH_SECRET = secrets.token_hex(32)


def get_refresh_token_secret() -> str:
    """Get refresh token secret from environment or process-wide fallback."""
    return os.environ.get("REFRESH_TOKEN_SECRET") or _DEFAULT_REFRESH_SECRET


def generate_refresh_token() -> str:
    """Generate a valid HMAC refresh token."""
    secret = get_refresh_token_secret()
    return hmac.new(
        secret.encode(),
        b"refresh",
        hashlib.sha256
    ).hexdigest()


def verify_refresh_token(token: Optional[str]) -> bool:
    """Verify refresh token using HMAC."""
    if not token:
        return False
    expected = generate_refresh_token()
    return hmac.compare_digest(token, expected)


# --- Error Response Helpers ---

def create_error_response(
    message: str,
    status_code: int = 500,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "success": False,
        "error": message,
        "request_id": request_id or generate_request_id()
    }


# --- Logging Helpers ---

def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str,
    **extra
) -> None:
    """Log API request with structured fields."""
    logger.info(
        f"{method} {path} -> {status_code}",
        extra={"extra_fields": {
            "event": "api_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "request_id": request_id,
            **extra
        }}
    )