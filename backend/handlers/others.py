"""
OPTIONS and 404 handlers.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from backend.handlers.common import (
    _build_cors_headers,
    _get_header,
    logger,
)
from backend.middleware import (
    create_error_response,
    handle_options_request,
)


def handle_options(
    request_headers: Dict[str, str]
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle OPTIONS preflight request."""
    origin = _get_header(request_headers, "Origin")
    status, headers = handle_options_request(origin)
    return status, headers, {}


def handle_not_found(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle 404 for unknown endpoints."""
    error_resp = create_error_response("Endpoint not found", 404, request_id)
    headers = _build_cors_headers(request_headers)
    return 404, headers, error_resp