"""
GET /api/weather handler.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from backend.handlers import common as common_h
from backend.weather import get_turkey_weather


def handle_get_weather(
    request_headers: Dict[str, str],
    request_id: str,
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle GET /api/weather request."""
    start_time = time.time()
    payload = get_turkey_weather()
    return common_h._finalize_response(
        request_headers,
        request_id,
        "GET",
        "/api/weather",
        200,
        payload,
        start_time,
        cache_control="public, max-age=300",
    )
