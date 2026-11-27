"""
GET /api/health handler.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, Tuple

from backend.models import HealthResponse
from backend.handlers import common as common_h
from backend.middleware import format_http_date
from backend.groq_service import get_groq_status
from backend.circuit_breaker import get_all_circuit_breakers


def handle_get_health(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle GET /api/health request."""
    start_time = time.time()

    repo = common_h.get_news_repository()
    articles = repo.load_or_fallback()

    groq_status = get_groq_status()
    circuit_breakers = get_all_circuit_breakers()

    response_data = HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="2.0.0",
        uptime_seconds=time.time() - common_h._server_start_time,
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

    return common_h._finalize_response(
        request_headers, request_id, "GET", "/api/health", 200, response_data.model_dump(), start_time
    )