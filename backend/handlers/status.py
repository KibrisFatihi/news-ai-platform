"""
GET /api/status handler.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from backend.models import StatusResponse
from backend.handlers import common as common_h
from backend.middleware import generate_refresh_token


def handle_get_status(
    request_headers: Dict[str, str],
    request_id: str
) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
    """Handle GET /api/status request."""
    start_time = time.time()

    groq_key = common_h.cfg.get_groq_api_key()

    # Groq model(ler)ini status yanıtına yaz. Rotasyon yapılandırması
    # bozuksa (ör. testlerde cfg mock edilmişse) default modele düş.
    groq_cfg = common_h.cfg.groq
    rotation = []
    if getattr(groq_cfg, "model_rotation_enabled", False):
        try:
            rotation = list(groq_cfg.model_rotation)
        except TypeError:
            rotation = []

    response_data = StatusResponse(
        status="online",
        version="2.0.0",
        groq_connected=bool(groq_key),
        model=(
            " → ".join(str(m) for m in rotation)
            if len(rotation) > 1
            else (rotation[0] if rotation else getattr(groq_cfg, "default_model", ""))
        ),
        mode="hybrid",
        uptime_seconds=time.time() - common_h._server_start_time,
        refresh_token=generate_refresh_token()
    ).model_dump()

    return common_h._finalize_response(
        request_headers, request_id, "GET", "/api/status", 200, response_data, start_time
    )