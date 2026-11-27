"""
Lightweight REST API & Static Server for AI News Platform
Built with Python standard library (http.server) - Zero extra dependencies required.

Architecture:
- api.py: Router (this file - main entry point)
- middleware.py: CORS, Request-ID, Error Handling, Rate Limiting, Conditional Requests
- handlers.py: Business logic for each endpoint

Endpoints:
- GET  /api/news        -> Fetch current aggregated news feed with metadata
- POST /api/refresh     -> Trigger live crawling & Groq/NLP summarization
- GET  /api/status      -> Get backend status and Groq AI key configuration
- GET  /api/health      -> Health check with detailed status
- Static: / or /frontend/ -> Serves modern web frontend
"""
from __future__ import annotations

import http.server
import json
import logging
import os
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure UTF-8 encoding on Windows for emoji support in logs
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", write_through=True)

from backend.handlers import (
    handle_get_health,
    handle_get_news,
    handle_get_status,
    handle_get_weather,
    handle_not_found,
    handle_options,
    handle_post_refresh,
)
from backend.logger import get_logger, setup_logging
from backend.middleware import generate_request_id
from config import get_config

# Initialize logging
cfg = get_config()
server_cfg = cfg.server
setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file)

logger = get_logger("server")

# Background scheduler state
_scheduler_thread: Optional[threading.Thread] = None
_scheduler_stop_event: Optional[threading.Event] = None


def _run_background_pipeline():
    """Background task that runs the pipeline periodically."""
    global _scheduler_stop_event
    interval = get_config().server.scheduler_interval_seconds
    logger.info(f"🕐 Arka plan zamanlayıcı başlatıldı (her {interval // 60} dakikada bir yenileme)")

    while _scheduler_stop_event and not _scheduler_stop_event.is_set():
        try:
            # Import here to avoid circular import
            from main import run_pipeline
            logger.info("⏰ Zamanlanmış pipeline çalıştırılıyor...")
            run_pipeline()
            logger.info("✅ Zamanlanmış pipeline başarıyla tamamlandı.")
        except Exception as e:
            logger.error(f"❌ Arka plan pipeline hatası: {e}")

        # Wait for interval or stop signal
        if _scheduler_stop_event:
            _scheduler_stop_event.wait(interval)

    logger.info("🛑 Arka plan zamanlayıcı durduruldu.")


def start_scheduler():
    """Start the background scheduler thread."""
    global _scheduler_thread, _scheduler_stop_event
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.warning("Zamanlayıcı zaten çalışıyor.")
        return

    _scheduler_stop_event = threading.Event()
    _scheduler_thread = threading.Thread(target=_run_background_pipeline, daemon=True, name="NewsPipelineScheduler")
    _scheduler_thread.start()
    logger.info("📅 Zamanlayıcı thread'i başlatıldı.")


def stop_scheduler():
    """Stop the background scheduler thread."""
    global _scheduler_thread, _scheduler_stop_event
    if _scheduler_stop_event:
        _scheduler_stop_event.set()
    if _scheduler_thread and _scheduler_thread.is_alive():
        _scheduler_thread.join(timeout=5.0)
        logger.info("Zamanlayıcı thread'i güvenle durduruldu.")


class NewsAPIRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with REST API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(cfg.frontend_dir), **kwargs)
    

    def handle_one_request(self):
        super().handle_one_request()

    def _send_response(
        self,
        status_code: int,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send HTTP response with headers and optional JSON body."""
        if body is not None:
            response_bytes = json.dumps(body, ensure_ascii=False, indent=2).encode("utf-8")
        else:
            response_bytes = b""

        self.send_response(status_code)

        # Content headers
        if response_bytes:
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_bytes)))
        else:
            self.send_header("Content-Length", "0")

        # Custom headers
        for key, value in headers.items():
            self.send_header(key, value)

        self.end_headers()
        if response_bytes:
            self.wfile.write(response_bytes)

    def end_headers(self):
        """Stik dosya yanıtlarına önbellek-devre dışı başlığı ekle.

        /api/ endpoint'lerinin zaten ayarladığı Cache-Control/ETag/Last-Modified
        başlıklarıyla çakışmasın diye sadece statik dosya yanıtlarına uygulanır.
        """
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        request_id = generate_request_id()
        request_start = time.time()
        origin = self.headers.get("Origin", "")

        status, headers, _ = handle_options({"Origin": origin})
        self._send_response(status, headers)
        self.log_request_with_timing("OPTIONS", self.path, status, request_start, request_id)

    def do_GET(self):
        """Handle GET requests."""
        request_id = generate_request_id()
        request_start = time.time()
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        try:
            # Redirect root to frontend
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/index.html")
                self.end_headers()
                self.log_request_with_timing("GET", path, 302, request_start, request_id)
                return

            # API: Get News
            if path == "/api/news":
                headers_dict = dict(self.headers)
                status, headers, body = handle_get_news(headers_dict, request_id)
                self._send_response(status, headers, body)
                self.log_request_with_timing("GET", path, status, request_start, request_id)
                return

            # API: Status check
            if path == "/api/status":
                headers_dict = dict(self.headers)
                status, headers, body = handle_get_status(headers_dict, request_id)
                self._send_response(status, headers, body)
                self.log_request_with_timing("GET", path, status, request_start, request_id)
                return

            # API: Health check
            if path == "/api/health":
                headers_dict = dict(self.headers)
                status, headers, body = handle_get_health(headers_dict, request_id)
                self._send_response(status, headers, body)
                self.log_request_with_timing("GET", path, status, request_start, request_id)
                return

            # API: Turkey Weather
            if path == "/api/weather":
                headers_dict = dict(self.headers)
                status, headers, body = handle_get_weather(headers_dict, request_id)
                self._send_response(status, headers, body)
                self.log_request_with_timing("GET", path, status, request_start, request_id)
                return

            # Default: Serve static files (HTML, JS, CSS)
            super().do_GET()
            self.log_request_with_timing("GET", path, 200, request_start, request_id)

        except Exception as e:
            logger.error(
                f"GET {path} error [{request_id}]: {e}",
                extra={"extra_fields": {"event": "api_error", "method": "GET", "path": path, "error": str(e), "request_id": request_id}}
            )
            self._send_response(500, {}, {"success": False, "error": "Internal server error", "request_id": request_id})
            self.log_request_with_timing("GET", path, 500, request_start, request_id)

    def do_POST(self):
        """Handle POST requests."""
        request_id = generate_request_id()
        request_start = time.time()
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        client_ip = self.client_address[0]

        logger.info(f"POST request: path={path}")

        try:
            if path == "/api/refresh":
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length) if content_length > 0 else b""

                headers_dict = dict(self.headers)
                status, headers, body = handle_post_refresh(headers_dict, post_data, client_ip, request_id)
                self._send_response(status, headers, body)
                self.log_request_with_timing("POST", path, status, request_start, request_id)
                return

            # Unknown POST endpoint
            status, headers, body = handle_not_found(dict(self.headers), request_id)
            self._send_response(status, headers, body)
            self.log_request_with_timing("POST", path, status, request_start, request_id)

        except Exception as e:
            logger.error(
                f"POST {path} error [{request_id}]: {e}",
                extra={"extra_fields": {"event": "api_error", "method": "POST", "path": path, "error": str(e), "request_id": request_id}}
            )
            self._send_response(500, {}, {"success": False, "error": "Internal server error", "request_id": request_id})
            self.log_request_with_timing("POST", path, 500, request_start, request_id)

    def log_request_with_timing(
        self,
        method: str,
        path: str,
        status_code: int,
        request_start: float,
        request_id: str
    ) -> None:
        """Log request with timing."""
        duration_ms = (time.time() - request_start) * 1000
        from backend.middleware import log_api_request
        log_api_request(logger, method, path, status_code, duration_ms, request_id)

    def log_message(self, format: str, *args) -> None:
        """Override to use our logger instead of stderr."""
        logger.debug(format % args)


def start_server():
    """Start the HTTP server."""
    os.chdir(cfg.base_dir)
    server_address = (server_cfg.host, server_cfg.port)

    # Enable address reuse for faster restarts
    socketserver.TCPServer.allow_reuse_address = False

    # NOTE: We intentionally do NOT call socket.setdefaulttimeout() here.
    # A global socket timeout would break the long-lived HTTP connection
    # that the browser holds open while /api/refresh runs the full pipeline
    # (RSS fetch + NLP + summarization can take 60-120s).
    # RSS fetch timeout is applied per-request via urllib.request.urlopen(timeout=).

    # Start background scheduler for automatic RSS refresh (unless disabled in tests)
    if os.environ.get("DISABLE_SCHEDULER") != "1":
        start_scheduler()

    # Use ThreadingHTTPServer for concurrent request handling (Python 3.7+)
    with http.server.ThreadingHTTPServer(server_address, NewsAPIRequestHandler) as httpd:
        logger.info("=" * 60)
        logger.info("AI News Platform Canli REST API & Web Sunucusu Baslatildi!")
        logger.info(f"Web Paneli: http://{server_cfg.host if server_cfg.host != '0.0.0.0' else 'localhost'}:{server_cfg.port}/index.html")
        logger.info(f"REST API:   http://localhost:{server_cfg.port}/api/news")
        logger.info(f"Yenile API: http://localhost:{server_cfg.port}/api/refresh")
        logger.info(f"Health API: http://localhost:{server_cfg.port}/api/health")
        logger.info(f"Status API: http://localhost:{server_cfg.port}/api/status")
        logger.info("🕐 Otomatik yenileme: Her 30 dakikada bir (arka planda)")
        logger.info("=" * 60)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\nSunucu durduruluyor...")
        finally:
            # Stop background scheduler on shutdown
            stop_scheduler()
            logger.info("Sunucu durduruldu.")


if __name__ == "__main__":
    start_server()