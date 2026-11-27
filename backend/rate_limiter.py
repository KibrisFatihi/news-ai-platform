"""
Rate Limiter for News AI Platform
Simplified: non-blocking sliding window only (no wait/blocking).
Thread-safe, per-model instances for Groq API.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from backend.logger import get_logger
from config import GroqConfig, get_config

logger = get_logger("rate_limiter")


@dataclass
class RateLimitWindow:
    """A single rate limit window (e.g., per minute, per day)."""
    max_requests: int
    window_seconds: int
    requests: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def acquire(self, count: int = 1) -> bool:
        """Try to acquire slots in this window. Returns True if successful."""
        with self._lock:
            now = time.time()
            # Remove expired requests
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            if len(self.requests) + count > self.max_requests:
                return False

            for _ in range(count):
                self.requests.append(now)
            return True

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            return max(0, self.max_requests - len(self.requests))

    def get_reset_time(self) -> Optional[float]:
        """Get timestamp when oldest request expires (window resets)."""
        with self._lock:
            if not self.requests:
                return None
            return self.requests[0] + self.window_seconds

    def wait_time(self) -> float:
        """Get seconds to wait until a slot is available."""
        with self._lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            if not self.requests:
                return 0.0
            return max(0.0, self.requests[0] + self.window_seconds - time.time())


class RateLimiter:
    """
    Multi-window rate limiter (e.g., RPM + RPD).
    Non-blocking only - returns False immediately if limit exceeded.
    Thread-safe and configurable via config.py.
    """

    __slots__ = ("windows", "_total_requests", "_rejected_requests", "_lock")

    def __init__(
        self,
        rpm: Optional[int] = None,
        rpd: Optional[int] = None,
        config: Optional[GroqConfig] = None
    ):
        cfg = config or get_config().groq

        self.windows = [
            RateLimitWindow(max_requests=rpm or cfg.rate_limit_rpm, window_seconds=60),
            RateLimitWindow(max_requests=rpd or cfg.rate_limit_rpd, window_seconds=86400),
        ]
        self._total_requests = 0
        self._rejected_requests = 0
        self._lock = threading.Lock()

    def acquire(self, count: int = 1, blocking: bool = False, timeout: Optional[float] = None) -> bool:
        """
        Try to acquire rate limit slots (non-blocking).

        Args:
            count: Number of slots to acquire (default 1)
            blocking: Ignored - kept for API compatibility, always non-blocking
            timeout: Ignored - kept for API compatibility

        Returns:
            True if acquired, False if rate limited
        """
        # Try all windows
        for window in self.windows:
            if not window.acquire(count):
                # Rollback any acquired
                for w in self.windows:
                    with w._lock:
                        for _ in range(count):
                            if w.requests:
                                w.requests.pop()
                with self._lock:
                    self._rejected_requests += count
                return False

        with self._lock:
            self._total_requests += count
        return True

    def get_retry_after(self) -> int:
        """Get Retry-After seconds for 429 response."""
        return int(max(w.wait_time() for w in self.windows)) + 1

    def get_status(self) -> dict:
        """Get current rate limiter status for monitoring."""
        return {
            "windows": [
                {
                    "max_requests": w.max_requests,
                    "window_seconds": w.window_seconds,
                    "current_usage": w.max_requests - w.get_remaining(),
                    "remaining": w.get_remaining(),
                    "reset_time": w.get_reset_time()
                }
                for w in self.windows
            ],
            "total_requests": self._total_requests,
            "rejected_requests": self._rejected_requests
        }

    def reset(self):
        """Reset all windows (for testing)."""
        for window in self.windows:
            with window._lock:
                window.requests.clear()
        with self._lock:
            self._total_requests = 0
            self._rejected_requests = 0


# Global rate limiter instances for Groq API (per-model)
_groq_rate_limiters: dict[str, RateLimiter] = {}
_rate_limiter_lock = threading.Lock()


def get_groq_rate_limiter(model_name: Optional[str] = None) -> RateLimiter:
    """Get or create a Groq API rate limiter (per-model for independent quotas)."""
    key = model_name or "__default__"
    with _rate_limiter_lock:
        if key not in _groq_rate_limiters:
            _groq_rate_limiters[key] = RateLimiter()
        return _groq_rate_limiters[key]


def reset_groq_rate_limiter(model_name: Optional[str] = None):
    """Reset the global or per-model rate limiter (for testing)."""
    global _groq_rate_limiters
    with _rate_limiter_lock:
        if model_name is None:
            _groq_rate_limiters = {}
        else:
            key = model_name or "__default__"
            _groq_rate_limiters.pop(key, None)