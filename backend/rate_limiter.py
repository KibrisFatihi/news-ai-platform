"""
Rate Limiter for News AI Platform
Implements sliding window rate limiting with proper blocking/backoff.
Thread-safe implementation using Condition variables (no busy-loop).
Supports Retry-After header for 429 responses.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any

from backend.logger import get_logger
from config import GroqConfig, get_config

logger = get_logger("rate_limiter")


@dataclass
class RateLimitWindow:
    """A single rate limit window (e.g., per minute, per day)."""
    max_requests: int
    window_seconds: int
    requests: deque[float] = field(default_factory=deque)
    # Use Condition instead of Lock for efficient waiting
    _condition: threading.Condition = field(default_factory=threading.Condition, init=False)

    def acquire(self, count: int = 1) -> bool:
        """Try to acquire slots in this window. Returns True if successful."""
        with self._condition:
            now = time.time()
            # Remove expired requests
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # Check if we have capacity
            if len(self.requests) + count > self.max_requests:
                return False

            # Reserve slots
            for _ in range(count):
                self.requests.append(now)
            # Notify waiters that slots may be available
            self._condition.notify_all()
            return True

    def get_remaining(self) -> int:
        """Get remaining requests in current window."""
        with self._condition:
            now = time.time()
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            return max(0, self.max_requests - len(self.requests))

    def get_reset_time(self) -> Optional[float]:
        """Get timestamp when oldest request expires (window resets)."""
        with self._condition:
            if not self.requests:
                return None
            return self.requests[0] + self.window_seconds

    def wait_time(self) -> float:
        """Get seconds to wait until a slot is available."""
        with self._condition:
            if len(self.requests) < self.max_requests:
                return 0.0
            if not self.requests:
                return 0.0
            oldest = self.requests[0]
            return max(0.0, oldest + self.window_seconds - time.time())

    def wait_for_slot(self, count: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait efficiently for a slot to become available.
        Uses Condition.wait() instead of busy-loop sleep.
        Returns True if slot acquired, False on timeout.
        """
        with self._condition:
            start_time = time.time()
            while True:
                now = time.time()
                # Clean expired
                cutoff = now - self.window_seconds
                while self.requests and self.requests[0] < cutoff:
                    self.requests.popleft()

                if len(self.requests) + count <= self.max_requests:
                    for _ in range(count):
                        self.requests.append(now)
                    return True

                # Calculate wait time
                if not self.requests:
                    return False  # Should not happen

                wait_until = self.requests[0] + self.window_seconds
                remaining = wait_until - now

                if timeout is not None:
                    elapsed = now - start_time
                    timeout_remaining = timeout - elapsed
                    if timeout_remaining <= 0:
                        return False
                    remaining = min(remaining, timeout_remaining)

                # Wait efficiently - this releases the lock and reacquires on notify/timeout
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                else:
                    # Small yield to avoid tight loop
                    self._condition.wait(timeout=0.01)


class RateLimiter:
    """
    Multi-window rate limiter (e.g., RPM + RPD).
    Thread-safe and configurable via config.py.
    Uses Condition variables for efficient blocking (no busy-loop).
    """

    def __init__(
        self,
        rpm: Optional[int] = None,
        rpd: Optional[int] = None,
        config: Optional[GroqConfig] = None
    ):
        cfg = config or get_config().groq

        self.windows = [
            RateLimitWindow(
                max_requests=rpm or cfg.rate_limit_rpm,
                window_seconds=60  # 1 minute
            ),
            RateLimitWindow(
                max_requests=rpd or cfg.rate_limit_rpd,
                window_seconds=86400  # 1 day
            )
        ]
        self._lock = threading.Lock()
        self._total_requests = 0
        self._rejected_requests = 0

    def acquire(self, count: int = 1, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Try to acquire rate limit slots.

        Args:
            count: Number of slots to acquire (default 1)
            blocking: If True, wait efficiently until slots available
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if acquired, False if not acquired (non-blocking) or timeout
        """
        start_time = time.time()

        if not blocking:
            # Non-blocking: try all windows once
            for window in self.windows:
                if not window.acquire(count):
                    # Rollback any acquired
                    for w in self.windows:
                        with w._condition:
                            for _ in range(count):
                                if w.requests:
                                    w.requests.pop()
                    with self._lock:
                        self._rejected_requests += count
                    return False
            with self._lock:
                self._total_requests += count
            return True

        # Blocking: wait efficiently for all windows
        while True:
            acquired_windows = []
            all_acquired = True

            for window in self.windows:
                # Try to acquire in this window
                if window.acquire(count):
                    acquired_windows.append(window)
                else:
                    all_acquired = False
                    break

            if all_acquired:
                with self._lock:
                    self._total_requests += count
                return True

            # Rollback acquired windows
            for window in acquired_windows:
                with window._condition:
                    for _ in range(count):
                        if window.requests:
                            window.requests.pop()

            # Wait for the window that needs the longest wait
            max_wait = max(w.wait_time() for w in self.windows)

            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    with self._lock:
                        self._rejected_requests += count
                    return False
                max_wait = min(max_wait, remaining)

            if max_wait > 0:
                logger.debug(
                    f"Rate limit reached, waiting {max_wait:.2f}s",
                    extra={"extra_fields": {
                        "event": "rate_limit_wait",
                        "wait_seconds": max_wait,
                        "count": count
                    }}
                )
                # Use the first window's condition to wait efficiently
                # We wait on the window with the longest wait time
                longest_window = max(self.windows, key=lambda w: w.wait_time())
                with longest_window._condition:
                    longest_window._condition.wait(timeout=max_wait)
            else:
                # Should not happen, but safety check
                time.sleep(0.01)

    def get_retry_after(self) -> int:
        """
        Get Retry-After seconds for 429 response.
        Returns the maximum wait time across all windows.
        """
        return int(max(w.wait_time() for w in self.windows)) + 1

    def get_status(self) -> dict:
        """Get current rate limiter status for monitoring."""
        return {
            "windows": [
                {
                    "max_requests": w.max_requests,
                    "window_seconds": w.window_seconds,
                    "current_usage": w.max_requests - w.get_remaining(),  # This cleans up expired entries first
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
            with window._condition:
                window.requests.clear()
        with self._lock:
            self._total_requests = 0
            self._rejected_requests = 0


# Global rate limiter instance for Groq API
_groq_rate_limiter: Optional[RateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_groq_rate_limiter() -> RateLimiter:
    """Get or create the global Groq API rate limiter."""
    global _groq_rate_limiter
    with _rate_limiter_lock:
        if _groq_rate_limiter is None:
            _groq_rate_limiter = RateLimiter()
        return _groq_rate_limiter


def reset_groq_rate_limiter():
    """Reset the global rate limiter (for testing)."""
    global _groq_rate_limiter
    with _rate_limiter_lock:
        _groq_rate_limiter = None


# --- Async-compatible rate limiter ---

class AsyncRateLimiter:
    """
    Async-compatible rate limiter using asyncio.
    For use with async HTTP clients (aiohttp, httpx).
    """

    def __init__(self, rpm: int = 30, rpd: int = 14400):
        self.rpm_window = RateLimitWindow(rpm, 60)
        self.rpd_window = RateLimitWindow(rpd, 86400)
        self._semaphore: Optional[Any] = None  # Will be initialized on first use

    async def acquire(self, count: int = 1) -> bool:
        """Acquire slots asynchronously."""
        import asyncio

        # Initialize semaphore on first use (needs running event loop)
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(1)

        async with self._semaphore:
            while True:
                # Try both windows
                rpm_ok = self.rpm_window.acquire(count)
                rpd_ok = self.rpd_window.acquire(count)

                if rpm_ok and rpd_ok:
                    return True

                # Wait for the longest window
                wait_time = max(self.rpm_window.wait_time(), self.rpd_window.wait_time())
                if wait_time > 0:
                    await asyncio.sleep(min(wait_time, 0.1))
                else:
                    await asyncio.sleep(0.01)

    def get_status(self) -> dict:
        return {
            "rpm": {
                "max": self.rpm_window.max_requests,
                "used": len(self.rpm_window.requests),
                "remaining": self.rpm_window.get_remaining()
            },
            "rpd": {
                "max": self.rpd_window.max_requests,
                "used": len(self.rpd_window.requests),
                "remaining": self.rpd_window.get_remaining()
            }
        }