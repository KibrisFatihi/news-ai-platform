"""
Circuit Breaker Pattern for News AI Platform
Simplified: fail-fast when failure threshold exceeded, auto-recovery after timeout.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from backend.logger import get_logger
from config import GroqConfig

logger = get_logger("circuit_breaker")


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5      # Failures before opening
    timeout_seconds: float = 60.0   # Time before allowing retry


class CircuitOpenError(Exception):
    """Raised when circuit breaker is OPEN and rejects a call."""
    pass


class CircuitBreaker:
    """
    Simplified circuit breaker: CLOSED -> OPEN -> (timeout) -> CLOSED.
    No HALF_OPEN state - just fails fast when open, auto-recovers after timeout.
    Thread-safe.
    """

    __slots__ = ("name", "config", "_state", "_failure_count", "_last_failure_time", "_lock", "stats")

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        groq_config: Optional[GroqConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig(
            failure_threshold=groq_config.circuit_failure_threshold if groq_config else 5,
            timeout_seconds=groq_config.circuit_timeout_seconds if groq_config else 60.0
        )
        self._state = "closed"  # "closed" or "open"
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        self.stats = {"total": 0, "success": 0, "failure": 0, "rejected": 0}

    @property
    def state(self) -> str:
        """Get current state, checking for auto-recovery timeout."""
        with self._lock:
            if self._state == "open" and self._last_failure_time:
                if time.time() - self._last_failure_time >= self.config.timeout_seconds:
                    self._state = "closed"
                    self._failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}': auto-recovered to CLOSED")
            return self._state

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection. Fails fast if OPEN."""
        # Check state (may auto-recover from OPEN to CLOSED)
        if self.state == "open":
            self.stats["rejected"] += 1
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")

        self.stats["total"] += 1
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            self.stats["success"] += 1
            if self._state == "closed":
                self._failure_count = 0  # Reset on success

    def _on_failure(self):
        with self._lock:
            self.stats["failure"] += 1
            self._last_failure_time = time.time()
            if self._state == "closed":
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._state = "open"
                    logger.warning(
                        f"Circuit breaker '{self.name}': OPEN (failures={self._failure_count})",
                        extra={"extra_fields": {"event": "circuit_open", "breaker": self.name, "failures": self._failure_count}}
                    )

    def get_status(self) -> dict:
        """Get circuit breaker status for monitoring."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failure_count": self._failure_count,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "timeout_seconds": self.config.timeout_seconds
                },
                "stats": dict(self.stats)
            }

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def force_open(self):
        """Manually force circuit breaker to OPEN state."""
        with self._lock:
            self._state = "open"
            self._last_failure_time = time.time()
            logger.warning(f"Circuit breaker '{self.name}' manually forced OPEN")


# Global circuit breaker registry
_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    groq_config: Optional[GroqConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(name, config, groq_config)
        return _breakers[name]


def get_all_circuit_breakers() -> dict[str, dict]:
    """Get status of all circuit breakers."""
    with _breakers_lock:
        return {name: cb.get_status() for name, cb in _breakers.items()}


def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    with _breakers_lock:
        for cb in _breakers.values():
            cb.reset()