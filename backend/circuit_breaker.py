"""
Circuit Breaker Pattern for News AI Platform
Prevents cascade failures when Groq API is unavailable.
Implements: CLOSED -> OPEN -> HALF_OPEN state machine.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Generic, Optional, TypeVar

from backend.logger import get_logger
from config import GroqConfig

logger = get_logger("circuit_breaker")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests go through
    OPEN = "open"          # Failing, requests blocked immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes in half-open before closing
    timeout_seconds: float = 60.0       # Time before half-open
    excluded_exceptions: tuple = ()     # Exceptions that don't count as failures
    expected_exceptions: tuple = (Exception,)  # Exceptions that count as failures


@dataclass
class CircuitBreakerStats:
    """Runtime statistics for monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: Optional[float] = None
    current_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker implementation for protecting external service calls.

    State transitions:
    - CLOSED: Normal operation. Failures counted. On threshold -> OPEN
    - OPEN: Requests fail fast. After timeout -> HALF_OPEN
    - HALF_OPEN: Test requests allowed. On success_threshold -> CLOSED, on failure -> OPEN
    """

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
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()
        self.stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for timeout transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and \
                   time.time() - self._last_failure_time >= self.config.timeout_seconds:
                    self._transition_to_half_open()
            return self._state

    def _transition_to_open(self):
        """Transition to OPEN state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
        self.stats.current_state = CircuitState.OPEN
        self.stats.state_changes += 1
        self.stats.last_state_change = time.time()
        logger.warning(
            f"Circuit breaker '{self.name}': {old_state.value} -> OPEN",
            extra={"extra_fields": {
                "event": "circuit_breaker_state_change",
                "breaker": self.name,
                "from_state": old_state.value,
                "to_state": "open",
                "failure_count": self._failure_count
            }}
        )

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self.stats.current_state = CircuitState.HALF_OPEN
        self.stats.state_changes += 1
        self.stats.last_state_change = time.time()
        logger.info(
            f"Circuit breaker '{self.name}': {old_state.value} -> HALF_OPEN",
            extra={"extra_fields": {
                "event": "circuit_breaker_state_change",
                "breaker": self.name,
                "from_state": old_state.value,
                "to_state": "half_open"
            }}
        )

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self.stats.current_state = CircuitState.CLOSED
        self.stats.state_changes += 1
        self.stats.last_state_change = time.time()
        logger.info(
            f"Circuit breaker '{self.name}': {old_state.value} -> CLOSED",
            extra={"extra_fields": {
                "event": "circuit_breaker_state_change",
                "breaker": self.name,
                "from_state": old_state.value,
                "to_state": "closed"
            }}
        )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function return value

        Raises:
            CircuitOpenError: If circuit is OPEN
            Original exception: If function fails
        """
        # Check state (may transition from OPEN to HALF_OPEN)
        current_state = self.state

        if current_state == CircuitState.OPEN:
            self.stats.rejected_calls += 1
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Retry after {self.config.timeout_seconds}s."
            )

        self.stats.total_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.excluded_exceptions:
            # Excluded exceptions don't count as failures
            self._on_success()
            raise
        except self.config.expected_exceptions:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def get_status(self) -> dict:
        """Get circuit breaker status for monitoring."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout_seconds": self.config.timeout_seconds
                },
                "stats": {
                    "total_calls": self.stats.total_calls,
                    "successful_calls": self.stats.successful_calls,
                    "failed_calls": self.stats.failed_calls,
                    "rejected_calls": self.stats.rejected_calls,
                    "state_changes": self.stats.state_changes,
                    "last_failure_time": self.stats.last_failure_time,
                    "last_success_time": self.stats.last_success_time,
                    "last_state_change": self.stats.last_state_change
                }
            }

    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self.stats.current_state = CircuitState.CLOSED
            logger.info(
                f"Circuit breaker '{self.name}' manually reset: {old_state.value} -> CLOSED",
                extra={"extra_fields": {
                    "event": "circuit_breaker_reset",
                    "breaker": self.name,
                    "from_state": old_state.value
                }}
            )

    def force_open(self):
        """Manually force circuit breaker to OPEN state."""
        with self._lock:
            old_state = self._state
            self._transition_to_open()
            logger.warning(
                f"Circuit breaker '{self.name}' manually forced open: {old_state.value} -> OPEN",
                extra={"extra_fields": {
                    "event": "circuit_breaker_force_open",
                    "breaker": self.name,
                    "from_state": old_state.value
                }}
            )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is OPEN and rejects a call."""
    pass


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