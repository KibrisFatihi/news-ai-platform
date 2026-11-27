"""
Tests for the Circuit Breaker module (simplified implementation).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from backend.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    get_all_circuit_breakers,
    get_circuit_breaker,
    reset_all_circuit_breakers,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.timeout_seconds == 60.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30.0,
        )
        assert config.failure_threshold == 3
        assert config.timeout_seconds == 30.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test initial state is CLOSED."""
        breaker = CircuitBreaker("test")
        assert breaker.state == "closed"
        assert breaker._failure_count == 0

    def test_successful_call_increments_stats(self):
        """Test successful call increments success stats."""
        breaker = CircuitBreaker("test")
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.stats["success"] == 1
        assert breaker.stats["total"] == 1

    def test_failed_call_increments_failure_count(self):
        """Test failed call increments failure count."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker._failure_count == 1
        assert breaker.stats["failure"] == 1

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == "open"
        assert breaker.stats["failure"] == 3

    def test_rejects_calls_when_open(self):
        """Test calls are rejected when OPEN."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        # Fail once to open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == "open"

        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "success")

        assert breaker.stats["rejected"] == 1

    def test_auto_recovers_after_timeout(self):
        """Test circuit auto-recovers to CLOSED after timeout."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        )

        # Fail to open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == "open"

        # Wait for timeout
        time.sleep(0.15)

        # State should transition to CLOSED on next check
        assert breaker.state == "closed"
        assert breaker._failure_count == 0

    def test_get_status(self):
        """Test status reporting."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))
        breaker.call(lambda: "success")
        breaker.call(lambda: "success")

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        status = breaker.get_status()
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 1
        assert status["stats"]["total"] == 3
        assert status["stats"]["success"] == 2
        assert status["stats"]["failure"] == 1

    def test_reset(self):
        """Test manual reset."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == "open"

        breaker.reset()

        assert breaker.state == "closed"
        assert breaker._failure_count == 0

    def test_force_open(self):
        """Test manual force open."""
        breaker = CircuitBreaker("test")
        assert breaker.state == "closed"

        breaker.force_open()

        assert breaker.state == "open"


class TestCircuitBreakerRegistry:
    """Tests for global circuit breaker registry."""

    def test_get_circuit_breaker_singleton(self):
        """Test get_circuit_breaker returns singleton."""
        reset_all_circuit_breakers()
        cb1 = get_circuit_breaker("test")
        cb2 = get_circuit_breaker("test")
        assert cb1 is cb2

    def test_get_all_circuit_breakers(self):
        """Test getting all breakers."""
        reset_all_circuit_breakers()
        get_circuit_breaker("breaker1")
        get_circuit_breaker("breaker2")

        all_breakers = get_all_circuit_breakers()
        assert "breaker1" in all_breakers
        assert "breaker2" in all_breakers

    def test_reset_all_circuit_breakers(self):
        """Test reset all."""
        reset_all_circuit_breakers()
        # Use failure_threshold=1 to open after single failure
        cb1 = get_circuit_breaker("test1", config=CircuitBreakerConfig(failure_threshold=1))
        cb2 = get_circuit_breaker("test2", config=CircuitBreakerConfig(failure_threshold=1))

        # Fail both
        for cb in [cb1, cb2]:
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert cb1.state == "open"
        assert cb2.state == "open"

        reset_all_circuit_breakers()

        assert cb1.state == "closed"
        assert cb2.state == "closed"


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_error_message(self):
        """Test error message format."""
        error = CircuitOpenError("test message")
        assert str(error) == "test message"

    def test_error_with_breaker_name(self):
        """Test error includes breaker name."""
        error = CircuitOpenError("Circuit breaker 'api' is OPEN. Retry after 60s.")
        assert "api" in str(error)
        assert "OPEN" in str(error)
        assert "60s" in str(error)