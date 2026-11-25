"""
Tests for the Circuit Breaker module.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from backend.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
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
        assert config.success_threshold == 2
        assert config.timeout_seconds == 60.0
        assert config.excluded_exceptions == ()
        assert config.expected_exceptions == (Exception,)

    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=30.0,
            excluded_exceptions=(ValueError,),
            expected_exceptions=(IOError,)
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 1
        assert config.timeout_seconds == 30.0
        assert config.excluded_exceptions == (ValueError,)
        assert config.expected_exceptions == (IOError,)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test initial state is CLOSED."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_successful_call_increments_stats(self):
        """Test successful call increments success stats."""
        breaker = CircuitBreaker("test")
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.stats.successful_calls == 1
        assert breaker.stats.total_calls == 1

    def test_failed_call_increments_failure_count(self):
        """Test failed call increments failure count."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker._failure_count == 1
        assert breaker.stats.failed_calls == 1

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=3))

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failed_calls == 3

    def test_rejects_calls_when_open(self):
        """Test calls are rejected when OPEN."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        # Fail once to open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == CircuitState.OPEN

        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "success")

        assert breaker.stats.rejected_calls == 1

    def test_transitions_to_half_open_after_timeout(self):
        """Test transition to HALF_OPEN after timeout."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(failure_threshold=1, timeout_seconds=0.1)
        )

        # Fail to open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # State should transition to HALF_OPEN on next check
        assert breaker.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold_in_half_open(self):
        """Test circuit closes after success threshold in HALF_OPEN."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=2,
                timeout_seconds=0.1
            )
        )

        # Fail to open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Succeed twice to close
        breaker.call(lambda: "success1")
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.call(lambda: "success2")
        assert breaker.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        """Test failure in HALF_OPEN reopens circuit."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=2,
                timeout_seconds=0.1
            )
        )

        # Fail to open
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Fail again - should reopen
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == CircuitState.OPEN

    def test_excluded_exceptions_dont_count(self):
        """Test excluded exceptions don't count as failures."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                excluded_exceptions=(ValueError,)
            )
        )

        # ValueError is excluded
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("excluded")))

        # Should still be CLOSED, failure count not incremented
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker.stats.successful_calls == 1  # Excluded counts as success

    def test_expected_exceptions_count(self):
        """Test expected exceptions count as failures."""
        breaker = CircuitBreaker(
            "test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                expected_exceptions=(IOError,)
            )
        )

        with pytest.raises(IOError):
            breaker.call(lambda: (_ for _ in ()).throw(IOError("expected")))

        assert breaker.state == CircuitState.OPEN
        assert breaker._failure_count == 1

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
        assert status["stats"]["total_calls"] == 3
        assert status["stats"]["successful_calls"] == 2
        assert status["stats"]["failed_calls"] == 1

    def test_reset(self):
        """Test manual reset."""
        breaker = CircuitBreaker("test", config=CircuitBreakerConfig(failure_threshold=1))

        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("error")))

        assert breaker.state == CircuitState.OPEN

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_force_open(self):
        """Test manual force open."""
        breaker = CircuitBreaker("test")
        assert breaker.state == CircuitState.CLOSED

        breaker.force_open()

        assert breaker.state == CircuitState.OPEN


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

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        reset_all_circuit_breakers()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED


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