# News AI Platform - Code Audit Report

**Date:** 2026-11-26  
**Project:** `C:\Users\umutg\projects\news-ai-platform`  
**Scope:** Full codebase audit for overengineering, unused code, missing parameters, and optimization opportunities

---

## Executive Summary

The News AI Platform is a well-structured, production-ready Turkish news aggregator with AI summarization. The codebase demonstrates solid architectural patterns (circuit breaker, rate limiting, model rotation, atomic persistence) but contains several areas of overengineering, redundant abstractions, and minor optimization opportunities.

**Overall Assessment:** **B+** - Good architecture with some over-engineered layers

---

## 1. Overengineering Analysis

### 1.1 Excessive Abstraction Layers

| File | Issue | Severity |
|------|-------|----------|
| `backend/summarizer.py` | 5 fallback strategy classes + Protocol + ModelRotator + 3 summarizer classes for essentially one task | **High** |
| `backend/circuit_breaker.py` | Full generic `CircuitBreaker[T]` with stats, state machine, registry - only used for Groq API | **Medium** |
| `backend/rate_limiter.py` | Dual sync/async implementations, Condition variables, per-model registry - only 2 windows (RPM/RPD) | **Medium** |
| `backend/models.py` | 11 Pydantic models for 5 API endpoints - many duplicate fields | **Low** |
| `backend/middleware.py` | 11 helper functions for CORS, caching, rate limiting, tokens - could be consolidated | **Low** |

### 1.2 Specific Overengineering Examples

#### `backend/summarizer.py` - The Main Culprit (550+ lines)
```python
# 5 FallbackStrategy dataclasses that are nearly identical
FALLBACK_CHAIN: tuple[FallbackStrategy, ...] = (
    FallbackStrategy(name="rate_limit_timeout", engine_label="Yerel NLP (Yedek Mod - Rate Limit)", model_label="Extractive TextRank"),
    FallbackStrategy(name="circuit_open", engine_label="Yerel NLP (Yedek Mod - Circuit Breaker)", model_label="Extractive TextRank"),
    FallbackStrategy(name="timeout", engine_label="Yerel NLP (Yedek Mod - Timeout)", model_label="Extractive TextRank"),
    FallbackStrategy(name="http_error", engine_label="Yerel NLP (Yedek Mod - HTTP Error)", model_label="Extractive TextRank"),
    FallbackStrategy(name="unexpected_error", engine_label="Yerel NLP (Yedek Mod - Error)", model_label="Extractive TextRank"),
)
```
**Problem:** All 5 strategies do the exact same thing (call local NLP) with only the log label differing. This could be a simple dict or function.

**Recommendation:** Replace with a single fallback function that takes an error type enum.

#### `backend/circuit_breaker.py` - Generic for Single Use
```python
class CircuitBreaker(Generic[T]):  # Only instantiated for "groq_api" and "groq_api:{model}"
```
**Problem:** Full generic implementation with stats tracking, excluded/expected exceptions, manual force_open/reset - but only protects Groq API calls.

**Recommendation:** Simplify to a non-generic class with hardcoded Groq config, or use a library like `pybreaker`.

#### `backend/rate_limiter.py` - Sync + Async + Per-Model Registry
```python
# Lines 299-346: AsyncRateLimiter (never used in codebase)
class AsyncRateLimiter: ...  # Only used in tests

# Lines 267-295: Global per-model registry
_groq_rate_limiters: dict[str, RateLimiter] = {}
```
**Problem:** Async version exists but `server.py` uses synchronous `ThreadPoolExecutor`. Per-model registry adds complexity for 3 models.

**Recommendation:** Remove `AsyncRateLimiter`. Keep simple sync limiter; per-model is fine but document why.

---

## 2. Unused / Dead Code

### 2.1 Completely Unused

| File | Symbol | Reason |
|------|--------|--------|
| `backend/rate_limiter.py` | `AsyncRateLimiter` class (lines 299-346) | Never imported/used in production code |
| `backend/circuit_breaker.py` | `CircuitBreaker.excluded_exceptions` / `expected_exceptions` config | Never customized; defaults used everywhere |
| `backend/models.py` | `AIEengine` enum | Defined but never used; handlers use raw strings |
| `backend/models.py` | `Category` enum | Used in tests only; runtime uses strings from config |
| `backend/summarizer.py` | `summarize_with_groq()` backwards compat function | Only used in tests and `groq_service.py` wrapper |
| `backend/groq_service.py` | `SYSTEM_PROMPT` constant | Duplicated in `GroqSummarizer.__init__` |
| `backend/groq_service.py` | `GROQ_API_URL`, `DEFAULT_MODEL`, `FAST_MODEL` | Re-exports from config; never used directly |
| `backend/middleware.py` | `create_success_response()` | Never called |
| `backend/repository.py` | `save_news_feed()` / `load_news_feed()` compat functions | Only used in tests |

### 2.2 Conditionally Unused

| File | Symbol | Condition |
|------|--------|-----------|
| `backend/summarizer.py` | `GroqSummarizer._make_request()` error handling for 401 | API key validation happens at request level, not here |
| `config.py` | `NLPConfig._compiled_patterns` lazy init | Patterns built on first `classify_category()` call - fine |
| `server.py` | `ThreadingHTTPServer` | Python 3.7+ only; Windows may have issues |

---

## 3. Missing Parameters / Configuration Gaps

### 3.1 Hardcoded Values That Should Be Configurable

| File | Line | Value | Should Be In Config |
|------|------|-------|---------------------|
| `backend/crawler.py` | 23-27 | `BROWSER_USER_AGENT` string | `RSSFeedConfig.user_agent` (already exists but not used!) |
| `backend/crawler.py` | 138 | `max_workers = min(12, ...)` | `RSSFeedConfig.max_workers` |
| `backend/nlp_engine.py` | 50 | `TITLE_WEIGHT = 2.0` | `NLPConfig.title_weight` |
| `backend/summarizer.py` | 130 | `num_sentences = 2` default | `NLPConfig.summary_sentences` (exists but not passed to LocalNLPSummarizer!) |
| `backend/summarizer.py` | 165-173 | `system_prompt` hardcoded | `GroqConfig.system_prompt` |
| `server.py` | 64 | `_scheduler_interval_seconds = 30 * 60` | `ServerConfig.scheduler_interval_seconds` |
| `server.py` | 324 | `pipeline_timeout_seconds = 120` | `ServerConfig.pipeline_timeout_seconds` |
| `middleware.py` | 138-139 | `REFRESH_RATE_LIMIT_WINDOW/MAX` from server_cfg | Already in config ✓ |

### 3.2 Config Values Defined But Not Used

| Config Field | Defined In | Used? |
|--------------|------------|-------|
| `RSSFeedConfig.user_agent` | `config.py:37` | **No** - crawler uses hardcoded `BROWSER_USER_AGENT` |
| `NLPConfig.summary_sentences` | `config.py:44` | **Partially** - used in `extractive_summarize()` but not passed to `LocalNLPSummarizer` |
| `GroqConfig.model_rotation` | `config.py:137` | **Yes** - used in `ResilientSummarizer` |
| `ServerConfig.refresh_rate_limit_*` | `config.py:164-165` | **Yes** - used in middleware |

---

## 4. Optimization Opportunities

### 4.1 Performance

| Area | Current | Optimized | Impact |
|------|---------|-----------|--------|
| `nlp_engine.py:145-161` | `_tokenize_and_vectorize()` caches by `hash(text)` | Cache by `(text, config_id)` tuple directly | Avoids hash collision risk, faster lookup |
| `nlp_engine.py:254-274` | Clustering re-tokenizes `main_article` for each comparison | Pre-compute all vectors once (already done for `article_texts`) | **Already optimized** - vectors cached |
| `crawler.py:138` | `ThreadPoolExecutor(max_workers=min(12, len(feeds)))` | Limit to CPU count or configurable | Prevents thread explosion on 34 feeds |
| `summarizer.py:143-173` | `GroqSummarizer` creates new `requests.post` each call | Use `requests.Session()` for connection pooling | **Medium** - reduces TCP handshakes |
| `repository.py:48-56` | `tempfile.NamedTemporaryFile` + `os.replace` | Good - already atomic | ✓ |

### 4.2 Code Simplification

#### Mergeable Modules
```
backend/
├── summarizer.py (550 lines)  <-- CAN BE REDUCED TO ~200
├── circuit_breaker.py (307 lines)  <-- CAN BE REDUCED TO ~80
├── rate_limiter.py (346 lines)  <-- CAN BE REDUCED TO ~120
├── groq_service.py (97 lines)  <-- MERGE INTO summarizer.py
├── middleware.py (244 lines)  <-- CAN BE REDUCED TO ~100
└── handlers.py (386 lines)  <-- OK, but could split
```

#### Specific Refactorings

**1. `summarizer.py` → Reduce 550 → 200 lines**
```python
# BEFORE: 5 FallbackStrategy classes, Protocol, ModelRotator, 3 summarizer classes
# AFTER: Single ResilientSummarizer with simple fallback dict

FALLBACKS = {
    "rate_limit": "Yerel NLP (Yedek Mod - Rate Limit)",
    "circuit_open": "Yerel NLP (Yedek Mod - Circuit Breaker)",
    "timeout": "Yerel NLP (Yedek Mod - Timeout)",
    "http_error": "Yerel NLP (Yedek Mod - HTTP Error)",
    "error": "Yerel NLP (Yedek Mod - Error)",
}

class ResilientSummarizer:
    def _fallback(self, reason: str, title: str, content: str) -> SummarizeResult:
        summary = self.local.summarize(title, content).summary
        return SummarizeResult(summary=summary, engine=FALLBACKS[reason], success=False, model="Extractive TextRank")
```

**2. `circuit_breaker.py` → Reduce 307 → 80 lines**
```python
# Use simple closure-based approach or pybreaker library
# Only need: state, failure_count, last_failure_time, call(func)
```

**3. `rate_limiter.py` → Reduce 346 → 120 lines**
```python
# Remove AsyncRateLimiter (unused)
# Simplify RateLimitWindow to use list + lock (Condition overkill for 30 RPM)
# Keep per-model registry but simplify
```

### 4.3 Duplicate Code

| Location | Duplication |
|----------|-------------|
| `summarizer.py:165-173` vs `groq_service.py:36-41` | `SYSTEM_PROMPT` defined twice |
| `config.py:224-230` vs multiple files | Backwards compat exports (`RSS_FEEDS`, `CATEGORY_KEYWORDS`, etc.) |
| `handlers.py:50-71` vs `models.py:51-63` | `_normalize_article()` duplicates Article model defaults |
| `nlp_engine.py:381-383` vs `config.py:225-226` | `TURKISH_STOPWORDS`, `CATEGORY_KEYWORDS` re-exported |

---

## 5. File-by-File Detailed Findings

### `config.py` (221 lines) ✓ Well Done
- **Good:** Centralized config, frozen dataclasses, env overrides, computed properties
- **Issues:** 
  - `RSSFeedConfig.user_agent` defined but not used (crawler uses hardcoded)
  - `NLPConfig.summary_sentences` not passed to `LocalNLPSummarizer`
  - Backwards compat exports at bottom create coupling

### `main.py` (148 lines) ✓ Good Pipeline Orchestrator
- **Good:** Clear 5-step pipeline, injected summarizer for testing, structured logging
- **Issues:**
  - `Optional[Any]` type hint for summarizer (should be `SummarizerProtocol`)
  - No timeout on individual pipeline steps (only overall in server.py)

### `backend/crawler.py` (203 lines) ✓ Solid
- **Good:** Concurrent fetching, browser UA, fallback to cache, HTML cleaning
- **Issues:**
  - Hardcoded `BROWSER_USER_AGENT` (config has `user_agent`)
  - Hardcoded `max_workers=12` 
  - `clean_html()` double `html.unescape()` (lines 34, 36)
  - Fallback cache normalization logic duplicated in comments (lines 158-170)

### `backend/nlp_engine.py` (383 lines) ✓ Good NLP Core
- **Good:** Tokenization, classification, cosine similarity with caching, clustering, extractive summarization
- **Issues:**
  - `TITLE_WEIGHT = 2.0` hardcoded (should be config)
  - `_vector_cache` global mutable state (test isolation needs `clear_vector_cache` fixture)
  - `tokenize()` creates new regex each call (pre-compile)
  - `cluster_similar_news()` O(n²) - fine for <100 articles but won't scale

### `backend/summarizer.py` (550 lines) ⚠️ **MAIN REFACTOR TARGET**
- **Good:** Resilient design, model rotation, circuit breaker integration, rate limiter integration, protocol for testing
- **Issues:** (See Section 1.1)
  - 5 identical `FallbackStrategy` dataclasses
  - `SummarizerProtocol` only used for type hint
  - `ModelRotator` overkill for 3 models
  - `LocalNLPSummarizer` doesn't use `config.nlp.summary_sentences`
  - `GroqSummarizer` creates new connection each call (no Session)
  - `ResilientSummarizer._groq_permanently_unavailable` flag but no reset mechanism

### `backend/circuit_breaker.py` (307 lines) ⚠️ Overengineered
- **Good:** Proper state machine, thread-safe, stats, registry
- **Issues:**
  - Generic `CircuitBreaker[T]` for single use case
  - `excluded_exceptions` / `expected_exceptions` never customized
  - `CircuitBreakerStats` dataclass overkill
  - Manual `force_open()` / `reset()` never used in production

### `backend/rate_limiter.py` (346 lines) ⚠️ Overengineered
- **Good:** Sliding window, Condition variables (efficient wait), per-model registry, Retry-After header
- **Issues:**
  - `AsyncRateLimiter` completely unused
  - `RateLimitWindow` uses `deque` + `Condition` - overkill for 30 RPM
  - `wait_for_slot()` complex for simple use case
  - Global registry `_groq_rate_limiters` adds indirection

### `backend/repository.py` (200 lines) ✓ Clean
- **Good:** Atomic writes, fallback loading, staleness detection, transaction context manager
- **Issues:**
  - Backwards compat functions only used in tests
  - `NewsFeed` type alias = `list[ArticleDict]` - loose typing

### `backend/models.py` (122 lines) ⚠️ Excessive Models
- **Good:** Pydantic validation, serialization, enums
- **Issues:**
  - 11 model classes for 5 endpoints
  - `AIEengine` enum unused (strings used everywhere)
  - `Category` enum unused at runtime (config strings used)
  - `Article` and `ArticleBase` share 90% fields
  - `NewsResponse` / `RefreshResponse` / `StatusResponse` / `HealthResponse` could use composition

### `backend/logger.py` (183 lines) ✓ Good
- **Good:** JSON + colored console, rotation, config hash for reinit, extra_fields pattern
- **Issues:**
  - Windows stdout reconfigure hack (needed but ugly)
  - `log_*` convenience functions couple logger to specific events

### `backend/middleware.py` (244 lines) ⚠️ Can Simplify
- **Good:** CORS, ETag, rate limiting, HMAC tokens, error helpers
- **Issues:**
  - `create_success_response()` unused
  - `parse_http_date()` uses `email.utils` + `calendar` - could use `datetime.fromisoformat` for ISO dates
  - `_refresh_rate_limit` global dict - test isolation needs fixture

### `backend/groq_service.py` (97 lines) ⚠️ Thin Wrapper
- **Good:** Status aggregation, backwards compat
- **Issues:**
  - `SYSTEM_PROMPT` duplicated from summarizer
  - Constants re-exported from config
  - Could be merged into `summarizer.py`

### `backend/handlers.py` (386 lines) ✓ Good Separation
- **Good:** Pure functions, testable, clear error handling, token verification
- **Issues:**
  - `_normalize_article()` duplicates `Article` model defaults
  - `handle_post_refresh()` imports `run_pipeline` inside function (circular import workaround)
  - Large function (150 lines) - could split

### `server.py` (312 lines) ✓ Good Stdlib Server
- **Good:** ThreadingHTTPServer, background scheduler, static file serving, graceful shutdown
- **Issues:**
  - `socketserver.TCPServer.allow_reuse_address = False` (should be True for dev)
  - Hardcoded `30*60` scheduler interval, `120` pipeline timeout
  - `sys.stdout.reconfigure()` duplicated from logger.py
  - `NewsAPIRequestHandler` inherits `SimpleHTTPRequestHandler` but overrides `end_headers`

### `tests/` (10 files, ~2000 lines) ✓ Comprehensive
- **Good:** Fixtures for isolation, mocks for external services, integration tests
- **Issues:**
  - `test_api_integration.py` is a verification script, not unit tests
  - Many tests patch internal implementation details
  - `conftest.py` has 7 autouse fixtures - complex setup

---

## 6. Refactoring Priority Matrix

| Priority | Task | Effort | Risk | Impact |
|----------|------|--------|------|--------|
| **P1** | Remove `AsyncRateLimiter` (unused) | 15 min | Zero | Cleanup |
| **P1** | Remove `AIEengine` enum (unused) | 10 min | Zero | Cleanup |
| **P1** | Remove `SYSTEM_PROMPT` duplication | 10 min | Zero | Dedupe |
| **P1** | Use `config.rss.user_agent` in crawler | 10 min | Low | Config consistency |
| **P1** | Pass `summary_sentences` to `LocalNLPSummarizer` | 10 min | Low | Config consistency |
| **P2** | Simplify `FALLBACK_CHAIN` to dict/function | 30 min | Low | -300 lines |
| **P2** | Use `requests.Session` in `GroqSummarizer` | 20 min | Low | Performance |
| **P2** | Move scheduler interval / pipeline timeout to config | 20 min | Low | Configurability |
| **P2** | Remove backwards compat exports from config.py | 30 min | Medium | Decoupling |
| **P3** | Simplify `CircuitBreaker` to non-generic ~80 lines | 1 hour | Medium | Maintainability |
| **P3** | Simplify `RateLimiter` (remove async, simplify window) | 1 hour | Medium | Maintainability |
| **P3** | Merge `groq_service.py` into `summarizer.py` | 30 min | Low | Cohesion |
| **P4** | Consolidate Pydantic models (composition over inheritance) | 2 hours | Medium | Type safety |
| **P4** | Split `handlers.py` into per-endpoint files | 1 hour | Low | Organization |
| **P4** | Pre-compile regex in `tokenize()` | 30 min | Low | Performance |

---

## 7. Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. Remove dead code: `AsyncRateLimiter`, `AIEengine`, `SYSTEM_PROMPT` dup, unused compat functions
2. Fix config usage: `user_agent`, `summary_sentences`, scheduler interval, pipeline timeout
3. Use `requests.Session` in `GroqSummarizer`
4. Simplify `FALLBACK_CHAIN` to dict

### Phase 2: Core Simplification (3-4 hours)
1. Refactor `CircuitBreaker` to simple class (remove generic, stats, unused methods)
2. Refactor `RateLimiter` (remove async, simplify window, keep per-model)
3. Merge `groq_service.py` → `summarizer.py`
4. Reduce `summarizer.py` from 550 → ~200 lines

### Phase 3: Polish (2-3 hours)
1. Consolidate Pydantic models
2. Split `handlers.py`
3. Pre-compile regex in tokenizer
4. Update tests for simplified internals
5. Run full test suite

---

## 8. Risk Assessment

| Change | Risk Level | Mitigation |
|--------|------------|------------|
| Remove dead code | **Zero** | Not referenced anywhere |
| Simplify fallback chain | **Low** | Same behavior, just less code |
| Config additions | **Low** | Defaults preserve current behavior |
| CircuitBreaker rewrite | **Medium** | Keep same public API (`call()`, `state`, `reset()`) |
| RateLimiter rewrite | **Medium** | Keep same public API (`acquire()`, `get_status()`) |
| Model consolidation | **Medium** | Run full test suite, check serialization |

---

## 9. Metrics Summary

| Metric | Current | Target (Post-Refactor) |
|--------|---------|------------------------|
| Total Python LOC | ~3,500 | ~2,500 (-28%) |
| `summarizer.py` lines | 550 | 200 |
| `circuit_breaker.py` lines | 307 | 80 |
| `rate_limiter.py` lines | 346 | 120 |
| Dead code symbols | 12+ | 0 |
| Hardcoded config values | 8 | 0 |
| Duplicate constants | 4 | 0 |
| Pydantic model classes | 11 | 6-7 |

---

## 10. Conclusion

The News AI Platform is **production-ready and well-tested** but carries significant **accidental complexity** in its resilience layer (summarizer, circuit breaker, rate limiter). The core pipeline (crawler → NLP → summarizer → repository) is clean and effective.

**Primary recommendation:** Focus refactoring on `backend/summarizer.py` and its dependencies (`circuit_breaker.py`, `rate_limiter.py`, `groq_service.py`). These four files comprise ~1,250 lines that can be reduced to ~400 lines with no loss of functionality.

The quick wins (Phase 1) can be done safely in under 2 hours. Phase 2 requires more care but pays off in long-term maintainability.

---

*Generated by code-audit analysis on 2026-11-26*