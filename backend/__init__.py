"""
News AI Platform - Backend Package
"""
from .circuit_breaker import CircuitBreaker, CircuitOpenError, get_circuit_breaker
from .crawler import clean_html, fetch_news
from .groq_service import get_groq_api_key, get_groq_status, summarize_with_groq
from .handlers import (
    handle_get_health,
    handle_get_news,
    handle_get_status,
    handle_not_found,
    handle_options,
    handle_post_refresh,
)
from .logger import get_logger, setup_logging
from .middleware import (
    check_conditional_request,
    check_refresh_rate_limit,
    compute_etag,
    create_error_response,
    format_http_date,
    generate_request_id,
    get_cors_headers,
    handle_options_request,
    log_api_request,
    parse_http_date,
    verify_refresh_token,
)
from .models import (
    Article,
    ArticleBase,
    ArticleDict,
    Category,
    Cluster,
    HealthResponse,
    NewsFeed,
    NewsResponse,
    RefreshResponse,
    StatusResponse,
    SummarizeResult,
)
from .nlp_engine import (
    classify_category,
    clear_vector_cache,
    cluster_similar_news,
    compute_cosine_similarity,
    extractive_summarize,
    tokenize,
)
from .rate_limiter import RateLimiter, get_groq_rate_limiter
from .repository import NewsRepository, get_news_repository, load_news_feed, save_news_feed
from .summarizer import ResilientSummarizer, create_resilient_summarizer

__all__ = [
    # Crawler
    "fetch_news",
    "clean_html",
    # NLP Engine
    "tokenize",
    "classify_category",
    "compute_cosine_similarity",
    "cluster_similar_news",
    "extractive_summarize",
    "clear_vector_cache",
    # Groq Service
    "get_groq_api_key",
    "summarize_with_groq",
    "get_groq_status",
    # Logger
    "get_logger",
    "setup_logging",
    # Rate Limiter
    "get_groq_rate_limiter",
    "RateLimiter",
    # Circuit Breaker
    "get_circuit_breaker",
    "CircuitBreaker",
    "CircuitOpenError",
    # Repository
    "get_news_repository",
    "NewsRepository",
    "load_news_feed",
    "save_news_feed",
    # Models
    "Article",
    "ArticleBase",
    "ArticleDict",
    "Category",
    "Cluster",
    "HealthResponse",
    "NewsFeed",
    "NewsResponse",
    "RefreshResponse",
    "StatusResponse",
    "SummarizeResult",
    # Summarizer
    "ResilientSummarizer",
    "create_resilient_summarizer",
    # Handlers
    "handle_get_news",
    "handle_get_status",
    "handle_get_health",
    "handle_post_refresh",
    "handle_options",
    "handle_not_found",
    # Middleware
    "generate_request_id",
    "get_cors_headers",
    "handle_options_request",
    "compute_etag",
    "format_http_date",
    "parse_http_date",
    "check_conditional_request",
    "check_refresh_rate_limit",
    "verify_refresh_token",
    "create_error_response",
    "log_api_request",
]