"""
Tests for the Models module.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.models import (
    Article,
    ArticleBase,
    Category,
    Cluster,
    HealthResponse,
    NewsResponse,
    RefreshResponse,
    StatusResponse,
    SummarizeResult,
)


class TestCategory:
    """Tests for Category enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert Category.TECHNOLOGY == "Savunma & Teknoloji"
        assert Category.ECONOMY == "Ekonomi"
        assert Category.POLITICS == "Politika & Gündem"
        assert Category.WORLD == "Dünya"
        assert Category.SPORTS == "Spor"
        assert Category.GENERAL == "Genel Gündem"


class TestArticleBase:
    """Tests for ArticleBase model."""

    def test_article_base_valid(self):
        """Test valid article base."""
        article = ArticleBase(
            id="1",
            source="Test",
            title="Test Title",
            content="Test content",
            link="https://example.com",
            published="2026-01-01"
        )
        assert article.id == "1"
        assert article.title == "Test Title"

    def test_article_base_validates_link(self):
        """Test link validation adds https."""
        article = ArticleBase(
            id="1",
            source="Test",
            title="Test",
            content="Content",
            link="example.com",  # No protocol
            published="2026-01-01"
        )
        assert article.link == "https://example.com"

    def test_article_base_preserves_https(self):
        """Test https links are preserved."""
        article = ArticleBase(
            id="1",
            source="Test",
            title="Test",
            content="Content",
            link="https://example.com",
            published="2026-01-01"
        )
        assert article.link == "https://example.com"

    def test_article_base_invalid_title(self):
        """Test title validation."""
        with pytest.raises(ValidationError):
            ArticleBase(
                id="1",
                source="Test",
                title="",  # Empty title
                content="Content",
                link="https://example.com",
                published="2026-01-01"
            )

    def test_article_base_optional_category(self):
        """Test optional category."""
        article = ArticleBase(
            id="1",
            source="Test",
            title="Test",
            content="Content",
            link="https://example.com",
            published="2026-01-01"
        )
        assert article.category is None

        article = ArticleBase(
            id="1",
            source="Test",
            title="Test",
            content="Content",
            link="https://example.com",
            published="2026-01-01",
            category=Category.TECHNOLOGY
        )
        assert article.category == Category.TECHNOLOGY


class TestArticle:
    """Tests for Article model."""

    def test_article_valid(self):
        """Test valid full article."""
        article = Article(
            id="1",
            source="Test",
            title="Test Title",
            content="Test content",
            link="https://example.com",
            published="2026-01-01",
            summary="Summary",
            original_content="Original content",
            ai_engine="Yerel NLP",
            ai_model="TextRank"
        )
        assert article.summary == "Summary"
        assert article.ai_engine == "Yerel NLP"

    def test_article_defaults(self):
        """Test default values."""
        article = Article(
            id="1",
            source="Test",
            title="Test",
            content="Content",
            link="https://example.com",
            published="2026-01-01",
            summary="Summary",
            original_content="Content",
            ai_engine="Test",
            ai_model="Test"
        )
        assert article.updated_at is not None
        assert article.related_count == 0
        assert article.related_sources == []

    def test_article_json_serialization(self):
        """Test JSON serialization includes datetime."""
        article = Article(
            id="1",
            source="Test",
            title="Test",
            content="Content",
            link="https://example.com",
            published="2026-01-01",
            summary="Summary",
            original_content="Content",
            ai_engine="Test",
            ai_model="Test"
        )
        data = article.model_dump()
        assert "updated_at" in data
        assert isinstance(data["updated_at"], str)


class TestSummarizeResult:
    """Tests for SummarizeResult model."""

    def test_summarize_result_valid(self):
        """Test valid summarize result."""
        result = SummarizeResult(
            summary="Test summary",
            engine="Groq LLaMA 3.3",
            success=True,
            model="llama-3.3-70b-versatile"
        )
        assert result.success is True
        assert result.engine == "Groq LLaMA 3.3"

    def test_summarize_result_fallback(self):
        """Test fallback result."""
        result = SummarizeResult(
            summary="Fallback summary",
            engine="Yerel NLP (Yedek Mod)",
            success=False,
            model="Extractive TextRank"
        )
        assert result.success is False
        assert "Yedek Mod" in result.engine


class TestCluster:
    """Tests for Cluster model."""

    def test_cluster_valid(self):
        """Test valid cluster."""
        main = ArticleBase(
            id="1",
            source="Test",
            title="Main",
            content="Content",
            link="https://example.com",
            published="2026-01-01"
        )
        cluster = Cluster(main_article=main, related_articles=[])
        assert cluster.main_article.id == "1"
        assert cluster.related_articles == []


class TestNewsResponse:
    """Tests for NewsResponse model."""

    def test_news_response_valid(self):
        """Test valid news response."""
        response = NewsResponse(
            success=True,
            count=1,
            categories={"Test": 1},
            groq_configured=False,
            active_engine="Yerel NLP",
            articles=[]
        )
        assert response.success is True
        assert response.count == 1


class TestRefreshResponse:
    """Tests for RefreshResponse model."""

    def test_refresh_response_valid(self):
        """Test valid refresh response."""
        response = RefreshResponse(
            success=True,
            message="Updated",
            count=1,
            articles=[]
        )
        assert response.success is True
        assert response.message == "Updated"


class TestStatusResponse:
    """Tests for StatusResponse model."""

    def test_status_response_valid(self):
        """Test valid status response."""
        response = StatusResponse(
            status="online",
            version="2.0.0",
            groq_connected=True,
            model="llama-3.3-70b-versatile",
            mode="hybrid",
            uptime_seconds=100.0
        )
        assert response.status == "online"
        assert response.groq_connected is True


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response_valid(self):
        """Test valid health response."""
        response = HealthResponse(
            status="healthy",
            timestamp="2026-01-01T12:00:00Z",
            version="2.0.0",
            uptime_seconds=100.0,
            components={
                "news_cache": {"loaded": True},
                "groq_api": {"api_key_configured": True},
                "circuit_breakers": {}
            }
        )
        assert response.status == "healthy"
        assert "news_cache" in response.components


