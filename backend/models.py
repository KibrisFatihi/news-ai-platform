"""
Data Models for News AI Platform
Pydantic models for type safety, validation, and serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict, field_serializer


class Category(str, Enum):
    """News categories."""
    TECHNOLOGY = "Savunma & Teknoloji"
    SCIENCE = "Bilim"
    ECONOMY = "Ekonomi"
    POLITICS = "Politika & Gündem"
    WORLD = "Dünya"
    SPORTS = "Spor"
    GENERAL = "Genel Gündem"


class AIEengine(str, Enum):
    """AI engine types."""
    GROQ_LLAMA = "Groq LLaMA 3.3"
    LOCAL_NLP = "Yerel NLP (Sıfır Token)"
    LOCAL_NLP_FALLBACK = "Yerel NLP (Yedek Mod)"


class ArticleBase(BaseModel):
    """Base article model with common fields."""
    id: str
    source: str
    title: str = Field(..., min_length=1, max_length=500)
    content: str
    link: str
    published: str
    category: Optional[Category] = None

    @field_validator("link", mode="before")
    @classmethod
    def validate_link(cls, v: Any) -> str:
        if isinstance(v, str) and not v.startswith(("http://", "https://")):
            return f"https://{v}"
        return str(v) if v is not None else v


class Article(ArticleBase):
    """Full article model with all fields."""
    summary: str
    original_content: str
    ai_engine: str
    ai_model: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    related_count: int = 0
    related_sources: list[str] = Field(default_factory=list)

    @field_serializer("updated_at")
    def serialize_updated_at(self, value: datetime) -> str:
        return value.isoformat()


class Cluster(BaseModel):
    """Article cluster model."""
    main_article: ArticleBase
    related_articles: list[dict] = Field(default_factory=list)


class NewsResponse(BaseModel):
    """API response for /api/news endpoint."""
    success: bool = True
    count: int
    categories: dict[str, int]
    groq_configured: bool
    active_engine: str
    articles: list[Article]


class RefreshResponse(BaseModel):
    """API response for /api/refresh endpoint."""
    success: bool
    message: str
    count: int
    articles: list[Article] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """API response for /api/status endpoint."""
    status: str = "online"
    version: str
    groq_connected: bool
    model: str
    mode: str
    uptime_seconds: float
    refresh_token: Optional[str] = None


class HealthResponse(BaseModel):
    """API response for /api/health endpoint."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class SummarizeResult(BaseModel):
    """Result from a summarization operation."""
    summary: str
    engine: str
    success: bool
    model: str


# Type aliases for backwards compatibility
ArticleDict = dict[str, Any]
ClusterDict = dict[str, Any]
NewsFeed = list[ArticleDict]