"""
Centralized Configuration for News AI Platform
All hardcoded values moved here for easy maintenance and environment-based overrides.
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Pattern


@dataclass(frozen=True)
class RSSFeedConfig:
    """RSS Feed sources configuration (20 active Turkish news & tech feeds)."""
    feeds: dict[str, str] = field(default_factory=lambda: {
        "BBC Türkçe": "https://feeds.bbci.co.uk/turkce/rss.xml",
        "TRT Haber": "https://www.trthaber.com/manset_articles.rss",
        "TRT Haber - Gündem": "https://www.trthaber.com/gundem_articles.rss",
        "TRT Haber - Dünya": "https://www.trthaber.com/dunya_articles.rss",
        "TRT Haber - Ekonomi": "https://www.trthaber.com/ekonomi_articles.rss",
        "TRT Haber - Bilim Teknoloji": "https://www.trthaber.com/bilim_teknoloji_articles.rss",
        "NTV - Ekonomi": "https://www.ntv.com.tr/ekonomi.rss",
        "NTV - Teknoloji": "https://www.ntv.com.tr/teknoloji.rss",
        "NTV Spor": "https://www.ntvspor.net/rss/anasayfa",
        "Anadolu Ajansı": "https://www.aa.com.tr/tr/rss/default?cat=guncel",
        "Habertürk": "https://www.haberturk.com/rss/kategori/gundem.xml",
        "Euronews Türkçe": "https://tr.euronews.com/rss?format=mrss&level=theme&name=news",
        "DW Türkçe": "https://rss.dw.com/xml/rss-tur-all",
        "Webrazzi": "https://webrazzi.com/feed/",
        "ShiftDelete.Net": "https://shiftdelete.net/feed",
        "DonanımHaber": "https://www.donanimhaber.com/rss/tum/",
        "Evrim Ağacı": "https://evrimagaci.org/rss.xml",
        "Bloomberg HT": "https://www.bloomberght.com/rss",
        "Ekonomim": "https://www.ekonomim.com/rss",
    })
    max_per_source: int = 6
    request_timeout: int = 10
    user_agent: str = "NewsAIPlatform/2.0 (+https://github.com/news-ai-platform)"


@dataclass(frozen=True)
class NLPConfig:
    """NLP Engine configuration."""
    similarity_threshold: float = 0.25
    summary_sentences: int = 2
    turkish_stopwords: frozenset = field(default_factory=lambda: frozenset({
        "acaba", "ama", "aslında", "az", "bazı", "belki", "biri", "birkaç", "birşey", "biz",
        "bu", "çok", "çünkü", "da", "daha", "de", "defa", "diye", "en", "gibi", "hem",
        "hep", "hepsi", "her", "hiç", "için", "ile", "ise", "kez", "ki", "kim", "mı",
        "mu", "mü", "nasıl", "ne", "neden", "nerde", "nerede", "nereye", "niçin", "niye",
        "o", "sanki", "şey", "siz", "şu", "tüm", "ve", "veya", "ya", "yani", "bir"
    }))

    # Category keywords - Title gets 2x weight in classification
    category_keywords: dict[str, list[str]] = field(default_factory=lambda: {
        "Savunma & Teknoloji": [
            "kızılelma", "envanter", "siber", "roket", "siha", "iha", "uydu",
            "drone", "füze", "radar", "savunma", "askeri", "mühimmat",
            "yapay zeka", "teknoloji", "algoritma", "model", "yazılım",
            "veri", "bulut", "siber güvenlik", "şifreleme", "blockchain",
            "işlemci", "ekran kartı", "apple", "google", "microsoft", "nvidia",
            "openai", "telefon", "güncelleme", "oyun", "elektrikli araç",
            "startup", "girişim", "uygulama", "donanım",
        ],
        "Bilim": [
            "bilim", "uzay", "nasa", "evrim", "deney", "araştırma",
            "kuantum", "biyoloji", "fizik", "gezegen", "teleskop",
            "iklim", "keşif", "fosil", "genetik", "molekül", "astronomi",
        ],
        "Ekonomi": [
            "dolar", "enflasyon", "faiz", "para", "fon", "banka", "piyasa", "borsa",
            "maaş", "zam", "ekonomi", "bütçe", "tüfe", "ithalat", "ihracat", "cari açık",
            "tl", "kur", "merkez bankası", "faiz kararı", "enflasyon verisi",
            "gsyh", "işsizlik", "asgari ücret", "emekli", "sgk", "vergi",
            "altın", "petrol", "kredi", "yatırım", "hisse",
        ],
        "Politika & Gündem": [
            "parti", "seçim", "belediye", "hükümet", "meclis", "başkan", "ak parti",
            "chp", "tbmm", "eylem", "miting", "oy", "sandık", "yargı", "anayasa",
            "milletvekili", "kanun", "teklif", "yasa", "genel kurul",
            "başbakan", "bakan", "kabine", "cumhurbaşkanı", "mahkeme", "soruşturma",
        ],
        "Dünya": [
            "uluslararası", "divan", "myanmar", "israil", "gazze", "abd", "rusya",
            "ukrayna", "ab", "avrupa", "soykırım", "nato", "bm", "barış", "savaş",
            "çin", "hindistan", "japonya", "kore", "orta doğu", "afrika",
            "brics", "g20", "bm güvenlik konseyi", "antlaşma", "diplomasi",
        ],
        "Spor": [
            "futbol", "maç", "gol", "lig", "galatasaray", "fenerbahçe", "beşiktaş",
            "şampiyon", "transfer", "kupa", "kupası", "milli takım", "süper lig",
            "trabzonspor", "basketbol", "voleybol", "tenis", "formula 1",
            "teknik direktör", "derbi", "şampiyonlar ligi", "nba", "euroleague",
        ],
    })

    # Pre-compiled regex patterns for each category (built on first access)
    _compiled_patterns: dict[str, list[Pattern]] = field(default_factory=dict, init=False, repr=False)
    default_category: str = "Genel Gündem"

    def get_compiled_patterns(self) -> dict[str, list[Pattern]]:
        """Get or build pre-compiled regex patterns for category keywords."""
        if not self._compiled_patterns:
            compiled = {}
            for category, keywords in self.category_keywords.items():
                patterns = []
                for kw in keywords:
                    if len(kw) >= 5 and ' ' not in kw:
                        pattern = r'\b' + re.escape(kw) + r'(?:\w*)?\b'
                    else:
                        pattern = r'\b' + re.escape(kw) + r'\b'
                    patterns.append(re.compile(pattern, re.IGNORECASE))
                compiled[category] = patterns
            object.__setattr__(self, "_compiled_patterns", compiled)
        return self._compiled_patterns


@dataclass(frozen=True)
class GroqConfig:
    """Groq API configuration."""
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    default_model: str = "llama-3.3-70b-versatile"
    fast_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 180
    request_timeout: int = 8
    # Rate limits for free tier
    rate_limit_rpm: int = 30
    rate_limit_rpd: int = 14400
    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: int = 60


@dataclass(frozen=True)
class ServerConfig:
    """HTTP Server configuration."""
    port: int = int(os.getenv("PORT", "8000"))
    host: str = "0.0.0.0"
    # CORS
    cors_allowed_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_allowed_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    cors_allowed_headers: list[str] = field(default_factory=lambda: [
        "Content-Type", "Authorization", "X-Groq-Api-Key", "X-Refresh-Token"
    ])
    # Cache
    cache_ttl_seconds: int = 300
    etag_enabled: bool = True
    # Request limits
    max_request_size_mb: int = 10
    # Static files
    static_dir: str = "frontend"
    index_file: str = "index.html"
    # Rate limiting for /api/refresh
    refresh_rate_limit_window: int = 60
    refresh_rate_limit_max: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    """Logging configuration."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Optional[str] = os.getenv("LOG_FILE")
    json_format: bool = os.getenv("LOG_JSON", "false").lower() == "true"


@dataclass(frozen=True)
class CacheConfig:
    """Cache configuration."""
    news_feed_path: str = "data/news_feed.json"
    max_age_seconds: int = 3600
    cleanup_on_startup: bool = False


@dataclass(frozen=True)
class AppConfig:
    """Main application configuration aggregator."""
    rss: RSSFeedConfig = field(default_factory=RSSFeedConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent)

    @property
    def news_feed_full_path(self) -> Path:
        return self.base_dir / self.cache.news_feed_path

    @property
    def frontend_dir(self) -> Path:
        return self.base_dir / self.server.static_dir

    def get_groq_api_key(self, override: Optional[str] = None) -> Optional[str]:
        """Get Groq API key with priority: override > env > None."""
        if override and override.strip():
            return override.strip()
        return os.environ.get("GROQ_API_KEY", "").strip() or None


# Global config instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


# For backwards compatibility and easy imports
RSS_FEEDS = config.rss.feeds
CATEGORY_KEYWORDS = config.nlp.category_keywords
TURKISH_STOPWORDS = config.nlp.turkish_stopwords
GROQ_API_URL = config.groq.api_url
DEFAULT_MODEL = config.groq.default_model
FAST_MODEL = config.groq.fast_model
PORT = config.server.port