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
        "Habertürk": "https://www.haberturk.com/rss/kategori/gundem.xml",
        "Euronews Türkçe": "https://tr.euronews.com/rss?format=mrss&level=theme&name=news",
        "DW Türkçe": "https://rss.dw.com/xml/rss-tur-all",
        "Webrazzi": "https://webrazzi.com/feed/",
        "ShiftDelete.Net": "https://shiftdelete.net/feed",
        "DonanımHaber": "https://www.donanimhaber.com/rss/tum/",
        "Sözcü": "https://www.sozcu.com.tr/feeds-rss-category-gundem",
        "Cumhuriyet": "https://www.cumhuriyet.com.tr/rss/son_dakika.xml",
        "Bloomberg HT": "https://www.bloomberght.com/rss",
        "Ekonomim": "https://www.ekonomim.com/rss",
    })
    max_per_source: int = int(os.getenv("MAX_PER_SOURCE", "5"))
    request_timeout: int = 10
    user_agent: str = "NewsAIPlatform/2.0 (+https://github.com/news-ai-platform)"


# Tek kaynak: kategori anahtar kelime listesi.
# Hem _CATEGORY_PATTERNS (derlenmiş regex) hem de
# NLPConfig.category_keywords (ham liste) bu tanımdan üretilir.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Savunma & Teknoloji": [
        "kızılelma", "envanter", "siber", "roket", "siha", "iha", "uydu",
        "drone", "füze", "radar", "savunma", "askeri", "mühimmat",
        "yapay zeka", "teknoloji", "algoritma", "model", "yazılım",
        "veri", "bulut", "siber güvenlik", "şifreleme", "blockchain",
        "işlemci", "ekran kartı", "apple", "google", "microsoft", "nvidia",
        "openai", "telefon", "güncelleme", "oyun", "elektrikli araç",
        "startup", "girişim", "uygulama", "donanım", "hacker", "siber saldırı", "veri sızıntısı",
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
        "üst düzey", "genel müdür", "yönetim kurulu", "ceo",
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
}

# Pre-compile category keyword patterns at module load (not lazy)
# This avoids first-request latency and ensures patterns are ready.
_CATEGORY_PATTERNS: dict[str, list[Pattern]] = {}
for _cat, _kws in _CATEGORY_KEYWORDS.items():
    _patterns = []
    for _kw in _kws:
        if _kw == "deney":
            # Avoid matching "deneyim" / "deneyimli" (experience) as science experiment.
            _pat = r'\bdeney(?!im)\w*\b'
        elif len(_kw) >= 5 and ' ' not in _kw:
            _pat = r'\b' + re.escape(_kw) + r'(?:\w*)?\b'
        else:
            _pat = r'\b' + re.escape(_kw) + r'\b'
        _patterns.append(re.compile(_pat, re.IGNORECASE))
    _CATEGORY_PATTERNS[_cat] = _patterns


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
    # Tek kaynaktan üretilir (_CATEGORY_KEYWORDS).
    category_keywords: dict[str, list[str]] = field(default_factory=lambda: dict(_CATEGORY_KEYWORDS))

    # Pre-compiled regex patterns for each category (built at module load)
    default_category: str = "Genel Gündem"

    def get_compiled_patterns(self) -> dict[str, list[Pattern]]:
        """Get pre-compiled regex patterns for category keywords."""
        return _CATEGORY_PATTERNS


@dataclass(frozen=True)
class GroqConfig:
    """Groq API configuration."""
    api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    default_model: str = "qwen/qwen3.8-27b"
    fast_model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.2
    max_tokens: int = 512
    request_timeout: int = 8
    # Rate limits for free tier
    rate_limit_rpm: int = 20
    rate_limit_rpd: int = 14400
    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_timeout_seconds: int = 60
    # Model rotation - Groq rate limits are per-model, so rotating across
    # models multiplies effective free-tier capacity (approx. N models x limits).
    # When enabled and no explicit model is chosen, articles are round-robined
    # across these models and a 429 on one model fails over to the next.
    model_rotation_enabled: bool = True
    model_rotation: list[str] = field(default_factory=lambda: [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",

    ])
    # Sequential model processing: process ALL articles with model 1, then model 2, etc.
    # Avoids hammering all models simultaneously (prevents 429 storms).
    # Trade-off: longer total time but much higher success rate per model.
    sequential_model_processing: bool = False

    # Hybrid Sequential Spillover Mode:
    # - max_workers parallel workers, EACH uses sequential spillover across models
    # - Worker starts with model[0], on 429 spills remaining to model[1], then model[2]
    # - Small delay between requests (0.3-0.5s) to respect RPM
    # - Reduced max_tokens and content truncation to stay within TPM
    sequential_spillover_enabled: bool = True
    spillover_max_workers: int = 2
    spillover_request_delay: float = 0.4  # seconds between requests
    spillover_max_tokens: int = 256
    spillover_max_content_chars: int = 1500


@dataclass(frozen=True)
class ServerConfig:
    """HTTP Server configuration."""
    port: int = int(os.getenv("PORT", "8000"))
    host: str = "0.0.0.0"
    # CORS — Render environment üzerinden override edilebilir;
    # varsayılan olarak tüm originlere izin verilir (*).
    cors_allowed_origins: list[str] = field(
        default_factory=lambda: os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
        if os.getenv("CORS_ALLOWED_ORIGINS")
        else ["*"]
    )
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
    # Rate limiting for /api/refresh — 10 dakikada 1 istek
    refresh_rate_limit_window: int = 600
    refresh_rate_limit_max: int = 1
    # Background scheduler
    scheduler_interval_seconds: int = 30 * 60  # 30 minutes
    # Pipeline timeout
    pipeline_timeout_seconds: int = 120


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


# --- Model ID doğrulama ---

ALLOWED_MODEL_IDS: set[str] = set()

def _build_allowed_model_ids() -> set[str]:
    """GroqConfig.model_rotation listesinden izin verilen model ID setini derler."""
    ids = set()
    cfg = get_config()
    for m in cfg.groq.model_rotation:
        if m:
            ids.add(m.strip())
    ids.add(cfg.groq.default_model)
    ids.add(cfg.groq.fast_model)
    return ids


ALLOWED_MODEL_IDS = _build_allowed_model_ids()


def validate_model_id(model_id: str) -> bool:
    """Verilen model ID'nin izinli listedede olup olmadığını döner.

    Sunucu tarafında model id whitelist koruması için kullanılır.
    Kullanıcının rastgele bir model id göndermesini engellemek için
    refresh/summarization giriş noktalarında çağrılmalı.
    """
    if not model_id or not isinstance(model_id, str):
        return False
    return model_id.strip() in ALLOWED_MODEL_IDS


# For backwards compatibility and easy imports
RSS_FEEDS = config.rss.feeds
CATEGORY_KEYWORDS = config.nlp.category_keywords
TURKISH_STOPWORDS = config.nlp.turkish_stopwords
GROQ_API_URL = config.groq.api_url
DEFAULT_MODEL = config.groq.default_model
FAST_MODEL = config.groq.fast_model
PORT = config.server.port