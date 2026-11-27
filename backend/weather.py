"""
Turkey Weather Service
Fetches current weather for major Turkish cities via Open-Meteo (free, no API key required)
with a 15-minute thread-safe in-memory cache.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.logger import get_logger

logger = get_logger("weather")

_CACHE_TTL_SECONDS = 900  # 15 minutes
_REQUEST_TIMEOUT_SECONDS = 5

TURKEY_CITIES: List[Dict[str, Any]] = [
    {"name": "İstanbul", "lat": 41.0082, "lon": 28.9784, "featured": True},
    {"name": "Ankara", "lat": 39.9334, "lon": 32.8597, "featured": True},
    {"name": "İzmir", "lat": 38.4237, "lon": 27.1428, "featured": True},
    {"name": "Antalya", "lat": 36.8969, "lon": 30.7133, "featured": False},
    {"name": "Bursa", "lat": 40.1885, "lon": 29.0610, "featured": False},
    {"name": "Adana", "lat": 37.0000, "lon": 35.3213, "featured": False},
    {"name": "Trabzon", "lat": 41.0027, "lon": 39.7168, "featured": False},
    {"name": "Gaziantep", "lat": 37.0662, "lon": 37.3833, "featured": False},
    {"name": "Konya", "lat": 37.8746, "lon": 32.4932, "featured": False},
    {"name": "Eskişehir", "lat": 39.7767, "lon": 30.5206, "featured": False},
    {"name": "Samsun", "lat": 41.2867, "lon": 36.3300, "featured": False},
    {"name": "Kayseri", "lat": 38.7312, "lon": 35.4787, "featured": False},
    {"name": "Diyarbakır", "lat": 37.9144, "lon": 40.2306, "featured": False},
    {"name": "Erzurum", "lat": 39.9000, "lon": 41.2700, "featured": False},
    {"name": "Muğla", "lat": 37.2153, "lon": 28.3636, "featured": False},
]

_WMO_MAP: Dict[int, tuple[str, str]] = {
    0: ("Açık", "☀️"),
    1: ("Az Bulutlu", "🌤️"),
    2: ("Parçalı Bulutlu", "⛅"),
    3: ("Çok Bulutlu", "☁️"),
    45: ("Sisli", "🌫️"),
    48: ("Kırağılı Sis", "🌫️"),
    51: ("Hafif Çisenti", "🌦️"),
    53: ("Çisenti", "🌦️"),
    55: ("Yoğun Çisenti", "🌧️"),
    61: ("Hafif Yağmur", "🌦️"),
    63: ("Yağmurlu", "🌧️"),
    65: ("Kuvvetli Yağmur", "🌧️"),
    71: ("Hafif Kar", "🌨️"),
    73: ("Kar Yağışlı", "❄️"),
    75: ("Yoğun Kar", "❄️"),
    80: ("Sağanak Yağış", "🌦️"),
    81: ("Sağanak", "🌧️"),
    82: ("Şiddetli Sağanak", "⛈️"),
    95: ("Gök Gürültülü Fırtına", "⛈️"),
    96: ("Dolu & Fırtına", "⛈️"),
    99: ("Şiddetli Fırtına", "⛈️"),
}

_cache_lock = threading.Lock()
_cached_payload: Optional[Dict[str, Any]] = None
_cached_at: float = 0.0


def _describe_weather_code(code: Optional[int]) -> tuple[str, str]:
    """Map WMO weather code to Turkish label and emoji icon."""
    if code is None:
        return ("Bilinmiyor", "🌡️")
    if code in _WMO_MAP:
        return _WMO_MAP[code]
    if 51 <= code <= 67:
        return ("Yağışlı", "🌧️")
    if 71 <= code <= 77:
        return ("Kar Yağışlı", "❄️")
    if 80 <= code <= 82:
        return ("Sağanak", "🌧️")
    if code >= 95:
        return ("Fırtına", "⛈️")
    return ("Bulutlu", "☁️")


def _fetch_open_meteo() -> List[Dict[str, Any]]:
    """Fetch current weather for all configured Turkish cities in one batch request."""
    lats = ",".join(str(c["lat"]) for c in TURKEY_CITIES)
    lons = ",".join(str(c["lon"]) for c in TURKEY_CITIES)
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}&longitude={lons}"
        f"&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
        f"&timezone=Europe/Istanbul"
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "GundemOzeti/1.0 (+https://open-meteo.com)"},
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    items = raw if isinstance(raw, list) else [raw]
    cities_out: List[Dict[str, Any]] = []
    for city, entry in zip(TURKEY_CITIES, items):
        current = entry.get("current") or {}
        code = current.get("weather_code")
        condition, icon = _describe_weather_code(code)
        temp = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        wind = current.get("wind_speed_10m")

        cities_out.append({
            "name": city["name"],
            "featured": city["featured"],
            "temp": round(temp) if isinstance(temp, (int, float)) else None,
            "feels_like": round(feels_like) if isinstance(feels_like, (int, float)) else None,
            "wind_kmh": round(wind) if isinstance(wind, (int, float)) else None,
            "weather_code": code,
            "condition": condition,
            "icon": icon,
        })
    return cities_out


def get_turkey_weather(force_refresh: bool = False) -> Dict[str, Any]:
    """Return cached or freshly fetched weather data for major Turkish cities."""
    global _cached_payload, _cached_at

    now = time.time()
    with _cache_lock:
        if not force_refresh and _cached_payload and (now - _cached_at) < _CACHE_TTL_SECONDS:
            return _cached_payload

    try:
        cities = _fetch_open_meteo()
        payload: Dict[str, Any] = {
            "success": True,
            "source": "Open-Meteo",
            "source_url": "https://www.mgm.gov.tr/tahmin/il-ve-ilceler.aspx",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cities": cities,
        }
        with _cache_lock:
            _cached_payload = payload
            _cached_at = time.time()
        return payload
    except Exception as exc:
        logger.warning(f"Weather fetch failed: {exc}")
        with _cache_lock:
            if _cached_payload:
                return _cached_payload
        return {
            "success": False,
            "source": "Open-Meteo",
            "source_url": "https://www.mgm.gov.tr/tahmin/il-ve-ilceler.aspx",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "cities": [],
            "error": "Hava durumu verisi şu an alınamadı.",
        }
