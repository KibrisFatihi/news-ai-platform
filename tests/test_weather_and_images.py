"""Tests for RSS image extraction and Turkey weather endpoint."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from backend.crawler import extract_image_url
from backend.handlers.weather import handle_get_weather
from backend.weather import TURKEY_CITIES, get_turkey_weather


def test_extract_image_url_from_media_content():
    entry = SimpleNamespace(
        media_content=[{"url": "https://cdn.example.com/photo.jpg", "medium": "image"}],
        summary="Haber özeti",
    )
    assert extract_image_url(entry) == "https://cdn.example.com/photo.jpg"


def test_extract_image_url_from_summary_html_skips_tracking_pixel():
    entry = SimpleNamespace(
        media_content=[],
        media_thumbnail=[],
        enclosures=[],
        summary='<img src="https://example.com/1x1.gif"><img src="https://cdn.example.com/news.webp">',
    )
    assert extract_image_url(entry) == "https://cdn.example.com/news.webp"


def test_extract_image_url_returns_none_when_missing():
    entry = SimpleNamespace(summary="Görselsiz metin", description="")
    assert extract_image_url(entry) is None


def test_get_turkey_weather_includes_featured_and_all_cities():
    fake_cities = [
        {
            "name": c["name"],
            "featured": c["featured"],
            "temp": 15,
            "feels_like": 14,
            "wind_kmh": 12,
            "weather_code": 1,
            "condition": "Az Bulutlu",
            "icon": "🌤️",
        }
        for c in TURKEY_CITIES
    ]
    with patch("backend.weather._fetch_open_meteo", return_value=fake_cities):
        result = get_turkey_weather(force_refresh=True)

    assert result["success"] is True
    assert len(result["cities"]) == len(TURKEY_CITIES)
    featured_names = {c["name"] for c in result["cities"] if c["featured"]}
    assert featured_names == {"İstanbul", "Ankara", "İzmir"}


def test_handle_get_weather_returns_200():
    fake_payload = {
        "success": True,
        "source": "Open-Meteo",
        "source_url": "https://www.mgm.gov.tr/tahmin/il-ve-ilceler.aspx",
        "updated_at": "2026-11-27T00:00:00+00:00",
        "cities": [{"name": "İstanbul", "featured": True, "temp": 14, "condition": "Açık", "icon": "☀️"}],
    }
    with patch("backend.handlers.weather.get_turkey_weather", return_value=fake_payload):
        status, headers, body = handle_get_weather({}, "req-weather-1")

    assert status == 200
    assert body["success"] is True
    assert body["cities"][0]["name"] == "İstanbul"
