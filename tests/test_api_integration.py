"""
Automated Verification Test for Free API Integration
Tests:
1. Groq service fallback and execution
2. NLP categorization & clustering integrity
3. Pipeline run with custom payload
4. REST API handler response generation
"""
from __future__ import annotations

import os
import sys

# Ensure workspace root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.groq_service import summarize_with_groq
from backend.nlp_engine import classify_category
from backend.summarizer import create_resilient_summarizer
from main import run_pipeline


def test_groq_fallback():
    print("🧪 Test 1: Testing Groq Service Fallback (0 Token / No Key)...")
    title = "Test Haber Başlığı: Yeni Teknoloji Geliştirildi"
    content = "Mühendisler yeni bir yapay zeka algoritması geliştirdi. Bu algoritma yüksek hızda çalışmaktadır. Testler başarıyla sonuçlandı."

    res = summarize_with_groq(title, content, api_key=None)
    assert res is not None, "Result should not be None"
    assert "summary" in res, "Summary key missing"
    assert "engine" in res, "Engine key missing"
    assert "Yerel NLP" in res["engine"], f"Expected Local NLP engine, got: {res['engine']}"
    print(f"   ✅ Başarılı! Çıktı: '{res['summary']}' (Motor: {res['engine']})")


def test_resilient_summarizer():
    print("\n🧪 Test 1b: Testing ResilientSummarizer Directly...")
    summarizer = create_resilient_summarizer(api_key=None)
    title = "Test Haber: Yapay Zeka Gelişmeleri"
    content = "Yapay zeka alanında yeni gelişmeler oldu. Araştırmacılar yeni modeller geliştirdi."

    result = summarizer.summarize(title, content)
    assert result.success is True
    assert "Yerel NLP" in result.engine
    assert len(result.summary) > 0
    print(f"   ✅ Başarılı! Çıktı: '{result.summary}' (Motor: {result.engine})")


def test_nlp_categorization():
    print("\n🧪 Test 2: Testing Category Classification...")
    cat1 = classify_category("KIZILELMA uçağı teste çıktı", "Savunma sanayiinde yeni adımlar atılıyor.")
    assert cat1 == "Savunma & Teknoloji", f"Expected Savunma & Teknoloji, got: {cat1}"

    cat2 = classify_category("Merkez Bankası faiz kararını açıkladı", "Piyasalarda dolar ve borsa hareketlendi.")
    assert cat2 == "Ekonomi", f"Expected Ekonomi, got: {cat2}"

    cat3 = classify_category("Cumhurbaşkanı Meclis'te konuştu", "AK Parti ve CHP hakkında açıklama yaptı.")
    assert cat3 == "Politika & Gündem", f"Expected Politika & Gündem, got: {cat3}"
    print("   ✅ Başarılı! Kategoriler doğru sınıflandırıldı.")


def test_pipeline_execution():
    print("\n🧪 Test 3: Testing Full Pipeline Execution...")
    results = run_pipeline(api_key=None, max_per_source=2)
    assert len(results) > 0, "Pipeline should return articles"
    first = results[0]
    assert "title" in first
    assert "summary" in first
    assert "category" in first
    assert "ai_engine" in first
    print(f"   ✅ Başarılı! Pipeline {len(results)} haberi başarıyla işledi ve kaydetti.")


def test_pipeline_with_injected_summarizer():
    print("\n🧪 Test 4: Testing Pipeline with Injected Summarizer (no @patch needed)...")
    from backend.summarizer import ResilientSummarizer, SummarizeResult

    class MockSummarizer:
        def summarize(self, title, content):
            return SummarizeResult(
                summary="Mock özeti",
                engine="Test Engine",
                success=True,
                model="Test Model"
            )

    mock_summarizer = MockSummarizer()
    # Use progressive=False to force full summarization with the mock (no delta mode)
    results = run_pipeline(api_key=None, max_per_source=1, summarizer=mock_summarizer, progressive=False)
    assert len(results) > 0
    assert results[0]["ai_engine"] == "Test Engine"
    assert results[0]["summary"] == "Mock özeti"
    print("   ✅ Başarılı! Pipeline injected summarizer ile çalıştı.")


if __name__ == "__main__":
    print("==========================================")
    print("🚀 RUNNING AUTOMATED VERIFICATION TESTS")
    print("==========================================")
    test_groq_fallback()
    test_resilient_summarizer()
    test_nlp_categorization()
    test_pipeline_execution()
    test_pipeline_with_injected_summarizer()
    print("\n🎉 BÜTÜN DOĞRULAMA TESTLERİ BAŞARIYLA GEÇTİ!")