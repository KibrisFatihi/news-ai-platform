"""
Main Pipeline Runner for AI News Aggregator
Orchestrates:
1. Data Ingestion (Crawler)
2. Topic Classification (NLP)
3. Similarity Deduplication & Clustering (NLP)
4. AI Summarization (ResilientSummarizer - Groq or Local NLP)
5. JSON Export via NewsRepository
"""
from __future__ import annotations

import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional


from backend.crawler import fetch_news
from backend.logger import get_logger, log_pipeline_complete, log_pipeline_start
from backend.nlp_engine import classify_category, cluster_similar_news
from backend.repository import get_news_repository
from backend.summarizer import create_resilient_summarizer
from config import get_config

logger = get_logger("pipeline")

BATCH_SIZE = 2  # Tek Groq isteğinde özetlenecek makale sayısı (79 haber -> ~16 istek)


def _chunk_list(items: List, chunk_size: int) -> List[List]:
    """Split a list into fixed-size chunks."""
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _build_article_record(cluster: Dict, ai_result, updated_at: str) -> Dict:
    """Build standardized output dictionary for a summarized cluster."""
    main_art = cluster["main_article"]
    return {
        "title": main_art["title"],
        "source": main_art["source"],
        "category": main_art["category"],
        "summary": ai_result.summary,
        "original_content": main_art["content"],
        "link": main_art["link"],
        "image_url": main_art.get("image_url"),
        "ai_engine": ai_result.engine,
        "ai_model": ai_result.model,
        "published": main_art.get("published", "Yeni"),
        "updated_at": updated_at,
        "related_count": len(cluster["related_articles"]),
        "related_sources": [r["article"]["source"] for r in cluster["related_articles"]],
    }


def run_pipeline(
    api_key: Optional[str] = None,
    max_per_source: Optional[int] = None,
    summarizer: Optional[Any] = None,  # Injected for testing
) -> List[Dict]:
    """
    Run the complete news aggregation and summarization pipeline.

    Args:
        api_key: Optional Groq API key override
        max_per_source: Max articles per RSS source (uses config default if None)
        summarizer: Optional injected summarizer (for testing without @patch)

    Returns:
        List of processed article dictionaries
    """
    cfg = get_config()
    rss_cfg = cfg.rss
    nlp_cfg = cfg.nlp

    max_items = max_per_source or rss_cfg.max_per_source

    pipeline_start = time.time()

    log_pipeline_start(logger, api_key_present=bool(api_key), max_per_source=max_items)

    logger.info("=" * 60)
    logger.info("🚀 AI-POWERED NEWS AGGREGATION & SUMMARIZATION PIPELINE")
    logger.info("=" * 60)

    # 1. Fetching news
    logger.info("[1/5] 📰 Haber kaynakları taranıyor (RSS Feeds)...")
    articles = fetch_news(max_per_source=max_items)
    logger.info(f"      -> Toplam {len(articles)} haber başarıyla işlendi.")

    if not articles:
        logger.warning("[!] Hiç haber bulunamadı.")
        return []

    # 2. Categorization
    logger.info("[2/5] 🏷️ Doğal Dil İşleme (NLP) ile Haberler Kategorize Ediliyor...")
    categorization_start = time.time()
    for article in articles:
        article["category"] = classify_category(article["title"], article["content"])
    categorization_duration = (time.time() - categorization_start) * 1000
    logger.info(f"      -> Kategorizasyon tamamlandı ({categorization_duration:.0f}ms)")

    # 3. Clustering / Deduplication
    logger.info("[3/5] 🔗 Benzer Haberler Gruplanıyor (Deduplication / Cosine Similarity)...")
    clusters = cluster_similar_news(articles, threshold=nlp_cfg.similarity_threshold)
    logger.info(f"      -> {len(articles)} haber {len(clusters)} benzersiz haber konusuna gruplandı.")

    # 4. Summarization (Batch + Parallel across models)
    logger.info("[4/5] ⚡ Özetleme Motoru Çalışıyor...")
    summarizer_instance = summarizer or create_resilient_summarizer(api_key=api_key)

    if summarizer_instance:
        active_key = get_config().get_groq_api_key(api_key)
        if active_key:
            logger.info("      -> Groq Cloud (Batch + Model Rotation) aktif")
        else:
            logger.info("      -> Yerel NLP (Sıfır Token / Çevrimdışı Mod) aktif")
    else:
        logger.info("      -> Yerel NLP (Sıfır Token / Çevrimdışı Mod) aktif")

    summarization_start = time.time()
    results: List[Dict] = []
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    use_batch = hasattr(summarizer_instance, "summarize_batch")

    if use_batch:
        batches = _chunk_list(clusters, BATCH_SIZE)
        max_workers = min(len(batches), 2)

        def _summarize_batch_clusters(batch_clusters: List[Dict]) -> List[Dict]:
            items = [
                (c["main_article"]["title"], c["main_article"]["content"])
                for c in batch_clusters
            ]
            ai_results = summarizer_instance.summarize_batch(items)
            return [
                _build_article_record(c, ai_res, now_iso)
                for c, ai_res in zip(batch_clusters, ai_results)
            ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_summarize_batch_clusters, b) for b in batches]
            for future in as_completed(futures):
                results.extend(future.result())
    else:
        max_workers = min(len(clusters), 3)

        def _summarize_single_cluster(cluster: Dict) -> Dict:
            main_art = cluster["main_article"]
            ai_result = summarizer_instance.summarize(main_art["title"], main_art["content"])
            return _build_article_record(cluster, ai_result, now_iso)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_summarize_single_cluster, c) for c in clusters]
            for future in as_completed(futures):
                results.append(future.result())

    summarization_duration = (time.time() - summarization_start) * 1000
    logger.info(
        f"      -> Özetleme tamamlandı ({summarization_duration:.0f}ms, {len(results)} makale, workers={max_workers}, batch={BATCH_SIZE if use_batch else 1})"
    )

    # 5. Save to JSON via NewsRepository (atomic write)
    logger.info("[5/5] 💾 Veriler NewsRepository ile kaydediliyor (Atomik Yazma)...")
    save_start = time.time()
    repo = get_news_repository()
    repo.save(results)
    save_duration = (time.time() - save_start) * 1000
    logger.info(f"      -> Kayıt tamamlandı ({save_duration:.0f}ms)")

    total_duration = (time.time() - pipeline_start) * 1000

    log_pipeline_complete(logger, len(results), total_duration)

    logger.info("=" * 60)
    logger.info(f"✨ İŞLEM TAMAMLANDI! {len(results)} haber hazırlandı.")
    logger.info(f"💾 Veriler kaydedildi: {repo.feed_path}")
    logger.info(f"⏱️  Toplam süre: {total_duration:.0f}ms")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_pipeline()