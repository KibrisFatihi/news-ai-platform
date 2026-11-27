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
from typing import Any, Dict, List, Optional

from backend.crawler import fetch_news
from backend.logger import get_logger, log_pipeline_complete, log_pipeline_start
from backend.nlp_engine import classify_category, cluster_similar_news
from backend.repository import get_news_repository
from backend.summarizer import create_resilient_summarizer
from config import get_config

logger = get_logger("pipeline")


def _build_article_record(cluster: Dict, ai_result, updated_at: str, summary_status: str = "completed") -> Dict:
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
        "summary_status": summary_status,
    }


def _build_pending_record(cluster: Dict) -> Dict:
    """Build record with pending summary for immediate display."""
    main_art = cluster["main_article"]
    return {
        "title": main_art["title"],
        "source": main_art["source"],
        "category": main_art["category"],
        "summary": "",  # Empty - will be filled async
        "original_content": main_art["content"],
        "link": main_art["link"],
        "image_url": main_art.get("image_url"),
        "ai_engine": "",
        "ai_model": "",
        "published": main_art.get("published", "Yeni"),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "related_count": len(cluster["related_articles"]),
        "related_sources": [r["article"]["source"] for r in cluster["related_articles"]],
        "summary_status": "pending",
    }


def run_pipeline(
    api_key: Optional[str] = None,
    max_per_source: Optional[int] = None,
    summarizer: Optional[Any] = None,  # Injected for testing
    progressive: bool = True,  # Enable progressive loading
) -> List[Dict]:
    """
    Run the complete news aggregation and summarization pipeline with progressive loading.

    Phase 1 (fast): Fetch -> Categorize -> Cluster -> Save immediately (summary_status="pending")
    Phase 2 (async): Summarize only NEW/CHANGED articles (delta) -> Update summaries in place

    Args:
        api_key: Optional Groq API key override
        max_per_source: Max articles per RSS source (uses config default if None)
        summarizer: Optional injected summarizer (for testing without @patch)
        progressive: If True, use progressive loading with delta summarization

    Returns:
        List of processed article dictionaries (with summaries if progressive=False, or pending if True)
    """
    cfg = get_config()
    rss_cfg = cfg.rss
    nlp_cfg = cfg.nlp

    max_items = max_per_source or rss_cfg.max_per_source
    pipeline_start = time.time()

    log_pipeline_start(logger, api_key_present=bool(api_key), max_per_source=max_items)

    logger.info("=" * 60)
    logger.info("🚀 AI-POWERED NEWS AGGREGATION & SUMMARIZATION PIPELINE")
    if progressive:
        logger.info("📦 Progressive Mode: Phase 1 (instant) + Phase 2 (delta summarization)")
    logger.info("=" * 60)

    # ============ PHASE 1: FAST PATH ============
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

    # 4. IMMEDIATE SAVE WITH PENDING SUMMARIES (Progressive Loading)
    logger.info("[4/5] ⚡ Hızlı Kayıt: Özetler 'pending' olarak kaydediliyor...")
    phase1_start = time.time()
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Build pending records for immediate display
    pending_results = [_build_pending_record(c) for c in clusters]

    # Progressive save: merges with existing, preserves completed summaries
    repo = get_news_repository()
    if progressive:
        repo.save_progressive(pending_results)
    else:
        repo.save(pending_results)

    phase1_duration = (time.time() - phase1_start) * 1000
    logger.info(f"      -> Anlık kayıt tamamlandı ({phase1_duration:.0f}ms, {len(pending_results)} makale)")

    # Return immediately for UI - articles visible with pending summaries
    if not progressive:
        # Legacy mode: do full summarization now
        return _run_summarization_phase(
            clusters=clusters,
            api_key=api_key,
            summarizer=summarizer,
            repo=repo,
            pipeline_start=pipeline_start,
        )

    # ============ PHASE 2: DELTA SUMMARIZATION (BACKGROUND) ============
    logger.info("[5/5] 🤖 Arka Plan Özetleme: Sadece YENİ/DEĞİŞEN haberler özetleniyor...")
    summarization_results = _run_summarization_phase(
        clusters=clusters,
        api_key=api_key,
        summarizer=summarizer,
        repo=repo,
        pipeline_start=pipeline_start,
        delta_mode=True,
    )

    total_duration = (time.time() - pipeline_start) * 1000

    log_pipeline_complete(logger, len(summarization_results), total_duration)

    logger.info("=" * 60)
    logger.info(f"✨ İŞLEM TAMAMLANDI! {len(summarization_results)} haber hazırlandı.")
    logger.info(f"💾 Veriler kaydedildi: {repo.feed_path}")
    logger.info(f"⏱️  Toplam süre: {total_duration:.0f}ms (Phase1: {phase1_duration:.0f}ms)")
    logger.info("=" * 60)

    return summarization_results


def _run_summarization_phase(
    clusters: List[Dict],
    api_key: Optional[str],
    summarizer: Optional[Any],
    repo,
    pipeline_start: float,
    delta_mode: bool = True,
) -> List[Dict]:
    """Run summarization phase with delta detection."""
    cfg = get_config()
    groq_api_key = cfg.get_groq_api_key(api_key)
    groq_cfg = cfg.groq

    summarizer_instance = summarizer or create_resilient_summarizer(api_key=api_key)
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if delta_mode:
        # Load existing to check what needs summarization
        existing = repo.load_or_fallback()
        existing_by_url = {a.get("link", ""): a for a in existing if a.get("link")}

        # Filter clusters needing summarization
        to_summarize: List[Dict] = []
        for cluster in clusters:
            url = cluster["main_article"]["link"]
            existing_article = existing_by_url.get(url)
            if not existing_article or existing_article.get("summary_status") == "pending":
                to_summarize.append(cluster)

        logger.info(f"      -> Delta: {len(to_summarize)}/{len(clusters)} haber özetlenecek (kalanı zaten özetli)")

        if not to_summarize:
            logger.info("      -> Hiç yeni haber yok, özetleme atlandı.")
            return repo.load_or_fallback()
    else:
        to_summarize = clusters

    # Prepare articles for summarization
    cluster_articles = [(c["main_article"]["title"], c["main_article"]["content"]) for c in to_summarize]

    if summarizer_instance and groq_api_key:
        logger.info("      -> Groq Cloud (Model Rotation) aktif")

        if groq_cfg.sequential_spillover_enabled and len(groq_cfg.model_rotation) > 1:
            logger.info(
                f"      -> Sequential Spillover: workers={groq_cfg.spillover_max_workers}, "
                f"delay={groq_cfg.spillover_request_delay}s, max_tokens={groq_cfg.spillover_max_tokens}"
            )
            all_ai_results = summarizer_instance.summarize_sequential_spillover(
                articles=cluster_articles,
                models=list(groq_cfg.model_rotation),
                max_workers=groq_cfg.spillover_max_workers,
                request_delay=groq_cfg.spillover_request_delay,
                max_tokens=groq_cfg.spillover_max_tokens,
                max_content_chars=groq_cfg.spillover_max_content_chars,
            )
        elif groq_cfg.sequential_model_processing and len(groq_cfg.model_rotation) > 1:
            logger.info("      -> Sequential Model Processing")
            all_ai_results = []
            for model in groq_cfg.model_rotation:
                model_results = summarizer_instance.summarize_batch_fixed_model(cluster_articles, model)
                all_ai_results.extend(model_results)
                time.sleep(1.0)
            all_ai_results = all_ai_results[:len(cluster_articles)]  # Trim if over
        else:
            logger.info("      -> Parallel Model Rotation")
            with ThreadPoolExecutor(max_workers=min(len(to_summarize), 2)) as executor:
                futures = [executor.submit(summarizer_instance.summarize, t, c) for t, c in cluster_articles]
                all_ai_results = [f.result() for f in as_completed(futures)]
    else:
        logger.info("      -> Yerel NLP (Sıfır Token)")
        all_ai_results = [summarizer_instance.summarize(t, c) for t, c in cluster_articles]

    # Build summary updates for repository
    summary_updates = []
    for cluster, ai_result in zip(to_summarize, all_ai_results):
        summary_updates.append({
            "link": cluster["main_article"]["link"],
            "summary": ai_result.summary,
            "ai_engine": ai_result.engine,
            "ai_model": ai_result.model,
            "summary_status": "completed" if ai_result.success else "failed",
            "updated_at": now_iso,
        })

    # Update summaries in repository
    updated = repo.update_article_summaries(summary_updates)
    logger.info(f"      -> {updated} haber özetleri güncellendi.")

    return repo.load_or_fallback()


if __name__ == "__main__":
    run_pipeline()