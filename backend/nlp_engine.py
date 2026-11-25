"""
NLP Engine Module
Implements:
1. Category Classification (Rule-based with precompiled regex, title weighting, root-prefix matching)
2. Semantic Similarity & Clustering / Deduplication (Cosine Similarity with vector caching)
3. Automatic Summarization (Extractive TextRank / Frequency-based)
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter
from functools import lru_cache
from typing import List, Optional

from backend.logger import get_logger
from config import NLPConfig, get_config

logger = get_logger("nlp_engine")


# --- Tokenization ---

def tokenize(text: str, nlp_config: Optional[NLPConfig] = None) -> List[str]:
    """
    Tokenize and normalize Turkish text.

    Args:
        text: Input text to tokenize
        nlp_config: Optional NLP config override (uses global config if None)

    Returns:
        List of normalized tokens
    """
    cfg = nlp_config or get_config().nlp
    # Handle Turkish İ (U+0130) and ı (U+0131) specially before casefold
    # İ -> i, ı -> i (for case-insensitive matching)
    text = text.replace('İ', 'i').replace('ı', 'i')
    # Now use casefold for other characters
    text = text.casefold()
    text = re.sub(r'[^a-zçğıöşü0-9\s]', ' ', text)
    tokens = [t.strip() for t in text.split() if t.strip() and t.strip() not in cfg.turkish_stopwords]
    return tokens


# --- Category Classification (Improved) ---

# Title weight multiplier for classification
TITLE_WEIGHT = 2.0


def classify_category(
    title: str,
    content: str,
    nlp_config: Optional[NLPConfig] = None
) -> str:
    """
    Classifies article into a category based on keyword density.
    Uses precompiled regex patterns, title weighting (2x), and root-prefix matching.

    Args:
        title: Article title
        content: Article content
        nlp_config: Optional NLP config override

    Returns:
        Category name
    """
    cfg = nlp_config or get_config().nlp
    patterns = cfg.get_compiled_patterns()

    scores = Counter()
    start_time = time.time()

    # Prepare text for matching (handle Turkish İ/ı)
    title_norm = title.replace('İ', 'i').replace('ı', 'i').casefold()
    content_norm = content.replace('İ', 'i').replace('ı', 'i').casefold()

    for category, cat_patterns in patterns.items():
        for pattern in cat_patterns:
            # Search in title (weighted 2x)
            title_matches = len(pattern.findall(title_norm))
            # Search in content
            content_matches = len(pattern.findall(content_norm))

            if title_matches or content_matches:
                # Title matches count double
                scores[category] += (title_matches * TITLE_WEIGHT) + content_matches

    if scores:
        result = scores.most_common(1)[0][0]
    else:
        result = cfg.default_category

    duration_ms = (time.time() - start_time) * 1000
    logger.debug(
        f"Category classified: {result}",
        extra={"extra_fields": {
            "event": "classify_category",
            "category": result,
            "scores": dict(scores),
            "duration_ms": duration_ms
        }}
    )

    return result


# --- Cosine Similarity with Vector Caching ---

# Cache for tokenized vectors to avoid recomputation
_vector_cache: dict[str, Counter] = {}
_vector_cache_lock = None  # Will be initialized on first use


def _get_vector_cache_lock():
    """Lazy initialization of threading lock."""
    global _vector_cache_lock
    if _vector_cache_lock is None:
        import threading
        _vector_cache_lock = threading.Lock()
    return _vector_cache_lock


def _get_cache_key(text: str, config_id: int) -> str:
    """Generate cache key for text + config combination."""
    return f"{config_id}:{hash(text)}"


def _get_cached_vector(text: str, cfg: NLPConfig) -> Optional[Counter]:
    """Get cached token frequency vector if available."""
    cache_key = _get_cache_key(text, id(cfg))
    with _get_vector_cache_lock():
        return _vector_cache.get(cache_key)


def _set_cached_vector(text: str, cfg: NLPConfig, vector: Counter) -> None:
    """Cache token frequency vector."""
    cache_key = _get_cache_key(text, id(cfg))
    with _get_vector_cache_lock():
        _vector_cache[cache_key] = vector


def _tokenize_and_vectorize(text: str, cfg: NLPConfig) -> Counter:
    """
    Tokenize text and return frequency vector.
    Uses caching to avoid recomputation.
    """
    # Check cache first
    cached = _get_cached_vector(text, cfg)
    if cached is not None:
        return cached

    tokens = tokenize(text, cfg)
    if not tokens:
        return Counter()

    vector = Counter(tokens)
    _set_cached_vector(text, cfg, vector)
    return vector


def compute_cosine_similarity(
    text1: str,
    text2: str,
    nlp_config: Optional[NLPConfig] = None
) -> float:
    """
    Calculates Cosine Similarity between two texts using token frequencies.
    Uses cached vectors for performance.

    Args:
        text1: First text
        text2: Second text
        nlp_config: Optional NLP config override

    Returns:
        Similarity score between 0.0 and 1.0
    """
    cfg = nlp_config or get_config().nlp

    vec1 = _tokenize_and_vectorize(text1, cfg)
    vec2 = _tokenize_and_vectorize(text2, cfg)

    if not vec1 or not vec2:
        return 0.0

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(val**2 for val in vec1.values())
    sum2 = sum(val**2 for val in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0

    return float(numerator) / denominator


def clear_vector_cache() -> None:
    """Clear the vector cache (useful for testing or config changes)."""
    global _vector_cache
    with _get_vector_cache_lock():
        _vector_cache.clear()
    logger.debug("Vector cache cleared", extra={"extra_fields": {"event": "vector_cache_cleared"}})


# --- Clustering with Deduplication ---

def cluster_similar_news(
    articles: List[dict],
    threshold: Optional[float] = None,
    nlp_config: Optional[NLPConfig] = None
) -> List[dict]:
    """
    Groups similar articles into clusters to prevent redundancy.
    Articles describing the same event are grouped together.
    Uses cached vectors for performance; deduplicates related_sources.

    Args:
        articles: List of article dictionaries with 'title' and 'content' keys
        threshold: Similarity threshold (uses config default if None)
        nlp_config: Optional NLP config override

    Returns:
        List of clusters, each with 'main_article' and 'related_articles'
    """
    cfg = nlp_config or get_config().nlp
    sim_threshold = threshold or cfg.similarity_threshold

    start_time = time.time()
    clusters = []
    assigned = [False] * len(articles)

    # Pre-compute combined text for each article (tokenized once)
    article_texts = []
    for article in articles:
        combined = f"{article['title']} {article['content']}"
        article_texts.append(combined)

    for i in range(len(articles)):
        if assigned[i]:
            continue

        current_cluster = {
            "main_article": articles[i],
            "related_articles": []
        }
        assigned[i] = True

        text_i = article_texts[i]
        vec_i = _tokenize_and_vectorize(text_i, cfg)

        for j in range(i + 1, len(articles)):
            if not assigned[j]:
                text_j = article_texts[j]
                vec_j = _tokenize_and_vectorize(text_j, cfg)

                # Compute cosine similarity from cached vectors
                if vec_i and vec_j:
                    intersection = set(vec_i.keys()) & set(vec_j.keys())
                    numerator = sum(vec_i[x] * vec_j[x] for x in intersection)
                    sum1 = sum(val**2 for val in vec_i.values())
                    sum2 = sum(val**2 for val in vec_j.values())
                    denominator = math.sqrt(sum1) * math.sqrt(sum2)

                    if denominator:
                        sim = float(numerator) / denominator
                    else:
                        sim = 0.0
                else:
                    sim = 0.0

                if sim >= sim_threshold:
                    current_cluster["related_articles"].append({
                        "article": articles[j],
                        "similarity_score": round(sim, 2)
                    })
                    assigned[j] = True

        clusters.append(current_cluster)

    # Deduplicate related_sources in each cluster
    for cluster in clusters:
        seen_sources = set()
        unique_related = []
        for rel in cluster["related_articles"]:
            source = rel["article"].get("source", "")
            if source not in seen_sources:
                seen_sources.add(source)
                unique_related.append(rel)
        cluster["related_articles"] = unique_related

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"Clustering completed: {len(articles)} articles -> {len(clusters)} clusters",
        extra={"extra_fields": {
            "event": "cluster_similar_news",
            "input_count": len(articles),
            "cluster_count": len(clusters),
            "threshold": sim_threshold,
            "duration_ms": duration_ms
        }}
    )

    return clusters


# --- Extractive Summarization ---

def extractive_summarize(
    text: str,
    num_sentences: Optional[int] = None,
    nlp_config: Optional[NLPConfig] = None
) -> str:
    """
    NLP Extractive Summarization (Sentence Scoring based on Word Frequency).
    Lightweight, fast, and does not require GPU or external API.

    Args:
        text: Input text to summarize
        num_sentences: Number of sentences in summary (uses config default if None)
        nlp_config: Optional NLP config override

    Returns:
        Extracted summary text
    """
    cfg = nlp_config or get_config().nlp
    sentences_count = num_sentences or cfg.summary_sentences

    start_time = time.time()

    # Handle None or empty input
    if not text:
        return ""

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if len(sentences) <= sentences_count:
        return text.strip()

    tokens = tokenize(text, cfg)
    if not tokens:
        return text[:150] + "..." if len(text) > 150 else text

    freq_table = Counter(tokens)
    max_freq = max(freq_table.values())
    for word in freq_table:
        freq_table[word] = freq_table[word] / max_freq

    sentence_scores = {}
    for idx, sentence in enumerate(sentences):
        words = tokenize(sentence, cfg)
        if not words:
            continue
        score = sum(freq_table.get(w, 0) for w in words) / (len(words) + 1)
        sentence_scores[idx] = score

    top_indices = sorted(
        sorted(sentence_scores.keys(), key=lambda i: sentence_scores[i], reverse=True)[:sentences_count]
    )
    summary = " ".join([sentences[i] for i in top_indices])

    duration_ms = (time.time() - start_time) * 1000
    logger.debug(
        f"Extractive summary generated: {len(summary)} chars",
        extra={"extra_fields": {
            "event": "extractive_summarize",
            "input_length": len(text),
            "output_length": len(summary),
            "sentences": sentences_count,
            "duration_ms": duration_ms
        }}
    )

    return summary


# For backwards compatibility
TURKISH_STOPWORDS = get_config().nlp.turkish_stopwords
CATEGORY_KEYWORDS = get_config().nlp.category_keywords