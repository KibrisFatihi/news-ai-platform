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
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import List, Optional

from backend.logger import get_logger
from config import NLPConfig, get_config

logger = get_logger("nlp_engine")


# --- Turkish Abbreviation Patterns for Sentence Splitting ---
# Common Turkish abbreviations that should NOT trigger sentence boundaries
# (without trailing period - we'll match them with optional period)
_TURKISH_ABBREVIATIONS = frozenset({
    "dr", "prof", "av", "doç", "yrd", "vb", "vd", "veya", "örn", "örnek",
    "mah", "cad", "sk", "sok", "no", "blv", "şube", "ptt",
    "kg", "km", "m", "cm", "mm", "ml", "lt", "g", "mg",
    "am", "pm", "gmt", "utc", "iso", "tcp", "ip", "url", "http", "https",
    "html", "css", "js", "api", "ui", "ux", "ai", "ml", "dl", "nlp",
    "usa", "eu", "uk", "un", "nato", "oecd", "gdp", "cpi", "ppi",
    "abd", "tbm", "tsk", "mgm", "thk", "tyb", "atm", "pos",
})

# Pattern to match abbreviation + period + space + capital letter
# We use this to TEMPORARILY protect the period during splitting
# Format: word boundary + abbreviation + period + whitespace + Turkish capital letter
# We build a regex that matches each abbreviation individually (fixed width)
_ABBREVIATION_PATTERNS = [
    re.compile(rf'\b{re.escape(abbr)}\.(?=\s+[A-ZÇĞİÖŞÜ])', re.IGNORECASE)
    for abbr in _TURKISH_ABBREVIATIONS
]

# Also match single capital letter + period + space + capital (e.g., "A. Başkan")
_SINGLE_LETTER_ABBR_PATTERN = re.compile(r'\b[A-ZÇĞİÖŞÜ]\.(?=\s+[A-ZÇĞİÖŞÜ])')

# Number with decimal pattern (e.g., 14.30, 3.14) - fixed width lookbehind OK
_NUMBER_DECIMAL_PATTERN = re.compile(r'(?<=\d)\.(?=\d)')


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


# --- Cosine Similarity with Vector Caching (LRU) ---

# Maximum cache size to prevent memory leaks in long-running processes
_MAX_VECTOR_CACHE_SIZE = 5000

# Cache for tokenized vectors to avoid recomputation (LRU using OrderedDict)
_vector_cache: OrderedDict[str, Counter] = OrderedDict()
_vector_cache_lock = None  # Will be initialized on first use


def _get_vector_cache_lock():
    """Lazy initialization of threading lock."""
    global _vector_cache_lock
    if _vector_cache_lock is None:
        import threading
        _vector_cache_lock = threading.Lock()
    return _vector_cache_lock


def _get_cache_key(text: str, config_version: int) -> str:
    """Generate cache key for text + config version.

    Uses config_version instead of id(cfg) to avoid issues when config
    object is recreated (new id but same content).
    """
    return f"{config_version}:{hash(text)}"


def _get_config_version(cfg: NLPConfig) -> int:
    """Generate a stable version identifier for config based on content.

    This avoids the issue where id(cfg) changes when config is recreated
    but content remains the same.
    """
    # Use a hash of the relevant config fields that affect tokenization
    import hashlib
    config_str = f"{cfg.similarity_threshold}:{cfg.summary_sentences}:{','.join(sorted(cfg.turkish_stopwords))}"
    return int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16)


def _get_cached_vector(text: str, cfg: NLPConfig) -> Optional[Counter]:
    """Get cached token frequency vector if available (LRU access)."""
    cache_key = _get_cache_key(text, _get_config_version(cfg))
    with _get_vector_cache_lock():
        if cache_key in _vector_cache:
            # Move to end (most recently used)
            _vector_cache.move_to_end(cache_key)
            return _vector_cache[cache_key]
        return None


def _set_cached_vector(text: str, cfg: NLPConfig, vector: Counter) -> None:
    """Cache token frequency vector with LRU eviction."""
    cache_key = _get_cache_key(text, _get_config_version(cfg))
    with _get_vector_cache_lock():
        if cache_key in _vector_cache:
            # Update and move to end
            _vector_cache[cache_key] = vector
            _vector_cache.move_to_end(cache_key)
        else:
            # Add new entry
            _vector_cache[cache_key] = vector
            # Evict oldest if over limit
            if len(_vector_cache) > _MAX_VECTOR_CACHE_SIZE:
                _vector_cache.popitem(last=False)  # Remove oldest (first item)


def _tokenize_and_vectorize(text: str, cfg: NLPConfig) -> Counter:
    """
    Tokenize text and return frequency vector.
    Uses LRU caching to avoid recomputation and prevent memory leaks.
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

        for j in range(i + 1, len(articles)):
            if not assigned[j]:
                text_j = article_texts[j]

                # Use the public compute_cosine_similarity function (DRY)
                # This uses cached vectors internally
                sim = compute_cosine_similarity(text_i, text_j, cfg)

                if sim >= sim_threshold:
                    current_cluster["related_articles"].append({
                        "article": articles[j],
                        "similarity_score": round(sim, 2)
                    })
                    assigned[j] = True

        clusters.append(current_cluster)

    # Deduplicate related_sources in each cluster
    # Keep the article with the highest similarity score per source
    for cluster in clusters:
        source_best: dict[str, tuple[float, dict]] = {}
        for rel in cluster["related_articles"]:
            source = rel["article"].get("source", "")
            score = rel["similarity_score"]
            # Keep the highest scoring article per source
            if source not in source_best or score > source_best[source][0]:
                source_best[source] = (score, rel)

        # Rebuild list with best per source
        cluster["related_articles"] = [rel for _, rel in source_best.values()]

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

# Minimum sentence length to be considered for summary (filters out fragments)
_MIN_SENTENCE_LENGTH = 25

# Maximum title overlap ratio to exclude a sentence (e.g., first sentence often repeats title)
_MAX_TITLE_OVERLAP_RATIO = 0.8

# Positional weight decay for early sentences (first sentence gets slightly less weight)
_POSITIONAL_DECAY = 0.95


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling Turkish abbreviations and decimal numbers.

    Uses a multi-step approach:
    1. Protect known abbreviations and decimal numbers by temporarily replacing periods
    2. Split on sentence boundaries
    3. Restore protected periods

    Args:
        text: Input text to split

    Returns:
        List of sentence strings (unfiltered - filtering done at call site)
    """
    # Step 1: Protect decimal numbers (e.g., 14.30 -> 14<DECIMAL>30)
    protected = _NUMBER_DECIMAL_PATTERN.sub('<DECIMAL>', text)

    # Step 2: Protect known abbreviations (e.g., Dr. -> Dr<ABBR>)
    # Match abbreviation followed by period and space + capital letter
    def _protect_abbr(match):
        return match.group(0).replace('.', '<ABBR>')

    # Apply each abbreviation pattern
    for pattern in _ABBREVIATION_PATTERNS:
        protected = pattern.sub(_protect_abbr, protected)

    # Also protect single letter abbreviations (A. Başkan -> A<ABBR> Başkan)
    protected = _SINGLE_LETTER_ABBR_PATTERN.sub(_protect_abbr, protected)

    # Step 3: Split on sentence boundaries (period, exclamation, question + space)
    sentences = re.split(r'(?<=[.!?]) +', protected.strip())

    # Step 4: Restore protected characters
    restored = [s.replace('<DECIMAL>', '.').replace('<ABBR>', '.') for s in sentences]

    # Return all sentences (filtering done at call site based on context)
    return [s.strip() for s in restored if s.strip()]


def _sentence_overlaps_title(sentence: str, title: str, cfg: NLPConfig) -> bool:
    """Check if sentence has high word overlap with title (likely a title repeat)."""
    if not title:
        return False
    sent_tokens = set(tokenize(sentence, cfg))
    title_tokens = set(tokenize(title, cfg))
    if not title_tokens:
        return False
    overlap = len(sent_tokens & title_tokens) / len(title_tokens)
    return overlap >= _MAX_TITLE_OVERLAP_RATIO


def extractive_summarize(
    text: str,
    num_sentences: Optional[int] = None,
    nlp_config: Optional[NLPConfig] = None,
    title: Optional[str] = None  # Optional title for overlap detection
) -> str:
    """
    NLP Extractive Summarization (Sentence Scoring based on Word Frequency).
    Lightweight, fast, and does not require GPU or external API.

    Improvements:
    - Handles Turkish abbreviations (Dr., Prof., vb., etc.) and decimals (14.30)
    - Filters out too-short fragments (< 25 chars)
    - Excludes sentences that are near-duplicates of the title
    - Applies mild positional decay (earlier sentences slightly downweighted)

    Args:
        text: Input text to summarize
        num_sentences: Number of sentences in summary (uses config default if None)
        nlp_config: Optional NLP config override
        title: Optional article title to detect/avoid title repeats in summary

    Returns:
        Extracted summary text
    """
    cfg = nlp_config or get_config().nlp
    sentences_count = num_sentences or cfg.summary_sentences

    start_time = time.time()

    # Handle None or empty input
    if not text:
        return ""

    # Split text into sentences with abbreviation/decimal protection
    sentences = _split_sentences(text)
    if len(sentences) <= sentences_count:
        return " ".join(sentences)

    # Filter out too-short fragments (likely split errors) ONLY if we have enough remaining
    # This preserves short sentences for short texts
    filtered_sentences = [s for s in sentences if len(s) >= _MIN_SENTENCE_LENGTH]
    if len(filtered_sentences) >= sentences_count:
        sentences = filtered_sentences
    # Otherwise keep all sentences (including short ones) to avoid returning empty

    tokens = tokenize(text, cfg)
    if not tokens:
        # Fallback: return first N characters
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

        # Base frequency score
        base_score = sum(freq_table.get(w, 0) for w in words) / (len(words) + 1)

        # Positional decay: slightly penalize very early sentences (often title repeats)
        # First sentence gets POSITIONAL_DECAY, second gets POSITIONAL_DECAY^2, etc.
        positional_factor = _POSITIONAL_DECAY ** (idx + 1)

        # Title overlap penalty: heavily penalize sentences too similar to title
        if title and _sentence_overlaps_title(sentence, title, cfg):
            positional_factor *= 0.1  # Heavy penalty

        sentence_scores[idx] = base_score * positional_factor

    # Select top N sentences by score, then sort by original position
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