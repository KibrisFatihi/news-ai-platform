"""
Tests for the NLP engine module.
"""
import pytest

from backend.nlp_engine import (
    classify_category,
    cluster_similar_news,
    compute_cosine_similarity,
    extractive_summarize,
    tokenize,
    clear_vector_cache,
)


class TestTokenize:
    """Tests for tokenize function."""

    def test_tokenize_basic(self):
        """Test basic tokenization."""
        text = "Bu bir test metnidir"
        tokens = tokenize(text)
        assert "bir" not in tokens  # stopword
        assert "test" in tokens
        assert "metnidir" in tokens

    def test_tokenize_lowercase(self):
        """Test that text is lowercased."""
        text = "BÜYÜK HARFLER"
        tokens = tokenize(text)
        assert all(t.islower() for t in tokens)

    def test_tokenize_punctuation_removed(self):
        """Test that punctuation is removed."""
        text = "Merhaba, dünya! Nasılsın?"
        tokens = tokenize(text)
        assert "merhaba" in tokens
        assert "dünya" in tokens
        # Turkish dotless ı (U+0131) is normalized to i for case-insensitive matching
        assert "nasilsin" in tokens

    def test_tokenize_turkish_chars(self):
        """Test Turkish character handling."""
        # Use chr() to avoid encoding issues on Windows
        # İ = U+0130, ü = U+00FC, ş = U+015F
        text = chr(0x0130) + "stanbul'da g" + chr(0x00FC) + "zel bir g" + chr(0x00FC) + "n"
        tokens = tokenize(text)
        # With casefold(), Turkish İ (U+0130) properly becomes 'i'
        # so "İstanbul" -> "istanbul" (not split)
        assert "istanbul" in tokens
        assert "g" + chr(0x00FC) + "zel" in tokens
        assert "g" + chr(0x00FC) + "n" in tokens

    def test_tokenize_stopwords_filtered(self):
        """Test that Turkish stopwords are filtered."""
        text = "bu ve bir için ile"
        tokens = tokenize(text)
        assert len(tokens) == 0  # all stopwords

    def test_tokenize_numbers_preserved(self):
        """Test that numbers are preserved."""
        text = "Fiyat 100 TL ve 50 kuruş"
        tokens = tokenize(text)
        assert "100" in tokens
        assert "50" in tokens


class TestClassifyCategory:
    """Tests for classify_category function."""

    def test_classify_technology(self):
        """Test technology category classification."""
        title = "Yapay Zeka Teknolojisi Gelişiyor"
        content = "Yeni yapay zeka modelleri piyasaya sürülüyor. Teknoloji şirketleri yatırım yapıyor."
        category = classify_category(title, content)
        assert category == "Savunma & Teknoloji"

    def test_classify_economy(self):
        """Test economy category classification."""
        title = "Dolar Kuru Yükseldi"
        content = "Merkez Bankası faiz kararı aldı. Dolar ve euro kurları değişti. Ekonomi etkileniyor."
        category = classify_category(title, content)
        assert category == "Ekonomi"

    def test_classify_politics(self):
        """Test politics category classification."""
        title = "Cumhurbaşkanı Meclis'te Konuştu"
        content = "Cumhurbaşkanı AK Parti ve CHP hakkında konuştu. Seçimler yaklaşıyor."
        category = classify_category(title, content)
        assert category == "Politika & Gündem"

    def test_classify_world(self):
        """Test world category classification."""
        title = "ABD ve Rusya İlişkileri"
        content = "ABD başkanı Rusya ile görüşecek. Ukrayna savaşını tartışacaklar."
        category = classify_category(title, content)
        assert category == "Dünya"

    def test_classify_sports(self):
        """Test sports category classification."""
        title = "Galatasaray Şampiyon Oldu"
        content = "Fenerbahçe ve Beşiktaş de ligde yarışıyor. Transfer haberleri devam ediyor."
        category = classify_category(title, content)
        assert category == "Spor"

    def test_classify_default(self):
        """Test default category when no keywords match."""
        title = "Genel Bir Haber"
        content = "Bu haberin kategorisi belirgin değil. Sadece bilgi veriyor."
        category = classify_category(title, content)
        assert category == "Genel Gündem"

    def test_classify_case_insensitive(self):
        """Test case insensitive matching."""
        title = "YAPAY ZEKÂ HABERİ"
        content = "TEKNOLOJİ ŞİRKETLERİ YATIRIM YAPIYOR"
        category = classify_category(title, content)
        assert category == "Savunma & Teknoloji"

    def test_classify_title_weighting(self):
        """Test that title matches are weighted 2x."""
        # Title has 1 tech keyword, content has 2 economy keywords
        # Title weight = 2, Content weight = 2 -> tie, but tech should win due to title
        title = "Yapay Zeka Haberi"  # 1 tech keyword
        content = "Dolar kuru arttı. Enflasyon yükseldi."  # 2 economy keywords
        category = classify_category(title, content)
        # Title weight 2*1=2, Content weight 2 -> tech should win (title has priority in tie)
        assert category == "Savunma & Teknoloji"

    def test_classify_root_prefix_matching(self):
        """Test root-prefix matching for Turkish agglutination."""
        # "teknoloji" should match "teknolojik", "teknolojiler", etc.
        title = "Teknolojik Gelişmeler"
        content = "Yeni teknolojiler piyasaya sürüldü."
        category = classify_category(title, content)
        assert category == "Savunma & Teknoloji"

    def test_classify_kizilelma_matching(self):
        """Test kızılelma keyword matching."""
        title = "KIZILELMA Uçağı Testi"
        content = "Yeni müihimmat envantere girdi."
        category = classify_category(title, content)
        assert category == "Savunma & Teknoloji"

    def test_classify_hacker_is_technology(self):
        """Hacker / cyberattack news should classify as Savunma & Teknoloji even with 'piyasa' in text."""
        title = "GTA 6'yı sızdıran hacker'a hapis cezası"
        content = "Oyunu piyasaya sürülmesinden önce sızdıran hacker ceza aldı."
        assert classify_category(title, content) == "Savunma & Teknoloji"

    def test_classify_deneyimli_not_bilim(self):
        """'deneyimli' (experienced) should not falsely match 'deney' (experiment) in Bilim."""
        title = "Şirkette üst düzey atama"
        content = "Deneyimli yönetici genel müdür olarak atandı."
        assert classify_category(title, content) == "Ekonomi"


class TestComputeCosineSimilarity:
    """Tests for compute_cosine_similarity function."""

    def test_similarity_identical(self):
        """Test similarity of identical texts."""
        text = "Bu bir test metnidir"
        sim = compute_cosine_similarity(text, text)
        assert sim == pytest.approx(1.0)

    def test_similarity_different(self):
        """Test similarity of completely different texts."""
        text1 = "Teknoloji haberleri"
        text2 = "Ekonomi verileri"
        sim = compute_cosine_similarity(text1, text2)
        assert sim == 0.0

    def test_similarity_partial(self):
        """Test partial similarity."""
        text1 = "Yapay zeka teknolojisi gelişiyor"
        text2 = "Yapay zeka modelleri piyasaya çıkıyor"
        sim = compute_cosine_similarity(text1, text2)
        assert 0.0 < sim < 1.0

    def test_similarity_empty(self):
        """Test similarity with empty text."""
        assert compute_cosine_similarity("", "test") == 0.0
        assert compute_cosine_similarity("test", "") == 0.0
        assert compute_cosine_similarity("", "") == 0.0

    def test_similarity_stopwords_ignored(self):
        """Test that stopwords don't affect similarity."""
        text1 = "Bu bir test metnidir"
        text2 = "Test metnidir bu bir"
        sim = compute_cosine_similarity(text1, text2)
        assert sim == pytest.approx(1.0)  # Same tokens after stopword removal

    def test_similarity_caching(self, reset_vector_cache):
        """Test that vector caching works."""
        text1 = "Yapay zeka teknolojisi gelişiyor"
        text2 = "Yapay zeka modelleri piyasaya çıkıyor"

        # First call populates cache
        sim1 = compute_cosine_similarity(text1, text2)
        # Second call should use cache
        sim2 = compute_cosine_similarity(text1, text2)
        assert sim1 == sim2


class TestClusterSimilarNews:
    """Tests for cluster_similar_news function."""

    def test_cluster_basic(self, sample_articles):
        """Test basic clustering."""
        clusters = cluster_similar_news(sample_articles, threshold=0.1)

        assert len(clusters) > 0
        assert len(clusters) <= len(sample_articles)
        for cluster in clusters:
            assert "main_article" in cluster
            assert "related_articles" in cluster

    def test_cluster_no_duplicates(self, sample_articles):
        """Test that each article appears only once."""
        clusters = cluster_similar_news(sample_articles, threshold=0.1)

        all_articles = []
        for cluster in clusters:
            all_articles.append(cluster["main_article"]["id"])
            for rel in cluster["related_articles"]:
                all_articles.append(rel["article"]["id"])

        assert len(all_articles) == len(set(all_articles))

    def test_cluster_high_threshold(self, sample_articles):
        """Test with high similarity threshold (no clustering)."""
        clusters = cluster_similar_news(sample_articles, threshold=0.9)

        # With high threshold, each article should be its own cluster
        assert len(clusters) == len(sample_articles)
        for cluster in clusters:
            assert len(cluster["related_articles"]) == 0

    def test_cluster_low_threshold(self, sample_articles):
        """Test with low similarity threshold (max clustering)."""
        clusters = cluster_similar_news(sample_articles, threshold=0.01)

        # Should have fewer clusters
        assert len(clusters) <= len(sample_articles)

    def test_cluster_similarity_scores(self):
        """Test that similarity scores are included."""
        articles = [
            {"id": "1", "title": "AI Test", "content": "Yapay zeka test"},
            {"id": "2", "title": "AI Test 2", "content": "Yapay zeka test benzer"},
        ]
        clusters = cluster_similar_news(articles, threshold=0.1)

        for cluster in clusters:
            for rel in cluster["related_articles"]:
                assert "similarity_score" in rel
                assert 0 <= rel["similarity_score"] <= 1.0

    def test_cluster_empty_input(self):
        """Test with empty article list."""
        clusters = cluster_similar_news([])
        assert clusters == []

    def test_cluster_single_article(self, sample_articles):
        """Test with single article."""
        clusters = cluster_similar_news([sample_articles[0]], threshold=0.1)
        assert len(clusters) == 1
        assert clusters[0]["main_article"]["id"] == sample_articles[0]["id"]

    def test_cluster_related_sources_dedupe(self):
        """Test that related_sources are deduplicated."""
        articles = [
            {"id": "1", "title": "Test", "content": "Test içerik", "source": "BBC"},
            {"id": "2", "title": "Test 2", "content": "Test içerik benzer", "source": "BBC"},
            {"id": "3", "title": "Test 3", "content": "Test içerik benzer", "source": "TRT"},
        ]
        clusters = cluster_similar_news(articles, threshold=0.1)

        for cluster in clusters:
            sources = [r["article"]["source"] for r in cluster["related_articles"]]
            assert len(sources) == len(set(sources))  # No duplicates


class TestExtractiveSummarize:
    """Tests for extractive_summarize function."""

    def test_summarize_basic(self):
        """Test basic summarization."""
        text = "İlk cümle buradadır. İkinci cümle de buradadır. Üçüncü cümle buradadır. Dördüncü cümle buradadır."
        summary = extractive_summarize(text, num_sentences=2)

        # Should have 2 sentences
        sentences = summary.split(". ")
        assert len(sentences) == 2

    def test_summarize_short_text(self):
        """Test with text shorter than requested sentences."""
        text = "Kısa metin."
        summary = extractive_summarize(text, num_sentences=3)
        assert summary == text

    def test_summarize_two_sentences(self):
        """Test default 2 sentence summary."""
        text = "Birinci cümle. İkinci cümle. Üçüncü cümle. Dördüncü cümle. Beşinci cümle."
        summary = extractive_summarize(text)  # default 2

        sentences = [s for s in summary.split(". ") if s]
        assert len(sentences) == 2

    def test_summarize_empty(self):
        """Test empty input."""
        assert extractive_summarize("") == ""
        assert extractive_summarize(None) == ""

    def test_summarize_no_stopwords(self):
        """Test with text containing only stopwords."""
        text = "Bu ve bir için ile da de"
        summary = extractive_summarize(text)
        # Should return truncated original
        assert len(summary) <= len(text)

    def test_summarize_preserves_key_terms(self):
        """Test that important terms are preserved."""
        text = "Yapay zeka teknolojisi hızla gelişiyor. Şirketler büyük yatırım yapıyor. Gelecek umut verici."
        summary = extractive_summarize(text, num_sentences=2)

        # Should contain important terms
        assert "yapay zeka" in summary.lower() or "teknoloji" in summary.lower()


class TestClearVectorCache:
    """Tests for clear_vector_cache function."""

    def test_clear_vector_cache(self, reset_vector_cache):
        """Test that vector cache can be cleared."""
        from backend.nlp_engine import _vector_cache, _get_vector_cache_lock

        # Add something to cache
        with _get_vector_cache_lock():
            _vector_cache["test_key"] = "test_value"

        clear_vector_cache()

        with _get_vector_cache_lock():
            assert len(_vector_cache) == 0