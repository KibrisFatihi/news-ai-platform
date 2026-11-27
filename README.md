# Gündem Özeti — AI Destekli Haber Derleyici & Editoryal Okuyucu

20'den fazla Türkçe ve uluslararası RSS kaynağını eşzamanlı tarayan, benzer haberleri kümeleyen, yapay zeka (Groq LLaMA 3.3) veya sıfır maliyetli yerel NLP (TextRank) ile özetleyen bağımsız haber platformu.

---

## Öne Çıkan Özellikler

* **Eşzamanlı RSS Tarama & Görsel Çıkarma:** `ThreadPoolExecutor` ile 20+ kaynaktan paralel haber çekimi; `media_content`, `media_thumbnail`, `enclosures` ve HTML içeriklerinden otomatik görsel URL çıkarımı.
* **Akıllı Kümeleme & Kategorizasyon (NLP):**
  * Kural tabanlı Türkçe kategori sınıflandırma (*Savunma & Teknoloji, Bilim, Ekonomi, Politika & Gündem, Dünya, Spor, Genel Gündem*).
  * TF-IDF & Kosinüs Benzerliği ile farklı kaynaklardaki aynı haberleri tek konuda birleştirme (`related_count`, `related_sources`).
* **Dayanıklı Özetleme Mimarisi:**
  * **Birincil Motor:** Groq API (LLaMA 3.3 / model rotasyonu, Rate Limiter ve Circuit Breaker korumalı).
  * **Otomatik Yedek (Fallback):** API anahtarı yoksa veya kota dolarsa yerel **Extractive TextRank** motoru devreye girer.
* **Editoryal Arayüz (70/30 Düzen):**
  * Üstte **Günün Manşetleri** (1 ana manşet + 2 ikincil vitrin kartı).
  * Solda görselli/tipografik haber akışı, sağda `sticky` yardımcı panel (**Çoklu Kaynakta Öne Çıkanlar**, **Son Dakika Akışı**, **Kategori Dağılımı**).
* **Türkiye Hava Durumu Bülteni:** Open-Meteo API üzerinden İstanbul, Ankara, İzmir ve 12 ek il için anlık sıcaklık, hissedilen sıcaklık ve rüzgar verileri (15 dk önbellekli).

---

## Teknoloji Yığını

* **Backend:** Python 3.9+, standart kütüphane `http.server` (sıfır ağır framework bağımlılığı), `feedparser`, `requests`, `pydantic`
* **Frontend:** Vanilla JavaScript, Tailwind CSS (CDN), Editorial Typography (`Newsreader` + `Inter`)
* **Test & Kalite:** `pytest` (260+ birim ve entegrasyon testi), `ruff`, `mypy`

---

## Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Haber Akışını Oluşturun (İlk Çalıştırma)
```bash
python main.py
```
> İşlenen haberler `data/news_feed.json` dosyasına atomik olarak kaydedilir.

### 3. Web Sunucusunu Başlatın
```bash
python server.py
```
Tarayıcıdan `http://localhost:8000` adresini açın. İsterseniz sağ üstteki **API Ayarı** butonundan ücretsiz Groq API anahtarınızı girebilirsiniz (anahtar yalnızca tarayıcınızda saklanır).

### 4. Testleri Çalıştırın
```bash
pytest -q
```

---

## API Uç Noktaları

| Metot | Yol | Açıklama |
| :--- | :--- | :--- |
| `GET` | `/api/news` | Özetlenmiş ve kümelenmiş haber akışını döndürür (ETag / 304 destekli) |
| `POST` | `/api/refresh` | Canlı RSS tarama ve özetleme boru hattını tetikler (`X-Refresh-Token` korumalı) |
| `GET` | `/api/weather` | Türkiye'deki 15 il için anlık hava durumu verisini döndürür |
| `GET` | `/api/status` | Sunucu durumu, aktif özetleme motoru ve refresh token bilgisi |
| `GET` | `/api/health` | Detaylı bileşen sağlık kontrolü |
