# Dosya İlişkileri ve Kronolojik Çalışma Sırası

## 1. KRONOLOJİK ÇALIŞMA SIRASI (Göreve Göre)

### Aşama 1: Model Yönetimi (Opsiyonel - İlk Kurulum)
**`add_model.py`**
- **Görev**: Yeni sentiment modelleri eklemek/çıkarmak için yardımcı script
- **Ne zaman çalışır**: İlk kurulumda veya yeni model eklerken
- **Çıktı**: `model_configs.json` dosyası oluşturur/günceller
- **Bağımlılık**: `sp500_ensemble_sentiment_pipeline.py` içindeki `DEFAULT_MODELS`'i import eder

### Aşama 2: Veri İşleme (Ana Pipeline)
**`sp500_ensemble_sentiment_pipeline.py`** ⭐ **ANA DOSYA**
- **Görev**: Ham haber başlıklarını okuyup, birden fazla sentiment modeliyle analiz eder ve ensemble sonuç üretir
- **Ne zaman çalışır**: Her yeni veri seti için veya model değişikliğinden sonra
- **Girdi**: `SP500_news/raw/sp500_headlines_2008_2024.csv` (ham veri)
- **Çıktı**: 
  - `SP500_news/processed/headlines_with_ensemble.csv` (işlenmiş başlıklar)
  - `SP500_news/processed/sp500_headlines_daily_agg_ensemble.csv` (günlük toplamlar)
  - `SP500_news/processed/sp500_headlines_daily_agg.csv` (legacy isim - backtest için)
- **Bağımlılık**: Bağımsız çalışır, ama `add_model.py` tarafından kullanılabilir

**ALTERNATİF: `sp500_finbert_pipeline.py`**
- **Görev**: Sadece FinBERT modeliyle sentiment analizi (ensemble olmadan)
- **Ne zaman çalışır**: Ensemble yerine tek model kullanmak istendiğinde
- **Not**: `sp500_ensemble_sentiment_pipeline.py` ile **alakasız** - ayrı bir pipeline

### Aşama 3: Backtest (Strateji Testi)
**`sp500_sentiment_backtest.py`** ⭐ **ANA DOSYA**
- **Görev**: Sentiment sonuçlarını kullanarak trading stratejisi test eder ve performans metrikleri hesaplar
- **Ne zaman çalışır**: Pipeline'dan sonra, strateji performansını görmek için
- **Girdi**: 
  - `SP500_news/processed/sp500_headlines_daily_agg.csv` (sentiment toplamları)
  - `SP500_news/raw/sp500_headlines_2008_2024.csv` (fiyat verisi)
- **Çıktı**: 
  - `SP500_news/results/backtest_metrics.json`
  - `SP500_news/results/backtest_equity_curve.csv`
  - `SP500_news/results/backtest_summary.txt`
- **Bağımlılık**: Pipeline'dan çıkan dosyalara ihtiyaç duyar

### Aşama 4: Karşılaştırma ve Optimizasyon (Opsiyonel)

**`compare_models.py`**
- **Görev**: Farklı modellerin performansını karşılaştırır
- **Ne zaman çalışır**: Hangi modelin daha iyi olduğunu görmek için
- **Çalışma sırası**:
  1. Önce `sp500_ensemble_sentiment_pipeline.py` çalıştırır (subprocess)
  2. Sonra `sp500_sentiment_backtest.py` çalıştırır (subprocess)
  3. Sonuçları karşılaştırır
- **Bağımlılık**: Hem pipeline hem backtest scriptlerini çağırır

**`optimize_weights.py`**
- **Görev**: Ensemble modellerinin ağırlıklarını optimize eder (grid search)
- **Ne zaman çalışır**: En iyi model ağırlıklarını bulmak için
- **Çalışma sırası**:
  1. `sp500_ensemble_sentiment_pipeline.py` fonksiyonlarını import eder
  2. Farklı ağırlık kombinasyonları için pipeline çalıştırır
  3. Her kombinasyon için `sp500_sentiment_backtest.py` çalıştırır (subprocess)
  4. En iyi ağırlıkları bulur
- **Bağımlılık**: Hem pipeline hem backtest scriptlerini kullanır

### Aşama 0: Test (Opsiyonel - Geliştirme Aşaması)
**`test_ensemble_pipeline.py`**
- **Görev**: Pipeline'ın düzgün çalışıp çalışmadığını test eder
- **Ne zaman çalışır**: Yeni model ekledikten sonra veya değişiklik yaptıktan sonra
- **Bağımlılık**: `sp500_ensemble_sentiment_pipeline.py` fonksiyonlarını import eder

---

## 2. DOĞRUDAN İLİŞKİLİ DOSYALAR

### Grup 1: Ensemble Pipeline Ekosistemi
**Doğrudan İlişkili:**
- `sp500_ensemble_sentiment_pipeline.py` ⭐ (merkez)
- `add_model.py` → `sp500_ensemble_sentiment_pipeline.py` (import eder)
- `test_ensemble_pipeline.py` → `sp500_ensemble_sentiment_pipeline.py` (import eder)
- `optimize_weights.py` → `sp500_ensemble_sentiment_pipeline.py` (import eder + subprocess çağırır)
- `compare_models.py` → `sp500_ensemble_sentiment_pipeline.py` (subprocess çağırır)
- `compare_models.py` → `sp500_sentiment_backtest.py` (subprocess çağırır)
- `optimize_weights.py` → `sp500_sentiment_backtest.py` (subprocess çağırır)

**İlişki Tipi:**
- `add_model.py` ve `test_ensemble_pipeline.py`: Fonksiyon import eder
- `compare_models.py` ve `optimize_weights.py`: Subprocess ile script çağırır

### Grup 2: Backtest Ekosistemi
**Doğrudan İlişkili:**
- `sp500_sentiment_backtest.py` ⭐ (merkez)
- `compare_models.py` → `sp500_sentiment_backtest.py` (subprocess çağırır)
- `optimize_weights.py` → `sp500_sentiment_backtest.py` (subprocess çağırır)

**İlişki Tipi:**
- Her ikisi de backtest scriptini subprocess ile çağırır

### Grup 3: Standalone (Bağımsız)
**Alakasız:**
- `sp500_finbert_pipeline.py` → **Hiçbir dosyayla doğrudan ilişkisi yok**
  - Alternatif bir pipeline
  - Ensemble pipeline'dan bağımsız çalışır
  - Sadece FinBERT kullanır

---

## 3. VERİ AKIŞI DİYAGRAMI

```
[RAW DATA]
    ↓
[sp500_ensemble_sentiment_pipeline.py]
    ↓
[Processed Data: daily_agg.csv]
    ↓
[sp500_sentiment_backtest.py]
    ↓
[Backtest Results: metrics.json, equity_curve.csv]
```

**Opsiyonel Dallar:**
- `add_model.py` → Model config günceller → Pipeline'a etki eder
- `compare_models.py` → Pipeline + Backtest → Karşılaştırma sonuçları
- `optimize_weights.py` → Pipeline (çoklu) + Backtest (çoklu) → Optimal ağırlıklar
- `test_ensemble_pipeline.py` → Pipeline fonksiyonlarını test eder

---

## 4. ÖZET TABLO

| Dosya | Ana Görev | Çalışma Sırası | Doğrudan Bağımlı Olduğu Dosyalar | Çağırdığı Dosyalar |
|-------|-----------|----------------|----------------------------------|-------------------|
| `add_model.py` | Model yönetimi | 1 (opsiyonel) | `sp500_ensemble_sentiment_pipeline.py` (import) | - |
| `sp500_ensemble_sentiment_pipeline.py` | Ana pipeline | 2 | - | - |
| `sp500_sentiment_backtest.py` | Backtest | 3 | - | - |
| `compare_models.py` | Model karşılaştırma | 4 (opsiyonel) | - | Pipeline + Backtest (subprocess) |
| `optimize_weights.py` | Ağırlık optimizasyonu | 4 (opsiyonel) | `sp500_ensemble_sentiment_pipeline.py` (import) | Pipeline + Backtest (subprocess) |
| `test_ensemble_pipeline.py` | Test | 0 (opsiyonel) | `sp500_ensemble_sentiment_pipeline.py` (import) | - |
| `sp500_finbert_pipeline.py` | Tek model pipeline | Alternatif | - | - |

---

## 5. TİPİK KULLANIM SENARYOLARI

### Senaryo 1: İlk Kurulum ve Çalıştırma
```
1. add_model.py (model ekleme - opsiyonel)
2. sp500_ensemble_sentiment_pipeline.py (veri işleme)
3. sp500_sentiment_backtest.py (backtest)
```

### Senaryo 2: Model Karşılaştırma
```
1. compare_models.py (otomatik olarak pipeline + backtest çalıştırır)
```

### Senaryo 3: Ağırlık Optimizasyonu
```
1. optimize_weights.py (otomatik olarak çoklu pipeline + backtest çalıştırır)
```

### Senaryo 4: Test
```
1. test_ensemble_pipeline.py (pipeline fonksiyonlarını test eder)
```

### Senaryo 5: Tek Model (Ensemble Olmadan)
```
1. sp500_finbert_pipeline.py (sadece FinBERT)
2. sp500_sentiment_backtest.py (backtest)
```

