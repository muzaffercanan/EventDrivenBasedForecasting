# SP500 Ensemble Sentiment Pipeline

Bu proje, birden fazla sentiment analizi modelini birleştirerek (ensemble) SP500 haber başlıklarından daha güvenilir duygu analizi yapmayı amaçlar.

## Özellikler

### Desteklenen Modeller

1. **ProsusAI/finbert** (Ağırlık: 0.4)
   - Finansal alan için özel eğitilmiş BERT modeli
   - Finansal haberlerde yüksek performans

2. **yiyanghkust/finbert-tone** (Ağırlık: 0.3)
   - Finansal ton analizi
   - Pozitif/negatif/nötr ayrımı

3. **distilbert/distilbert-base-uncased-finetuned-sst-2-english** (Ağırlık: 0.2)
   - Genel amaçlı sentiment analizi
   - Hızlı ve hafif model

4. **cardiffnlp/twitter-roberta-base-sentiment-latest** (Ağırlık: 0.1)
   - Sosyal medya/sosyal platform sentiment analizi
   - Kısa metinlerde etkili

### Ensemble Stratejileri

1. **Weighted Average** (Varsayılan)
   - Her modelin signed score'larını ağırlıklı ortalaması
   - Model ağırlıkları ayarlanabilir

2. **Voting**
   - Çoğunluk oylaması
   - Her model bir etiket önerir, en çok oy alan kazanır

## Kurulum

```bash
pip install pandas numpy transformers torch pyarrow
```

## Kullanım

### Temel Kullanım

```bash
python sp500_ensemble_sentiment_pipeline.py \
    --raw_csv SP500_news/raw/sp500_headlines_2008_2024.csv \
    --date_col Date \
    --title_col Title \
    --ensemble_method weighted_average
```

### Tüm Parametreler

```bash
python sp500_ensemble_sentiment_pipeline.py \
    --raw_csv SP500_news/raw/sp500_headlines_2008_2024.csv \
    --date_col Date \
    --title_col Title \
    --ensemble_method weighted_average \
    --batch_size 32 \
    --max_length 128 \
    --device 0 \
    --save_parquet
```

### Belirli Modelleri Kullanma

```bash
python sp500_ensemble_sentiment_pipeline.py \
    --models finbert finbert_tone \
    --ensemble_method weighted_average
```

### Backtest ile Kullanım

Ensemble pipeline çalıştıktan sonra:

```bash
python sp500_sentiment_backtest.py \
    --agg_csv SP500_news/processed/sp500_headlines_daily_agg.csv \
    --price_csv SP500_news/raw/sp500_headlines_2008_2024.csv \
    --tpos 0.10 \
    --tneg 0.10 \
    --nmin 5
```

### Model Karşılaştırma

```bash
python compare_models.py \
    --raw_csv SP500_news/raw/sp500_headlines_2008_2024.csv \
    --price_csv SP500_news/raw/sp500_headlines_2008_2024.csv
```

## Çıktılar

### 1. Headlines with Ensemble (`headlines_with_ensemble.csv`)
- Her başlık için:
  - Her modelin ayrı sonuçları (`{model}_label`, `{model}_score`, `{model}_signed`)
  - Ensemble sonuçları (`ensemble_label`, `ensemble_score`, `ensemble_signed`)
  - Uyumluluk için eski kolonlar (`finbert_label`, vb.)

### 2. Daily Aggregation (`sp500_headlines_daily_agg_ensemble.csv`)
- Günlük özetler:
  - `news_count`: Günlük haber sayısı
  - `pos_count`, `neg_count`, `neu_count`: Etiket sayıları
  - `signed_mean`, `signed_std`, `signed_min`, `signed_max`: İstatistikler
  - Her model için ayrı `{model}_signed_mean` kolonları

### 3. Summary (`ensemble_summary.txt`)
- Kullanılan modeller
- Ensemble yöntemi
- Temel istatistikler

## Model Ağırlıklarını Özelleştirme

`sp500_ensemble_sentiment_pipeline.py` dosyasındaki `DEFAULT_MODELS` dictionary'sini düzenleyerek ağırlıkları değiştirebilirsiniz:

```python
DEFAULT_MODELS = {
    "finbert": {
        "model_id": "ProsusAI/finbert",
        "weight": 0.5,  # Ağırlığı artır
        "description": "Financial domain BERT"
    },
    # ...
}
```

## Performans İpuçları

1. **GPU Kullanımı**: `--device 0` ile GPU kullanın (çok daha hızlı)
2. **Batch Size**: GPU belleğine göre `--batch_size` değerini artırın
3. **Paralel İşleme**: Her model sırayla çalışır, paralelleştirme için kod geliştirilebilir
4. **Caching**: İlk çalıştırmada modeller indirilir, sonraki çalıştırmalarda cache'den yüklenir

## Sorun Giderme

### Model Yükleme Hatası
Eğer bir model yüklenemezse, otomatik olarak atlanır ve diğer modellerle devam edilir. En az bir model başarılı olmalıdır.

### Bellek Hatası
- `--batch_size` değerini düşürün (örn: 16 veya 8)
- `--max_length` değerini düşürün (örn: 64)
- CPU kullanın: `--device -1`

### Uyumluluk
Ensemble pipeline'ın çıktıları, mevcut backtest scripti ile uyumludur. `sp500_headlines_daily_agg.csv` dosyası otomatik olarak oluşturulur.

## Gelecek Geliştirmeler

- [ ] Stacking ensemble (meta-learner)
- [ ] Dinamik ağırlık optimizasyonu
- [ ] Paralel model inference
- [ ] Model performans metrikleri
- [ ] Fine-tuning desteği

## Lisans

Bu proje mevcut proje lisansı altındadır.

