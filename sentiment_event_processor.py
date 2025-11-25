import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SentimentEventProcessor:
    def __init__(self, sentiment_data_path="./Gold_sentiment/processed/processed_gold_sentiment.csv"):
        """
        Event-driven sentiment modeli için veri işleyici
        """
        self.sentiment_data_path = sentiment_data_path
        self.events_path = "./Gold_sentiment/events/"
        self.features_path = "./Gold_sentiment/features/"
        
        # Klasörleri oluştur
        os.makedirs(self.events_path, exist_ok=True)
        os.makedirs(self.features_path, exist_ok=True)
    
    def load_sentiment_data(self):
        """Sentiment verilerini yükle"""
        try:
            df = pd.read_csv(self.sentiment_data_path)
            print(f"Sentiment verisi yüklendi: {df.shape}")
            return df
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return None
    
    def create_events(self, df):
        """Sentiment verilerini event formatına dönüştür"""
        print("Event-driven format oluşturuluyor...")
        
        # Tarih formatını düzelt
        df['Dates'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')
        
        # Event oluştur
        events = []
        
        for idx, row in df.iterrows():
            event = {
                'event_id': f"sentiment_{idx}",
                'timestamp': row['Dates'],
                'news_text': row['News'],
                'url': row['URL'],
                
                # Sentiment bilgileri
                'sentiment_label': row['Price Sentiment'],
                'price_direction_up': row['Price Direction Up'],
                'price_direction_constant': row['Price Direction Constant'],
                'price_direction_down': row['Price Direction Down'],
                
                # Analiz kategorileri
                'has_asset_comparison': row['Asset Comparision'],
                'has_past_info': row['Past Information'],
                'has_future_info': row['Future Information'],
                
                # Hedef değişken (fiyat yönü)
                'target_direction': self._determine_target_direction(row)
            }
            events.append(event)
        
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Event sayısı: {len(events_df)}")
        print(f"Tarih aralığı: {events_df['timestamp'].min()} - {events_df['timestamp'].max()}")
        
        return events_df
    
    def _determine_target_direction(self, row):
        """Fiyat yönünü belirle (0=down, 1=constant, 2=up)"""
        if row['Price Direction Up'] == 1:
            return 2  # Up
        elif row['Price Direction Constant'] == 1:
            return 1  # Constant
        elif row['Price Direction Down'] == 1:
            return 0  # Down
        else:
            return 1  # Default: Constant
    
    def extract_text_features(self, events_df):
        """Haber metinlerinden özellikler çıkar"""
        print("Metin özellikleri çıkarılıyor...")
        
        text_features = []
        
        for idx, row in events_df.iterrows():
            text = str(row['news_text']).lower()
            
            # Temel metin özellikleri
            features = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(re.split(r'[.!?]+', text)),
                
                # Finansal kelimeler
                'gold_mentions': text.count('gold'),
                'price_mentions': text.count('price') + text.count('$'),
                'market_mentions': text.count('market'),
                'investor_mentions': text.count('investor') + text.count('investment'),
                
                # Duygusal kelimeler
                'positive_words': self._count_positive_words(text),
                'negative_words': self._count_negative_words(text),
                'uncertainty_words': self._count_uncertainty_words(text),
                
                # Zaman ifadeleri
                'future_tense': self._count_future_tense(text),
                'past_tense': self._count_past_tense(text),
                
                # Sayısal ifadeler
                'number_count': len(re.findall(r'\d+', text)),
                'percentage_count': text.count('%'),
                'dollar_count': text.count('$'),
            }
            
            text_features.append(features)
        
        return pd.DataFrame(text_features)
    
    def _count_positive_words(self, text):
        """Pozitif kelimeleri say"""
        positive_words = ['up', 'rise', 'gain', 'increase', 'bullish', 'strong', 'growth', 'profit', 'success']
        return sum(text.count(word) for word in positive_words)
    
    def _count_negative_words(self, text):
        """Negatif kelimeleri say"""
        negative_words = ['down', 'fall', 'drop', 'decline', 'bearish', 'weak', 'loss', 'crisis', 'crash']
        return sum(text.count(word) for word in negative_words)
    
    def _count_uncertainty_words(self, text):
        """Belirsizlik kelimelerini say"""
        uncertainty_words = ['maybe', 'might', 'could', 'possibly', 'uncertain', 'volatile', 'unpredictable']
        return sum(text.count(word) for word in uncertainty_words)
    
    def _count_future_tense(self, text):
        """Gelecek zaman ifadelerini say"""
        future_words = ['will', 'going to', 'expected', 'forecast', 'prediction', 'outlook']
        return sum(text.count(word) for word in future_words)
    
    def _count_past_tense(self, text):
        """Geçmiş zaman ifadelerini say"""
        past_words = ['was', 'were', 'had', 'previous', 'historical', 'past']
        return sum(text.count(word) for word in past_words)
    
    def extract_temporal_features(self, events_df):
        """Zaman bazlı özellikler çıkar"""
        print("Zaman özellikleri çıkarılıyor...")
        
        temporal_features = []
        
        for idx, row in events_df.iterrows():
            timestamp = row['timestamp']
            
            features = {
                'year': timestamp.year,
                'month': timestamp.month,
                'day': timestamp.day,
                'day_of_week': timestamp.dayofweek,
                'day_of_year': timestamp.dayofyear,
                'quarter': timestamp.quarter,
                'is_weekend': 1 if timestamp.dayofweek >= 5 else 0,
                'is_month_start': 1 if timestamp.day <= 7 else 0,
                'is_month_end': 1 if timestamp.day >= 25 else 0,
            }
            
            temporal_features.append(features)
        
        return pd.DataFrame(temporal_features)
    
    def create_sentiment_features(self, events_df):
        """Sentiment özelliklerini oluştur"""
        print("Sentiment özellikleri oluşturuluyor...")
        
        # Sentiment label'ları sayısal değerlere dönüştür
        sentiment_mapping = {
            'positive': 1,
            'negative': -1,
            'neutral': 0,
            'none': 0
        }
        
        events_df['sentiment_numeric'] = events_df['sentiment_label'].map(sentiment_mapping)
        
        # Sentiment özellikleri
        sentiment_features = []
        
        for idx, row in events_df.iterrows():
            features = {
                'sentiment_positive': 1 if row['sentiment_label'] == 'positive' else 0,
                'sentiment_negative': 1 if row['sentiment_label'] == 'negative' else 0,
                'sentiment_neutral': 1 if row['sentiment_label'] == 'neutral' else 0,
                'sentiment_none': 1 if row['sentiment_label'] == 'none' else 0,
                'sentiment_numeric': row['sentiment_numeric'],
                
                # Fiyat yönü özellikleri
                'direction_up': row['price_direction_up'],
                'direction_constant': row['price_direction_constant'],
                'direction_down': row['price_direction_down'],
                'target_direction': row['target_direction'],
                
                # Analiz kategorileri
                'has_asset_comparison': row['has_asset_comparison'],
                'has_past_info': row['has_past_info'],
                'has_future_info': row['has_future_info'],
            }
            
            sentiment_features.append(features)
        
        return pd.DataFrame(sentiment_features)
    
    def process_events(self):
        """Tüm event işleme sürecini çalıştır"""
        print("=== EVENT-DRIVEN SENTIMENT PROCESSING ===")
        
        # 1. Veriyi yükle
        df = self.load_sentiment_data()
        if df is None:
            return None
        
        # 2. Event'leri oluştur
        events_df = self.create_events(df)
        
        # 3. Özellikleri çıkar
        text_features = self.extract_text_features(events_df)
        temporal_features = self.extract_temporal_features(events_df)
        sentiment_features = self.create_sentiment_features(events_df)
        
        # 4. Tüm özellikleri birleştir
        all_features = pd.concat([
            events_df[['event_id', 'timestamp', 'news_text']],
            text_features,
            temporal_features,
            sentiment_features
        ], axis=1)
        
        # 5. Event'leri kaydet
        events_output = os.path.join(self.events_path, "sentiment_events.csv")
        all_features.to_csv(events_output, index=False)
        print(f"Event'ler kaydedildi: {events_output}")
        
        # 6. Özellikleri kaydet
        features_output = os.path.join(self.features_path, "sentiment_features.csv")
        all_features.to_csv(features_output, index=False)
        print(f"Özellikler kaydedildi: {features_output}")
        
        print(f"\n[SUCCESS] Event-driven sentiment processing tamamlandi!")
        print(f"Toplam event: {len(all_features)}")
        print(f"Toplam özellik: {len(all_features.columns)}")
        
        return all_features

def main():
    """Ana fonksiyon"""
    processor = SentimentEventProcessor()
    events_df = processor.process_events()
    
    if events_df is not None:
        print("\n=== EVENT ANALİZİ ===")
        print(f"Event sayısı: {len(events_df)}")
        print(f"Tarih aralığı: {events_df['timestamp'].min()} - {events_df['timestamp'].max()}")
        print(f"Sentiment dağılımı:")
        print(events_df['sentiment_label'].value_counts())
        print(f"Hedef dağılımı:")
        print(events_df['target_direction'].value_counts())

if __name__ == "__main__":
    main()
