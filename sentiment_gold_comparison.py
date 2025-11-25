import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SentimentGoldComparison:
    def __init__(self):
        """
        Sentiment analizi ile gerçek altın fiyat hareketlerini karşılaştırma
        """
        self.sentiment_data_path = "./Gold_sentiment/processed/processed_gold_sentiment.csv"
        self.gold_data_path = "./Commodities/rectangle/01/Price/TrainPrice/GC=F.npy"
        self.gold_dates_path = "./Commodities/rectangle/01/Date/TrainDate/GC=F.npy"
        self.results_path = "./Gold_sentiment/results/"
        
        # Klasörleri oluştur
        os.makedirs(self.results_path, exist_ok=True)
    
    def load_sentiment_data(self):
        """Sentiment verilerini yükle"""
        print("Sentiment verisi yükleniyor...")
        df = pd.read_csv(self.sentiment_data_path)
        df['Dates'] = pd.to_datetime(df['Dates'], format='%d-%m-%Y')
        print(f"Sentiment veri boyutu: {df.shape}")
        return df
    
    def load_gold_data(self):
        """Altın fiyat verilerini yükle"""
        print("Altın fiyat verisi yükleniyor...")
        
        # Fiyat verilerini yükle
        gold_prices = np.load(self.gold_data_path)
        
        # Tarih verilerini yükle
        gold_dates = np.load(self.gold_dates_path, allow_pickle=True)
        
        # DataFrame oluştur
        gold_df = pd.DataFrame({
            'Date': gold_dates,
            'Price': gold_prices
        })
        
        # Tarih formatını düzelt
        gold_df['Date'] = pd.to_datetime(gold_df['Date'])
        
        print(f"Altın veri boyutu: {gold_df.shape}")
        print(f"Altın tarih aralığı: {gold_df['Date'].min()} - {gold_df['Date'].max()}")
        
        return gold_df
    
    def analyze_quarterly_sentiment(self, sentiment_df):
        """Çeyrek bazlı sentiment analizi"""
        print("Çeyrek bazlı sentiment analizi yapılıyor...")
        
        # Yıl ve çeyrek bilgilerini ekle
        sentiment_df['Year'] = sentiment_df['Dates'].dt.year
        sentiment_df['Quarter'] = sentiment_df['Dates'].dt.quarter
        
        # Çeyrek bazlı sentiment sayıları
        quarterly_analysis = []
        
        for year in sorted(sentiment_df['Year'].unique()):
            for quarter in [1, 2, 3, 4]:
                quarter_data = sentiment_df[
                    (sentiment_df['Year'] == year) & 
                    (sentiment_df['Quarter'] == quarter)
                ]
                
                if len(quarter_data) > 0:
                    # Sentiment sayıları
                    positive_count = len(quarter_data[quarter_data['Price Sentiment'] == 'positive'])
                    negative_count = len(quarter_data[quarter_data['Price Sentiment'] == 'negative'])
                    neutral_count = len(quarter_data[quarter_data['Price Sentiment'] == 'neutral'])
                    none_count = len(quarter_data[quarter_data['Price Sentiment'] == 'none'])
                    
                    # Toplam none + neutral
                    hold_count = neutral_count + none_count
                    
                    # Karar verme
                    if positive_count > negative_count and positive_count > hold_count:
                        decision = 'BUY'
                        confidence = positive_count / len(quarter_data)
                    elif negative_count > positive_count and negative_count > hold_count:
                        decision = 'SELL'
                        confidence = negative_count / len(quarter_data)
                    else:
                        decision = 'HOLD'
                        confidence = hold_count / len(quarter_data)
                    
                    quarterly_analysis.append({
                        'Year': year,
                        'Quarter': quarter,
                        'Total_News': len(quarter_data),
                        'Positive': positive_count,
                        'Negative': negative_count,
                        'Neutral': neutral_count,
                        'None': none_count,
                        'Hold_Total': hold_count,
                        'Decision': decision,
                        'Confidence': confidence,
                        'Start_Date': quarter_data['Dates'].min(),
                        'End_Date': quarter_data['Dates'].max()
                    })
        
        return pd.DataFrame(quarterly_analysis)
    
    def calculate_gold_price_change(self, gold_df, start_date, end_date):
        """Belirli tarih aralığındaki altın fiyat değişimini hesapla"""
        # Tarih aralığındaki fiyatları bul
        period_data = gold_df[
            (gold_df['Date'] >= start_date) & 
            (gold_df['Date'] <= end_date)
        ]
        
        if len(period_data) < 2:
            return None, None, None
        
        # Başlangıç ve bitiş fiyatları
        start_price = period_data['Price'].iloc[0]
        end_price = period_data['Price'].iloc[-1]
        
        # Fiyat değişimi
        price_change = end_price - start_price
        price_change_percent = (price_change / start_price) * 100
        
        # Gerçek karar (fiyat hareketine göre)
        if price_change_percent > 2:  # %2'den fazla artış
            actual_decision = 'BUY'
        elif price_change_percent < -2:  # %2'den fazla düşüş
            actual_decision = 'SELL'
        else:  # %2'den az değişim
            actual_decision = 'HOLD'
        
        return price_change, price_change_percent, actual_decision
    
    def compare_sentiment_with_reality(self, quarterly_analysis, gold_df):
        """Sentiment kararları ile gerçek fiyat hareketlerini karşılaştır"""
        print("Sentiment kararları ile gerçek fiyat hareketleri karşılaştırılıyor...")
        
        comparison_results = []
        
        for _, row in quarterly_analysis.iterrows():
            # Çeyrek sonundaki tarihi hesapla
            quarter_end = row['End_Date'] + timedelta(days=30)  # Çeyrek sonundan 30 gün sonra
            
            # Altın fiyat değişimini hesapla
            price_change, price_change_percent, actual_decision = self.calculate_gold_price_change(
                gold_df, row['Start_Date'], quarter_end
            )
            
            if price_change is not None:
                # Karşılaştırma
                sentiment_decision = row['Decision']
                is_correct = (sentiment_decision == actual_decision)
                
                comparison_results.append({
                    'Year': row['Year'],
                    'Quarter': row['Quarter'],
                    'Sentiment_Decision': sentiment_decision,
                    'Actual_Decision': actual_decision,
                    'Price_Change_Percent': price_change_percent,
                    'Is_Correct': is_correct,
                    'Confidence': row['Confidence'],
                    'Total_News': row['Total_News'],
                    'Start_Date': row['Start_Date'],
                    'End_Date': quarter_end
                })
        
        return pd.DataFrame(comparison_results)
    
    def calculate_accuracy_metrics(self, comparison_results):
        """Doğruluk metriklerini hesapla"""
        print("Doğruluk metrikleri hesaplanıyor...")
        
        total_predictions = len(comparison_results)
        correct_predictions = comparison_results['Is_Correct'].sum()
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        
        # Karar türüne göre doğruluk
        buy_predictions = comparison_results[comparison_results['Sentiment_Decision'] == 'BUY']
        sell_predictions = comparison_results[comparison_results['Sentiment_Decision'] == 'SELL']
        hold_predictions = comparison_results[comparison_results['Sentiment_Decision'] == 'HOLD']
        
        buy_accuracy = (buy_predictions['Is_Correct'].sum() / len(buy_predictions) * 100) if len(buy_predictions) > 0 else 0
        sell_accuracy = (sell_predictions['Is_Correct'].sum() / len(sell_predictions) * 100) if len(sell_predictions) > 0 else 0
        hold_accuracy = (hold_predictions['Is_Correct'].sum() / len(hold_predictions) * 100) if len(hold_predictions) > 0 else 0
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': accuracy,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'hold_accuracy': hold_accuracy,
            'buy_count': len(buy_predictions),
            'sell_count': len(sell_predictions),
            'hold_count': len(hold_predictions)
        }
    
    def calculate_yearly_accuracy(self, comparison_results):
        """Yıl bazlı doğruluk metriklerini hesapla"""
        print("Yıl bazlı doğruluk metrikleri hesaplanıyor...")
        
        yearly_metrics = []
        
        for year in sorted(comparison_results['Year'].unique()):
            year_data = comparison_results[comparison_results['Year'] == year]
            
            if len(year_data) > 0:
                # Yıl bazlı metrikler
                total_predictions = len(year_data)
                correct_predictions = year_data['Is_Correct'].sum()
                accuracy = (correct_predictions / total_predictions) * 100
                
                # Karar türüne göre yıl bazlı doğruluk
                buy_data = year_data[year_data['Sentiment_Decision'] == 'BUY']
                sell_data = year_data[year_data['Sentiment_Decision'] == 'SELL']
                hold_data = year_data[year_data['Sentiment_Decision'] == 'HOLD']
                
                buy_accuracy = (buy_data['Is_Correct'].sum() / len(buy_data) * 100) if len(buy_data) > 0 else 0
                sell_accuracy = (sell_data['Is_Correct'].sum() / len(sell_data) * 100) if len(sell_data) > 0 else 0
                hold_accuracy = (hold_data['Is_Correct'].sum() / len(hold_data) * 100) if len(hold_data) > 0 else 0
                
                # Ortalama güven skoru
                avg_confidence = year_data['Confidence'].mean()
                
                # Ortalama fiyat değişimi
                avg_price_change = year_data['Price_Change_Percent'].mean()
                
                yearly_metrics.append({
                    'Year': year,
                    'Total_Predictions': total_predictions,
                    'Correct_Predictions': correct_predictions,
                    'Accuracy_Percent': accuracy,
                    'Buy_Count': len(buy_data),
                    'Buy_Accuracy': buy_accuracy,
                    'Sell_Count': len(sell_data),
                    'Sell_Accuracy': sell_accuracy,
                    'Hold_Count': len(hold_data),
                    'Hold_Accuracy': hold_accuracy,
                    'Avg_Confidence': avg_confidence,
                    'Avg_Price_Change_Percent': avg_price_change
                })
        
        return pd.DataFrame(yearly_metrics)
    
    def run_comparison(self):
        """Ana karşılaştırma sürecini çalıştır"""
        print("=== SENTIMENT vs GERÇEK ALTIN FİYAT KARŞILAŞTIRMASI ===")
        
        # 1. Verileri yükle
        sentiment_df = self.load_sentiment_data()
        gold_df = self.load_gold_data()
        
        # 2. Çeyrek bazlı sentiment analizi
        quarterly_analysis = self.analyze_quarterly_sentiment(sentiment_df)
        print(f"Çeyrek analizi tamamlandı: {len(quarterly_analysis)} çeyrek")
        
        # 3. Gerçek fiyat hareketleri ile karşılaştır
        comparison_results = self.compare_sentiment_with_reality(quarterly_analysis, gold_df)
        print(f"Karşılaştırma tamamlandı: {len(comparison_results)} sonuç")
        
        # 4. Doğruluk metriklerini hesapla
        metrics = self.calculate_accuracy_metrics(comparison_results)
        
        # 5. Yıl bazlı doğruluk metriklerini hesapla
        yearly_metrics = self.calculate_yearly_accuracy(comparison_results)
        
        # 6. Sonuçları kaydet
        quarterly_analysis.to_csv(os.path.join(self.results_path, "quarterly_sentiment_analysis.csv"), index=False)
        comparison_results.to_csv(os.path.join(self.results_path, "sentiment_vs_reality_comparison.csv"), index=False)
        yearly_metrics.to_csv(os.path.join(self.results_path, "yearly_accuracy_analysis.csv"), index=False)
        
        # 7. Sonuçları yazdır
        print("\n=== GENEL SONUÇLAR ===")
        print(f"Toplam tahmin sayısı: {metrics['total_predictions']}")
        print(f"Doğru tahmin sayısı: {metrics['correct_predictions']}")
        print(f"Genel doğruluk: {metrics['overall_accuracy']:.2f}%")
        print(f"\nKarar türüne göre doğruluk:")
        print(f"  BUY kararları: {metrics['buy_count']} adet, doğruluk: {metrics['buy_accuracy']:.2f}%")
        print(f"  SELL kararları: {metrics['sell_count']} adet, doğruluk: {metrics['sell_accuracy']:.2f}%")
        print(f"  HOLD kararları: {metrics['hold_count']} adet, doğruluk: {metrics['hold_accuracy']:.2f}%")
        
        return quarterly_analysis, comparison_results, metrics

def main():
    """Ana fonksiyon"""
    comparator = SentimentGoldComparison()
    quarterly_analysis, comparison_results, metrics, yearly_metrics = comparator.run_comparison()
    
    print("\n[SUCCESS] Sentiment vs gerçek altın fiyat karşılaştırması tamamlandı!")
    print("Yıl bazlı analiz sonuçları 'yearly_accuracy_analysis.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()
