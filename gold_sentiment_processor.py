import pandas as pd
import numpy as np
import os

def load_sentiment_data():
    """Gold sentiment verilerini yükle"""
    
    # Gold_sentiment klasöründeki CSV dosyasını yükle
    sentiment_path = "./Gold_sentiment/"
    csv_file = "gold-dataset-sinha-khandait.csv"
    file_path = os.path.join(sentiment_path, csv_file)
    
    if not os.path.exists(file_path):
        print(f"CSV dosyası bulunamadı: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"Veri yüklendi: {csv_file}")
        print(f"Veri boyutu: {df.shape}")
        print(f"Sütunlar: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Veri yükleme hatası: {e}")
        return None

def analyze_data(df):
    """Veriyi analiz et"""
    print("\n=== GOLD SENTIMENT VERİ ANALİZİ ===")
    print(f"Toplam satır: {len(df)}")
    print(f"Toplam sütun: {len(df.columns)}")
    print(f"Eksik değerler:\n{df.isnull().sum()}")
    print(f"\nİlk 5 satır:\n{df.head()}")
    
    # Kategorik değişkenlerin dağılımı
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"\n{col} değer dağılımı:")
            print(df[col].value_counts().head())
    
    # Sayısal sütunların istatistikleri
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nSayısal sütunların istatistikleri:")
        print(df[numeric_cols].describe())

def preprocess_data(df):
    """Veriyi ön işleme"""
    print("\n=== VERİ ÖN İŞLEME ===")
    
    # Eksik değerleri kontrol et
    missing_before = df.isnull().sum().sum()
    print(f"Ön işleme öncesi eksik değer sayısı: {missing_before}")
    
    # Eksik değerleri temizle
    df_cleaned = df.dropna()
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"Ön işleme sonrası eksik değer sayısı: {missing_after}")
    print(f"Temizlenen satır sayısı: {len(df) - len(df_cleaned)}")
    
    # İşlenmiş veriyi kaydet
    processed_path = "./Gold_sentiment/processed/"
    os.makedirs(processed_path, exist_ok=True)
    
    output_file = os.path.join(processed_path, "processed_gold_sentiment.csv")
    df_cleaned.to_csv(output_file, index=False)
    print(f"İşlenmiş veri kaydedildi: {output_file}")
    
    return df_cleaned

def main():
    """Ana fonksiyon"""
    print("Gold Sentiment veri işleyici başlatılıyor...")
    
    # Veriyi yükle
    df = load_sentiment_data()
    
    if df is not None:
        # Veriyi analiz et
        analyze_data(df)
        
        # Veriyi ön işle
        processed_df = preprocess_data(df)
        
        print("\n[SUCCESS] Gold Sentiment veri isleme tamamlandi!")
        print(f"Islenmis veri boyutu: {processed_df.shape}")
    else:
        print("[ERROR] Veri yuklenemedi!")

if __name__ == "__main__":
    main()
