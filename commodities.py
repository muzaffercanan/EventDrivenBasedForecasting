import os
import yfinance as yf
import pandas as pd
import numpy as np
import talib as tb
import time
import traceback

'''
Bazı Değişkenlerin Tanımlanması
'''
startDate = '2001-10-11'
endDate = '2025-03-16'
axes = ['Date', 'Value']
headers = ['RSI', 'CMO', 'PLUS_DI', 'MINUS_DI', 'WILLR', 'CCI', 'ULTOSC', 'AROONOSC', 'MFI', 'MOM', 'MACD', 'MACDFIX', 
           'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'ROCP', 'ROC', 'ROCR', 'ROCR100', 'SLOWK', 'FASTD', 'SLOWD', 
           'AROONUP', 'AROONDOWN', 'APO', 'MACDEXT', 'FASTK', 'PPO', 'MINUS_DM', 'ADOSC', 'FASTDRSI', 'FASTKRSI', 
           'TRANGE', 'TRIX', 'STD', 'BOP', 'VAR', 'PLUS_DM', 'CORREL', 'AD', 'BETA', 'WCLPRICE', 'TSF', 'TYPPRICE', 
           'AVGPRICE', 'MEDPRICE', 'BBANDSL', 'LINEARREG', 'OBV', 'BBANDSM', 'TEMA', 'BBANDSU', 'DEMA', 'MIDPRICE', 
           'MIDPOINT', 'WMA', 'EMA', 'HT_TRENDLINE', 'KAMA', 'SMA', 'MA', 'ADXR', 'ADX', 'TRIMA', 'LINEARREG_INTERCEPT', 'DX', 'SAR']

commoditiesList = [
    'GC=F', 'SI=F', 'PL=F', 'PA=F', 'CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F', 'SB=F', 'CT=F', 
    'KC=F', 'CC=F', 'OJ=F', 'LE=F', 'HE=F'
]

threshold = 0.01
pd.set_option('display.max_rows', None)

'''
Gerekli Klasörlerin Oluşturulması
'''
folders = [
    "./Commodities/rectangle/01/TrainData",
    "./Commodities/rectangle/01/TestData",
    "./Commodities/rectangle/01/Date/TrainDate",
    "./Commodities/rectangle/01/Date/TestDate",
    "./Commodities/rectangle/01/Price/TrainPrice",
    "./Commodities/rectangle/01/Price/TestPrice"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

'''
Veri İndirme Fonksiyonu
'''
def download_data_with_retry(ticker, start, end, retries=5, initial_delay=120):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end)
            if data is None or data.empty:
                if attempt == retries - 1:
                    print(f"{ticker} için veri bulunamadı, atlanıyor (tüm denemeler başarısız).")
                    return None
                delay = initial_delay * (2 ** attempt)
                print(f"{ticker} için veri alınamadı. {delay} saniye sonra tekrar denenecek... (Deneme {attempt + 1}/{retries})")
                time.sleep(delay)
                continue
            return data
        except Exception as e:
            if "YFPricesMissingError" in str(e):
                print(f"{ticker} için veri yok (muhtemelen listelenmemiş): {str(e)}. Atlanıyor.")
                return None
            if "Too Many Requests" in str(e) or "Rate limit" in str(e):
                if attempt < retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"{ticker} için hız sınırı aşıldı. {delay} saniye sonra tekrar denenecek... (Deneme {attempt + 1}/{retries})")
                    time.sleep(delay)
                else:
                    raise
            elif attempt < retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"{ticker} indirilirken hata: {str(e)}. {delay} saniye sonra tekrar denenecek... (Deneme {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise

'''
Verilerin İndirilmesi ve İşlenmesi
'''
for i, commodity in enumerate(commoditiesList):
    print(f"\n{commodity} işleniyor ({i+1}/{len(commoditiesList)})...")
    
    try:
        imageList = []
        labelList = []

        data = download_data_with_retry(commodity, startDate, endDate)
        if data is None or data.empty:
            print(f"{commodity} için veri yok, atlanıyor.")
            continue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        print(f"{commodity} veri şekli: {data.shape}")
        print(f"Veri sütunları: {data.columns.tolist()}")
        print(f"İlk birkaç satır:\n{data.head()}")

        nan_counts = {
            "Close": data["Close"].isna().sum(),
            "High": data["High"].isna().sum(),
            "Low": data["Low"].isna().sum(),
            "Open": data["Open"].isna().sum(),
            "Volume": data["Volume"].isna().sum()
        }
        print(f"NaN sayıları (doldurulmadan önce): {nan_counts}")

        close_values = np.ascontiguousarray(data["Close"].ffill().bfill().values.flatten(), dtype=np.float64)
        high_values = np.ascontiguousarray(data["High"].ffill().bfill().values.flatten(), dtype=np.float64)
        low_values = np.ascontiguousarray(data["Low"].ffill().bfill().values.flatten(), dtype=np.float64)
        open_values = np.ascontiguousarray(data["Open"].ffill().bfill().values.flatten(), dtype=np.float64)
        volume_values = np.ascontiguousarray(data["Volume"].ffill().bfill().values.flatten(), dtype=np.float64)

        print(f"Tüm veri dizileri 1D'ye düzleştirildi ve float64 türüne dönüştürüldü")

        # İndikatör Hesaplamaları (tam liste)
        wma = pd.Series(tb.WMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ema = pd.Series(tb.EMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        sma = pd.Series(tb.SMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        tema = pd.Series(tb.TEMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        sar = pd.Series(tb.SAR(high_values, low_values, acceleration=0.02, maximum=0.2), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        try:
            bbands_upperband, bbands_middleband, bbands_lowerband = tb.BBANDS(close_values, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        except:
            sma_temp = pd.Series(close_values).rolling(window=5).mean()
            std_temp = pd.Series(close_values).rolling(window=5).std()
            bbands_upperband = (sma_temp + 2 * std_temp).values
            bbands_middleband = sma_temp.values
            bbands_lowerband = (sma_temp - 2 * std_temp).values
        bbands_upperband = pd.Series(bbands_upperband, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        bbands_middleband = pd.Series(bbands_middleband, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        bbands_lowerband = pd.Series(bbands_lowerband, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        dema = pd.Series(tb.DEMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ht_trendline = pd.Series(tb.HT_TRENDLINE(close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        kama = pd.Series(tb.KAMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ma = pd.Series(tb.MA(close_values, timeperiod=30, matype=0), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        midpoint = pd.Series(tb.MIDPOINT(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        midprice = pd.Series(tb.MIDPRICE(high_values, low_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        trima = pd.Series(tb.TRIMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        rsi = pd.Series(tb.RSI(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        roc = pd.Series(tb.ROC(close_values, timeperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        cmo = pd.Series(tb.CMO(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        cci = pd.Series(tb.CCI(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ppo = pd.Series(tb.PPO(close_values, fastperiod=12, slowperiod=26, matype=0), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        willr = pd.Series(tb.WILLR(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        macd, _, _ = tb.MACD(close_values, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = pd.Series(macd, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        adx = pd.Series(tb.ADX(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        adxr = pd.Series(tb.ADXR(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        apo = pd.Series(tb.APO(close_values, fastperiod=12, slowperiod=26, matype=0), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        aroondown, aroonup = tb.AROON(high_values, low_values, timeperiod=14)
        aroondown = pd.Series(aroondown, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        aroonup = pd.Series(aroonup, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        aroonosc = pd.Series(tb.AROONOSC(high_values, low_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        bop = pd.Series(tb.BOP(open_values, high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        dx = pd.Series(tb.DX(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        macdext, _, _ = tb.MACDEXT(close_values, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        macdext = pd.Series(macdext, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        macdfix, _, _ = tb.MACDFIX(close_values, signalperiod=9)
        macdfix = pd.Series(macdfix, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        mfi = pd.Series(tb.MFI(high_values, low_values, close_values, volume_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        minus_di = pd.Series(tb.MINUS_DI(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        minus_dm = pd.Series(tb.MINUS_DM(high_values, low_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        mom = pd.Series(tb.MOM(close_values, timeperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        plus_di = pd.Series(tb.PLUS_DI(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        plus_dm = pd.Series(tb.PLUS_DM(high_values, low_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        rocp = pd.Series(tb.ROCP(close_values, timeperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        rocr = pd.Series(tb.ROCR(close_values, timeperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        rocr100 = pd.Series(tb.ROCR100(close_values, timeperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        slowk, slowd = tb.STOCH(high_values, low_values, close_values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        slowk = pd.Series(slowk, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        slowd = pd.Series(slowd, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        fastk, fastd = tb.STOCHF(high_values, low_values, close_values, fastk_period=5, fastd_period=3, fastd_matype=0)
        fastk = pd.Series(fastk, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        fastd = pd.Series(fastd, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        fastkrsi, fastdrsi = tb.STOCHRSI(close_values, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        fastkrsi = pd.Series(fastkrsi, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        fastdrsi = pd.Series(fastdrsi, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        trix = pd.Series(tb.TRIX(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ultosc = pd.Series(tb.ULTOSC(high_values, low_values, close_values, timeperiod1=7, timeperiod2=14, timeperiod3=28), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        obv = pd.Series(tb.OBV(close_values, volume_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ad = pd.Series(tb.AD(high_values, low_values, close_values, volume_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        adosc = pd.Series(tb.ADOSC(high_values, low_values, close_values, volume_values, fastperiod=3, slowperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        trange = pd.Series(tb.TRANGE(high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        avgprice = pd.Series(tb.AVGPRICE(open_values, high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        medprice = pd.Series(tb.MEDPRICE(high_values, low_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        typprice = pd.Series(tb.TYPPRICE(high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        wclprice = pd.Series(tb.WCLPRICE(high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        std = pd.Series(tb.STDDEV(close_values, timeperiod=5, nbdev=1), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        beta = pd.Series(tb.BETA(high_values, low_values, timeperiod=5), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        correl = pd.Series(tb.CORREL(high_values, low_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg = pd.Series(tb.LINEARREG(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg_angle = pd.Series(tb.LINEARREG_ANGLE(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg_intercept = pd.Series(tb.LINEARREG_INTERCEPT(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg_slope = pd.Series(tb.LINEARREG_SLOPE(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        tsf = pd.Series(tb.TSF(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        var = pd.Series(tb.VAR(close_values, timeperiod=5, nbdev=1), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)

        indicators = [rsi, cmo, plus_di, minus_di, willr, cci, ultosc, aroonosc, mfi, mom, macd, macdfix, rocp, roc, rocr, rocr100, 
                      slowk, fastd, slowd, aroonup, aroondown, fastk, apo, macdext, ppo, minus_dm, fastdrsi, fastkrsi, trix, bop, 
                      plus_dm, adxr, adx, dx, wma, ema, ht_trendline, midpoint, sma, ma, midprice, trima, dema, kama, 
                      bbands_upperband, bbands_middleband, tema, bbands_lowerband, std, var, linearreg_slope, linearreg_angle, 
                      linearreg_intercept, linearreg, tsf, beta, correl, avgprice, medprice, typprice, wclprice, obv, ad, adosc, trange, sar]

        nIndicators = len(indicators)
        maxNullVal = max([indicator['Value'].isnull().sum() for indicator in indicators])
        print(f"Max Null Değeri: {maxNullVal}")
        print(f"İndikatör Sayısı: {nIndicators}")
        print(f"Ham veri gün sayısı: {len(data)}")
        print(f"Beklenen görüntü sayısı: {len(data) - maxNullVal - nIndicators}")

        for i, indicator in enumerate(indicators):
            null_count = indicator['Value'].isnull().sum()
            if null_count > 0:
                print(f"{headers[i]}: {null_count} NaN")

        indicatorValues = [indicator['Value'].iloc[maxNullVal:] for indicator in indicators]
        indicatorValuesMatrix = pd.concat(indicatorValues, axis=1, keys=headers)

        nDays = len(indicatorValues[0])
        print(f"nDays: {nDays}")
        for idx in range(nDays - nIndicators):
            image = [indicatorValue.iloc[idx:idx + nIndicators].values for indicatorValue in indicatorValues]
            image_array = np.array(image)
            if np.isnan(image_array).any():
                print(f"Uyarı: {idx}. indekste NaN değerler bulundu. 0 ile dolduruluyor.")
                image_array = np.nan_to_num(image_array, nan=0.0)
            imageList.append(image_array)

        print(f"Görüntü oluşturma tamamlandı, imageList uzunluğu: {len(imageList)}")

        data_close = data[maxNullVal + nIndicators - 1:]["Close"]
        close_prices = data_close.values
        print(f"data_close uzunluğu: {len(data_close)}")

        for i in range(len(close_prices) - 1):
            closePriceDifference = close_prices[i + 1] - close_prices[i]
            thresholdPrice = threshold * close_prices[i]
            if closePriceDifference > 0:
                if closePriceDifference <= thresholdPrice:
                    labelList.append(np.array([1.0]))  # HOLD
                else:
                    labelList.append(np.array([0.0]))  # BUY
            elif closePriceDifference < 0:
                if abs(closePriceDifference) <= thresholdPrice:
                    labelList.append(np.array([1.0]))  # HOLD
                else:
                    labelList.append(np.array([2.0]))  # SELL
            else:
                labelList.append(np.array([1.0]))  # HOLD

        print(f"Etiket oluşturma tamamlandı, labelList uzunluğu: {len(labelList)}")

        print(f"Görüntü sayısı: {len(imageList)}")
        print(f"Etiket sayısı: {len(labelList)}")
        print(f"Kapanış verisi sayısı: {len(data_close[:-1])}")

        standartized_image_list = []
        for img in imageList:
            m = np.mean(img, axis=1, keepdims=True)
            s = np.std(img, axis=1, keepdims=True)
            s[s == 0] = 1
            standartized_image = np.expand_dims((img - m) / s, axis=-1)
            standartized_image_list.append(standartized_image)

        print(f"Standartizasyon tamamlandı, standartized_image_list uzunluğu: {len(standartized_image_list)}")

        split_index = int(len(standartized_image_list) * 0.8)
        print(f"split_index: {split_index}")
        
        x_train, x_test = standartized_image_list[:split_index], standartized_image_list[split_index:]
        y_train, y_test = labelList[:split_index], labelList[split_index:]
        train_date, test_date = data_close.index[:split_index], data_close.index[split_index:]
        train_price, test_price = data_close.iloc[:split_index], data_close.iloc[split_index:]

        print(f"Eğitim ve test kümeleri ayrıldı")

        np.save(f"./Commodities/rectangle/01/TrainData/x_{commodity}.npy", x_train)
        np.save(f"./Commodities/rectangle/01/TrainData/y_{commodity}.npy", y_train)
        np.save(f"./Commodities/rectangle/01/TestData/x_{commodity}.npy", x_test)
        np.save(f"./Commodities/rectangle/01/TestData/y_{commodity}.npy", y_test)
        np.save(f"./Commodities/rectangle/01/Date/TrainDate/{commodity}.npy", train_date)
        np.save(f"./Commodities/rectangle/01/Date/TestDate/{commodity}.npy", test_date)
        np.save(f"./Commodities/rectangle/01/Price/TrainPrice/{commodity}.npy", train_price)
        np.save(f"./Commodities/rectangle/01/Price/TestPrice/{commodity}.npy", test_price)

        print(f"{commodity} için veri kaydedildi")

    except Exception as e:
        print(f"{commodity} işlenirken hata oluştu: {str(e)}")
        print(f"Hata türü: {type(e).__name__}")
        print(f"Hata detayları: {e.args}")
        traceback.print_exc()
        continue




        