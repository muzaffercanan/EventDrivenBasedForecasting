import yfinance as yf
import pandas as pd
import numpy as np
import talib as tb
import time

'''
DEFINING SOME VARIABLES
'''
startDate = '2001-10-11'
endDate = '2022-04-15'
axes = ['Date', 'Value']
headers = ['RSI', 'CMO', 'PLUS_DI', 'MINUS_DI', 'WILLR', 'CCI', 'ULTOSC', 'AROONOSC', 'MFI', 'MOM', 'MACD', 'MACDFIX', 'LINEARREG_ANGLE', 'LINEARREG_SLOPE', 'ROCP', 'ROC', 'ROCR', 'ROCR100', 'SLOWK',
           'FASTD', 'SLOWD', 'AROONUP', 'AROONDOWN', 'APO', 'MACDEXT', 'FASTK', 'PPO', 'MINUS_DM', 'ADOSC', 'FASTDRSI', 'FASTKRSI', 'TRANGE', 'TRIX', 'STD', 'BOP', 'VAR', 'PLUS_DM', 'CORREL', 'AD',
           'BETA', 'WCLPRICE', 'TSF', 'TYPPRICE', 'AVGPRICE', 'MEDPRICE', 'BBANDSL', 'LINEARREG', 'OBV', 'BBANDSM', 'TEMA', 'BBANDSU', 'DEMA', 'MIDPRICE', 'MIDPOINT', 'WMA', 'EMA',
           'HT_TRENDLINE', 'KAMA', 'SMA', 'MA', 'ADXR', 'ADX', 'TRIMA', 'LINEARREG_INTERCEPT', 'DX']


etfList = ['XLF', 'XLU', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE'] 
#threshold = 0.0038  # Re-arrange the Threshold Value
threshold = 0.01

pd.set_option('display.max_rows', None)

def download_data_with_retry(ticker, start, end, retries=5, initial_delay=120):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start, end=end)
            if data is None or data.empty:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"No data received for {ticker}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                continue
            return data
        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate limit" in str(e):
                if attempt < retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limit hit for {ticker}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(delay)
                else:
                    raise
            elif attempt < retries - 1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                print(f"Error downloading {ticker}: {str(e)}. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise

'''
DOWNLOADING THE DATA
'''
# DataFrame, size=(n_days, 6), col_names=["Open", "High", "Low", "Close", "Adj Close", "Volume"]
for i, etf in enumerate(etfList):
    print(f"\nProcessing {etf} ({i+1}/{len(etfList)})...")
    
    try:
        if i > 0:  # Add delay between processing different ETFs
            delay = 5  # 5 minutes delay between ETFs
            print(f"Waiting {delay} seconds before processing next ETF...")
            time.sleep(delay)

        imageList = []
        labelList = []

        data = download_data_with_retry(etf, startDate, endDate)
        if data is None or data.empty:
            print(f"Skipping {etf} due to no data available")
            continue

        print(f"Data shape for {etf}: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"First few rows of data:\n{data.head()}")

        # Ensure all data arrays are 1D, contiguous, and float64 type
        close_values = data["Close"].values
        print(f"Close values shape: {close_values.shape}, dtype: {close_values.dtype}")
        if close_values.ndim != 1:
            print(f"Close values has {close_values.ndim} dimensions, flattening...")
            close_values = close_values.flatten()
            print(f"After flattening, shape: {close_values.shape}")
        
        # Check for NaN values
        nan_count = np.isnan(close_values).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values found in close_values. Filling with forward fill method.")
            # Convert to pandas Series to use fillna methods
            close_series = pd.Series(close_values)
            close_series = close_series.fillna(method='ffill').fillna(method='bfill')
            close_values = close_series.values
            
        # Ensure the array is contiguous and of the right type
        close_values = np.ascontiguousarray(close_values, dtype=np.float64)
        print(f"After contiguous conversion: dtype={close_values.dtype}, contiguous={close_values.flags['C_CONTIGUOUS']}")
            
        high_values = data["High"].values
        print(f"High values dtype before conversion: {high_values.dtype}")
        if high_values.ndim != 1:
            high_values = high_values.flatten()
        # Check for NaN values
        if np.isnan(high_values).any():
            high_series = pd.Series(high_values)
            high_values = high_series.fillna(method='ffill').fillna(method='bfill').values
        high_values = np.ascontiguousarray(high_values, dtype=np.float64)
        
        low_values = data["Low"].values
        if low_values.ndim != 1:
            low_values = low_values.flatten()
        # Check for NaN values
        if np.isnan(low_values).any():
            low_series = pd.Series(low_values)
            low_values = low_series.fillna(method='ffill').fillna(method='bfill').values
        low_values = np.ascontiguousarray(low_values, dtype=np.float64)
        
        open_values = data["Open"].values
        if open_values.ndim != 1:
            open_values = open_values.flatten()
        # Check for NaN values
        if np.isnan(open_values).any():
            open_series = pd.Series(open_values)
            open_values = open_series.fillna(method='ffill').fillna(method='bfill').values
        open_values = np.ascontiguousarray(open_values, dtype=np.float64)
        
        volume_values = data["Volume"].values
        if volume_values.ndim != 1:
            volume_values = volume_values.flatten()
        # Check for NaN values
        if np.isnan(volume_values).any():
            volume_series = pd.Series(volume_values)
            volume_values = volume_series.fillna(method='ffill').fillna(method='bfill').values
        volume_values = np.ascontiguousarray(volume_values, dtype=np.float64)
            
        print(f"All data arrays flattened to 1D and converted to float64 type")

        '''
        CALCULATING THE INDICATOR VALUES
        '''

        """ Overlap Studies """
        try:
            print(f"Calculating WMA for {etf}...")
            wma = pd.Series(tb.WMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        except Exception as e:
            print(f"Error calculating WMA: {str(e)}")
            raise
            
        try:
            print(f"Calculating EMA for {etf}...")
            ema = pd.Series(tb.EMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        except Exception as e:
            print(f"Error calculating EMA: {str(e)}")
            raise
            
        try:
            print(f"Calculating SMA for {etf}...")
            sma = pd.Series(tb.SMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        except Exception as e:
            print(f"Error calculating SMA: {str(e)}")
            raise
            
        try:
            print(f"Calculating TEMA for {etf}...")
            tema = pd.Series(tb.TEMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        except Exception as e:
            print(f"Error calculating TEMA: {str(e)}")
            raise
            
        try:
            print(f"Calculating SAR for {etf}...")
            # Default values for SAR are typically: acceleration=0.02, maximum=0.2
            print(f"High values shape: {high_values.shape}")
            print(f"Low values shape: {low_values.shape}")
            sar = pd.Series(tb.SAR(high_values, low_values, acceleration=0.02, maximum=0.2), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        except Exception as e:
            print(f"Error calculating SAR: {str(e)}")
            raise
            
        try:
            print(f"Calculating BBANDS for {etf}...")
            print(f"BBANDS input shape: {close_values.shape}, dtype: {close_values.dtype}")
            print(f"First few values: {close_values[:5]}")
            
            # Try with explicit casting to double
            try:
                bbands_upperband, bbands_middleband, bbands_lowerband = tb.BBANDS(close_values, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            except Exception as e:
                print(f"First BBANDS attempt failed: {str(e)}, trying with manual calculation...")
                # Manual calculation as fallback
                sma = pd.Series(close_values).rolling(window=5).mean()
                std = pd.Series(close_values).rolling(window=5).std()
                bbands_upperband = (sma + 2 * std).values
                bbands_middleband = sma.values
                bbands_lowerband = (sma - 2 * std).values
                print("Manual calculation completed")
                
            bbands_upperband = pd.Series(bbands_upperband, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
            bbands_middleband = pd.Series(bbands_middleband, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
            bbands_lowerband = pd.Series(bbands_lowerband, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        except Exception as e:
            print(f"Error calculating BBANDS: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise
        dema = pd.Series(tb.DEMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ht_trendline = pd.Series(tb.HT_TRENDLINE(close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        kama = pd.Series(tb.KAMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ma = pd.Series(tb.MA(close_values, timeperiod=30, matype=0), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        midpoint = pd.Series(tb.MIDPOINT(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        midprice = pd.Series(tb.MIDPRICE(high_values, low_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        trima = pd.Series(tb.TRIMA(close_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)

        """ Momentum Indicators """
        rsi = pd.Series(tb.RSI(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        roc = pd.Series(tb.ROC(close_values, timeperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        cmo = pd.Series(tb.CMO(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        cci = pd.Series(tb.CCI(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ppo = pd.Series(tb.PPO(close_values, fastperiod=12, slowperiod=26, matype=0), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        willr = pd.Series(tb.WILLR(high_values, low_values, close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        macd, macdsignal, macdhist = tb.MACD(close_values, fastperiod=12, slowperiod=26, signalperiod=9)
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
        macdext, macdextsignal, macdexthist = tb.MACDEXT(close_values, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        macdext = pd.Series(macdext, index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        macdfix, macdfixsignal, macdfixhist = tb.MACDFIX(close_values, signalperiod=9)
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

        """ Volume Indicators """
        obv = pd.Series(tb.OBV(close_values, volume_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        ad = pd.Series(tb.AD(high_values, low_values, close_values, volume_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        adosc = pd.Series(tb.ADOSC(high_values, low_values, close_values, volume_values, fastperiod=3, slowperiod=10), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)

        """ Volatility Indicators """
        trange = pd.Series(tb.TRANGE(high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)

        """ Price Transform Indicators """
        avgprice = pd.Series(tb.AVGPRICE(open_values, high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        medprice = pd.Series(tb.MEDPRICE(high_values, low_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        typprice = pd.Series(tb.TYPPRICE(high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        wclprice = pd.Series(tb.WCLPRICE(high_values, low_values, close_values), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)

        """ Cycle Indicators """

        """ Pattern Recognition """

        """ Statistical Functions """
        std = pd.Series(tb.STDDEV(close_values, timeperiod=5, nbdev=1), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        beta = pd.Series(tb.BETA(high_values, low_values, timeperiod=5), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        correl = pd.Series(tb.CORREL(high_values, low_values, timeperiod=30), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg = pd.Series(tb.LINEARREG(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg_angle = pd.Series(tb.LINEARREG_ANGLE(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg_intercept = pd.Series(tb.LINEARREG_INTERCEPT(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        linearreg_slope = pd.Series(tb.LINEARREG_SLOPE(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        tsf = pd.Series(tb.TSF(close_values, timeperiod=14), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)
        var = pd.Series(tb.VAR(close_values, timeperiod=5, nbdev=1), index=data.index).to_frame().reset_index().set_axis(axes, axis=1)

        '''
        PREPROCESSING INDICATOR DATA
        '''
        # List of (indicators) DataFrames, size=n_indicators
        indicators = [rsi, cmo, plus_di, minus_di, willr, cci, ultosc, aroonosc, mfi, mom, macd, macdfix, rocp, roc, rocr, rocr100, slowk, fastd, slowd, aroonup,
                      aroondown, fastk, apo, macdext, ppo, minus_dm, fastdrsi, fastkrsi, trix, bop, plus_dm, adxr, adx, dx, wma, ema, ht_trendline, midpoint, sma, ma, midprice,
                      trima, dema, kama, bbands_upperband, bbands_middleband, tema, bbands_lowerband, std, var, linearreg_slope, linearreg_angle, linearreg_intercept, linearreg,
                      tsf, beta, correl, avgprice, medprice, typprice, wclprice, obv, ad, adosc, trange]

        # Number of indicators (int)
        nIndicators = len(indicators)

        # Calculating the most number of null values in an indicator DataFrame's "Value" column
        maxNullVal = -1
        for indicator in indicators:
            if(indicator['Value'].isnull().sum() > maxNullVal):
                maxNullVal = indicator['Value'].isnull().sum()

        # List of (indicators "Value" column) DataFrames, size=n_indicators
        indicatorValues = []
        for indicator in indicators:
            # Check if the indicator's Value column is a Series or DataFrame
            if isinstance(indicator['Value'], pd.Series) or isinstance(indicator['Value'], pd.DataFrame):
                # Handle MultiIndex if present
                if hasattr(indicator['Value'], 'columns') and indicator['Value'].columns.nlevels > 1:
                    # If it's a DataFrame with MultiIndex columns, take the first column
                    value_series = indicator['Value'].iloc[:, 0]
                elif hasattr(indicator['Value'], 'index') and indicator['Value'].index.nlevels > 1:
                    # If it's a Series with MultiIndex, convert to numpy array
                    value_series = pd.Series(indicator['Value'].values)
                else:
                    value_series = indicator['Value']
                
                # Getting rid of null values
                indicatorValues.append(value_series.iloc[maxNullVal:])
            else:
                # If it's already a numpy array or other type
                print(f"Warning: indicator['Value'] is not a Series or DataFrame, it's a {type(indicator['Value'])}")
                # Convert to Series and append
                value_series = pd.Series(indicator['Value'])[maxNullVal:]
                indicatorValues.append(value_series)

        # DataFrame, size=(n_days, n_indicators, col_names=headers)
        indicatorValuesMatrix = pd.concat(indicatorValues, axis=1, keys=headers)
        indicatorCorr = indicatorValuesMatrix.corr(method='pearson')

        '''
        dictCor = {}
        for header, value in zip(headers, indicatorCorr.iloc[0]):
            dictCor[header] = value
        sortedDictCor = {k: v for k, v in sorted(dictCor.items(), key=lambda item: abs(item[1]), reverse=True)}
        for k,v in sortedDictCor.items():
            print(k, v)

        '''

        '''
        CREATING THE IMAGES
        '''
        nDays = len(indicatorValues[0])
        for idx in range(nDays-nIndicators):
            # List, size=n_indicators, contains imageRows of size (n_indicators, 1)
            image = []
            for indicatorValue in indicatorValues:
                # NumPy Array, size=(n_indicators, 1)
                # Handle different types of indicatorValue
                if hasattr(indicatorValue, 'iloc'):
                    # If it's a pandas Series or DataFrame
                    imageRow = indicatorValue.iloc[idx:idx+nIndicators].values
                else:
                    # If it's already a numpy array
                    imageRow = indicatorValue[idx:idx+nIndicators]
                
                # Ensure it's a numpy array
                if not isinstance(imageRow, np.ndarray):
                    imageRow = np.array(imageRow)
                
                image.append(imageRow)
            
            # Convert the entire image to a numpy array
            image_array = np.array(image)
            
            # Check for NaN values in the image
            if np.isnan(image_array).any():
                print(f"Warning: NaN values found in image at index {idx}. Filling with 0.")
                image_array = np.nan_to_num(image_array, nan=0.0)
            
            imageList.append(image_array)

        '''
        CREATING THE LABELS
        '''
        # Pandas Series, size=n_days-(maxNullVal+nIndicators-1) -> Check this, size is imageList+1, might be a bug.
        data_close = data[maxNullVal+nIndicators-1:]["Close"]
        
        # Handle MultiIndex structure if present
        if isinstance(data_close, pd.DataFrame) or (isinstance(data_close, pd.Series) and data_close.index.nlevels > 1):
            print(f"Data close type: {type(data_close)}, shape: {data_close.shape}")
            print(f"Data close index levels: {data_close.index.nlevels}")
            print(f"Data close columns: {data_close.columns if hasattr(data_close, 'columns') else 'N/A'}")
            
            # Extract the actual close values - if it's a DataFrame with MultiIndex columns
            if isinstance(data_close, pd.DataFrame) and data_close.columns.nlevels > 1:
                data_close = data_close.iloc[:, 0]  # Take the first column
                print(f"After extraction, data_close type: {type(data_close)}")
            # If it's a Series with MultiIndex
            elif isinstance(data_close, pd.Series) and data_close.index.nlevels > 1:
                data_close = data_close.values  # Convert to numpy array
                print(f"After extraction, data_close type: {type(data_close)}")

        # Buy : 0
        # Hold: 1
        # Sell: 2
        for i in range(len(data_close)-1):
            # Extract scalar values to avoid Series comparison
            current_close = data_close.iloc[i] if hasattr(data_close, 'iloc') else data_close[i]
            next_close = data_close.iloc[i+1] if hasattr(data_close, 'iloc') else data_close[i+1]
            
            # Ensure we have scalar values
            if hasattr(current_close, 'item'):
                current_close = current_close.item()
            if hasattr(next_close, 'item'):
                next_close = next_close.item()
                
            closePriceDifference = next_close - current_close
            thresholdPrice = threshold * current_close
            
            # If the price has increased
            if(closePriceDifference > 0):
                # but not enough to pass the threshold
                if(closePriceDifference <= thresholdPrice):
                    labelList.append(np.array([1.0]))  # HOLD
                # enough to pass the threshold
                else:
                    labelList.append(np.array([0.0]))  # BUY
            # If the price has decreased
            elif(closePriceDifference < 0):
                # but not so much to pass the thresshold
                if(abs(closePriceDifference) <= thresholdPrice):
                    labelList.append(np.array([1.0]))  # HOLD
                # so much to pass the threshold
                else:
                    labelList.append(np.array([2.0]))  # SELL
            # If the price hasn't changed
            else:
                labelList.append(np.array([1.0]))  # HOLD

        print(len(imageList))
        print(len(labelList))
        print(len(data_close[:-1]))

        # imageList = np.array(imageList)
        # labelList = np.array(labelList)

        # unique, counts = np.unique(labelList, return_counts=True)
        # print(np.asarray((unique, counts)).T)

        # imageList_copy = imageList[:]
        # imageList_copy = imageList_copy.reshape(len(imageList), -1)
        # # df_before = pd.DataFrame(imageList_copy, columns=np.repeat(
        # #     np.array(headers), nIndicators))
        # # df_before.to_csv("df_before.csv", encoding='utf-8', index=False)
        # mean = np.mean(imageList_copy, axis=0)
        # # mean_df = pd.DataFrame(mean)
        # # mean_df.to_csv("mean.csv", encoding='utf-8', index=False)
        # std = np.std(imageList_copy, axis=0)
        # # std_df = pd.DataFrame(std)
        # # std_df.to_csv("std.csv", encoding='utf-8', index=False)
        # imageList_copy = (imageList_copy - mean) / std
        # # df_after = pd.DataFrame(imageList_copy, columns=np.repeat(
        # #     np.array(headers), nIndicators))
        # # df_after.to_csv("df_after.csv", encoding='utf-8', index=False)
        # imageList = imageList_copy.reshape(
        #     len(imageList), len(indicators), len(indicators), 1)
        
        standartized_image_list = []
        for img in imageList:
            m = np.mean(img, axis=1, keepdims=True)
            s = np.std(img, axis=1, keepdims=True)
            standartized_image = np.expand_dims((img - m) / s, axis=-1)
            standartized_image_list.append(standartized_image)

        x_train = []
        y_train = []

        x_test = []
        y_test = []

        train_date = []
        test_date = []

        train_price = []
        test_price = []

        for index in range(len(standartized_image_list)):
            if(index < (len(standartized_image_list) * 0.8)):
                x_train.append(standartized_image_list[index])
                y_train.append(labelList[index])
                train_date.append(data_close.index[index])
                train_price.append(data_close.iloc[index])
            else:
                x_test.append(standartized_image_list[index])
                y_test.append(labelList[index])
                test_date.append(data_close.index[index])
                test_price.append(data_close.iloc[index])

        np.save(f"./ETF/strategy/01/TrainData/x_{etf}.npy", x_train)
        np.save(f"./ETF/strategy/01/TrainData/y_{etf}.npy", y_train)
        np.save(f"./ETF/strategy/01/TestData/x_{etf}.npy", x_test)
        np.save(f"./ETF/strategy/01/TestData/y_{etf}.npy", y_test)

        np.save(f"./ETF/strategy/01/Date/TrainDate/{etf}.npy", train_date)
        np.save(f"./ETF/strategy/01/Date/TestDate/{etf}.npy", test_date)
        np.save(f'./ETF/strategy/01/Price/TrainPrice/{etf}.npy', train_price)
        np.save(f'./ETF/strategy/01/Price/TestPrice/{etf}.npy', test_price)

    except Exception as e:
        print(f"Error processing {etf}: {str(e)}")
        continue