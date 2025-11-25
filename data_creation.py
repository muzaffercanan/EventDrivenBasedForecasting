import yfinance as yf
import pandas as pd
import numpy as np
import talib as tb

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
threshold = 0.01  # Re-arrange the Threshold Value

pd.set_option('display.max_rows', None)

'''
DOWNLOADING THE DATA
'''
for etf in etfList:

    imageList = []
    labelList = []

    data = yf.download(etf, start=startDate, end=endDate)

    '''
    CALCULATING THE INDICATOR VALUES
    '''
    # Önce ilgili sütunları 1D numpy array olarak alıyoruz.
    close = data["Close"].values.astype(float).ravel()
    high = data["High"].values.astype(float).ravel()
    low = data["Low"].values.astype(float).ravel()
    volume = data["Volume"].values.astype(float).ravel()

    rsi = pd.DataFrame(tb.RSI(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    wma = pd.DataFrame(tb.WMA(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    ema = pd.DataFrame(tb.EMA(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    sma = pd.DataFrame(tb.SMA(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    roc = pd.DataFrame(tb.ROC(close, timeperiod=10), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    cmo = pd.DataFrame(tb.CMO(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    cci = pd.DataFrame(tb.CCI(high, low, close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    ppo = pd.DataFrame(tb.PPO(close, fastperiod=12, slowperiod=26, matype=0), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    tema = pd.DataFrame(tb.TEMA(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    willr = pd.DataFrame(tb.WILLR(high, low, close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    macd, macdsignal, macdhist = tb.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd = pd.DataFrame(macd, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    sar = pd.DataFrame(tb.SAR(high, low, acceleration=0, maximum=0), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    adx = pd.DataFrame(tb.ADX(high, low, close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    std = pd.DataFrame(tb.STDDEV(close, timeperiod=5, nbdev=1), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    obv = pd.DataFrame(tb.OBV(close, volume), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    adxr = pd.DataFrame(tb.ADXR(high, low, close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    apo = pd.DataFrame(tb.APO(close, fastperiod=12, slowperiod=26, matype=0), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    aroondown, aroonup = tb.AROON(high, low, timeperiod=14)
    aroondown = pd.DataFrame(aroondown, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    aroonup = pd.DataFrame(aroonup, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    aroonosc = pd.DataFrame(tb.AROONOSC(high, low, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    bop = pd.DataFrame(tb.BOP(data["Open"].values.astype(float).ravel(), high, low, close), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    dx = pd.DataFrame(tb.DX(high, low, close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    macdext, macdextsignal, macdexthist = tb.MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    macdext = pd.DataFrame(macdext, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    macdfix, macdfixsignal, macdfixhist = tb.MACDFIX(close, signalperiod=9)
    macdfix = pd.DataFrame(macdfix, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    mfi = pd.DataFrame(tb.MFI(high, low, close, volume, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    minus_di = pd.DataFrame(tb.MINUS_DI(high, low, close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    minus_dm = pd.DataFrame(tb.MINUS_DM(high, low, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    mom = pd.DataFrame(tb.MOM(close, timeperiod=10), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    plus_di = pd.DataFrame(tb.PLUS_DI(high, low, close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    plus_dm = pd.DataFrame(tb.PLUS_DM(high, low, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    rocp = pd.DataFrame(tb.ROCP(close, timeperiod=10), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    rocr = pd.DataFrame(tb.ROCR(close, timeperiod=10), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    rocr100 = pd.DataFrame(tb.ROCR100(close, timeperiod=10), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    slowk, slowd = tb.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    slowk = pd.DataFrame(slowk, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    slowd = pd.DataFrame(slowd, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    fastk, fastd = tb.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastk = pd.DataFrame(fastk, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    fastd = pd.DataFrame(fastd, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    fastkrsi, fastdrsi = tb.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    fastkrsi = pd.DataFrame(fastkrsi, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    fastdrsi = pd.DataFrame(fastdrsi, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    trix = pd.DataFrame(tb.TRIX(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    ultosc = pd.DataFrame(tb.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    bbands_upperband, bbands_middleband, bbands_lowerband = tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    bbands_upperband = pd.DataFrame(bbands_upperband, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    bbands_middleband = pd.DataFrame(bbands_middleband, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    bbands_lowerband = pd.DataFrame(bbands_lowerband, index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    dema = pd.DataFrame(tb.DEMA(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    ht_trendline = pd.DataFrame(tb.HT_TRENDLINE(close), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    kama = pd.DataFrame(tb.KAMA(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    ma = pd.DataFrame(tb.MA(close, timeperiod=30, matype=0), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    midpoint = pd.DataFrame(tb.MIDPOINT(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    midprice = pd.DataFrame(tb.MIDPRICE(high, low, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    sarext = pd.DataFrame(tb.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0,
                       accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    trima = pd.DataFrame(tb.TRIMA(close, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    ad = pd.DataFrame(tb.AD(high, low, close, volume), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    adosc = pd.DataFrame(tb.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    trange = pd.DataFrame(tb.TRANGE(high, low, close), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    avgprice = pd.DataFrame(tb.AVGPRICE(data["Open"].values.astype(float).ravel(), high, low, close), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    medprice = pd.DataFrame(tb.MEDPRICE(high, low), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    typprice = pd.DataFrame(tb.TYPPRICE(high, low, close), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    wclprice = pd.DataFrame(tb.WCLPRICE(high, low, close), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    beta = pd.DataFrame(tb.BETA(high, low, timeperiod=5), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    correl = pd.DataFrame(tb.CORREL(high, low, timeperiod=30), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    linearreg = pd.DataFrame(tb.LINEARREG(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    linearreg_angle = pd.DataFrame(tb.LINEARREG_ANGLE(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    linearreg_intercept = pd.DataFrame(tb.LINEARREG_INTERCEPT(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    linearreg_slope = pd.DataFrame(tb.LINEARREG_SLOPE(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    tsf = pd.DataFrame(tb.TSF(close, timeperiod=14), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)
    var = pd.DataFrame(tb.VAR(close, timeperiod=5, nbdev=1), index=data.index, columns=['Value']).reset_index().set_axis(axes, axis=1)

    '''
    PREPROCESSING INDICATOR DATA
    '''
    # List of (indicators) DataFrames
    indicators = [rsi, cmo, plus_di, minus_di, willr, cci, ultosc, aroonosc, mfi, mom, macd, macdfix, linearreg_angle, linearreg_slope, rocp, roc, rocr, rocr100, slowk, fastd, slowd, aroonup, aroondown, apo,
                  macdext, fastk, ppo, minus_dm, adosc, fastdrsi, fastkrsi, trange, trix, std, bop, var, plus_dm, correl, ad, beta, wclprice, tsf, typprice, avgprice, medprice, bbands_lowerband, linearreg, obv,
                  bbands_middleband, tema, bbands_upperband, dema, midprice, midpoint, wma, ema, ht_trendline, kama, sma, ma, adxr, adx, trima, linearreg_intercept, dx]
    nIndicators = len(indicators)

    maxNullVal = -1
    for indicator in indicators:
        if indicator['Value'].isnull().sum() > maxNullVal:
            maxNullVal = indicator['Value'].isnull().sum()

    indicatorValues = []
    for indicator in indicators:
        indicatorValues.append(indicator['Value'].iloc[maxNullVal:])

    indicatorValuesMatrix = pd.concat(indicatorValues, axis=1, keys=headers)
    indicatorCorr = indicatorValuesMatrix.corr(method='pearson')

    '''
    CREATING THE IMAGES
    '''
    nDays = len(indicatorValues[0])
    for idx in range(nDays - 2 * nIndicators):
        image = []
        for indicatorValue in indicatorValues:
            imageRow = indicatorValue[idx:idx + 2 * nIndicators].values
            image.append(imageRow)
        imageList.append(np.array(image))

    '''
    CREATING THE LABELS
    '''
    # Get the close prices starting from the appropriate offset
    data_close = data.iloc[maxNullVal + 2 * nIndicators - 1:]["Close"]
    
    # Convert to regular Python list for easier iteration
    close_prices = data_close.values
    
    # Create labels based on price differences
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

    print(len(imageList))
    print(len(labelList))
    print(len(data_close[:-1]))

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
        if index < (len(standartized_image_list) * 0.8):
            x_train.append(standartized_image_list[index])
            y_train.append(labelList[index])
            train_date.append(data_close.index[index])
            train_price.append(data_close.iloc[index])
        else:
            x_test.append(standartized_image_list[index])
            y_test.append(labelList[index])
            test_date.append(data_close.index[index])
            test_price.append(data_close.iloc[index])

    np.save(f"./ETF/rectangle/01/TrainData/x_{etf}.npy", x_train)
    np.save(f"./ETF/rectangle/01/TrainData/y_{etf}.npy", y_train)
    np.save(f"./ETF/rectangle/01/TestData/x_{etf}.npy", x_test)
    np.save(f"./ETF/rectangle/01/TestData/y_{etf}.npy", y_test)

    np.save(f"./ETF/rectangle/01/Date/TrainDate/{etf}.npy", train_date)
    np.save(f"./ETF/rectangle/01/Date/TestDate/{etf}.npy", test_date)
    np.save(f'./ETF/rectangle/01/Price/TrainPrice/{etf}.npy', train_price)
    np.save(f'./ETF/rectangle/01/Price/TestPrice/{etf}.npy', test_price)