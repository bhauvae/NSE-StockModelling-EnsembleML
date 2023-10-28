import talib as tb


def format_data(df, avg_days=5):
    short_ind = 5
    long_ind = 10
    extra_long_ind = 50
    # super_long_ind = 250
    # OVERLAP INDICATORS
    df["ma"] = tb.MA(df["Close"], timeperiod=short_ind)
    df["ema"] = tb.EMA(df["Close"], timeperiod=long_ind)
    df["dema"] = tb.DEMA(df["Close"], timeperiod=short_ind)
    df["kama"] = tb.KAMA(df["Close"], timeperiod=short_ind)
    df["sma"] = tb.SMA(df["Close"], timeperiod=long_ind)
    df["sar"] = tb.SAR(df["High"], df["Low"])

    df["long_ma"] = tb.MA(df["Close"], timeperiod=extra_long_ind)
    df["long_ema"] = tb.EMA(df["Close"], timeperiod=extra_long_ind)

    # df["super_ma"] = tb.MA(df["Close"], timeperiod=super_long_ind)
    # df["super_ema"] = tb.EMA(df["Close"], timeperiod=super_long_ind)

    # MOMENTUM INDICATORS
    df["adx"] = tb.ADX(df["High"], df["Low"],
                       df["Close"], timeperiod=long_ind)
    df["cci"] = tb.CCI(df["High"], df["Low"],
                       df["Close"], timeperiod=long_ind)
    df["apo"] = tb.APO(df["Close"], fastperiod=long_ind,
                       slowperiod=short_ind)
    df["bop"] = tb.BOP(df["Open"], df["High"], df["Low"], df["Close"])
    df["macd"], df["macdsignal"], df["macdhist"] = tb.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["mfi"] = tb.MFI(df["High"], df["Low"], df["Close"],
                       df["Volume"], timeperiod=long_ind)
    df["mom"] = tb.MOM(df["Close"], timeperiod=long_ind)
    df["rsi"] = tb.RSI(df["Close"], timeperiod=long_ind)
    df["adxr"] = tb.ADXR(df["High"], df["Low"], df["Close"], timeperiod=long_ind)
    df["aroon_up"], df["aroon_down"] = tb.AROON(df["High"], df["Low"], timeperiod=long_ind)


    # OSCILLATOR INDICATORS
    df["stoch"], df["stoch_signal"] = tb.STOCH(df["High"], df["Low"], df["Close"])
    df["rsi"] = tb.RSI(df["Close"], timeperiod=long_ind)
    df["williamsr"] = tb.WILLR(df["High"], df["Low"], df["Close"], timeperiod=long_ind)

    # VOLUME INDICATORS
    df["ad"] = tb.AD(df["High"], df["Low"], df["Close"], df["Volume"])
    df["adosc"] = tb.ADOSC(df["High"], df["Low"], df["Close"],
                           df["Volume"], fastperiod=short_ind, slowperiod=long_ind)
    df["obv"] = tb.OBV(df["Close"], df["Volume"])
    df["trange"] = tb.TRANGE(df["High"], df["Low"], df["Close"])
    df["atr"] = tb.ATR(df["High"], df["Low"],
                       df["Close"], timeperiod=long_ind)
    df["natr"] = tb.NATR(df["High"], df["Low"],
                         df["Close"], timeperiod=long_ind)
    df["roc"] = tb.ROC(df["Close"], timeperiod=long_ind)
    df["cmo"] = tb.CMO(df["Close"], timeperiod=long_ind)

    # VOLATILITY INDICATORS
    df["bbands_upper"], df["bbands_middle"], df["bbands_lower"] = tb.BBANDS(df["Close"], timeperiod=long_ind)
    df["kc_middle"] = tb.EMA(df["Close"], timeperiod=long_ind)
    df["kc_upper"] = df["kc_middle"] + df["atr"]
    df["kc_lower"] = df["kc_middle"] - df["atr"]

    df['closingmarubozu'] = tb.CDLCLOSINGMARUBOZU(df["Open"], df["High"], df["Low"], df["Close"])
    df['harami'] = tb.CDLHARAMI(df["Open"], df["High"], df["Low"], df["Close"])
    df['cdlshortline'] = tb.CDLSHORTLINE(df["Open"], df["High"], df["Low"], df["Close"])
    df['spinningtop'] = tb.CDLSPINNINGTOP(df["Open"], df["High"], df["Low"], df["Close"])
    df['cdllongline'] = tb.CDLLONGLINE(df["Open"], df["High"], df["Low"], df["Close"])
    df['breakaway'] = tb.CDLBREAKAWAY(df["Open"], df["High"], df["Low"], df["Close"])
    df['hammer'] = tb.CDLHAMMER(df["Open"], df["High"], df["Low"], df["Close"])
    df['doji'] = tb.CDLDOJI(df["Open"], df["High"], df["Low"], df["Close"])
    df['engulfing'] = tb.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])
    df['morning_star'] = tb.CDLMORNINGSTAR(df["Open"], df["High"], df["Low"], df["Close"])
    df['evening_star'] = tb.CDLEVENINGSTAR(df["Open"], df["High"], df["Low"], df["Close"])

    # 8 TRIGRAMS

    trigrams = []
    for i in range(1, len(df)):
        current_row = df.index[i]
        previous_row = df.index[i - 1]
        if (df.loc[current_row, "High"] > df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] < df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] > df.loc[previous_row, "Close"]):
            signal = 100  # "BullishHorn"
        elif (df.loc[current_row, "High"] > df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] < df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] < df.loc[previous_row, "Close"]):
            signal = -100  # "BearHorn"
        elif (df.loc[current_row, "High"] > df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] > df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] > df.loc[previous_row, "Close"]):
            signal = 100  # "BullishHigh"
        elif (df.loc[current_row, "High"] > df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] > df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] < df.loc[previous_row, "Close"]):
            signal = -100  # "BearHigh"
        elif (df.loc[current_row, "High"] < df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] < df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] > df.loc[previous_row, "Close"]):
            signal = 100  # "BullishLow"
        elif (df.loc[current_row, "High"] < df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] < df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] < df.loc[previous_row, "Close"]):
            signal = -100  # "BearLow"
        elif (df.loc[current_row, "High"] < df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] > df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] > df.loc[previous_row, "Close"]):
            signal = 100  # "BullishHarami"
        elif (df.loc[current_row, "High"] < df.loc[previous_row, "High"]) & (df.loc[current_row, "Low"] > df.loc[previous_row, "Low"]) & (
                df.loc[current_row, "Close"] < df.loc[previous_row, "Close"]):
            signal = -100  # "BearHarami"
        else:
            signal = 0
        trigrams.append(signal)

    df.drop(df.index[0], inplace=True)
    df["trigrams"] = trigrams

    # TARGET
    df["target"] = df["Close"].pct_change().rolling(avg_days).mean().shift(avg_days)

    df.dropna(inplace=True)
    return df


