import numpy as np
import pandas as pd


FEATURES = [
    "ret_1",
    "ret_3",
    "ema_gap",
    "rsi",
    "macd",
    "macd_signal",
    "atr_pct",
    "volume_z"
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    required = {
        "open",
        "high",
        "low",
        "close",
        "volume"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing columns: {sorted(missing)}"
        )

    close = df["close"]

    # Price momentum
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)

    # EMA relationship
    ema_fast = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["ema_gap"] = (
        (ema_fast - ema_slow) / close
    )

    # RSI
    delta = close.diff()

    gain = (
        delta
        .clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    df["rsi"] = (
        100 - (100 / (1 + rs))
    )

    # MACD
    df["macd"] = ema_fast - ema_slow

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    # ATR
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - close.shift()).abs(),
            (df["low"] - close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(14).mean()

    df["atr_pct"] = atr / close

    # Volume
    volume_mean = (
        df["volume"]
        .rolling(30)
        .mean()
    )

    volume_std = (
        df["volume"]
        .rolling(30)
        .std()
        .replace(0, np.nan)
    )

    df["volume_z"] = (
        (df["volume"] - volume_mean)
        / volume_std
    )

    return df
