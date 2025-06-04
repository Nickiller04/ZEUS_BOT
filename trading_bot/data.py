import logging
from typing import List, Dict

import pandas as pd
from binance.client import Client

from .config import load_config

logger = logging.getLogger(__name__)


def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Fetch historical klines from Binance."""
    cfg = load_config()
    client = Client(cfg.binance_api_key, cfg.binance_secret_key)
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(
        raw,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df = df.astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate features for the ML model."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["ma_fast"] = df["close"].rolling(window=5).mean()
    df["ma_slow"] = df["close"].rolling(window=20).mean()
    df["volatility"] = df["returns"].rolling(window=20).std()
    df.dropna(inplace=True)
    return df
