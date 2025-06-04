import logging

import numpy as np

from .data import fetch_klines, compute_features
from .model import train_supervised_model
from .rl_agent import train_rl_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(symbol: str = "BTCUSDT", interval: str = Client.KLINE_INTERVAL_5MINUTE):
    df = fetch_klines(symbol, interval)
    df = compute_features(df)

    clf, accuracy = train_supervised_model(df)
    logger.info("Trained supervised model with accuracy %.2f%%", accuracy * 100)

    env_data = df[["close", "returns", "ma_fast", "ma_slow", "volatility"]].values
    model = train_rl_agent(env_data)
    logger.info("RL agent trained")

    # Placeholder for live trading integration
    logger.info("Trading bot setup complete. Implement order execution logic here.")


if __name__ == "__main__":
    from binance.client import Client

    run()
