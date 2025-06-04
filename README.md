# ZEUS_BOT

A framework for experimenting with high-frequency cryptocurrency trading using machine learning and reinforcement learning techniques.

## Features

- Fetches 5-minute OHLCV data from Binance.
- Generates technical indicators for supervised learning.
- Trains a gradient boosting classifier to predict short-term price direction.
- Provides a simple PPO reinforcement learning environment for strategy optimisation.
- Designed to maintain a positive risk–reward ratio with dynamic stop-loss and take-profit logic (to be implemented).

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Export required API keys as environment variables:

- `BINANCE_API_KEY`
- `BINANCE_SECRET_KEY`
- `SANTIMENT_API_KEY` (optional)
- `NEWS_API_KEY` (optional)
- `CRYPTO_PANIC_API_KEY` (optional)
- `REDDIT_CLIENT_ID` and `REDDIT_SECRET` (optional)

3. Run the training pipeline:

```bash
python -m trading_bot.main
```

This will download recent market data, train the supervised model, and perform a short reinforcement learning session. Order execution logic should be added where indicated in `trading_bot/main.py`.
