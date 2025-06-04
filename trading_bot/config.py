import os
from dataclasses import dataclass


@dataclass
class APIConfig:
    binance_api_key: str = os.getenv("BINANCE_API_KEY", "")
    binance_secret_key: str = os.getenv("BINANCE_SECRET_KEY", "")
    santiment_key: str = os.getenv("SANTIMENT_API_KEY", "")
    news_key: str = os.getenv("NEWS_API_KEY", "")
    crypto_panic_key: str = os.getenv("CRYPTO_PANIC_API_KEY", "")
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_secret: str = os.getenv("REDDIT_SECRET", "")


def load_config() -> APIConfig:
    """Load API keys from environment variables."""
    return APIConfig()
