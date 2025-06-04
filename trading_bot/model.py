import logging
from typing import Tuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_supervised_model(data: pd.DataFrame) -> Tuple[Pipeline, float]:
    """Train a simple classifier to predict next 5-minute return direction."""
    df = data.copy()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    features = ["returns", "ma_fast", "ma_slow", "volatility"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    clf = Pipeline(
        [("scaler", StandardScaler()), ("gbc", GradientBoostingClassifier())]
    )
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    logger.info("Supervised model accuracy: %.2f%%", score * 100)
    return clf, score
