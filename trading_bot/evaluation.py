import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def calculate_profit(trades: List[float]) -> float:
    return float(np.sum(trades))


def risk_reward_ratio(profits: List[float], losses: List[float]) -> float:
    if not losses:
        return float("inf")
    return abs(np.mean(profits)) / abs(np.mean(losses))
