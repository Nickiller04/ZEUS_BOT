import logging
from typing import Any

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """A simple trading environment for reinforcement learning."""

    def __init__(self, data: np.ndarray, initial_balance: float = 1000.0):
        super().__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # +1 for long, -1 for short
        self.index = 0
        return self.data[self.index]

    def step(self, action: int):
        done = False
        reward = 0.0
        if action == 1 and self.position == 0:
            self.position = 1
        elif action == 2 and self.position == 0:
            self.position = -1
        elif action == 0:
            pass

        self.index += 1
        if self.index >= len(self.data) - 1:
            done = True

        price_change = self.data[self.index][0] - self.data[self.index - 1][0]
        reward = self.position * price_change
        obs = self.data[self.index]
        return obs, reward, done, {}


def train_rl_agent(data: np.ndarray, timesteps: int = 10000) -> PPO:
    """Train a PPO agent on the trading environment."""
    env = TradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    logger.info("RL agent training completed")
    return model
