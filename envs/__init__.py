"""
Environment package initialization for the balancing bot.
Registers BalancingBot-v0 with Gymnasium.
"""

from gymnasium.envs.registration import register
from envs.balance_bot_env import BalanceBotEnv

register(
    id="BalancingBot-v0",
    entry_point="envs.balance_bot_env:BalanceBotEnv",
    max_episode_steps=10000,
)

__all__ = ["BalanceBotEnv"]
