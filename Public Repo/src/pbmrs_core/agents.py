from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseAgent:
    """A minimal, extensible agent interface for the model."""
    name: str
    state: float = 0.0
    risk_tolerance: float = 0.5
    reaction_coefficient: float = 1.0
    memory: float = 0.0
    activity_probability: float = 0.5
    order_size: float = 1.0

    def step(self, market_signal: float) -> float:
        return self.state + self.reaction_coefficient * market_signal


@dataclass
class MomentumAgent(BaseAgent):
    def step(self, market_signal: float) -> float:
        return self.state + self.reaction_coefficient * market_signal + self.memory


@dataclass
class FundamentalAgent(BaseAgent):
    def step(self, market_signal: float) -> float:
        return self.state + 0.5 * self.reaction_coefficient * market_signal


@dataclass
class NoiseAgent(BaseAgent):
    def step(self, market_signal: float) -> float:
        return self.state + self.reaction_coefficient * (market_signal * 0.1)


@dataclass
class LiquidityProvider(BaseAgent):
    def step(self, market_signal: float) -> float:
        return self.state - 0.25 * self.reaction_coefficient * market_signal


@dataclass
class Contrarian(BaseAgent):
    def step(self, market_signal: float) -> float:
        return self.state - self.reaction_coefficient * market_signal


@dataclass
class Institutional(BaseAgent):
    def step(self, market_signal: float) -> float:
        return self.state + self.reaction_coefficient * market_signal * 1.5


@dataclass
class Retail(BaseAgent):
    def step(self, market_signal: float) -> float:
        return self.state + self.reaction_coefficient * market_signal * 0.8
