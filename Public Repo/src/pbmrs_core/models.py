from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class SimConfig:
    seed: int = 42
    timesteps: int = 2000
    dt: float = 1.0
    n_agents: int = 2000
    q0: float = 5e-4
    beta: float = 1.2
    J: float = 0.5
    alpha_r: float = 1.0
    alpha_v: float = 0.001
    alpha_l: float = 0.1
    alpha_0: float = 0.0
    mu0: float = 0.0
    lam: float = 0.05
    sigma_eps: float = 0.01
    kappa_v: float = 0.05
    theta_v: float = 1.0
    eta_v: float = 0.85
    gamma_v: float = 0.050
    kappa_l: float = 0.03
    l0: float = 1.0
    eta_l: float = 0.015
    gamma_l: float = 0.030
    impact_eps: float = 0.010
    min_vol: float = 0.0
    min_liquidity: float = 1e-6
    x0: float = 0.0
    v0: float = 1.0

    def __post_init__(self) -> None:
        errors: List[str] = []
        if self.timesteps <= 0:
            errors.append(f"timesteps must be > 0 (got {self.timesteps})")
        if self.n_agents <= 0:
            errors.append(f"n_agents must be > 0 (got {self.n_agents})")
        if self.dt <= 0.0:
            errors.append(f"dt must be > 0 (got {self.dt})")
        if self.theta_v <= 0.0:
            errors.append(f"theta_v must be > 0 (got {self.theta_v})")
        if self.l0 <= 0.0:
            errors.append(f"l0 must be > 0 (got {self.l0})")
        if self.min_liquidity <= 0.0:
            errors.append(f"min_liquidity must be > 0 (got {self.min_liquidity})")
        if not (0.0 < self.kappa_v < 1.0):
            errors.append(f"kappa_v must be in (0, 1) for EWMA stability (got {self.kappa_v})")
        if not (0.0 < self.kappa_l < 1.0):
            errors.append(f"kappa_l must be in (0, 1) for EWMA stability (got {self.kappa_l})")
        if errors:
            raise ValueError("Invalid SimConfig (" + str(len(errors)) + " error(s)):\n" + "\n".join(f"  - {e}" for e in errors))

        jb = self.J * self.beta
        if jb >= 1.0:
            warnings.warn(f"J * beta = {jb:.4f} >= 1.0 (supercritical)", UserWarning, stacklevel=2)

        flow_scale = self.q0 * self.n_agents
        if not (0.5 <= flow_scale <= 2.0):
            warnings.warn(f"q0 * n_agents = {flow_scale:.3f} is outside [0.5, 2.0]", UserWarning, stacklevel=2)


class SimResult(tuple):
    __slots__ = ()

    def __new__(cls, x, v, l, m, Q, r, h, prices):
        return super().__new__(cls, (x, v, l, m, Q, r, h, prices))

    @property
    def x(self):
        return self[0]

    @property
    def v(self):
        return self[1]

    @property
    def l(self):
        return self[2]

    @property
    def m(self):
        return self[3]

    @property
    def Q(self):
        return self[4]

    @property
    def r(self):
        return self[5]

    @property
    def h(self):
        return self[6]

    @property
    def prices(self):
        return self[7]

    def _replace(self, **kwargs):
        data = {
            "x": self.x,
            "v": self.v,
            "l": self.l,
            "m": self.m,
            "Q": self.Q,
            "r": self.r,
            "h": self.h,
            "prices": self.prices,
        }
        data.update(kwargs)
        return SimResult(**data)


@dataclass
class ScenarioSpec:
    name: str
    description: str
    parameters: dict


@dataclass
class Scenario:
    name: str
    description: str
    config: SimConfig
