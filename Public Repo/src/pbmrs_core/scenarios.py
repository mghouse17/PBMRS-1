from __future__ import annotations

from .models import Scenario, ScenarioSpec, SimConfig


_SCENARIOS = {
    "flash_crash": ScenarioSpec(
        name="flash_crash",
        description="A sudden liquidity drought and volatility surge.",
        parameters={"alpha_r": 12.0, "gamma_v": 0.08, "eta_l": 0.03, "gamma_l": 0.05},
    ),
    "liquidity_drought": ScenarioSpec(
        name="liquidity_drought",
        description="Persistent liquidity stress with weakening market depth.",
        parameters={"alpha_r": 9.0, "gamma_v": 0.06, "eta_l": 0.025, "gamma_l": 0.04},
    ),
    "central_bank_shock": ScenarioSpec(
        name="central_bank_shock",
        description="A regime shift driven by policy surprise and feedback amplification.",
        parameters={"alpha_r": 10.0, "gamma_v": 0.07, "eta_l": 0.02, "gamma_l": 0.035},
    ),
}


def build_default_scenario(name: str) -> Scenario:
    spec = _SCENARIOS[name]
    base_cfg = SimConfig()
    config = SimConfig(**{**base_cfg.__dict__, **spec.parameters})
    return Scenario(name=spec.name, description=spec.description, config=config)
