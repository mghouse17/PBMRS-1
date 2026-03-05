# PBMRS: Physics-Based Market Risk Simulator

A research framework for studying market fragility, instability, and systemic risk using physics-inspired modelling.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Portfolio project.** This public repository demonstrates the framework's architecture and modelling approach. The full production implementation is maintained privately.

---

## What it is

PBMRS treats financial markets as complex adaptive systems: populations of interacting agents operating under uncertainty, coupled to a shared price and liquidity environment through explicit feedback.

The framework is not a trading system. It does not forecast prices or generate signals. Its purpose is analytical — understanding how instability, herding, and liquidity crises *emerge* from the structure of the system rather than from external shocks.

---

## Motivation

Standard risk models assume markets are near equilibrium and returns are approximately Gaussian. Real markets show volatility clustering, fat tails, sudden liquidity withdrawal, and sharp regime shifts — all of which emerge from feedback between agents and market state, not from the size of individual shocks.

PBMRS addresses this by borrowing from:

- **Statistical physics** — Ising-inspired agent dynamics, phase transitions, order parameters
- **Stochastic processes** — explicit noise in returns, volatility, and liquidity
- **Dynamical systems** — coupled state evolution, nonlinear feedback, fixed-point analysis

Physics serves as a modelling language here, not a literal claim that markets obey physical laws.

---

## Model architecture

The system is a closed feedback loop evolving in discrete time:

```
Agents → mt → Qt → rt / xt → vt , ℓt → ht → Agents
```

Each arrow is a named equation from the model spec. No step is implicit.

### State variables

| Variable | Description | Constraint |
|----------|-------------|------------|
| `x_t` | Log-price | — |
| `v_t` | Volatility (risk temperature proxy) | `≥ 0` |
| `ℓ_t` | Liquidity depth | `> 0` |
| `m_t` | Magnetization — aggregate agent sentiment | `∈ [−1, +1]` |
| `Q_t` | Order flow (function of `m_t` and liquidity) | — |

### Equations

**Eq. 1 — Order flow**
```
Qt = q0 · N · mt
```
Scales aggregate sentiment into a signed order imbalance. Default sizing: `q0 · N = 1`, so `Qt ∈ [−1, +1]`.

**Eq. 2 — Log-return**
```
rt = μ0·Δt  +  λ · Qt / max(ℓt, ε)  +  √(vt·Δt) · σ · εt
```
Three components: drift, liquidity-adjusted price impact, and volatility-scaled noise. The floor `max(ℓt, ε)` caps impact when liquidity is near-zero.

**Eq. 3 — Volatility update**
```
vt+1 = (1−κv)·vt + κv·θv + ηv·rt² + γv·mt²
```
Mean-reverts to baseline `θv`, amplified by return shocks (ARCH-like) and by crowding. Fixed point: `vt = θv, rt = mt = 0 → vt+1 = θv`.

**Eq. 4 — Liquidity update**
```
ℓt+1 = (1−κℓ)·ℓt + κℓ·ℓ0 − ηℓ·|Qt| − γℓ·max(vt − θv, 0)
```
Replenishes toward baseline `ℓ0`, depleted by flow and by *excess* volatility above baseline. Calm markets (`vt ≈ θv, Qt ≈ 0`) maintain `ℓt ≈ ℓ0`. Only stress above the baseline drains liquidity. Fixed point: `ℓt = ℓ0, Qt = 0, vt = θv → ℓt+1 = ℓ0`.

**Eq. 5 — Market field**
```
ht = αr·rt  −  αv·vt  −  αℓ·(ℓ0/ℓt − 1)  +  α0
```
The signal each agent receives: trend-following (`αr·rt`), volatility aversion (`αv·vt`), liquidity-stress aversion (`αℓ·(ℓ0/ℓt − 1)`), and a baseline bias (`α0`).

**Eq. 6 — Agent update (Ising logistic)**
```
P(si,t+1 = +1) = σ(β · (J·mt + ht))
```
Agents update probabilistically. The coupling `J·mt` drives herding; `ht` drives market-state responsiveness. The logit is clipped to `[−50, +50]` for numerical safety.

### Criticality

The Ising subcritical condition is `J·β < 1`. Below this boundary, `mt` fluctuates around zero — agents disagree and liquidity is stable. Above it (`J·β ≥ 1`), the system can spontaneously lock to `m ≈ ±1` (full herding), triggering volatility spikes and liquidity withdrawal. The transition region (`J·β` near 1) is where fat tails, volatility clustering, and magnetization persistence are most pronounced.

---

## Repository structure

```
pbmrs/
├── src/pbmrs_core/
│   ├── sim.py          # SimConfig, SimResult, run_sim, run_ensemble
│   ├── diagnostics.py  # drawdown, ACF(r²), magnetization persistence, tail stats
│   └── analysis.py     # phase_map — 2-D parameter sweep over any config pair
├── tests/
│   └── test_sim.py     # 35+ unit, regression, edge-case, and diagnostic tests
├── configs/
│   └── base.yaml       # default parameters with inline documentation
├── notebooks/
│   └── 00_pbmrs_mvp.ipynb
└── docs/
    ├── Model_Spec.md
    ├── ARCHITECTURE.md
    ├── CONFIGS.md
    ├── TEST_PLAN.md
    └── ROADMAP.md
```

**What's included:** core framework, simulation engine, diagnostics, test suite, example analysis.

**What's private:** full production implementation, calibration tools, advanced analysis modules, proprietary optimisations.

---

## Installation

```bash
git clone https://github.com/yourusername/pbmrs.git
cd pbmrs
pip install -r requirements.txt
```

---

## Quick start

```python
from pbmrs_core.sim import SimConfig, run_sim, run_ensemble, check_invariants
from pbmrs_core.diagnostics import max_drawdown, acf_squared_returns, tail_stats

# Single run
cfg = SimConfig(seed=42, timesteps=2000, n_agents=2000)
out = run_sim(cfg)
check_invariants(out, cfg)

print(f"Max drawdown:     {max_drawdown(out.prices):.3f}")
print(f"Return std:       {out.r.std():.4f}")
print(f"ACF(r²) lag 1:    {acf_squared_returns(out.r)[1]:.3f}")

# Monte Carlo ensemble
results = run_ensemble(cfg, n_runs=100)
summary = tail_stats(results, l0=cfg.l0)
print(f"MDD mean ± std:   {summary['mdd_mean']:.3f} ± {summary['mdd_std']:.3f}")
print(f"Excess kurtosis:  {summary['excess_kurtosis']:.2f}")
print(f"Liq stressed:     {summary['liq_stressed_frac']:.1%} of runs")
```

```python
# Phase map — sweep J × gamma_v, colour by mean max drawdown
from pbmrs_core.analysis import phase_map
import numpy as np

result = phase_map(
    cfg_base = cfg,
    param_x  = "J",       values_x = np.linspace(0.3, 0.9, 10).tolist(),
    param_y  = "gamma_v", values_y = np.linspace(0.01, 0.1, 8).tolist(),
    n_runs   = 20,
    metrics  = ["mdd_mean", "herd_fraction", "acf_r2_lag1"],
)
# result["mdd_mean"] is a (8, 10) matrix ready for imshow / contourf
```

---

## Outputs and diagnostics

PBMRS produces interpretable risk metrics, not trading signals.

| Diagnostic | What it measures |
|------------|-----------------|
| `max_drawdown` | Worst peak-to-trough loss on a price path |
| `recovery_time` | Steps from trough until prices recover to prior peak |
| `acf_squared_returns` | ACF of r² — confirms volatility clustering when positive at small lags |
| `magnetization_persistence` | ACF of \|m\| and fraction of time in herding state |
| `tail_stats` | Ensemble summary: MDD distribution, excess kurtosis, liquidity stress fraction |
| `fragility_index` | Composite index combining crowding, vol, and liquidity stress |
| `regime_labels` | Stable / Fragile / Unstable classification by fragility percentile |
| `phase_map` | 2-D metric grid over any pair of config parameters |

---

## Calibration and reproducibility

A simulation run is fully determined by `(config + seed + code version)`. All parameters live in `configs/base.yaml` with inline documentation. Changing `seed` with the same config produces an independent path; changing any parameter updates the regression test baseline intentionally.

Key parameters to understand before varying anything:

- `J · beta` — distance from the critical point. Keep below 0.83 for subcritical dynamics; approach it to study near-critical behaviour.
- `gamma_v` — crowding amplification of volatility. The main knob for fat tails and clustering.
- `eta_l`, `gamma_l` — flow and excess-vol depletion of liquidity. Increase together to study liquidity crises.
- `lam` — price impact strength. Scales with `1/ℓt`, so liquidity and impact interact.

---

## Use cases

- **Systemic risk research** — how does herding interact with liquidity to produce crises?
- **Stress testing** — what parameter regions produce extreme drawdowns or no recovery?
- **Phase diagram analysis** — map the stable / fragile / unstable boundary across (J, γv) space
- **Stylized fact validation** — does the simulator reproduce fat tails, volatility clustering, and magnetization persistence at the right parameter values?
- **Risk model comparison** — baseline the simulator's tail behaviour against standard Gaussian models

---

## Disclaimer

PBMRS is a research tool for studying market dynamics. It is not intended for trading, investment decisions, or financial advice. This repository contains a demonstration implementation only. The author assumes no liability for any use of this software.

---

## Contact

For enquiries about employment, research collaboration, or access to the full implementation:

- LinkedIn: [linkedin.com/in/salman-g-664b812ab](https://www.linkedin.com/in/salman-g-664b812ab)
- Email: m.ghouse1720@gmail.com

---

**Status:** Active development &nbsp;|&nbsp; **Version:** 0.2.0 &nbsp;|&nbsp; **Last updated:** March 2026