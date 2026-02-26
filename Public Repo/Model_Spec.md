
---

# PBMRS Model Spec

This document defines the PBMRS state variables, update equations, constraints, and diagnostics.  
It is the **source of truth** for implementation and testing.

## State Variables (time t)
- **x_t**: price/return proxy state (log-price or price index, depending on implementation choice)
- **v_t**: volatility state (must satisfy v_t >= 0)
- **ℓ_t**: liquidity state (must satisfy ℓ_t > 0)
- **m_t**: magnetization / aggregate sentiment in [-1, 1]
- **Q_t**: flow / order imbalance (function of m_t, liquidity, and scaling)

## Agent Layer
### Agent spin/state
Each agent i has spin s_i(t) ∈ {−1, +1}.

### Magnetization
m_t = (1/N) * Σ_i s_i(t)

### Spin update probability (logistic)
P[s_i(t+1)=+1] = 1 / (1 + exp(−β * h_i(t)))
Where h_i(t) represents the field (trend, mean reversion, volatility/liquidity coupling).

> Implementation note: document the exact h_i(t) definition used in code.

## Flow Layer
Q_t = κ * m_t * f(ℓ_t)
- κ: flow scaling constant
- f(ℓ_t): liquidity response term (choose and document)

## Return / Price Update
r_t = α * Q_t + σ_t * ε_t
- ε_t ~ Normal(0,1)
- σ_t is linked to v_t (e.g., σ_t = sqrt(v_t) or σ_t = v_t depending on definition)

x_{t+1} = x_t + r_t

## Volatility Update
v_{t+1} = (1−λ_v) * v_t + λ_v * g(r_t) + η_v
- g(r_t) could be r_t^2 (common)
- η_v optional noise term

Constraint: v_{t+1} >= 0 (enforced via clamp or reparameterization)

## Liquidity Update
ℓ_{t+1} = (1−λ_ℓ) * ℓ_t + λ_ℓ * h(v_t) + η_ℓ
- h(v_t) typically decreasing in v_t (liquidity dries up when vol rises)

Constraint: ℓ_{t+1} > 0 (enforce by softplus / clamp + epsilon)

## Diagnostics (summary)
### Drawdown
DD_t = 1 − x_t / max_{τ≤t}(x_τ)

### Max drawdown
MDD = max_t DD_t

### Recovery time
Time from a peak to the point where x_t exceeds that peak again.

### Tail risk (Monte Carlo)
Run K simulations → compute probability of DD exceeding a threshold, and distribution of losses.

## Implementation Contracts
- m_t must always be in [-1, 1]
- v_t must always be >= 0
- ℓ_t must always be > 0
- Runs must be reproducible with fixed seed and config