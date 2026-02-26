# PBMRS Architecture

## Modules
### 1) Core State (`pbmrs/core/state.py`)
Holds x_t, v_t, ℓ_t, and metadata (seed, step, timestamps).

### 2) Agents (`pbmrs/core/agents.py`)
- Represents agent population
- Computes magnetization m_t
- Updates spins using logistic probability

### 3) Market (`pbmrs/core/market.py`)
- Computes Q_t from m_t and liquidity
- Updates returns and x_{t+1}
- Updates v_{t+1} and ℓ_{t+1} with constraints

### 4) Simulation Orchestrator (`pbmrs/core/sim.py`)
Single-run loop:
1) agent update
2) compute m_t
3) compute Q_t
4) compute r_t and x_{t+1}
5) update v_{t+1}
6) update ℓ_{t+1}
7) log outputs

### 5) Diagnostics (`pbmrs/diagnostics/*`)
Drawdown, recovery, fragility regimes, tail risk.

## Data Flow (one step)
Agents → m_t → Q_t → r_t/x → v → ℓ → back into agent field h_i(t)

## Design Principles
- Config-driven parameters
- Deterministic + seeded RNG
- Separation of core update equations from logging/IO
- Clear “contracts” for constraints and invariants