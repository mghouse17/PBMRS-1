# Configs and Reproducibility

## Goals
- A simulation is fully determined by (config + seed + code version).
- Configs are human-readable and version-controlled.

## Recommended Config Fields
- `seed`
- `timesteps`
- `n_agents`
- `beta` (agent sensitivity)
- `kappa` (flow scaling)
- `alpha` (impact of flow on returns)
- `lambda_v`, `lambda_l`
- noise params (`sigma_eps`, `eta_v`, `eta_l`)
- constraint params (`min_liquidity_epsilon`, clamp settings)

## Example (base.yaml)
seed: 42
timesteps: 2000
n_agents: 5000
beta: 1.2
kappa: 0.8
alpha: 0.05
lambda_v: 0.05
lambda_l: 0.03
sigma_eps: 1.0
eta_v: 0.0
eta_l: 0.0
min_liquidity_epsilon: 1e-6

## Output Folder Convention
results/
  run_YYYYMMDD_HHMM/
    config.yaml
    timeseries.csv
    summary.json
    diagnostics.json