# PBMRS: Physics-Based Market Risk Simulator

PBMRS is a modular Python research framework for studying market fragility, volatility regimes, liquidity stress, and feedback-driven instability. It is designed as a scientific simulation platform rather than a trading system.

## What PBMRS does

PBMRS studies how systemic instability emerges from interacting agents, feedback loops, and market state variables such as:

- log price
- volatility
- liquidity
- magnetization
- order flow

The framework centers on a discrete-time feedback loop:

Agents → order flow → returns → volatility/liquidity → market field → agents

## Features

- deterministic simulation core with configurable seeds
- scenario presets for stress and crash-like conditions
- diagnostics for drawdown, fragility, regime labels, and tail behavior
- auditable FRED/CFTC data access and fixed-grid finite-sample calibration
- optional FastAPI service and Streamlit dashboard entry points
- regression and integration tests covering core behavior

## Requirements

- Python 3.10 or newer
- pip

## Installation

From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
```

For development and notebook execution:

```bash
python -m pip install -e ".[dev,notebook]"
```

The exact environment used for the committed notebook outputs is recorded in requirements-lock.txt.

If you are using the project from a different working directory, point pip at the repository folder explicitly:

```bash
python -m pip install -e "C:/Users/Mghou/PBMRS-1/Public Repo"
```

## Quick start

Run a short simulation from Python:

```python
from pbmrs_core import SimConfig, run_sim, max_drawdown

cfg = SimConfig(seed=7, timesteps=200, n_agents=200, q0=0.01)
out = run_sim(cfg)
print(f"Final price: {out.prices[-1]:.4f}")
print(f"Max drawdown: {max_drawdown(out.prices):.4f}")
```

Run the built-in demo script:

```bash
python demo.py
```

Explore the maintained notebooks:

- [MVP simulation](notebooks/00_pbmrs_mvp.ipynb)
- [Phase-transition diagnostics](notebooks/01_pbmrs_phase_transition.ipynb)
- [FRED WTI regime calibration](notebooks/02_wti_regime_calibration.ipynb)

Rebuild the application analysis and notebook with:

```bash
python notebooks/build_gmsg_analysis.py
python notebooks/build_gmsg_notebook.py
```

## Running the API and dashboard

Start the FastAPI service:

```bash
uvicorn pbmrs_core.api:app --reload
```

Start the Streamlit dashboard:

```bash
streamlit run src/pbmrs_core/dashboard.py
```

## Testing

Run the full test suite:

```bash
pytest -q
```

## Project layout

- `src/pbmrs_core/` — simulation core, models, diagnostics, config, scenarios, API, dashboard, and visualization
- `tests/` — regression and integration tests
- `configs/` — YAML configuration examples
- `docs/` — architecture and specification notes
- `demo.py` — minimal runnable example

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Notes

PBMRS uses physics-inspired mathematical structure as a modelling language, not as a claim that markets literally obey physical laws.
