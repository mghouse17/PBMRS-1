# PBMRS Architecture

## Overview

PBMRS is structured as a modular Python research platform with a clear separation between configuration, mathematical dynamics, diagnostics, and optional interfaces.

## Components

- `models.py`: typed dataclasses for configs, results, and scenarios
- `math.py`: isolated equation implementations for order flow, returns, volatility, liquidity, and agent updates
- `simulation.py`: orchestration of the discrete-time feedback loop
- `diagnostics.py`: drawdown, recovery, fragility, and tail metrics
- `scenarios.py`: scenario presets for stress testing
- `api.py` and `dashboard.py`: optional execution interfaces

## Data Flow

Agents → order flow → returns → volatility/liquidity → market field → agents

## Design Principles

- no global state
- injectable configuration
- typed public interfaces
- scientific clarity over marketing claims