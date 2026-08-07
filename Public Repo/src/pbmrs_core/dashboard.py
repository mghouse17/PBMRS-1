from __future__ import annotations

from pathlib import Path

import streamlit as st

from .models import SimConfig
from .simulation import run_sim


def main() -> None:
    st.set_page_config(page_title="PBMRS Dashboard", layout="wide")
    st.title("PBMRS: Physics-Based Market Risk Simulator")
    st.write("Interactive stress simulation and diagnostics")

    timesteps = st.sidebar.slider("Timesteps", 50, 2000, 300)
    n_agents = st.sidebar.slider("Agents", 50, 5000, 500)
    seed = st.sidebar.number_input("Seed", 0, 100000, 42, step=1)

    if st.button("Run simulation"):
        cfg = SimConfig(seed=int(seed), timesteps=int(timesteps), n_agents=int(n_agents))
        result = run_sim(cfg)
        st.line_chart(result.prices)
        st.metric("Final price", f"{result.prices[-1]:.4f}")


if __name__ == "__main__":
    main()
