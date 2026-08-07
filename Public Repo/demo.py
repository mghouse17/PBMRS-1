"""Simple PBMRS demo script for running a short simulation and printing diagnostics."""

from __future__ import annotations

from pbmrs_core import SimConfig, max_drawdown, run_sim, tail_stats


def main() -> None:
    cfg = SimConfig(seed=7, timesteps=120, n_agents=200, q0=0.01)
    result = run_sim(cfg)

    print("PBMRS demo")
    print("-" * 40)
    print(f"Final price: {result.prices[-1]:.4f}")
    print(f"Max drawdown: {max_drawdown(result.prices):.4f}")
    print(f"Mean return: {result.r.mean():.6f}")
    print(f"Return std: {result.r.std():.6f}")
    print(f"Min liquidity: {result.l.min():.6f}")

    stats = tail_stats([result], l0=cfg.l0)
    print("\nTail stats:")
    print(f"  n_runs: {stats['n_runs']}")
    print(f"  mdd_mean: {stats['mdd_mean']:.6f}")
    print(f"  liq_stressed_frac: {stats['liq_stressed_frac']:.6f}")


if __name__ == "__main__":
    main()
