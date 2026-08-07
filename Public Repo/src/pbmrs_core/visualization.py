from __future__ import annotations

import matplotlib.pyplot as plt


def plot_series(values, title: str = "Series") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.show()
