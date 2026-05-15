"""End-to-end example: load data, build signal, backtest, plot."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

# allow `python examples/run_analysis.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import run_backtest
from src.data_loader import load_data
from src.signals import backwardation_signal
from src.term_structure import fit_panel
from src import visualization as viz


def main() -> None:
    panel = load_data(start="2015-01-01", use_live=False)
    print(f"Loaded panel: {panel.n_obs} obs, tenors = {list(panel.prices.columns)}")

    factors = fit_panel(panel.prices, lam=6.0)
    print("\nNelson-Siegel factor summary:")
    print(factors.describe().round(4))

    signal = backwardation_signal(
        panel.prices,
        method="roll_yield",
        near=1, far=3,
        z_window=252,
        threshold=1.0,
    )
    print(f"\nFraction of days in long position: {signal['position'].mean():.2%}")
    print("Regime distribution (1=backwardation, 0=neutral, -1=contango):")
    print(signal["regime"].value_counts(normalize=True).round(3).to_string())

    bt = run_backtest(panel.prices, signal, front=1, cost_bps=2.0)
    print("\nBacktest summary")
    print("-" * 40)
    print(bt.summary())

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    sample_dates = [
        panel.prices.index[len(panel.prices) // 4],
        panel.prices.index[len(panel.prices) // 2],
        panel.prices.index[-1],
    ]
    viz.plot_term_structure_snapshot(panel.prices, sample_dates, ax=axes[0, 0])
    viz.plot_signal(signal, ax=axes[0, 1])
    viz.plot_heatmap(panel.prices, ax=axes[1, 0])
    viz.plot_backtest(bt, ax=axes[1, 1])
    fig.tight_layout()

    out = Path(__file__).resolve().parent / "analysis_output.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
