"""Simple long-only backwardation backtest.

The strategy goes long the front-month proxy when the backwardation
z-score exceeds a threshold; flat otherwise. The 'asset' here is a
synthetic excess return from holding the front contract — illustrative
only, not production alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series       # daily strategy returns
    equity: pd.Series        # cumulative equity (base 1.0)
    benchmark: pd.Series     # buy-and-hold front-month equity
    stats: dict

    def summary(self) -> str:
        s = self.stats
        return (
            f"Sharpe (strategy):  {s['sharpe']:.2f}\n"
            f"Sharpe (benchmark): {s['sharpe_bm']:.2f}\n"
            f"CAGR (strategy):    {s['cagr']:.2%}\n"
            f"CAGR (benchmark):   {s['cagr_bm']:.2%}\n"
            f"Max DD (strategy):  {s['mdd']:.2%}\n"
            f"Max DD (benchmark): {s['mdd_bm']:.2%}\n"
            f"Turnover (annual):  {s['turnover']:.2f}\n"
            f"Time in market:     {s['time_in_market']:.2%}"
        )


def _max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1.0
    return float(dd.min())


def _stats(returns: pd.Series, periods: int = 252) -> dict:
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0.0, "cagr": 0.0, "mdd": 0.0}
    sharpe = float(np.sqrt(periods) * returns.mean() / returns.std())
    equity = (1 + returns).cumprod()
    years = len(returns) / periods
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0
    mdd = _max_drawdown(equity)
    return {"sharpe": sharpe, "cagr": cagr, "mdd": mdd}


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    front: int = 1,
    cost_bps: float = 2.0,
) -> BacktestResult:
    """Run a long-only backtest driven by `signal['position']`.

    Parameters
    ----------
    prices : pd.DataFrame
        Term structure panel (date x tenor_months).
    signal : pd.DataFrame
        Output of `signals.backwardation_signal`. Uses the `position` col.
    front : int
        Tenor (months) used as the traded contract.
    cost_bps : float
        Round-trip transaction cost in basis points, charged on |dPos|.

    Returns
    -------
    BacktestResult
    """
    if front not in prices.columns:
        raise KeyError(f"Front tenor {front} not in panel.")

    asset_ret = np.log(prices[front]).diff().fillna(0.0)
    pos = signal["position"].shift(1).fillna(0.0)        # signal acts next day
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = (cost_bps / 1e4) * turnover

    strat_ret = pos * asset_ret - cost
    bm_ret = asset_ret.copy()

    equity = (1 + strat_ret).cumprod()
    bm_equity = (1 + bm_ret).cumprod()

    s = _stats(strat_ret)
    b = _stats(bm_ret)
    stats = {
        "sharpe": s["sharpe"],
        "sharpe_bm": b["sharpe"],
        "cagr": s["cagr"],
        "cagr_bm": b["cagr"],
        "mdd": s["mdd"],
        "mdd_bm": b["mdd"],
        "turnover": float(turnover.mean() * 252),
        "time_in_market": float((pos > 0).mean()),
    }

    return BacktestResult(
        returns=strat_ret.rename("strategy"),
        equity=equity.rename("strategy_equity"),
        benchmark=bm_equity.rename("benchmark_equity"),
        stats=stats,
    )
