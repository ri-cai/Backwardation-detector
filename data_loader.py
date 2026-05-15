"""Crude oil futures data loader.

Multi-maturity WTI history requires a paid feed (CME, Bloomberg, Refinitiv).
To keep the project fully runnable, a calibrated synthetic generator
produces a realistic WTI-like panel: mean-reverting log-level, a slope
process that switches between contango and backwardation regimes, and a
small curvature factor. If `yfinance` is installed, the live front-month
(`CL=F`) can be attached as a reference series.

To plug in real data, replace the body of `load_data` so it returns a
`FuturesPanel` whose `prices` DataFrame is indexed by date with
tenor-in-months columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FuturesPanel:
    """Container for a futures term structure panel.

    Attributes
    ----------
    prices : pd.DataFrame
        Index = trading dates, columns = maturity tenors in months
        (e.g., 1, 2, 3, 6, 12, 18, 24). Values are settlement prices.
    spot : pd.Series, optional
        Reference spot / front-month series.
    """

    prices: pd.DataFrame
    spot: Optional[pd.Series] = None

    @property
    def tenors(self) -> np.ndarray:
        return np.asarray(self.prices.columns, dtype=float)

    @property
    def n_obs(self) -> int:
        return len(self.prices)


def fetch_yahoo_front_month(start: str = "2015-01-01") -> Optional[pd.Series]:
    """Try to fetch front-month WTI from Yahoo. Returns None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        logger.info("yfinance not installed; skipping live fetch.")
        return None

    try:
        ticker = yf.Ticker("CL=F")
        hist = ticker.history(start=start, auto_adjust=False)
        if hist.empty:
            return None
        return hist["Close"].rename("CL_front")
    except Exception as exc:  # pragma: no cover - network path
        logger.warning("Yahoo fetch failed: %s", exc)
        return None


def generate_synthetic_panel(
    start: str = "2015-01-01",
    end: Optional[str] = None,
    tenors: Optional[list] = None,
    seed: int = 42,
) -> FuturesPanel:
    """Generate a realistic synthetic WTI term-structure panel.

    Models:
      * Level  L_t : log-price OU around log(65), shocked occasionally
      * Slope  S_t : OU around 0 with regime jumps (contango/backwardation)
      * Curv   C_t : small AR(1)
    Term structure built from Nelson-Siegel-style loadings.
    """
    if tenors is None:
        tenors = [1, 2, 3, 6, 9, 12, 18, 24]
    tenors_arr = np.asarray(tenors, dtype=float)

    end = end or datetime.today().strftime("%Y-%m-%d")
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n == 0:
        raise ValueError("Empty date range.")

    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    # ---- Level: mean-reverting log price (Ornstein-Uhlenbeck) ----
    kappa_L, theta_L, sigma_L = 0.5, np.log(65.0), 0.30
    log_level = np.empty(n)
    log_level[0] = theta_L
    z_L = rng.standard_normal(n)
    for t in range(1, n):
        log_level[t] = (
            log_level[t - 1]
            + kappa_L * (theta_L - log_level[t - 1]) * dt
            + sigma_L * np.sqrt(dt) * z_L[t]
        )

    # Regime shocks (oil crashes / supply spikes)
    if n > 120:
        shock_count = max(1, n // 750)
        shock_dates = rng.choice(np.arange(60, n - 60), size=shock_count, replace=False)
        for sd in shock_dates:
            magnitude = rng.uniform(-0.4, 0.25)
            decay = np.exp(-np.arange(n - sd) / 60)
            log_level[sd:] += magnitude * decay

    level = np.exp(log_level)

    # ---- Slope: mean-reverting with regime jumps ----
    # slope > 0 => contango (far above near in our parameterisation),
    # slope < 0 => backwardation. This is the *factor*, not the
    # observed price slope, but it controls the same shape.
    kappa_S, sigma_S = 1.5, 0.15
    slope = np.empty(n)
    slope[0] = 0.0
    regime_state = 0.0
    z_S = rng.standard_normal(n)
    for t in range(1, n):
        if rng.uniform() < 1.0 / 180.0:  # avg regime switch ~9 months
            regime_state = rng.uniform(-0.18, 0.10)
        slope[t] = (
            slope[t - 1]
            + kappa_S * (regime_state - slope[t - 1]) * dt
            + sigma_S * np.sqrt(dt) * z_S[t]
        )

    # ---- Curvature: small AR(1) ----
    curvature = np.zeros(n)
    z_C = rng.standard_normal(n)
    for t in range(1, n):
        curvature[t] = 0.95 * curvature[t - 1] + 0.05 * z_C[t] * 0.05

    # ---- Build full term structure (Nelson-Siegel loadings) ----
    lam_months = 6.0
    x = tenors_arr / lam_months  # x = tau/lam
    # (1 - e^-x)/x  with the limit at x=0 handled
    slope_load = np.where(x == 0, 1.0, (1 - np.exp(-x)) / x)
    curv_load = slope_load - np.exp(-x)

    log_prices = (
        np.log(level)[:, None]
        + slope[:, None] * slope_load[None, :]
        + curvature[:, None] * curv_load[None, :]
    )
    prices_arr = np.exp(log_prices)

    prices = pd.DataFrame(
        prices_arr,
        index=dates,
        columns=[int(t) for t in tenors_arr],
    )
    prices.index.name = "date"
    prices.columns.name = "tenor_months"

    return FuturesPanel(prices=prices)


def load_data(
    start: str = "2015-01-01",
    end: Optional[str] = None,
    use_live: bool = True,
    tenors: Optional[list] = None,
) -> FuturesPanel:
    """Load crude oil futures panel.

    Parameters
    ----------
    start, end : str
        Date range (YYYY-MM-DD).
    use_live : bool
        If True, attempt to fetch a live front-month series from Yahoo
        and attach it as `spot`. The term-structure panel itself is
        always synthetic in this open-source build.
    tenors : list of int, optional
        Tenors in months.

    Returns
    -------
    FuturesPanel
    """
    panel = generate_synthetic_panel(start=start, end=end, tenors=tenors)
    if use_live:
        spot = fetch_yahoo_front_month(start=start)
        if spot is not None:
            panel.spot = spot.reindex(panel.prices.index).ffill()
            logger.info("Attached live front-month reference (%d obs).", len(spot))
    return panel
