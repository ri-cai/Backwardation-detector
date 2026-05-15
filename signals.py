"""Backwardation signal construction.

A futures market is in **backwardation** when near contracts trade above
far contracts (negative slope of the log-price curve in maturity), and in
**contango** in the opposite case.

This module exposes three raw indicators:
    * roll_yield(near, far)  - annualised carry between two tenors
    * front_back_slope       - average forward log-slope across the curve
    * curve_convexity        - second-difference of the log-price curve

plus a rolling z-score and a one-shot `backwardation_signal` builder that
returns raw, z-scored, regime-classified, and position-rule outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def front_back_slope(prices: pd.DataFrame) -> pd.Series:
    """Average forward log-slope across the curve, in 1/month.

    Negative => backwardation (near above far).
    """
    log_p = np.log(prices.values)
    tenors = np.asarray(prices.columns, dtype=float)
    dlog = np.diff(log_p, axis=1)
    dtau = np.diff(tenors)
    slope_per_month = dlog / dtau[None, :]
    return pd.Series(slope_per_month.mean(axis=1), index=prices.index, name="slope")


def roll_yield(prices: pd.DataFrame, near: int = 1, far: int = 2) -> pd.Series:
    """Annualised roll yield between two specific tenors.

        RY = (12 / (T_far - T_near)) * ln(F_near / F_far)

    Positive => backwardation. Mirrors the carry earned by holding the
    near contract and rolling into the far at expiry.
    """
    if near not in prices.columns or far not in prices.columns:
        raise KeyError(
            f"Tenors {near}, {far} not in panel columns {list(prices.columns)}"
        )
    if far <= near:
        raise ValueError("`far` tenor must be greater than `near` tenor.")
    dt = far - near
    ry = (12.0 / dt) * np.log(prices[near] / prices[far])
    return ry.rename(f"roll_yield_{near}_{far}")


def curve_convexity(prices: pd.DataFrame) -> pd.Series:
    """Average second-difference of log price across tenors."""
    log_p = np.log(prices.values)
    second = np.diff(np.diff(log_p, axis=1), axis=1)
    return pd.Series(second.mean(axis=1), index=prices.index, name="convexity")


def zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Rolling z-score with a configurable look-back window."""
    mu = series.rolling(window, min_periods=min_periods).mean()
    sd = series.rolling(window, min_periods=min_periods).std()
    name = series.name if series.name is not None else "x"
    return ((series - mu) / sd).rename(f"{name}_z")


def backwardation_signal(
    prices: pd.DataFrame,
    method: str = "roll_yield",
    near: int = 1,
    far: int = 3,
    z_window: int = 252,
    threshold: float = 1.0,
    regime_band: float = 0.5,
) -> pd.DataFrame:
    """Build a backwardation signal panel.

    Returns a DataFrame with columns:
        raw       : raw indicator (positive => backwardation)
        z         : rolling z-score of the raw indicator
        regime    : {-1, 0, +1} = {contango, neutral, backwardation}
        position  : {0, 1} long-only rule from `threshold`
    """
    if method == "roll_yield":
        raw = roll_yield(prices, near=near, far=far)
    elif method == "slope":
        raw = (-front_back_slope(prices)).rename("neg_slope")
    else:
        raise ValueError(f"Unknown method {method!r}")

    z = zscore(raw, window=z_window)
    regime = pd.Series(0, index=prices.index, dtype=int)
    regime[z > regime_band] = 1
    regime[z < -regime_band] = -1
    position = (z > threshold).astype(int)

    return pd.DataFrame(
        {"raw": raw, "z": z, "regime": regime, "position": position}
    )
