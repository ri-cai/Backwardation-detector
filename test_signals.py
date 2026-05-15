"""Unit tests for the crude term structure toolkit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_panel
from src.signals import (
    backwardation_signal,
    curve_convexity,
    front_back_slope,
    roll_yield,
    zscore,
)
from src.term_structure import fit_nelson_siegel, fit_panel, nelson_siegel
from src.backtest import run_backtest


def make_curve(near_above_far: bool = True) -> pd.DataFrame:
    """Tiny 3-day, 4-tenor panel with a known shape."""
    tenors = [1, 3, 6, 12]
    if near_above_far:
        rows = [[80, 78, 76, 74], [82, 80, 77, 74], [85, 82, 78, 75]]
    else:
        rows = [[70, 72, 75, 78], [71, 73, 76, 80], [72, 75, 78, 82]]
    return pd.DataFrame(
        rows,
        index=pd.bdate_range("2024-01-01", periods=3),
        columns=tenors,
    )


# ------------- signal tests -------------

def test_roll_yield_sign_backwardation():
    panel = make_curve(near_above_far=True)
    ry = roll_yield(panel, near=1, far=3)
    assert (ry > 0).all()


def test_roll_yield_sign_contango():
    panel = make_curve(near_above_far=False)
    ry = roll_yield(panel, near=1, far=3)
    assert (ry < 0).all()


def test_roll_yield_invalid_tenors():
    panel = make_curve()
    with pytest.raises(KeyError):
        roll_yield(panel, near=1, far=99)
    with pytest.raises(ValueError):
        roll_yield(panel, near=6, far=3)


def test_slope_sign_backwardation():
    panel = make_curve(near_above_far=True)
    slope = front_back_slope(panel)
    assert (slope < 0).all()


def test_slope_sign_contango():
    panel = make_curve(near_above_far=False)
    slope = front_back_slope(panel)
    assert (slope > 0).all()


def test_zscore_basic():
    s = pd.Series(np.arange(500, dtype=float))
    z = zscore(s, window=100, min_periods=20)
    # rising series -> latest z should be strongly positive
    assert z.iloc[-1] > 0

    # for a stationary i.i.d. series the rolling z mean is ~0
    rng = np.random.default_rng(0)
    s2 = pd.Series(rng.standard_normal(2000))
    z2 = zscore(s2, window=100, min_periods=20)
    assert abs(z2.dropna().mean()) < 0.3


def test_signal_dataframe_shape():
    panel = make_curve(near_above_far=True)
    big = pd.concat([panel] * 200, ignore_index=True)
    big.index = pd.bdate_range("2020-01-01", periods=len(big))
    big.columns = [1, 3, 6, 12]
    sig = backwardation_signal(
        big, method="roll_yield", near=1, far=3, z_window=60, threshold=1.0
    )
    assert set(sig.columns) == {"raw", "z", "regime", "position"}
    assert len(sig) == len(big)


def test_signal_method_slope():
    panel = generate_synthetic_panel(start="2020-01-01", end="2021-12-31").prices
    sig = backwardation_signal(panel, method="slope", z_window=60)
    assert "raw" in sig.columns


def test_signal_unknown_method():
    panel = make_curve()
    with pytest.raises(ValueError):
        backwardation_signal(panel, method="bogus")


# ------------- NS tests -------------

def test_nelson_siegel_recovers_inputs():
    tenors = np.array([1.0, 3, 6, 12, 24, 36])
    true = nelson_siegel(tenors, beta0=4.2, beta1=-0.3, beta2=0.1, lam=8.0)
    prices = np.exp(true)
    fitted = fit_nelson_siegel(tenors, prices, lam=8.0)
    assert abs(fitted.beta0 - 4.2) < 1e-6
    assert abs(fitted.beta1 - (-0.3)) < 1e-6
    assert abs(fitted.beta2 - 0.1) < 1e-6


def test_nelson_siegel_fits_lam_jointly():
    tenors = np.array([1.0, 3, 6, 12, 24, 36])
    true = nelson_siegel(tenors, beta0=4.2, beta1=-0.4, beta2=0.2, lam=10.0)
    prices = np.exp(true)
    fitted = fit_nelson_siegel(tenors, prices, lam=None)
    # joint fit should reconstruct the curve almost perfectly even if
    # lam itself is weakly identified
    reconstructed = fitted.evaluate(tenors)
    assert np.allclose(reconstructed, true, atol=5e-3)
    assert 1.0 <= fitted.lam <= 60.0


def test_fit_panel_shape():
    panel = generate_synthetic_panel(start="2022-01-01", end="2022-12-31").prices
    factors = fit_panel(panel, lam=6.0)
    assert factors.shape == (len(panel), 3)
    assert list(factors.columns) == ["level", "slope", "curvature"]


def test_curve_convexity_runs():
    panel = make_curve()
    c = curve_convexity(panel)
    assert len(c) == len(panel)


# ------------- backtest -------------

def test_backtest_runs_and_returns_finite_stats():
    panel = generate_synthetic_panel(start="2018-01-01", end="2022-12-31").prices
    sig = backwardation_signal(panel, method="roll_yield", near=1, far=3,
                               z_window=126, threshold=1.0)
    res = run_backtest(panel, sig, front=1, cost_bps=2.0)
    for k in ["sharpe", "cagr", "mdd", "turnover", "time_in_market"]:
        assert k in res.stats
        assert np.isfinite(res.stats[k])
    assert len(res.equity) == len(panel)


def test_backtest_invalid_front():
    panel = generate_synthetic_panel(start="2020-01-01", end="2022-06-30").prices
    sig = backwardation_signal(panel, near=1, far=3, z_window=126)
    with pytest.raises(KeyError):
        run_backtest(panel, sig, front=99)
