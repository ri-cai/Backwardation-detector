"""Term structure modelling utilities.

Nelson-Siegel parameterisation of the log-price curve:

    log F(tau) = beta0
               + beta1 * (1 - exp(-tau/lam)) / (tau/lam)
               + beta2 * [ (1 - exp(-tau/lam)) / (tau/lam)  -  exp(-tau/lam) ]

with `tau` in months. The three loadings have natural interpretations:

    beta0  - level     (long-run log price)
    beta1  - slope     (negative => backwardation)
    beta2  - curvature (hump in the middle of the curve)

Reference
---------
Nelson, C. R., & Siegel, A. F. (1987). *Parsimonious modeling of yield
curves.* Journal of Business, 60(4), 473-489.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class NSParams:
    """Nelson-Siegel parameters: level, slope, curvature, decay."""

    beta0: float
    beta1: float
    beta2: float
    lam: float

    def evaluate(self, tenors: np.ndarray) -> np.ndarray:
        """Evaluate the NS log-price curve at `tenors` (months)."""
        return nelson_siegel(tenors, self.beta0, self.beta1, self.beta2, self.lam)


def _ns_loadings(tau: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_load, curv_load) at given tenors and decay."""
    tau = np.asarray(tau, dtype=float)
    x = tau / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_load = np.where(x == 0, 1.0, (1.0 - np.exp(-x)) / x)
    curv_load = slope_load - np.exp(-x)
    return slope_load, curv_load


def nelson_siegel(
    tau: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float
) -> np.ndarray:
    """Nelson-Siegel log-price curve evaluated at `tau` (months)."""
    slope_load, curv_load = _ns_loadings(tau, lam)
    return beta0 + beta1 * slope_load + beta2 * curv_load


def fit_nelson_siegel(
    tenors: np.ndarray,
    prices: np.ndarray,
    lam: Optional[float] = None,
) -> NSParams:
    """Fit Nelson-Siegel to a single observation of the term structure.

    Parameters
    ----------
    tenors : array-like, shape (N,)
        Tenors in months.
    prices : array-like, shape (N,)
        Settlement prices at those tenors. Internally we fit on log prices.
    lam : float, optional
        If given, fit beta0..beta2 by linear regression at that decay.
        Otherwise jointly optimise lam in [1, 60].

    Returns
    -------
    NSParams
    """
    tenors = np.asarray(tenors, dtype=float)
    prices = np.asarray(prices, dtype=float)
    log_prices = np.log(prices)

    def fit_betas(lam_value: float) -> tuple[np.ndarray, float]:
        slope_load, curv_load = _ns_loadings(tenors, lam_value)
        X = np.column_stack([np.ones_like(tenors), slope_load, curv_load])
        betas, *_ = np.linalg.lstsq(X, log_prices, rcond=None)
        resid = log_prices - X @ betas
        return betas, float(resid @ resid)

    if lam is not None:
        betas, _ = fit_betas(lam)
        return NSParams(beta0=betas[0], beta1=betas[1], beta2=betas[2], lam=lam)

    res = minimize(
        lambda l: fit_betas(l[0])[1],
        x0=np.array([6.0]),
        bounds=[(1.0, 60.0)],
        method="L-BFGS-B",
    )
    lam_opt = float(res.x[0])
    betas, _ = fit_betas(lam_opt)
    return NSParams(beta0=betas[0], beta1=betas[1], beta2=betas[2], lam=lam_opt)


def fit_panel(prices: pd.DataFrame, lam: float = 6.0) -> pd.DataFrame:
    """Fit Nelson-Siegel to every row of a panel with fixed `lam`.

    Returns
    -------
    pd.DataFrame
        Indexed by date with columns ['level', 'slope', 'curvature'].
    """
    tenors = np.asarray(prices.columns, dtype=float)
    log_prices = np.log(prices.values)
    slope_load, curv_load = _ns_loadings(tenors, lam)
    X = np.column_stack([np.ones_like(tenors), slope_load, curv_load])
    # vectorised OLS across rows: betas[t] = (X'X)^{-1} X' y[t]
    XtX_inv_Xt = np.linalg.pinv(X)              # shape (3, N)
    betas = log_prices @ XtX_inv_Xt.T           # shape (T, 3)
    out = pd.DataFrame(
        betas,
        index=prices.index,
        columns=["level", "slope", "curvature"],
    )
    return out
