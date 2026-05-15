# Crude Oil Term Structure & Backwardation Signals

A Python toolkit for building the WTI crude oil futures term structure, fitting Nelson-Siegel factors, identifying backwardation regimes, and backtesting a simple long-only strategy driven by the regime signal.

![tests](https://github.com/<your-username>/crude-term-structure/actions/workflows/tests.yml/badge.svg)

---

## Why backwardation?

A futures market is in **backwardation** when contracts with nearer expiry trade above contracts with later expiry — the curve slopes downward in maturity. The opposite shape is **contango**.

Empirically, backwardation in crude is associated with positive *roll yield*: a long position in the front contract that rolls forward into the cheaper second contract earns the spread. Identifying these regimes systematically — and tracking how strong they are relative to history — gives a tradable signal as well as a useful diagnostic for inventory pressure and supply tightness in the physical market.

---

## Methodology

### Term structure

Let $F_i^t$ denote the settlement price at time $t$ of the futures contract maturing in $T_i$ months, $i = 1, \dots, N$. The full term structure is the cross-section $\{F_i^t\}_{i=1}^{N}$.

Following Nelson and Siegel (1987), each cross-section of log-prices is fit with three factors plus a decay parameter $\lambda$:

$$
\log F(\tau) \;=\; \beta_0 \;+\; \beta_1 \cdot \frac{1 - e^{-\tau / \lambda}}{\tau / \lambda} \;+\; \beta_2 \cdot \left( \frac{1 - e^{-\tau / \lambda}}{\tau / \lambda} \;-\; e^{-\tau / \lambda} \right)
$$

where $\tau$ is the tenor in months. The factors have natural interpretations:

- $\beta_0$ — **level** (long-run log price)
- $\beta_1$ — **slope** (negative $\beta_1$ ⇒ backwardation in the front of the curve)
- $\beta_2$ — **curvature** (hump in the middle)
- $\lambda$ — decay parameter controlling where the curvature loading peaks

For a fixed $\lambda$, $(\beta_0, \beta_1, \beta_2)$ are recovered by ordinary least squares on the log-price cross-section. The codebase fits the panel with a fixed $\lambda$ in vectorised form for speed, and exposes a joint $\lambda$ optimiser for single-date fits.

### Backwardation signals

Two complementary measures are provided.

**1. Roll yield** between two specific tenors $T_n$ and $T_f$ (annualised):

$$
RY^t \;=\; \frac{12}{T_f - T_n} \cdot \ln\!\left( \frac{F_n^t}{F_f^t} \right)
$$

Positive $RY^t$ ⇒ backwardation. This corresponds directly to the carry earned by holding the near contract.

**2. Average forward log-slope** across the curve:

$$
\beta^t \;=\; \frac{1}{N - 1} \sum_{i=1}^{N - 1} \frac{\log F_{i+1}^t - \log F_i^t}{T_{i+1} - T_i}
$$

Negative $\beta^t$ ⇒ backwardation. The codebase reports $-\beta^t$ when used as a signal so that the convention "positive value = backwardation" holds across both methods.

To compare these across regimes we standardize via a rolling z-score:

$$
Z^t \;=\; \frac{X^t - \mu_w(X^t)}{\sigma_w(X^t)}
$$

where $\mu_w, \sigma_w$ are rolling mean and standard deviation over a window $w$ (default 252 trading days).

A regime label is assigned by

$$
\text{regime}^t \;=\;
\begin{cases}
+1, & Z^t > +0.5 \quad \text{(backwardation)} \\
-1, & Z^t < -0.5 \quad \text{(contango)} \\
\phantom{+}0, & \text{otherwise}
\end{cases}
$$

### Strategy

A simple long-only rule: hold the front-month contract when $Z^t$ on roll yield exceeds an entry threshold $k$ (default $+1.0$); flat otherwise. Round-trip costs of $c$ basis points are charged on changes in position:

$$
r^t_{\text{strat}} \;=\; \mathbb{1}\!\left[ Z^{t-1} > k \right] \cdot r^t_{\text{front}} \;-\; \frac{c}{10{,}000} \cdot \big| \Delta\text{pos}^t \big|
$$

This is intentionally minimal. The aim is to demonstrate that the signal carries information, not to claim production-grade alpha.

---

## Installation

```bash
git clone https://github.com/<your-username>/crude-term-structure.git
cd crude-term-structure
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

```python
from src.data_loader import load_data
from src.signals import backwardation_signal
from src.backtest import run_backtest

panel = load_data(start="2015-01-01")
signal = backwardation_signal(
    panel.prices,
    method="roll_yield",
    near=1, far=3,
    z_window=252,
    threshold=1.0,
)
result = run_backtest(panel.prices, signal, front=1, cost_bps=2.0)
print(result.summary())
```

Or run the full example, which produces a 4-panel diagnostic figure:

```bash
python examples/run_analysis.py
```

---

## Project layout

```
crude-term-structure/
├── src/
│   ├── data_loader.py      # WTI panel loader + synthetic generator
│   ├── term_structure.py   # Nelson-Siegel fitting (single + panel)
│   ├── signals.py          # Roll yield, slope, z-score, regime
│   ├── backtest.py         # Long-only backtest
│   └── visualization.py    # Plotting helpers
├── tests/
│   └── test_signals.py     # pytest suite
├── examples/
│   └── run_analysis.py     # End-to-end demonstration
├── .github/workflows/
│   └── tests.yml           # CI on Python 3.10–3.12
├── requirements.txt
└── README.md
```

---

## Data

Multi-maturity crude futures history requires a paid feed (CME, Bloomberg, Refinitiv). To keep this project fully runnable, `data_loader.py` ships with a calibrated synthetic generator that produces a realistic WTI-like panel:

- Mean-reverting log-level (Ornstein–Uhlenbeck) around $\log 65$
- Slope process that switches between contango and backwardation regimes
- Small AR(1) curvature factor

If `yfinance` is installed, the live front-month series (`CL=F`) is attached to the panel as a reference (`panel.spot`).

To plug in real data, replace the body of `load_data` so that it returns a `FuturesPanel` whose `prices` DataFrame is indexed by date with tenor-in-months columns. Everything downstream (NS fitting, signals, backtest, plotting) is data-source agnostic.

---

## Testing

```bash
pytest -q
```

The suite covers signal sign correctness in known contango / backwardation regimes, Nelson-Siegel parameter recovery, vectorised panel fit shape, z-score behaviour, and end-to-end backtest finiteness.

---

## References

- Nelson, C. R., & Siegel, A. F. (1987). *Parsimonious modeling of yield curves.* Journal of Business, 60(4), 473–489.
- Gorton, G., & Rouwenhorst, K. G. (2006). *Facts and fantasies about commodity futures.* Financial Analysts Journal, 62(2), 47–68.
- Szymanowska, M., de Roon, F., Nijman, T., & Van Den Goorbergh, R. (2014). *An anatomy of commodity futures risk premia.* Journal of Finance, 69(1), 453–482.

---

## License

MIT
