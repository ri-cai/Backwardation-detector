# Crude oil backwardation detector

I built this during a Goldman Sachs markets mentorship to get a feel for how the WTI crude futures curve moves between backwardation and contango, and whether that shape carries any tradable information. It fits a Nelson-Siegel curve to the futures term structure each day, turns the slope into a standardised signal, and backtests a minimal long-only rule on top of it.

It's a learning project, not a trading system — the point was to understand the mechanics of term-structure signals, not to claim real alpha.

## The idea

A futures curve is in **backwardation** when near-dated contracts trade above far-dated ones (the curve slopes down in maturity), and in **contango** when it's the other way around.

Backwardation matters because it tends to come with positive *roll yield*: if you're long the front contract and roll into a cheaper second contract, you pick up the spread. So the slope of the curve is both a rough read on physical supply tightness and something you can try to trade.

This project measures that slope two ways, standardises it against its own history, and labels each day as backwardation / contango / neutral.

## How it works

**Term structure (`term_structure.py`).**
Each day's curve of log futures prices is fit with a Nelson-Siegel model:

```
log F(tau) = beta0
           + beta1 * (1 - exp(-tau/lam)) / (tau/lam)
           + beta2 * [ (1 - exp(-tau/lam)) / (tau/lam) - exp(-tau/lam) ]
```

with `tau` the tenor in months. The three loadings mean:

- `beta0` — the level of the curve
- `beta1` — the slope; negative slope is backwardation at the front
- `beta2` — curvature, the hump in the middle
- `lam` — decay, controlling where that hump sits

For a fixed `lam` the model is linear in the betas, so they come straight out of ordinary least squares. `fit_panel` does this across every day at once (vectorised), and there's a joint optimiser over `lam` for single-day fits.

**Signals (`signals.py`).**
Two measures of backwardation, both built so that a positive number always means backwardation:

- *Roll yield* between a near and far tenor — the annualised log ratio of the two prices. This is the carry you'd actually earn holding the near contract.
- *Average forward log-slope* across the whole curve.

Each is turned into a rolling z-score (default 252-day window) so it can be compared against its own recent history, and a regime label is assigned when the z-score clears +/- 0.5.

**Backtest (`backtest.py`).**
A deliberately simple rule: go long the front-month contract when the roll-yield z-score is above an entry threshold, flat otherwise, with round-trip costs charged whenever the position changes. It's meant to show the signal carries information, nothing more.

**Data (`data_loader.py`).**
Real multi-maturity crude history needs a paid feed (CME / Bloomberg / Refinitiv), so to keep the repo runnable out of the box `data_loader.py` generates a synthetic WTI-like panel: a mean-reverting log level, a slope process that flips between contango and backwardation, and a small curvature factor. If `yfinance` is installed it also pulls the live front-month series (`CL=F`) as a reference.

To use real data, swap out the body of `load_data` to return a price panel indexed by date with tenor-in-months columns — everything downstream is agnostic to where the prices came from.

## Files

```
term_structure.py   Nelson-Siegel fitting (single day + whole panel)
signals.py          roll yield, curve slope, z-score, regime labels
backtest.py         long-only backtest with transaction costs
data_loader.py      synthetic WTI panel + optional live front-month
run_analysis.py     end-to-end run / diagnostic plots
test_signals.py     sanity-check tests for the signal logic
```

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy matplotlib yfinance
python run_analysis.py
```

⟨FILL IN: if run_analysis.py takes arguments or writes a figure to a specific path, say so here⟩

To call the pieces directly:

```python
⟨FILL IN: paste the actual import + call you use, e.g.
from data_loader import load_data
from signals import backwardation_signal
from backtest import run_backtest
... exactly as your files expose them⟩
```

## Tests

```
pytest test_signals.py
```

The tests check the signal has the right sign in known contango / backwardation cases, that the Nelson-Siegel fit recovers sensible parameters, and that the backtest runs end to end.

## References

- Nelson & Siegel (1987), *Parsimonious modeling of yield curves*, Journal of Business 60(4).
- Gorton & Rouwenhorst (2006), *Facts and fantasies about commodity futures*, FAJ 62(2).
- Szymanowska et al. (2014), *An anatomy of commodity futures risk premia*, Journal of Finance 69(1).

## License

MIT
