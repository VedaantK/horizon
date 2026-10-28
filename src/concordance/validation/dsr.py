"""Probabilistic and Deflated Sharpe Ratio (Bailey & López de Prado 2012, 2014).

Ported from QuantMega `validation/deflated_sharpe.py`. Two fixes:
- a non-positive variance term (extreme skew/kurtosis) now returns DSR = 0, not a z-score
  divided by sqrt(1e-6), which inflated it instead of widening the error;
- database coupling removed: trials come from the ledger or the caller.

Everything here is per-period (e.g. daily); annualize only for display.
Kurtosis is non-excess (3 for normal returns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: float, var_sharpe: float, mean_sharpe: float = 0.0) -> float:
    """E[max SR] of `n_trials` independent trials with Sharpe variance `var_sharpe`."""
    n = float(n_trials)
    if n < 1:
        raise ValueError("n_trials must be >= 1")
    if n == 1:
        return mean_sharpe
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    # The extreme-value approximation dips below the mean for n just above 1 (-0.48 sd at
    # n = 1.05), which would make a near-duplicate trial *lower* the bar. The best of
    # n >= 1 trials is never expected below one trial's mean.
    return float(mean_sharpe + np.sqrt(var_sharpe)
                 * max((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2, 0.0))


def probabilistic_sharpe(sr: float, sr_benchmark: float, T: int, skew: float,
                         kurt: float) -> float:
    """P(true SR > sr_benchmark) given an observed per-period `sr` over T observations."""
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if not np.isfinite(var_term) or var_term <= 0:
        return 0.0
    z = (sr - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(var_term)
    return float(stats.norm.cdf(z))


@dataclass(frozen=True)
class DSRResult:
    dsr: float
    sharpe: float  # per period
    sr0: float  # noise ceiling, per period
    n_trials: float
    T: int
    skew: float
    kurt: float

    def annualized(self, periods: int = 252) -> dict:
        a = np.sqrt(periods)
        return {"dsr": self.dsr, "sharpe": self.sharpe * a, "noise_ceiling": self.sr0 * a,
                "n_trials": self.n_trials}


def deflated_sharpe(returns, n_trials: float, var_sharpe: float | None = None) -> DSRResult:
    """DSR of a return series.

    `var_sharpe` is the cross-trial variance of per-period Sharpe estimates. If omitted, the
    estimation-noise variance under the null, 1/(T-1), is used (QuantMega's default: the
    raw dispersion of real trials mixes in genuine quality differences).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 3:
        raise ValueError("need at least 3 observations")
    sd = r.std(ddof=1)
    if sd == 0:
        return DSRResult(0.0, 0.0, np.nan, n_trials, T, np.nan, np.nan)
    sr = float(r.mean() / sd)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    v = 1.0 / (T - 1) if var_sharpe is None else float(var_sharpe)
    sr0 = expected_max_sharpe(n_trials, v)
    return DSRResult(probabilistic_sharpe(sr, sr0, T, skew, kurt), sr, sr0, n_trials, T,
                     skew, kurt)


DSR_MIN = 0.95  # Bailey & López de Prado's 95% confidence that the true Sharpe beats the noise


def dsr_check(result: DSRResult, threshold: float = DSR_MIN, periods: int = 252) -> dict:
    """Pass/fail verdict on a DSR: does the Sharpe clear the best of `n_trials` noise trials
    with `threshold` confidence? A verdict, not a run failure: metrics stay valid either way."""
    a = result.annualized(periods)
    passed = bool(result.dsr >= threshold)
    reason = (f"DSR {result.dsr:.3f} {'>=' if passed else '<'} {threshold}: annual Sharpe "
              f"{a['sharpe']:.2f} vs a noise ceiling of {a['noise_ceiling']:.2f} from "
              f"{result.n_trials:.1f} effective trials")
    return {"passed": passed, "threshold": threshold, "dsr": result.dsr, "reason": reason}


def effective_independent_trials(trial_returns) -> float:
    """Participation ratio (Σλ)²/Σλ² of the trials' return correlation matrix.

    Identical trials -> 1, independent trials -> N. Carried over unchanged from QuantMega
    (no library provides it). Rows with any non-finite value and zero-variance columns are
    dropped.
    """
    m = np.asarray(trial_returns, dtype=float)
    if m.ndim != 2 or m.size == 0:
        return 0.0
    m = m[np.isfinite(m).all(axis=1)]
    if m.shape[0] < 2:
        return 0.0
    m = m[:, m.std(axis=0, ddof=1) > 1e-15]
    n = m.shape[1]
    if n <= 1:
        return float(n)
    corr = np.nan_to_num(np.corrcoef(m, rowvar=False))
    lam = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    ss = float(np.sum(lam**2))
    if ss <= 0:
        return 1.0
    return float(np.clip(np.sum(lam) ** 2 / ss, 1.0, n))
