import numpy as np
import pytest
from conftest import long_frame, path

from concordance.costs import ZERO_COST
from concordance.panel import build_panel
from concordance.strategy import LookaheadError, Strategy, weight_schedule
from concordance.validation.dsr import (
    deflated_sharpe,
    effective_independent_trials,
    expected_max_sharpe,
    probabilistic_sharpe,
)
from concordance.validation.lookahead import check_truncation_invariance
from concordance.validation.overfitting import cpcv_select, pbo_cscv
from concordance.validation.placebo import random_selection_placebo


def test_dsr_matches_bailey_lopez_de_prado_2014_example():
    # Paper's worked example: annual SR 2.5 over 5y daily (T=1250), N=100 trials,
    # annualized V[SR]=0.5, skew -3, kurtosis 10 -> DSR ~ 0.9004.
    sr = 2.5 / np.sqrt(250)
    var = 0.5 / 250
    sr0 = expected_max_sharpe(100, var)
    assert sr0 * np.sqrt(250) == pytest.approx(1.7899, abs=1e-3)
    assert probabilistic_sharpe(sr, sr0, 1250, -3.0, 10.0) == pytest.approx(0.9004, abs=1e-3)


def test_dsr_agrees_with_vectorbt_implementation():
    from vectorbt.returns.metrics import deflated_sharpe_ratio
    rng = np.random.default_rng(3)
    r = rng.standard_t(4, size=1500) * 0.01 + 0.0006
    ours = deflated_sharpe(r, n_trials=37, var_sharpe=0.002)
    theirs = deflated_sharpe_ratio(est_sharpe=np.array([ours.sharpe]), var_sharpe=0.002,
                                   nb_trials=37, backtest_horizon=len(r),
                                   skew=np.array([ours.skew]), kurtosis=np.array([ours.kurt]))
    assert ours.dsr == pytest.approx(float(theirs[0]), abs=1e-12)


def test_dsr_extreme_moments_fail_closed():
    # 1 - skew*sr + (k-1)/4*sr^2 <= 0 -> probability 0 (QuantMega inflated z here).
    assert probabilistic_sharpe(0.5, 0.0, 1000, skew=10.0, kurt=1.0) == 0.0


def test_more_trials_raise_the_bar():
    r = np.random.default_rng(0).normal(0.0008, 0.01, 2000)
    assert deflated_sharpe(r, 1).dsr > deflated_sharpe(r, 50).dsr > deflated_sharpe(r, 5000).dsr


def test_effective_trials_limits():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 1))
    assert effective_independent_trials(np.repeat(x, 8, axis=1)) == pytest.approx(1.0)
    assert effective_independent_trials(rng.normal(size=(5000, 8))) == pytest.approx(8, abs=0.2)


def test_pbo_noise_vs_real_edge():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, size=(2000, 20))
    assert 0.25 < pbo_cscv(noise, 10).pbo < 0.75  # pure noise: selection is a coin flip
    edge = noise.copy()
    edge[:, 7] += 0.004  # one configuration truly better everywhere
    assert pbo_cscv(edge, 10).pbo < 0.05


def test_cpcv_picks_real_edge_out_of_sample():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.01, size=(1000, 6))
    r[:, 2] += 0.003
    out = cpcv_select(r, n_folds=8, n_test_folds=2, purged_size=3, embargo_size=3)
    assert set(out.chosen) == {2}
    assert (out.path_sharpes > 0).all()


class _Peeker(Strategy):
    """Cheats: decides using the panel's final close, captured at construction."""

    def __init__(self, panel):
        self.final = panel["close"][-1]

    def decide(self, view):
        best = int(np.nanargmax(self.final))
        w = np.zeros(len(view.assets))
        w[best] = 1.0
        return w


class _Honest(Strategy):
    def __init__(self, panel):
        pass

    def decide(self, view):
        c = view["close"][-1]
        w = np.zeros(len(view.assets))
        w[int(np.nanargmax(c))] = 1.0
        return w


def _two_asset_panel():
    a = path("A", closes=[10, 11, 12, 13, 14, 15, 9, 8, 7, 6, 5, 4])
    b = path("B", closes=[12, 11.5, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5])
    return build_panel(long_frame(a + b))


def test_truncation_invariance_catches_peeking():
    panel = _two_asset_panel()
    check_truncation_invariance(_Honest, panel, cuts=6)
    with pytest.raises(LookaheadError):
        check_truncation_invariance(_Peeker, panel, cuts=6)


def test_placebo_is_reproducible_and_bounded():
    panel = _two_asset_panel()
    w, has = weight_schedule(_Honest(panel), panel)
    fee = ZERO_COST.fee_rates(panel)
    a = random_selection_placebo(panel, w, has, fee, n=30, seed=5)
    b = random_selection_placebo(panel, w, has, fee, n=30, seed=5)
    assert np.array_equal(a.placebo_sharpes, b.placebo_sharpes)
    assert 1 / 31 <= a.p_value <= 1.0
