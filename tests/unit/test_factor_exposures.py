"""The factor regression recovers known betas and alpha from constructed returns."""

import numpy as np
import pandas as pd
import pytest

from concordance.report import FACTORS, factor_exposures


def test_recovers_betas_and_alpha():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2005-01-03", periods=2500)
    f = pd.DataFrame({c: rng.normal(0, 0.01, len(dates)) for c in FACTORS})
    f["rf"] = 0.0001
    f.insert(0, "date", dates)
    # 2 bps a day of alpha (5.04% a year), 1.2 market, 0.8 momentum, small noise.
    r = 0.0002 + f["rf"] + 1.2 * f["mktrf"] + 0.8 * f["umd"] + rng.normal(0, 0.002, len(dates))
    out = factor_exposures(pd.Series(r.to_numpy(), index=dates), f)
    assert out["factor_betas"]["mktrf"] == pytest.approx(1.2, abs=0.02)
    assert out["factor_betas"]["umd"] == pytest.approx(0.8, abs=0.02)
    assert abs(out["factor_betas"]["hml"]) < 0.02
    assert out["factor_alpha_annual"] == pytest.approx(0.0504, abs=0.012)
    assert out["factor_alpha_t"] > 3
    assert out["factor_r2"] > 0.9


def test_too_little_data_is_reported_not_guessed():
    dates = pd.bdate_range("2005-01-03", periods=30)
    f = pd.DataFrame({"date": dates, **{c: 0.0 for c in FACTORS}, "rf": 0.0})
    out = factor_exposures(pd.Series(0.001, index=dates), f)
    assert "not enough" in out["factor_model"]
