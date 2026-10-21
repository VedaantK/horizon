import numpy as np
import pandas as pd

from concordance.validation.ledger import TrialLedger


def test_same_name_and_parameters_preserve_different_source_and_cost_trials(tmp_path):
    ledger = TrialLedger(tmp_path, "shared")
    returns = pd.Series(
        np.random.default_rng(1).normal(0, 0.01, 120),
        index=pd.bdate_range("2019-01-02", periods=120),
    )
    first = ledger.record("same", {}, returns, context={"source": "a", "cost": 1})
    second = ledger.record("same", {}, returns, context={"source": "b", "cost": 1})
    third = ledger.record("same", {}, returns / 2, context={"source": "a", "cost": 2})
    assert len({first, second, third}) == 3
    assert ledger.n_trials() == 3
    np.testing.assert_array_equal(
        pd.read_parquet(ledger.dir / (first + ".parquet"))["ret"], returns
    )
    assert ledger.record("same", {}, returns, context={"source": "a", "cost": 1}) == first
    assert ledger.n_trials() == 3  # exact reproductions do not invent new designs


def test_disjoint_windows_do_not_reset_effective_search_to_one(tmp_path):
    ledger = TrialLedger(tmp_path, "shared")
    for year in (2017, 2023):
        returns = pd.Series(
            np.random.default_rng(year).normal(0, 0.01, 120),
            index=pd.bdate_range(f"{year}-01-02", periods=120),
        )
        ledger.record("same", {}, returns, context={"year": year})
    assert ledger.n_effective() == 2
