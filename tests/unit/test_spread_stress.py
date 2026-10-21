import numpy as np


def test_stress_doubles_quoted_spread_as_well_as_fallback():
    from types import SimpleNamespace

    from concordance.costs import CostModel

    class Panel(SimpleNamespace):
        def __getitem__(self, key):
            return self.fields[key]

    panel = Panel(shape=(3, 1), liquidate=np.zeros((3, 1), dtype=bool),
                  fields={"bid": np.array([[99.], [98.], [99.]]),
                          "ask": np.array([[101.], [102.], [101.]])})
    base = CostModel(commission_bps=1).fee_rates(panel)
    stressed = CostModel(commission_bps=2, spread_multiplier=2).fee_rates(panel)
    np.testing.assert_allclose(stressed, 2 * base)

