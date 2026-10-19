"""Apple's fiscal 2019 figures, checked by hand against its 10-K (filed 2019-10-31).

Revenue FY2019 $260,174M (10-K). Compustat COGS excludes depreciation: cost of sales
$161,782M - D&A $12,547M = $149,235M. Total assets 2019-09-28: $338,516M. Q4 FY2018
revenue $62,900M. The FY2019 Q4 results were released 2019-10-30 after the close, so the
year is visible from the 2019-10-31 close and not on 2019-10-30.
"""

import numpy as np
import pandas as pd
import pytest

from concordance.data.ingest import default_lake

LAKE = default_lake()
pytestmark = pytest.mark.skipif(not (LAKE / "comp_fundq.parquet").exists(),
                                reason="needs the real lake")
AAPL = 14593


def test_apple_fy2019_visible_only_after_report():
    from concordance.data.crsp import load_sp500_panel
    from concordance.data.fundamentals import attach_fundamentals
    from concordance.strategy import PointInTimeView

    panel = attach_fundamentals(load_sp500_panel("2019-10-01", "2019-11-15", lake=LAKE))
    j = int(np.flatnonzero(np.asarray(panel.meta["permno"]) == AAPL)[0])
    view = PointInTimeView(panel, panel.shape[0] - 1)
    s = list(panel.sessions)
    before, after = s.index(pd.Timestamp("2019-10-30")), s.index(pd.Timestamp("2019-10-31"))
    rev = view.fundamental("revtq_ttm")[:, j]
    assert rev[before] == 62_900 + 84_310 + 58_015 + 53_809  # through Q3 FY2019
    assert rev[after] == 260_174
    assert view.fundamental("cogsq_ttm")[after, j] == 149_235
    assert view.fundamental("atq")[after, j] == 338_516
    assert view.fundamental("revtq")[after, j] == 64_040
