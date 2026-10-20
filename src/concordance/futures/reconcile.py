"""Reference vs Nautilus for the futures margin account.

Integer contracts leave no rounding grid to flip, so positions and every fill's session,
phase, market and quantity must match exactly. Fees must match to a money tick per fill.
Equity may differ only by Nautilus's 1e-9 money rounding per fill plus float
accumulation (1e-9 of equity), the same allowances as docs/SEMANTICS.md "Tolerance".
"""

from __future__ import annotations

import numpy as np

from concordance.reconcile import MONEY_TICK, REL_FLOAT
from concordance.result import BacktestResult


def reconcile_futures(ref: BacktestResult, other: BacktestResult) -> dict:
    problems = []
    if not np.array_equal(ref.positions, other.positions):
        bad = np.argwhere(ref.positions != other.positions)
        problems.append(f"positions differ first at session {int(bad[0][0])}")
    cols = ["session", "phase", "asset", "qty"]
    a, b = ref.fills[cols].to_numpy(), other.fills[cols].to_numpy()
    if a.shape != b.shape or not np.array_equal(a, b):
        problems.append(f"fills differ ({len(a)} vs {len(b)})")
    elif not np.allclose(ref.fills["fee"], other.fills["fee"], rtol=0,
                         atol=MONEY_TICK * 10):
        problems.append("fill fees differ")
    tol = MONEY_TICK * len(ref.fills) + REL_FLOAT * float(np.abs(ref.equity).max())
    diff = float(np.abs(ref.equity - other.equity).max())
    if not diff <= tol:
        problems.append(f"equity differs by {diff:.3e} (allowed {tol:.3e})")
    summary = (f"[{'OK ' if not problems else 'FAIL'}] {other.engine} vs {ref.engine}: "
               f"fills={len(ref.fills)} max|Δequity|={diff:.3e}")
    if problems:
        summary += "\n" + "\n".join(f"       - {p}" for p in problems)
    return {"ok": not problems, "summary": summary, "max_equity_diff": diff}
