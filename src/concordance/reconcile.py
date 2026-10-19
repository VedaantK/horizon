"""Cross-engine reconciliation: fills, positions, cash and equity must agree.

Tolerances come from known rounding granularity (see docs/SEMANTICS.md), not from tuning:
money is stored at 1e-9 in Nautilus, so each fill may contribute up to 1e-9 of drift.

That drift can tip an order size that sits on a half step of the 1e-6 quantity grid to the
neighbouring step ("grid flip"). A flip is allowed only as exactly one step, and the money
it explains is added to the cash/equity allowance: price x step plus the fee change, and
afterwards the value of the one-step position difference while it stays open.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from concordance.result import BacktestResult

MONEY_TICK = 1e-9  # per-fill money rounding allowance
REL_FLOAT = 1e-9  # float64 accumulation allowance relative to equity
QTY_TOL = 1e-9  # quantities are on a 1e-6 grid; anything beyond float noise is a mismatch
QTY_STEP = 1e-6  # one grid step: the most a rounding-boundary flip can move a quantity
FEE_SLOPE = 1.5  # d(fee)/d(qty) <= 1.5 x fee/qty for a linear rate plus sqrt market impact
PRICE_REL = 1e-12


@dataclass
class PairCheck:
    engine: str
    against: str
    problems: list[str] = field(default_factory=list)
    max_equity_diff: float = 0.0
    max_cash_diff: float = 0.0
    n_fills: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class ReconciliationReport:
    checks: list[PairCheck]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            status = "OK " if c.ok else "FAIL"
            lines.append(f"[{status}] {c.engine} vs {c.against}: fills={c.n_fills} "
                         f"max|Δequity|={c.max_equity_diff:.3e} max|Δcash|={c.max_cash_diff:.3e}")
            lines.extend(f"       - {p}" for p in c.problems[:20])
        return "\n".join(lines)


def _compare(a: BacktestResult, b: BacktestResult,
             marks: np.ndarray | None = None) -> PairCheck:
    chk = PairCheck(engine=b.engine, against=a.engine, n_fills=len(a.fills))
    sessions = a.sessions

    key = ["session", "phase", "asset"]
    for f, e in ((a.fills, a.engine), (b.fills, b.engine)):
        if f.duplicated(key).any():
            chk.problems.append(f"{e} has two fills for one (session, phase, asset)")
    fa = a.fills.sort_values(key).reset_index(drop=True)
    fb = b.fills.sort_values(key).reset_index(drop=True)
    m = fa.merge(fb, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    only = m[m["_merge"] != "both"]
    for _, r in only.head(10).iterrows():
        side = a.engine if r["_merge"] == "left_only" else b.engine
        chk.problems.append(f"fill only in {side}: {sessions[int(r['session'])].date()} "
                            f"phase={int(r['phase'])} asset={a.assets[int(r['asset'])]}")
    both = m[m["_merge"] == "both"]
    dq = (both["qty_a"] - both["qty_b"]).abs()
    flip = (dq > QTY_TOL) & (dq <= QTY_STEP + QTY_TOL)
    for _, r in both[dq > QTY_STEP + QTY_TOL].head(10).iterrows():
        chk.problems.append(f"qty differs {sessions[int(r['session'])].date()} "
                            f"asset={a.assets[int(r['asset'])]}: {r['qty_a']} vs {r['qty_b']}")
    dp = (both["price_a"] - both["price_b"]).abs() > PRICE_REL * both["price_a"].abs() + 1e-12
    for _, r in both[dp].head(10).iterrows():
        chk.problems.append(f"price differs {sessions[int(r['session'])].date()} "
                            f"asset={a.assets[int(r['asset'])]}: {r['price_a']} vs {r['price_b']}")
    fee_gap = (both["fee_a"] - both["fee_b"]).abs()
    per_share = (both["fee_a"].abs() / both["qty_a"].abs().where(both["qty_a"] != 0)).fillna(0)
    flip_fee = np.where(flip, FEE_SLOPE * per_share * dq, 0.0)
    dfee = fee_gap > MONEY_TICK + REL_FLOAT * both["fee_a"].abs() + flip_fee
    for _, r in both[dfee].head(10).iterrows():
        chk.problems.append(f"fee differs {sessions[int(r['session'])].date()} "
                            f"asset={a.assets[int(r['asset'])]}: {r['fee_a']} vs {r['fee_b']}")

    cum_fills = np.zeros(len(sessions))
    if len(fa):
        counts = np.bincount(fa["session"].to_numpy(), minlength=len(sessions))
        cum_fills = np.cumsum(counts)
    tol = MONEY_TICK * (cum_fills + 1) + REL_FLOAT * np.abs(a.equity)

    # Grid flips: cash they explain, and how far each asset's position may drift.
    flips = both[flip]
    flip_cash = np.zeros(len(sessions))
    flip_steps = np.zeros(a.positions.shape)
    if len(flips):
        s = flips["session"].to_numpy(dtype=np.int64)
        money = (flips["price_a"].abs() * dq[flip] + fee_gap[flip]).to_numpy()
        flip_cash = np.cumsum(np.bincount(s, weights=money, minlength=len(sessions)))
        np.add.at(flip_steps, (s, flips["asset"].to_numpy(dtype=np.int64)), 1.0)
        flip_steps = np.cumsum(flip_steps, axis=0)
    dpos = np.abs(a.positions - b.positions)
    open_value = np.zeros(len(sessions))
    if marks is not None:
        open_value = np.nansum(np.where(dpos > QTY_TOL, dpos * np.abs(marks), 0.0), axis=1)
    tolerances = {"cash": tol + flip_cash, "equity": tol + flip_cash + open_value}

    for label, xa, xb in (("equity", a.equity, b.equity), ("cash", a.cash, b.cash)):
        tol = tolerances[label]
        d = np.abs(xa - xb)
        bad = ~(d <= tol)  # NaN counts as bad
        if label == "equity":
            chk.max_equity_diff = float(np.nanmax(d)) if len(d) else 0.0
        else:
            chk.max_cash_diff = float(np.nanmax(d)) if len(d) else 0.0
        if bad.any():
            i = int(np.argmax(bad))
            chk.problems.append(f"{label} diverges first at {sessions[i].date()}: "
                                f"{xa[i]!r} vs {xb[i]!r} (tol {tol[i]:.2e})")

    pos_bad = dpos > QTY_STEP * flip_steps + QTY_TOL
    if pos_bad.any():
        i, j = np.argwhere(pos_bad)[0]
        chk.problems.append(f"position differs at {sessions[i].date()} asset={a.assets[j]}: "
                            f"{a.positions[i, j]} vs {b.positions[i, j]}")
    if a.scaled_sessions != b.scaled_sessions:
        chk.problems.append("buy scaling happened on different sessions")
    return chk


def valuation_marks(tr_close: np.ndarray) -> np.ndarray:
    """Per-session valuation prices as the engines mark them: last finite tr_close."""
    return pd.DataFrame(tr_close).ffill().to_numpy()


def reconcile(results: dict[str, BacktestResult],
              marks: np.ndarray | None = None) -> ReconciliationReport:
    """Compare every engine with the reference. `marks` (T, N) values open grid-flip
    differences; without it any position difference must be zero-cost to pass equity."""
    names = list(results)
    base = results["reference"] if "reference" in results else results[names[0]]
    checks = [_compare(base, r, marks) for name, r in results.items() if r is not base]
    return ReconciliationReport(checks)


def equity_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({k: v.equity_series for k, v in results.items()})
