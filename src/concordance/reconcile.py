"""Cross-engine reconciliation: fills, positions, cash and equity must agree.

Tolerances come from known rounding granularity (see docs/SEMANTICS.md), not from tuning:
money is stored at 1e-9 in Nautilus, so each fill may contribute up to 1e-9 of drift.

That drift can tip an order size that sits on a half step of the 1e-6 quantity grid to the
neighbouring step ("grid flip"). A fill may therefore differ by one step, plus:
- the position gap the asset already had (closing a position closes what each engine
  holds),
- at the open, `|position after| x G_V / V`: targets are w x V / price, so a gap G_V in the
  pre-trade value V (the carried cash gap plus open position gaps at their marks) moves
  every target in proportion, and
- on a session whose buys were scaled to fit cash, one more step (the scaled size is
  rounded down) plus `|qty| x G / B`: the scale factor is cash after sells over buy cost B,
  so a gap G in either moves every scaled buy in proportion. G = the cash gap carried into
  the open + the money of that session's differing sells + one step of every buy at its
  price and fee (each pre-scaling size may itself have flipped).
The money every explained difference accounts for is added to the cash/equity allowance:
price x |qty gap| plus the fee change, and the value of open position gaps.
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
    bad_sessions: list[int] = field(default_factory=list)  # session index of each problem

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def first_bad_session(self) -> int | None:
        return min(self.bad_sessions) if self.bad_sessions else None

    def flag(self, session, message: str) -> None:
        self.problems.append(message)
        if session is not None:
            self.bad_sessions.append(int(session))


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
        chk.flag(r["session"], f"fill only in {side}: {sessions[int(r['session'])].date()} "
                            f"phase={int(r['phase'])} asset={a.assets[int(r['asset'])]}")
    both = m[m["_merge"] == "both"]
    dq = (both["qty_a"] - both["qty_b"]).abs()
    fee_gap = (both["fee_a"] - both["fee_b"]).abs()
    allowed = _qty_allowance(a, b, both, dq, fee_gap, marks)
    flip = (dq > QTY_TOL) & (dq <= allowed + QTY_TOL)
    for _, r in both[dq > allowed + QTY_TOL].head(10).iterrows():
        chk.flag(r["session"], f"qty differs {sessions[int(r['session'])].date()} "
                            f"asset={a.assets[int(r['asset'])]}: {r['qty_a']} vs {r['qty_b']}")
    dp = (both["price_a"] - both["price_b"]).abs() > PRICE_REL * both["price_a"].abs() + 1e-12
    for _, r in both[dp].head(10).iterrows():
        chk.flag(r["session"], f"price differs {sessions[int(r['session'])].date()} "
                            f"asset={a.assets[int(r['asset'])]}: {r['price_a']} vs {r['price_b']}")
    per_share = (both["fee_a"].abs() / both["qty_a"].abs().where(both["qty_a"] != 0)).fillna(0)
    flip_fee = np.where(flip, FEE_SLOPE * per_share * dq, 0.0)
    dfee = fee_gap > MONEY_TICK + REL_FLOAT * both["fee_a"].abs() + flip_fee
    for _, r in both[dfee].head(10).iterrows():
        chk.flag(r["session"], f"fee differs {sessions[int(r['session'])].date()} "
                            f"asset={a.assets[int(r['asset'])]}: {r['fee_a']} vs {r['fee_b']}")

    cum_fills = np.zeros(len(sessions))
    if len(fa):
        counts = np.bincount(fa["session"].to_numpy(), minlength=len(sessions))
        cum_fills = np.cumsum(counts)
    if a.borrow is not None:  # each borrow charge is one more rounded money amount
        cum_fills = cum_fills + np.cumsum(a.borrow > 0)
    tol = MONEY_TICK * (cum_fills + 1) + REL_FLOAT * np.abs(a.equity)

    # Explained quantity gaps: cash they account for, and how far each position may drift.
    flips = both[flip]
    flip_cash = np.zeros(len(sessions))
    pos_allow = np.zeros(a.positions.shape)
    if len(flips):
        s = flips["session"].to_numpy(dtype=np.int64)
        money = (flips["price_a"].abs() * dq[flip] + fee_gap[flip]).to_numpy()
        flip_cash = np.cumsum(np.bincount(s, weights=money, minlength=len(sessions)))
        np.add.at(pos_allow, (s, flips["asset"].to_numpy(dtype=np.int64)), dq[flip].to_numpy())
        pos_allow = np.cumsum(pos_allow, axis=0)
    dpos = np.abs(a.positions - b.positions)
    open_value = np.zeros(len(sessions))
    if marks is not None:
        open_value = np.nansum(np.where(dpos > QTY_TOL, dpos * np.abs(marks), 0.0), axis=1)
        if a.borrow is not None and a.borrow.any():
            # a gap in a short position also changes that night's borrow fee
            short_value = np.nansum(np.where(a.positions < 0, -a.positions * marks, 0.0),
                                    axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rate = float(np.nanmax(np.where(short_value > 0, a.borrow / short_value,
                                                np.nan)))
            flip_cash = flip_cash + rate * np.cumsum(open_value)
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
            chk.flag(i, f"{label} diverges first at {sessions[i].date()}: "
                                f"{xa[i]!r} vs {xb[i]!r} (tol {tol[i]:.2e})")

    pos_bad = dpos > pos_allow + QTY_TOL
    if pos_bad.any():
        i, j = np.argwhere(pos_bad)[0]
        chk.flag(i, f"position differs at {sessions[i].date()} asset={a.assets[j]}: "
                            f"{a.positions[i, j]} vs {b.positions[i, j]}")
    if a.scaled_sessions != b.scaled_sessions:
        differ = set(a.scaled_sessions) ^ set(b.scaled_sessions)
        chk.flag(min(differ) if differ else None, "buy scaling happened on different sessions")
    return chk


def _qty_allowance(a: BacktestResult, b: BacktestResult, both: pd.DataFrame,
                   dq: pd.Series, fee_gap: pd.Series,
                   marks: np.ndarray | None = None) -> np.ndarray:
    """Largest quantity gap each matched fill may have (see the module docstring). Uses the
    gaps actually observed before the fill; those are checked in their own right."""
    T = len(a.sessions)
    s = both["session"].to_numpy(dtype=np.int64)
    j = both["asset"].to_numpy(dtype=np.int64)
    prev = np.maximum(s - 1, 0)
    pos_gap = np.where(s > 0, np.abs(a.positions[prev, j] - b.positions[prev, j]), 0.0)
    allowed = QTY_STEP + pos_gap
    # Value gap carried into each open: cash gap + open position gaps at their marks
    # (without marks, the observed equity gap).
    gap_v = np.zeros(T)
    if marks is not None:
        held_gap = np.nansum(np.abs(a.positions - b.positions) * np.abs(marks), axis=1)
        gap_v[1:] = np.abs(a.cash[:-1] - b.cash[:-1]) + held_gap[:-1]
    else:
        gap_v[1:] = np.abs(a.equity[:-1] - b.equity[:-1])
    value = np.ones(T)
    value[1:] = np.abs(a.equity[:-1])
    at_open = both["phase"].to_numpy() == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        target_shift = np.abs(a.positions[s, j]) * gap_v[s] / value[s]
    allowed = allowed + np.where(at_open & (s > 0), np.nan_to_num(target_shift), 0.0)
    scaled = np.zeros(T, dtype=bool)
    scaled[list(a.scaled_sessions)] = True
    buy = both["qty_a"].to_numpy() > 0
    in_scaled = scaled[s] & buy
    if in_scaled.any():
        buys = a.fills[a.fills["qty"] > 0]
        cost = np.bincount(buys["session"].to_numpy(dtype=np.int64),
                           weights=(buys["qty"] * buys["price"] + buys["fee"]).to_numpy(),
                           minlength=T)
        gap_in = np.zeros(T)
        gap_in[1:] = np.abs(a.cash[:-1] - b.cash[:-1])
        sell_gap = (~buy) & (dq.to_numpy() > QTY_TOL)
        sell_money = np.bincount(s[sell_gap], weights=(both["price_a"].abs() * dq + fee_gap)
                                 .to_numpy()[sell_gap], minlength=T)
        # Buy cost B can differ too: before scaling, each buy's size may sit one step apart
        # (a flip), so B moves by up to price x step (plus its fee) per buy.
        rate = (buys["fee"] / (buys["qty"] * buys["price"])).fillna(0).to_numpy()
        cost_gap = np.bincount(buys["session"].to_numpy(dtype=np.int64),
                               weights=(buys["price"].to_numpy() * QTY_STEP * (1 + rate)),
                               minlength=T)
        g = gap_in[s] + sell_money[s] + cost_gap[s]
        with np.errstate(invalid="ignore", divide="ignore"):
            prop = np.abs(both["qty_a"].to_numpy()) * g / cost[s]
        # one more step: the scaled size is rounded down to the grid (a floor can flip too)
        allowed = allowed + np.where(in_scaled, QTY_STEP + np.nan_to_num(prop), 0.0)
    return allowed


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
