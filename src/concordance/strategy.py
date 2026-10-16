"""Strategy interface, the point-in-time view that prevents look-ahead, and the order schedule.

A strategy sees only a `PointInTimeView` truncated at the decision session. Its decisions
(target weights plus optional stop-loss, take-profit and limit-entry orders) are turned into
a `Schedule` once, and every engine executes that same schedule. See docs/SEMANTICS.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from concordance.panel import PRICE_DECIMALS, MarketPanel

# exit_kind codes (Schedule.exit_kind, PositionState.exit_today)
NO_EXIT, STOP, TAKE_PROFIT = 0, 1, 2


class LookaheadError(RuntimeError):
    pass


@dataclass(frozen=True)
class PositionState:
    """What the strategy's own orders have done, as of the decision close.

    These are the *intended* holdings implied by the schedule. The engines may hold less
    (cash or volume limits), never more. Prices are on today's raw price scale.
    """

    held: np.ndarray  # bool: a position is open
    entry_session: np.ndarray  # int: session index the position was opened, -1 if flat
    entry_price: np.ndarray  # fill price of the opening trade, NaN if flat
    stop_level: np.ndarray  # active stop-loss level, NaN if none
    take_profit_level: np.ndarray  # active take-profit level, NaN if none
    exit_today: np.ndarray  # int8: STOP / TAKE_PROFIT if an exit order filled today


class PointInTimeView:
    """Data known at the close of session `i`: every array is sliced to rows 0..i (read-only)."""

    def __init__(self, panel: MarketPanel, i: int, state: PositionState | None = None):
        self._panel = panel
        self.i = i
        self.date = panel.sessions[i]
        self.sessions = panel.sessions[: i + 1]
        self.assets = panel.assets
        self._state = state

    def _cut(self, arr: np.ndarray) -> np.ndarray:
        out = arr[: self.i + 1].view()
        out.flags.writeable = False
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        if name == "tr_close":
            return self._cut(self._panel.tr_close)
        if name == "member":
            return self._cut(self._panel.member)
        return self._cut(self._panel.fields[name])

    def frame(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self[name], index=self.sessions, columns=self.assets)

    def customer_weights(self, imputed: bool = True) -> np.ndarray:
        """(N, N) read-only revenue shares known at this close: row = supplier, column =
        customer (e.g. 0.35 = that customer bought 35% of the supplier's revenue). Zero where
        no link is visible. `imputed=False` drops links whose share was not disclosed (they
        are set to the 10% disclosure threshold). Needs `data.customer_links: true`."""
        if self._panel.links is None:
            raise LookaheadError("customer links are not loaded for this run: set "
                                 "data.customer_links to true in the request")
        return self._panel.links.weights_at(self.i, imputed)

    @property
    def members(self) -> np.ndarray:
        """Boolean mask of universe members on the decision session."""
        return self._panel.member[self.i].copy()

    @property
    def position(self) -> PositionState:
        if self._state is None:
            n = len(self.assets)
            nan = np.full(n, np.nan)
            return PositionState(np.zeros(n, bool), np.full(n, -1), nan, nan, nan,
                                 np.zeros(n, np.int8))
        return self._state


@dataclass
class Orders:
    """A decision with exit and entry orders. Every field is optional.

    weights          target weights, as a plain return value would be (None = no rebalance)
    stop_loss        {asset: raw price}: sell the whole position if the price falls to it
    take_profit      {asset: raw price}: sell the whole position if the price rises to it
    stop_loss_pct    {asset: fraction}: stop at entry price * (1 - fraction)
    take_profit_pct  {asset: fraction}: take profit at entry price * (1 + fraction)
    entry_limit      {asset: raw price}: open the position (to its target weight) only if the
                     price falls to this level next session; cancelled at that session's close
    hold             assets (ids, or a bool array) whose position the rebalance leaves exactly
                     as it is; their weight must be 0 / NaN. Lets you add or drop names
                     without resizing the others.

    A stop field that is given (`stop_loss` or `stop_loss_pct`) replaces every stop; assets
    missing from it lose their stop. A field left as None changes nothing. Same for take
    profit. Levels may be dicts/Series keyed by asset id or arrays over `view.assets` (NaN =
    no order).
    """

    weights: object = None
    stop_loss: object = None
    take_profit: object = None
    stop_loss_pct: object = None
    take_profit_pct: object = None
    entry_limit: object = None
    hold: object = None


class Strategy(ABC):
    """Override `decide`. Return weights ({asset: w} / Series / array of length N), an
    `Orders`, or None to hold."""

    name: str = "strategy"

    def warmup(self) -> int:
        """Sessions of history required before the first decision."""
        return 0

    @abstractmethod
    def decide(self, view: PointInTimeView):
        ...


def _row(values, panel: MarketPanel, what: str) -> np.ndarray:
    """Any accepted per-asset container -> float array over panel.assets (missing = NaN)."""
    n = len(panel.assets)
    if isinstance(values, pd.Series):
        unknown = values.index.difference(panel.assets)
        if len(unknown):
            raise ValueError(f"{what} for unknown assets: {list(unknown)[:5]}")
        return values.reindex(panel.assets).to_numpy(float)
    if isinstance(values, Mapping):
        return _row(pd.Series(values, dtype=float), panel, what)
    row = np.asarray(values, dtype=float)
    if row.shape != (n,):
        raise ValueError(f"{what} array must have shape ({n},), got {row.shape}")
    return row


def _to_row(decision, panel: MarketPanel) -> np.ndarray:
    row = np.nan_to_num(_row(decision, panel, "weights"))
    if (row < 0).any():
        raise ValueError("negative weights are not supported yet (long-only v1)")
    if row.sum() > 1.0 + 1e-12:
        raise ValueError(f"gross weight {row.sum():.6f} exceeds 1 (no leverage in v1)")
    return row


def _hold_row(values, panel: MarketPanel) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == bool:
        if arr.shape != (len(panel.assets),):
            raise ValueError(f"hold mask must have shape ({len(panel.assets)},)")
        return arr.copy()
    if isinstance(values, Mapping):
        values = [k for k, v in values.items() if v]
    ids = pd.Index(list(values))
    unknown = ids.difference(panel.assets)
    if len(unknown):
        raise ValueError(f"hold for unknown assets: {list(unknown)[:5]}")
    return panel.assets.isin(ids)


def _level_row(values, panel: MarketPanel, what: str) -> np.ndarray:
    row = _row(values, panel, what)
    bad = np.isfinite(row) & (row <= 0)
    if bad.any() or np.isinf(row).any():
        raise ValueError(f"{what} levels must be positive and finite (NaN = none)")
    return row


def _pct_row(values, panel: MarketPanel, what: str, below_one: bool) -> np.ndarray:
    row = _row(values, panel, what)
    fin = np.isfinite(row)
    bad = fin & ((row <= 0) | ((row >= 1) if below_one else False))
    if bad.any() or np.isinf(row).any():
        rng = "in (0, 1)" if below_one else "> 0"
        raise ValueError(f"{what} must be a fraction {rng} (NaN = none)")
    return row


def _round(x):
    return np.round(x, PRICE_DECIMALS)


def intraday_range(panel: MarketPanel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(open, low, high) of each session on the total-return scale, NaN when untradable.

    The intraday leg uses raw same-day ratios (as tr_open does). The range is the envelope
    of open, low, high and close, so it always contains the open and the close even when a
    vendor high/low disagrees with them (5 rows in the 1990-2025 CRSP data).
    """
    tradable = np.isfinite(panel.exec_price) & ~panel.liquidate
    tr_open = np.where(tradable, panel.exec_price, np.nan)
    close = panel.fields["close"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = panel.tr_close / close
        lo = _round(ratio * panel.fields["low"]) if "low" in panel.fields else tr_open
        hi = _round(ratio * panel.fields["high"]) if "high" in panel.fields else tr_open
    low = np.fmin(np.fmin(lo, tr_open), panel.tr_close)
    high = np.fmax(np.fmax(hi, tr_open), panel.tr_close)
    return tr_open, np.where(tradable, low, np.nan), np.where(tradable, high, np.nan)


@dataclass
class Schedule:
    """Everything the engines execute, aligned to the **execution** session.

    weights/has      row i+1 holds the target decided at the close of session i
    hold             with `has`: assets the rebalance leaves untouched
    gap_exit         an exit level was already crossed at the open: sell at the open
    stop/take_profit resting exit levels during session i (TR scale), for positions held
                     after the open
    entry_limit      resting limit-buy level during session i (TR scale)
    exit_price       fill price of a resting exit that triggered in session i (NaN if none)
    exit_kind        STOP / TAKE_PROFIT for gap and resting exits
    limit_filled     the resting limit buy was reached in session i
    """

    weights: np.ndarray
    has: np.ndarray
    hold: np.ndarray
    gap_exit: np.ndarray
    stop: np.ndarray
    take_profit: np.ndarray
    entry_limit: np.ndarray
    exit_price: np.ndarray
    exit_kind: np.ndarray
    limit_filled: np.ndarray
    notes: dict = field(default_factory=dict)

    @classmethod
    def from_weights(cls, weights: np.ndarray, has: np.ndarray) -> Schedule:
        T, N = weights.shape
        nan = np.full((T, N), np.nan)
        return cls(weights, has, np.zeros((T, N), bool), np.zeros((T, N), bool), nan,
                   nan.copy(), nan.copy(), nan.copy(), np.zeros((T, N), np.int8),
                   np.zeros((T, N), bool))

    @property
    def uses_orders(self) -> bool:
        return bool(self.gap_exit.any() or np.isfinite(self.stop).any()
                    or np.isfinite(self.take_profit).any()
                    or np.isfinite(self.entry_limit).any())

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ("weights", "has", "hold", "gap_exit", "stop",
                                              "take_profit", "entry_limit", "exit_price",
                                              "exit_kind", "limit_filled")}


def build_schedule(strategy: Strategy, panel: MarketPanel) -> Schedule:
    """Run the strategy session by session and resolve its orders against prices.

    Which orders trigger depends only on prices and the strategy's own levels, never on
    cash, so it is computed once here. Order sizes are left to the engines.
    """
    T, N = panel.shape
    tr_open, low, high = intraday_range(panel)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = pd.DataFrame(panel.tr_close / panel.fields["close"]).ffill().to_numpy()

    s = Schedule.from_weights(np.zeros((T, N)), np.zeros(T, dtype=bool))
    notes = {"stops_at_or_above_close": 0, "take_profits_at_or_below_close": 0}
    held = np.zeros(N, bool)
    entry_tr = np.full(N, np.nan)
    entry_sess = np.full(N, -1)
    stop = np.full(N, np.nan)
    tp = np.full(N, np.nan)
    stop_pct = np.full(N, np.nan)
    tp_pct = np.full(N, np.nan)
    limit_next = np.full(N, np.nan)

    def resolve_pct(mask):
        sp = mask & np.isfinite(stop_pct)
        stop[sp] = _round(entry_tr[sp] * (1 - stop_pct[sp]))
        stop_pct[sp] = np.nan
        tq = mask & np.isfinite(tp_pct)
        tp[tq] = _round(entry_tr[tq] * (1 + tp_pct[tq]))
        tp_pct[tq] = np.nan

    for i in range(T):
        # ---- session i: open -----------------------------------------------------------
        liq = panel.liquidate[i]
        held[liq] = False
        trad = np.isfinite(tr_open[i])
        want = np.where(s.hold[i], held, s.weights[i] > 0) if s.has[i] else held.copy()
        want = np.where(trad, want, held)
        lim = np.where(trad, limit_next, np.nan)
        limit_next = np.full(N, np.nan)
        with np.errstate(invalid="ignore"):
            resting = np.isfinite(lim) & ~(tr_open[i] <= lim)  # marketable limit = open buy
        enter = want & ~held & ~resting & trad
        entry_tr[enter] = tr_open[i][enter]
        entry_sess[enter] = i
        resolve_pct(enter)
        after = np.where(trad, want & ~resting, held)
        with np.errstate(invalid="ignore"):
            gap_stop = after & trad & (tr_open[i] <= stop)
            gap_tp = after & trad & ~gap_stop & (tr_open[i] >= tp)
        gap = gap_stop | gap_tp
        s.gap_exit[i] = gap
        s.exit_kind[i][gap_stop] = STOP
        s.exit_kind[i][gap_tp] = TAKE_PROFIT
        after &= ~gap

        # ---- session i: intraday, path open -> low -> high -> close -------------------
        active = after & trad & (np.isfinite(stop) | np.isfinite(tp))
        s.stop[i] = np.where(active, stop, np.nan)
        s.take_profit[i] = np.where(active, tp, np.nan)
        with np.errstate(invalid="ignore"):
            hit_stop = active & (low[i] <= stop)
            hit_tp = active & ~hit_stop & (high[i] >= tp)
        s.exit_price[i] = np.where(hit_stop, stop, np.where(hit_tp, tp, np.nan))
        s.exit_kind[i][hit_stop] = STOP
        s.exit_kind[i][hit_tp] = TAKE_PROFIT
        after &= ~(hit_stop | hit_tp)

        s.entry_limit[i] = np.where(resting, lim, np.nan)
        with np.errstate(invalid="ignore"):
            filled = resting & (low[i] <= lim)
        s.limit_filled[i] = filled
        after |= filled
        entry_tr[filled] = lim[filled]
        entry_sess[filled] = i
        resolve_pct(filled)  # active from the next session

        held = after
        flat = ~held
        for arr in (entry_tr, stop, tp, stop_pct, tp_pct):
            arr[flat] = np.nan
        entry_sess[flat] = -1

        # ---- session i: close -> decision for session i+1 ------------------------------
        if i < strategy.warmup() or i >= T - 1:
            continue
        r = ratio[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            state = PositionState(
                held=held.copy(), entry_session=entry_sess.copy(),
                entry_price=entry_tr / r, stop_level=stop / r, take_profit_level=tp / r,
                exit_today=s.exit_kind[i].copy())
        for a in state.__dict__.values():
            a.flags.writeable = False
        decision = strategy.decide(PointInTimeView(panel, i, state))
        if decision is None:
            continue
        if not isinstance(decision, Orders):
            decision = Orders(weights=decision)
        if decision.weights is not None:
            s.weights[i + 1] = _to_row(decision.weights, panel)
            s.has[i + 1] = True
            if decision.hold is not None:
                h = _hold_row(decision.hold, panel)
                if (s.weights[i + 1][h] > 0).any():
                    raise ValueError("a held-as-is asset (hold) must have weight 0 / NaN")
                s.hold[i + 1] = h
        elif decision.hold is not None:
            raise ValueError("hold only applies to a rebalance: give weights too")
        _apply_levels(decision, panel, i, r, held, entry_tr, stop, tp, stop_pct, tp_pct,
                      notes)
        if decision.entry_limit is not None:
            lv = _level_row(decision.entry_limit, panel, "entry_limit")
            fin = np.isfinite(lv)
            if fin.any():
                if not s.has[i + 1] or (s.weights[i + 1][fin] <= 0).any():
                    raise ValueError("entry_limit needs a positive target weight in the same "
                                     "Orders for every asset it names")
                if held[fin].any():
                    raise ValueError("entry_limit is only for opening a position; asset "
                                     f"{panel.assets[np.flatnonzero(fin & held)[0]]} is held")
                limit_next = np.where(fin, _round(lv * r), np.nan)
                if np.isnan(limit_next[fin]).any():
                    raise ValueError("entry_limit on an asset with no price yet")
    s.notes = notes
    return s


def _apply_levels(d: Orders, panel, i, r, held, entry_tr, stop, tp, stop_pct, tp_pct, notes):
    close = panel.fields["close"][i]
    for raw_f, pct_f, lvl, pct, below, key in (
            ("stop_loss", "stop_loss_pct", stop, stop_pct, True, "stops_at_or_above_close"),
            ("take_profit", "take_profit_pct", tp, tp_pct, False,
             "take_profits_at_or_below_close")):
        raw_v, pct_v = getattr(d, raw_f), getattr(d, pct_f)
        if raw_v is None and pct_v is None:
            continue
        n = len(panel.assets)
        raw = _level_row(raw_v, panel, raw_f) if raw_v is not None else np.full(n, np.nan)
        frac = (_pct_row(pct_v, panel, pct_f, below_one=below) if pct_v is not None
                else np.full(n, np.nan))
        both = np.isfinite(raw) & np.isfinite(frac)
        if both.any():
            raise ValueError(f"{raw_f} and {pct_f} both set for {panel.assets[both][0]}")
        fin = np.isfinite(raw)
        with np.errstate(invalid="ignore"):
            wrong = fin & ((raw >= close) if below else (raw <= close))
        notes[key] += int(wrong.sum())
        lvl[:] = _round(raw * r)
        if np.isnan(lvl[fin]).any():
            raise ValueError(f"{raw_f} on an asset with no price yet")
        pct[:] = np.nan
        now = np.isfinite(frac) & held
        lvl[now] = _round(entry_tr[now] * ((1 - frac[now]) if below else (1 + frac[now])))
        later = np.isfinite(frac) & ~held
        pct[later] = frac[later]


def weight_schedule(strategy: Strategy, panel: MarketPanel) -> tuple[np.ndarray, np.ndarray]:
    """(weights, has) of `build_schedule`, for callers that only need the targets."""
    s = build_schedule(strategy, panel)
    return s.weights, s.has
