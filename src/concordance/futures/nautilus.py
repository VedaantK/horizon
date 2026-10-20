"""Nautilus futures engine: Nautilus's margin account, futures instruments and matching do
the P&L; the reference loop is in `futures/engine.py` (docs/SEMANTICS.md, "Futures").

Each market is a Nautilus FuturesContract with the market's multiplier. Bars carry the
Panama-adjusted open and settle, shifted by a constant per market so every price is
positive (a constant shift changes no difference, so no P&L). Market orders fill at the
open bar; the account marks every held market at each bar. Equity at a close is Nautilus's
account balance plus its unrealized P&L. Interest on the previous close's equity and roll
fees are booked through the simulated exchange's account ledger, like stock borrow.
Order sizes come from the shared `target_contracts` rule; Nautilus margin checks are off
because that rule enforces the margin cap.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    AssetClass,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy as NautilusStrategy

from concordance.engines.nautilus_engine import (
    CLOSE_OFFSET,
    OPEN_OFFSET,
    VENUE,
    _currency,
    _VenueHandle,
)
from concordance.futures.engine import DEFAULT_RULES, ROLL_PHASE, MarginRules, target_contracts
from concordance.futures.panel import FuturesData
from concordance.panel import PRICE_DECIMALS, MarketPanel
from concordance.result import FILL_COLUMNS, AccountWipedOut, BacktestResult
from concordance.strategy import Schedule


BAR_VOLUME = Quantity(1_000_000_000, 0)  # whole contracts: size precision 0


class _PerContractFee(FeeModel):
    def __init__(self, fee_per, inst_to_market, ccy):
        super().__init__()
        self.fee_per, self.inst_to_market, self.ccy = fee_per, inst_to_market, ccy

    def get_commission(self, order, fill_qty, fill_px, instrument):
        return Money(float(fill_qty) * self.fee_per[self.inst_to_market[instrument.id]],
                     self.ccy)


class _Executor(NautilusStrategy):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.seen = defaultdict(int)
        self.fills = []
        self.problems = []

    def on_start(self):
        for bt in self.ctx["bar_types"]:
            self.subscribe_bars(bt)

    def on_bar(self, bar: Bar):
        ts = bar.ts_event
        kind, i = self.ctx["ts_kind"][ts]
        self.seen[ts] += 1
        if self.seen[ts] == self.ctx["expected"][ts]:
            if kind == "open":
                self._open(i)
            else:
                self._close(i)

    def _positions(self):
        return np.array([float(self.portfolio.net_position(inst.id))
                         for inst in self.ctx["instruments"]])

    def _equity(self):
        c = self.ctx
        bal = self.portfolio.account(VENUE).balance_total(c["ccy"]).as_double()
        upnl = 0.0
        for inst in c["instruments"]:
            u = self.portfolio.unrealized_pnl(inst.id)
            upnl += u.as_double() if u is not None else 0.0
        return bal + upnl

    def _open(self, i):
        c = self.ctx
        e_prev = c["equity"][i - 1] if i > 0 else c["init_cash"]
        interest = e_prev * c["rf"][i]
        if interest:
            c["venue"].venue_exchange.adjust_account(Money(interest, c["ccy"]))
        sc = c["sched"]
        if not sc.has[i]:
            return
        fd = c["fd"]
        q, ok, scaled = target_contracts(sc.weights[i], e_prev, fd.multiplier,
                                         fd.open_unadj[i], c["margin_rate"], c["rules"])
        if scaled:
            c["scaled"].append(i)
        pos = self._positions()
        for j in np.flatnonzero(ok & (q != pos)):
            delta = q[j] - pos[j]
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            self.submit_order(self.order_factory.market(
                c["instruments"][j].id, side, Quantity(abs(delta), 0)))

    def _close(self, i):
        c = self.ctx
        pos = self._positions()
        fd = c["fd"]
        for j in np.flatnonzero(fd.roll[i] & (pos != 0)):
            fee = 2.0 * abs(pos[j]) * c["fee_per"][j]
            c["venue"].venue_exchange.adjust_account(Money(-fee, c["ccy"]))
            self.fills.append((i, ROLL_PHASE, j, 0.0, float(fd.close_unadj[i, j]), fee))
        eq = self._equity()
        c["equity"][i] = eq
        c["cash"][i] = eq
        c["positions"][i] = pos
        c["recorded"][i] = True
        if not eq > 0:
            self.problems.append(f"equity {eq:.2f} <= 0 at session {i}")
            c["wiped_out"] = True

    def on_order_filled(self, event):
        c = self.ctx
        _, i = c["ts_kind"][event.ts_event]
        j = c["inst_to_market"][event.instrument_id]
        sign = 1.0 if event.order_side == OrderSide.BUY else -1.0
        # Record the contract's own (unadjusted) open, as the reference does.
        self.fills.append((i, 0, j, sign * float(event.last_qty),
                           float(c["fd"].open_unadj[i, j]), event.commission.as_double()))

    def on_order_denied(self, event):
        self.problems.append(f"order denied: {event.reason}")

    def on_order_rejected(self, event):
        self.problems.append(f"order rejected: {event.reason}")


def run(panel: MarketPanel, sched: Schedule, init_cash: float,
        rules: MarginRules = DEFAULT_RULES) -> BacktestResult:
    fd: FuturesData = panel.meta["futures"]
    T, N = panel.shape
    ccy = _currency()
    shift = np.zeros(N)  # make every bar price positive; differences are unchanged
    for j in range(N):
        lo = np.nanmin(np.r_[fd.adj_open[:, j], fd.adj_close[:, j]])
        shift[j] = max(0.0, 1.0 - lo)
    fee_per = np.array([s.commission + s.slippage_ticks * s.tick * s.multiplier
                        for s in fd.specs])
    far = pd.Timestamp("2200-01-01").tz_localize("UTC").value
    instruments = [
        FuturesContract(
            InstrumentId(Symbol(f"F{j}"), VENUE), Symbol(f"F{j}"), AssetClass.INDEX, ccy,
            PRICE_DECIMALS, Price(10.0 ** -PRICE_DECIMALS, PRICE_DECIMALS),
            Quantity(spec.multiplier, 0) if float(spec.multiplier).is_integer()
            else Quantity(spec.multiplier, 6),
            Quantity.from_int(1), spec.root, 0, far, 0, 0,
            margin_init=Decimal(0), margin_maint=Decimal(0),
            maker_fee=Decimal(0), taker_fee=Decimal(0))
        for j, spec in enumerate(fd.specs)
    ]
    bar_types = [BarType(inst.id, BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
                         AggregationSource.EXTERNAL) for inst in instruments]
    ts_kind, expected, bars = {}, {}, []

    def flat(t, row):
        js = np.flatnonzero(np.isfinite(row))
        expected[t] = len(js)
        for j in js:
            p = Price(float(row[j] + shift[j]), PRICE_DECIMALS)
            bars.append(Bar(bar_types[j], p, p, p, p, BAR_VOLUME, t, t))

    for i, day in enumerate(panel.sessions):
        t_open = (day + OPEN_OFFSET).tz_localize("UTC").value
        t_close = (day + CLOSE_OFFSET).tz_localize("UTC").value
        ts_kind[t_open], ts_kind[t_close] = ("open", i), ("close", i)
        flat(t_open, fd.adj_open[i])
        flat(t_close, fd.adj_close[i])
    # A session with no open bar at all still needs its interest and rebalance check.
    for t, n in list(expected.items()):
        if n == 0:
            raise ValueError(f"session {ts_kind[t][1]} has no bars; drop it from the panel")

    ctx = {
        "fd": fd, "sched": sched, "rf": panel.meta["rf"], "rules": rules, "ccy": ccy,
        "init_cash": float(init_cash), "fee_per": fee_per,
        "margin_rate": np.array([s.margin_rate for s in fd.specs]),
        "instruments": instruments, "bar_types": bar_types,
        "inst_to_market": {inst.id: j for j, inst in enumerate(instruments)},
        "ts_kind": ts_kind, "expected": expected, "scaled": [],
        "equity": np.full(T, np.nan), "cash": np.full(T, np.nan),
        "positions": np.zeros((T, N)), "recorded": np.zeros(T, bool),
        "venue": _VenueHandle(), "wiped_out": False,
    }
    engine = BacktestEngine(BacktestEngineConfig(
        trader_id=TraderId("CONCORDANCE-FUT"),
        logging=LoggingConfig(bypass_logging=True),
        risk_engine=RiskEngineConfig(bypass=True),
    ))
    engine.add_venue(VENUE, OmsType.NETTING, AccountType.MARGIN,
                     starting_balances=[Money(init_cash, ccy)], base_currency=ccy,
                     fee_model=_PerContractFee(fee_per, ctx["inst_to_market"], ccy),
                     bar_execution=True, modules=[ctx["venue"]], default_leverage=Decimal(1))
    for inst in instruments:
        engine.add_instrument(inst)
    engine.add_data(bars)
    strat = _Executor(ctx)
    engine.add_strategy(strat)
    try:
        engine.run()
    finally:
        engine.dispose()
    if ctx["wiped_out"]:
        raise AccountWipedOut("Nautilus: " + strat.problems[-1])
    if strat.problems:
        raise RuntimeError("Nautilus refused orders: " + "; ".join(strat.problems[:5]))
    missing = np.flatnonzero(~ctx["recorded"])
    if len(missing):
        raise RuntimeError(f"no close recorded for sessions {missing[:5].tolist()}")
    fills = pd.DataFrame(sorted(strat.fills, key=lambda f: (f[0], f[1], f[2])),
                         columns=FILL_COLUMNS).astype(
        {"session": np.int64, "phase": np.int64, "asset": np.int64})
    return BacktestResult("nautilus", panel.sessions, panel.assets, ctx["equity"],
                          ctx["cash"], ctx["positions"], fills, sorted(ctx["scaled"]),
                          np.zeros(T), meta={"account": "margin"})
