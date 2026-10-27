"""Nautilus Trader engine: Nautilus's matching engine, cash account and portfolio do the work.

Each session becomes bars per asset on the same bar type:
- an **open bar** at 14:30Z with O=H=L=C=execution price (the total-return open, or the
  final value for a forced liquidation),
- when the schedule has a resting order for the asset that session, two **range bars** at
  14:31Z and 14:32Z that walk the documented intraday path open -> low -> high -> close:
  (O, H=O, L, C=L) then (O=L, H, L, C), and
- a **close bar** at 21:00Z with O=H=L=C=total-return close.

Market orders submitted once every open bar of a session has arrived fill at that open
price (bar execution, L1). Resting exits are real Nautilus STOP_MARKET and LIMIT orders
(one cancels the other), and limit entries real LIMIT buys, so Nautilus's matching engine
decides independently whether and where they fill; unfilled ones are cancelled at the
close. Fees, including market impact, are computed by the fee model below, separately from
the shared planner. Equity at each close is read from Nautilus: account balance plus signed
net positions marked at the last bar price Nautilus delivered. Stock borrow on shorts is
charged at the close through the simulated exchange's account ledger. Money uses a
9-decimal currency and quantities a 6-decimal grid, matching docs/SEMANTICS.md.
"""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.accounting.factory import AccountFactory
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.backtest.modules import SimulationModule, SimulationModuleConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    AssetClass,
    BarAggregation,
    CurrencyType,
    InstrumentClass,
    OmsType,
    OrderSide,
    PriceType,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Currency, Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy as NautilusStrategy

from concordance.costs import ExecTables
from concordance.panel import PRICE_DECIMALS, QTY_DECIMALS, MarketPanel
from concordance.planner import plan_session
from concordance.result import FILL_COLUMNS, AccountWipedOut, BacktestResult
from concordance.strategy import Schedule

VENUE = Venue("SIM")
OPEN_OFFSET = pd.Timedelta(hours=14, minutes=30)
RANGE_OFFSETS = (pd.Timedelta(hours=14, minutes=31), pd.Timedelta(hours=14, minutes=32))
CLOSE_OFFSET = pd.Timedelta(hours=21)
BAR_VOLUME = Quantity(1_000_000_000, QTY_DECIMALS)


def _currency() -> Currency:
    # Not Currency.from_str: for an unknown code it silently invents an 8-decimal currency.
    ccy = Currency("CUSD", PRICE_DECIMALS, 0, "Horizon USD", CurrencyType.FIAT)
    Currency.register(ccy, overwrite=True)
    assert Currency.from_str("CUSD", strict=True).precision == PRICE_DECIMALS
    return ccy


def _impact(qty: float, adv: float, sigma: float, coef: float, ceiling: float) -> float:
    """Square-root impact rate, written out here independently of the shared planner."""
    if coef == 0.0 and ceiling == 0.0:
        return 0.0
    if not adv > 0.0 or not math.isfinite(sigma):
        return ceiling
    return min(coef * sigma * math.sqrt(qty / adv), ceiling)


class _TableFee(FeeModel):
    """Fee = |qty| * price * (base_fee[session, asset] + impact(qty)), looked up from the
    order's timestamp. Forced liquidations pay nothing."""

    def __init__(self, tables: ExecTables, liquidate, ts_to_session, inst_to_asset, ccy):
        super().__init__()
        self.t = tables
        self.liquidate = liquidate
        self.ts_to_session = ts_to_session
        self.inst_to_asset = inst_to_asset
        self.ccy = ccy

    def get_commission(self, order, fill_qty, fill_px, instrument):
        i = self.ts_to_session[order.ts_init]
        j = self.inst_to_asset[instrument.id]
        q = float(fill_qty)
        notional = q * float(fill_px)
        if self.liquidate[i, j]:
            return Money(0.0, self.ccy)
        t = self.t
        rate = float(t.base_fee[i, j]) + _impact(q, float(t.adv[i, j]), float(t.sigma[i, j]),
                                                 float(t.impact_coef), float(t.max_impact))
        return Money(notional * rate, self.ccy)


class _VenueHandle(SimulationModule):
    """Gives the strategy the simulated exchange, so stock borrow is charged through
    Nautilus's own account ledger (`adjust_account`), as its FX rollover module does."""

    def __init__(self):
        super().__init__(SimulationModuleConfig())
        self.venue_exchange = None

    def register_venue(self, exchange):
        super().register_venue(exchange)
        self.venue_exchange = exchange

    def pre_process(self, data):
        pass

    def process(self, ts_now: int):
        pass

    def log_diagnostics(self, logger):
        pass

    def reset(self):
        pass


class _Executor(NautilusStrategy):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.seen = defaultdict(int)
        self.open_px = {}
        self.last_close = np.full(ctx["N"], np.nan)
        self.fills = []
        self.problems = []
        self.sibling = {}  # client order id -> the other half of a stop / take-profit pair
        self.awaiting = set()  # open-phase market orders not yet filled
        self.resting = []  # orders to place once they have

    def on_start(self):
        for bt in self.ctx["bar_types"]:
            self.subscribe_bars(bt)

    def on_bar(self, bar: Bar):
        ts = bar.ts_event
        j = self.ctx["inst_to_asset"][bar.bar_type.instrument_id]
        kind, i = self.ctx["ts_kind"][ts]
        if kind == "open":
            self.open_px[j] = bar.close.as_double()
        else:
            self.last_close[j] = bar.close.as_double()
        self.seen[ts] += 1
        if self.seen[ts] == self.ctx["expected"][ts]:
            if kind == "open":
                self._rebalance(i)
            elif kind == "close":
                self._end_of_day(i)
                self._record(i)

    def _positions(self) -> np.ndarray:
        pos = np.zeros(self.ctx["N"])
        for j, inst in enumerate(self.ctx["instruments"]):
            pos[j] = float(self.portfolio.net_position(inst.id))
        return pos

    def _cash(self) -> float:
        return self.portfolio.account(VENUE).balance_total(self.ctx["ccy"]).as_double()

    def _rebalance(self, i: int):
        c = self.ctx
        exec_price = np.full(c["N"], np.nan)
        for j, px in self.open_px.items():
            exec_price[j] = px
        self.open_px = {}
        sc, t = c["sched"], c["tables"]
        exit_active = np.isfinite(sc.stop[i]) | np.isfinite(sc.take_profit[i])
        (delta, _, order, m, scaled, limit_qty, _, exit_qty, _, pending) = plan_session(
            self._cash(), self._positions(), c["pending"], exec_price, self.last_close.copy(),
            t.base_fee[i], sc.weights[i], bool(sc.has[i]), sc.hold[i], c["liquidate"][i],
            sc.gap_exit[i],
            sc.entry_limit[i], exit_active, t.cap_buy[i], t.cap_sell[i], t.adv[i],
            t.sigma[i], t.impact_coef, t.max_impact, t.min_trade)
        c["pending"] = pending
        if scaled:
            c["scaled"].append(i)
        f = self.order_factory
        # Resting orders are placed once the open fills have settled (so the cash from
        # today's sells is free); Nautilus then decides if and where they trigger.
        resting = []
        for j in np.flatnonzero(exit_qty > 0):
            inst = c["instruments"][j].id
            qty = Quantity(float(exit_qty[j]), QTY_DECIMALS)
            pair = []
            if np.isfinite(sc.stop[i, j]):
                pair.append(f.stop_market(inst, OrderSide.SELL, qty, reduce_only=True,
                                          trigger_price=Price(sc.stop[i, j], PRICE_DECIMALS)))
            if np.isfinite(sc.take_profit[i, j]):
                pair.append(f.limit(inst, OrderSide.SELL, qty, reduce_only=True,
                                    price=Price(sc.take_profit[i, j], PRICE_DECIMALS)))
            if len(pair) == 2:
                self.sibling[pair[0].client_order_id] = pair[1]
                self.sibling[pair[1].client_order_id] = pair[0]
            resting += pair
        for j in np.flatnonzero(limit_qty > 0):
            resting.append(f.limit(c["instruments"][j].id, OrderSide.BUY,
                                   Quantity(float(limit_qty[j]), QTY_DECIMALS),
                                   price=Price(sc.entry_limit[i, j], PRICE_DECIMALS)))
        self.resting = resting
        self.awaiting = set()
        market = []
        for k in range(m):
            j = int(order[k])
            qty = Quantity(abs(float(delta[j])), QTY_DECIMALS)
            side = OrderSide.BUY if delta[j] > 0 else OrderSide.SELL
            market.append(f.market(c["instruments"][j].id, side, qty))
        self.awaiting = {o.client_order_id for o in market}
        for o in market:
            self.submit_order(o)
        if not self.awaiting:
            self._place_resting()

    def _place_resting(self):
        for o in self.resting:
            self.submit_order(o)
        self.resting = []

    def _end_of_day(self, i: int):
        """Cancel resting orders that did not fill, and drop carried quantity for positions
        the schedule closed (also when this engine held nothing to sell)."""
        for order in self.cache.orders_open():
            self.cancel_order(order)
        self.sibling = {}
        closed = np.isfinite(self.ctx["sched"].exit_price[i])
        self.ctx["pending"][closed] = 0.0

    def _record(self, i: int):
        c = self.ctx
        pos = self._positions()
        held = pos != 0.0
        short = pos < 0.0
        rate = float(c["tables"].borrow_rate)
        if rate > 0.0 and short.any():
            fee = rate * float(np.dot(-pos[short], self.last_close[short]))
            c["borrow"][i] = fee
            c["venue"].venue_exchange.adjust_account(Money(-fee, c["ccy"]))
        cash = self._cash()
        # Signed: portfolio.net_exposures reports a short's notional as positive.
        c["cash"][i] = cash
        c["equity"][i] = cash + float(np.dot(pos[held], self.last_close[held]))
        c["positions"][i] = pos
        c["recorded"][i] = True
        if short.any() and not c["equity"][i] > 0:
            self.problems.append(f"equity {c['equity'][i]:.2f} <= 0 at session {i}")
            c["wiped_out"] = True

    def on_order_filled(self, event):
        c = self.ctx
        kind, i = c["ts_kind"][event.ts_event]
        j = c["inst_to_asset"][event.instrument_id]
        other = self.sibling.pop(event.client_order_id, None)
        if other is not None and other.is_open:
            self.cancel_order(other)
        if event.client_order_id in self.awaiting:
            if self.cache.order(event.client_order_id).is_closed:
                self.awaiting.discard(event.client_order_id)
                if not self.awaiting:
                    self._place_resting()
        sign = 1.0 if event.order_side == OrderSide.BUY else -1.0
        self.fills.append((i, 0 if kind == "open" else 1, j, sign * float(event.last_qty),
                           event.last_px.as_double(), event.commission.as_double()))

    def on_order_denied(self, event):
        o = self.cache.order(event.client_order_id)
        self.problems.append(f"order denied: {event.reason} [{o.side_string()} {o.type_string()} "
                             f"{o.quantity} open_orders={len(self.cache.orders_open())}]")

    def on_order_rejected(self, event):
        self.problems.append(f"order rejected: {event.reason}")


def _instrument(j: int, ccy: Currency) -> Instrument:
    sym = Symbol(f"A{j}")
    return Instrument(
        InstrumentId(sym, VENUE), sym, AssetClass.EQUITY, InstrumentClass.SPOT, ccy, False,
        PRICE_DECIMALS, QTY_DECIMALS, Quantity(10.0 ** -QTY_DECIMALS, QTY_DECIMALS),
        Quantity.from_int(1), Decimal(0), Decimal(0), Decimal(0), Decimal(0), 0, 0,
        price_increment=Price(10.0 ** -PRICE_DECIMALS, PRICE_DECIMALS))


def run(panel: MarketPanel, sched: Schedule, tables: ExecTables,
        init_cash: float) -> BacktestResult:
    T, N = panel.shape
    ccy = _currency()
    instruments = [_instrument(j, ccy) for j in range(N)]
    bar_types = [BarType(inst.id, BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
                         AggregationSource.EXTERNAL) for inst in instruments]

    from concordance.strategy import intraday_range

    tr_open, low, high = intraday_range(panel)
    resting = (np.isfinite(sched.stop) | np.isfinite(sched.take_profit)
               | np.isfinite(sched.entry_limit))

    def px(x):
        return Price(float(x), PRICE_DECIMALS)

    ts_kind, expected, bars = {}, {}, []

    def flat_bars(t, row):
        js = np.flatnonzero(np.isfinite(row))
        expected[t] = len(js)
        for j in js:
            p = px(row[j])
            bars.append(Bar(bar_types[j], p, p, p, p, BAR_VOLUME, t, t))

    for i, day in enumerate(panel.sessions):
        t_open = (day + OPEN_OFFSET).tz_localize("UTC").value
        t_a, t_b = ((day + off).tz_localize("UTC").value for off in RANGE_OFFSETS)
        t_close = (day + CLOSE_OFFSET).tz_localize("UTC").value
        ts_kind[t_open] = ("open", i)
        ts_kind[t_a] = ("range", i)
        ts_kind[t_b] = ("range", i)
        ts_kind[t_close] = ("close", i)
        flat_bars(t_open, panel.exec_price[i])
        js = np.flatnonzero(resting[i])
        expected[t_a] = expected[t_b] = len(js)
        path = [(px(tr_open[i, j]), px(low[i, j]), px(high[i, j]), px(panel.tr_close[i, j]))
                for j in js]
        for j, (o, lo, _, _) in zip(js, path, strict=True):
            bars.append(Bar(bar_types[j], o, o, lo, lo, BAR_VOLUME, t_a, t_a))
        for j, (_, lo, hi, cl) in zip(js, path, strict=True):
            bars.append(Bar(bar_types[j], lo, hi, lo, cl, BAR_VOLUME, t_b, t_b))
        flat_bars(t_close, panel.tr_close[i])

    ctx = {
        "N": N, "ccy": ccy, "instruments": instruments, "bar_types": bar_types,
        "inst_to_asset": {inst.id: j for j, inst in enumerate(instruments)},
        "ts_kind": ts_kind, "expected": expected,
        "sched": sched, "tables": tables, "pending": np.zeros(N),
        "liquidate": panel.liquidate, "scaled": [],
        "equity": np.full(T, np.nan), "cash": np.full(T, np.nan),
        "positions": np.zeros((T, N)), "recorded": np.zeros(T, dtype=bool),
        "borrow": np.zeros(T), "venue": _VenueHandle(), "wiped_out": False,
    }

    engine = BacktestEngine(BacktestEngineConfig(
        trader_id=TraderId("CONCORDANCE-001"),
        logging=LoggingConfig(bypass_logging=True),
        risk_engine=RiskEngineConfig(max_order_submit_rate="1000000/00:00:01"),
    ))
    fee_model = _TableFee(tables, panel.liquidate, {t: i for t, (_, i) in ts_kind.items()},
                          ctx["inst_to_asset"], ccy)
    # allow_cash_borrowing registers the venue in a process-wide table that Nautilus never
    # clears, so a second run in the same process would fail: clear it around each run.
    AccountFactory.deregister_cash_borrowing(VENUE.value)
    engine.add_venue(VENUE, OmsType.NETTING, AccountType.CASH,
                     starting_balances=[Money(init_cash, ccy)], base_currency=ccy,
                     fee_model=fee_model, bar_execution=True, modules=[ctx["venue"]],
                     # borrow fees can take a fully invested long/short book's cash a
                     # little below 0; the shared planner still never lets a buy overdraw
                     allow_cash_borrowing=True)
    for inst in instruments:
        engine.add_instrument(inst)
    engine.add_data(bars)
    strat = _Executor(ctx)
    engine.add_strategy(strat)
    try:
        engine.run()
    finally:
        engine.dispose()
        AccountFactory.deregister_cash_borrowing(VENUE.value)

    if ctx["wiped_out"]:
        raise AccountWipedOut("Nautilus: " + strat.problems[-1])
    if strat.problems:
        raise RuntimeError("Nautilus refused orders: " + "; ".join(strat.problems[:5]))
    missing = np.flatnonzero(~ctx["recorded"])
    if len(missing):
        raise RuntimeError(f"no close recorded for sessions {missing[:5].tolist()}")

    fills = pd.DataFrame(strat.fills, columns=FILL_COLUMNS).astype(
        {"session": np.int64, "phase": np.int64, "asset": np.int64})
    return BacktestResult("nautilus", panel.sessions, panel.assets, ctx["equity"], ctx["cash"],
                          ctx["positions"], fills, sorted(ctx["scaled"]), ctx["borrow"])
