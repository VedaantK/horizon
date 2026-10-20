"""What Concordance can test, as data. `concordance capabilities` prints it as JSON.

A coordinator (Colloquy) builds its instructions to strategy authors from this instead of
keeping its own copy. Limits are read from the code that enforces them; `not_supported`
is the list of requests to turn away rather than approximate.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from concordance.data.fundamentals import VIEW_FIELDS

SCHEMA_VERSION = 1

# Order fields that only apply to long positions (strategy.build_schedule rejects the rest).
LONG_ONLY_ORDERS = ("stop_loss", "take_profit", "stop_loss_pct", "take_profit_pct",
                    "entry_limit")

NOT_SUPPORTED = [
    "intraday, minute or tick data (decisions use daily closes)",
    "stocks outside the S&P 500 (members as of each date, survivorship-free)",
    "net long weight above 1 (no margin loans)",
    "gross weight above 2",
    "stop-loss, take-profit or limit orders on short positions",
    "per-stock borrow fees, locate limits or short-sale bans (borrow is a flat fee)",
    "maintenance-margin calls",
    "annual fundamentals, as-first-reported (unrestated) figures, earnings-announcement "
    "events or analyst data in the strategy view (quarterly Compustat values are restated)",
    "options, ETFs, cash bonds or spot currencies (futures: only the markets listed under "
    "`futures`)",
]

FUTURES_NOT_SUPPORTED = [
    "stop-loss, take-profit or limit orders on futures (target weights only)",
    "calendar-spread or butterfly positions (one continuous position per market)",
    "markets other than those listed; intraday data",
    "maintenance-margin calls (a margin cap scales targets instead)",
]


def capabilities(lake: str | Path | None = None) -> dict:
    from concordance.api import DEFAULTS, ENGINES, STATUSES
    from concordance.data.crsp import PANEL_COLUMNS
    from concordance.data.ingest import default_lake
    from concordance.strategy import MAX_GROSS, MAX_NET, Orders

    lake = Path(lake or default_lake())
    manifest = lake / "manifest.json"
    files = json.loads(manifest.read_text()).get("files", {}) if manifest.exists() else {}
    datasets = {name: {k: f.get(k) for k in ("start", "end", "rows") if f.get(k) is not None}
                for name, f in sorted(files.items())}
    order_fields = [f.name for f in fields(Orders)]
    return {
        "schema_version": SCHEMA_VERSION,
        "frequency": "daily",
        "universe": ["sp500", "futures"],
        "decision": "at each session's close, from data up to and including that close",
        "execution": {
            "rebalance": "market orders at the next session's open",
            "exits_and_entries": "stop-loss and take-profit orders fill inside the next "
                                 "session's range; limit entries fill if the price reaches "
                                 "the level next session",
        },
        "view_fields": ["tr_close", "member", *PANEL_COLUMNS],
        "view_extras": {
            "members": "today's S&P 500 membership mask",
            "position": "what the strategy's own orders hold (entry session/price, levels)",
            "customer_weights": "supplier -> customer revenue shares known at the close; "
                                "needs data.customer_links: true",
            "fundamental": "view.fundamental(name) -> (sessions, assets) Compustat quarterly "
                           "values known at each close ($ millions; quarter visible from the "
                           "session after its report date, stale after 275 days; NaN where "
                           "unknown; fiscal 1992 on); needs data.fundamentals: true. Names: "
                           + ", ".join(VIEW_FIELDS),
        },
        "weights": {"max_net": MAX_NET, "max_gross": MAX_GROSS, "short_selling": True},
        "orders": {"fields": order_fields,
                   "long_only": [f for f in LONG_ONLY_ORDERS if f in order_fields]},
        "costs": DEFAULTS["costs"],
        "liquidity": DEFAULTS["liquidity"],
        "engines": list(ENGINES),
        "statuses": list(STATUSES),
        "datasets": datasets,
        "not_supported": NOT_SUPPORTED,
        "futures": _futures(files),
    }


def _futures(files: dict) -> dict:
    """The futures universe (data.universe = "futures"), when the lake has it."""
    from concordance.api import DEFAULTS, FUTURES_ENGINES
    from concordance.data.futures import MARKETS
    from concordance.futures.panel import WEIGHT_LIMITS

    have = files.get("futures_bars", {})
    if have.get("status") != "ok":
        return {"available": False}
    return {
        "available": True,
        "request": 'data.universe = "futures"; options in the request\'s "futures" section',
        "defaults": DEFAULTS["futures"],
        "markets": {m.root: {"sector": m.sector, "liquid_months": m.months}
                    for m in MARKETS.values() if m.root in have.get("markets", [])},
        "data": {"start": have.get("start"), "end": have.get("end"),
                 "note": "markets join when their data begins (e.g. NQ 1999, 6E 1998, "
                         "BRN 2003, RB 2005)"},
        "view_fields": {"open": "held contract's own open (unadjusted)",
                        "close": "held contract's own settle (unadjusted; jumps at rolls)",
                        "ret": "excess return of a fully collateralized position (point "
                               "change over own previous settle; no roll jumps, no interest)",
                        "member": "market has data that session"},
        "weights": {**WEIGHT_LIMITS, "meaning": "notional exposure / equity per market"},
        "account": "margin: positions cost no cash, P&L settles daily, cash earns the "
                   "daily risk-free rate; integer contracts; initial margin capped at "
                   "futures.max_margin of equity (targets scaled toward zero)",
        "rolls": "Panama continuous series; roll at the close futures.sessions_before "
                 "sessions before the deadline (last trade, or first notice / month before "
                 "delivery for physical markets); a roll costs two legs",
        "costs": "per contract per side: futures.commission USD + futures.slippage_ticks "
                 "ticks",
        "engines": list(FUTURES_ENGINES),
        "not_supported": FUTURES_NOT_SUPPORTED,
    }
