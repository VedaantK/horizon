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
    "annual fundamentals without an explicit prepared annual_lagged source; "
    "as-first-reported (unrestated) figures, earnings-announcement "
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
    us = files.get("crsp_daily_us", {}).get("status") == "ok"
    pit = all(files.get(name, {}).get("status") == "ok"
              for name in ("first_reported_quarterly", "first_reported_admission"))
    return {
        "schema_version": SCHEMA_VERSION,
        "frequency": "daily",
        "universe": ["sp500", *(["us_liquid"] if us else []), "futures"],
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
        "memory": {"defaults": DEFAULTS["memory"], "disk_universes": ["us_liquid"],
                   "description": "Disk-backed arrays, bounded asset loading and continuous "
                                  "streamed Nautilus execution; portfolio state is preserved."},
        "engines": list(ENGINES),
        "statuses": list(STATUSES),
        "datasets": datasets,
        "not_supported": [n for n in NOT_SUPPORTED if not (
            (us and n.startswith("stocks outside"))
            or (pit and n.startswith("annual fundamentals")))
        ] + (["annual fundamentals without an explicit prepared annual_lagged source; "
              "earnings-announcement events and analyst data"] if pit else []),
        "annual_lagged": {
            "supported": True,
            "requires_explicit_source": True,
            "data_setting": "data.fundamentals=true, data.fundamentals_basis=annual_lagged, "
                            "data.annual_source=<prepared directory>",
            "fields": ["at", "ceq", "revt", "cogs"],
            "timing": "prior-calendar-year data at the last June close; next-open execution",
            "limitation": "Current annual STD values may include revisions; "
                          "not authenticated first-reported data.",
        },
        "first_reported": {
            "available": pit,
            "data_setting": "data.fundamentals=true and data.fundamentals_basis=first_reported",
            "fields": ["atq", "seqq", "ceqq", "pstkq", "txditcq", "revtq", "cogsq",
                       "revtq_ttm", "cogsq_ttm"] if pit else [],
            "limitation": "Operator-reviewed source attestation; source authenticity is not "
                          "established by a schema or hash alone. No restated fallback.",
        },
        "us_liquid": _us_liquid(files) if us else None,
        "futures": _futures(files),
    }


def _us_liquid(files: dict) -> dict:
    """All US common stocks (data.universe = "us_liquid"), when the lake has them."""
    from concordance.data.us_universe import MIN_PRICE, NYSE_SIZE_PCT

    have = files.get("crsp_daily_us", {})
    return {
        "request": 'data.universe = "us_liquid"',
        "data": {k: have.get(k) for k in ("start", "end", "permnos_out")},
        "members": f"view.members / view['member']: US common stocks (NYSE, NYSE American, "
                   f"Nasdaq) that closed >= ${MIN_PRICE:g} with market cap >= the NYSE "
                   f"{int(NYSE_SIZE_PCT * 100)}th percentile at the previous close",
        "fields": "view['sp500'] = 1 on S&P 500 member days (exclude them for small and "
                  "mid caps); every other view field as for sp500",
        "benchmark": "cap-weighted total return of the members (an equal-weighted one is "
                     "in the run manifest)",
        "fundamentals": (
            "Admitted first-reported quarterly fields require data.fundamentals_basis: "
            "first_reported; no restated fallback. See first_reported.fields."
            if all(files.get(name, {}).get("status") == "ok" for name in (
                "first_reported_quarterly", "first_reported_admission")) else
            "data.fundamentals: true reads restated quarterly Compustat; exploratory only"
            if files.get("comp_fundq_us", {}).get("status") == "ok" else
            "Quarterly fundamentals are not available in this lake"
        ),
        "caveats": ["shorting small stocks is costly and often impossible, but borrow is a "
                    "flat fee here: prefer long-only designs",
                    "costs use each stock's quoted spread and square-root impact; small "
                    "names are expensive to trade, so keep turnover low"],
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
