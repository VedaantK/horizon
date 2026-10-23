"""Machine-readable run interface: JSON request in, JSON result plus artifact files out.

    from concordance.api import run_backtest
    result = run_backtest({"strategy": {"path": "s.py"}, "data": {"start": ..., "end": ...}},
                          output_dir="runs/x")

or `concordance backtest request.json -o runs/x`. The result always comes back as a dict
with a `status`; failures are reported, not raised, so a coordinator can act on them.
See docs/RUN_API.md for the request and result fields.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.costs import CostModel, LiquidityModel, exec_tables
from concordance.panel import MarketPanel, PanelError
from concordance.result import AccountWipedOut
from concordance.strategy import STOP, TAKE_PROFIT, LookaheadError, Schedule, Strategy

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
ENGINES = ("reference", "vectorbt", "nautilus")
FUTURES_ENGINES = ("reference", "nautilus")  # vectorbt has no margined futures model
UNIVERSES = ("sp500", "us_liquid", "futures")

DEFAULTS: dict = {
    "run_id": None,
    "strategy": {"path": None, "params": {}},
    "data": {"universe": "sp500", "start": None, "end": None, "lake": None,
             "customer_links": False, "fundamentals": False, "fundamentals_basis": "restated",
             "annual_source": None},
    "initial_capital": 1_000_000.0,
    "engines": list(ENGINES),
    "costs": {"commission_bps": 1.0, "default_half_spread_bps": 5.0,
              "use_prior_quoted_spread": True, "borrow_bps": 50.0, "min_trade_bps": 0.0,
              "spread_multiplier": 1.0},
    "liquidity": {"participation": 0.10, "impact_coef": 1.0, "max_impact_bps": 500.0,
                  "window": 20, "min_obs": 10},
    "validation": {"program": "default", "placebo": 0, "lookahead_cuts": 3},
    "lockbox": {"unlock_reason": None},
    "report": False,
    "memory": {"mode": "ram", "asset_batch_size": 256, "bar_batch_sessions": 20},
    # Used only when data.universe is "futures" (docs/SEMANTICS.md, "Futures").
    "futures": {"markets": None, "commission": 2.5, "slippage_ticks": 1.0,
                "max_margin": 0.5, "sessions_before": 5},
}

# Statuses a result can have. Only "ok" carries metrics you may act on.
STATUSES = ("ok", "rejected", "lookahead", "strategy_error", "data_error",
            "invalid_request", "error")


class RequestError(ValueError):
    pass


class StrategyError(RuntimeError):
    pass


# --------------------------------------------------------------------------- strategies


def load_strategy(path: str | Path, params: dict | None = None) -> Strategy:
    """A fresh strategy from a file: `make_strategy(**params)` if the file defines it,
    else its `STRATEGY` (which then takes no params). Each call re-executes the module, so
    no state leaks between instances."""
    path = Path(path)
    if not path.is_file():
        raise RequestError(f"strategy file not found: {path}")
    name = f"concordance_user_strategy_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise StrategyError(f"importing {path.name} failed: {e!r}") from e
    params = params or {}
    if hasattr(mod, "make_strategy"):
        try:
            strat = mod.make_strategy(**params)
        except TypeError as e:
            raise StrategyError(f"make_strategy rejected params {params}: {e}") from e
    elif hasattr(mod, "STRATEGY"):
        if params:
            raise RequestError("params given but the file has no make_strategy(**params)")
        strat = mod.STRATEGY
    else:
        raise StrategyError(f"{path.name} must define make_strategy(**params) or STRATEGY")
    if not isinstance(strat, Strategy):
        raise StrategyError(f"{path.name} produced {type(strat).__name__}, not a Strategy")
    return strat


# ------------------------------------------------------------------------------ request


def _merge(base: dict, over: dict, where: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise RequestError(f"unknown request field {where}{k!r}")
        if isinstance(base[k], dict) and base[k] and v is not None:  # {} = free-form
            if not isinstance(v, dict):
                raise RequestError(f"{where}{k} must be an object")
            out[k] = _merge(base[k], v, f"{where}{k}.")
        else:
            out[k] = v
    return out


def resolve_request(request: dict) -> dict:
    """Defaults filled in and checked. `liquidity: null` turns the liquidity model off."""
    liq_off = "liquidity" in request and request["liquidity"] is None
    req = _merge(DEFAULTS, {k: v for k, v in request.items()
                            if not (k == "liquidity" and v is None)})
    if liq_off:
        req["liquidity"] = None
    memory = req["memory"]
    if not isinstance(memory, dict) or memory.get("mode") not in ("ram", "disk"):
        raise RequestError("memory.mode must be ram or disk")
    for key in ("asset_batch_size", "bar_batch_sessions"):
        if (isinstance(memory[key], bool) or not isinstance(memory[key], int)
                or not 1 <= memory[key] <= 4096):
            raise RequestError(f"memory.{key} must be an integer in [1,4096]")
    if memory["mode"] == "disk" and req["data"]["universe"] != "us_liquid":
        raise RequestError("disk mode currently supports data.universe=us_liquid")
    if not req["strategy"]["path"]:
        raise RequestError("strategy.path is required")
    if req["data"]["universe"] not in UNIVERSES:
        raise RequestError(f"data.universe must be one of {UNIVERSES}")
    if req["data"]["universe"] == "futures":
        if "engines" not in request:
            req["engines"] = list(FUTURES_ENGINES)
        if set(req["engines"]) - set(FUTURES_ENGINES):
            raise RequestError(f"futures run on {FUTURES_ENGINES} only (vectorbt has no "
                               "margined futures model)")
        if req["data"]["customer_links"] or req["data"]["fundamentals"]:
            raise RequestError("customer_links and fundamentals are stock data")
        f = req["futures"]
        from concordance.data.futures import MARKETS

        if f["markets"] is not None and (not isinstance(f["markets"], list)
                                         or set(f["markets"]) - set(MARKETS)):
            raise RequestError(f"futures.markets must be a list from {sorted(MARKETS)}")
        for k in ("commission", "slippage_ticks"):
            if isinstance(f[k], bool) or not isinstance(f[k], int | float) or f[k] < 0:
                raise RequestError(f"futures.{k} must be a number >= 0")
        if not 0 < f["max_margin"] <= 1:
            raise RequestError("futures.max_margin must be in (0, 1]")
        if not isinstance(f["sessions_before"], int) or not 0 <= f["sessions_before"] <= 20:
            raise RequestError("futures.sessions_before must be an integer in [0, 20]")
    for k in ("start", "end"):
        if not req["data"][k]:
            raise RequestError(f"data.{k} is required")
        try:
            pd.Timestamp(req["data"][k])
        except ValueError as e:
            raise RequestError(f"data.{k}: {e}") from e
    for flag in ("customer_links", "fundamentals"):
        if not isinstance(req["data"][flag], bool):
            raise RequestError(f"data.{flag} must be true or false")
    if req["data"]["fundamentals_basis"] not in ("restated", "first_reported", "annual_lagged"):
        raise RequestError("data.fundamentals_basis must be restated, first_reported "
                           "or annual_lagged")
    annual_source = req["data"]["annual_source"]
    if req["data"]["fundamentals_basis"] == "annual_lagged":
        if (not req["data"]["fundamentals"] or not isinstance(annual_source, str)
                or not annual_source):
            raise RequestError("annual_lagged requires fundamentals=true and data.annual_source")
    elif annual_source is not None:
        raise RequestError("data.annual_source requires annual_lagged basis")
    if pd.Timestamp(req["data"]["start"]) >= pd.Timestamp(req["data"]["end"]):
        raise RequestError("data.start must be before data.end")
    bad = set(req["engines"]) - set(ENGINES)
    if bad or not req["engines"]:
        raise RequestError(f"engines must be a non-empty subset of {ENGINES}")
    if not req["initial_capital"] or req["initial_capital"] <= 0:
        raise RequestError("initial_capital must be positive")
    if req["liquidity"] is not None:
        q = req["liquidity"]
        if not 0 < q["participation"] <= 1:
            raise RequestError("liquidity.participation must be in (0, 1]")
        if q["min_obs"] > q["window"] or q["window"] < 1:
            raise RequestError("liquidity: need 1 <= min_obs <= window")
    for k in ("commission_bps", "default_half_spread_bps", "borrow_bps", "min_trade_bps",
              "spread_multiplier"):
        v = req["costs"][k]
        if isinstance(v, bool) or not isinstance(v, int | float) or not v >= 0:
            raise RequestError(f"costs.{k} must be a number >= 0")
    if req["costs"]["min_trade_bps"] >= 1e4:
        raise RequestError("costs.min_trade_bps must be below 10000 (100% of the portfolio)")
    if not isinstance(req["strategy"]["params"], dict):
        raise RequestError("strategy.params must be an object")
    req["run_id"] = req["run_id"] or (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-"
                                      + uuid.uuid4().hex[:8])
    return req


# ------------------------------------------------------------------------------ helpers


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git() -> dict:
    def g(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    try:
        return {"commit": g("rev-parse", "HEAD") or None,
                "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _versions() -> dict:
    out = {}
    for pkg in ("numpy", "pandas", "numba", "vectorbt", "nautilus_trader", "concordance"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def equity_hash(equity: np.ndarray) -> str:
    """Fingerprint of an equity curve at 1e-6 resolution (for reproduction checks)."""
    return hashlib.sha256(np.round(np.asarray(equity, float), 6).tobytes()).hexdigest()


def _fill_kind(panel: MarketPanel, s: Schedule, f: pd.DataFrame) -> np.ndarray:
    i, j, ph = f["session"].to_numpy(), f["asset"].to_numpy(), f["phase"].to_numpy()
    kind = np.full(len(f), "rebalance", dtype=object)
    if not len(f):
        return kind
    k = s.exit_kind[i, j]
    kind[(ph == 0) & s.has[i]] = "rebalance"
    kind[(ph == 0) & ~s.has[i]] = "carried"
    kind[(ph == 0) & s.gap_exit[i, j] & (k == STOP)] = "stop_at_open"
    kind[(ph == 0) & s.gap_exit[i, j] & (k == TAKE_PROFIT)] = "take_profit_at_open"
    kind[(ph == 0) & panel.liquidate[i, j]] = "delisting"
    kind[(ph == 1) & (k == STOP)] = "stop"
    kind[(ph == 1) & (k == TAKE_PROFIT)] = "take_profit"
    kind[(ph == 1) & s.limit_filled[i, j]] = "limit_entry"
    return kind


def _write_artifacts(out: Path, panel: MarketPanel, res, s: Schedule) -> dict:
    from concordance.memory import allocate

    sessions, assets = panel.sessions, np.asarray(panel.assets)
    val = panel.val_price
    value = allocate(panel.shape)
    with np.errstate(invalid="ignore"):
        for lo in range(0, len(sessions), 128):
            sl = slice(lo, lo + 128)
            value[sl] = np.nan_to_num(res.positions[sl] * val[sl])
    eq = pd.DataFrame({"date": sessions, "equity": res.equity, "cash": res.cash,
                       "gross_exposure": np.abs(value).sum(1) / res.equity,
                       "n_positions": (res.positions != 0).sum(1)})
    bench = panel.meta.get("benchmark")
    if bench is not None:
        eq["benchmark_return"] = pd.Series(bench).reindex(sessions).to_numpy()
    eq.to_parquet(out / "equity.parquet", index=False)

    f = res.fills
    fills = pd.DataFrame({
        "date": sessions[f["session"].to_numpy()], "asset": assets[f["asset"].to_numpy()],
        "phase": np.where(f["phase"].to_numpy() == 0, "open", "intraday"),
        "kind": _fill_kind(panel, s, f), "qty": f["qty"].to_numpy(),
        "price": f["price"].to_numpy(), "fee": f["fee"].to_numpy()})
    fills.to_parquet(out / "fills.parquet", index=False)

    ii, jj = np.nonzero(res.positions)
    pos = pd.DataFrame({"date": sessions[ii], "asset": assets[jj],
                        "qty": res.positions[ii, jj], "value": value[ii, jj],
                        "weight": value[ii, jj] / res.equity[ii]})
    pos.to_parquet(out / "positions.parquet", index=False)

    rows = []
    for name, arr in (("stop", s.stop), ("take_profit", s.take_profit),
                      ("entry_limit", s.entry_limit)):
        ii, jj = np.nonzero(np.isfinite(arr))
        if name == "entry_limit":
            hit = s.limit_filled[ii, jj]
            px = np.where(hit, arr[ii, jj], np.nan)
        else:
            code = STOP if name == "stop" else TAKE_PROFIT
            hit = np.isfinite(s.exit_price[ii, jj]) & (s.exit_kind[ii, jj] == code)
            px = np.where(hit, s.exit_price[ii, jj], np.nan)
        rows.append(pd.DataFrame({"date": sessions[ii], "asset": assets[jj], "order": name,
                                  "level_tr": arr[ii, jj], "triggered": hit,
                                  "fill_price_tr": px}))
    ii, jj = np.nonzero(s.gap_exit)
    rows.append(pd.DataFrame({"date": sessions[ii], "asset": assets[jj],
                              "order": np.where(s.exit_kind[ii, jj] == STOP, "stop",
                                                "take_profit"),
                              "level_tr": np.nan, "triggered": True,
                              "fill_price_tr": panel.exec_price[ii, jj]}))
    orders = pd.concat(rows, ignore_index=True).sort_values(["date", "asset"])
    orders.to_parquet(out / "orders.parquet", index=False)
    return {"equity": "equity.parquet", "fills": "fills.parquet",
            "positions": "positions.parquet", "orders": "orders.parquet"}


def _order_stats(s: Schedule, res) -> dict:
    k = s.exit_kind
    return {
        "stop_exits": int(((k == STOP) & (s.gap_exit | np.isfinite(s.exit_price))).sum()),
        "take_profit_exits": int(((k == TAKE_PROFIT)
                                  & (s.gap_exit | np.isfinite(s.exit_price))).sum()),
        "exits_at_open": int(s.gap_exit.sum()),
        "limit_entries_placed": int(np.isfinite(s.entry_limit).sum()),
        "limit_entries_filled": int(s.limit_filled.sum()),
        "intraday_fills": int((res.fills["phase"] == 1).sum()),
        "stops_set_at_or_above_close": s.notes.get("stops_at_or_above_close", 0),
        "take_profits_set_at_or_below_close": s.notes.get("take_profits_at_or_below_close", 0),
    }


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def listing_gap_exits(panel: MarketPanel, res) -> pd.DataFrame:
    """Forced exits at the start of a listing gap that held a position (a listing whose
    PERMNO relists later inside the panel)."""
    from concordance.data.crsp import permno_of

    gaps = panel.meta.get("listing_gaps")
    f = res.fills
    if gaps is None or not len(gaps) or not len(f):
        return pd.DataFrame(columns=["date", "permno", "qty", "price"])
    ids = np.asarray(panel.assets)[f["asset"].to_numpy()]
    when = panel.sessions[f["session"].to_numpy()]
    forced = panel.liquidate[f["session"].to_numpy(), f["asset"].to_numpy()]
    # The first relisting after the exit tells whether CRSP gave a delisting value for it.
    def unknown_exit(p, d):
        g = gaps[(gaps["permno"] == p) & (gaps["date"] >= d)].sort_values("date")
        return len(g) > 0 and not bool(g["gap_exit_known"].iloc[0])

    relists_later = np.array([unknown_exit(p, d)
                              for p, d in zip(permno_of(ids), when, strict=True)])
    hit = forced & relists_later
    f = f[hit]
    return pd.DataFrame({"date": panel.sessions[f["session"].to_numpy()].date,
                         "permno": permno_of(ids[hit]), "qty": f["qty"].to_numpy(),
                         "price": f["price"].to_numpy()})


# ---------------------------------------------------------------------------------- run


def load_panel(req: dict, strategy: Strategy, log=lambda msg: None):
    """The request's CRSP panel (with warm-up and, if asked, customer links), the lake path
    and the data section of the manifest."""
    from concordance.data.crsp import load_sp500_panel
    from concordance.data.ingest import default_lake

    lake = Path(req["data"]["lake"] or default_lake())
    expected_basis = getattr(strategy, "fundamentals_basis", None)
    if expected_basis and (not req["data"]["fundamentals"]
                           or req["data"]["fundamentals_basis"] != expected_basis):
        raise RequestError(f"strategy requires data.fundamentals=true and "
                           f"data.fundamentals_basis={expected_basis}")
    log(f"loading panel {req['data']['start']} .. {req['data']['end']}")
    if req["data"]["universe"] == "futures":
        return _load_futures(req, strategy, lake)
    us = req["data"]["universe"] == "us_liquid"
    # A short-lookback signal still needs prior observations for ADV/impact sizing.
    # Without this, an otherwise valid first rebalance can be silently untradeable.
    liquidity = req.get("liquidity", DEFAULTS["liquidity"])
    execution_history = int(liquidity["window"]) + 1 if liquidity else 0
    available_execution_history = execution_history
    lm = lake / "manifest.json"
    if lm.exists():
        files = json.loads(lm.read_text()).get("files", {})
        first = files.get("crsp_daily_us" if us else "crsp_daily", {}).get("start")
        if first:
            from concordance.data.crsp import xnys_sessions

            first, start = pd.Timestamp(first), pd.Timestamp(req["data"]["start"])
            available_execution_history = (len(xnys_sessions(first, start - pd.Timedelta(days=1)))
                                           if first < start else 0)
    # Never invent unavailable volume history or weaken a strategy's own requirement.
    warmup = max(strategy.warmup(), min(execution_history, available_execution_history))
    if us:
        from concordance.memory import STORE

        if STORE.get() is not None:
            from concordance.data.disk_panel import load_disk_us_panel as load_us_panel
        else:
            from concordance.data.us_universe import load_us_panel

        panel = load_us_panel(req["data"]["start"], req["data"]["end"], lake=lake,
                              warmup_sessions=warmup,
                              unlock_reason=req["lockbox"]["unlock_reason"])
    else:
        panel = load_sp500_panel(req["data"]["start"], req["data"]["end"], lake=lake,
                                 warmup_sessions=warmup,
                                 unlock_reason=req["lockbox"]["unlock_reason"])
    lm = lake / "manifest.json"
    data = {"lake": str(lake), "universe": req["data"]["universe"],
            "lake_manifest_sha256": _sha256_file(lm) if lm.exists() else None}
    if warmup < execution_history:
        data["partial_execution_history"] = {"requested": execution_history, "loaded": warmup,
                                             "behavior": "existing insufficient-ADV rules apply"}
    if us:
        data["universe_rule"] = panel.meta["universe_rule"]
        data["members_per_day"] = round(panel.meta["members_per_day"], 1)
    if req["data"]["customer_links"]:
        from concordance.data.customers import attach_customer_links

        panel = attach_customer_links(panel, lake)
        data["customer_links"] = {**panel.meta["customer_links"],
                                  "sha256": _sha256_file(lake / "customer_links.parquet")}
    if req["data"]["fundamentals"]:
        if req["data"]["fundamentals_basis"] == "annual_lagged":
            from concordance.data.annual import MANIFEST, attach

            source = Path(req["data"]["annual_source"])
            panel = attach(panel, source)
            data["fundamentals"] = {**panel.meta["fundamentals"],
                                    "manifest_sha256": _sha256_file(source / MANIFEST)}
            return panel, lake, data
        if req["data"]["fundamentals_basis"] == "first_reported":
            from concordance.data.first_reported_panel import ADMISSION, FILE, attach

            panel = attach(panel, lake)
            data["fundamentals"] = {**panel.meta["fundamentals"], "file": FILE,
                                    "admission_sha256": _sha256_file(lake / ADMISSION)}
            return panel, lake, data
        from concordance.data.fundamentals import attach_fundamentals

        fundq = "comp_fundq_us.parquet" if us else "comp_fundq.parquet"
        panel = attach_fundamentals(panel, lake, fundq)
        data["fundamentals"] = {**panel.meta["fundamentals"], "file": fundq,
                                "sha256": _sha256_file(lake / fundq), "basis": "restated"}
    return panel, lake, data


def _load_futures(req: dict, strategy: Strategy, lake: Path):
    from dataclasses import replace

    from concordance.data import lockbox
    from concordance.data.futures import load_futures_panel

    lockbox.enforce(req["data"]["end"], req["lockbox"]["unlock_reason"])
    f = req["futures"]
    panel = load_futures_panel(req["data"]["start"], req["data"]["end"], lake,
                               roots=f["markets"], sessions_before=f["sessions_before"],
                               warmup_sessions=strategy.warmup())
    fd = panel.meta["futures"]
    specs = tuple(replace(s, commission=float(f["commission"]),
                          slippage_ticks=float(f["slippage_ticks"])) for s in fd.specs)
    panel.meta["futures"] = replace(fd, specs=specs)
    panel.meta["warmup_sessions"] = strategy.warmup()
    lm = lake / "manifest.json"
    data = {"lake": str(lake), "universe": "futures",
            "lake_manifest_sha256": _sha256_file(lm) if lm.exists() else None,
            "futures": {"bars_sha256": _sha256_file(lake / "futures_bars.parquet"),
                        "contracts_sha256": _sha256_file(lake / "futures_contracts.parquet"),
                        "markets": list(panel.assets),
                        "skipped_contracts": panel.meta.get("skipped_contracts", {})}}
    return panel, lake, data


def request_tables(req: dict, panel: MarketPanel):
    """Fee rates, volume caps and the other execution tables a request asks for."""
    c, q = req["costs"], req["liquidity"]
    costs = CostModel(commission_bps=c["commission_bps"],
                      default_half_spread_bps=c["default_half_spread_bps"],
                      use_prior_quoted_spread=c["use_prior_quoted_spread"],
                      borrow_bps=c["borrow_bps"], min_trade_bps=c["min_trade_bps"],
                      spread_multiplier=c["spread_multiplier"])
    liquidity = LiquidityModel(**q) if q is not None else None
    return exec_tables(panel, costs, liquidity)


def run_backtest(request: dict, output_dir: str | Path | None = None,
                 panel: MarketPanel | None = None, quiet: bool = True) -> dict:
    from concordance.memory import ArrayStore

    try:
        req = resolve_request(request)
    except RequestError:
        return _run_backtest(request, output_dir, panel, quiet)
    if req["memory"]["mode"] == "ram":
        return _run_backtest(req, output_dir, panel, quiet)
    out = Path(output_dir or REPO / "runs" / req["run_id"])
    out.mkdir(parents=True, exist_ok=True)
    with ArrayStore(out, req["memory"]["asset_batch_size"],
                    req["memory"]["bar_batch_sessions"]):
        return _run_backtest(req, out, panel, quiet)


def _run_backtest(request: dict, output_dir: str | Path | None = None,
                 panel: MarketPanel | None = None, quiet: bool = True) -> dict:
    """Run one request. Writes artifacts and result.json to `output_dir` (default
    runs/<run_id>). `panel` overrides the data section (tests, synthetic data)."""
    t0 = time.time()
    started = datetime.now(UTC).isoformat()
    result: dict = {"schema_version": SCHEMA_VERSION, "run_id": None, "status": "error",
                    "diagnostics": {"messages": [], "warnings": []}, "metrics": {},
                    "artifacts": {}, "manifest": {}}
    diag = result["diagnostics"]

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    try:
        req = resolve_request(request)
    except RequestError as e:
        result["status"] = "invalid_request"
        diag["messages"].append(str(e))
        return result
    result["run_id"] = req["run_id"]
    out = Path(output_dir or REPO / "runs" / req["run_id"])
    out.mkdir(parents=True, exist_ok=True)
    src = Path(req["strategy"]["path"]).resolve()
    manifest = result["manifest"] = {
        "run_id": req["run_id"], "started_at": started, "request": req,
        "strategy": {"path": str(src), "sha256": _sha256_file(src) if src.is_file() else None,
                     "params": req["strategy"]["params"]},
        "concordance": _git(), "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": _versions()}}

    try:
        if src.is_file():
            shutil.copyfile(src, out / "strategy.py")
        resolved = copy.deepcopy(req)
        resolved["strategy"]["path"] = "strategy.py"
        (out / "request.json").write_text(json.dumps(resolved, indent=2))

        def fresh():
            return load_strategy(src, req["strategy"]["params"])

        strategy = fresh()
        manifest["strategy"].update(name=getattr(strategy, "name", type(strategy).__name__),
                                    cls=type(strategy).__name__)
        if panel is None:
            panel, lake, manifest["data"] = load_panel(req, strategy, log)
        else:  # caller-supplied panel: keep its trial ledger out of the real lake
            lake = Path(req["data"]["lake"]) if req["data"]["lake"] else out
            manifest["data"] = {"lake": None, "note": "panel supplied by caller"}
        manifest["data"]["panel"] = {
            "sessions": int(panel.shape[0]), "securities": int(panel.shape[1]),
            "first": str(panel.sessions[0].date()), "last": str(panel.sessions[-1].date()),
            "coverage": panel.meta.get("coverage_overall")}
        result["status"] = "running"
        result.update(_evaluate(req, panel, strategy, fresh, out, lake, log, diag))
    except LookaheadError as e:
        result["status"] = "lookahead"
        diag["messages"].append(str(e))
    except StrategyError as e:
        result["status"] = "strategy_error"
        diag["messages"].append(str(e))
        diag["traceback"] = traceback.format_exc(limit=-6)
    except RequestError as e:
        result["status"] = "invalid_request"
        diag["messages"].append(str(e))
    except Exception as e:  # reported, not raised
        from concordance.data.crsp import CoverageError
        from concordance.data.lockbox import LockboxViolation

        if isinstance(e, AccountWipedOut):  # a (terrible) outcome of the strategy itself
            result["status"] = "strategy_error"
        elif isinstance(e, CoverageError | LockboxViolation | PanelError | FileNotFoundError):
            result["status"] = "data_error"
        else:
            result["status"] = "error"
        diag["messages"].append(f"{type(e).__name__}: {e}")
        diag["traceback"] = traceback.format_exc(limit=-8)
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    from concordance.memory import STORE

    if STORE.get() is not None:
        manifest["memory"] = {"mode": "disk", "mapped_bytes": STORE.get().bytes,
                              "mapped_files": STORE.get().files,
                              "asset_batch_size": STORE.get().batch_assets,
                              "bar_batch_sessions": STORE.get().bar_sessions}
    manifest["elapsed_s"] = round(time.time() - t0, 2)
    result = _jsonable(result)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    result["output_dir"] = str(out)
    return result


def _evaluate(req, panel, strategy, fresh, out, lake, log, diag) -> dict:
    from concordance.reconcile import ReconciliationReport, reconcile
    from concordance.report import factor_exposures, html_report, leg_attribution, metrics
    from concordance.runner import run_engine
    from concordance.strategy import build_schedule
    from concordance.validation.dsr import deflated_sharpe
    from concordance.validation.ledger import TrialLedger
    from concordance.validation.lookahead import check_truncation_invariance
    from concordance.validation.placebo import random_selection_placebo

    v = req["validation"]
    try:
        sched = build_schedule(strategy, panel)
        if v["lookahead_cuts"]:
            log(f"lookahead: truncation check at {v['lookahead_cuts']} cut points")
            check_truncation_invariance(lambda p: fresh(), panel, cuts=v["lookahead_cuts"])
    except (LookaheadError, StrategyError):
        raise
    except Exception as e:
        raise StrategyError(f"strategy failed while deciding: {type(e).__name__}: {e}") from e
    if panel.meta.get("account") == "margin":
        return _evaluate_futures(req, panel, strategy, sched, out, lake, log, diag)
    tables = request_tables(req, panel)
    results = {}
    checks = []
    engine_names = sorted(req["engines"], key=lambda e: e != "reference")
    for e in engine_names:
        log(f"engine: {e}")
        candidate = run_engine(e, panel, sched, tables, req["initial_capital"])
        if not results:
            results[e] = candidate
        else:
            pair = reconcile({**results, e: candidate}, panel.val_price)
            checks.extend(pair.checks)
            del candidate
            import gc

            gc.collect()
    report = ReconciliationReport(checks) if len(engine_names) > 1 else None
    if report is not None:
        diag["reconciliation"] = report.summary()
        if not report.ok:
            return {"status": "rejected",
                    "diagnostics": {**diag, "messages": diag["messages"]
                                    + ["engines disagree: result rejected"]}}
    else:
        diag["warnings"].append("single engine: not reconciled")
    res = results.get("reference") or next(iter(results.values()))

    start = panel.meta.get("warmup_sessions", 0)
    bench = panel.meta.get("benchmark")
    m = metrics(res, start=start, benchmark=bench)
    m.update(leg_attribution(res, panel.val_price, start))
    ff_path = lake / "ff_factors_daily.parquet"
    if ff_path.exists():
        m.update(factor_exposures(res.returns.iloc[max(start, 1):],
                                  pd.read_parquet(ff_path)))
    else:
        diag["warnings"].append("no ff_factors_daily in the lake: factor exposures skipped")
    m["orders"] = _order_stats(sched, res)
    gap = listing_gap_exits(panel, res)
    if len(gap):
        m["listing_gap_exits"] = len(gap)
        diag["warnings"].append(f"{len(gap)} positions exited at a listing gap at the last "
                                "price before it (CRSP has no return across the gap)")
    if sched.notes.get("stops_at_or_above_close"):
        diag["warnings"].append(f"{sched.notes['stops_at_or_above_close']} stop levels were "
                                "set at or above the close (they fire at the next open)")

    rets = res.returns.iloc[max(start, 1):]
    ledger = TrialLedger(root=lake / "ledger", program=v["program"])
    params = getattr(strategy, "params", req["strategy"]["params"])
    trial = ledger.record(getattr(strategy, "name", type(strategy).__name__), params, rets,
                          note=f"run {req['run_id']}", context={
                              "source": _sha256_file(Path(req["strategy"]["path"])),
                              "data": req["data"], "costs": req.get("costs"),
                              "futures": req.get("futures"),
                              "capital": req["initial_capital"],
                              "data_manifest": _sha256_file(lake / "manifest.json")
                              if (lake / "manifest.json").exists() else None})
    n_eff = max(ledger.n_effective(), 1.0)
    m["dsr"] = deflated_sharpe(rets.to_numpy(), n_trials=n_eff).annualized()
    m["trials_recorded"] = ledger.n_trials()
    m["trials_effective"] = n_eff
    m["trial_hash"] = trial
    if v["placebo"]:
        log(f"placebo: {v['placebo']} random portfolios")
        pl = random_selection_placebo(panel, sched.weights, sched.has, tables,
                                      req["initial_capital"], n=v["placebo"])
        m["placebo_p_value"] = pl.p_value
        if sched.uses_orders:
            diag["warnings"].append("placebo replays target weights only; stop, take-profit "
                                    "and limit orders are not part of it")
    m["equity_sha256"] = equity_hash(res.equity)
    artifacts = _write_artifacts(out, panel, res, sched)
    cp = _crash_protection(rets, lake, req["costs"], bench, diag)
    if cp is not None:
        m["crash_protection"], frame = cp
        frame.rename_axis("date").reset_index().to_parquet(out / "crash_protection.parquet",
                                                           index=False)
        artifacts["crash_protection"] = "crash_protection.parquet"
    if req["report"]:
        extra = {"reconciliation": diag.get("reconciliation", "single engine"),
                 "metrics": json.dumps(_jsonable(m), indent=2, default=str)}
        limitation = panel.meta.get("fundamentals", {}).get("limitation")
        if limitation:
            extra["accounting data limitation"] = limitation
        if diag["warnings"]:
            extra["execution/data notes"] = "\n".join(diag["warnings"])
        html_report(res, out / "report.html", title=getattr(strategy, "name", "strategy"),
                    extra=extra,
                    benchmark=bench, start=start)
        artifacts["report"] = "report.html"
    artifacts.update(strategy="strategy.py", request="request.json")
    return {"status": "ok", "metrics": m, "artifacts": artifacts, "diagnostics": diag}


def _crash_protection(rets, lake, costs, bench, diag):
    """The crash-protection report (overlay.py); a missing input is a warning, because the
    run itself is complete without it."""
    from concordance.overlay import crash_protection

    cost = (costs["commission_bps"]
            + costs["default_half_spread_bps"] * costs["spread_multiplier"])
    try:
        return crash_protection(rets, lake, cost, benchmark=None if bench is None
                                else pd.Series(bench))
    except (FileNotFoundError, ValueError) as e:
        diag["warnings"].append(f"crash protection not reported: {e}")
        return None


def _evaluate_futures(req, panel, strategy, sched, out, lake, log, diag) -> dict:
    """Margin-account evaluation: reference and Nautilus must agree exactly on positions
    and fills, and on equity to within money rounding."""
    from concordance.futures import engine as fut_ref
    from concordance.futures import nautilus as fut_nt
    from concordance.futures.reconcile import reconcile_futures
    from concordance.report import factor_exposures, metrics
    from concordance.validation.dsr import deflated_sharpe
    from concordance.validation.ledger import TrialLedger

    if sched.uses_orders:
        raise StrategyError("futures take target weights only (no stop, take-profit or "
                            "limit orders)")
    rules = fut_ref.MarginRules(max_margin=float(req["futures"]["max_margin"]))
    runners = {"reference": fut_ref.run, "nautilus": fut_nt.run}
    results = {}
    for e in req["engines"]:
        log(f"engine: {e}")
        results[e] = runners[e](panel, sched, req["initial_capital"], rules)
    if len(results) > 1:
        rep = reconcile_futures(results["reference"], results["nautilus"])
        diag["reconciliation"] = rep["summary"]
        if not rep["ok"]:
            return {"status": "rejected",
                    "diagnostics": {**diag, "messages": diag["messages"]
                                    + ["engines disagree: result rejected"]}}
    else:
        diag["warnings"].append("single engine: not reconciled")
    res = results.get("reference") or next(iter(results.values()))
    start = panel.meta.get("warmup_sessions", 0)
    m = metrics(res, start=start, benchmark=None)
    ff_path = lake / "ff_factors_daily.parquet"
    if ff_path.exists():
        m.update(factor_exposures(res.returns.iloc[max(start, 1):],
                                  pd.read_parquet(ff_path)))
    fills = res.fills
    m["n_rolls"] = int((fills["phase"] == 2).sum())
    m["roll_fees"] = float(fills.loc[fills["phase"] == 2, "fee"].sum())
    m["margin_scaled_sessions"] = len(res.scaled_sessions)
    held = res.positions[start:] != 0
    m["avg_markets_held"] = float(held.sum(axis=1).mean()) if len(held) else 0.0
    if req["validation"]["placebo"]:
        diag["warnings"].append("the stock placebo does not apply to futures; skipped")
    rets = res.returns.iloc[max(start, 1):]
    ledger = TrialLedger(root=lake / "ledger", program=req["validation"]["program"])
    params = getattr(strategy, "params", req["strategy"]["params"])
    trial = ledger.record(getattr(strategy, "name", type(strategy).__name__), params, rets,
                          note=f"run {req['run_id']}", context={
                              "source": _sha256_file(Path(req["strategy"]["path"])),
                              "data": req["data"], "costs": req.get("costs"),
                              "futures": req.get("futures"),
                              "capital": req["initial_capital"],
                              "data_manifest": _sha256_file(lake / "manifest.json")
                              if (lake / "manifest.json").exists() else None})
    n_eff = max(ledger.n_effective(), 1.0)
    m["dsr"] = deflated_sharpe(rets.to_numpy(), n_trials=n_eff).annualized()
    m["trials_recorded"] = ledger.n_trials()
    m["trials_effective"] = n_eff
    m["trial_hash"] = trial
    m["equity_sha256"] = equity_hash(res.equity)
    eq = pd.DataFrame({"date": panel.sessions, "equity": res.equity, "cash": res.cash,
                       "gross_exposure": np.nan, "n_positions": (res.positions != 0).sum(axis=1),
                       "benchmark_return": np.nan})
    eq.to_parquet(out / "equity.parquet", index=False)
    res.fills_frame().to_parquet(out / "fills.parquet", index=False)
    pos = pd.DataFrame(res.positions, index=panel.sessions, columns=panel.assets)
    pos.stack().rename("contracts").reset_index().query("contracts != 0").to_parquet(
        out / "positions.parquet", index=False)
    artifacts = {"equity": "equity.parquet", "fills": "fills.parquet",
                 "positions": "positions.parquet", "strategy": "strategy.py",
                 "request": "request.json"}
    return {"status": "ok", "metrics": m, "artifacts": artifacts, "diagnostics": diag}


def reproduce(run_dir: str | Path, output_dir: str | Path | None = None) -> dict:
    """Re-run a saved run from its own request.json and strategy.py copy and compare."""
    run_dir = Path(run_dir).resolve()
    old = json.loads((run_dir / "result.json").read_text())
    req = json.loads((run_dir / "request.json").read_text())
    req["strategy"]["path"] = str(run_dir / "strategy.py")
    req["run_id"] = f"{req['run_id']}-repro"
    new = run_backtest(req, output_dir or run_dir / "repro")
    same = {
        "status": old.get("status") == new.get("status"),
        "equity": old.get("metrics", {}).get("equity_sha256")
        == new.get("metrics", {}).get("equity_sha256"),
        "strategy_source": old["manifest"]["strategy"]["sha256"]
        == new["manifest"]["strategy"]["sha256"],
        "data": old["manifest"].get("data", {}).get("lake_manifest_sha256")
        == new["manifest"].get("data", {}).get("lake_manifest_sha256"),
        "code": old["manifest"]["concordance"]["commit"]
        == new["manifest"]["concordance"]["commit"],
    }
    return {"reproduced": same["status"] and same["equity"], "checks": same,
            "original": str(run_dir), "rerun": new.get("output_dir")}
