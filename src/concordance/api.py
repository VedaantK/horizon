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
from concordance.strategy import STOP, TAKE_PROFIT, LookaheadError, Schedule, Strategy

SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[2]
ENGINES = ("reference", "vectorbt", "nautilus")

DEFAULTS: dict = {
    "run_id": None,
    "strategy": {"path": None, "params": {}},
    "data": {"universe": "sp500", "start": None, "end": None, "lake": None},
    "initial_capital": 1_000_000.0,
    "engines": list(ENGINES),
    "costs": {"commission_bps": 1.0, "default_half_spread_bps": 5.0,
              "use_prior_quoted_spread": True},
    "liquidity": {"participation": 0.10, "impact_coef": 1.0, "max_impact_bps": 500.0,
                  "window": 20, "min_obs": 10},
    "validation": {"program": "default", "placebo": 0, "lookahead_cuts": 3},
    "lockbox": {"unlock_reason": None},
    "report": False,
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
    if not req["strategy"]["path"]:
        raise RequestError("strategy.path is required")
    if req["data"]["universe"] != "sp500":
        raise RequestError("data.universe: only 'sp500' is available")
    for k in ("start", "end"):
        if not req["data"][k]:
            raise RequestError(f"data.{k} is required")
        try:
            pd.Timestamp(req["data"][k])
        except ValueError as e:
            raise RequestError(f"data.{k}: {e}") from e
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
    sessions, assets = panel.sessions, np.asarray(panel.assets)
    val = pd.DataFrame(panel.tr_close).ffill().to_numpy()
    with np.errstate(invalid="ignore"):
        value = np.nan_to_num(res.positions * val)
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


def run_backtest(request: dict, output_dir: str | Path | None = None,
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
            from concordance.data.crsp import load_sp500_panel
            from concordance.data.ingest import default_lake

            lake = Path(req["data"]["lake"] or default_lake())
            log(f"loading panel {req['data']['start']} .. {req['data']['end']}")
            panel = load_sp500_panel(req["data"]["start"], req["data"]["end"], lake=lake,
                                     warmup_sessions=strategy.warmup(),
                                     unlock_reason=req["lockbox"]["unlock_reason"])
            lm = lake / "manifest.json"
            manifest["data"] = {"lake": str(lake),
                                "lake_manifest_sha256": _sha256_file(lm) if lm.exists()
                                else None}
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

        if isinstance(e, CoverageError | LockboxViolation | PanelError | FileNotFoundError):
            result["status"] = "data_error"
        else:
            result["status"] = "error"
        diag["messages"].append(f"{type(e).__name__}: {e}")
        diag["traceback"] = traceback.format_exc(limit=-8)
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["elapsed_s"] = round(time.time() - t0, 2)
    result = _jsonable(result)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    result["output_dir"] = str(out)
    return result


def _evaluate(req, panel, strategy, fresh, out, lake, log, diag) -> dict:
    from concordance.reconcile import reconcile
    from concordance.report import html_report, metrics
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
    c, q = req["costs"], req["liquidity"]
    costs = CostModel(commission_bps=c["commission_bps"],
                      default_half_spread_bps=c["default_half_spread_bps"],
                      use_prior_quoted_spread=c["use_prior_quoted_spread"])
    liquidity = LiquidityModel(**q) if q is not None else None
    tables = exec_tables(panel, costs, liquidity)
    results = {}
    for e in req["engines"]:
        log(f"engine: {e}")
        results[e] = run_engine(e, panel, sched, tables, req["initial_capital"])
    report = reconcile(results) if len(results) > 1 else None
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
                          note=f"run {req['run_id']}")
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
    if req["report"]:
        html_report(res, out / "report.html", title=getattr(strategy, "name", "strategy"),
                    extra={"reconciliation": diag.get("reconciliation", "single engine"),
                           "metrics": json.dumps(_jsonable(m), indent=2, default=str)},
                    benchmark=bench, start=start)
        artifacts["report"] = "report.html"
    artifacts.update(strategy="strategy.py", request="request.json")
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
