"""concordance command line.

  concordance ingest RAW_DIR               validate WRDS exports into the lake
  concordance permnos MEMBERSHIP_FILE       PERMNO list for the CRSP web-form upload
  concordance run STRATEGY.py --start --end  run all engines, reconcile, validate, report
  concordance demo                          the same pipeline on synthetic data (no WRDS)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.costs import CostModel
from concordance.panel import MarketPanel
from concordance.report import html_report, metrics
from concordance.runner import ALL_ENGINES, run
from concordance.strategy import Strategy, weight_schedule

ALIASES = {"ref": "reference", "vbt": "vectorbt", "naut": "nautilus"}


def load_strategy(path: str) -> Strategy:
    spec = importlib.util.spec_from_file_location("user_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "STRATEGY"):
        return mod.STRATEGY
    raise SystemExit(f"{path} must define STRATEGY = <Strategy instance>")


def evaluate(panel: MarketPanel, strategy: Strategy, engines, costs: CostModel,
             init_cash: float, program: str, report: Path | None, placebo: int,
             lookahead_cuts: int, ledger_root: Path | None = None) -> dict:
    from concordance.validation.dsr import deflated_sharpe
    from concordance.validation.ledger import TrialLedger
    from concordance.validation.lookahead import check_truncation_invariance
    from concordance.validation.placebo import random_selection_placebo

    if lookahead_cuts:
        print(f"lookahead: truncation check at {lookahead_cuts} cut points ...", flush=True)
        check_truncation_invariance(lambda p: strategy, panel, cuts=lookahead_cuts)
    out = run(panel, strategy, costs=costs, init_cash=init_cash, engines=engines,
              strict=False)
    if out.report is not None:
        print(out.report.summary())
        if not out.report.ok:
            raise SystemExit("ENGINES DISAGREE: result rejected.")
    res = out.result
    start = panel.meta.get("warmup_sessions", 0)
    benchmark = panel.meta.get("benchmark")
    m = metrics(res, start=start, benchmark=benchmark)
    gap_exits = listing_gap_exits(panel, res)
    if len(gap_exits):
        m["listing_gap_exits"] = len(gap_exits)
        print("WARNING: positions exited at a listing gap, valued at the last price before "
              "the gap (CRSP has no return across it):\n" + gap_exits.to_string(index=False))

    rets = res.returns.iloc[max(start, 1):]
    ledger = TrialLedger(root=ledger_root, program=program)
    params = getattr(strategy, "params", {})
    ledger.record(getattr(strategy, "name", type(strategy).__name__), params, rets)
    n_eff = max(ledger.n_effective(), 1.0)
    d = deflated_sharpe(rets.to_numpy(), n_trials=n_eff)
    m["dsr"] = d.annualized()
    m["trials_recorded"] = ledger.n_trials()

    if placebo:
        w, has = weight_schedule(strategy, panel)
        pl = random_selection_placebo(panel, w, has, costs.fee_rates(panel), init_cash,
                                      n=placebo)
        m["placebo_p_value"] = pl.p_value
    if report:
        extra = {"reconciliation": out.report.summary() if out.report else "single engine",
                 "metrics": json.dumps(m, indent=2, default=str)}
        html_report(res, report, title=getattr(strategy, "name", "strategy"), extra=extra,
                    benchmark=benchmark, start=start)
        m["report"] = str(report)
    return m


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


def cmd_ingest(a):
    from concordance.data.ingest import ingest

    print(json.dumps(ingest(Path(a.raw_dir), Path(a.lake) if a.lake else None), indent=2))


def cmd_permnos(a):
    """Membership download -> one PERMNO per line, for the CRSP web form's code-list upload."""
    from concordance.data.ingest import clean_membership, read_export
    from concordance.data.schema import SP500_MEMBERSHIP

    spells, _ = clean_membership(read_export(Path(a.membership_file), SP500_MEMBERSHIP))
    if a.since:
        spells = spells[spells["end"] >= pd.Timestamp(a.since)]
    ids = sorted(spells["permno"].unique())
    Path(a.output).write_text("\n".join(str(i) for i in ids) + "\n")
    print(f"wrote {len(ids)} PERMNOs to {a.output}")


def cmd_run(a):
    from concordance.data.crsp import load_sp500_panel

    strategy = load_strategy(a.strategy)
    panel = load_sp500_panel(a.start, a.end, lake=Path(a.lake) if a.lake else None,
                             warmup_sessions=strategy.warmup(),
                             unlock_reason=a.unlock_lockbox)
    print(f"panel: {panel.shape[0]} sessions x {panel.shape[1]} securities, "
          f"coverage {panel.meta['coverage_overall']:.2%}")
    from concordance.data.ingest import default_lake

    m = evaluate(panel, strategy, _engines(a.engines), _costs(a), a.init_cash, a.program,
                 Path(a.report) if a.report else None, a.placebo, a.lookahead_cuts,
                 ledger_root=Path(a.lake or default_lake()) / "ledger")
    print(json.dumps(m, indent=2, default=str))


def cmd_demo(a):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "property"))
    from random_market import RandomRebalancer, random_long_frame

    from concordance.panel import build_panel

    panel = build_panel(random_long_frame(42, n_assets=40, n_sessions=400))
    m = evaluate(panel, RandomRebalancer(42, p=0.1), _engines(a.engines), _costs(a), 1e6,
                 "demo", Path(a.report) if a.report else None, 0, 0)
    print(json.dumps(m, indent=2, default=str))


def _engines(s: str):
    names = [ALIASES.get(x.strip(), x.strip()) for x in s.split(",")]
    bad = set(names) - set(ALL_ENGINES)
    if bad:
        raise SystemExit(f"unknown engines {bad}; choose from {ALL_ENGINES}")
    return names


def _costs(a) -> CostModel:
    return CostModel(commission_bps=a.commission_bps,
                     default_half_spread_bps=a.default_half_spread_bps)


def main(argv=None):
    p = argparse.ArgumentParser(prog="concordance")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="validate WRDS exports into the lake")
    s.add_argument("raw_dir")
    s.add_argument("--lake")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("permnos", help="PERMNO list file for the CRSP web-query upload")
    s.add_argument("membership_file")
    s.add_argument("-o", "--output", default="permnos.txt")
    s.add_argument("--since", default="1994-01-01",
                   help="only members whose spell ends on/after this date")
    s.set_defaults(fn=cmd_permnos)

    def common(s):
        s.add_argument("--engines", default="reference,vectorbt,nautilus")
        s.add_argument("--commission-bps", type=float, default=1.0)
        s.add_argument("--default-half-spread-bps", type=float, default=5.0)
        s.add_argument("--report", help="write an HTML tear sheet here")

    s = sub.add_parser("run", help="backtest a strategy file on CRSP S&P 500 data")
    s.add_argument("strategy")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--lake")
    s.add_argument("--init-cash", type=float, default=1_000_000.0)
    s.add_argument("--program", default="default", help="trial-ledger program name")
    s.add_argument("--placebo", type=int, default=0, help="random-selection placebo runs")
    s.add_argument("--lookahead-cuts", type=int, default=3)
    s.add_argument("--unlock-lockbox", help="reason for a logged one-shot lockbox read")
    common(s)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("demo", help="full pipeline on a synthetic market")
    common(s)
    s.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    pd.set_option("display.width", 120)
    np.set_printoptions(linewidth=120)
    a.fn(a)


if __name__ == "__main__":
    main()
