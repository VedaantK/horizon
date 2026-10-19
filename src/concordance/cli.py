"""concordance command line.

  concordance ingest RAW_DIR                 validate WRDS exports into the lake
  concordance permnos MEMBERSHIP_FILE         PERMNO list for the CRSP web-form upload
  concordance link-customers                  point-in-time supplier -> customer links
  concordance run STRATEGY.py --start --end   run all engines, reconcile, validate, report
  concordance backtest REQUEST.json           the same, driven by a JSON request (docs/RUN_API.md)
  concordance reproduce RUN_DIR               re-run a saved run and check it matches
  concordance diagnose RUN_DIR                where and why engines stopped agreeing
  concordance validate STRATEGY.py            check a strategy file against the contract
  concordance formalize IDEA.txt -o SPEC.json  plain-language idea -> StrategySpec (AI)
  concordance author SPEC.json --out DIR      StrategySpec -> validated strategy.py (AI)
  concordance demo                            the pipeline on synthetic data (no WRDS)
  concordance capabilities                    what can be tested, as JSON (for coordinators)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.api import ENGINES, reproduce, run_backtest

ALIASES = {"ref": "reference", "vbt": "vectorbt", "naut": "nautilus"}
REPO = Path(__file__).resolve().parents[2]


def cmd_ingest(a):
    from concordance.data.ingest import ingest

    print(json.dumps(ingest(Path(a.raw_dir), Path(a.lake) if a.lake else None), indent=2))


def cmd_link_customers(a):
    from concordance.data.customers import link_customers

    entry = link_customers(Path(a.lake) if a.lake else None,
                           Path(a.review) if a.review else None)
    print(json.dumps(entry, indent=2, default=str))


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


def _request_from_flags(a) -> dict:
    req = {
        "strategy": {"path": str(Path(a.strategy).resolve()),
                     "params": json.loads(a.params) if a.params else {}},
        "data": {"start": a.start, "end": a.end, "lake": a.lake},
        "initial_capital": a.init_cash,
        "engines": _engines(a.engines),
        "costs": {"commission_bps": a.commission_bps,
                  "default_half_spread_bps": a.default_half_spread_bps},
        "liquidity": None if a.no_liquidity else {"participation": a.participation,
                                                  "impact_coef": a.impact_coef},
        "validation": {"program": a.program, "placebo": a.placebo,
                       "lookahead_cuts": a.lookahead_cuts},
        "lockbox": {"unlock_reason": a.unlock_lockbox},
        "report": bool(a.report),
    }
    if a.run_id:
        req["run_id"] = a.run_id
    return req


def _print_result(res: dict, report_to: str | None = None) -> int:
    d = res.get("diagnostics", {})
    if d.get("reconciliation"):
        print(d["reconciliation"])
    for w in d.get("warnings", []):
        print(f"WARNING: {w}")
    for msg in d.get("messages", []):
        print(f"{res['status'].upper()}: {msg}")
    if res["status"] == "ok":
        print(json.dumps(res["metrics"], indent=2, default=str))
        if report_to and "report" in res.get("artifacts", {}):
            Path(report_to).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(res["output_dir"]) / "report.html", report_to)
            print(f"report: {report_to}")
    print(f"status: {res['status']}  run: {res.get('output_dir')}")
    return 0 if res["status"] == "ok" else 1


def _done(code: int):
    if code:
        sys.exit(code)


def cmd_run(a):
    res = run_backtest(_request_from_flags(a), a.out, quiet=False)
    _done(_print_result(res, a.report))


def cmd_backtest(a):
    request = json.loads(Path(a.request).read_text())
    path = Path(request.get("strategy", {}).get("path") or "")
    if path and not path.is_absolute():  # relative to the request file
        request["strategy"]["path"] = str((Path(a.request).parent / path).resolve())
    res = run_backtest(request, a.output, quiet=not a.verbose)
    print(json.dumps(res, indent=2, default=str))
    _done(0 if res["status"] == "ok" else 1)


def cmd_reproduce(a):
    out = reproduce(a.run_dir, a.output)
    print(json.dumps(out, indent=2))
    _done(0 if out["reproduced"] else 1)


def cmd_diagnose(a):
    from concordance.diagnose import diagnose, render

    out = diagnose(a.run_dir, _engines(a.engines) if a.engines else None, top=a.top)
    print(json.dumps(out, indent=2, default=str) if a.json else render(out))
    _done(0 if out["ok"] else 1)


def cmd_validate(a):
    from concordance.authoring.validate import validate_file

    real = tuple(a.real.split(":")) if a.real else None
    rep = validate_file(a.strategy, params=json.loads(a.params) if a.params else None,
                        real_window=real, timeout=a.timeout)
    print(json.dumps(rep, indent=2, default=str))
    _done(0 if rep["ok"] else 1)


def cmd_formalize(a):
    from concordance.authoring.drive import formalize

    idea = Path(a.idea).read_text() if Path(a.idea).is_file() else a.idea
    out = formalize(idea, a.tool, a.output, model=a.model)
    print(json.dumps(out, indent=2))
    _done(0 if out["ok"] else 1)


def cmd_author(a):
    from concordance.authoring.drive import author

    out = author(a.spec, a.tool, a.out, model=a.model, rounds=a.rounds,
                 allow_open_questions=a.allow_open_questions)
    short = {k: v for k, v in out.items() if k not in ("attempts", "validation")}
    if "attempts" in out:
        short["attempts"] = [{k: v for k, v in t.items() if k != "answer"}
                             for t in out["attempts"]]
        short["answer"] = out["attempts"][-1].get("answer")
    print(json.dumps(short, indent=2, default=str))
    _done(0 if out["ok"] else 1)


def cmd_capabilities(a):
    from concordance.capabilities import capabilities

    print(json.dumps(capabilities(a.lake), indent=2))


def cmd_demo(a):
    from concordance.panel import build_panel
    from concordance.synthetic import random_long_frame

    panel = build_panel(random_long_frame(42, n_assets=40, n_sessions=400))
    a.strategy, a.start, a.end = str(REPO / "examples" / "swing_pullback.py"), "x", "y"
    a.params, a.lake, a.init_cash, a.program = None, None, 1e6, "demo"
    a.placebo, a.lookahead_cuts, a.unlock_lockbox = 0, 2, None
    req = _request_from_flags(a)
    req["data"] = {"start": str(panel.sessions[0].date()), "end": str(panel.sessions[-1].date())}
    req["liquidity"] = None if a.no_liquidity else {**req["liquidity"], "window": 10,
                                                    "min_obs": 5}
    res = run_backtest(req, a.out, panel=panel, quiet=False)
    _done(_print_result(res, a.report))


def _engines(s: str):
    names = [ALIASES.get(x.strip(), x.strip()) for x in s.split(",")]
    bad = set(names) - set(ENGINES)
    if bad:
        raise SystemExit(f"unknown engines {bad}; choose from {ENGINES}")
    return names


def main(argv=None):
    p = argparse.ArgumentParser(prog="concordance")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="validate WRDS exports into the lake")
    s.add_argument("raw_dir")
    s.add_argument("--lake")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("link-customers",
                       help="match customer names and build lake/customer_links.parquet")
    s.add_argument("--lake")
    s.add_argument("--review", help="review CSV (default lake/customer_matches_review.csv); "
                   "fill decision=accept/reject and rerun to apply")
    s.set_defaults(fn=cmd_link_customers)

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
        s.add_argument("--participation", type=float, default=0.10,
                       help="max share of trailing ADV traded per asset per session")
        s.add_argument("--impact-coef", type=float, default=1.0,
                       help="square-root impact coefficient (0 = no impact)")
        s.add_argument("--no-liquidity", action="store_true",
                       help="turn off the volume cap and market impact")
        s.add_argument("--report", help="also copy an HTML tear sheet here")
        s.add_argument("--out", help="run directory (default runs/<run_id>)")
        s.add_argument("--run-id")

    s = sub.add_parser("run", help="backtest a strategy file on CRSP S&P 500 data")
    s.add_argument("strategy")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--params", help="JSON object passed to make_strategy(**params)")
    s.add_argument("--lake")
    s.add_argument("--init-cash", type=float, default=1_000_000.0)
    s.add_argument("--program", default="default", help="trial-ledger program name")
    s.add_argument("--placebo", type=int, default=0, help="random-selection placebo runs")
    s.add_argument("--lookahead-cuts", type=int, default=3)
    s.add_argument("--unlock-lockbox", help="reason for a logged one-shot lockbox read")
    common(s)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("backtest", help="run a JSON request (docs/RUN_API.md)")
    s.add_argument("request")
    s.add_argument("-o", "--output", help="run directory (default runs/<run_id>)")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser("reproduce", help="re-run a saved run directory and compare")
    s.add_argument("run_dir")
    s.add_argument("-o", "--output")
    s.set_defaults(fn=cmd_reproduce)

    s = sub.add_parser("diagnose", help="rerun a saved run's engines; show where they differ")
    s.add_argument("run_dir")
    s.add_argument("--engines", help="engines to compare with the reference "
                   "(default: the run's own)")
    s.add_argument("--top", type=int, default=10, help="rows per table")
    s.add_argument("--json", action="store_true", help="print the full JSON report")
    s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("validate", help="check a strategy file against the contract")
    s.add_argument("strategy")
    s.add_argument("--params", help="JSON object passed to make_strategy(**params)")
    s.add_argument("--real", help="also run START:END on real CRSP data (reference engine)")
    s.add_argument("--timeout", type=float, default=300.0)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("formalize", help="AI: plain-language idea -> StrategySpec JSON")
    s.add_argument("idea", help="a text file, or the idea itself in quotes")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--tool", choices=["claude", "codex"], default="claude")
    s.add_argument("--model")
    s.set_defaults(fn=cmd_formalize)

    s = sub.add_parser("author", help="AI: StrategySpec -> validated strategy.py")
    s.add_argument("spec")
    s.add_argument("--out", required=True, help="candidate workspace directory")
    s.add_argument("--tool", choices=["claude", "codex"], default="claude")
    s.add_argument("--model")
    s.add_argument("--rounds", type=int, default=3, help="max write/repair rounds")
    s.add_argument("--allow-open-questions", action="store_true")
    s.set_defaults(fn=cmd_author)

    s = sub.add_parser("capabilities", help="what Concordance can test, as JSON")
    s.add_argument("--lake")
    s.set_defaults(fn=cmd_capabilities)

    s = sub.add_parser("demo", help="full pipeline on a synthetic market")
    common(s)
    s.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    pd.set_option("display.width", 120)
    np.set_printoptions(linewidth=120)
    a.fn(a)


if __name__ == "__main__":
    main()
