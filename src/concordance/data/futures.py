"""LSEG Datastream futures (WRDS library trdstrm) -> the lake, and futures panels from it.

Source: one WRDS "Futures Contracts" export (daily bars joined with contract details),
converted to parquet once. Ingest keeps the markets in MARKETS and writes:
  lake/futures_bars.parquet       root, futcode, date, open, high, low, settle, volume, oi
  lake/futures_contracts.parquet  root, futcode, contract month/year, last trade date,
                                  first notice date, the roll deadline and how it was set

Data rules (raise, never repair silently):
- The export repeats some (contract, date) rows; copies that differ only in an unrelated
  Datastream column `P` are dropped. Copies that disagree on any price, volume or open
  interest raise.
- Only the liquid contract months of each market are kept (e.g. gold G J M Q V Z), so a
  continuous series never holds a thin serial month.
- Roll deadline = the last session the position may be held:
    cash-settled or delivered after the last trade (indexes, FX, energy, hogs): last trade;
    physically delivered with an earlier notice period (Treasuries, metals, grains, cattle):
    the earliest of the last trade, the day before first notice, and the last business day
    of the month before the contract month. The month rule covers old contracts whose
    first-notice date is missing and is never later than the exchange's first notice.
  A missing last-trade date falls back to the contract's last bar ("last_bar" source).
Multipliers convert Datastream's quote units to USD per 1.0 of price (checked on
2010-03-01: grains and livestock are in cents, the yen per 100 yen).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from concordance.futures.continuous import Contract, build_continuous
from concordance.futures.panel import Spec, build_futures_panel

MONTH_CODES = "FGHJKMNQUVXZ"
ALL = MONTH_CODES
QUARTERLY = "HMUZ"


@dataclass(frozen=True)
class Market:
    root: str
    contrcode: int
    sector: str
    multiplier: float  # USD per 1.0 of Datastream's quoted price
    tick: float  # in quoted price units
    margin_rate: float
    months: str  # liquid contract months
    early_delivery: bool  # physical delivery with notice before the last trade


# Datastream series codes verified against the export (name, currency, span) 2026-10-19.
MARKETS = {m.root: m for m in [
    Market("SP", 1382, "equity", 250, 0.10, 0.06, QUARTERLY, False),
    Market("NQ", 323, "equity", 20, 0.25, 0.06, QUARTERLY, False),
    Market("ZT", 3377, "rates", 2000, 0.0078125, 0.01, QUARTERLY, True),
    Market("ZF", 2680, "rates", 1000, 0.0078125, 0.02, QUARTERLY, True),
    Market("ZN", 2682, "rates", 1000, 0.015625, 0.03, QUARTERLY, True),
    Market("ZB", 3271, "rates", 1000, 0.03125, 0.04, QUARTERLY, True),
    Market("6E", 2683, "fx", 125_000, 0.0001, 0.04, QUARTERLY, False),
    Market("6J", 2846, "fx", 125_000, 0.0001, 0.04, QUARTERLY, False),  # quoted per 100 yen
    Market("6B", 2759, "fx", 62_500, 0.0001, 0.04, QUARTERLY, False),
    Market("6C", 2776, "fx", 100_000, 0.0001, 0.04, QUARTERLY, False),
    Market("6S", 2807, "fx", 125_000, 0.0001, 0.05, QUARTERLY, False),
    Market("6A", 2744, "fx", 100_000, 0.0001, 0.05, QUARTERLY, False),
    Market("CL", 1986, "energy", 1000, 0.01, 0.10, ALL, False),
    Market("BRN", 1575, "energy", 1000, 0.01, 0.10, ALL, False),
    Market("NG", 2060, "energy", 10_000, 0.001, 0.12, ALL, False),
    Market("HO", 2029, "energy", 42_000, 0.0001, 0.10, ALL, False),
    Market("RB", 2091, "energy", 42_000, 0.0001, 0.10, ALL, False),
    Market("GC", 2020, "metals", 100, 0.10, 0.07, "GJMQVZ", True),
    Market("SI", 2108, "metals", 5000, 0.005, 0.09, "HKNUZ", True),
    Market("HG", 2026, "metals", 25_000, 0.0005, 0.07, "HKNUZ", True),
    Market("ZC", 3247, "grains", 50, 0.25, 0.06, "HKNUZ", True),  # cents per bushel
    Market("ZW", 3272, "grains", 50, 0.25, 0.07, "HKNUZ", True),
    Market("ZS", 3376, "grains", 50, 0.25, 0.06, "FHKNX", True),
    Market("LE", 2675, "livestock", 400, 0.025, 0.05, "GJMQVZ", True),  # cents per pound
    Market("HE", 2676, "livestock", 400, 0.025, 0.06, "GJMNQVZ", False),  # cash-settled
]}
COMMISSION = 2.5  # USD per contract per side (broker, exchange and regulatory fees)
SLIPPAGE_TICKS = 1.0


def spec(root: str) -> Spec:
    m = MARKETS[root]
    return Spec(root, m.multiplier, m.tick, m.margin_rate, COMMISSION, SLIPPAGE_TICKS)


def _contract_year(contrdate: str, ltd: pd.Timestamp) -> int:
    """MMYY -> the century that puts the contract nearest its own last trade."""
    yy = int(contrdate[2:])
    return min((1900 + yy, 2000 + yy, 2100 + yy), key=lambda y: abs(y - ltd.year))


def convert_export(csv_gz: Path, parquet: Path) -> int:
    """The raw WRDS "Futures Contracts" CSV (gzip) -> parquet with typed columns.
    Quoting is explicit: some contract names contain commas."""
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute(f"""
        COPY (SELECT FutCode::BIGINT futcode, Date_::DATE date, Open_::DOUBLE open,
              High::DOUBLE high, Low::DOUBLE low, Settlement::DOUBLE settle,
              Volume::DOUBLE volume, OpenInterest::DOUBLE oi, P::VARCHAR p,
              ContrCode::BIGINT contrcode, DSMnem dsmnem, ContrName contrname,
              ContrDate::VARCHAR contrdate, ISOCurrCode currency, UnitDesc unit,
              StartDate::DATE startdate, LastTrdDate::DATE lasttrddate,
              FirstNoticeDate::DATE firstnoticedate, ExchTickerSymb ticker,
              TrdStatCode trdstat
        FROM read_csv('{Path(csv_gz).as_posix()}', header=true, all_varchar=true, delim=',',
                      quote='"', escape='"', strict_mode=true))
        TO '{Path(parquet).as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    n = con.execute(f"SELECT count(*) FROM '{Path(parquet).as_posix()}'").fetchone()[0]
    con.close()
    return int(n)


def ingest(raw: Path, lake: Path) -> dict:
    """Write the lake's futures tables from the WRDS export (.csv.gz or converted .parquet)
    and record them in the lake manifest. Returns stats."""
    raw = Path(raw)
    if raw.name.endswith(".csv.gz"):
        parquet = raw.with_name(raw.name[: -len(".csv.gz")] + ".parquet")
        convert_export(raw, parquet)
        raw = parquet
    codes = {m.contrcode: m.root for m in MARKETS.values()}
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.register("codes", pd.DataFrame({"contrcode": list(codes), "root": list(codes.values())}))
    src = f"read_parquet('{Path(raw).as_posix()}')"
    conflicts = con.execute(f"""
        SELECT count(*) FROM (SELECT futcode, date FROM {src} JOIN codes USING (contrcode)
        GROUP BY 1, 2 HAVING count(DISTINCT (coalesce(open, -1), coalesce(high, -1),
          coalesce(low, -1), coalesce(settle, -1), coalesce(volume, -1), coalesce(oi, -1))) > 1)
    """).fetchone()[0]
    if conflicts:
        raise ValueError(f"{conflicts} contract-days have conflicting duplicate rows")
    bars = con.execute(f"""
        SELECT DISTINCT c.root, f.futcode, f.date, f.open, f.high, f.low, f.settle, f.volume,
               f.oi FROM {src} f JOIN codes c USING (contrcode)
        WHERE f.settle IS NOT NULL AND f.settle > 0 ORDER BY c.root, f.futcode, f.date
    """).df()
    info = con.execute(f"""
        SELECT c.root, f.futcode, any_value(f.contrdate) AS contrdate,
               any_value(f.lasttrddate) AS lasttrddate,
               any_value(f.firstnoticedate) AS firstnoticedate, max(f.date) AS last_bar
        FROM {src} f JOIN codes c USING (contrcode) GROUP BY 1, 2
    """).df()
    con.close()
    rows = []
    for r in info.itertuples():
        ltd_src = "vendor" if pd.notna(r.lasttrddate) else "last_bar"
        ltd = pd.Timestamp(r.lasttrddate if pd.notna(r.lasttrddate) else r.last_bar)
        if not isinstance(r.contrdate, str) or len(r.contrdate) != 4:
            continue
        month = int(r.contrdate[:2])
        year = _contract_year(r.contrdate, ltd)
        m = MARKETS[r.root]
        deadline, how = ltd, "last_trade"
        if m.early_delivery:
            month_rule = pd.Timestamp(year, month, 1) - pd.offsets.BDay(1)
            cands = [(ltd, "last_trade"), (month_rule, "month_before_delivery")]
            if pd.notna(r.firstnoticedate):
                cands.append((pd.Timestamp(r.firstnoticedate) - pd.Timedelta(days=1),
                              "first_notice"))
            deadline, how = min(cands, key=lambda c: c[0])
        rows.append({"root": r.root, "futcode": int(r.futcode), "month": month, "year": year,
                     "month_code": MONTH_CODES[month - 1], "liquid": MONTH_CODES[month - 1]
                     in m.months, "last_trade": ltd, "last_trade_source": ltd_src,
                     "first_notice": r.firstnoticedate, "deadline": deadline,
                     "deadline_rule": how})
    contracts = pd.DataFrame(rows).sort_values(["root", "deadline"]).reset_index(drop=True)
    lake = Path(lake)
    bars.to_parquet(lake / "futures_bars.parquet", index=False)
    contracts.to_parquet(lake / "futures_contracts.parquet", index=False)
    stats = {"bars": len(bars), "contracts": len(contracts),
             "liquid_contracts": int(contracts["liquid"].sum()),
             "first": str(bars["date"].min().date()), "last": str(bars["date"].max().date())}
    _record_in_manifest(lake, raw, bars, contracts)
    return stats


def _record_in_manifest(lake: Path, raw: Path, bars: pd.DataFrame, contracts: pd.DataFrame):
    import hashlib
    import json
    from datetime import UTC, datetime

    path = lake / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"files": {}}
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    manifest.setdefault("files", {})["futures_bars"] = {
        "status": "ok", "rows": len(bars), "start": str(bars["date"].min().date()),
        "end": str(bars["date"].max().date()), "sources": {raw.name: digest},
        "markets": sorted(bars["root"].unique().tolist()),
        "ingested_at": datetime.now(UTC).isoformat()}
    manifest["files"]["futures_contracts"] = {"status": "ok", "rows": len(contracts),
                                              "sources": {raw.name: digest}}
    path.write_text(json.dumps(manifest, indent=2))


FIRST_YEAR_DAYS = 400  # a roll gap this soon after a market's first bar is a vendor artifact


def _chain(contracts: list[Contract], sessions_before: int) -> tuple[list[Contract], list]:
    """The contracts actually held, in order. A next contract with no settle on or before
    the front's roll date (a hole in the vendor's history) is skipped for the following
    one; every skip is returned so it is reported, never silent."""
    from concordance.calendar import calendar

    cal = calendar("XNYS")
    chain, skipped = [contracts[0]], []
    rest = contracts[1:]
    while rest:
        front = chain[-1]
        dl = cal.date_to_session(front.deadline, direction="previous")
        cand = cal.session_offset(dl, -sessions_before).tz_localize(None) \
            if sessions_before else pd.Timestamp(dl).tz_localize(None)
        front_days = front.bars["close"].dropna().index
        for k, nxt in enumerate(rest):
            both = front_days.intersection(nxt.bars["close"].dropna().index)
            if len(both[both <= cand]):
                skipped += [(c.id, front.id) for c in rest[:k]]
                chain.append(nxt)
                rest = rest[k + 1:]
                break
        else:
            raise RollGap(front, f"{front.id}: no later contract has a settle by its roll "
                                 f"date {cand.date()}")
    return chain, skipped


def load_series(root: str, lake: Path, sessions: pd.DatetimeIndex, sessions_before: int = 5
                ) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Panama continuous series for one market over `sessions` (XNYS days only).
    Returns (series, roll table, skipped contracts). Raises if the market has no data."""
    lake = Path(lake)
    cons = pd.read_parquet(lake / "futures_contracts.parquet", filters=[("root", "==", root)])
    cons = cons[cons["liquid"]].sort_values("deadline")
    bars = pd.read_parquet(lake / "futures_bars.parquet", filters=[("root", "==", root)])
    bars["date"] = pd.to_datetime(bars["date"])
    first_bar = bars["date"].min()
    bars = bars[bars["date"].isin(sessions)]
    lo, hi = sessions[0], sessions[-1]
    # Contracts that can be held inside the window: deadline on/after its start, plus only
    # the first one whose deadline falls after its end (later ones are never held).
    after = cons[cons["deadline"] > hi]
    cons = cons[(cons["deadline"] >= lo) & (cons["deadline"] <= hi)]
    if len(after):
        cons = pd.concat([cons, after.iloc[:1]])
    contracts = []
    for c in cons.itertuples():
        b = bars[bars["futcode"] == c.futcode].set_index("date").sort_index()
        b = b[["open", "settle"]].rename(columns={"settle": "close"})
        b["open"] = b["open"].where(b["open"] > 0)
        if b.empty:
            continue  # no bar inside the window at all
        contracts.append(Contract(f"{root}{c.month_code}{c.year}", last_trade=c.deadline,
                                  bars=b))
    if len(contracts) < 2:
        raise NoData(f"{root}: fewer than two contracts in the window")
    try:
        chain, skipped = _chain(contracts, sessions_before)
    except RollGap as e:
        # Datastream's first year of a market holds only contracts expiring that year (LE
        # 1979), so its first roll can be impossible: the market starts after that contract,
        # reported as skipped. A hole anywhere later is real and raises.
        if e.front.deadline - first_bar > pd.Timedelta(days=FIRST_YEAR_DAYS):
            raise
        k = next(i for i, c in enumerate(contracts) if c is e.front)
        dropped, contracts = contracts[:k + 1], contracts[k + 1:]
        if len(contracts) < 2:
            raise NoData(f"{root}: no rollable contracts after its first year") from None
        chain, skipped = _chain(contracts, sessions_before)
        skipped = [(c.id, "first-year history") for c in dropped] + skipped
    series, table = build_continuous(chain, sessions_before=sessions_before)
    return series[(series.index >= lo) & (series.index <= hi)], table, skipped


class RollGap(ValueError):
    """No later contract has a settle by `front`'s roll date."""

    def __init__(self, front: Contract, message: str):
        super().__init__(message)
        self.front = front


class NoData(ValueError):
    """The market has no contracts in the requested window (e.g. NQ before 1999)."""


def load_futures_panel(start, end, lake: Path, roots=None, sessions_before: int = 5,
                       warmup_sessions: int = 0):
    from concordance.data.crsp import xnys_sessions

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sessions = xnys_sessions(start, end)
    if warmup_sessions:
        before = xnys_sessions(start - pd.Timedelta(days=int(warmup_sessions * 1.6) + 10),
                               start - pd.Timedelta(days=1))
        sessions = before[-warmup_sessions:].append(sessions)
    roots = sorted(roots or MARKETS)
    series, rolls, skipped = {}, {}, {}
    for root in roots:
        try:
            s, t, sk = load_series(root, lake, sessions, sessions_before)
        except NoData:
            continue  # only "no contracts in this window"; any other problem raises
        series[root], rolls[root], skipped[root] = s, t, sk
    from concordance.data.riskfree import load_rf

    rf = load_rf(lake)
    missing_rf = sessions.difference(rf.index)
    if len(missing_rf) > 5:
        raise ValueError(f"risk-free rate missing on {len(missing_rf)} sessions")
    panel = build_futures_panel(series, {r: spec(r) for r in series}, rf, sessions=sessions)
    panel.meta["rolls"] = rolls
    panel.meta["skipped_contracts"] = {r: v for r, v in skipped.items() if v}
    panel.meta["sectors"] = {r: MARKETS[r].sector for r in series}
    return panel


def roll_report(panel) -> pd.DataFrame:
    """Rolls per market with the roll gap as a share of price: a quick data sanity view."""
    out = []
    for root, t in panel.meta["rolls"].items():
        if len(t):
            out.append({"root": root, "rolls": len(t),
                        "median_abs_gap": float(np.median(np.abs(t["gap"])))})
    return pd.DataFrame(out)
