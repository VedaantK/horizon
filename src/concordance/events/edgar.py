"""SEC EDGAR earnings events (8-K Item 2.02) with exact, point-in-time timing.

Salvaged from QuantMega `eventlab/{events,edgar}.py` (the most careful code in that repo):
- acceptance times converted UTC -> America/New_York;
- data.sec.gov sometimes labels Eastern time as UTC ('Z'); suspect values are re-read from
  the raw submission header (`ACCEPTANCE-DATETIME`), which is unambiguous;
- BMO / DMH / AMC classification -> day0 = first session whose prices reflect the news.

Fixed here: QuantMega used a fixed 16:00 close, so filings after 13:00 on early-close days
(day after Thanksgiving, Christmas Eve, July 3) were dated to the wrong session. The open
and close of each session now come from the exchange calendar.

Features computed on day0 are known at the day0 close; the earliest entry is day0+1's open.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from concordance.calendar import calendar

ITEM_202_START = pd.Timestamp("2004-08-23")
# Converted-ET hours where an ET-mislabelled-as-UTC stamp would move day0 (QuantMega,
# verified against raw headers): the shift is 4-5h, so only these hours can cross an
# open/close boundary or midnight.
SUSPECT_ET_HOURS = set(range(0, 6)) | set(range(10, 16))
NY = "America/New_York"


# ── timing ──────────────────────────────────────────────────────────────────
def session_bounds(sessions: pd.DatetimeIndex, cal_name: str = "XNYS"):
    """(open, close) in naive New York time for each session (early closes included)."""
    cal = calendar(cal_name)
    opens = cal.schedule.loc[sessions, "open"].dt.tz_convert(NY).dt.tz_localize(None)
    closes = cal.schedule.loc[sessions, "close"].dt.tz_convert(NY).dt.tz_localize(None)
    return opens.to_numpy(), closes.to_numpy()


def assign_day0(accepted_et: pd.Series, cal_name: str = "XNYS") -> pd.DataFrame:
    """Return timing (BMO/DMH/AMC/NON-SESSION) and day0 for naive-ET acceptance times."""
    cal = calendar(cal_name)
    t = pd.to_datetime(accepted_et)
    lo = t.min().normalize() - pd.Timedelta(days=10)
    hi = t.max().normalize() + pd.Timedelta(days=10)
    sessions = pd.DatetimeIndex(cal.sessions_in_range(lo, hi)).tz_localize(None)
    opens, closes = session_bounds(sessions, cal_name)
    day = t.dt.normalize().to_numpy()
    pos = np.searchsorted(sessions.values, day, side="left")
    is_session = (pos < len(sessions)) & (sessions.values[np.minimum(pos, len(sessions) - 1)]
                                          == day)
    p = np.minimum(pos, len(sessions) - 1)
    tv = t.to_numpy()
    before_open = is_session & (tv < opens[p])
    during = is_session & (tv >= opens[p]) & (tv < closes[p])
    after = is_session & (tv >= closes[p])
    timing = np.select([before_open, during, after], ["BMO", "DMH", "AMC"], "NON-SESSION")
    # Non-session days (weekend/holiday): first session after -> pos already points there.
    day0_pos = pos + after.astype(int)
    day0 = pd.Series(pd.NaT, index=accepted_et.index, dtype="datetime64[ns]")
    ok = day0_pos < len(sessions)
    day0[ok] = sessions.values[day0_pos[ok]]
    return pd.DataFrame({"timing": timing, "day0": day0}, index=accepted_et.index)


def utc_to_et(utc: pd.Series) -> pd.Series:
    return pd.to_datetime(utc, utc=True).dt.tz_convert(NY).dt.tz_localize(None)


def needs_header_check(accepted_et: pd.Series) -> pd.Series:
    return accepted_et.dt.hour.isin(SUSPECT_ET_HOURS)


def parse_header_acceptance(text: str) -> pd.Timestamp | None:
    """ACCEPTANCE-DATETIME from a raw EDGAR submission header (already Eastern time)."""
    m = re.search(r"ACCEPTANCE-DATETIME>\s*(\d{14})", text)
    return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d%H%M%S")) if m else None


def earnings_8k(filings: pd.DataFrame) -> pd.DataFrame:
    """Original 8-Ks carrying Item 2.02 from a submissions table (`recent` block columns)."""
    f = filings[(filings["form"] == "8-K")
                & filings["items"].fillna("").str.contains(r"\b2\.02\b")].copy()
    if f.empty:
        return f
    f["accepted_et"] = utc_to_et(f["acceptanceDateTime"])
    f = f[f["accepted_et"] >= ITEM_202_START]
    return f.rename(columns={"accessionNumber": "accession"}).drop_duplicates("accession")


def cluster(events: pd.DataFrame, gap_sessions: int = 20) -> pd.DataFrame:
    """Per asset, keep the first event of each cluster (later 2.02s within `gap_sessions`
    sessions are usually corrections or a parent/subsidiary duplicate). Adds n_clustered.

    Known limitation (from the QuantMega audit): a pre-announcement followed by the actual
    report within the gap keeps the pre-announcement. Inspect `n_clustered > 0` events.
    """
    ev = events.sort_values(["asset", "day0", "accepted_et"]).copy()
    sess = pd.DatetimeIndex(calendar().sessions).tz_localize(None)
    ev["_s"] = sess.searchsorted(ev["day0"].values)
    keep = np.zeros(len(ev), dtype=bool)
    n_cl = np.zeros(len(ev), dtype=int)
    last_a, last_s, last_i = None, -(10**9), -1
    for i, (a, s) in enumerate(zip(ev["asset"].values, ev["_s"].values, strict=True)):
        if a != last_a or s - last_s > gap_sessions:
            keep[i] = True
            last_a, last_s, last_i = a, s, i
        else:
            n_cl[last_i] += 1
    ev["n_clustered"] = n_cl
    return ev[keep].drop(columns="_s").reset_index(drop=True)


def event_flags(events: pd.DataFrame, sessions: pd.DatetimeIndex, assets) -> np.ndarray:
    """(T, N) bool: True on each event's day0 row (known at that session's close)."""
    out = np.zeros((len(sessions), len(assets)), dtype=bool)
    col = {a: j for j, a in enumerate(assets)}
    for a, d in zip(events["asset"], events["day0"], strict=True):
        j = col.get(a)
        i = sessions.searchsorted(d)
        if j is not None and i < len(sessions) and sessions[i] == d:
            out[i, j] = True
    return out


# ── network (SEC fair-access: <=10 req/s, a real User-Agent) ────────────────
class SecClient:
    def __init__(self, cache_dir: Path, user_agent: str | None = None):
        ua = user_agent or os.environ.get("SEC_USER_AGENT")
        if not ua:
            raise RuntimeError("set SEC_USER_AGENT='Your Name your@email' (SEC requires it)")
        self.ua, self.cache, self._last = ua, Path(cache_dir), 0.0
        self.cache.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, name: str, headers: dict | None = None) -> str | None:
        import requests

        path = self.cache / name
        if path.exists():
            txt = path.read_text()
            return None if txt == "null" else txt
        for attempt in range(4):
            wait = 0.125 - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            r = requests.get(url, headers={"User-Agent": self.ua, **(headers or {})}, timeout=30)
            if r.status_code == 404:
                path.write_text("null")
                return None
            if r.status_code in (429, 503):
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            path.write_text(r.text)
            return r.text
        raise RuntimeError(f"SEC fetch failed: {url}")

    def submissions(self, cik: int) -> pd.DataFrame:
        """All filings for a CIK (recent block plus older pages)."""
        name = f"CIK{int(cik):010d}.json"
        txt = self._get(f"https://data.sec.gov/submissions/{name}", name)
        if txt is None:
            return pd.DataFrame()
        sub = json.loads(txt)
        cols = ("form", "filingDate", "acceptanceDateTime", "items", "accessionNumber")
        rows = {c: list(sub["filings"]["recent"].get(c, [])) for c in cols}
        for page in sub["filings"].get("files", []):
            ptxt = self._get(f"https://data.sec.gov/submissions/{page['name']}", page["name"])
            if ptxt:
                p = json.loads(ptxt)
                for c in cols:
                    rows[c].extend(p.get(c, []))
        df = pd.DataFrame(rows)
        df["cik"] = int(cik)
        return df

    def header_acceptance(self, cik: int, accession: str) -> pd.Timestamp | None:
        acc = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{accession}.txt"
        txt = self._get(url, f"hdr_{accession}.txt", {"Range": "bytes=0-4000"})
        return parse_header_acceptance(txt) if txt else None

    def earnings_events(self, cik: int, asset) -> pd.DataFrame:
        f = earnings_8k(self.submissions(cik))
        if f.empty:
            return f
        f["time_source"] = "json"
        for i in f.index[needs_header_check(f["accepted_et"])]:
            t = self.header_acceptance(cik, f.at[i, "accession"])
            f.at[i, "time_source"] = "header" if t is not None else "json_unverified"
            if t is not None:
                f.at[i, "accepted_et"] = t
        f = f.join(assign_day0(f["accepted_et"]))
        f["asset"] = asset
        return f
