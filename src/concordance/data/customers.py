"""Point-in-time supplier -> customer links from Compustat customer segments.

US filers must name every customer worth 10% or more of revenue. Compustat's historical
customer segments keep those disclosures as originally reported, one row per supplier,
fiscal year (`srcdate`) and customer name (`cnms`). This module turns them into links a
strategy can read without look-ahead:

1. Clean: company customers only, unnamed placeholders dropped, a customer split across
   business segments summed.
2. Match customer names to Compustat companies (then PERMNOs). Exact normalized names are
   matched automatically, using the link valid at the fiscal date to resolve reused names.
   Everything else goes to a review CSV; only rows the user accepts are used.
3. Availability: a fiscal year's links become visible at the first XNYS session after
   `srcdate + 6 months` (the Cohen-Frazzini convention, leaving room for a late 10-K) and
   stop after `srcdate + 18 months`. A supplier's newest visible fiscal year replaces its
   older ones, so a customer that drops below 10% disappears with the next filing.
4. `CustomerLinks.weights_at(i)` gives the (N, N) supplier x customer revenue-share matrix
   known at the close of session i. The panel carries it; `PointInTimeView.customer_weights`
   exposes it. Shares with no disclosed amount are imputed at the 10% disclosure threshold
   and flagged, so a strategy can drop them.

Limitation: when a newer filing names no company customers at all, the supplier's older
links stay visible until they expire.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

AVAILABLE_LAG = pd.DateOffset(months=6)
EXPIRY = pd.DateOffset(months=18)
IMPUTED_SHARE = 0.10  # disclosure threshold, used when a customer's sales are not given
FUZZY_MIN = 0.86  # review suggestions only; never used without the user's acceptance

SUFFIXES = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COS", "LTD", "LIMITED",
    "LLC", "LP", "LLP", "PLC", "SA", "AG", "NV", "BV", "SE", "SPA", "AB", "ASA", "OYJ",
    "GMBH", "KK", "HOLDINGS", "HOLDING", "HLDGS", "GROUP", "GRP", "THE", "COM", "NEW", "DEL",
}
# Short forms filers commonly use, keyed and valued by normalized names.
ALIASES = {
    "IBM": "INTERNATIONAL BUSINESS MACHINES",
    "GE": "GENERAL ELECTRIC",
    "GM": "GENERAL MOTORS",
    "HEWLETT PACKARD": "HP",
    "HP ENTERPRISE": "HEWLETT PACKARD ENTERPRISE",
    "HPE": "HEWLETT PACKARD ENTERPRISE",
    "P AND G": "PROCTER AND GAMBLE",
    "J AND J": "JOHNSON AND JOHNSON",
    "WALMART STORES": "WALMART",
    "WAL MART STORES": "WALMART",
    "WAL MART": "WALMART",
    "ATT": "AT AND T",
    "UPS": "UNITED PARCEL SERVICE",
}
UNNAMED = re.compile(
    r"^(NOT REPORTED|NONE|N ?A|UNKNOWN|VARIOUS|OTHERS?|"
    r"(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+) (MAJOR |LARGEST |SIGNIFICANT )?"
    r"(CUSTOMERS?|CLIENTS?|DISTRIBUTORS?|RETAILERS?)|"
    r"(MAJOR |LARGEST |SIGNIFICANT |TOP )?(CUSTOMERS?|CLIENTS?|DISTRIBUTORS?)( [A-Z0-9])?|"
    r"CUSTOMER [A-Z0-9]+|[A-Z0-9] CUSTOMER)$")


def normalize_name(name) -> str:
    """Upper case, '&' -> AND, punctuation dropped, legal suffixes removed."""
    if not isinstance(name, str):
        return ""
    s = name.upper().replace("&", " AND ")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = s.split()
    while toks and toks[-1] in SUFFIXES:
        toks.pop()
    while toks and toks[0] == "THE":
        toks.pop(0)
    return " ".join(toks)


def gvkey_str(s: pd.Series) -> pd.Series:
    """Compustat gvkeys as 6-digit strings, whether a CSV gave them as text, int or float."""
    num = pd.to_numeric(s, errors="coerce")
    if num.isna().any():
        from concordance.data.ingest import IngestError

        raise IngestError(f"unreadable gvkey values, e.g. {s[num.isna()].iloc[0]!r}")
    if not (num == num.round()).all():
        from concordance.data.ingest import IngestError

        raise IngestError("gvkey values must be whole numbers")
    return num.astype("int64").astype(str).str.zfill(6)


def _alias(norm: str) -> str:
    return ALIASES.get(norm, norm)


# ------------------------------------------------------------------------------ ingest


def clean_seg_customer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Company customers with a name; one row per (supplier, fiscal year, customer name)."""
    n_in = len(df)
    # A few WRDS rows carry no supplier id at all; they can't be attributed, so they are
    # dropped and counted.
    blank = df["gvkey"].isna() | (df["gvkey"].astype(str).str.strip() == "")
    df = df[~blank].copy()
    df["gvkey"] = gvkey_str(df["gvkey"])
    if "datadate" in df.columns and df["datadate"].notna().any():
        # Source documents restate earlier years; keep each filing's own fiscal year.
        df = df[df["datadate"].isna() | (df["datadate"] == df["srcdate"])]
    company = df["ctype"].astype(str).str.upper() == "COMPANY"
    df = df[company]
    norm = df["cnms"].map(normalize_name)
    unnamed = (norm == "") | norm.str.fullmatch(UNNAMED)
    df = df[~unnamed].assign(cnms=df["cnms"].astype(str).str.strip())
    # One customer reported under several business segments: sum its sales.
    g = df.groupby(["gvkey", "srcdate", "cnms"], as_index=False)
    out = g.agg(salecs=("salecs", lambda s: s.sum(min_count=1)),
                n_segments=("salecs", "size"),
                conm=("conm", "first") if "conm" in df else ("cnms", "size"))
    out = out.sort_values(["gvkey", "srcdate", "cnms"]).reset_index(drop=True)
    return out, {"rows": len(out), "rows_in": n_in, "no_gvkey_dropped": int(blank.sum()),
                 "company_rows": int(company.sum()),
                 "unnamed_dropped": int(unnamed.sum()),
                 "suppliers": int(out["gvkey"].nunique()),
                 "start": str(out["srcdate"].min().date()),
                 "end": str(out["srcdate"].max().date()),
                 "missing_sales": int(out["salecs"].isna().sum())}


def clean_funda(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Industrial-format annual rows (the FS format repeats financial firms)."""
    df = df.copy()
    df["gvkey"] = gvkey_str(df["gvkey"])
    n_fs = 0
    if "indfmt" in df.columns:
        fs = df["indfmt"] != "INDL"
        n_fs = int(fs.sum())
        df = df[~fs]
    exact = df.duplicated(keep="first")
    df = df[~exact]
    conflict = df.duplicated(["gvkey", "datadate"], keep=False)
    if conflict.any():
        from concordance.data.ingest import IngestError

        raise IngestError(f"comp_funda: {int(conflict.sum())} rows share (gvkey, datadate) "
                          f"but differ:\n{df[conflict].sort_values('gvkey').head(6)}")
    return df.reset_index(drop=True), {
        "rows": len(df), "non_indl_dropped": n_fs, "exact_duplicates_dropped": int(exact.sum()),
        "missing_sale": int(df["sale"].isna().sum()) if "sale" in df else None}


# ---------------------------------------------------------------------------- matching


def _valid_links(links: pd.DataFrame, gvkeys=None) -> pd.DataFrame:
    lk = links[["gvkey", "permno", "linkprim", "linkdt", "linkenddt"]].copy()
    lk["gvkey"] = lk["gvkey"].astype(str).str.zfill(6)
    if gvkeys is not None:
        lk = lk[lk["gvkey"].isin(gvkeys)]
    return lk


def link_on(rows: pd.DataFrame, links: pd.DataFrame, key: str, date: str, out: str
            ) -> pd.DataFrame:
    """Attach `out` = PERMNO of `rows[key]` whose link is valid on `rows[date]`
    (primary link preferred). Rows without a valid link are dropped."""
    lk = _valid_links(links).rename(columns={"gvkey": key, "permno": out})
    m = rows.reset_index(names="_row").merge(lk, on=key)
    m = m[(m[date] >= m["linkdt"]) & (m[date] <= m["linkenddt"])]
    m["_rank"] = (m["linkprim"] != "P").astype(int)
    m = m.sort_values(["_row", "_rank"]).drop_duplicates("_row")
    return m.drop(columns=["_row", "_rank", "linkprim", "linkdt", "linkenddt"])


def candidate_names(funda: pd.DataFrame, links: pd.DataFrame, sp500: pd.DataFrame,
                    crsp_names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Companies a customer may be matched to: every gvkey ever linked to an S&P 500 member,
    under each name and ticker Compustat records for it, plus every historical CRSP name
    (Compustat keeps only the current name: AmerisourceBergen is listed as Cencora)."""
    lk = _valid_links(links)
    members = set(sp500["permno"].astype(int))
    gv = set(lk.loc[lk["permno"].isin(members), "gvkey"])
    f = funda[funda["gvkey"].isin(gv)]
    names = []
    if crsp_names is not None and len(crsp_names):
        h = crsp_names[["permno", "name"]].dropna().drop_duplicates()
        h = h.merge(lk[["gvkey", "permno"]].drop_duplicates(), on="permno")
        h = h[h["gvkey"].isin(gv)][["gvkey", "name"]].assign(source="crsp_name")
        names.append(h)
    for col in ("conm", "tic"):
        if col in f.columns:
            v = f[["gvkey", col]].dropna().rename(columns={col: "name"})
            v["source"] = col
            names.append(v)
    c = pd.concat(names, ignore_index=True).drop_duplicates(["gvkey", "name"])
    c["norm"] = c["name"].map(normalize_name)
    tic = c["source"] == "tic"
    # Tickers only count when they look like a name a filer would write (IBM, AT&T...).
    c = c[(c["norm"] != "") & (~tic | (c["norm"].str.len() >= 2))]
    return c.drop_duplicates(["gvkey", "norm"]).reset_index(drop=True)


def _compact(s: pd.Series) -> pd.Series:
    return s.str.replace(" ", "", regex=False)


def match_customers(seg: pd.DataFrame, funda: pd.DataFrame, links: pd.DataFrame,
                    sp500: pd.DataFrame, decisions: pd.DataFrame | None = None,
                    crsp_names: pd.DataFrame | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (matched rows, review table, report).

    matched: seg rows with `customer_gvkey` and `match` in {exact, alias, accepted}.
    review: one row per customer name needing a decision, largest sales first.
    `decisions` (customer_name, decision, gvkey) are the user's accept/reject entries."""
    cand = candidate_names(funda, links, sp500, crsp_names)
    lk = _valid_links(links, set(cand["gvkey"]))
    seg = seg.copy()
    seg["norm"] = seg["cnms"].map(normalize_name).map(_alias)
    def index(c: pd.DataFrame, key: pd.Series) -> dict:
        return c.assign(k=key).groupby("k")["gvkey"].agg(lambda s: sorted(set(s))).to_dict()

    # Company names first; a ticker only when no company name matches ("HP" is HP Inc's
    # short name but Helmerich & Payne's ticker).
    names_c = cand[cand["source"] != "tic"]
    tics = cand[cand["source"] == "tic"]
    lookups = (index(names_c, names_c["norm"]), index(names_c, _compact(names_c["norm"])),
               index(tics, tics["norm"]))

    def gv_for(norm: str) -> list[str]:
        for i, table in enumerate(lookups):
            hit = table.get(norm.replace(" ", "") if i == 1 else norm)
            if hit:
                return hit
        return []

    names = seg[["cnms", "norm"]].drop_duplicates()
    names["cands"] = names["norm"].map(gv_for)
    seg = seg.merge(names[["cnms", "cands"]], on="cnms", how="left")

    # Resolve each row's exact-name candidates by the link valid at its fiscal date.
    one = seg[seg["cands"].str.len() >= 1].explode("cands").rename(columns={"cands": "cg"})
    one = one.reset_index(names="_r").merge(lk.rename(columns={"gvkey": "cg"}), on="cg")
    one = one[(one["srcdate"] >= one["linkdt"]) & (one["srcdate"] <= one["linkenddt"])]
    n_valid = one.groupby("_r")["cg"].nunique()
    unique_r = n_valid[n_valid == 1].index
    pick = one[one["_r"].isin(unique_r)].drop_duplicates("_r").set_index("_r")["cg"]
    seg["customer_gvkey"] = pick.reindex(seg.index)
    aliased = seg["cnms"].map(normalize_name) != seg["norm"]
    seg["match"] = np.where(seg["customer_gvkey"].notna(),
                            np.where(aliased, "alias", "exact"), None)
    ambiguous_r = set(n_valid[n_valid > 1].index)

    # Everything unresolved is reviewed by name, ranked by disclosed sales.
    left = seg[seg["customer_gvkey"].isna()]
    rev = left.groupby("cnms").agg(norm=("norm", "first"), rows=("cnms", "size"),
                                   suppliers=("gvkey", "nunique"),
                                   total_sales=("salecs", "sum"),
                                   first_year=("srcdate", "min"), last_year=("srcdate", "max"))
    rev["ambiguous"] = rev.index.isin(set(seg.loc[sorted(ambiguous_r), "cnms"]))
    sug = _suggest(rev["norm"], cand)
    rev = rev.join(sug).reset_index().rename(columns={"cnms": "customer_name"})
    rev["decision"], rev["gvkey"] = "", rev["suggested_gvkey"].fillna("")
    if decisions is not None and len(decisions):
        d = decisions.dropna(subset=["decision"])
        d = d[d["decision"].astype(str).str.strip() != ""]
        d = d.assign(decision=d["decision"].astype(str).str.strip().str.lower(),
                     gvkey=d["gvkey"].astype(str).str.strip().str.zfill(6))
        bad = ~d["decision"].isin(["accept", "reject"])
        if bad.any():
            raise ValueError(f"review decisions must be 'accept' or 'reject', got "
                             f"{d.loc[bad, 'decision'].unique().tolist()[:5]}")
        d = d.drop_duplicates("customer_name", keep="last").set_index("customer_name")
        rev = rev.set_index("customer_name")
        rev.update(d[["decision", "gvkey"]])
        rev = rev.reset_index()
        acc = d[d["decision"] == "accept"]
        hit = seg["customer_gvkey"].isna() & seg["cnms"].isin(acc.index)
        seg.loc[hit, "customer_gvkey"] = seg.loc[hit, "cnms"].map(acc["gvkey"])
        seg.loc[hit, "match"] = "accepted"
    rev = rev.sort_values(["total_sales", "rows"], ascending=False).reset_index(drop=True)
    rev["first_year"] = rev["first_year"].dt.year
    rev["last_year"] = rev["last_year"].dt.year
    matched = seg[seg["customer_gvkey"].notna()].drop(columns=["cands"])
    sales = seg["salecs"].sum()
    report = {
        "customer_rows": len(seg), "names": int(seg["cnms"].nunique()),
        "matched_rows": len(matched),
        "matched_by": matched["match"].value_counts().to_dict(),
        "matched_sales_share": round(float(matched["salecs"].sum() / sales), 4) if sales else 0,
        "review_names": len(rev),
        "review_with_suggestion": int(rev["suggested_gvkey"].notna().sum()),
        "review_ambiguous": int(rev["ambiguous"].sum()),
        "decisions_accepted": int((rev["decision"] == "accept").sum()),
        "decisions_rejected": int((rev["decision"] == "reject").sum()),
    }
    return matched, rev, report


def _suggest(norms: pd.Series, cand: pd.DataFrame) -> pd.DataFrame:
    """Closest candidate name sharing the first word (blocking keeps this fast)."""
    first = cand["norm"].str.split().str[0]
    blocks = {k: g for k, g in cand.assign(first=first).groupby("first")}
    rows = {}
    for name, norm in norms.items():
        if not norm:
            continue
        g = blocks.get(norm.split()[0])
        if g is None:
            continue
        best, score = None, 0.0
        for r in g.itertuples():
            sc = difflib.SequenceMatcher(None, norm, r.norm).ratio()
            # "LUMENTUM OPERATIONS" vs "LUMENTUM": a word-prefix match is a strong hint.
            if norm.startswith(r.norm + " ") or r.norm.startswith(norm + " "):
                sc = max(sc, 0.9)
            if sc > score:
                best, score = r, sc
        if best is not None and score >= FUZZY_MIN:
            rows[name] = {"suggested_gvkey": best.gvkey, "suggested_name": best.name,
                          "score": round(score, 3)}
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["suggested_gvkey", "suggested_name", "score"])


# ----------------------------------------------------------------------- availability


def build_links(matched: pd.DataFrame, funda: pd.DataFrame, links: pd.DataFrame,
                sessions: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """One row per (supplier PERMNO, customer PERMNO, fiscal year) with its revenue share and
    visibility window [available, expires] on XNYS sessions."""
    m = matched.copy()
    m["customer_gvkey"] = m["customer_gvkey"].astype(str).str.zfill(6)
    sale = funda[["gvkey", "datadate", "sale"]].rename(columns={"datadate": "srcdate"})
    m = m.merge(sale, on=["gvkey", "srcdate"], how="left")
    m["available"] = _first_session_after(m["srcdate"] + AVAILABLE_LAG, sessions)
    m["expires"] = m["srcdate"] + EXPIRY
    m = m.dropna(subset=["available"])
    m = link_on(m, links, "gvkey", "available", "supplier_permno")
    m = link_on(m, links, "customer_gvkey", "available", "customer_permno")
    m = m[m["supplier_permno"] != m["customer_permno"]]
    # Several names for one customer in one filing: sum what was disclosed.
    keys = ["supplier_permno", "customer_permno", "srcdate"]
    g = m.groupby(keys, as_index=False).agg(
        gvkey=("gvkey", "first"), customer_gvkey=("customer_gvkey", "first"),
        salecs=("salecs", lambda s: s.sum(min_count=1)), sale=("sale", "first"),
        available=("available", "first"), expires=("expires", "first"),
        names=("cnms", lambda s: "; ".join(sorted(set(s)))))
    with np.errstate(divide="ignore", invalid="ignore"):
        share = g["salecs"] / g["sale"].where(g["sale"] > 0)
    impossible = share > 1.0 + 1e-9
    g["imputed"] = ~np.isfinite(share)
    g["share"] = np.where(g["imputed"], IMPUTED_SHARE, share)
    g = g[~impossible & (g["share"] > 0)]
    g = g.sort_values(keys).reset_index(drop=True)
    return g, {"links": len(g), "suppliers": int(g["supplier_permno"].nunique()),
               "customers": int(g["customer_permno"].nunique()),
               "imputed_share": int(g["imputed"].sum()),
               "dropped_share_over_100pct": int(impossible.sum()),
               "first_available": str(g["available"].min().date()) if len(g) else None,
               "last_available": str(g["available"].max().date()) if len(g) else None,
               "rules": {"available": "first XNYS session after srcdate + 6 months",
                         "expires": "srcdate + 18 months; newest fiscal year wins",
                         "imputed_share": IMPUTED_SHARE}}


def _first_session_after(dates: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    idx = np.searchsorted(sessions.values, dates.values.astype("datetime64[ns]"), side="right")
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = idx < len(sessions)
    out[ok] = sessions[idx[ok]]
    return out


def link_customers(lake: Path | None = None, review_path: Path | None = None) -> dict:
    """`concordance link-customers`: build lake/customer_links.parquet and the review CSV.

    The review CSV is both output and input: fill `decision` (accept / reject) and, when
    accepting, the `gvkey` to use; rerun to apply. Decisions already made are kept."""
    from concordance.data.crsp import xnys_sessions
    from concordance.data.ingest import _sha256, default_lake

    lake = Path(lake or default_lake())
    review_path = Path(review_path or lake / "customer_matches_review.csv")
    need = ["comp_seg_customer", "comp_funda", "ccm_link", "sp500_membership"]
    missing = [n for n in need if not (lake / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"ingest these exports first: {missing}")
    seg, funda, links, sp500 = (pd.read_parquet(lake / f"{n}.parquet") for n in need)
    decisions = (pd.read_csv(review_path, dtype=str, keep_default_na=False)
                 if review_path.exists() else None)
    names_path = lake / "crsp_names.parquet"
    crsp_names = pd.read_parquet(names_path) if names_path.exists() else None
    matched, review, report = match_customers(seg, funda, links, sp500, decisions, crsp_names)
    report["historical_names"] = crsp_names is not None
    import exchange_calendars as xcals

    last = xcals.get_calendar("XNYS").last_session.tz_localize(None)
    sessions = xnys_sessions(seg["srcdate"].min(),
                             min(seg["srcdate"].max() + pd.DateOffset(years=2), last))
    out, stats = build_links(matched, funda, links, sessions)
    out.to_parquet(lake / "customer_links.parquet", index=False)
    cols = ["customer_name", "decision", "gvkey", "suggested_name", "score", "total_sales",
            "rows", "suppliers", "first_year", "last_year", "ambiguous"]
    review[cols].to_csv(review_path, index=False)
    mpath = lake / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"files": {}}
    entry = {"status": "ok", **report, **stats, "review_file": str(review_path),
             "review_sha256": _sha256(review_path)}
    manifest["files"]["customer_links"] = entry
    mpath.write_text(json.dumps(manifest, indent=2))
    return entry


# -------------------------------------------------------------------------- panel side


@dataclass
class CustomerLinks:
    """Links on a panel's columns. Visibility is by session index, so a truncated panel
    (look-ahead check) sees exactly the links it would have seen."""

    supplier: np.ndarray  # int column index
    customer: np.ndarray  # int column index
    share: np.ndarray  # float
    imputed: np.ndarray  # bool
    fiscal: np.ndarray  # int64 fiscal year-end (ns), newest per supplier wins
    start: np.ndarray  # int first session index visible
    stop: np.ndarray  # int first session index no longer visible
    n_assets: int
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        self._events = np.unique(np.concatenate([self.start, self.stop]))

    def weights_at(self, i: int, imputed: bool = True) -> np.ndarray:
        state = (int(np.searchsorted(self._events, i, side="right")), bool(imputed))
        hit = self._cache.get(state)
        if hit is not None:
            return hit
        act = (self.start <= i) & (i < self.stop)
        if act.any():
            newest = np.full(self.n_assets, np.iinfo(np.int64).min)
            np.maximum.at(newest, self.supplier[act], self.fiscal[act])
            act &= self.fiscal == newest[self.supplier]
        if not imputed:
            act &= ~self.imputed
        w = np.zeros((self.n_assets, self.n_assets))
        w[self.supplier[act], self.customer[act]] = self.share[act]
        w.flags.writeable = False
        self._cache.clear()  # sessions arrive in order; one state is enough
        self._cache[state] = w
        return w


def from_frame(links: pd.DataFrame, sessions: pd.DatetimeIndex, permnos: np.ndarray
               ) -> CustomerLinks:
    """Map PERMNO links onto panel columns (a PERMNO with several listings maps to each)."""
    col: dict[int, list[int]] = {}
    for j, p in enumerate(permnos):
        col.setdefault(int(p), []).append(j)
    sess = sessions.values.astype("datetime64[ns]")
    sup, cus, share, imp, fis, start, stop = ([] for _ in range(7))
    for r in links.itertuples():
        for a in col.get(int(r.supplier_permno), ()):
            for b in col.get(int(r.customer_permno), ()):
                if a == b:
                    continue
                sup.append(a)
                cus.append(b)
                share.append(r.share)
                imp.append(bool(r.imputed))
                fis.append(pd.Timestamp(r.srcdate).value)
                start.append(np.searchsorted(sess, np.datetime64(r.available, "ns"), "left"))
                stop.append(np.searchsorted(sess, np.datetime64(r.expires, "ns"), "right"))
    a = np.asarray
    return CustomerLinks(a(sup, int), a(cus, int), a(share, float), a(imp, bool),
                         a(fis, np.int64), a(start, int), a(stop, int), len(permnos))


def attach_customer_links(panel, lake: Path | None = None):
    """Load lake/customer_links.parquet onto `panel` (returns a new panel)."""
    from dataclasses import replace

    from concordance.data.ingest import default_lake

    lake = Path(lake or default_lake())
    path = lake / "customer_links.parquet"
    if not path.exists():
        raise FileNotFoundError("lake/customer_links.parquet is missing: run "
                                "`concordance link-customers` first")
    frame = pd.read_parquet(path)
    permnos = panel.meta.get("permno", np.asarray(panel.assets))
    links = from_frame(frame, panel.sessions, np.asarray(permnos))
    meta = dict(panel.meta)
    meta["customer_links"] = {"links": int(len(links.share)),
                              "suppliers": int(len(np.unique(links.supplier)))}
    return replace(panel, links=links, meta=meta)


def random_links(panel, seed: int, per_supplier: int = 3):
    """Synthetic links for the strategy validator: fiscal years every 252 sessions."""
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    T, N = panel.shape
    rows = []
    for year_start in range(0, T, 252):
        for s in range(N):
            if rng.random() < 0.4:
                continue
            for c in rng.choice(np.delete(np.arange(N), s), size=min(per_supplier, N - 1),
                                replace=False):
                imputed = rng.random() < 0.3
                rows.append((s, int(c), 0.1 if imputed else float(rng.uniform(0.1, 0.6)),
                             imputed, year_start, year_start + int(rng.integers(5, 40)),
                             year_start + 252 + int(rng.integers(5, 40))))
    a = np.array(rows, dtype=float).reshape(-1, 7)
    links = CustomerLinks(a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2],
                          a[:, 3].astype(bool), a[:, 4].astype(np.int64), a[:, 5].astype(int),
                          a[:, 6].astype(int), N)
    return replace(panel, links=links)
