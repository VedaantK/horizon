"""Expected WRDS exports and their column mappings (see docs/WRDS_EXPORT.md).

Column names are matched case-insensitively. The CRSP mapping is carried over from
QuantMega's validated CIZ extract (`research/quantlab/crsp.py`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExportSpec:
    name: str  # file stem the user saves the export as (any of .parquet/.csv/.csv.gz)
    wrds_table: str
    columns: dict[str, str]  # WRDS column -> Horizon column
    required: tuple[str, ...]  # Horizon columns that must be present
    date_columns: tuple[str, ...] = ()


CRSP_DAILY = ExportSpec(
    name="crsp_daily",
    wrds_table="crsp.dsf_v2",
    columns={
        "permno": "permno",
        "dlycaldt": "date",
        "ticker": "ticker",
        "siccd": "siccd",
        "primaryexch": "primaryexch",
        "sharetype": "sharetype",
        "securitytype": "securitytype",
        "dlyopen": "open",
        "dlyhigh": "high",
        "dlylow": "low",
        "dlyclose": "close_trade",
        "dlyprc": "prc",
        "dlybid": "bid",
        "dlyask": "ask",
        "dlyvol": "volume",
        "dlyret": "ret",
        "dlyretx": "retx",
        "dlycap": "mktcap_k",
        "shrout": "shrout",
        "dlydelflg": "delist_flag",
        "dlyprevdt": "prev_date",
        "sprtrn": "sp500_ret",  # S&P 500 composite return that day (benchmark, optional)
    },
    required=("permno", "date", "open", "prc", "ret"),
    date_columns=("date", "prev_date"),
)

SP500_MEMBERSHIP = ExportSpec(
    name="sp500_membership",
    wrds_table="crsp.dsp500list_v2",
    columns={"permno": "permno", "mbrstartdt": "start", "mbrenddt": "end"},
    required=("permno", "start", "end"),
    date_columns=("start", "end"),
)

CCM_LINK = ExportSpec(
    name="ccm_link",
    wrds_table="crsp.ccmxpf_lnkhist",
    columns={"gvkey": "gvkey", "lpermno": "permno", "permno": "permno",
             "linktype": "linktype", "linkprim": "linkprim", "linkdt": "linkdt",
             "linkenddt": "linkenddt"},
    required=("gvkey", "permno", "linktype", "linkprim", "linkdt"),
    date_columns=("linkdt", "linkenddt"),
)

_FUNDQ_COLS = [
    "gvkey", "datadate", "fyearq", "fqtr", "rdq", "atq", "ltq", "seqq", "ceqq", "saleq",
    "revtq", "cogsq", "xsgaq", "ibq", "niq", "oiadpq", "txditcq", "pstkq", "cshoq", "prccq",
    "oancfy", "capxy", "dlttq", "dlcq", "cheq", "actq", "lctq", "xrdq", "curcdq",
]
COMP_FUNDQ = ExportSpec(
    name="comp_fundq",
    wrds_table="comp.fundq",
    columns={c: c for c in _FUNDQ_COLS},
    required=("gvkey", "datadate", "fyearq", "fqtr", "rdq"),
    date_columns=("datadate", "rdq"),
)

_FUNDA_COLS = [
    "gvkey", "datadate", "fyear", "at", "lt", "seq", "ceq", "pstk", "pstkl", "pstkrv",
    "txditc", "act", "lct", "che", "dltt", "dlc", "sale", "revt", "cogs", "xsga", "xrd", "dp",
    "oiadp", "ib", "ni", "oancf", "capx", "dvc", "csho", "prcc_f",
]
COMP_FUNDA = ExportSpec(
    name="comp_funda",
    wrds_table="comp.funda",
    columns={c: c for c in [*_FUNDA_COLS, "conm", "tic", "cusip", "indfmt", "curcd", "costat"]},
    required=("gvkey", "datadate", "fyear"),
    date_columns=("datadate",),
)

COMP_SEG_CUSTOMER = ExportSpec(
    name="comp_seg_customer",
    wrds_table="Compustat Historical Segments: Customer (comp_segments_hist_daily.seg_customer)",
    columns={c: c for c in ("gvkey", "srcdate", "datadate", "conm", "tic", "cid", "cnms",
                            "ctype", "salecs", "sid", "stype", "gareac", "gareat")},
    required=("gvkey", "srcdate", "cnms", "ctype", "salecs"),
    date_columns=("srcdate", "datadate"),
)

CRSP_NAMES = ExportSpec(
    name="crsp_names",
    wrds_table="CRSP stock names history (crsp.stocknames / v2 security info history)",
    columns={"permno": "permno", "comnam": "name", "issuernm": "name", "namedt": "start",
             "secinfostartdt": "start", "nameenddt": "end", "nameendt": "end",
             "secinfoenddt": "end",
             "ticker": "ticker"},
    required=("permno", "name"),
    date_columns=("start", "end"),
)

FF_DAILY = ExportSpec(
    name="ff_factors_daily",
    wrds_table="ff.factors_daily",
    columns={"date": "date", "mktrf": "mktrf", "smb": "smb", "hml": "hml", "rmw": "rmw",
             "cma": "cma", "rf": "rf", "umd": "umd"},
    required=("date", "mktrf", "rf"),
    date_columns=("date",),
)

CRSP_DELIST = ExportSpec(
    name="crsp_delist",
    wrds_table="crsp.stkdelists (v2; legacy crsp.dsedelist names also accepted)",
    columns={"permno": "permno", "delistingdt": "date", "dlstdt": "date", "delret": "delret",
             "dlret": "delret", "delnextprc": "delnextprc", "delactiontype": "action",
             "delreasontype": "reason", "delpaymenttype": "payment", "dlstcd": "code"},
    required=("permno", "date", "delret"),
    date_columns=("date",),
)

# Delistings are ingested before the daily file, which uses them at listing gaps.
ALL_SPECS = (CRSP_DELIST, CRSP_DAILY, SP500_MEMBERSHIP, CCM_LINK, COMP_FUNDQ, COMP_FUNDA,
             COMP_SEG_CUSTOMER, CRSP_NAMES, FF_DAILY)
