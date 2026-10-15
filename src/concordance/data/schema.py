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
    columns: dict[str, str]  # WRDS column -> Concordance column
    required: tuple[str, ...]  # Concordance columns that must be present
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
    "oancfy", "capxy", "dlttq", "dlcq", "cheq", "actq", "lctq", "xrdq",
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
    columns={c: c for c in _FUNDA_COLS},
    required=("gvkey", "datadate", "fyear"),
    date_columns=("datadate",),
)

FF_DAILY = ExportSpec(
    name="ff_factors_daily",
    wrds_table="ff.factors_daily",
    columns={"date": "date", "mktrf": "mktrf", "smb": "smb", "hml": "hml", "rf": "rf",
             "umd": "umd"},
    required=("date", "mktrf", "rf"),
    date_columns=("date",),
)

ALL_SPECS = (CRSP_DAILY, SP500_MEMBERSHIP, CCM_LINK, COMP_FUNDQ, COMP_FUNDA, FF_DAILY)
