# WRDS exports for Concordance

> **IP-Auth / web-form only?** Use [`WRDS_WEB_EXPORT.md`](WRDS_WEB_EXPORT.md) instead.

Run these queries on WRDS. The easiest place is **WRDS Cloud → JupyterHub** (Python, with
`wrds` preinstalled), or the **SAS Studio / PostgreSQL** query tool. Save each result
with the **exact file name** shown, as `.parquet` (preferred) or `.csv.gz`. Put them all in one
folder, then run:

```bash
uv run concordance ingest /path/to/that/folder
```

Ingest checks the columns, rejects conflicting duplicates and writes `lake/manifest.json`
(row counts, date ranges, SHA-256 of each file). If a column name differs on your account,
ingest names it. Send me that message and I'll adjust `src/concordance/data/schema.py`.

WRDS licence terms apply. The lake is git-ignored, so never commit or share these files.

In JupyterHub, each query below is run with:
```python
import wrds; db = wrds.Connection()
df = db.raw_sql("""<query>""")
df.to_parquet("<file name>.parquet", index=False)
```

## 1. `crsp_daily` — CRSP daily stock file (CIZ / v2) **[required]**
Every daily row for every PERMNO that was ever an S&P 500 member. Include days outside the
membership spell, so exits and delistings are kept.
```sql
SELECT d.permno, d.dlycaldt, d.ticker, d.siccd, d.primaryexch, d.sharetype, d.securitytype,
       d.dlyopen, d.dlyhigh, d.dlylow, d.dlyclose, d.dlyprc, d.dlybid, d.dlyask, d.dlyvol,
       d.dlyret, d.dlyretx, d.dlycap, d.shrout, d.dlydelflg, d.dlyprevdt
FROM crsp.dsf_v2 d
WHERE d.permno IN (SELECT DISTINCT permno FROM crsp.dsp500list_v2)
  AND d.dlycaldt >= '1994-01-01'
```
About 8M rows. If it times out, split it by year (`AND dlycaldt BETWEEN …`) and concatenate
the pieces before saving.

## 2. `sp500_membership` — S&P 500 membership spells **[required]**
```sql
SELECT permno, mbrstartdt, mbrenddt FROM crsp.dsp500list_v2
```

## 3. `ccm_link` — CRSP/Compustat link table (needed for fundamentals)
```sql
SELECT gvkey, lpermno, linktype, linkprim, linkdt, linkenddt
FROM crsp.ccmxpf_lnkhist
WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
```

## 4. `comp_fundq` — Compustat quarterly fundamentals
```sql
SELECT gvkey, datadate, fyearq, fqtr, rdq, atq, ltq, seqq, ceqq, saleq, revtq, cogsq, xsgaq,
       ibq, niq, oiadpq, txditcq, pstkq, cshoq, prccq, oancfy, capxy, dlttq, dlcq, cheq,
       actq, lctq, xrdq
FROM comp.fundq
WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
  AND datadate >= '1992-01-01'
```

## 5. `comp_funda` — Compustat annual fundamentals (optional)
```sql
SELECT gvkey, datadate, fyear, at, lt, seq, ceq, pstk, pstkl, pstkrv, txditc, act, lct, che,
       dltt, dlc, sale, revt, cogs, xsga, xrd, dp, oiadp, ib, ni, oancf, capx, dvc, csho,
       prcc_f
FROM comp.funda
WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
  AND datadate >= '1992-01-01'
```

## 6. `ff_factors_daily` — Fama-French daily factors (benchmark and sanity checks)
```sql
SELECT date, mktrf, smb, hml, rf, umd FROM ff.factors_daily WHERE date >= '1994-01-01'
```

## Later phases (not needed yet)
- **TAQ** (`taqmsec.ctm_YYYYMMDD`, `cqm_YYYYMMDD`, `nbbom_YYYYMMDD`): trades, quotes and the
  national best bid/offer for SPY/QQQ intraday studies.
- **OptionMetrics** (`optionm.opprcdYYYY`, `secprd`, `zerocd`): options.
- **IBES** (`ibes.statsum_epsus`, `actu_epsus`): earnings surprises.
- **Compustat Snapshot / Point-in-Time**, if your licence has it: as-first-reported
  fundamentals. `comp.fundq` contains restated values.

## Things worth knowing
- The legacy CRSP files (`crsp.dsf`, `crsp.msf`, `crsp.msedelist`) stopped updating after
  2024. Concordance is built on the CIZ (`_v2`) tables, where `dlyret` already includes the
  delisting return.
- `dlyprc < 0` means there was no trade that day and the price is the bid/ask midpoint.
  Ingest takes the absolute value and counts those rows.
- Membership is keyed by PERMNO. Tickers get reused, so they are never used as identifiers.
