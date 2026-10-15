# Getting the data through WRDS web query forms (IP-Auth accounts)

IP-Auth accounts can't use WRDS Cloud, JupyterHub or the PostgreSQL connection. The web
query forms on wrds-www.wharton.upenn.edu still give you everything Concordance needs.
Connect from a CMU IP (on campus or through the VPN).

Every WRDS web query form follows the same 4 steps:
1. **Date range**
2. **Company codes**: type them, upload a `.txt` file, or "Search the entire database"
3. **Variables**: tick checkboxes
4. **Output**: format and date format

Use these settings for **every** query in step 4:
- **Output format:** comma-delimited text (`.csv`). Compressed `.csv.gz` is fine too.
- **Date format:** `YYYY-MM-DD` (e.g. `YYMMDD10.`) or `YYYYMMDD` (`YYMMDDn8.`). Ingest
  accepts both, plus `02JAN2020`.

Save everything into one folder, e.g. `~/wrds_exports`, under the file names given below.
Menu names can be worded slightly differently on your screen; look for the closest match.

---

## Query 1: S&P 500 membership (do this first) — save as `sp500_membership.csv`
**Menu:** CRSP → (Version 2 / CIZ) Index → **S&P 500 Constituents**

| Step | Setting |
|---|---|
| Date range | Widest available (e.g. 1925 → latest) |
| Company codes | Search the entire database |
| Variables | `PERMNO`, `MbrStartDt`, `MbrEndDt` (all the form offers is fine) |

Then make the PERMNO list for Query 2:
```bash
cd ~/concordance
uv run concordance permnos ~/wrds_exports/sp500_membership.csv -o ~/wrds_exports/permnos.txt
```
This prints how many PERMNOs it wrote (expect roughly 1,500–2,000).

## Query 2: CRSP daily stock file, in chunks — save as `crsp_daily_1994_1998.csv`, etc.
**Menu:** CRSP → (Version 2 / CIZ) Stock → **Daily Stock File**

| Step | Setting |
|---|---|
| Date range | One 5-year chunk per query (see the table below) |
| Company codes | Code type **PERMNO**, then **upload `permnos.txt`** |
| Variables | `PERMNO`, `DlyCalDt`, `Ticker`, `SICCD`, `PrimaryExch`, `ShareType`, `SecurityType`, `DlyOpen`, `DlyHigh`, `DlyLow`, `DlyClose`, `DlyPrc`, `DlyBid`, `DlyAsk`, `DlyVol`, `DlyRet`, `DlyRetx`, `DlyCap`, `ShrOut`, `DlyDelFlg`, `DlyPrevDt` |

Run it once per chunk. Any file named `crsp_daily_<anything>.csv` is combined automatically,
and overlapping days are de-duplicated.

| Query | Dates | Save as |
|---|---|---|
| 2a | 1994-01-01 → 1998-12-31 | `crsp_daily_1994_1998.csv` |
| 2b | 1999-01-01 → 2003-12-31 | `crsp_daily_1999_2003.csv` |
| 2c | 2004-01-01 → 2008-12-31 | `crsp_daily_2004_2008.csv` |
| 2d | 2009-01-01 → 2013-12-31 | `crsp_daily_2009_2013.csv` |
| 2e | 2014-01-01 → 2018-12-31 | `crsp_daily_2014_2018.csv` |
| 2f | 2019-01-01 → latest | `crsp_daily_2019_latest.csv` |

If WRDS says a query is too large or it times out, split that chunk in half.
The **required** variables are `PERMNO`, `DlyCalDt`, `DlyOpen`, `DlyPrc` and `DlyRet`.
If any of the others are missing on your form, skip them.

**You can run your first backtest after Queries 1 and 2.** The rest add fundamentals and
benchmarks.

## Query 3: CRSP/Compustat link table — save as `ccm_link.csv`
**Menu:** CRSP → CRSP/Compustat Merged → **Linking Table** (it may not be offered on the
IP-Auth account; if so, skip it and tell me)

| Step | Setting |
|---|---|
| Company codes | Search the entire database |
| Link types | `LU`, `LC`; link primacy `P`, `C` |
| Variables | `gvkey`, `LPERMNO`, `LINKTYPE`, `LINKPRIM`, `LINKDT`, `LINKENDDT` |

## Query 4: Compustat quarterly fundamentals — save as `comp_fundq.csv`
**Menu:** Compustat → North America → **Fundamentals Quarterly**

| Step | Setting |
|---|---|
| Date range | 1992-01 → latest |
| Company codes | Search the entire database (or gvkey list if it is too big; ask me) |
| Screening | Consolidation **C**, Industry format **INDL**, Data format **STD**, Population source **D** (usually the defaults) |
| Variables | `gvkey`, `datadate`, `fyearq`, `fqtr`, `rdq`, `atq`, `ltq`, `seqq`, `ceqq`, `saleq`, `revtq`, `cogsq`, `xsgaq`, `ibq`, `niq`, `oiadpq`, `txditcq`, `pstkq`, `cshoq`, `prccq`, `oancfy`, `capxy`, `dlttq`, `dlcq`, `cheq`, `actq`, `lctq`, `xrdq` |

## Query 5: Fama-French daily factors — save as `ff_factors_daily.csv`
**Menu:** Fama-French Portfolios and Factors → **Fama-French Factors - Daily Frequency**

| Step | Setting |
|---|---|
| Date range | 1994-01-01 → latest |
| Variables | `date`, `mktrf`, `smb`, `hml`, `rf`, `umd` |

---

## Then
```bash
cd ~/concordance
uv run concordance ingest ~/wrds_exports
```
Ingest prints a summary of every file. If it reports a missing or renamed column, paste the
message to Claude. It lists the columns it found, so the schema can be adjusted in a minute.

## Checklist
- [ ] `sp500_membership.csv`
- [ ] `permnos.txt` (generated, not downloaded)
- [ ] `crsp_daily_*.csv` × 6 chunks
- [ ] `ccm_link.csv` (optional; may be unavailable)
- [ ] `comp_fundq.csv` (optional for now)
- [ ] `ff_factors_daily.csv` (optional for now)
