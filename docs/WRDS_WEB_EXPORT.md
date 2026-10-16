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

## Query 6: CRSP delisting returns — save as `crsp_delist_<years>.csv`
Ingest checks that every final delisting return is already in the daily file (622 of 622
on the 1994–2025 export). The v2 table records only a stock's final delisting, so it does not
value the 18 temporary listing gaps (3 began while the stock was an S&P member: HealthSouth
2003, Comverse 2007, Washington Mutual 2008). Those exits stay at the last price, and
`concordance run` warns if a strategy held one. A delisting record dated inside a gap is used
if one ever appears (e.g. from the legacy file).

**Menu:** CRSP → Stock (v2) → **Delisting information** (the legacy Annual Update form with
`DLSTDT`, `DLRET` works too)

| Step | Setting |
|---|---|
| Date range | **Type the start date: 1990-01-01.** The form defaults to 2020. |
| Company codes | PERMNO, upload `permnos.txt` |
| Variables | all (needed: `PERMNO`, `DelistingDt`, `DelRet`, `DelNextPrc`) |

---|---|
| Date range | 1990-01-01 → latest |
| Company codes | PERMNO, upload `permnos.txt` |
| Variables | `PERMNO`, `DLSTDT`, `DLSTCD`, `DLRET`, `DLRETX`, `DLPRC`, `NWPERM`, `NWCOMP` |

---

## Customer links (supplier -> customer strategies)
Three exports, saved in their own folder (e.g. `~/wrds_exports/customer/`). Ingest that
folder by itself: datasets it doesn't contain keep their earlier ingest.

| Save as | Menu | Settings |
|---|---|---|
| `comp_seg_customer.csv` | Compustat → North America → **Historical Segments → Customer** (not the non-historical Segments, which are restated and start in 2014) | Type the start date **1988-01-01** (data begins with fiscal 1998); search the entire database; variables `GVKEY`, `SRCDATE`, `CONM`, `TIC`, `CNMS`, `CTYPE`, `SALECS`, `CID`, `SID`, `STYPE`, `GAREAC`, `GAREAT` |
| `ccm_link.csv` | CRSP/Compustat Merged → **Linking Table** | Start 1950-01-01; entire database; `GVKEY`, `LPERMNO`, `LINKTYPE`, `LINKPRIM`, `LINKDT`, `LINKENDDT` (Query 3 above) |
| `comp_funda.csv` | Compustat → North America → **Fundamentals Annual** | Start **1997-01-01**; entire database; default screens; `GVKEY`, `DATADATE`, `FYEAR`, `CONM`, `TIC`, `CUSIP`, `SALE`, `REVT` |
| `crsp_names.csv` (optional, improves matching) | CRSP → Stock → **Stock Header / Names** history | Widest dates; PERMNO, upload `permnos.txt`; `PERMNO`, `COMNAM` (or `IssuerNm`), `NAMEDT`, `NAMEENDDT`, `TICKER` |

Then:
```bash
uv run concordance ingest ~/wrds_exports/customer
uv run concordance link-customers
```
`link-customers` matches customer names to S&P 500 companies (exact normalized names and a
short alias list automatically, resolved by the link valid at the fiscal date) and writes
`lake/customer_matches_review.csv`, largest unmatched customers first. Put `accept` or
`reject` in its `decision` column (and the `gvkey` to use when accepting), then rerun to
apply. Compustat keeps only each company's current name, so without `crsp_names` renamed
customers (AmerisourceBergen → Cencora, CVS Caremark → CVS Health) need review.

The October 2026 customer export: 595,637 rows, fiscal 1998–2026; 1,046 rows had no
`GVKEY` (dropped); 129,156 company rows were unnamed placeholders ("Not Reported",
"5 Customers"). The fundamentals export included the financial-services format (`FS`)
next to `INDL`; ingest keeps `INDL`.

---

## What was actually downloaded (October 2026)
- `sp500_membership.csv` was extracted from the S&P 500 Constituents form's output. That form
  also returns daily stock rows, but only for days inside each membership spell, so it can't
  replace Query 2.
- `crsp_daily_1990_2019.csv.gz` and `crsp_daily_2020_2025.csv.gz`: one Daily Stock File query
  each. They include `sprtrn` (the S&P 500 return), which ingest keeps as the benchmark.
- CRSP opening prices start in mid-1992, so backtests can start in 1993 at the earliest.

## Checklist
- [x] `sp500_membership.csv`
- [x] `permnos.txt` (generated, not downloaded)
- [x] `crsp_daily_*.csv.gz`
- [x] `ff_factors_daily.csv.gz`
- [x] `crsp_delist_2020_2025.csv.gz`
- [x] `crsp_delist_1990_2019.csv.gz`
- [x] `ccm_link.csv`
- [x] `comp_seg_customer.csv`, `comp_funda.csv` (customer links)
- [ ] `crsp_names.csv` (optional; better customer-name matching)
- [ ] `comp_fundq.csv` (optional for now)
