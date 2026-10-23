# Annual Value and gross profitability

`examples/value_annual.py` is a separate exploratory implementation of the Value-then-
profitability strategy. Its accounting inputs come from current Compustat annual STD
exports. They can contain later revisions. A conservative lag does not make them
historical original vintages, and they cannot satisfy a first-reported policy.

## Fixed baseline

- Universe: the existing `us_liquid` historical common-stock universe (prior-close price
  at least $5 and market cap at least the NYSE 20th percentile), excluding contemporaneous
  S&P 500 members. This is not an exact small/mid-cap index.
- Signal date: last XNYS session close in June. Orders execute at the next available open
  under the existing engine's execution, liquidity, cost and delisting rules.
- Accounting cohort: fiscal periods ending in the preceding calendar year, choosing the
  latest fiscal end per company as one entire row. No older-year or missing-field fallback.
  No inference of actual filing dates. Missing reporting years expire at the next formation.
- Value: positive common equity (`ceq`) / contemporaneous market capitalization. No
  deferred-tax or preferred-stock book-equity substitutions.
- Keep the cheaper 50%, then select the most profitable 20% by `(revt-cogs)/at` with
  positive assets. Missing/nonfinite inputs and nonpositive capitalization are excluded.
  Negative gross profit remains a legitimate rankable value.
- Equal weights, 80% target gross, 1,000-position cap; deterministic asset-ID tie breaks.
  No eligible stocks means a cash target. There is no monthly reranking in this version.
  A run starting before its first June formation holds cash until that formation;
  positions are not backdated to before the run.
- Fractions and gross are configurable for explicitly recorded later experiments; baseline
  defaults were selected before looking at historical performance. No optimization was run.

## Prepare a source

Run from this checkout with its source directory on PYTHONPATH:

```bash
PYTHONPATH=src /home/vkulk/concordance/.venv/bin/python -m concordance.data.annual \
  --source /path/to/comp_funda_annual.csv.gz \
  --links /path/to/ccm_link.parquet \
  --destination /path/to/new-prepared-directory --cutoff 2019-12-31
```

Required annual fields: `gvkey, datadate, curcd, indfmt, datafmt, consol, at, ceq, revt,
cogs`. Only USD, INDL, STD, C observations are retained. `popsrc=D` is enforced when the
column is supplied; its absence is recorded, not silently claimed as verified. No currency
conversion, missing-value imputation or revenue substitution occurs. Raw downloads remain
unchanged. Statements/formation dates after the cutoff are excluded; availability intervals
are clipped to that cutoff. The manifest hashes the source, link file and prepared rows.

CCM LC/LU and P/C links are accepted only during their historical spells. Overlapping
assignments to a PERMNO fail rather than choose a company by row order. Company links that
start during a holding year become visible only from their start. Expired links disappear
from the view; existing held positions still follow normal execution/delisting rules.

## Run through the normal API

```json
{
  "strategy": {"path": "examples/value_annual.py"},
  "data": {
    "universe": "us_liquid", "start": "1998-01-01", "end": "2019-12-31",
    "lake": "/path/to/research-lake",
    "fundamentals": true, "fundamentals_basis": "annual_lagged",
    "annual_source": "/path/to/prepared-directory"
  },
  "engines": ["reference", "vectorbt", "nautilus"],
  "validation": {"program": "annual_value_1998_2019", "lookahead_cuts": 3}
}
```

The source hash and limitation are recorded in run metadata. A first_reported request
cannot accept annual_source. This route does not mutate the registered quarterly study,
admit annual data into a sealed generation, or schedule autonomous model work.

The standard strategy validator supplies synthetic annual inputs when the strategy
declares `fundamentals_basis = "annual_lagged"`. Real validation uses an explicit run
request as above so the source cannot be guessed from the global lake. Isolated validator
subprocesses use this checkout's source tree even when another checkout is installed in
the shared interpreter.

## Verification and resource limits

Unit tests exercise June weekend timing, missing years/fields, fiscal-year transitions,
link expiry/start, ambiguous assignments, source hashes, cache identity, deterministic
selection, S&P exclusion and next-open execution across three engines. The normal strategy
validator also checks synthetic hostile markets and truncation invariance.

Use a short real-data window first. For long histories set `memory.mode: "disk"`;
[Memory](MEMORY.md) documents batched loading, disk arrays, streaming execution and
resource measurement. The earlier RAM loader exceeded a 16 GiB address-space cap.
Disk mode preserves continuous portfolio state instead of restarting yearly accounts.
It still needs physical-memory and disk monitoring; mapped virtual address space is
not equivalent to resident RAM.
