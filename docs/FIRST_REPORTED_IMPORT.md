# First-reported fundamentals staging

Value research remains blocked until actual historical source evidence is available.
The importer validates a supplied canonical export; it neither downloads licensed data
nor authenticates a vendor's first-reported claim. It does not amend any sealed lake,
enable fundamentals in the existing backtester, or make restated Compustat acceptable.

The export is CSV or Parquet, with exactly these columns:

| Columns | Contract |
| --- | --- |
| permno | Positive historical security identifier, already linked without future identifiers |
| datadate | Fiscal-quarter end, a date without a time zone |
| fyearq, fqtr | Integer fiscal year and quarter 1–4; no ambiguous calendar duplicates |
| published_at, vintage_at | Actual publication and original source-vintage timestamps, both with time zones |
| source_record_id | Nonempty original filing/snapshot identity |
| is_original | Boolean true; revisions/restatements are rejected |
| currency, units | USD and USD_millions |
| atq, seqq, ceqq, pstkq, txditcq, revtq, cogsq | Numeric or missing; flows are individual quarters, never YTD |

The independent review JSON must include schema_version=1, basis=as_first_reported,
provider, dataset, reviewed_by, reviewed_at, methodology, and source_sha256 matching the
exact export bytes. This attestation needs a real review of vendor documentation and
sample source records. Filling in a JSON form is not proof of historical availability.

```sh
PYTHONPATH=/home/vkulk/horizon-review/src \
  /home/vkulk/horizon/.venv/bin/python -m concordance.data.first_reported \
  EXPORT.parquet REVIEW.json NEW_STAGING_DIRECTORY --cutoff 2019-12-31
```

Availability is conservatively the first XNYS session after the UTC calendar date of
the later publication or vintage timestamp. Reports not available by the cutoff are
excluded. Metadata describes retained rows only. TTM event construction uses only
quarters visible at each event: a late filing can complete a trailing year when it
arrives, but cannot rewrite earlier history. Missing quarters and financial values
remain missing. Fiscal-calendar changes requiring judgment stop rather than guess.

The staging output must still pass independent provenance review, historical identifier
review, a versioned backtester adapter and generation admission checks before Value is
runnable. Public modern company-facts exports or a delayed restated table must not be
substituted for historical first-reported observations.

## Explicit admitted-data adapter

The research-next checkout includes a separate adapter selected by both
`data.fundamentals: true` and `data.fundamentals_basis: first_reported`. The default
remains `restated` for legacy requests. There is no fallback when first-reported data
or its approval is missing. The API reports which basis and evidence hash it used.

After an independent review of real source evidence, the approved lake must contain
`first_reported_quarterly.parquet` and `first_reported_admission.json`. The admission
has exactly these fields: schema_version=1, basis=as_first_reported, data_sha256,
evidence_sha256, approved_by, cutoff. The evidence hash refers to the reviewed operator
evidence bundle, which stays outside the researcher boundary. A JSON approval is an
operator attestation, not a vendor-authenticated signature. Do not approve a synthetic
test fixture or populate this record merely to unblock research.

The adapter checks the canonical file hash, schema, metadata, timezone-bearing source
timestamps, fiscal keys, original-report flags, exact next-exchange-session availability
and cutoff. It derives events as reports arrive, preserves late-quarter TTM updates,
never backfills them into earlier dates, and returns missing values after staleness.
Supported fields are atq, seqq, ceqq, pstkq, txditcq, revtq, cogsq and the two TTM flows.
Other fields and annual reports remain unsupported. The panel cache distinguishes dates
and asset identities. Capabilities describe this source separately from restated data.
