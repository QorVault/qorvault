# Sample — 2026-02-11 Custodial

- **Set:** `2026-02-11:Custodial`
- **Source file:** `/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data/2026-02-11-regular-meeting-630-pm/Trust_Vouchers_02-11-26.pdf`
- **SHA-256:** `f1fa5e245000d3c5a7cb575778f400644f79180f11f137add58d44a18ad05cb0`
- **Set totals:** stated 400.00, parsed 400.00, 3 lines, 1 checks
- **Sample:** 3 of 3 lines, seed `20260914`

Open the source file at the page in each row and confirm the vendor, date, check number and both amounts match. Mark anything that does not.

**If you find zero errors in these 3 lines**, the error rate for this set is below roughly 100.0% at 95% confidence (the rule of three: 3/3). A smaller sample gives a correspondingly weaker bound, and this says nothing about any set that was not sampled.

> **3 of these 3 rows name an individual rather than a business, and the name is withheld.** The same classifier the published exports use decides this, so the two cannot drift apart. A withheld row still carries its page, check number and both amounts, which is everything needed to find it in the source PDF and check the arithmetic — open the page and the name is there. The verbatim quote is redacted for the same reason the column is: the quote is the source line, and it carries the name.

| # | Page | Payee | Check date | Check no. | Check amt | Invoice amt | Description | Flags |
|---:|---:|---|---|---|---:|---:|---|---|
| 1 | 1 | (name withheld) | 2026-02-05 | 700142 | 400.00 | 400.00 | Trust and Custodial Fund Warrants 01/09/26 through 02/05/26 |  |
| 2 | 2 | (name withheld) | — | — | — | — |  | **COLUMN_AMBIGUOUS** |
| 3 | 2 | (name withheld) | — | — | — | — |  | **COLUMN_AMBIGUOUS** |

## Verbatim quotes

Each line as extracted, for exact comparison against the page:

- `#1` p1 @595: `(name withheld) 02/05/2026 700142 400.00 400.00 T`
- `#2` p2 @5524: `(name withheld)`
- `#3` p2 @5694: `(name withheld) — quote suppressed: the payee name could not be located to remove it`
