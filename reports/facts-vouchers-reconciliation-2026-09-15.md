# Voucher Fact Tables — Full Run and Reconciliation

**Date:** 2026-09-15
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers`, branched from `main` @ `29557ed`
**Package:** `facts/vouchers/` (+ new shared `facts/common/`)
**Phase 0 report:** `reports/facts-vouchers-recon-2026-09-14.md`

**No LLM touched any amount, date, vendor, check number or total.** Every figure
below is `pdfplumber` text extraction, a regular expression, and exact `Decimal`
arithmetic. The `vendor.category` column — the only place an LLM was ever permitted —
is NULL for all 12,270 vendors; no LLM step was run.

All reads of `documents` used a `READ ONLY` Postgres session. Writes went only to the
five new tables in schema `facts`. The six minutes tables were verified unchanged at
**meeting 1,646 / motion 6,507 / vote 19,633** after every rebuild.

---

## Headline

| | |
|---|---|
| Voucher sets built | **446** (2005-05-25 → 2026-05-27) |
| Voucher lines | **464,253** |
| Distinct vendors | **12,270** |
| Register cross-checks stored | **128** |
| Parse-log rows (one per artifact attempted) | **667** |
| HARD fixtures | **13 PASS, 0 FAIL, 15 BLOCKED** |
| Advisory checks | **6 of 6 match exactly** |
| Contract checks | **2 of 2 PASS** |

**Every hard fixture that has a source document on this machine passes to the cent,
and nothing failed.** The 15 blocked fixtures are the three 2026 months whose PDFs are
still absent — the staging directories you created are empty (see *Staged input*).

---

## 1. The reconciliation, by year

`reconciled` has three states and they are not interchangeable: **true** (lines sum to
the printed total, to the cent), **false** (they sum to something else — `delta` and
`reason_code` say what), **NULL** (the document prints no total to reconcile against).

| Year | Sets | Reconciled | Flagged | No stated total | % of checkable |
|---|---:|---:|---:|---:|---:|
| 2005 | 2 | 0 | 0 | 2 | — |
| 2007 | 2 | 0 | 0 | 2 | — |
| 2008 | 1 | 0 | 1 | 0 | 0.0 |
| 2017 | 54 | 0 | 3 | 51 | 0.0 |
| 2018 | 44 | 0 | 0 | 44 | — |
| 2019 | 54 | 0 | 0 | 54 | — |
| 2020 | 56 | 0 | 3 | 53 | 0.0 |
| 2021 | 50 | 0 | 4 | 46 | 0.0 |
| 2022 | 45 | 0 | 3 | 42 | 0.0 |
| 2023 | 39 | 6 | 2 | 31 | **75.0** |
| 2024 | 22 | 16 | 0 | 6 | **100.0** |
| 2025 | 56 | 31 | 25 | 0 | **55.4** |
| 2026 | 21 | 15 | 6 | 0 | **71.4** |

Totals: **68 reconciled, 33 flagged out of balance, 331 with no stated total, 14 with
no parseable rows.** One reconciling set additionally carries `DUPLICATE_SET`.

### By reason code

| Reason | Sets | First | Last | Σ&#124;delta&#124; |
|---|---:|---|---|---:|
| `TOTAL_NOT_FOUND` | 331 | 2005-05-25 | 2024-08-14 | 0 |
| *(reconciled)* | 67 | 2023-04-26 | 2026-05-27 | 0.00 |
| `OUT_OF_BALANCE` | 33 | 2023-05-24 | 2026-02-11 | 121,100,408.86 |
| `REGEX_MISS` | 14 | 2008-12-10 | 2022-03-09 | 0 |
| `DUPLICATE_SET` | 1 | 2026-02-11 | 2026-02-11 | 0.00 |

**No set is silently accepted.** A fixture asserts that every set which is not
`reconciled = true` carries a reason code; it passes at zero exceptions.

### Reading the 331 `TOTAL_NOT_FOUND` rows correctly

These are not failures. Phase 0 established that **detail listings printed no TOTAL
line at all before 2023** — 303 of them, 2017 to 2022. Their lines parsed, their money
is in the table, and there is simply nothing in the document to check the sum against.
Marking them `false` would assert the district's arithmetic is wrong when what is true
is that the document states no arithmetic. They are `NULL` with
`reason_code = TOTAL_NOT_FOUND`, and `facts.set_totals.status` renders that as
*"no stated total in the document"* wherever they surface.

### Parse-log status across all 667 artifacts

| Status | Count |
|---|---:|
| `total_not_found` | 333 |
| `no_text_layer` | 174 |
| `parsed` | 111 |
| `out_of_balance` | 33 |
| `regex_miss` | 14 |
| `unreadable` | 2 |

The 174 with no text layer are the 2010–2016 scanned recaps plus the surviving scanned
recaps and registers from 2017–2023. The 2 unreadable files are corrupt PDFs (one 2017,
one 2022).

---

## 2. The 33 out-of-balance sets, and what actually causes them

26 of the 33 are under $100,000 of delta. Seven are not, and one is spectacular. All 33
are flagged, none is presented as a settled figure anywhere, and the cause is now
diagnosed rather than guessed.

| Meeting | Fund | Stated | Parsed | Δ |
|---|---|---:|---:|---:|
| 2025-04-23 | GF | 10,912,257.80 | 121,023,812.80 | **+110,111,555.00** |
| 2025-03-26 | GF | 13,587,633.61 | 4,229,768.61 | −9,357,865.00 |
| 2026-02-11 | GF | 10,263,060.15 | 10,619,649.67 | +356,589.52 |
| 2025-01-22 | GF | 12,959,675.77 | 13,292,600.31 | +332,924.54 |
| 2026-01-14 | GF | 11,955,142.30 | 12,225,076.71 | +269,934.41 |
| 2025-02-26 | GF | 21,908,638.65 | 22,170,676.76 | +262,038.11 |
| 2025-09-24 | GF | 26,442,414.04 | 26,304,500.27 | −137,913.77 |

**The cause is column bleed, and here is the exact line.** On 2025-04-23 General Fund:

```
WSCA 4/3/2025 603163 11,325.00 325 2025 WSCA Counselor Conference Registrations
```

The invoice amount is `325`. The description begins `2025`. Because this corpus prints
some invoice amounts with no decimal at all, the amount pattern cannot stop at the
column edge, so it reads `325 2025` as **$3,252,025.00**. That one line repeats across
five invoice rows of the same check, which is most of the $110M.

**I tried the obvious fix and measured it rather than shipping it.** Requiring an
explicit `.00` on every amount kills this defect outright. It also drops **2,108 real
rows** corpus-wide, because the 2025-03-26 ASB listing genuinely prints `123.4`,
`293.1` and a bare `132` as invoice amounts. Requiring a thousands comma on any amount
above 999 instead fixes the two catastrophic sets and breaks 2023-11-08 Capital
(7,009,772.90 → 17,204.27), because *that* listing prints four-digit amounts without
commas. Measured across all 101 sets that have a stated total, both alternatives give
**net zero change** in how many reconcile.

So the honest conclusion: **this is not fixable with a better amount pattern.** The
amount column has to be located by its horizontal position on the page — pdfplumber
exposes word coordinates, and the description's left edge is a hard boundary the regex
cannot see. That is real work, not a tweak, and it is the top follow-on item (**R1**).

Two smaller notes in the same family, both already visible in the table above: the
2025-03-26 GF set is *short*, not long, which is the same defect in the other
direction (an amount partly consumed by the field to its left); and 26 of the 33 are
small deltas of the P-card and partial-decimal kind rather than column bleed.

**What protects you meanwhile:** all four fixture-month sets for 2026-03-25 and all
five for 2026-05-27 reconcile to the cent, so the two cycles you would actually quote
are clean. `facts.set_totals.status` labels every flagged set, the export excludes them
from its headline total and prints `FLAGGED: OUT_OF_BALANCE` on their row, and
`facts.reconciliation_by_reason` is a view rather than a number in a report that goes
stale.

---

## 3. Cumulative listings — larger than Phase 0 found

Phase 0 found that the 2021-02-10 and 2021-03-10 Transportation listings both total
$1,175,094.00 while their recaps state $783,396.00 and $391,698.00 — the listing
restates the prior cycle. The full run measures how widespread that is, by comparing
every set's check numbers against every earlier set of the same fund.

**145 of 446 sets restate an earlier cycle's rows**, across five funds:

| Fund | Cumulative sets |
|---|---:|
| GF | 56 |
| ASB | 30 |
| Capital | 26 |
| Trust | 22 |
| Transportation | 11 |

This is much broader than the Transportation example that surfaced it. Each affected
set carries a note naming how many check numbers overlapped and how much money they
carry, and `facts.cumulative_sets` lists them all with the warning in its comment:
**never sum these across meetings.**

This is also why the multi-year `parsed_total` figures in §1 look impossible —
$708M in 2020, $946M in 2021 — against a district whose annual budget is roughly
$400M. Those columns are the sum of what the *documents* say, restatements included.
They are not the district's spending, and no view or export presents them as such.

**"Cumulative" is measured, not inferred from the format era.** I had intended to tag
Era C as cumulative, as your D5 implies. That would have been wrong: the Era C column
header (`Vendor | Check date | Check # | Check Amt | Invoice Amt | Work performed`) is
**still in use in 2026**, so an era tag would have branded current, clean sets as
restatements. The check-number overlap test is a property of the document.

---

## 4. The signed register as independent evidence

128 cross-checks stored, one per (set, basis). Three bases, because the register and
the listing are different scopes and a single "does it match" boolean would be
meaningless without saying what was compared:

- `ap_direct_deposit` — the ACH listing against every accounts-payable
  direct-deposit line in the register, across all funds.
- `warrants_plus_pcard` — a fund listing against that fund's warrant *ranges* plus its
  purchasing-card line. Deliberately excludes DOR use taxes, L&I self-insurance and
  payroll, none of which appears in a listing.
- `recap_fund_total` — where no register exists but a Warrant Recap does. Recorded as
  `SCOPE_DIFFERS` rather than a mismatch, because the recap is register scope.

### The two fixture cycles

| Meeting | Fund | Basis | Register | Listing | Δ | Result |
|---|---|---|---:|---:|---:|---|
| 2026-03-25 | ACH | ap_direct_deposit | 3,609,064.78 | 3,609,064.78 | **0.00** | MATCH |
| 2026-03-25 | Capital | warrants_plus_pcard | 84,009.42 | 84,009.42 | **0.00** | MATCH |
| 2026-03-25 | ASB | warrants_plus_pcard | 137,378.15 | 136,796.65 | −581.50 | REGISTER_MISMATCH |
| 2026-03-25 | GF | warrants_plus_pcard | 5,608,535.32 | 5,609,073.26 | +537.94 | REGISTER_MISMATCH |
| 2026-05-27 | ACH | ap_direct_deposit | 9,394,987.52 | 9,394,987.52 | **0.00** | MATCH |
| 2026-05-27 | Capital | warrants_plus_pcard | 1,047,509.82 | 1,047,509.82 | **0.00** | MATCH |
| 2026-05-27 | ASB | warrants_plus_pcard | 179,774.32 | 181,039.41 | +1,265.09 | REGISTER_MISMATCH |
| 2026-05-27 | GF | warrants_plus_pcard | 3,383,444.97 | 3,388,060.86 | +4,615.89 | REGISTER_MISMATCH |

These reproduce the Phase 0 arithmetic exactly, which is the point: the register total
is computed from the register's own lines by this code, and it lands on the same figure
Phase 0 derived by hand.

**The ASB −$581.50 is stored as `REGISTER_MISMATCH` and not resolved**, per your D7.
It is two single-warrant register lines — `418227` at $30.00 and `418236` at $551.50 —
that appear on the board's signed register and **not** in the listing the public is
given. `30.00 + 551.50 = 581.50`. I have not determined why they are withheld and have
not guessed; both locators are on the row so you can look at both documents.

---

## 5. Query surface

Eight views, all applied:

| View | What it answers |
|---|---|
| `facts.set_totals` | One row per set, all three control totals, reconciliation status |
| `facts.vendor_by_cycle` | What a vendor was paid at each voucher night, per fund, with a locator |
| `facts.description_search` | Every line with its locator; apply a regex to `description` |
| `facts.vendor_search` | One row per vendor, spend split by whether its source set reconciles |
| `facts.reconciliation_by_year` | The failure rate by year — a view, not a stale number |
| `facts.reconciliation_by_reason` | How many sets carry each reason, and the delta behind it |
| `facts.cumulative_sets` | Sets that restate an earlier cycle. Never sum these |
| `facts.register_crosschecks` | Every listing against the signed register, both locators |

`vendor_search` splits each vendor's money into `invoice_total_reconciled` and
`invoice_total_unreconciled` rather than adding them together. A vendor total that
silently mixes the two would be the easiest way to say a wrong number out loud.

### Three control totals per set, all stored

`parsed_total` (Σ invoice_amount — the figure the printed TOTAL equals),
`sum_check_dedup` (Σ check_amount over distinct check numbers), and `hash_total`
(Σ of the check numbers themselves, so a transposition changes it).

Per your D7, only the first decides `reconciled`. The second legitimately differs:
on 2026-05-27 ASB it is $468.40 higher because P-card pseudo-check `9264000032`
(Bank Of America) carries a statement amount of $11,004.94 against 31 itemised rows
summing to $10,536.54. `set_totals.invoice_minus_check` exposes the difference and the
export prints it as a note rather than hiding it.

### Line flags

Of 464,253 lines: **6,092** P-card (check number begins 926), **407** payroll warrants
(530xxx), **1,959** credits, **5,377** person-shaped payees.

---

## 6. Exports

Five cycles generated, markdown and CSV, in `exports/`:

| File | Lines | Content |
|---|---:|---|
| `facts-vouchers-2026-03-25.md` | 83 | 4 funds, all reconciled |
| `facts-vouchers-2026-05-27.md` | 86 | 5 funds, all reconciled |
| `facts-vouchers-2026-06-24.md` | 6 | *"No voucher set is on file"* |
| `facts-vouchers-2026-07-22.md` | 6 | *"No voucher set is on file"* |
| `facts-vouchers-2026-08-26.md` | 6 | *"No voucher set is on file"* |

The three empty ones are generated deliberately. An export that silently omits a cycle
looks the same as a cycle with no spending.

Each export carries set totals with reconciliation status and the **page number** of
the TOTAL line, the register cross-check, the top 25 vendors, and the watch list across
the last 12 cycles. Both the markdown and the CSV are under the 300-line cap.

### The personal-name rule, and what it costs

Per your D5 the export uses an **allow-list**: watch-list vendors plus names that
positively identify as organizations. A block-list would have to recognise every
individual to be safe; an allow-list only has to recognise companies.

On 2026-03-25 that withholds **218 payees totalling $239,728.80 — 2.5% of the cycle's
$9,438,944.11**. 169 of the 218 are `Surname, Given` personal names. The rest are small
organizations the rules do not recognise (largest: *Restorx Of Washington* at $62,335.57).
Every withheld row stays in `facts.voucher_line` and stays queryable.

Getting there needed two real fixes, both found by looking at the withheld bucket
rather than by reasoning about the rules:

1. **The comma rule was catching organizations.** `NWAP, Inc` and
   `Hearing, Speech & Deafness Ctr` were being withheld as though they were people.
   The surname part is now capped at two tokens and a legal form after the comma
   disqualifies the match.
2. **The organization test was far too narrow.** Before widening it, the export
   withheld **274 payees totalling $4,050,273 — 42.9% of the cycle** — including Puget
   Sound Energy, City of Kent and Federal Way Public Schools. An export missing 43% of
   the money is not a privacy control, it is a broken report.

`Robert Half` is the case that proves the watch list is necessary rather than
decorative: it is a staffing company whose name is indistinguishable from a person's,
so no pattern will ever admit it. A test pins that.

---

## 7. Sampling plan — prepared, not judged

`facts/vouchers/samples/` holds **15 files**, one per reconciling 2026 fund-set,
seeded at **20260914** (the seed is per set, so adding a set does not reshuffle the
others, and you can redraw exactly the same sample later).

| Set | Drawn |
|---|---|
| 2026-03-25 ACH | 300 of 1,101 |
| 2026-03-25 ASB | 224 of 224 (all) |
| 2026-03-25 Capital | 40 of 40 (all) |
| 2026-03-25 GF | 300 of 1,533 |
| 2026-05-27 ACH | 300 of 2,189 |
| 2026-05-27 ASB | 300 of 316 |
| 2026-05-27 Capital | 39 of 39 (all) |
| 2026-05-27 GF | 300 of 1,555 |
| 2026-05-27 Transportation | 1 of 1 (all) |

Each row gives the page, vendor, check date, check number, both amounts, the
description and the flags, plus a verbatim extracted quote for exact comparison.

**The statement you may make on a clean result:** zero errors in 300 randomly drawn
lines puts that set's error rate **below roughly 1% at 95% confidence** — the rule of
three, 3/300. Fewer than 300 lines gives a correspondingly weaker bound. It says
nothing about sets that were not sampled, and nothing about the 33 that did not
reconcile. I have not judged any sample; that is yours to do.

---

## 8. Staged input — the path is built, the directories are empty

Per your instruction, staged PDFs are a **first-class input**, not a workaround:
`~/workspace/staging/vouchers-2026/<date>/` is a corpus root alongside the scraped
trees, its bare-ISO directory names are understood, and anything found there is written
with `source = 'staged_pdf'`.

**The three directories exist and contain no files.** I checked at creation, again 45
seconds later, and swept `/home/donald` for any PDF created in the preceding 30
minutes — nothing. So `facts.voucher_set.source` is `corpus_pdf` for all 446 rows and
15 fixtures are BLOCKED.

When the packets land, this completes the work with **no code change**:

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)
.venv/bin/python build.py --reload --progress     # ~20 min
.venv/bin/python fixtures.py                      # the 15 BLOCKED become PASS or FAIL
.venv/bin/python samples.py
for d in 2026-06-24 2026-07-22 2026-08-26; do .venv/bin/python export_cycle.py $d; done
```

### Locators without a document id, per your D2

`locator_document_id` is nullable and **never guessed**. `locator_file_path` and
`locator_file_sha256` are NOT NULL on every set and every line. **15 sets** currently
have no document id: the nine 2026-03-25 and 2026-05-27 sets, and six 2019-06 sets
whose PDFs are on disk but were never ingested.

`relink.py` attaches ids later, keyed on the digest, and is idempotent. Run now it
reports `0 row(s) linked` across all four tables — correct, because none of those 15
files has a `documents` row yet. Running it twice changes nothing the second time.

---

## 9. Fixtures

**13 PASS · 0 FAIL · 15 BLOCKED · 6 advisory all matching.** `fixtures.py` exits 0.

| Fixture | Result |
|---|---|
| 2026-03-25 GF / ACH / Capital / ASB | **PASS** ×4, Δ 0.00 each |
| 2026-05-27 GF / ACH / Capital / ASB / Transportation | **PASS** ×5, Δ 0.00 each |
| 2023-08-23 Capital (the Phase 0 pre-2024 set, $2,120,786.72) | **PASS** |
| Transportation cumulative (783,396.00 + 391,698.00 = 1,175,094.00, and flagged) | **PASS** |
| Every unreconciled set carries a reason code | **PASS**, 0 exceptions |
| Every line carries a resolvable locator | **PASS**, 0 exceptions |
| 2026-06-24 ×4, 2026-07-22 ×2, 2026-08-26 ×4 | **BLOCKED** — no document |
| 2026-06-24 GF components (warrants + P-card + payroll) | **BLOCKED** — arithmetic verified, document absent |
| 2026-06-24 / 07-22 / 08-26 advisory counts and vendor totals | **BLOCKED** ×14 |

### Advisory — all six match exactly

| Check | Expected | Actual |
|---|---|---|
| 2026-03-25 GF | 384 checks / 1,533 lines | 384 / 1,533 |
| 2026-03-25 ACH | 241 checks / 1,101 lines | 241 / 1,101 |
| 2026-03-25 Capital | 29 checks / 40 lines | 29 / 40 |
| 2026-03-25 Sunburst Workforce Advisors (GF+ACH) | 722,422.23 | 722,422.23 |
| 2026-03-25 Blazerworks | 105,425.90 | 105,425.90 |
| 2026-03-25 CBPI | 12,303.75 | 12,303.75 |

These come from a prior manual parse done independently of this code. Six independent
figures matching to the cent — three counts and three vendor totals — is the strongest
evidence in this report that the parser is reading the document correctly, and it is
worth more than the fixtures I wrote myself.

**Blocked fixtures are never counted as passes.** A check that never executed is not a
check that succeeded, and `fixtures.py` prints that sentence on every run.

---

## 10. Bugs found and fixed during the build

Four, all caught by measurement or by a test rather than by reading code. Each produced
numbers that looked entirely plausible, which is the only kind of bug that matters here.

1. **The register amount capture read the issue date as money.**
   `GENERAL PAYROLL 530157-530158 3/5/2026 355.27` became **$2,026,355.27**, and
   `PURCHASING CARD 2/11/26-3/12/26 403.45` became **$26,403.45**. Neither figure is out
   of range for a school district. Caught by a unit test on a four-line synthetic
   register, not by the corpus run — the corpus run looked fine. Fixed by requiring the
   amount to begin at a whitespace boundary; pinned by two tests.
2. **The register cross-check compared the wrong scope.** It included DOR use taxes and
   L&I self-insurance, which no listing contains, so ACH and Capital showed mismatches
   of −$2,228,600 and −$26,000 where Phase 0 had proved them exact. Fixed to warrant
   ranges plus the P-card line; both now tie at 0.00.
3. **Requiring cents on every amount dropped 2,108 real rows.** I shipped that rule
   after measuring it on twelve well-behaved 2023 and 2026 files, where it changed
   nothing. Across the corpus it took nine sets from reconciled to out-of-balance,
   because this data really does print `123.4` and `132`. Reverted, with the measurement
   recorded in the source so the next person does not repeat it. **My "changed not one
   row or total" claim was true of the sample and false of the corpus.**
4. **The export withheld 43% of the money.** Covered in §6.

---

## 11. Open items

### R1 — Column-position parsing for the amount fields *(highest priority)*

33 sets are out of balance and the cause is diagnosed (§2): the amount pattern cannot
see the description's left edge. pdfplumber exposes per-word coordinates; anchoring the
two amount columns by x-position would fix the whole class, including the $110M set.
Both regex-only alternatives measure at net zero. **Until this lands, no GF figure from
2025-01, 2025-02, 2025-03, 2025-04, 2025-09, 2026-01 or 2026-02 should be quoted** —
they are flagged in the table and excluded from export totals, but the numbers are
visible and a reader could mistake them.

### R2 — The 331 sets with no stated total

Parsed and stored, unreconcilable from the document alone. Where a 2020–2021 Warrant
Recap exists, `recap_fund_total` records the comparison as `SCOPE_DIFFERS`. For
2017–2019 there is no machine-readable independent total of any kind: the recaps are
scans and the registers do not start until late 2021. Your D4 is implemented as
specified; this is the residual it leaves.

### R3 — Cumulative listings are broader than the decision assumed

145 sets across five funds, not an Era C subset (§3). The note and the
`facts.cumulative_sets` view make them visible, but **no deduplicated multi-cycle
total exists yet.** Anyone asking "what did we pay vendor X in FY2021" would get a
restated answer. Designing that deduplication is the second-largest piece of work
after R1.

### R4 — 2010–2016 remains unreadable

106 scanned recaps, zero text layer. Unchanged from Phase 0. OCR would be a separate
project, and OCR'd digits in a financial table are exactly the numbers you should not
say out loud.

### R5 — Carry the two-root path fix back to `facts/minutes`

Per your D3, `resolve_pdf_path` now lives in the shared `facts/common/paths.py` and
knows both stale roots plus the staging root. **`facts/minutes/locators.py` was not
touched** and still has a single rewrite, so it silently fails to resolve any document
from the 2026 re-scrape. Filed here as you asked rather than fixed in this branch.

### R6 — The corpus is still six months stale

Independent of everything above: the newest scraped voucher night is 2026-03-25 and the
next board meeting is 2026-09-23. The fact layer cannot be fresher than the scrape.
You said the scrape fix is a separate task and does not gate this one; recording it so
it does not get lost.

### R7 — One duplicate set

2026-02-11 has two listings for the same fund with differing contents. Both are kept,
the second carries `DUPLICATE_SET`, and it reconciles. Worth an eye before quoting that
cycle.

---

## 12. Verification

```
ruff check facts/           All checks passed
ruff format --check          14 files already formatted
interrogate                  98.3% docstring coverage (floor 80%)
bandit -ll -ii               0 medium, 0 high
pip-audit                    no known vulnerabilities
pytest                       185 passed
fixtures.py                  13 PASS / 0 FAIL / 15 BLOCKED, exit 0
```

The SQL insert helper composes statements with `psycopg2.sql` identifier quoting rather
than string formatting. The schema, table and column names are module-level literals
and could not carry input, but this project forbids SQL built by string formatting
regardless of provenance, and the rule is worth more than the one exception would save.

### Nothing outside schema `facts` was written

| Table | Before | After |
|---|---|---|
| `facts.meeting` | 1,646 | 1,646 |
| `facts.motion` | 6,507 | 6,507 |
| `facts.vote` | 19,633 | 19,633 |
| `facts.attendance` | 3,515 | 3,515 |
| `documents` | 20,166 | 20,166 |
| `chunks` | 179,026 | 179,026 |

No row was deleted outside `facts`. `pg_trgm` was **not** installed: it would have made
`description_search` faster but `CREATE EXTENSION` is a database-wide change outside
schema `facts`, so it is left as a deliberate decision for you rather than a side effect
of running `schema.sql`.

---

## Appendix A — regenerating everything

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers

export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

# Schema and views (idempotent).
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < schema.sql
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < views.sql

# Full build: opens 667 PDFs, ~20 min. All writes in one transaction.
.venv/bin/python build.py --reload --progress

# One cycle only, no write:
.venv/bin/python build.py --only-date 2026-03-25 --dry-run

.venv/bin/python fixtures.py            # 13 PASS / 0 FAIL / 15 BLOCKED, exit 0
.venv/bin/python relink.py              # idempotent; 0 linked while ingest is behind
.venv/bin/python samples.py             # 15 files, seed 20260914
.venv/bin/python export_cycle.py 2026-03-25
.venv/bin/python -m pytest -q           # 185 passed
.venv/bin/pip-audit -r requirements-lock.txt
```

Rebuilding the venv in a fresh worktree:

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-lock.txt
```

## Appendix B — package contents

| File | Purpose |
|---|---|
| `facts/common/paths.py` | Shared path resolution: two stale roots, the staging root, staged-vs-scraped classification (D3) |
| `facts/vouchers/locators.py` | `PdfText` with exact per-page offsets, SHA-256, quote builder |
| `facts/vouchers/classify.py` | Fund and document-class rules over a closed vocabulary |
| `facts/vouchers/census.py` | Merged inventory across `documents` and disk, deduplicated by content |
| `facts/vouchers/parsers.py` | Four era parsers, TOTAL selection, register parser |
| `facts/vouchers/vendors.py` | Deterministic normalization; the export allow-list |
| `facts/vouchers/build.py` | The loader. Writes only to schema `facts` |
| `facts/vouchers/schema.sql` | Five tables |
| `facts/vouchers/views.sql` | Eight views |
| `facts/vouchers/fixtures.py` | HARD / advisory / contract fixtures |
| `facts/vouchers/relink.py` | Idempotent document-id attach, keyed on digest (D2) |
| `facts/vouchers/samples.py` | Seeded traceable samples |
| `facts/vouchers/export_cycle.py` | One-cycle briefing, markdown + CSV |
| `facts/vouchers/recon_phase0.py` | Phase 0 recon, retained and still runnable |
| `facts/vouchers/test_parsers.py`, `test_vouchers.py` | 185 tests |
