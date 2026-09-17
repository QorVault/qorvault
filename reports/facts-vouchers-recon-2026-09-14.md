# Phase 0 Recon — Voucher Fact Tables

**Date:** 2026-09-14
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers`, branched from `main` @ `29557ed`
**Package:** `facts/vouchers/`
**Database:** `boarddocs` on `127.0.0.1:5432`. Every corpus read used a `READ ONLY`
Postgres session. Nothing was written to `documents`, `chunks`, Qdrant or `rag_api`.
No `facts` table was created; Phase 0 writes no rows anywhere.

**No LLM touched any amount, date, vendor, check number or total in this report.**
Every figure below comes from `pdfplumber` text extraction, a regular expression,
and exact decimal arithmetic. Where I state a number, the command that produced it
is in Appendix A.

---

## STOP — and why

**The build brief names three stop conditions. One of them fires.**

| Stop condition | Threshold | Measured | Result |
|---|---|---|---|
| Monthly sets since 2024-09 | ~24 | 18 voucher nights (91 fund-sets) | **short by 6 nights** |
| Sets since 2015 with a text layer | ≥60% | **82.8%** (512 of 618) | PASS |
| 2026 fixture months on disk | 03-25, 06-24, 07-22, 08-26 | **only 03-25** | **FAIL** |

### The three missing fixture months do not exist on this machine

`2026-06-24`, `2026-07-22` and `2026-08-26` are not present anywhere under
`/home/donald` — not as meeting directories, not as PDFs, not inside any archive.
I searched for them four ways: by directory name, by file name in every date
notation the district uses, by PDF modification date after 2026-04-01, and by
listing the two zip archives on the machine. All four came back empty
(Appendix A, §A4).

The reason is upstream of this task. **The scraped corpus stops at 2026-03-25.**
The newest meeting directory in either corpus root is
`2026-03-25-regular-meeting-6-30-p-m-`, and the newest voucher PDF in the database
is dated 2026-02-11. The single exception is a hand-placed
`~/workspace/meeting_files/2026-05-27-regular-meeting-6-30-p-m-/` that somebody
copied in on 2026-05-27 and that no ingest run has ever seen.

Today is 2026-09-14. **The voucher record on this machine is six months stale**, and
the next voucher night is 2026-09-23 — nine days away. Fetching from BoardDocs is
explicitly out of scope for this task and `curl` is blocked by policy, so I did not
attempt it. This is a pipeline problem, not a parser problem, and it is the single
most consequential thing in this report: the fact layer cannot be more current than
the scrape that feeds it.

Ten of the fourteen HARD fixture totals in the brief belong to those three missing
months. I will not weaken them, invent them, or substitute a different month's
figures for them. Phase 1 stops here pending your decision (**D1** below).

---

## What I proved anyway

Stopping on the fixtures did not stop the recon. Everything else in Phase 0 ran to
completion, and the four fixtures that *are* reachable all pass.

**All four 2026-03-25 HARD totals match the PDFs exactly, to the cent.** I read them
out of the documents' own `TOTAL` lines rather than trusting the brief:

| Fund | Brief says | PDF `TOTAL` line | Page | Match |
|---|---:|---:|---|---|
| GF | 5,609,073.26 | 5,609,073.26 | p17 | ✓ |
| ACH | 3,609,064.78 | 3,609,064.78 | p13 | ✓ |
| Capital | 84,009.42 | 84,009.42 | p1 | ✓ |
| ASB | 136,796.65 | 136,796.65 | p3 | ✓ |

**And all four reconcile when parsed** — the sum of the invoice amounts equals the
printed total exactly, and the advisory line and check counts in the brief match to
the row:

| Set | Rows parsed | Brief (advisory) | Checks | Brief (advisory) | Parsed total | Δ vs stated |
|---|---:|---:|---:|---:|---:|---:|
| GF | 1,533 | 1,533 | 384 | 384 | 5,609,073.26 | **0.00** |
| ACH | 1,101 | 1,101 | 241 | 241 | 3,609,064.78 | **0.00** |
| Capital | 40 | 40 | 29 | 29 | 84,009.42 | **0.00** |
| ASB | 224 | — | 66 | — | 136,796.65 | **0.00** |

**I found the $1,322.35.** The brief records that a previous attempt parsed
2026-03-25 ASB at $135,474.30 against a stated $136,796.65. It is exactly one row,
on page 3:

```
THE HEATHMAN LODGE AND HUDSONS BAR AN03 /12/2026 418256   1,322.35   1,322.35 Cheet to State Hotel fee
```

Two defects collide on that one line. The vendor name is long enough to run into the
date column with **no space at all** between them (`...BAR AN` immediately followed by
`03 /12/2026`), and `pdfplumber` has split the date itself with a stray space. The
brief's row regex requires `\s{1,}` before the date and a date with no internal
space, so it fails, silently, on that row alone. `135,474.30 + 1,322.35 =
136,796.65`. Two changes to the regex — `\s*` instead of `\s{1,}`, and optional
spaces inside the date — close it, and they change nothing on any other row in any
of the four sets (pinned by tests `TestBriefRegexVersusFixed`).

**The 2026-06-24 double-TOTAL pitfall reproduces on a document I do have.** The
brief warns that 2026-06-24 GF prints `$2,026,444.30` and then a lone page reading
`$4,052,888.60`, exactly 2×. The 2026-05-27 Transportation listing does the same
thing: `TOTAL $173,922.13` on page 1 immediately after its single data row, then
pages 2–4 carrying nothing but the running header, then `TOTAL $347,844.26` on
page 5 — exactly 2×. The brief's rule ("the first TOTAL after the last data row is
the stated total") picks the right one, and the signed register confirms it
independently: `TOTAL TRANSPORTATION VEHICLE FUND $173,922.13`. **This pitfall can
be tested without the missing June PDF.**

**A whole substitute fixture month is available and clean.** Every fund in the
2026-05-27 set reconciles to the cent (§Fixture register, below).

---

## 1. Inventory

### What counts as a voucher document

I swept `documents` two ways — by title (`%voucher%`, `%warrant%`, `%bdmtg%`,
`%bd mtg%`, `%board mtg%`) and by first-page phrase (`General Fund Warrants`,
`ACH Payments`, `Capital Projects Fund Warrants`, and six more) — and then walked
the corpus directories on disk for voucher PDFs the database has never seen.
Deduplication is by SHA-256 of file content, not by name: the same money is filed
under `GF Vouchers 3-25-26.pdf` in one place and `General Fund Vouchers
03-25-26.pdf` in another, and a name-based rule would count it twice.

**859 distinct files. 815 are voucher artifacts. 44 are not, and are listed in
Appendix B rather than silently dropped.**

**A completeness check confirms nothing was missed.** I re-walked every corpus
directory afterwards and compared, by content hash, every PDF filed under a voucher
agenda item against the inventory. 407 such PDFs are not in the inventory; **all 407
are byte-identical duplicates** of a file that is. Truly missing: **zero**. A filter
that silently drops a voucher set looks exactly like one that drops a transcript,
so this check is the difference between believing the inventory and knowing it.

| Artifact class | n | What it is |
|---|---:|---|
| `detail_listing` | **446** | The monthly per-check listing. This is what Phase 1 parses. |
| `warrant_recap` | 173 | Fund-level totals only, no vendor rows. The *only* voucher artifact for 2010–2016. |
| `voucher_agenda_item` | 96 | The board's index entry for a voucher night (88 vouchers + 8 warrant-cancellation resolutions). |
| `warrant_register` | 48 | The signed register, certified under penalty of perjury. Independent evidence — see §4. |
| `voucher_by_vendor` | 26 | Year-to-date listing sorted by vendor. Overlaps the monthly sets; **never** a monthly set. |
| `warrant_cancellation` | 25 | Stale/outstanding warrant resolutions. An accounting correction, not a payment. |
| `unnamed_in_voucher_item` | 1 | A PDF named nothing useful, filed under a voucher agenda item. |

Treating the year-to-date vendor reports as monthly sets would have double-counted a
year of payments each time one appeared. They are classified out, not filtered away.

### Detail listings by fiscal year and fund

Washington school fiscal years run 1 September to 31 August, so `FY2026` is
1 Sep 2025 – 31 Aug 2026.

| FY | GF | ACH | Capital | ASB | Trust | Transp | Custod | Perm | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FY2005 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 |
| FY2007 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| FY2009 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| FY2010–FY2016 | — | — | — | — | — | — | — | — | **0** |
| FY2017 | 10 | 0 | 9 | 10 | 3 | 1 | 0 | 0 | 33 |
| FY2018 | 12 | 0 | 13 | 13 | 8 | 1 | 0 | 0 | 47 |
| FY2019 | 17 | 0 | 15 | 15 | 5 | 2 | 0 | 0 | 54 |
| FY2020 | 15 | 0 | 15 | 15 | 5 | 4 | 0 | 0 | 54 |
| FY2021 | 11 | 0 | 10 | 10 | 10 | 9 | 0 | 0 | 50 |
| FY2022 | 11 | 0 | 10 | 10 | 10 | 10 | 0 | 0 | 51 |
| FY2023 | 12 | 0 | 12 | 12 | 5 | 0 | 0 | 0 | 41 |
| FY2024 | 5 | 0 | 5 | 5 | 5 | 0 | 0 | 0 | 20 |
| FY2025 | 11 | 11 | 11 | 11 | 2 | 1 | 8 | 0 | 55 |
| FY2026 | 8 | 6 | 7 | 7 | 4 | 2 | 1 | 1 | 36 |
| **Total** | **116** | **17** | **108** | **108** | **57** | **30** | **9** | **1** | **446** |

Earliest detail listing **2005-05-25**; latest **2026-05-27**. Earliest voucher
artifact of any kind **2005-05-11**.

**There is a seven-year hole.** FY2010 through FY2016 contain no detail listing at
all — only 106 `Warrant_Recap.pdf` scans, none of which has a text layer (§2). For
those years the corpus records that money was approved and how much in total, and
nothing about who was paid.

### The fund vocabulary in the brief is too small

The brief specifies five funds (`GF|ACH|Capital|ASB|Trust`). The corpus publishes
**nine** listings, and the four extra ones pay real district money:

- **Transportation Vehicle Fund** — 30 listings, 2017–2026. The 2026-05-27 set alone
  is $173,922.13 (one school bus).
- **Custodial Fund** — 9 listings, all FY2025–FY2026. Appears on the signed register
  with its own fund total.
- **Permanent Fund** — 1 listing, 2026-01-14.
- **Vision Trust** — a short-lived 2017–2018 variant, classified under Trust.

A five-value `CHECK` constraint would reject 40 real sets, and the export would show
a vendor's Transportation spend as zero. See **D2**.

### Voucher nights on the agenda but with no listing

88 voucher agenda items across 87 distinct meeting dates, 2018-09-12 to 2026-03-25.
Only **two** have a voucher agenda item and no listing of any kind on file:
**2019-02-13** and **2019-04-24**.

### The voucher nights since 2024-09

18 nights, against the ~24 months the brief expects:

```
2024-10-09  2024-11-13  2024-12-11  2025-01-22  2025-02-26  2025-03-26
2025-04-23  2025-05-28  2025-06-25  2025-07-23  2025-08-13  2025-09-24
2025-10-22  2025-12-10  2026-01-14  2026-02-11  2026-03-25  2026-05-27
```

Missing: **2024-09, 2025-11, 2026-04, 2026-06, 2026-07, 2026-08, 2026-09**. The
first two may be genuine (boards skip months); the last five are the staleness
described in the STOP section.

---

## 2. PDF resolution and text layer

### Resolution: 693 of 693

**Every voucher artifact that could be parsed resolves to a file on disk.** That is
not what the minutes package's path rewrite would have produced, and the difference
is worth stating plainly.

`documents.file_path` is stale under **two** different roots, not one:

| Stored prefix | Resolves to | Why |
|---|---|---|
| `/home/donald/ksd_forensic/...` | `.../qorvault-dev-archive/framework-backup/home/ksd_forensic/...` | The bulk 2005–2026 corpus; the backup is its only copy. |
| `/home/donald/workspace/projects/ksd_forensic/boarddocs/data/...` | `.../boarddocs/data_DO_NOT_LOAD/...` | A later 2026 re-scrape whose `data` directory was renamed after ingest. |

`facts/minutes/locators.py` implements only a single generic
`/home/donald/` → archive rewrite. Applied to the second root it produces a path that
does not exist, and a missing rewrite is indistinguishable from a missing file. The
voucher package tries the rewrites **in order**, most specific first, and a test
pins that ordering (`TestPathResolution.test_workspace_rewrite_is_tried_first`).
This is worth carrying back to the minutes package — see **D6**.

- 693 parseable artifacts, **693 resolved (100%)**.
- **649 have a `documents` row** and therefore a document id.
- **44 have no `documents` row at all.** They exist on disk and have never been
  ingested. They include *every file in both 2026 fixture-capable sets* — the
  2026-03-25 and 2026-05-27 listings and both signed registers. This breaks the
  locator contract as written; see **D3**.

### Text layer, full pass over all 693

Not a sample. Every PDF was opened with `pdfplumber` and every page's text
extracted. Two files are unreadable (corrupt), one from 2017 and one from 2022.

| Year | With text | Image only | Unreadable |
|---|---:|---:|---:|
| 2005 | 2 | 0 | 0 |
| 2007 | 2 | 0 | 0 |
| 2008 | 1 | 0 | 0 |
| **2010–2016** | **0** | **106** | **0** |
| 2017 | 54 | 17 | 1 |
| 2018 | 44 | 13 | 0 |
| 2019 | 54 | 17 | 0 |
| 2020 | 76 | 2 | 0 |
| 2021 | 70 | 0 | 0 |
| 2022 | 54 | 9 | 1 |
| 2023 | 41 | 9 | 0 |
| 2024 | 27 | 1 | 0 |
| 2025 | 67 | 0 | 0 |
| 2026 | 25 | 0 | 0 |

Since 2015: **512 of 618 (82.8%)** carry a text layer — comfortably past the 60%
stop threshold.

**The 2010–2016 block is a hard floor, not a parsing problem.** Those 106 files are
scanned images of a printed recap. No regex will ever read them. Making them
machine-readable means OCR, which is a separate project with its own accuracy
problem — and OCR'd digits in a financial table are exactly the kind of number you
should not say out loud at a board meeting. I recommend leaving them out and
recording *why* they are out, so the gap is visible rather than silent.

### Where a stated total actually exists

This is the table that decides how much of the corpus the "reconcile or flag"
contract can even apply to, because a set with no printed total has nothing to
reconcile *against*.

| Year | Detail listings (n / with text / **printing a TOTAL**) | Recaps (n / text / TOTAL) | Registers (n / text / TOTAL) |
|---|---|---|---|
| 2005–2008 | 5 / 5 / **0** | 0 | 0 |
| 2010–2016 | 0 | 106 / 0 / 0 | 0 |
| 2017 | 54 / 54 / **0** | 17 / 0 / 0 | 1 / 0 / 0 |
| 2018 | 44 / 44 / **0** | 13 / 0 / 0 | 0 |
| 2019 | 54 / 54 / **0** | 17 / 0 / 0 | 0 |
| 2020 | 56 / 56 / **0** | 13 / 11 / 9 | 0 |
| 2021 | 50 / 50 / **0** | 6 / 6 / 6 | 4 / 4 / 4 |
| 2022 | 45 / 45 / **0** | 1 / 0 / 0 | 11 / 2 / 2 |
| 2023 | 39 / 39 / **8** | 0 | 11 / 2 / 1 |
| 2024 | 22 / 22 / **16** | 0 | 6 / 5 / 4 |
| 2025 | 56 / 56 / **56** | 0 | 11 / 11 / 10 |
| 2026 | 21 / 21 / **21** | 0 | 4 / 4 / 3 |

**Before 2023 the detail listings do not print a total at all.** 303 listings,
2017–2022, with vendor rows and no stated total. The contract "lines must sum to it
to the cent, or the set is `reconciled=false` with a delta" has no `it` for those
years. They would all be `TOTAL_NOT_FOUND`, forever, which is a true statement about
the record but not a useful one. See **D4**.

---

## 3. Format eras

The brief asks for eras grouped by layout. Grouping on the printed column-header
line alone produced **55** distinct signatures, which is wrong — the header is two
physical lines that wrap differently depending on page width, so the same report
generator yields several signatures. **The reliable era key is the column *order*
plus whether amounts carry a `$`.** On that basis there are four eras, and I read a
document from each rather than inferring them.

### Era A — 2005 to 2008 · "KSD Voucher Register"

Columns: `Voucher Number | Vendor Name | Amount | Description`. **No check date and
no check number at all**, and a single amount rather than a check/invoice pair. The
brief's row model does not apply to this era; it would need its own parser and would
produce rows that cannot carry `check_date` or `check_number`.

> Locator: `2007-03-28` · doc `81a56777-6fce-428b-9f6e-0594d5027ec1` ·
> `General_Fund.pdf` · p1 ·
> "KSD VOUCHER REGISTER / VOUCHER DATES: 09-MAR-07 TO 22-MAR-07 / GENERAL FUND WARRANTS1"

5 listings. No TOTAL line.

### Era B — 2017 to 2019 · "Voucher Register", check number first

Columns: `CHECK NO. | VENDOR | DATE | DESCRIPTION | INVOICE | INV. TOTAL`.
**The check number comes first, before the vendor** — the reverse of every later
era. Amounts are `$`-prefixed. Descriptions wrap onto continuation lines.

> Locator: `2017-01-25` · doc `ea009d2a-2937-4685-adf5-9e0a71fd7875` ·
> `ASB_Fund_Vouchers_1-25-17.pdf` · p1 ·
> "CHECK NO. VENDOR DATE DESCRIPTION INVOICE INV. TOTAL /
> 411037 Area 5 DECA 1/12/2017 Registration/Testing Fees $ 171.00 $ 171.00"
>
> Also: `2019-01-09` · doc `97d72e50-fcaf-4bf8-b43a-72bc900033ca` ·
> `ASB Vouchers 1.9.19.pdf` · p1

No TOTAL line.

### Era C — 2020 to 2022 · "Voucher Register", vendor first, `$` amounts

Columns: `Vendor | Check date | Check # | Check Amt | Invoice Amt | Work performed`.
Vendor-first, which is the modern order, but the amounts are `$`-prefixed and
`pdfplumber` frequently splits a digit off (`$  2 25.00`).

> Locator: `2021-01-13` · doc `bb01122b-80f5-472c-bcc4-74b046d0df8d` ·
> `ASB Vouchers 1.13.2021.pdf` · p1 ·
> "(individual payee, name withheld) 9/24/2020 414835 $ 450.00 $ 2 25.00 Choreography for Northwood Dance team"
>
> Also: `2022-01-12` · doc `025c10bd-a3cb-4d26-b4a4-b1ea67f54a7f` ·
> `ASB Vouchers 1.12.2022.pdf` · p1

No TOTAL line.

### Era D — 2023 to 2026 · current, with a printed TOTAL

Columns: `Vendor | Check Date | Check Number | Check Amount | Invoice Amount |
Description`. This is the era the brief's regex targets, and it is the first era
that prints a whole-line `TOTAL`. It has **two sub-variants** that matter:

- **D1, `$`-prefixed** — e.g. `2026-02-11`, `2026-01-14`, `2025-01-22`.
  `Academy Schs 01/29/2026 607141 $ 8 ,108.50 $ 8 ,108.50 ...`
  > Locator: `2026-02-11` · doc `1104b09c-1b80-4dfe-accc-492317e87bec` ·
  > `General_Fund_Vouchers_02-11-26.pdf` · p1
- **D2, plain** — e.g. `2026-03-25`, `2026-05-27`.
  `911 Interpreters Inc 02/12/2026 607335 1,435.54 1,435.54 Open PO for 2025-2026 school year`
  > Locator: `2026-03-25` · no `documents` row ·
  > `.../2026-03-25-.../9-12-ds4muh5ca61b-vouchers/General Fund Vouchers 03-25-26.pdf` · p1

### What the era structure costs

Running the brief's row regex, as written, over all 446 detail listings:

| Outcome | n | Meaning |
|---|---:|---|
| **reconciles** | 47 | Rows sum to the printed total, to the cent |
| out of balance | 15 | Rows parsed, total found, they disagree |
| `TOTAL_NOT_FOUND` | 116 | Rows parsed, no printed total to check against |
| `REGEX_MISS` | 268 | Zero rows parsed |

**268 zero-row listings sounds like the data is unusable. It is not.** I tested an
era-aware regex (allowing a `$` prefix and, separately, the check-first column
order) against a stratified random sample of 37 of the 268, one to four per year:

- **30 of 37 recovered rows** — the single dominant cause is the `$` prefix, which
  the brief's amount pattern does not admit.
- 7 stayed at zero: the five Era A documents (different columns entirely), one
  by-vendor report, one 2021 Transportation file.
- Of the 30 that recovered, **22 have no TOTAL line** (they are Era B/C), 3
  reconciled, and 5 were out of balance.

So the corpus is parseable with four era parsers. What it is *not* is reconcilable
before 2023, because the anchor does not exist in the document.

### Era C listings are cumulative — a trap worth naming

While checking whether the Warrant Recap could serve as the missing Era C anchor, I
found something more important. The 2021-02-10 and 2021-03-10 Transportation
listings **both parse to exactly $1,175,094.00.** The recaps for those two meetings
state $783,396.00 and $391,698.00 — and `783,396.00 + 391,698.00 = 1,175,094.00`.
The same pattern holds for 2020-06-24 (listing $855,839.26; recap $129,299.81; the
$726,539.45 difference is precisely the 2020-05-27 listing).

**In Era C the detail listing restates prior cycles' rows rather than covering only
the new period.** Summing Era C listings across meetings would double- and
triple-count real money. Any Era C parser must deduplicate by check number across
sets, or filter by the stated period — and until that is designed, no Era C figure
should be quoted aloud. This is the sharpest edge found in Phase 0.

---

## 4. Signed warrant registers

### Coverage

48 registers. They begin in earnest in September 2021.

| Year | Registers | Meetings with a register | Meetings with a detail listing | **Both** |
|---|---:|---:|---:|---:|
| 2017 | 1 | 1 | 16 | 1 |
| 2018–2020 | 0 | 0 | 42 | 0 |
| 2021 | 4 | 4 | 10 | 4 |
| 2022 | 11 | 10 | 11 | 10 |
| 2023 | 11 | 11 | 11 | 11 |
| 2024 | 6 | 5 | 5 | 5 |
| 2025 | 11 | 11 | 11 | 11 |
| 2026 | 4 | 4 | 4 | 4 |

**45 meetings have both artifacts.** From 2021-09 onward, essentially every voucher
night does. Nine of the eleven 2022 registers are scans with no text layer; from
2023 on they are all machine-readable.

### Is the register independent evidence, or the same listing with a signature page?

**Independent, decisively — and I can prove it arithmetically rather than assert it.**

The register is a different document produced from a different query. It is
organised by fund and payment *type*, it lists warrant number ranges and issue
dates rather than vendors, it includes payroll and electronic transfers that the
detail listings never show, and it carries the auditing officer's certification
under penalty of perjury.

Take 2026-03-25. The register's `TOTAL GENERAL FUND` is **$45,109,568.76**; the
General Fund detail listing's `TOTAL` is **$5,609,073.26**. Those are not
contradictory — the register includes $19.4M of payroll direct deposit, $6.9M of
payroll taxes, $5.65M of SEBB medical and more, none of which is a voucher.

The relationship is exact at the **warrant-range level**, which is what makes it a
real cross-check:

| Check | Computed from the register | Detail listing total | Δ |
|---|---:|---:|---:|
| **ACH** = every `ACCOUNTS PAYABLE / DIRECT DEPOSIT-ELECTRONIC TRANSFER` line, all funds | 3,609,064.78 | 3,609,064.78 | **0.00** |
| **Capital** = Capital AP warrant ranges + Capital P-card | 84,009.42 | 84,009.42 | **0.00** |
| **ASB** = ASB AP warrant ranges + ASB P-card | 137,378.15 | 136,796.65 | −581.50 |
| **GF** = General AP warrant ranges + General P-card | 5,608,535.32 | 5,609,073.26 | +537.94 |

Two of the four tie to the cent. The ASB gap of exactly **$581.50** is two
single-warrant register lines — `418227` ($30.00) and `418236` ($551.50) — that
appear on the register and not in the listing. `30.00 + 551.50 = 581.50`. I have not
determined why they are withheld; a refund to a named individual is the obvious
hypothesis and would be consistent with the export rule in the brief, but I did not
verify it and am not asserting it.

**This is a finding, not a defect.** Two warrants approved by the board do not
appear in the document the public is given. Whatever the reason, `facts.voucher_
reconciliation` should store that delta as a first-class value rather than log it.

**The ACH result also settles a modelling question.** ACH is not a fund — it is a
payment method that spans General, Capital, ASB and Custodial. The district
publishes it as its own listing with its own total, so `voucher_set.fund = 'ACH'` is
reasonable bookkeeping, but "a vendor's ACH total" is not "a vendor's General Fund
spend". The export's watch-list (GF + ACH) is correct as specified; it just needs
to say what it is measuring.

---

## 5. Vendor name shape

Sampled from the parsed rows of every detail listing that parses: **51,338 vendor
strings, 3,527 distinct.** One caveat, stated because it affects how you read the
distinct count: the sample takes the first 400 rows of each listing, and listings
are alphabetical, so the *distinct* figure is biased toward names early in the
alphabet. The *shape* findings below are not affected by that bias.

| Signal | Count (of 3,527 distinct) | What it implies |
|---|---:|---|
| Leading/trailing whitespace | **0** | A trim is still required, but nothing depends on it |
| Internal double spaces | **0** | No whitespace collapsing needed beyond the trim |
| Corporate suffix (`Inc`, `LLC`, `Co`, `Corp`, …) | 890 | **Keep them.** Stripping `Inc` merges distinct legal entities |
| `Surname, Given Middle` — person-shaped | **654 (18.5%)** | Refund and reimbursement payees. Critical for the export rule |
| All-caps | 70 | `KCDA`, `WA ST Patrol`, `(individual payee, name withheld)` |
| Case-only duplicates | **11 pairs** | e.g. `AMAZON CAPITAL SERVICES` / `Amazon Capital Services` |
| Contains `&` | 91 | `JW Pepper & Son Inc` |
| Contains `-` | 113 | `Genuine Parts Co-Seattle DC`, `US Foods - Seattle` |
| Contains `'` | 36 | `Coeur D'Alene French Baking Co`, `Penny's Salsa Inc` |
| Contains a digit | 46 | `911 Interpreters Inc`, `Area 5 DECA` |
| Single token | 71 | `Comcast`, `KCDA`, `Costco` |
| Trailing punctuation | 8 | Strip `.` `,` `-` `&` from the end |
| Longest name | 47 chars | No fixed-width truncation; the distribution tapers smoothly |

### The normalization rules the data actually requires

Deterministic, exact-and-case-insensitive only, no fuzzy matching — as the brief
requires, and now with the evidence for each rule:

1. **Trim, then collapse internal whitespace.** Costs nothing; guards against a
   future extractor that is less tidy than this one.
2. **Strip trailing `.`, `,`, `-`, `&`.** 8 names need it.
3. **Casefold for the key; keep the raw string for `display_name`.** This merges
   exactly 11 pairs and nothing else — measured, not assumed. `KCDA` must not become
   `Kcda` on screen.
4. **Do not strip corporate suffixes, and do not expand abbreviations.** The corpus
   is full of `Svc`/`Svcs`, `Sys`, `Sol`, `Publ`, `Prod`, `Mgmt`, `Intl`, `Lrning
   Diff`. Every expansion rule is a guess, and a guess that merges two vendors is
   unrecoverable once the rows are written.
5. **Flag `Surname, Given` as person-shaped** with a literal comma rule.

### The export's "never a personal name" rule needs a decision

The brief says the export never lists a personal name. The comma form catches 654 of
them reliably. But the corpus also pays individuals in `Given Surname` form —
`(individual payee, name withheld)`, `(individual payee, name withheld)`, `(individual payee, name withheld)`, `(individual payee, name withheld)` — and **no
deterministic rule distinguishes those from a two-word company** without a name
dictionary, which is a guess by another route. The only safe deterministic rule is
positive: an allow-list of watch-list vendors plus everything carrying a corporate
suffix; everything else is withheld from the export and stays in the table. See
**D5**.

(One vendor-master artifact is worth knowing about:
`(individual payee, name withheld)` is a real vendor string in this data.)

---

## Fixture register — every fixture in the brief

### HARD — reachable and passing

| Fixture | Status |
|---|---|
| 2026-03-25 GF 5,609,073.26 | **PASS** — stated total read from p17; parsed 1,533 rows / 384 checks, Δ 0.00 |
| 2026-03-25 ACH 3,609,064.78 | **PASS** — p13; 1,101 rows / 241 checks, Δ 0.00 |
| 2026-03-25 Capital 84,009.42 | **PASS** — p1; 40 rows / 29 checks, Δ 0.00 |
| 2026-03-25 ASB 136,796.65 | **PASS** — p3; 224 rows / 66 checks, Δ 0.00 (the missing $1,322.35 row found and explained) |

### HARD — blocked, source document absent

| Fixture | Status |
|---|---|
| 2026-06-24 GF / ACH / Capital / ASB | **BLOCKED** — no 2026-06-24 PDF on this machine |
| 2026-07-22 ACH / Capital | **BLOCKED** — no 2026-07-22 PDF |
| 2026-08-26 GF / ACH / Capital / ASB | **BLOCKED** — no 2026-08-26 PDF |
| 2026-06-24 GF components (warrants 608270–608506 = 1,800,211.05; P-cards 9261000039–42 = 224,719.75; payroll 530162 = 1,513.50) | **BLOCKED** — arithmetic checks out (`1,800,211.05 + 224,719.75 + 1,513.50 = 2,026,444.30`) but cannot be verified against a document |

### HARD — two pre-2024 sets, one per older era

The brief asks for two pre-2024 sets with stated totals read from their PDFs and
recorded here before parsing. I can deliver one honestly and not the other, and the
reason is the §2 finding.

**Era D-early — 2023-08-23 Capital. A clean fixture.**

> `Capital Projects Fund Vouchers 7-21-23 to 8-10-23.pdf`
> doc `1eb3a878-8998-4f41-a966-2b2c3b3ef38b`
> sha256 `eb157ca7a6109cdc93c47c2294b252c580214253fd2d3db58f2261237d51e345`
> **Stated TOTAL: $2,120,786.72** (page 2, read from the document's own TOTAL line)
> Parsed: 67 rows, 41 checks, sum(invoice) = sum(deduped check) = **2,120,786.72**, Δ **0.00**
> Hash total (sum of check numbers): 1,018,491,476

**Era C — 2021-10-13 Capital. Cannot be a to-the-cent fixture.**

> `Capital Projects Vouchers 10.13.2021.pdf`
> doc `d069f24b-be3c-4ca5-8361-5ae04de92a6b`
> **The document prints no TOTAL line.** The nearest stated figure is on the signed
> register `BD Mtg - 10-13-21.pdf` (doc `d9c77b51-16ae-44c0-aa56-6f111679750b`):
> `TOTAL CAPITAL PROJECTS FUND $6,239,703.47` — which is register scope, including
> electronic transfers the listing never shows, so equality is not expected and a
> HARD fixture asserting it would be asserting something false.

I am recording it as an **advisory cross-check at $6,239,703.47**, not a fixture,
and flagging that no pre-2023 to-the-cent fixture can be constructed from these
documents. Manufacturing one by choosing a tolerance would defeat the point of
having it.

### Substitute month available — 2026-05-27, every fund reconciles

Totals read from each PDF's own TOTAL line, then parsed independently:

| Fund | Stated TOTAL | Page | Rows | Checks | Parsed | Δ |
|---|---:|---|---:|---:|---:|---:|
| GF | 3,388,060.86 | p26 | 1,555 | 350 | 3,388,060.86 | **0.00** |
| ACH | 9,394,987.52 | p54 | 2,189 | 402 | 9,394,987.52 | **0.00** |
| Capital | 1,047,509.82 | p1 | 39 | 27 | 1,047,509.82 | **0.00** |
| ASB | 181,039.41 | p8 | 316 | 100 | 181,039.41 | **0.00** |
| Transportation | 173,922.13 | p1 | 1 | 1 | 173,922.13 | **0.00** |

Signed register `BDMTG - 5-27-2026 SIGNED.pdf` present, with text.

### ADVISORY — vendor watch list

The vendor-level advisory figures (Sunburst Workforce Advisors, Elevation
Healthcare, Blazerworks, CBPI, Positive Behavior Supports) are all for 2026-03-25,
06-24, 07-22 and 08-26. Only 2026-03-25 is reachable. I have **not** checked the
03-25 vendor figures: doing so requires a vendor rollup, which is Phase 1 work, and
Phase 1 has not been authorised. They are carried forward, not dropped.

---

## The control-total contract needs one change

The brief specifies three control totals per set and asserts
`sum(invoice_amount) == sum(deduped check_amount)`. **That assertion is false on
real data, and the reason is benign.**

On 2026-05-27 ASB: sum(invoice) = 181,039.41 (which equals the stated TOTAL exactly)
but sum(deduped check) = 181,507.81. The whole 468.40 difference is one P-card
pseudo-check — `9264000032`, Bank Of America, 31 itemised rows summing to 10,536.54
against a statement amount of 11,004.94. The P-card statement total exceeds the
transactions itemised in the listing.

On 2026-05-27 ACH: one row, check `8888888888` (Electrocom Inc), 6 lines summing to
0.00 invoice against a check amount of **−2,887.19**. A credit, carried on a
sentinel check number.

So: **`sum(invoice_amount) == stated_total` is the hard reconciliation.
`sum(deduped check_amount)` is a second control total that legitimately differs on
P-card and credit rows.** Both should be stored, and the difference between them
recorded — but only the first should decide `reconciled`. Asserting equality would
have failed 127 of the sets that parse, every one of them for a correct reason.
See **D7**.

---

## Decisions I need from you

**D1 — The three missing fixture months.** *(blocking Phase 1)*
Three of four HARD fixture months are not on this machine and I cannot fetch them.
Options, in the order I would rank them:

- **(a) You supply the PDFs.** Drop the 2026-06-24, 07-22 and 08-26 voucher packets
  into a directory and tell me where. You have BoardDocs access; I do not. This
  keeps every HARD fixture in the brief intact and is the only option that does.
- **(b) Re-scope the fixtures to what exists.** 2026-03-25 (4 funds, all verified
  above) plus 2026-05-27 (5 funds, all verified above) gives nine to-the-cent HARD
  fixtures across two months, including a live reproduction of the double-TOTAL
  pitfall. Phase 1 starts immediately; the specific June/July/August totals in the
  brief remain unverified.
- **(c) Fix the scrape first.** The corpus is six months stale and the next voucher
  night is 2026-09-23. Whatever we build now will be showing you March data on the
  night. This is the option that fixes the underlying problem, but it is a different
  task.

*(a) and (c) are not exclusive, and doing (b) now does not prevent either.*

**D2 — Extend the fund vocabulary from five to nine.**
`Transportation`, `Custodial`, `Permanent` (and `Trust` covering the old Vision
Trust). 40 real sets and roughly $174k of the 2026-05-27 night alone fall outside
the five-value list. I recommend nine. The cost is a wider `CHECK` constraint; the
cost of not doing it is a bus that the export says was never bought.

**D3 — Locators for PDFs that have no `documents` row.**
44 voucher PDFs exist on disk with no database row — including **every file in both
2026 sets that carry the fixtures**. `locator_document_id uuid REFERENCES
documents(id)` cannot be satisfied for them. My proposal, and I will do this unless
you say otherwise:

- `locator_document_id` stays a nullable FK — real document id where one exists.
- Add `source_path` and `source_sha256` to `voucher_set`, always populated. The
  SHA-256 makes the citation verifiable even if the file moves.
- Add `agenda_item_document_id` — the voucher agenda item *does* have a row
  (2026-03-25 is `9.12 Vouchers`), so every set stays anchored to a real document.
- `locator_page`, `locator_char_offset` and `locator_quote` resolve against the PDF
  regardless.

The alternative is re-ingesting those 44 files into `documents`, which writes
outside `facts` and is forbidden by this task's rules. I am not doing that without
you explicitly asking.

**D4 — What to do with 2017–2022.**
303 detail listings with vendor rows and no printed total. They can be parsed, but
they cannot be reconciled, and Era C additionally restates prior cycles (§3). Three
options: (a) parse and store them with `reason_code = TOTAL_NOT_FOUND` and a
prominent "not reconciled" flag everywhere they surface; (b) parse them but keep
them out of the query surface until the Era C cumulative problem is solved; (c)
scope Phase 1 to 2023-onward and treat the earlier era as a follow-on. I lean
**(c) then (b)** — it gets you correct numbers for the 2026-09-23 meeting first and
does not risk a double-counted figure being said aloud.

**D5 — The export's personal-name rule.**
An allow-list (watch-list vendors + names carrying a corporate suffix) is the only
deterministic rule that cannot leak a `Given Surname` payee. It is conservative: it
will also withhold some legitimate small businesses from the export. They stay in
the table and remain queryable. Confirm that trade is the right one.

**D6 — Carry the two-root path fix back to `facts/minutes`.**
`facts/minutes/locators.py` has a single rewrite and will silently fail to resolve
any document from the 2026 re-scrape. I have not touched the minutes package — it is
outside this task — but it is a live defect in shipped code.

**D7 — Accept `sum(invoice_amount) == stated_total` as the hard reconciliation**,
with `sum(deduped check_amount)` stored as a second control total whose difference
is recorded rather than asserted to be zero. Evidence above.

---

## Risks and open items

1. **Staleness is the biggest risk to the 2026-09-23 meeting.** The newest voucher
   data on this machine is 2026-05-27, hand-copied. Nothing built here changes that.
2. **Era C cumulative listings.** Until the deduplication rule is designed and
   tested, no 2020–2022 figure should be quoted. Named in §3; not yet solved.
3. **Two corrupt PDFs** (one 2017, one 2022) could not be opened at all. They are
   counted as `unreadable`, not as missing.
4. **Fractional-cent parses.** Five of the 15 out-of-balance sets parse to amounts
   with more than two decimal places (e.g. 2023-05-24 Capital at
   `1,874,334.4036`), which means the permissive amount pattern is capturing across
   a column boundary. It surfaces as out-of-balance rather than as a silently wrong
   number — the design working as intended — but it is Phase 1 work.
5. **Gross over-parses.** 2025-04-23 GF parses to $127.4M against a stated $10.9M.
   Same root cause as (4). Flagged, not hidden.
6. **A mislabelled file.** `TVF_Vouchers_02-11-26.pdf` (doc
   `a2325718-7bc7-4cfa-aeb3-73a325f2796b`) prints `Permanent Fund Warrants` as its
   header. The title says Transportation. **This is why the printed phrase must beat
   the file name when they disagree** — and it is a small but real example of the
   district's own labelling being wrong.
7. **13 artifacts I could not classify** are listed in Appendix B and excluded. Per
   the stop rule, they are left out and listed rather than guessed at.
8. **The vendor distinct-count is alphabetically biased** (first 400 rows per
   listing). Shape findings are unaffected; the 3,527 figure is a floor.

## Documentation impact

- `CLAUDE.md` documents the database as `qorvault/qorvault`; it is
  `boarddocs/boarddocs`. Flagged only — I do not modify CLAUDE.md files.
- `documents.file_path` is stale under two roots, not one (§2). Anything reading
  that column without both rewrites is silently broken.
- The pre-2019 corpus exists only inside a backup directory
  (`/home/donald/qorvault-dev-archive/framework-backup/`). Fourteen years of
  financial records with a backup as their sole copy is a durability risk, and it is
  the same risk the minutes debrief raised on 2026-09-13. It has not changed.
- The live corpus directory is named `data_DO_NOT_LOAD`, which suggests somebody
  deliberately took it out of the ingest path. I have only read from it.

## System state

- **Nothing was written outside `facts/vouchers/` and `reports/`.** No `facts` table
  was created. No row was written or deleted anywhere in Postgres.
- All corpus reads used `READ ONLY` sessions.
- Credentials injected at runtime from the container config. `.env` was neither read
  nor edited. No password was printed.
- Virtualenv at `facts/vouchers/.venv`, gitignored, built from
  `requirements-lock.txt` only. `pip-audit` reports **no known vulnerabilities**.
- **99 unit tests, all passing.** `ruff check` clean, `ruff format` clean,
  `interrogate` docstring coverage 98% (floor 80%), `bandit` 0 medium and 0 high.
- No hook was bypassed. No hook blocked anything this session.

---

## Appendix A — commands

Every command below was run from `~/workspace/projects/ksd-vouchers/facts/vouchers`
unless stated. `PGPASSWORD` is injected from the container config in every shell
that needs it and is never echoed:

```bash
export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)
```

### A1 — worktree and environment

```bash
cd ~/workspace/projects/ksd-main
git worktree add ~/workspace/projects/ksd-vouchers -b claude/facts-vouchers main

cd ~/workspace/projects/ksd-vouchers/facts/vouchers
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements-lock.txt
.venv/bin/pip-audit -r requirements-lock.txt          # no known vulnerabilities
.venv/bin/python -m pytest test_vouchers.py -q        # 99 passed
```

### A2 — corpus survey (all read-only)

```bash
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -c \
  "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY; \d documents"

# title shapes across every voucher/warrant-shaped document
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -c \
  "SELECT regexp_replace(title,'[0-9]{1,4}','#','g') AS shape, document_type,
          count(*), min(meeting_date), max(meeting_date)
     FROM documents
    WHERE title ILIKE '%voucher%' OR title ILIKE '%warrant%' OR title ILIKE '%BDMTG%'
    GROUP BY 1,2 ORDER BY 3 DESC"

# corpus end date
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -c "SELECT max(meeting_date) FROM documents"
```

### A3 — corpus roots on disk

```bash
ls /home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data | wc -l   # 1684
ls /home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD | wc -l               # 729
ls /home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD/2026-03-25-regular-meeting-6-30-p-m-/9-12-ds4muh5ca61b-vouchers/
ls /home/donald/workspace/meeting_files/2026-05-27-regular-meeting-6-30-p-m-/9-07-du7rdn6d8261-vouchers/
```

### A4 — the four searches for the missing fixture months

```bash
# by directory name
find /home/donald -maxdepth 9 -type d \
  \( -name '2026-06*' -o -name '2026-07*' -o -name '2026-08*' \)          # empty

# by file name, every date notation the district uses
find /home/donald -type f \
  \( -iname '*06-24-26*' -o -iname '*6-24-26*'  -o -iname '*06.24.26*' \
  -o -iname '*07-22-26*' -o -iname '*7-22-26*'  -o -iname '*07.22.26*' \
  -o -iname '*08-26-26*' -o -iname '*8-26-26*'  -o -iname '*08.26.26*' \
  -o -iname '*062426*'   -o -iname '*072226*'   -o -iname '*082626*' \)   # empty

# any voucher PDF modified after 2026-04-01
find /home/donald -type f -iname '*.pdf' -newermt '2026-04-01' \
  | xargs -r -I{} basename {} | grep -iE 'voucher|warrant|bdmtg|board mtg'  # empty

# the two zip archives on the machine, listed not extracted
.venv/bin/python -c "import zipfile,sys; print(zipfile.ZipFile(sys.argv[1]).namelist())" \
  /home/donald/files.zip                                                   # no hits
```

### A5 — Phase 0 recon (the numbered sections above)

```bash
# inventory only, no PDF extraction
.venv/bin/python recon_phase0.py --no-probe

# full pass: opens all 693 resolvable voucher PDFs, ~25 min, cached by SHA-256
.venv/bin/python recon_phase0.py --progress --json-out _build/recon_findings.json \
  > _build/recon_stdout.json 2> _build/recon_progress.log
```

Findings JSON: `facts/vouchers/_build/recon_findings.json` (gitignored — it is
regenerable, and it contains verbatim vendor and description text).

### A6 — the fixture verifications quoted above

```bash
V="/home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD/\
2026-03-25-regular-meeting-6-30-p-m-/9-12-ds4muh5ca61b-vouchers"

# TOTAL lines in every 2026-03-25 PDF
.venv/bin/python - "$V" <<'PY'
import sys, os, re
from locators import PdfText
TOTAL = re.compile(r'^\s*TOTAL\b.*$', re.I | re.M)
for name in sorted(os.listdir(sys.argv[1])):
    if not name.lower().endswith('.pdf'): continue
    t = PdfText(os.path.join(sys.argv[1], name))
    for m in TOTAL.finditer(t.text):
        print(name, 'p%d' % t.page_for_offset(m.start()),
              re.sub(r'\s+', ' ', m.group(0)).strip())
PY

# brief regex vs fixed regex over all four sets (the $1,322.35 result)
# and the register-to-detail arithmetic (the ACH / Capital exact ties)
# are reproduced by the same pattern against ROW_RX_BRIEF and ROW_RX
# in facts/vouchers/recon_phase0.py; both are pinned by tests:
.venv/bin/python -m pytest test_vouchers.py -q -k "BriefRegexVersusFixed or TotalLine"
```

---

## Appendix B — artifacts excluded, and why

44 of the 859 distinct files matched the voucher vocabulary but are not voucher
artifacts. 31 were excluded by a named rule:

| Reason | n | Example |
|---|---:|---|
| `meeting_attachment` | 13 | `Attachment for 20240208 Board Meeting - NS Equipment 2.pdf` |
| `policy_or_contract` | 7 | files matching `Employment` / `Contract` / `Capital Facilities Plan` |
| `meeting_minutes` | 6 | `Board Meeting Minutes Executive Session 20220914.pdf` |
| `audit_report` | 2 | `2024 - Audit of Expenditures - Final.pdf` |
| `warranty_not_warrant` | 2 | `3b_KW_E_Parking_Lot_Statutory_Warranty_Deed.pdf` |
| `donation_listing` | 1 | `Donations for 05.11.2022 Board Meeting.pdf` |

The 216 board-meeting transcripts never reach the exclusion rules at all: the title
sweep matches `%board mtg%` and `%bdmtg%` but not `%board meeting%`, so
`Board Meeting Transcript - 2019-05-22` is never a candidate. The exclusion rule for
transcripts is retained as a guard in case the sweep is ever widened, and a test
pins it.

13 could not be classified with confidence and are **left out and listed**, per the
stop rule:

```
Board_Minutes_022818.pdf
Donation List for 12.12.18 Board Meeting.pdf
EXTENS~1.PDF
EXTENS~2.PDF
KPD Liasion Agreement 2023-2024.pdf
KPD School Resource Officer 2023-2024.pdf
Liaison Extension Letter June 2024_Final.pdf
N2Y, LLC (Unique Learning System).pdf
Procedures 5111P.1819.FINAL.pdf
REV- 200M25036 Kentwood Schools- Repairs.pdf
SIP Board Meeting 10.23.24.pdf
SRO Extension Letter June 2024_Final.pdf
Warrants.pdf                          <- 2010-10-27, 2 pages, no text layer
```

Of these, only `Warrants.pdf` (2010-10-27) might be a voucher artifact. It is a
scanned image with no text layer, so it would be unparseable even if it is one.

---

## Next step

Phase 1 does not start until you answer **D1**. **D2** through **D7** can be answered
in the same pass and I will carry them into the schema. Nothing in this report
depends on Phase 1 having run.
