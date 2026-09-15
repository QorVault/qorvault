# Addendum — staged 2026 packets loaded, fifteen fixtures unblocked

**Date:** 2026-09-15
**Addendum to:** `reports/facts-vouchers-reconciliation-2026-09-15.md`
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers`
**Scope of this addendum:** STEP 1 only. **Steps 2–5 were not started.**

**No LLM touched any amount, date, vendor, check number or total.** Every figure
below is `pdfplumber` text extraction, a regular expression, and exact `Decimal`
arithmetic.

---

## Why this stops here

The task's stop rule: *"Any HARD failure among the newly unblocked fifteen: stop
and report the delta and the line; do not proceed to step 2."*

**Six HARD fixtures fail.** All six are the **R1 column-collision defect** —
the same defect step 2 exists to fix — reaching lines the old corpus never
exercised. Every failing cent is accounted for below, to the line.

No previously passing fixture regressed. The nine 2026-03-25 and 2026-05-27
fixtures, the 2023-08-23 pre-2024 fixture, the Transportation cumulative
fixture, and both contract fixtures all still pass.

---

## 1. Input verification

All 16 staged files present and byte-identical to `MANIFEST.md`:

```
$ cd ~/workspace/staging/vouchers-2026 && sha256sum */*.pdf
16 files, all 16 SHA-256 digests match the manifest exactly.
```

June and July carry five files each (GF, ACH, Capital, ASB + signed register);
August carries six (adds Trust). That matches the manifest's statement that no
Trust listing was published in June or July — it is an absence at source, not a
retrieval gap.

---

## 2. The build

```
$ .venv/bin/python build.py --reload --progress
sets=459 lines=469716 vendors=12454 reconciliations=133 parse_log=683
loaded
```

| | Before (446-set run) | After |
|---|---:|---:|
| Voucher sets | 446 | **459** (+13) |
| Voucher lines | 464,253 | **469,716** (+5,463) |
| Distinct vendors | 12,270 | **12,454** |
| Register cross-checks | 128 | **133** |
| Parse-log rows | 667 | **683** (+16) |
| Sets sourced `staged_pdf` | 0 | **13** |

Reconciliation state across all 459 sets:

| State | Sets |
|---|---:|
| `reconciled = true` | **74** (was 68) |
| `reconciled = false` | **54** (was 47) |
| `reconciled = NULL` (`TOTAL_NOT_FOUND`) | 331 |

By reason code: `TOTAL_NOT_FOUND` 331, *(reconciled)* 73, `OUT_OF_BALANCE` **38**
(was 33), `REGEX_MISS` **15** (was 14), `MULTIPLE_TOTALS` **1** (new),
`DUPLICATE_SET` 1.

Of the 13 new sets, **6 reconcile** (2026-06-24 Capital; all five August sets)
and **7 are flagged**. No new set is silently accepted — the
`contract_no_silent_acceptance` fixture still passes at zero exceptions.

### Nothing outside schema `facts` was written

| Table | Count after build |
|---|---:|
| `facts.meeting` | 1,646 |
| `facts.motion` | 6,507 |
| `facts.vote` | 19,633 |
| `facts.attendance` | 3,515 |
| `documents` | 20,166 |
| `chunks` | 179,026 |

All unchanged. Corpus reads used a `READ ONLY` session; writes went only to the
five `facts.voucher_*` / `facts.vendor` tables.

---

## 3. All 28 HARD fixtures

**18 PASS · 6 FAIL · 0 BLOCKED · 23 advisory REPORT.** `fixtures.py` exits 1.

Every one of the fifteen previously BLOCKED fixtures now ran. A blocked fixture
is a check that never executed; these have now executed, and six of them say no.

| Fixture | Result | Expected | Actual | Δ |
|---|---|---:|---:|---:|
| hard_total 2026-03-25 ACH | PASS | 3,609,064.78 | 3,609,064.78 | 0.00 |
| hard_total 2026-03-25 ASB | PASS | 136,796.65 | 136,796.65 | 0.00 |
| hard_total 2026-03-25 Capital | PASS | 84,009.42 | 84,009.42 | 0.00 |
| hard_total 2026-03-25 GF | PASS | 5,609,073.26 | 5,609,073.26 | 0.00 |
| hard_total 2026-05-27 ACH | PASS | 9,394,987.52 | 9,394,987.52 | 0.00 |
| hard_total 2026-05-27 ASB | PASS | 181,039.41 | 181,039.41 | 0.00 |
| hard_total 2026-05-27 Capital | PASS | 1,047,509.82 | 1,047,509.82 | 0.00 |
| hard_total 2026-05-27 GF | PASS | 3,388,060.86 | 3,388,060.86 | 0.00 |
| hard_total 2026-05-27 Transportation | PASS | 173,922.13 | 173,922.13 | 0.00 |
| **hard_total 2026-06-24 ACH** | **FAIL** | 5,490,076.68 | 5,473,967.68 | **−16,109.00** |
| **hard_total 2026-06-24 ASB** | **FAIL** | 149,650.79 | 149,412.79 | **−238.00** |
| hard_total 2026-06-24 Capital | PASS | 932,245.73 | 932,245.73 | 0.00 |
| **hard_total 2026-06-24 GF** | **FAIL** | 2,026,444.30 | 2,025,360.30 | **−1,084.00** |
| **hard_total 2026-07-22 ACH** | **FAIL** | 5,585,581.92 | 0.00 | **whole set** |
| **hard_total 2026-07-22 Capital** | **FAIL** | 127,028.73 | 3,196.96 | **−123,831.77** |
| hard_total 2026-08-26 ACH | PASS | 15,001,072.89 | 15,001,072.89 | 0.00 |
| hard_total 2026-08-26 ASB | PASS | 14,330.26 | 14,330.26 | 0.00 |
| hard_total 2026-08-26 Capital | PASS | 403,772.77 | 403,772.77 | 0.00 |
| hard_total 2026-08-26 GF | PASS | 2,474,622.84 | 2,474,622.84 | 0.00 |
| hard_pre_2024 2023-08-23 Capital | PASS | 2,120,786.72 | 2,120,786.72 | 0.00 |
| **hard_components 2026-06-24 GF** | **FAIL** | warrants 1,800,211.05 | 2,948,911.05 | **+1,148,700.00** |
| hard_cumulative Transportation | PASS | 783,396.00 + 391,698.00 = 1,175,094.00 | both listings 1,175,094.00, 1 flagged | — |
| contract_no_silent_acceptance | PASS | 0 | 0 | — |
| contract_every_line_has_a_locator | PASS | 0 | 0 | — |

**In every one of the six failures the *stated* total is read correctly.** The
document's own printed TOTAL matches the operator's figure to the cent in all
six. What differs is the sum of the parsed lines. The defect is in reading the
detail rows, never in reading the total.

### Advisory checks — 20 of 23 match exactly

All 2026-03-25, 2026-06-24 and 2026-08-26 line and check counts match the prior
independent manual parse **to the row**:

| Check | Expected | Actual |
|---|---|---|
| 2026-06-24 GF | 242 checks / 1,129 lines | 242 / 1,129 |
| 2026-06-24 ACH | 342 checks / 1,596 lines | 342 / 1,596 |
| 2026-06-24 Capital | 11 checks / 13 lines | 11 / 13 |
| 2026-06-24 ASB | 75 checks / 206 lines | 75 / 206 |
| 2026-08-26 GF | 272 checks / 708 lines | 272 / 708 |
| 2026-08-26 ACH | 382 checks / 981 lines | 382 / 981 |
| 2026-08-26 Capital | 21 checks / 26 lines | 21 / 26 |
| 2026-08-26 ASB | 12 checks / 14 lines | 12 / 14 |
| 2026-06-24 Sunburst Workforce Advisors | 300,252.85 | 300,252.85 |
| 2026-08-26 Sunburst / Elevation / Blazerworks / CBPI / Positive Behavior | all five | **all match to the cent** |

Three deviations, all July, all the same cause as the July HARD failures:
2026-07-22 ACH 336 checks / 1,541 lines expected against **0 / 0** actual;
2026-07-22 Capital 9 / 12 expected against **2 / 2**; 2026-07-22 Sunburst
949,107.73 expected against **0**.

**June and August are line-for-line correct in count.** June loses money on
five individual lines, not on rows it failed to find. July is a different and
much larger failure — see §5.

---

## 4. The June failures, line by line

All three June deltas are **whole dollars**, and all three are fully accounted
for. The signature is identical every time: **an invoice amount printed with no
decimal part, immediately followed by a description that begins with a digit.**
The amount pattern cannot see where the amount column ends, so it swallows the
first token of the description.

### 2026-06-24 ASB — −238.00, one line, page 2

```
Head Quarters Corp 6/11/2026 418449 355 240 2 Standard Portable Toilets for KM Athletics Use
```

| | Check amount | Invoice amount | Description |
|---|---:|---:|---|
| As printed | 355 | **240** | 2 Standard Portable Toilets… |
| As parsed | **355240.00** | **2.00** | Standard Portable Toilets… |

Loss on this line: 240 − 2 = **238.00**. That is the entire set delta.
Corroborated independently: check `418449` is the only check in the set carrying
two different `check_amount` values (`355.00` and `355240.00`).

### 2026-06-24 GF — −1,084.00, two lines

```
GRMEA-Enumclaw HS 5/21/2026 608288 760 700 2 Groups Choir Festival Registration - Kent Meridian HS   (page 1, line 25)
UW Botanic Gardens 6/11/2026 608502 388 388 2nd-grade Field Trip admission fee: 05/28/26            (page 7, line 370)
```

| Line | Invoice as printed | Invoice as parsed | Check amount as parsed | Loss |
|---|---:|---:|---:|---:|
| 608288 | 700 | 2.00 | 760700.00 | 698.00 |
| 608502 | 388 | 2.00 | 388388.00 | 386.00 |
| | | | **Total** | **1,084.00** |

698 + 386 = **1,084.00**, exactly the set delta.

**These same two lines are the whole of the `hard_components` failure.** The
warrant-range sum comes out 1,148,700.00 high, and
(760,700 − 760) + (388,388 − 388) = **1,148,700.00**. One defect, two fixtures.

### 2026-06-24 ACH — −16,109.00, two lines, page 27

Both on check `9252601789`, Pacifica Law Group LLP:

```
Pacifica Law Group LLP 6/4/2026 9252601789 40,817.50 106 25-26 Legal Services
Pacifica Law Group LLP 6/4/2026 9252601789 40,817.50  53 25-26 Legal Services
```

| Line | Invoice as printed | Invoice as parsed | Loss |
|---|---:|---:|---:|
| first | 106 | **−10,625.00** | 10,731.00 |
| second | 53 | **−5,325.00** | 5,378.00 |
| | | **Total** | **16,109.00** |

Here the collision also **flips the sign**. The description begins `25-26`, and
this corpus writes some negatives with a *trailing* minus, so `106 25-` reads as
`−10625`. A legal-services invoice of $106 is recorded as a **credit of
$10,625**.

Independent confirmation from the document's own arithmetic: the other twelve
invoice rows against check `9252601789` sum to 40,658.50, and the check amount
printed on every one of those rows is 40,817.50. The difference is
**159.00 = 106 + 53** — exactly the two printed amounts, and nothing else fits.

### 2026-06-24 GF also carries `MULTIPLE_TOTALS`

That listing prints a second TOTAL of `4,052,888.60`, which is exactly twice
`2,026,444.30`. The first-TOTAL-after-the-last-row rule picks the right one; the
extra is recorded, and the set is flagged `MULTIPLE_TOTALS` rather than
`OUT_OF_BALANCE` because both conditions hold.

---

## 5. The July packet is a different and worse layout

July is not "a few bad lines." **The entire 2026-07-22 packet is rendered with
no whitespace between columns at all**, and `pdfplumber`'s layout mode cannot
recover it. Raw page-1 text of `ACH Vouchers 07-22-26.pdf`:

```
       Vendor                Check Date Check Number Amount Invoice Amount Description
       AIRGAS USA LLC         6/18/20269252601881 3.15 3.152025-26 Helium tank rental and refills
       ALL HANDS CMTY INTERP SVCS 6/18/20269252601882 20,594.57 1,517.352025-2026 ASL Services for Employee ADA...
```

The check date runs into the check number (`6/18/20269252601881`) and the
invoice amount runs into the description (`3.152025-26`). No regex over this
text can be correct, and the parser correctly declines to guess: **0 of 1,541
ACH rows parse**, and the set is flagged `REGEX_MISS` rather than reported at a
wrong number.

`Capital Vouchers 07-22-26.pdf` shows both layouts on the same page —
`7/9/2026208930 4659.93 1605KL Field Renovation` fuses on both sides, while
`Ashurst Perkins Coie US LLP 6/18/2026208926 863.50 863.50Bid Bond Clarification`
fuses only between date and check number. 2 of 12 rows parse.

July ASB (no HARD fixture, but flagged) shows the sign-flip failure four times
over, all from descriptions beginning `2026-27`:

```
Elite Performance Dance Camps 6/25/2026 418468 14100 7375 2026-27 KR Dance Elite Performance Dance Camp registration
  → check 141,007,375.00   invoice −2,026.00   (printed: check 14,100, invoice 7,375)
Happy Feet Boots 7/9/2026 418488 596 596 2026-27 Dance uniform boots
  → check 596,596.00       invoice −2,026.00   (printed: check 596, invoice 596)
Weissman 7/9/2026 418492 1018 1018 2026-27 uniform order shoes, tights, etc
  → check 10,181,018.00    invoice −2,026.00   (printed: check 1,018, invoice 1,018)
```

### The signed registers for June and July are scans

`BDMTG - 6-24-2026 SIGNED.pdf` and `BDMTG - 7-22-2026 SIGNED.pdf` have **no text
layer** (2 pages each, zero extractable characters). They are logged
`no_text_layer` and produce no cross-check. `BDMTG Signed 08-26-26.pdf` does
carry text and produced **5 register cross-checks** for the August cycle.

So for June and July there is no independent register evidence on this machine —
the listings are the only machine-readable source, which makes getting the
listing parse right the whole of the evidence.

---

## 6. What this says about R1

R1 was ranked highest priority on the evidence of 33 out-of-balance sets. The
staged packets raise that to **38**, and — more to the point — they move the
defect from "old sets nobody quotes" to **the three most recent voucher nights
the board has approved**.

The task's prescription is right, and the source documents confirm it is
achievable. Character-level coordinates from `pdfplumber` are clean even where
the text stream is fused. On July ACH page 2, every data row places its
characters in the same columns:

| Column | x-range (points) |
|---|---|
| Vendor | 40.1 → ~110 |
| Check date | 202.6 → 226.2 |
| Check number | 228.6 → 256.2 |
| Check amount | right-aligned, ends **294.4** |
| Invoice amount | right-aligned, ends **336.3** |
| Description | begins **338.6** |

and the printed header row on page 1 lands on the same edges
(`Amount` ends 294.6, `Invoice Amount` ends 336.5, `Description` begins 338.6).
The boundary the regex cannot see is a **2.3-point gap** that is plainly there
in the coordinates.

That is a design note, not an implementation. **No parser change was made.**

---

## 7. State of the tree

- `facts.*` holds the 459-set build described above. It is the current truth.
- The **exports in `exports/` are now stale** — they were generated from the
  446-set build and do not include June, July or August. They were deliberately
  not regenerated, because step 4 requires the deduped views from step 3.
- The **sampling files in `facts/vouchers/samples/` are stale** for the same
  reason (seed 20260914, 15 files, 2026-03-25 and 2026-05-27 only).
- No source file under `facts/` was modified in this session.

**Until step 2 lands, no figure from 2026-06-24 or 2026-07-22 should be
quoted.** 2026-08-26 reconciles to the cent on all five funds and is safe;
2026-06-24 Capital reconciles and is safe.

---

## 8. Regenerating exactly this

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers

export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

# Verify the staged input first.
( cd ~/workspace/staging/vouchers-2026 && sha256sum -c <(
    grep -oE '`[^`]+\.pdf` \| `[0-9a-f]{64}`' MANIFEST.md ) ) 2>/dev/null || \
  ( cd ~/workspace/staging/vouchers-2026 && sha256sum */*.pdf )

.venv/bin/python build.py --reload --progress     # ~22 min, 683 artifacts
.venv/bin/python fixtures.py                      # 18 PASS / 6 FAIL / 0 BLOCKED, exit 1

# One cycle, no write:
.venv/bin/python build.py --only-date 2026-07-22 --dry-run
```
