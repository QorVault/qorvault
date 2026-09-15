# Session debrief — voucher fact layer, R1 column assignment by geometry

**Date:** 2026-09-15
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers` (not merged; the operator merges)
**Parent commit at session start:** `512e837`
**Report:** `reports/vouchers-r1-2026-09-15.md`

## Outcome

R1 is fixed and the remaining steps of the original delegation are done.
**35 HARD fixtures pass, 0 fail, 0 blocked; 23 advisory checks all match
exactly; no set that reconciled to the cent at `512e837` stopped reconciling.**
2026-07-22 ACH went from 0 of 1,541 rows parsed to all 1,541, tying to its
printed total of 5,585,581.92 exactly.

## Decisions

1. **The printed header names the columns; the measured corridor places the
   boundary.** The task specified deriving boundaries from the header's
   x-extents. Phase 0 measured that this cannot work in this corpus: label
   alignment inside a column is not consistent between formats, and **193 of 458
   listings (42%) have at least one boundary outside the band between the two
   header labels it separates**. Every fixed rule — band midpoint, band left
   edge, band right edge — was measured falling inside real data on some format.
   So the header is used to identify and order the columns, and the boundary is
   the empty corridor measured between the two columns' own printed runs.
   Deviation from the letter of the instruction, taken deliberately, evidenced
   in report §0.2.

2. **Characters are assigned by midpoint, and the straddle test is applied to
   midpoints, not to glyph boxes.** Glyph boxes in this corpus overlap: on
   2026-03-25 ASB page 3 a truncated vendor name's final `N` and the check
   date's first `0` overlap by 1.43 pt, so *no* boundary point is outside both
   boxes. A box-straddle test would reject that row, which is 1,322.35 of a set
   that reconciles to the cent and is a passing HARD fixture — it would have
   tripped the task's own regression stop rule. The midpoint test fires 3 times
   in 482,395 rows.

3. **Exactly one reason code added: `COLUMN_AMBIGUOUS`**, used at row level, set
   level and in the parse log — one name, three places. A new `reason_detail`
   column carries which column failed. "Register unavailable" needed **no**
   addition: `REGISTER_NOT_FOUND` and `REGISTER_NO_TEXT` were already in
   `voucher_reconciliation.reason`'s closed vocabulary and had never been
   emitted.

4. **Unread rows are held, never dropped and never summed.** A row that fails
   column assignment is written to `facts.voucher_line` with its page, its text
   and its reason, and is excluded from every control total. This is what makes
   "zero silent drops" checkable.

5. **Parenthesised negatives were left unread on purpose.** 1,385 rows print
   `$ (1,430.20)`-style amounts, $2.06M absolute. The task's sign rule
   recognises only a leading or trailing minus. Reading parentheses as negative
   would be me deciding what the district's accounting means. Held, counted,
   reported, and raised for approval.

6. **Era B's two money columns kept their existing positional mapping.** The
   2008 header reads `INV. AMT.` then `TOTAL AMT.`, which arguably reverses the
   2017 wording. Mapped fifth→`check_amount`, sixth→`invoice_amount` exactly as
   the regex parser did, so no historical figure silently changed meaning.
   Raised for approval instead.

## What changed

**New files**

| File | What |
|---|---|
| `facts/vouchers/geometry.py` | the column engine: header matching, corridor measurement, character assignment |
| `facts/vouchers/recon_r1.py` | the header-edge survey and the corpus sweep |
| `facts/vouchers/report_r1.py` | before/after tables and the regression gate |
| `facts/vouchers/test_geometry.py` | 27 tests over synthetic pages built from measured coordinates |

**Modified**

| File | What |
|---|---|
| `facts/vouchers/parsers.py` | `parse_listing_geometric` is the parse path; `read_cells` type-checks each column independently; the regexes are retained only as a per-line cross-check |
| `facts/vouchers/locators.py` | `PdfText` carries page geometry alongside the layout text and derives the document grid |
| `facts/vouchers/build.py` | writes held rows, the regex verdict and the grid method; emits register-unavailable cross-checks; `--dry-run` prints a per-set summary |
| `facts/vouchers/schema.sql` | `voucher_line.reason_code` / `reason_detail` / `regex_verdict`; six parse-log columns; idempotent ALTERs for an existing database |
| `facts/vouchers/views.sql` | six new views (cross-cycle dedupe, register availability, unread lines) |
| `facts/vouchers/fixtures.py` | 5 line-level, 4 check-level and 2 no-silent-drop HARD fixtures |
| `facts/vouchers/export_cycle.py` | reads the deduped views; states register availability per set; Held column |
| `facts/vouchers/samples.py` | double sample for 2026-07-22; file names from the set id; held rows flagged |
| `facts/vouchers/test_parsers.py` | two tests for held-row arithmetic |

**Data:** 459 sets rebuilt. 469,716 → 482,395 lines (**no set lost a line**;
215 gained). 74 → 88 sets reconciling. 133 → 381 register cross-check rows.
Nothing outside `facts.voucher_*` was touched — `facts.meeting` 1,646,
`facts.motion` 6,507, `facts.vote` 19,633, `facts.attendance` 3,515, `documents`
20,166, `chunks` 179,026, all identical before and after.

## Findings

- **The "independent manual parse" compares row counts, not amounts.** It is
  operator-supplied and genuinely independent of this code — the fixtures were
  committed at 09:00:48 and the documents arrived at 10:50, an hour and fifty
  minutes later — but its author, input and method are recorded nowhere on this
  machine. It matched June while three June totals failed because it was never
  checking those amounts. Relabelled in the report's tier table.
- **Check 9252601789 prints 12 lines, not the 13 the task states**, and those 12
  sum to 40,817.50 exactly. Counted twice by two paths.
- **$3.3M of 2005 General Fund payments were printed and not in the table.** The
  Era A regex needed two spaces between vendor and amount, and a wide amount is
  the one that crowds a long vendor name — so it dropped precisely the large
  rows. `2005-05-25 GF` went from 453 rows / $93,093 to 594 rows / $3,403,044.
  Same mechanism as R1, one era earlier.
- **Four General Fund sets from 2008 and 2017 had zero rows** because the era
  detector labelled a vendor-first document Era B. Geometry reads the header
  instead of guessing the era: 1,867 + 686 + 676 + 676 rows recovered.
- **88 rows have a check number printed as `#########`** — the source
  spreadsheet overflowed its column before the PDF existed. Not recoverable from
  the document.
- **Real payments carry 1- and 3-digit check numbers**, including a $3.1M
  payroll warrant on 2017-03-08. The `\d{5,11}` rule is unchanged from the regex
  parser, so this is not a regression, but the rows are visible for the first
  time.
- **155 sets restate $1,959,290,346.64 already counted at an earlier meeting.**
  That is what the new dedupe views exist to keep out of a cross-cycle sum.
- **Sample files silently overwrote each other.** `2026-02-11` has two Permanent
  Fund sets; both wrote to `2026-02-11-Permanent.md`. Fixed; names now come from
  the set id.
- **`schema.sql` says nine funds; the table holds eight.**
- **The project `CLAUDE.md` names the database/role `qorvault`; on this host
  both are `boarddocs`.** `psql -U qorvault` fails outright.

## Open items

1. **Parenthesised negatives.** *Symptom:* 1,385 rows held, $2.06M absolute,
   printed `$ (1,430.20)`. *Tried:* nothing — deliberately. *Diagnosis:*
   accounting-style negatives; the task's sign rule names only the minus sign.
   *Next step:* operator decides whether parentheses mean negative here; if so,
   one change to `money()` and a rebuild. *Urgency:* medium — the money is held,
   not wrong, but 26 residual out-of-balance sets probably contain some of it.

2. **Personal names in git.** *Symptom:* 277 distinct "Surname, Given" payee
   names appear in the 28 committed sample files. *Tried:* nothing — this
   predates the session (15 were already committed at `512e837`) and changing
   the privacy posture is not mine to decide. *Diagnosis:* the export layer
   withholds payees that are not positively identifiable as organizations; the
   sample files deliberately do not, because they are the audit trail — but they
   are committed to a source-available repository. *Next step:* keep samples out
   of git, redact the vendor column in the committed copy, or record it as a
   deliberate decision. *Urgency:* **highest item here.**

3. **26 sets still out of balance**, down from 38. *Symptom:* listed by year and
   fund in the report. *Tried:* nothing beyond what geometry fixed — new rules
   for the legacy sets were out of scope. *Diagnosis:* mixed; at least
   `2026-01-14 Custodial` (+10,180.00, exactly double) is a listing that prints
   one row twice, and `2025-08-13 Capital` is a stated total that does not match
   its own rows. *Next step:* one session per cause, largest delta first.
   *Urgency:* low — every one is flagged and excluded from export totals.

4. **Check-number floor of 5 digits.** *Symptom:* real payments with 1- and
   3-digit numbers are held. *Diagnosis:* `\d{5,11}`, inherited unchanged from
   the regex parser. *Next step:* decide whether to lower it or make it per-era.
   *Urgency:* low.

5. **Two by-vendor summaries are classified as `detail_listing`.**
   *Symptom:* `2020-09-09 GF#2` gets no grid; `2022-11-09 ASB` needs the
   header-band fallback. *Diagnosis:* `census.py` never assigns the
   `voucher_by_vendor` doc class, though the vocabulary has it. *Next step:* one
   classifier rule. *Urgency:* low — both produce correct output today.

6. **Era B money-column order.** As in Decisions 6. *Urgency:* low — all Era B
   sets are `TOTAL_NOT_FOUND`, so nothing reconciles against it either way.

7. **Five empty Transportation listings hold one row each.** *Symptom:* the
   printed `Total as of 2/10/22 $ -` line is read as a data-row candidate and
   flagged. *Tried:* using the vendor label's x0 as the fallback data margin —
   **rejected after measuring**: in the fragmented-amount format the vendor
   label sits 45 pt from the data margin, and the change would have dropped
   20,032 real lines from `2022-05-11 GF`. *Diagnosis:* the fallback margin is
   the modal row start, and on a page with no data rows that is the total line.
   *Next step:* a furniture rule keyed on the row's leading word, if it is worth
   it. *Urgency:* very low — the sets are flagged either way and no voucher row
   is involved.

8. **OCR of the June and July signed registers.** Out of scope as instructed.
   Both are scans with no text layer, so those two cycles still have no
   independent register evidence. Every affected set now carries
   `REGISTER_NO_TEXT` and both exports say so on every line. *Urgency:* medium —
   it is the only missing cross-check on two recent voucher nights.

## Documentation impact

- `reports/vouchers-r1-2026-09-15.md` is new and is the record of this work.
- The previous addendum,
  `reports/facts-vouchers-reconciliation-2026-09-15-addendum-staged-build.md`,
  is now **superseded in three places** and should be read with this report
  beside it: its §6 design note proposes deriving boundaries from the header
  x-extents (measured not to work, report §0.2); its §4 says check 9252601789
  has "twelve other rows" (it has ten others, twelve in total); and its
  statement that June and August "match the independent manual parse line for
  line" is true of counts only.
- `docs/session-logs/session-debrief-2026-09-15-facts-vouchers-staged-build.md`
  says "Next session starts at step 2 (R1)". Done.
- `facts/vouchers/CLAUDE.md` does not exist and was not created; per the global
  rule no `CLAUDE.md` was modified.
- The project `CLAUDE.md`'s Postgres credentials are stale (see Findings). **Not
  edited** — `CLAUDE.md` files are never modified by me. Flagged for the operator.

## System state summary

- **Git:** branch `claude/facts-vouchers`, one new commit on top of `512e837`.
  Nothing merged, nothing pushed, nothing rebased. `main` untouched.
- **Database:** `boarddocs` on 127.0.0.1:5432. `facts.voucher_set` 459,
  `facts.voucher_line` 482,395, `facts.vendor` 13,291,
  `facts.voucher_reconciliation` 381, `facts.voucher_parse_log` 683. Schema and
  views applied. Everything outside `facts.voucher_*` is byte-for-byte as it was.
- **Files on disk:** 10 export files and 28 sample files regenerated; 14 new
  sample files added.
- **Safe to quote:** every 2026 cycle. All 22 sets across 2026-03-25, -05-27,
  -06-24, -07-22 and -08-26 reconcile to their printed totals to the cent.
  2026-06-24 and 2026-07-22 have **no independent register cross-check** — the
  signed registers are scans — and every export states that on every line.
- **Not safe to quote:** the 26 residual out-of-balance sets, all flagged and
  all listed in the report.

## Regenerate every output

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers
export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

# Verify the staged input first
( cd ~/workspace/staging/vouchers-2026 && sha256sum */*.pdf )   # compare to MANIFEST.md

# Schema and views (idempotent)
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < schema.sql
podman exec -i -e PGPASSWORD="$PGPASSWORD" boarddocs-postgres \
  psql -U boarddocs -d boarddocs -v ON_ERROR_STOP=1 < views.sql

# Snapshot the current set table BEFORE rebuilding, if you want a
# before/after comparison. build.py --reload writes in one transaction at
# the very end, so the old state is readable until it commits.
.venv/bin/python - <<'PY'
import csv, db
rows = db.query_dicts("SELECT set_id, meeting_date::text AS meeting_date, fund, "
                      "format_era, source, stated_total, parsed_total, sum_check_dedup, "
                      "line_count, check_count, reconciled, delta, reason_code, notes "
                      "FROM facts.voucher_set ORDER BY meeting_date, fund", None)
with open("_build/before.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
PY

.venv/bin/python build.py --reload --progress     # ~40 min
#   -> sets=459 lines=482395 vendors=13291 reconciliations=381 parse_log=683

.venv/bin/python fixtures.py                      # PASS 35 / FAIL 0 / BLOCKED 0, exit 0
.venv/bin/python -m pytest -q                     # 214 passed
.venv/bin/python report_r1.py --before _build/before.csv --table   # exits 1 on any regression

for d in 2026-03-25 2026-05-27 2026-06-24 2026-07-22 2026-08-26; do
  .venv/bin/python export_cycle.py $d
done
.venv/bin/python samples.py                       # 28 files, seed 20260914

# Phase 0 survey, if it needs re-doing
.venv/bin/python recon_r1.py --corpus --out _build/r1_sweep.json   # first 6 pages per listing
.venv/bin/python recon_r1.py --survey "<pdf path>"

# One cycle, writing nothing
.venv/bin/python build.py --only-date 2026-07-22 --dry-run
```

## Next session starts at

The operator's call on Open Items 1 and 2 — parenthesised negatives, and whether
payee names belong in git. Both are decisions, not work.
