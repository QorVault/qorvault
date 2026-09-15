# Session debrief — voucher fact tables, staged 2026 packets (STEP 1 only)

**Date:** 2026-09-15
**Worktree:** `~/workspace/projects/ksd-vouchers`
**Branch:** `claude/facts-vouchers` (not merged)
**Parent commit at session start:** `3d220b3`
**Report:** `reports/facts-vouchers-reconciliation-2026-09-15-addendum-staged-build.md`

## What was asked

Five steps: (1) rebuild against the staged 2026 PDFs and report all 28 fixtures;
(2) R1 — anchor amount columns by page x-position; (3) cross-cycle dedupe views;
(4) regenerate exports from deduped views; (5) sampling files.

## What was done

**Step 1 only.** The task's stop rule fired:

> *Any HARD failure among the newly unblocked fifteen: stop and report the delta
> and the line; do not proceed to step 2.*

Steps 2, 3, 4 and 5 were **not started**. No file under `facts/` was modified.
No test, view or export was written.

## Outcome

- All 16 staged PDFs verified byte-identical to `MANIFEST.md` by SHA-256.
- `build.py --reload` ran clean: **459 sets, 469,716 lines, 12,454 vendors,
  133 register cross-checks, 683 parse-log rows.** 13 sets carry
  `source = 'staged_pdf'`.
- Fixtures: **18 PASS · 6 FAIL · 0 BLOCKED · 23 advisory REPORT**, exit 1.
  Zero blocked — every one of the fifteen previously blocked fixtures ran.
- **No previously passing fixture regressed.**
- Guardrail counts unchanged: `facts.meeting` 1,646 / `facts.motion` 6,507 /
  `facts.vote` 19,633 / `facts.attendance` 3,515 / `documents` 20,166 /
  `chunks` 179,026.

### The six failures, all one root cause

| Fixture | Δ |
|---|---:|
| 2026-06-24 ACH | −16,109.00 (2 lines, check 9252601789) |
| 2026-06-24 ASB | −238.00 (1 line, check 418449) |
| 2026-06-24 GF | −1,084.00 (2 lines, checks 608288 / 608502) |
| 2026-06-24 GF components | +1,148,700.00 (the same 2 lines) |
| 2026-07-22 Capital | −123,831.77 (10 of 12 rows unparseable) |
| 2026-07-22 ACH | whole set — 0 of 1,541 rows parse |

All six are **R1**: the amount pattern cannot see where the amount column ends,
so a bare whole-dollar invoice amount followed by a digit-leading description
gets merged. Every failing cent is reconciled to a named line in the addendum,
§4 and §5.

June and August match the independent manual parse **line for line** on every
count. July's PDFs are rendered with no inter-column whitespace at all and are a
categorically harder case.

The signed registers for June and July are **scans with no text layer**, so
those two cycles have no independent register evidence on this machine.

## Deviations from the task

- Deliverables tied to steps 2–5 (tests, views, exports, sampling files, git
  regenerate commands for them) are not present. This is the stop rule, not an
  omission.
- Exports in `exports/` and samples in `facts/vouchers/samples/` are now
  **stale** — they predate the 13 new sets. Left stale deliberately: step 4
  requires the step 3 deduped views, and regenerating them from the current
  parser would publish the June and July figures the fixtures just rejected.
- The ASB −$581.50 register mismatch on 2026-03-25 was not touched and remains
  recorded as `REGISTER_MISMATCH`.

## Not safe to quote

Anything from **2026-06-24** (except Capital) or **2026-07-22**. 2026-08-26
reconciles to the cent on all five funds.

## Regenerate

```bash
cd ~/workspace/projects/ksd-vouchers/facts/vouchers

export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

cd ~/workspace/staging/vouchers-2026 && sha256sum */*.pdf   # compare to MANIFEST.md
cd ~/workspace/projects/ksd-vouchers/facts/vouchers

.venv/bin/python build.py --reload --progress   # ~22 min
.venv/bin/python fixtures.py                    # 18 PASS / 6 FAIL / 0 BLOCKED, exit 1

# Inspect one cycle without writing:
.venv/bin/python build.py --only-date 2026-07-22 --dry-run
```

## Next session starts at

Step 2 (R1). The addendum §6 records the measured column geometry that makes it
tractable, including the exact x-edges on the July ACH pages. Nothing was
adopted or prototyped into the tree.
