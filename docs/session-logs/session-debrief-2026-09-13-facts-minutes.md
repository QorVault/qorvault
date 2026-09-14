# Session Debrief — Minutes Fact Tables (Phase 0 + Phase 1)

**Date:** 2026-09-13
**Worktree:** `~/workspace/projects/ksd-main`
**Branch:** `claude/feat-facts-minutes`
**Git refs:** `313ce1d` (Phase 0 recon), `2f5ab0f` (Phase 1 build), branched from `main` @ `0ba6cc0`
**Package:** `facts/minutes/`

---

## Decisions

**Votes are sourced from BoardDocs agenda items, not the minutes.** Phase 0
found that across all 874 minutes documents from 2004-2026, exactly two record
votes with director names and two more use a roll-call format. Every other
motion reads "Motion carried." with no mover, no second, no names, no tally.
Building `facts.vote` from minutes alone would have produced a table covering
four documents. The 2,543 agenda items carrying a `Motion & Voting` block hold
mover, second, disposition and a named yea/nay/abstain roll for 2018-2026, so
those became the vote source. Every motion and vote row is tagged with its
`source`, and the two are never blended. This was put to the operator as Phase 0
decision D1 and approved as option (b).

**`facts.meeting` is seeded from the scraped meeting directory census.** This
was not in the original design and proved necessary: the required
`meetings_missing_minutes` view is unanswerable if meetings only exist when
their minutes do. The `agenda` document type could not serve as the census —
it holds 3 rows for 2019 and 4 for 2020. The scraped directory names are
complete for 2005-2026 and encode date and meeting type.

**`documents.meeting_date` populates `approved_at_meeting_id`, never
`meeting_date`.** All 796 resolvable attachment offsets are positive, 88.6% of
them exact multiples of 7 days: minutes are attached to the agenda of the later
meeting that approves them. `meeting_date` comes from the minutes body.
Reversing these two would have misdated the entire fact layer by one meeting —
the single highest-consequence modelling decision in this build.

**`purpose_category` is populated only where the text cites an RCW subsection.**
Phase 0 found exactly one such citation in the whole corpus. The column is
therefore almost entirely NULL by design. Inferring a statutory category from
prose like "to discuss personnel evaluations" would be reading a legal
classification into a record that does not state one.

**An executive-session adjournment is an end time, not a session.** This
decision is why a hard fixture does not pass — see Open Items.

**No LLM in any date, name, vote, motion or count path.** Everything is regex
and arithmetic over extracted text. No `summary` column was populated.

## What changed

New package `facts/minutes/` (16 files) and schema `facts` (6 tables, 5 views):

| Table | Rows |
|---|---:|
| `facts.meeting` | 1,646 |
| `facts.attendance` | 3,515 |
| `facts.motion` | 6,507 (2,293 minutes + 4,214 agenda items) |
| `facts.vote` | 19,625 |
| `facts.executive_session` | 280 |
| `facts.minutes_parse_log` | 874 |

Views: `motions_by_date`, `votes_by_director`, `exec_sessions_by_year`,
`meetings_missing_minutes`, `votes_unnamed_by_year`.

Reports: `reports/facts-minutes-recon-2026-09-12.md` (Phase 0),
`reports/facts-minutes-run-2026-09-13.md` (Phase 1),
`reports/meeting-export-2026-02-04.md` (export of the most recent meeting).

35 unit tests, all passing. Full corpus run takes ~71 seconds with zero parse
failures.

**Nothing was written outside schema `facts`.** All reads against `documents`
and `chunks` used a `READ ONLY` Postgres session, so an accidental write would
have failed at the database rather than succeeding quietly. Qdrant and `rag_api`
were never touched. `ksd-boarddocs-rag` and production were never touched.

## Findings

**The premise that a `facts` package already existed was wrong.** No `facts`
package exists in any branch, no commit ever added one, and the database had no
`facts` schema. There was nothing to extend; the fact layer starts here. The
only voucher code on the machine is `ksd_forensic/scripts/analyze_vouchers.py`,
a different project using pandas/numpy on system python3.

**Postgres holds no page-level text at all.** `document_pages` is empty and
`chunks.source_page` is NULL for all 179,081 chunks. The locator rule could not
be satisfied from the database, which is why `pdfplumber` became a genuine
dependency. Page coverage is now 100% on motions, votes and executive sessions.

**Before 2018 the record contains no named votes whatsoever** — not one of the
783 motions recorded between 2004 and 2017 names how any director voted. This is
the headline civic finding and is now queryable via
`facts.votes_unnamed_by_year`.

**2023 and 2024 have real minutes gaps**: 33.3% and 38.9% of regular meetings
have no minutes document in the corpus, against 0% for both 2022 and 2025. I
verified the 2024 gap by hand rather than trusting the date parser. The data
cannot distinguish "never adopted" from "never posted" from "missed by the
scraper", and I have not drawn a conclusion about which.

**Five parser bugs were found and fixed**, each silently wrong beforehand:

1. Executive-session announcements that state a time were missed **entirely** —
   the pattern used `[^.\n]` between the verb and "executive session", which
   cannot cross the period inside "6:00 **p.m.**". Fixing it raised executive
   sessions from 170 to 280.
2. RCW citations were truncated to "RCW 42." by the same class of bug.
3. Named votes were double-counted where an agenda item prints a surname-only
   `ROLL CALL VOTE` roll before its resolution as well as the canonical roll
   after it — reporting 10 votes on a 5-member board.
4. Attendance was under-captured four ways ("Board President" vs "President",
   "calling the meeting to order" vs "presiding", "(via telephone)" glued into
   the name, "X was excused" outside both lists). Fixing these took attendance
   from 2,950 to 3,515 rows.
5. Child rows were emitted from duplicate documents describing one meeting,
   double-counting attendance and motions.

Bugs 1 and 2 were caught by unit tests, not by the corpus run — the run looked
plausible while silently missing most of its subject matter.

**`.env` is stale for Postgres.** Its password is 32 characters; the running
container's is 13. Credentials are injected at runtime from the container
config; `.env` was not edited.

## Open Items

### 1. "27 executive sessions in 2024" fixture does not pass (26)

- **Symptom:** Hard fixture expects 27; the build produces 26.
- **Tried:** Counted three ways — 24 executive sessions convened as their own
  meeting (census, after excluding 2 cancellation notices), plus 2 announced
  inside other meetings' minutes.
- **Diagnosis:** The 2024-09-11 minutes contain three executive-session
  sentences: an announcement at 10:09 p.m., an extension at 10:12 p.m., and
  "The Executive Session was adjourned at 10:25 p.m." I model the third as the
  **end time of the session already open** (`actual_end_time`). Counting it as a
  separate session yields exactly 27. The fixture is reproducible only under a
  definition that counts adjournment sentences as sessions, which would
  double-count every session whose end time happens to be recorded.
- **Next step:** Operator confirms which definition the 2026 evaluation used. I
  will either accept 26 as correct or change the model deliberately. **I did not
  loosen the fixture to reach 27.**
- **Urgency:** Medium — blocks a hard fixture, but the discrepancy is understood
  and is one row.

### 2. `tally_within_attendance` fixture does not pass (117 of 2,613)

- **Symptom:** Votes cast exceed directors recorded present.
- **Tried:** Fixed four attendance extraction bugs, reducing violations from 490
  to 117.
- **Diagnosis:** All 117 fall in **6 meetings**. One (2023-12-13) is a board
  transition where the minutes record the outgoing board's roll and the votes
  were cast by the newly seated board — both records are accurate. The rest are
  minutes that genuinely list fewer directors present than later voted.
- **Next step:** Recommend keeping the fixture and treating the residual as a
  data-quality finding; the list of 6 meetings is the useful output, not a green
  check. Operator decides.
- **Urgency:** Low — now surfacing real record discrepancies, not parser defects.

### 3. `disposition_and_locator` fixture does not pass (43 of 6,507)

- **Symptom:** 43 locator quotes do not contain a word matching the recorded
  disposition. `missing_locator` is 0 — every motion has a disposition and a
  locator.
- **Tried:** Made the quote bridge the motion opening to the disposition
  sentence, since consent-agenda motions run to thousands of characters and a
  fixed-length excerpt truncated before "Motion carried". Reduced 362 → 43.
- **Diagnosis:** Residual cases record e.g. `withdrawn` where the text says the
  mover "withdrew" it; the quote is correct, the word-match is not.
- **Next step:** Widen the disposition word list, or accept as a known residual.
- **Urgency:** Low.

### 4. Operator hand counts never supplied — fixture BLOCKED, not passed

- **Symptom:** The hard fixture requiring six operator-chosen meetings with hand
  counts could not run.
- **Diagnosis:** Requested in the Phase 0 report (D3); not supplied. Reported as
  blocked rather than passing, because a check that never executed is not a
  check that succeeded. Phase 0 found **two** eras, not three, so three meetings
  per era is the sensible split.
- **Next step:** Operator supplies six meetings and hand counts.
- **Urgency:** Medium — this is the only independent check on parser accuracy.

### 5. Research documents were never available

- `research-04-document-identity.md`, `research-02 §2.6`, ADR 0007 and
  `session-debrief-2026-09-08-retrieval-smoke-test.md` are not in the repo and I
  have no access to project knowledge. Phase 0 did not depend on them, but ADR
  0007's fact-layer schema would have informed table naming.
- **Urgency:** Low, but table names may need revisiting if ADR 0007 differs.

### 6. I committed a report I did not write

`git add -A reports/` swept in `reports/boarddocs-coverage-2026-09-12.md`, an
untracked report from a previous session's coverage audit. It is a legitimate
project artifact but was not mine to commit. Say the word and I will remove it
from the branch with `git rm --cached`.

### 7. `pre_amendment_text` is NULL throughout

The column exists as specified, but amendments to a motion before its vote are
not reliably distinguishable in this corpus. No row populates it.

## Documentation impact

- **`CLAUDE.md` documents the database as `qorvault/qorvault`; it is
  `boarddocs/boarddocs`.** The documented connection string fails outright.
  Flagged only — I do not modify CLAUDE.md files.
- `documents.file_path` is stale corpus-wide (rooted at a path that no longer
  exists). This package works around it in `locators.resolve_pdf_path`; anything
  else reading that column is silently broken.
- The pre-2019 corpus exists **only** inside a backup directory. Fourteen years
  of records with a backup as their sole copy is a durability risk.
- 5 meetings are scraped twice under differently punctuated slugs, all in 2026.

## System state summary

- Schema `facts` created and populated in database `boarddocs`. Nothing outside
  it was modified. No rows were deleted anywhere.
- Branch `claude/feat-facts-minutes`, two commits ahead of `main` @ `0ba6cc0`.
  Not pushed. Pre-commit hooks ran and passed on both commits (ruff, bandit,
  secret detection, interrogate); none were bypassed.
- Two hooks escalated during the session and were respected, not worked around:
  the privilege-escalation guard blocked installing `poppler-utils` via `sudo`
  (hence `pdfplumber` instead of `pdftotext`), and the dependency guard required
  operator approval for the venv. A third guard correctly blocked a command of
  mine that would have printed the live Postgres password to the transcript.
- Virtualenv at `facts/minutes/.venv`, gitignored. `pip-audit` against the lock
  file reports no known vulnerabilities.

## Regenerating the export

```bash
cd ~/workspace/projects/ksd-main/facts/minutes

export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

.venv/bin/python export_meeting.py 2026-02-04 \
  -o ../../reports/meeting-export-2026-02-04.md
```

Full rebuild from scratch:

```bash
cd ~/workspace/projects/ksd-main/facts/minutes
export PGPASSWORD=...   # as above

podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs \
  -v ON_ERROR_STOP=1 < schema.sql
podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs \
  -v ON_ERROR_STOP=1 < views.sql

.venv/bin/python build.py --reload          # ~71s
.venv/bin/python -m pytest test_parsers.py -q
.venv/bin/python fixtures.py
```
