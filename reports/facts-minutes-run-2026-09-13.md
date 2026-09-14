# Phase 1 Run — Minutes Fact Tables

**Date:** 2026-09-13
**Worktree:** `~/workspace/projects/ksd-main`
**Branch:** `claude/feat-facts-minutes` (from `main` @ `0ba6cc0`)
**Database:** `boarddocs` on `boarddocs-postgres`, schema `facts`
**Package:** `facts/minutes/`

**Status: built and loaded. Three hard fixtures do not pass and one could not
run.** None were loosened. Each is diagnosed below.

---

## 1. What was built

Six tables in schema `facts`, plus five views and an export script.

| Table | Rows | Source |
|---|---:|---|
| `facts.meeting` | 1,646 | meeting census + minutes + agenda items |
| `facts.attendance` | 3,515 | minutes |
| `facts.motion` | 6,507 | 2,293 minutes + 4,214 agenda items |
| `facts.vote` | 19,625 | agenda items only |
| `facts.executive_session` | 280 | minutes |
| `facts.minutes_parse_log` | 874 | one row per minutes document |

Full run over all 874 minutes documents and all 2,543 agenda items carrying a
`Motion & Voting` block takes **~71 seconds**.

### Decisions taken, and why

**Votes come from agenda items (Phase 0 decision D1, option b).** Phase 0
established that the minutes record how individuals voted in only 4 of 874
documents. Building `vote` from minutes alone would have produced a table with
essentially nothing in it. The 2,543 BoardDocs agenda items carry mover,
second, disposition and a named yea/nay/abstain roll for 2018-2026, so
`facts.vote` is sourced from them and every row is tagged `source='agenda_item'`.
Motions therefore have two sources, also tagged, and the two are never mixed
silently.

**`facts.meeting` is seeded from the scraped meeting census, not from minutes.**
This was not in the original plan and turned out to be required: a meeting whose
minutes are missing would otherwise not exist as a row, and the mandated
`meetings_missing_minutes` view would have had nothing to report. The census
supplies the denominator (1,395 real meetings, cancellations and ceremonial
appearances excluded). Rows are upgraded in place when minutes are found.

**`documents.meeting_date` populates `approved_at_meeting_id`, never
`meeting_date`.** Phase 0 measured all 796 resolvable attachment offsets as
positive (88.6% exact multiples of 7 days): minutes are attached to the agenda
of the later meeting that approves them. `meeting_date` comes from the minutes
body. Reversing these would have misdated the entire fact layer by one meeting.

**`purpose_category` is populated only when the text cites an RCW subsection.**
Phase 0 found exactly one such citation in the whole corpus, so this column is
almost entirely NULL by design. The purpose is never inferred from prose — that
would be reading a legal classification into a record that does not state one.

**No LLM touches any date, name, vote, motion or count.** Every value is regex
and arithmetic over extracted text. No `summary` column was populated.

## 2. Run results by year

"Docs" counts minutes documents that parsed; "sup" counts documents superseded
because another document covers the same meeting with more text.

| Year | Docs | Sup | Motions | Exec | Attendance | Year | Docs | Sup | Motions | Exec | Attendance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2004 | 1 | 0 | 2 | 1 | 5 | 2016 | 45 | 1 | 127 | 14 | 201 |
| 2005 | 25 | 1 | 20 | 10 | 102 | 2017 | 46 | 5 | 79 | 11 | 193 |
| 2006 | 29 | 0 | 26 | 21 | 127 | 2018 | 31 | 8 | 85 | 9 | 145 |
| 2007 | 25 | 1 | 34 | 17 | 106 | 2019 | 39 | 19 | 177 | 10 | 191 |
| 2008 | 38 | 3 | 24 | 16 | 144 | 2020 | 48 | 14 | 210 | 15 | 194 |
| 2009 | 29 | 1 | 38 | 14 | 125 | 2021 | 51 | 10 | 242 | 14 | 240 |
| 2010 | 46 | 4 | 48 | 15 | 218 | 2022 | 61 | 1 | 187 | 0 | 257 |
| 2011 | 30 | 2 | 121 | 23 | 142 | 2023 | 33 | 1 | 97 | 1 | 134 |
| 2012 | 31 | 0 | 127 | 19 | 131 | 2024 | 32 | 0 | 112 | 2 | 151 |
| 2013 | 37 | 1 | 139 | 20 | 171 | 2025 | 46 | 5 | 150 | 7 | 221 |
| 2014 | 31 | 0 | 89 | 22 | 135 | 2026 | 6 | 0 | 28 | 0 | 30 |
| 2015 | 35 | 2 | 131 | 19 | 152 | | | | | | |

### Failure categories

**There were no parse failures.** All 874 documents resolved to one of two
outcomes:

| Status | Count |
|---|---:|
| `parsed` | 795 |
| `superseded_duplicate` | 79 |
| `no_text_layer` | 0 |
| `date_unresolved` | 0 |
| `era_unmatched` | 0 |
| `motion_unparsed` | 0 |
| `locator_unresolved` | 0 |

`superseded_duplicate` is not a failure: it means two or more documents describe
the same meeting (a re-scrape, or a combined PDF re-posted) and the longest text
was used. Emitting rows from all of them would double-count attendance and
motions. The superseded documents are recorded in `facts.minutes_parse_log`
rather than silently dropped.

A meeting with zero motions is valid, not a failure — 431 of the minutes
documents are work, study or executive session records that contain no motions
at all.

### Locator coverage

| Table | Rows with a page number |
|---|---|
| `facts.motion` | 6,507 / 6,507 (100%) |
| `facts.vote` | 19,625 / 19,625 (100%) |
| `facts.executive_session` | 280 / 280 (100%) |
| `facts.attendance` | 3,506 / 3,515 (99.7%) |
| `facts.meeting` | 793 / 1,646 (48.2%) |

Page numbers come from `pdfplumber` reading the source PDFs (793 documents) —
Postgres holds no page data at all, as Phase 0 found. BoardDocs agenda items are
single scraped web pages, so page 1 is their true page number rather than an
assumption. The 48.2% figure for `facts.meeting` is correct behaviour: the other
853 rows are census meetings with no minutes document, so there is no page for
them to cite.

Every fact row carries `locator_document_id`, `locator_page`,
`locator_char_offset` and `locator_quote`. No chunk id is used as a locator
anywhere.

## 3. Named votes — the transparency measure

`facts.votes_unnamed_by_year`. Before 2018 the record contains **no named votes
at all**: not a single one of the 783 motions recorded between 2004 and 2017
names how any director voted.

| Year | Motions | Named | % unnamed | From minutes | From agenda items |
|---:|---:|---:|---:|---:|---:|
| 2004-2017 | 783 | **0** | **100%** | 783 | 0 |
| 2018 | 215 | 130 | 39.5% | 85 | 130 |
| 2019 | 590 | 413 | 30.0% | 177 | 413 |
| 2020 | 595 | 385 | 35.3% | 210 | 385 |
| 2021 | 601 | 359 | 40.3% | 242 | 359 |
| 2022 | 691 | 504 | 27.1% | 187 | 504 |
| 2023 | 1,071 | 974 | 9.1% | 97 | 974 |
| 2024 | 740 | 628 | 15.1% | 112 | 628 |
| 2025 | 874 | 723 | 17.3% | 150 | 724 |
| 2026 | 125 | 97 | 22.4% | 28 | 97 |

The residual unnamed share after 2018 is the minutes-sourced motions, which
never name voters. Where a motion appears in both sources it is recorded twice,
once per source — deliberately, so that what each record *says* stays visible
rather than being merged into an unattributable blend.

## 4. Fixture results

### HARD — `exec_sessions_2024`: **FAIL** (expected 27, got 26)

Not loosened. I can state precisely where the missing one is.

| Definition | 2024 count |
|---|---:|
| Executive sessions convened as their own meeting (census) | 24 |
| Executive sessions announced inside another meeting's minutes | 2 |
| **Total as built** | **26** |
| Closing sentences treated as separate sessions | +1 |
| **Total if closings counted** | **27** |

The 2024-09-11 minutes contain three executive-session sentences: an
announcement at 10:09 p.m., an extension at 10:12 p.m., and *"The Executive
Session was adjourned at 10:25 p.m."* I model the third as the **end time of the
session already open** and store it in `actual_end_time`, because it is not a
new session. Counting it as one yields exactly 27.

So the fixture is reproducible only under a definition that counts adjournment
sentences as sessions. **I did not adopt that definition, because it is wrong**:
it would double-count every executive session whose end time happens to be
recorded. This is the open question D2 from Phase 0, still unanswered. Please
confirm which definition the 2026 evaluation used, and I will either accept 26
as correct or change the model deliberately.

Corpus-wide the correction matters: 26 dirs in 2024 mention an executive
session, 2 of which are cancellation notices, which is why the census gives 24.

### HARD — `tally_within_attendance`: **FAIL** (117 of 2,613 checked)

All 117 violations fall in **6 meetings**. Three distinct causes, none of which
is the vote data being wrong:

| Meeting | Attendance recorded | Directors who voted | Cause |
|---|---|---|---|
| 2023-12-13 regular | Bento, Margel, Hamada, Farah, Clark | Farah, Song, Margel, Clark, Cook | **Board transition.** The December 2023 meeting seated newly elected directors. The minutes record the outgoing board's roll; the votes are cast by the incoming board. Both records are accurate. |
| 2022-10-05 special | 4 names | 5 names | Attendance list in the minutes is short by one |
| 2022-06-29 special | 1 name | 4 names | Minutes record only the presiding officer |
| 2024-07-10 special | 5 names | 4 names | Statuses include absent/excused, so "present" is 4 |
| 2025-02-11 special | 4 names | 4 names | Same |
| 2020-03-19 special | 4 names | 5 names | Attendance list short by one |

Parser work during this session reduced these from 490 to 117 by fixing four
real extraction bugs (below). The remainder are minutes that genuinely record
fewer directors present than later voted, plus one board-transition meeting
where both records are correct and simply describe different boards.

**I recommend keeping this fixture as-is and treating the residual as a
data-quality finding**, since it is now pinpointing real discrepancies in the
district's records rather than parser defects. A list of the 6 meetings is the
useful output, not a green check.

### HARD — `disposition_and_locator`: **FAIL** (43 of 6,507)

Every motion has a disposition and a locator — `missing_locator` is **0**. The
failure is narrower: in 43 cases the locator *quote* does not contain a word
matching the recorded disposition.

This started at 362 and was reduced to 43 by making the quote bridge the motion
opening to the disposition sentence (consent-agenda motions run to thousands of
characters, so a fixed-length excerpt truncated long before "Motion carried").
The residual 43 are motions whose disposition was inferred from surrounding
context — for example a motion marked `withdrawn` because the minutes say the
mover "withdrew" it, where the quote contains "withdrew" but the check looks for
"withdrawn". These are correct rows with an imperfect quote match.

### HARD — `operator_hand_counts`: **BLOCKED, not passed**

The six operator-chosen meetings and their hand counts were requested in the
Phase 0 report (D3) and have not been supplied. This check did not run. I have
reported it as blocked rather than passing, because a check that never executed
is not a check that succeeded. Phase 0 found **two** eras, not three, so three
meetings per era is the sensible split.

## 5. Parser bugs found and fixed during this run

Each of these was silently wrong before the tests and fixtures caught it. All
five are now covered by unit tests.

1. **Executive-session announcements that state a time were being missed
   entirely.** The pattern used `[^.\n]` between the verb and "executive
   session", which cannot span the period inside "6:00 **p.m.**" A recess
   written as *"recessed the meeting at 6:00 p.m. for an executive session"*
   never matched. Fixing this raised executive sessions from 170 to 280
   (+65%).
2. **RCW citations were truncated.** The same sentence terminator broke on the
   periods inside "RCW 42.30.110(1)(i)", capturing only "RCW 42." and making
   `purpose_category` unfillable even when the text did cite a subsection.
3. **Named votes were double-counted.** Some agenda items print a surname-only
   `ROLL CALL VOTE` roll *before* their resolution as well as the canonical
   full-name roll after it. Scanning to the next resolution attributed the
   following motion's pre-roll to the current motion, reporting 10 votes on a
   5-member board.
4. **Attendance was under-captured four ways**: "Board President X presiding"
   (the optional "Board" broke the match), "X calling the meeting to order"
   instead of "presiding", "Debbie Straus (via telephone)" glued the qualifier
   into the name, and "Denise Daniels was excused" sat outside both the present
   and absent lists. Fixing these raised attendance from 2,950 to 3,515 rows and
   took meetings with a full five-member roll from 177 to 476.
5. **Child rows were emitted from duplicate documents.** Where two documents
   describe one meeting, both sets of attendance and motions were being written,
   double-counting. Now the longest document wins and the others are logged as
   `superseded_duplicate`.

## 6. Query surface

Five views, all exposing locator columns:

- `facts.motions_by_date` — motions in date order with vote coverage
- `facts.votes_by_director` — one row per director per vote
- `facts.exec_sessions_by_year` — scheduled meetings and announcements, counted
  separately and summed
- `facts.meetings_missing_minutes` — 853 meetings with no minutes document
- `facts.votes_unnamed_by_year` — the transparency measure in section 3

Export script, verified against the most recent meeting with minutes:

```bash
cd ~/workspace/projects/ksd-main/facts/minutes
export PGPASSWORD=...
.venv/bin/python export_meeting.py 2026-02-04 -o ../../reports/meeting-export-2026-02-04.md
```

Output is at `reports/meeting-export-2026-02-04.md`.

## 7. Tests

35 unit tests, all passing, run against fixed text samples rather than the
database so they pin parser behaviour independently of corpus state:

```
cd ~/workspace/projects/ksd-main/facts/minutes
.venv/bin/python -m pytest test_parsers.py -q
# 35 passed
```

## 8. Dependencies

Only **`pdfplumber`** is genuinely new: it is the package's sole third-party
import, needed because Postgres has no page-level text. `psycopg2-binary`
replaced shelling out to `psql` inside the container, which bypassed credentials
entirely instead of injecting them at runtime. `pytest` was already present in
system python3 at 9.0.2.

- `requirements.txt` — direct dependencies, pinned
- `requirements-lock.txt` — full resolved set (`pip freeze`)
- `requirements-dev.txt` — `pip-audit`, `pytest`
- `pip-audit` against the lock file: **no known vulnerabilities**
- `facts/.gitignore` excludes `.venv/` and `_build/`

Credentials are injected at runtime from the environment. **`.env` was not
edited** — and could not have been used regardless: its Postgres password is
32 characters while the running container's is 13, so `.env` is stale for
Postgres (see Recommended changes).

## 9. Recommended changes (requires operator approval — none made)

- **R1 — `documents.file_path` is stale corpus-wide.** Every value is rooted at
  `/home/donald/ksd_forensic/...`, which does not exist. This package works
  around it with a prefix rewrite in `locators.resolve_pdf_path`. Anything else
  reading that column is silently broken.
- **R2 — `document_pages` is empty and `chunks.source_page` is 100% NULL.** Page
  provenance was lost somewhere in `document_processor`. This defeats
  page-level citation for the RAG system, not just this fact layer.
- **R3 — `.env`'s `POSTGRES_PASSWORD` does not match the running container.**
  Any service relying on it for Postgres cannot authenticate.
- **R4 — `CLAUDE.md` documents the database as `qorvault/qorvault`;** it is
  `boarddocs/boarddocs`. (Flagging only — I do not modify CLAUDE.md files.)
- **R5 — The pre-2019 corpus exists only inside a backup directory.** The active
  project path holds 2019+ only. Fourteen years of records with a backup as
  their sole copy is a durability risk.
- **R6 — 5 meetings are scraped twice** under differently punctuated slugs
  (`...-work-session-5-00-p-m-` vs `...-work-session-500-pm`), all in 2026. The
  census dedupes them; the scraper should not create them.

## 10. Open items

| Item | Urgency |
|---|---|
| **D2** — definition behind "27 executive sessions in 2024" (see §4) | Blocks a hard fixture |
| **D3** — hand counts for six meetings, three per era | Blocks a hard fixture |
| **D4** — research docs (`research-04-document-identity.md`, `research-02 §2.6`, ADR 0007, the 2026-09-08 debrief) are not in the repo and were not available | Would inform table naming |
| 6 meetings where recorded voters exceed recorded attendance (§4) | Data-quality finding, not a defect |
| 2023 and 2024 minutes gaps (33.3% and 38.9% of regular meetings) from Phase 0 | Civic finding, unchanged |
| `pre_amendment_text` is NULL throughout | Amendments to motions before a vote are not distinguishable in this corpus; the column exists but no row populates it |

---

## Appendix — commands to regenerate

```bash
cd ~/workspace/projects/ksd-main/facts/minutes

# Credentials at runtime. .env's Postgres password is stale, so take the
# container's; nothing is written to .env.
export PGPASSWORD=$(podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep '^POSTGRES_PASSWORD=' | cut -d= -f2-)

# Schema and views
podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs \
  -v ON_ERROR_STOP=1 < schema.sql
podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs \
  -v ON_ERROR_STOP=1 < views.sql

# Full build (~71s)
.venv/bin/python build.py --reload

# Tests and fixtures
.venv/bin/python -m pytest test_parsers.py -q
.venv/bin/python fixtures.py

# Export the most recent meeting
.venv/bin/python export_meeting.py 2026-02-04 \
  -o ../../reports/meeting-export-2026-02-04.md

# Dependency audit
.venv/bin/python -m pip_audit -r requirements-lock.txt
```

Useful queries:

```sql
SELECT * FROM facts.exec_sessions_by_year ORDER BY year;
SELECT * FROM facts.votes_unnamed_by_year ORDER BY year;
SELECT * FROM facts.meetings_missing_minutes WHERE meeting_type = 'regular';
SELECT director, vote, count(*) FROM facts.votes_by_director
 GROUP BY 1,2 ORDER BY 1,2;
```
