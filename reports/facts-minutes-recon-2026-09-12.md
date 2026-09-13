# Phase 0 Recon — Minutes Fact Tables

**Date:** 2026-09-12
**Worktree:** `~/workspace/projects/ksd-main`
**Branch:** `claude/feat-facts-minutes` (branched from `main` @ `0ba6cc0`)
**Database:** `boarddocs` on `boarddocs-postgres` (podman), read-only throughout
**Status:** Phase 0 complete. **STOPPED for operator review.** No `facts` schema
objects were created. Nothing was written to `documents`, `chunks`, Qdrant or
`rag_api`.

---

## Executive summary — read this first

Four findings change the shape of the build. Two of them are premise
corrections, and one of them decides whether a `vote` table is worth building
at all.

1. **There is no existing `facts` package or `facts` schema to extend.** The
   task brief says the voucher fact table already exists and should be
   extended. It does not exist — not in any branch, not in any commit, and not
   in the database. This build starts the fact layer from scratch.
2. **Named per-director votes do not exist in the minutes.** Across all 874
   minutes documents spanning 2005–2026, exactly **2 documents** record votes
   with director names and **2 more** use a roll-call vote format. Every other
   motion is recorded as a bare "Motion carried." with no mover, no second, no
   names and no tally. A `vote` table sourced from minutes would contain
   roughly four documents' worth of rows.
3. **Named votes do exist — but in the agenda items, not the minutes.** 2,543
   BoardDocs `agenda_item` documents (2018–2026) carry a `Motion & Voting`
   block with mover, second, disposition and a named `Yea:`/`Nay:`/`Abstain:`
   roll. This is the only real source of vote-level fact data in the corpus,
   and it is outside the literal scope of "from board meeting minutes." **This
   is the main decision I need from you** (see Open Decisions, D1).
4. **There is no page-level text anywhere in the database.** `document_pages`
   has 0 rows and `chunks.source_page` is NULL for all 179,081 chunks. The
   locator rule ("document_id, page, and a text quote", chunk ids never the
   locator) therefore cannot be satisfied from the database. It requires
   re-extracting text per page from the source PDFs, which needs a PDF library
   that is **currently blocked pending your approval** (see Blockers, B1).

### STOP-rule evaluation

All three Phase 0 stop conditions were evaluated. **None are breached**, so the
build may proceed once the decisions below are resolved.

| Stop condition | Threshold | Measured | Verdict |
|---|---|---|---|
| Regular meetings since 2015 lacking minutes | > 15% | **12.3%** (25 of 203) | **PASS** — but see the 2023/2024 concentration below |
| Sampled minutes with a usable text layer | < 60% | **100%** (874 of 874 have ≥500 chars) | **PASS** |
| Attachment offset not consistent enough to state as a rule | — | 796/796 positive; 88.6% exact multiples of 7 days | **PASS** — rule is stateable |

---

## 1. Inventory

Minutes are **attachments**, not a document type of their own. The corpus holds
874 minutes PDFs. Identification was by title match on `%minutes%` against
`document_type='attachment'`; the filenames are highly regular and encode the
date of the meeting the minutes describe.

**Text-layer presence is 100%.** Every one of the 874 documents has at least
500 characters of extracted text; the mean is ~5,300 characters. All 874 are
flagged `ocr_applied` and all are `processing_status='complete'`. No document
required sampling to establish this — the check ran over the full set, which is
stronger than the 3-per-year sample the brief asked for.

| Year | Docs | Text ≥500 chars | Mean chars | Year | Docs | Text ≥500 chars | Mean chars |
|---|---|---|---|---|---|---|---|
| 2005 | 26 | 26 (100%) | 8,299 | 2016 | 46 | 46 (100%) | 5,953 |
| 2006 | 29 | 29 (100%) | 9,342 | 2017 | 51 | 51 (100%) | 5,134 |
| 2007 | 26 | 26 (100%) | 11,088 | 2018 | 39 | 39 (100%) | 4,319 |
| 2008 | 41 | 41 (100%) | 6,734 | 2019 | 57 | 57 (100%) | 4,867 |
| 2009 | 30 | 30 (100%) | 7,332 | 2020 | 63 | 63 (100%) | 4,173 |
| 2010 | 46 | 46 (100%) | 3,934 | 2021 | 59 | 59 (100%) | 4,936 |
| 2011 | 36 | 36 (100%) | 6,106 | 2022 | 65 | 65 (100%) | 3,558 |
| 2012 | 31 | 31 (100%) | 7,261 | 2023 | 34 | 34 (100%) | 3,334 |
| 2013 | 35 | 35 (100%) | 7,072 | 2024 | 31 | 31 (100%) | 2,992 |
| 2014 | 33 | 33 (100%) | 6,816 | 2025 | 51 | 51 (100%) | 3,280 |
| 2015 | 37 | 37 (100%) | 6,584 | 2026 | 8 | 8 (100%) | 3,343 |

Note the years above are the year of the meeting the minutes were *attached
to*. Grouped by the meeting date stated in the minutes body, the earliest is
**2004-08-31** and the latest **2026-02-04**.

Date recovery from the documents is reliable: the meeting date parsed out of
the minutes body for **866 of 874** documents (99.1%), and out of the filename
for 858. Where both parsed, they disagree for only 33 documents (3.8%) — those
33 need individual review in Phase 1 and are listed in the run report, not
guessed at.

### Corpus geography (a correction worth recording)

The live data root `~/workspace/projects/ksd_forensic/boarddocs/data` holds only
**2019 onward** (729 meeting directories). The complete 2005–2026 corpus (1,686
directories, 10,677 PDFs) exists only under
`~/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data`.

The `documents.file_path` column stores stale paths rooted at
`/home/donald/ksd_forensic/...`, which no longer exists. A single prefix
rewrite — `/home/donald/` → `/home/donald/qorvault-dev-archive/framework-backup/home/`
— resolves **874 of 874 (100%)** minutes PDFs to real files on disk. Phase 1
can rely on this mapping, but the stale column is a latent trap for anything
else that reads `file_path` (see Recommended changes, R1).

## 2. Completeness audit

The `agenda` document type is **not** a usable meeting census: it holds only 3
rows for 2019 and 4 for 2020. I used the scraped meeting directory names
instead, which encode both date and meeting type, and excluded non-meetings
(cancellations, receptions, WSSDA conference attendance, community events).
That yields a census of **1,395 meetings**.

A meeting counts as covered when some minutes document's parsed date equals
that meeting's date. Same-day meetings are correctly credited by a single
combined minutes document, which is the normal Kent pattern (one PDF covering
the regular meeting plus that evening's work session and executive session).

| Year | Regular (total/missing) | Special | Exec session | Work/study |
|---|---|---|---|---|
| 2005 | 15 / 5 (33.3%) | 11 / 1 (9.1%) | 1 / 1 (100%) | — |
| 2006 | 18 / 3 (16.7%) | 12 / 1 (8.3%) | 2 / 2 (100%) | — |
| 2007 | 19 / 6 (31.6%) | 9 / 1 (11.1%) | 1 / 1 (100%) | — |
| 2008 | 20 / 4 (20.0%) | 21 / 1 (4.8%) | 2 / 0 (0%) | 1 / 0 (0%) |
| 2009 | 20 / 4 (20.0%) | 14 / 3 (21.4%) | 17 / 10 (58.8%) | 1 / 0 (0%) |
| 2010 | 19 / 1 (5.3%) | 24 / 3 (12.5%) | 15 / 4 (26.7%) | 6 / 0 (0%) |
| 2011 | 18 / 1 (5.6%) | 14 / 0 (0%) | 19 / 1 (5.3%) | 1 / 0 (0%) |
| 2012 | 16 / 0 (0%) | 23 / 2 (8.7%) | 8 / 0 (0%) | — |
| 2013 | 18 / 1 (5.6%) | 28 / 1 (3.6%) | 11 / 2 (18.2%) | — |
| 2014 | 18 / 4 (22.2%) | 27 / 5 (18.5%) | 20 / 6 (30.0%) | — |
| 2015 | 18 / 0 (0%) | 27 / 4 (14.8%) | 23 / 11 (47.8%) | 2 / 0 (0%) |
| 2016 | 20 / 1 (5.0%) | 37 / 0 (0%) | 11 / 1 (9.1%) | 1 / 0 (0%) |
| 2017 | 18 / 2 (11.1%) | 31 / 2 (6.5%) | 22 / 8 (36.4%) | 9 / 0 (0%) |
| 2018 | 18 / 5 (27.8%) | 8 / 3 (37.5%) | 34 / 17 (50.0%) | 30 / 9 (30.0%) |
| 2019 | 18 / 1 (5.6%) | 9 / 0 (0%) | 31 / 7 (22.6%) | 29 / 1 (3.4%) |
| 2020 | 18 / 2 (11.1%) | 14 / 3 (21.4%) | 29 / 7 (24.1%) | 31 / 3 (9.7%) |
| 2021 | 19 / 0 (0%) | 16 / 2 (12.5%) | 29 / 5 (17.2%) | 26 / 0 (0%) |
| 2022 | 17 / 0 (0%) | 17 / 2 (11.8%) | 35 / 5 (14.3%) | 28 / 1 (3.6%) |
| **2023** | **18 / 6 (33.3%)** | 8 / 4 (50.0%) | 29 / 12 (41.4%) | 31 / 11 (35.5%) |
| **2024** | **18 / 7 (38.9%)** | 9 / 5 (55.6%) | 22 / 13 (59.1%) | 32 / 15 (46.9%) |
| 2025 | 18 / 0 (0%) | 8 / 0 (0%) | 26 / 1 (3.8%) | 22 / 1 (4.5%) |
| 2026 | 3 / 1 (33.3%) | 1 / 0 (0%) | 2 / 1 (50.0%) | 4 / 1 (25.0%) |

The aggregate for regular meetings since 2015 is 12.3% missing, which clears the
15% stop threshold. **But the gap is not evenly spread.** 2023 and 2024 are
outliers at 33.3% and 38.9%, against 0% for both 2022 and 2025 on either side.

I verified the 2024 gap is real rather than a date-parsing artifact by listing
every minutes attachment in 2024 H1 by hand. Minutes exist for the 01-10,
02-28, 04-24, 05-01, 06-26 and 07-10 meetings. There is genuinely nothing in
the corpus for the 01-24, 02-14, 03-13, 03-27, 05-08, 05-22 and 06-12 regular
meetings.

What this does **not** tell us is *why*: the data cannot distinguish "minutes
were never adopted" from "minutes were adopted but never posted to BoardDocs"
from "posted but missed by the scraper." That distinction matters and is not
answerable from this corpus. I have not drawn a conclusion about it.

## 3. Attachment offset

**The rule holds and can be stated.** Minutes are attached to the agenda of a
*later* meeting — the one at which they are approved. Of 796 documents where
both the minutes date and the attachment date resolved:

- **796 of 796 (100%) offsets are positive.** There are no exceptions. No
  minutes document is ever attached to its own meeting or an earlier one.
- **705 of 796 (88.6%) are exact multiples of 7 days**, consistent with a board
  meeting on a fixed weekday.
- **751 of 796 (94.3%) fall between 7 and 35 days.**

| Offset | Count | Offset | Count |
|---|---|---|---|
| 14 days | 331 | 8 days | 14 |
| 7 days | 151 | 15 days | 13 |
| 21 days | 96 | 16 days | 12 |
| 28 days | 66 | 63 days | 11 |
| 35 days | 37 | 6 days | 10 |

14 days dominates because regular meetings are biweekly. The non-multiples of 7
are mostly minutes of a *special* meeting held midweek, attached to the next
regular meeting. The long tail (49, 63 days) is the summer recess.

**Practical consequence for the schema:** `meeting.meeting_date` must come from
the minutes body, and `documents.meeting_date` must be treated as the
*approval* meeting, i.e. it populates `approved_at_meeting_id`, never
`meeting_date`. Getting this backwards would misdate the entire fact layer by
one meeting. Example: `Board Minutes 2024 01 10.pdf` carries
`documents.meeting_date = 2024-01-24`; the meeting it documents is 2024-01-10.

### Amendments at approval

**Essentially not recorded.** "As corrected" appears in only **2** of 874
minutes documents, and in **0** agenda items. "As amended" appears in 56
minutes documents, but inspection shows these almost always refer to amending
*that evening's agenda*, not correcting prior minutes.

The approval event itself is recorded — as a consent-agenda line item, e.g.
"9.10 Minutes of February 25, 2026 Regular Meeting and Special Meetings (Work
Session, Audit Entrance Conference and Accountability Audit, and Executive
Session)". So `approved_at_meeting_id` is recoverable, but
`approved_as_corrected` will be false and `correction_note` NULL for
essentially the whole corpus. I recommend keeping both columns anyway — their
emptiness is itself a finding about record-keeping practice.

## 4. Vote format — the most important finding

I classified all 874 documents rather than the 20 requested, because a full
pass cost no more than a sample.

**First, a correction to my own method.** An initial pass reported a large
`tally_only` population. That was wrong — a loose `\d-\d` regex was matching
motion numbers ("Motion No. 03-10"), school years ("2005-06") and grade ranges
("1-12"). After requiring vote language adjacent to the digits, **the corpus
contains zero vote tallies.** I am flagging this because the uncorrected number
would have led directly to building a `tally_yes`/`tally_no` extraction path
that has nothing to extract.

| Year | Docs | Motions | carried_no_names | no_motion_language | named | roll_call |
|---|---|---|---|---|---|---|
| 2005 | 26 | 21 | 16 | 10 | — | — |
| 2006 | 29 | 28 | 19 | 10 | — | — |
| 2007 | 26 | 41 | 20 | 6 | — | — |
| 2008 | 41 | 26 | 19 | 22 | — | — |
| 2009 | 30 | 39 | 18 | 12 | — | — |
| 2010 | 50 | 52 | 23 | 27 | — | — |
| 2011 | 32 | 122 | 16 | 16 | — | — |
| 2012 | 31 | 127 | 18 | 13 | — | — |
| 2013 | 38 | 149 | 26 | 12 | — | — |
| 2014 | 31 | 92 | 18 | 12 | — | — |
| 2015 | 37 | 133 | 23 | 14 | — | — |
| 2016 | 46 | 128 | 25 | 20 | 1 | — |
| 2017 | 51 | 84 | 22 | 28 | 1 | — |
| 2018 | 39 | 87 | 15 | 24 | — | — |
| 2019 | 58 | 192 | 23 | 35 | — | — |
| 2020 | 62 | 213 | 25 | 37 | — | — |
| 2021 | 61 | 244 | 27 | 34 | — | — |
| 2022 | 62 | 189 | 28 | 34 | — | — |
| 2023 | 34 | 98 | 14 | 20 | — | — |
| 2024 | 32 | 117 | 13 | 17 | — | 2 |
| 2025 | 51 | 173 | 27 | 24 | — | — |
| 2026 | 6 | 28 | 2 | 4 | — | — |
| **Total** | **874** | **2,385** | **438** | **431** | **2** | **2** |

Corroborating counts across the whole minutes corpus: `seconded by` appears
**0** times. `moved by`/`motion by` appears in **14** documents out of 874.
"voted no"/"voted against" appears **0** times. "abstain" appears in 8.

`no_motion_language` (431 documents) is not a parse failure — these are work
session, study session and executive session minutes that genuinely contain no
motions. The brief anticipates this: a meeting row with zero motions is valid.

**What a typical motion actually looks like (2005–2022):**

> Motion No. 51-11 That the Board of Directors approves adoption of Revised
> Policy 3207: Prohibition of Harassment, Intimidation, and Bullying.
>
> Motion carried.

**And in the modern era (2022–2026):**

> A motion was made to approve the agenda as presented.
> The motion carried.

Neither records who moved, who seconded, or how anyone voted.

### Where the named votes actually are

The BoardDocs `agenda_item` documents carry a structured `Motion & Voting`
block that the minutes PDFs do not:

> Motion & Voting
> A motion was made to approve Second Reading and Approval of Policy 3410 Student Health.
> Motion by Laura Williams, second by Andy Song.
> Final Resolution: Motion Carries
> Yea: Meghin Margel, Donald Cook, Andy Song, Teresa Gregory, Laura Williams

Coverage, 2018–2026 only (BoardDocs did not expose this before 2018):

| Year | Blocks | Yea | Nay | Abstain | Mover+second | Carries | Fails |
|---|---|---|---|---|---|---|---|
| 2018 | 113 | 113 | 20 | 33 | 113 | 112 | 1 |
| 2019 | 415 | 412 | 23 | 2 | 413 | 411 | 2 |
| 2020 | 381 | 379 | 72 | 3 | 379 | 376 | 3 |
| 2021 | 363 | 359 | 38 | 8 | 360 | 351 | 8 |
| 2022 | 321 | 318 | 31 | 25 | 319 | 316 | 3 |
| 2023 | 361 | 361 | 25 | 28 | 361 | 361 | 2 |
| 2024 | 214 | 214 | 59 | 18 | 214 | 212 | 12 |
| 2025 | 330 | 326 | 73 | 29 | 329 | 324 | 37 |
| 2026 | 45 | 43 | 14 | 0 | 45 | 43 | 1 |
| **Total** | **2,543** | **2,525** | **355** | **146** | **2,533** | **2,500** | **29** |

Only two dispositions occur: `Motion Carries` (2,500) and `Motion Fails` (29).
There is no tabled/withdrawn disposition in this vocabulary, so the schema's
`{adopted|lost|tabled|withdrawn}` enum will only ever see its first two values
from this source.

## 5. Format eras

Two eras with a clean summer-2022 boundary. Grouping is by which layout signals
fire in each document.

**Era A — "Numbered motion" (2004-08-31 → 2022-05-11).** Motions carry a
sequential number scoped to the year (`Motion No. 51-11`), the motion text
begins "That the Board of Directors…", and disposition is a standalone
"Motion carried." line. Attendance is prose in the opening paragraph: "…with
President Bill Boyce presiding. Other board members present: Jim Berrios, Tim
Clark, Karen DeBruler and Debbie Straus."
Sample locator: `Board_Meeting_Minutes_060811.pdf`, doc
`3e98f997-2876-4328-9761-135868c1645d` era-mate; quote "Motion No. 51-11 That
the Board of Directors approves adoption of Revised Policy 3207".

**Era B — "Roll call / passive motion" (2022-08-24 → 2026-02-04).** A `Roll
Call` heading introduces a structured attendance block ("President Margel:
Present / Vice President Cook: Present / Director Song: Present (attended
virtually)"). Motions are passive and unnumbered: "A motion was made to approve
the agenda as presented. The motion carried."
Sample locator: `Board Minutes 2025 02 26.pdf`; quote "Roll Call / President
Margel: Present / Vice President Cook: Present".

**Boundary evidence:** the last document using `Motion No.` has body date
**2022-05-11**; the first using `Roll Call` has body date **2022-08-24**. No
document uses both conventions. A parser selected by date range with a cut at
2022-07-01 will route every document correctly.

Important subtlety: in Era B, **"Roll Call" means attendance, not a recorded
vote.** Treating that heading as a vote roll would fabricate vote rows for
every meeting. It must map to the `attendance` table.

385 documents fire no motion signals at all; they span both eras and are the
work/study/executive session minutes.

## 6. Roster

**No director roster exists** — not in the repo (no file matching
roster/director/seat/term), not in the `facts` schema (which does not exist),
and not as a database table. The `public` schema holds only `documents`,
`chunks`, `document_pages`, `tenants` and session/logging tables.

Therefore, per the brief: **votes and attendance will store raw names only**,
with `director_norm` NULL throughout. No fuzzy matching will be attempted.

Worth noting for a future task: a roster *could* be derived deterministically,
because Era B roll-call blocks give full name plus office ("President Margel",
"Vice President Cook") and the agenda-item `Yea:` lines give full names. That
is a separate piece of work and I have not started it.

---

## Blockers requiring your decision

### B1 — PDF text extraction is blocked (blocks Phase 1 locators)

**What happened.** `pdftotext` is not installed on this machine. Installing it
needs `sudo`, which your `block-dangerous-commands.sh` hook correctly refuses.
I then tried to create a local Python virtualenv with `pdfplumber` and `pypdf`
instead, and your `ai-review-ask-commands.sh` hook escalated that for human
approval, because those packages are not in an existing `requirements.txt`.

**I did not work around either hook,** and I am not asking you to disable them.

**Why this matters.** The locator rule requires a page number and a quote from
the source PDF page, and forbids using chunk ids as the locator. The database
cannot supply a page number for any document: `document_pages` is empty and
`chunks.source_page` is NULL for all 179,081 chunks. Page numbers can only come
from re-reading the PDFs. Roughly 45% of minutes documents carry a "Page N"
footer in their extracted text, so a text-only fallback would leave more than
half the corpus without a page locator.

**What I need:** approval to run, inside the worktree,

```
cd ~/workspace/projects/ksd-main/facts/minutes
python3 -m venv .venv
.venv/bin/pip install pdfplumber pypdf psycopg2-binary
```

`pdfplumber` with `layout=True` is the closest available substitute for
`pdftotext -layout`; both are pure Python, no system packages, no sudo.
`psycopg2-binary` replaces the `podman exec psql` shelling I used for Phase 0,
which is fine for recon but not for a production parser. I will add a
`requirements.txt` so the hook has something to check against next time.

If you would rather not add dependencies, the fallback is to accept
character-offset-into-`content_text` as the locator with a page number only
where a footer exists. That is weaker and I do not recommend it.

### D1 — Should the `vote` table be sourced from agenda items?

This is the real decision from Phase 0. The brief says to build the fact tables
"from board meeting minutes." Taken literally, the `vote` table will hold rows
from 4 documents out of 874, because the minutes simply do not record how
individuals voted. Meanwhile 2,543 agenda items record exactly that, with
mover, second and per-director yea/nay/abstain, for 2018–2026.

Three options:

- **(a) Minutes only, as written.** `vote` stays nearly empty. Honest to the
  brief and honest about what the minutes contain. The emptiness is itself a
  civically meaningful finding about Kent's minute-taking.
- **(b) Minutes for meeting/attendance/motion/exec-session, agenda items for
  `vote` and for motion mover/second/disposition where available.** Every row
  still carries `document_id` + page + quote, pointing at the agenda-item
  document rather than the minutes PDF, so the locator rule is honored. This
  produces a genuinely useful vote table for 2018 onward. **This is my
  recommendation.**
- **(c) Defer votes entirely** to a separate task scoped to agenda items.

I have not built any of these. Phase 1 waits on your answer.

### D2 — Pin down the definition behind the "27 executive sessions in 2024" fixture

The hard fixture says calendar 2024 contains 27 executive sessions. I cannot
yet reproduce that number, and I want the definition fixed before I build to
it rather than tuning a regex until it hits 27 — which would make the fixture
meaningless as a check.

What the data shows for 2024, three different ways of counting:

- **22** — meetings whose scraped directory is an executive session
  (`special-meeting-executive-session` and variants).
- **14** — sentences in 2024-dated minutes containing "executive session", but
  this count is badly contaminated: most matches are consent-agenda line items
  like "08 – Minutes of 13 December 2023 Regular Meeting, and Executive
  Session", not announcements.
- **A small number** of genuine in-meeting announcements, e.g. from
  2024-09-11: "President Margel announced an Executive Session for
  approximately three minutes at 10:09 p.m.", followed by "President Margel
  announced an extension of the Executive Session for approximately ten minutes
  at 10:12 p.m." and "The Executive Session was adjourned at 10:25 p.m."

22 scheduled executive-session meetings plus a handful of executive sessions
announced mid-meeting plausibly reaches 27, but "plausibly" is not good enough
for a hard fixture. **Please confirm:** does the 27 count scheduled
executive-session meetings, announcements entered in minutes, or both — and
does an "extension" of a session count as a second session? The brief's own
schema note says "Executive session extended or reconvened: one row per
announcement", which suggests extensions *do* count separately; that reading
should be confirmed against whatever produced the 27.

### D3 — Source documents for the six hand-counted meetings

The brief requires the operator to supply, before Phase 1, a hand count of
motions and dispositions for six meetings (two per era). Phase 0 found **two**
eras, not the three or more the brief may have assumed, so that is three
meetings per era. I suggest drawing them from Era A (say 2011, 2016, 2019) and
Era B (2023, 2024, 2025), but the choice should be yours since it is the
independent check on my parser.

### D4 — Research documents are not in the repo

The brief cites `research-04-document-identity.md`, `research-02 §2.6`,
`ADR 0007 draft` and `session-debrief-2026-09-08-retrieval-smoke-test.md`, and
says to copy them into `docs/` if they live only in project knowledge. **None
of them are in this repo** — `docs/` contains only the OSPI and agenda-analysis
memos. I could not copy them because I have no access to your Claude.ai project
knowledge. Phase 0 did not depend on them, but the ADR 0007 fact-layer schema
would materially inform Phase 1 table naming. Please paste or upload them.

---

## Recommended changes (requires operator approval — none made)

- **R1 — `documents.file_path` is stale corpus-wide.** Every value is rooted at
  `/home/donald/ksd_forensic/...`, which does not exist. Anything that trusts
  this column to open a file is silently broken today. Suggest a one-time
  `UPDATE` to repoint at the archive root, or a documented resolver function.
  I have not touched `documents`.
- **R2 — `document_pages` is empty and `chunks.source_page` is entirely NULL.**
  Page provenance was dropped somewhere in `document_processor`. This defeats
  page-level citation for the RAG system as well as this fact layer. Worth
  investigating independently of this task.
- **R3 — `CLAUDE.md` documents the database as `qorvault/qorvault`;** the
  running container is `boarddocs/boarddocs`. The documented connection string
  fails outright. (Flagging only — I do not modify CLAUDE.md files.)
- **R4 — The pre-2019 corpus exists only inside a backup directory.** The
  active project path holds 2019+ only. A backup being the sole copy of 14
  years of records is a durability risk worth addressing on its own merits.

## System state

Nothing was created, altered or deleted in `documents`, `chunks`,
`document_pages`, Qdrant or `rag_api`. No `facts` schema was created. All
database access was `SELECT`-only via `podman exec … psql`. All file access
outside the worktree was read-only.

Files added to the worktree, uncommitted:

- `facts/minutes/recon_phase0.py`
- `facts/minutes/recon_completeness.py`
- `reports/facts-minutes-recon-2026-09-12.md`

Branch `claude/feat-facts-minutes` was created off `main` @ `0ba6cc0`; no
commit has been made yet.

---

## Appendix — commands

Environment and schema discovery:

```bash
cd ~/workspace/projects/ksd-main
git worktree list
git branch -a
git ls-files | grep -i -E 'fact|voucher'          # empty: no facts package
git log --all --oneline --diff-filter=A -- '*facts*'   # empty: never existed

podman ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
podman inspect boarddocs-postgres \
  --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i '^POSTGRES_'

podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\dn"
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\dt public.*"
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\d documents"
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\d chunks"
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\d document_pages"
```

Confirming there is no page-level data:

```bash
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c \
  "SELECT count(*) AS chunks_total, count(source_page) AS with_page FROM chunks;"
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c \
  "SELECT count(*) FROM document_pages;"
```

Inventory and text layer (item 1):

```bash
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "
SELECT extract(year from meeting_date)::int AS yr, count(*) AS docs,
       count(*) FILTER (WHERE content_text IS NOT NULL AND length(content_text) >= 500) AS text_ge500,
       round(avg(length(coalesce(content_text,''))))::int AS avg_len,
       count(*) FILTER (WHERE ocr_applied) AS ocr,
       count(*) FILTER (WHERE processing_status <> 'complete') AS not_complete
FROM documents WHERE document_type='attachment' AND title ILIKE '%minutes%'
GROUP BY 1 ORDER BY 1;"
```

Corpus geography and path resolution:

```bash
A=~/workspace/projects/ksd_forensic/boarddocs/data
B=~/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data
for R in "$A" "$B"; do
  echo "$R  dirs: $(ls "$R" | wc -l)  pdfs: $(find "$R" -name '*.pdf' | wc -l)"
  ls "$R" | grep -oE '^[0-9]{4}' | sort | uniq -c
done
```

Items 3, 4, 5 (offset, vote format, eras) — full-corpus pass:

```bash
cd ~/workspace/projects/ksd-main
python3 facts/minutes/recon_phase0.py > /tmp/p0.json
```

Item 2 (completeness) — meeting census from scraped directories:

```bash
cd ~/workspace/projects/ksd-main/facts/minutes
python3 recon_completeness.py > /tmp/p0c.json
```

Named votes in agenda items (the D1 evidence):

```bash
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "
SELECT extract(year from meeting_date)::int AS yr,
 count(*) FILTER (WHERE content_text ~ 'Yea:')     AS yea,
 count(*) FILTER (WHERE content_text ~ 'Nay:')     AS nay,
 count(*) FILTER (WHERE content_text ~ 'Abstain:') AS abstain,
 count(*) FILTER (WHERE content_text ~* 'Motion by .+ second by') AS mover_second,
 count(*) FILTER (WHERE content_text ~* 'Final Resolution: *Motion Carries') AS carries,
 count(*) FILTER (WHERE content_text ~* 'Final Resolution: *Motion Fails')   AS fails
FROM documents WHERE document_type='agenda_item' AND content_text ~* 'motion *& *voting'
GROUP BY 1 ORDER BY 1;"
```

Era boundary:

```bash
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -t -A -c "
SELECT 'FIRST_ROLLCALL', to_char(meeting_date,'YYYY-MM-DD'), title FROM documents
WHERE document_type='attachment' AND title ILIKE '%minutes%' AND content_text ~ 'Roll Call'
  AND meeting_date > '2020-01-01' ORDER BY meeting_date LIMIT 3;
SELECT 'LAST_MOTIONNO', to_char(meeting_date,'YYYY-MM-DD'), title FROM documents
WHERE document_type='attachment' AND title ILIKE '%minutes%' AND content_text ~ 'Motion No\.'
ORDER BY meeting_date DESC LIMIT 3;"
```

Roster check (item 6):

```bash
cd ~/workspace/projects/ksd-main
git ls-files | grep -iE 'roster|director|board_member|person|seat'   # empty
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -t -A -c \
  "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
```
