# Ingest Degradation Diagnostic — 2026 Unlinked Attachments and Unprocessed Agenda Items

**Date:** 2026-09-13
**Scope:** Read-only diagnostic. No re-ingestion, no record patching, no scraper execution, no network access to BoardDocs.
**Inputs:** PostgreSQL `boarddocs` (container `boarddocs-postgres`), loader source at `boarddocs_loader/`, scraper output trees on disk.
**Predecessor:** `reports/boarddocs-coverage-2026-09-12.md` (Phase 0 coverage audit).

---

## Executive summary

The Phase 0 report read this as a single degrading trend — 26 unlinked in 2024, 16 in 2025,
73 in 2026. It is not one trend. It is **three unrelated things**, and one of the three is not
a defect at all.

1. **The 2024–2025 counts are not a BoardDocs problem.** All 42 of them are *email*
   attachments loaded from a separate source tree on 2026-02-25. They have no `meeting_id`,
   no agenda item exists for them to link to, and none is expected. BoardDocs attachments
   with no agenda-item link in 2024 and 2025: **zero**. The regression line in the Phase 0
   report should be redrawn.

2. **The 2026 problem is real, and it is 71 records, not 73.** (The other 2 are email
   attachments, same as above.) The cause is **not** a code regression, a BoardDocs markup
   change, or rate-limit truncation. It is that **a second, older scraper was run against
   2026 meetings on 2026-02-22**, and it writes a directory layout the loader is designed to
   treat as unlinkable. The loader behaved exactly as written. The input changed.

   *Scope corrected 2026-09-13: that run covered **806 meetings over 4h36m**, not only 2026
   ones. The 71-record count here is unaffected — 2026 is merely where it collided. See*
   **Correction C1** *in §0.2.*

3. **The 48 unprocessed agenda items are not stuck or failed. They were never attempted.**
   All 48 were loaded on 2026-03-25 and the `document_processor` stage has not run since.
   They carry full `content_raw` and `content_text`, no `processing_error`. This is an
   unfinished pipeline run, not a failure.

Separately, and more valuable than any of the above: **`source_url` is derivable for 12,207
of 12,305 attachments (99.2%) from data already sitting in PostgreSQL.** No re-scrape and no
network access are required. The Phase 0 report concluded this was unrecoverable; that
conclusion was based on `external_id` alone and missed two other columns that carry the
BoardDocs file IDs. Also note: the URL pattern in the task brief is wrong — the live markup
uses `/files/`, not `/pfiles/`.

**Cause classification, ranked:**

| Rank | Cause | Records | Evidence strength |
|---|---|---|---|
| 1 | **Scraper substitution** — legacy flat-format scraper run against 2026 meetings | 71 | Confirmed, offline-reproduced |
| 2 | **Pipeline stage never run** — `document_processor` not run after 2026-03-25 ingest | 48 | Confirmed from DB state |
| 3 | **Misclassification in the prior report** — email attachments counted as BoardDocs | 44 | Confirmed |
| — | Site markup change | 0 | Ruled out |
| — | Key format change | 0 | Ruled out |
| — | Scraper code regression | 0 | Ruled out |
| — | Rate-limit truncation | 0 | Ruled out |

---

## Phase 0 — Characterization

### 0.1 The affected records

**Unlinked attachments, 2024-01-01 onward, clustered by meeting** (query A1):

Two clearly separate populations, distinguishable by `meeting_id` and ingest timestamp.

*Population A — email attachments (44 records, not a defect):*

| Span | Records | `meeting_id` | Ingested |
|---|---|---|---|
| 2024-02-17 → 2024-12-21 | 26 | *(empty)* | 2026-02-25 14:10:23–25 |
| 2025-02-22 → 2025-12-16 | 16 | *(empty)* | 2026-02-25 14:10:25 |
| 2026-01-17 | 2 | *(empty)* | 2026-02-25 14:10:26 |

`external_id` prefix is `email_<Message-ID>_<filename>`; `file_path` is under
`input/Email/attachments/`. Example:

```
email_<DM6PR08MB3947...@DM6PR08MB3947.namprd08.prod.outlook.com>_Stipend Sheet Break out 1-16-2026 - 24-25 Full list.pdf
```

These are public-records-request email attachments, not BoardDocs documents. They correctly
have no agenda item.

*Population B — BoardDocs attachments (71 records, the actual defect):*

| Meeting date | `meeting_id` | Unlinked | Linked (same meeting) | Ingested |
|---|---|---|---|---|
| 2026-01-28 | DQHKYP542B0C | 30 | 65 | 2026-02-23 23:21:08 |
| 2026-02-04 | DQU2PT0331CA | 1 | 1 | 2026-02-23 23:21:08 |
| 2026-02-11 | DQU45Y09F4B0 | 21 | **0** | 2026-02-23 23:21:08 |
| 2026-02-11 | DQU47R0A3706 | 19 | **0** | 2026-02-23 23:21:08 |

All 71 were written in a **single loader run** at 2026-02-23 23:21:08. File types: 71/71
`.pdf`. No scraper version or commit is recorded on the document row — the loader does not
capture one (see *Gaps in recorded provenance* below).

Meetings **before** 2026-01-28 are clean: 2026-01-07 (2 linked / 0 unlinked), 2026-01-14
(29 / 0 and 57 / 0).

**Unprocessed agenda items — all 48** (query A5):

| Meeting date | `meeting_id` | Count | Loaded | `content_text` null | `processing_error` |
|---|---|---|---|---|---|
| 2026-03-25 | DS4MSK5CA5C5 | 42 | 2026-03-25 23:02:54 | 0 | none |
| 2026-03-25 | DS4MVC5CCBE6 | 2 | 2026-03-25 23:03:01 | 0 | none |
| 2026-03-25 | DSCM9K5A287D | 4 | 2026-03-25 23:03:02 | 0 | none |

All 48 are from one date and one loader run. This is a different incident from the
attachment problem — a month later, different meetings, different pipeline stage.

### 0.2 Onset date

**Onset: 2026-01-28, introduced retroactively by a scrape run on 2026-02-22**
~~**07:02–07:05.**~~ → **06:43:20–11:19:07.** *(time window corrected 2026-09-13 —
see* **Correction C1** *at the end of this section)*

The onset is not where it appears. The first *meeting date* with an unlinked BoardDocs
attachment is 2026-01-28. But that meeting was scraped correctly on 2026-02-04 and its
attachments linked fine — the 65 linked records prove it. The 30 unlinked records for the
same meeting were created by a **second scrape of the same meeting** two and a half weeks
later.

Evidence — two directories exist for the same meeting, with different naming conventions
and different `meeting.json` schemas (query/command A6, A7):

| | Structured scrape | Flat scrape |
|---|---|---|
| Directory | `2026-01-28-regular-meeting-6-30-p-m-` | `2026-01-28-regular-meeting-630-pm` |
| Directory mtime | 2026-02-04 01:26 | **2026-02-22 07:05** |
| `meeting.json` `scrapedAt` | `2026-02-04T09:24:23.013Z` | `2026-02-22T07:05:11.370662` |
| JSON key style | camelCase: `categories`, `meetingSlug`, `meetingType`, `meetingUrl`, `scrapedAt` | snake_case: `meeting_id`, `slug`, `name`, `source_url`, `scraped_at`, `committee_id`, `files_found` |
| Item subdirectories | 47 | **0** |
| Filenames | `1210 - Proposed.pdf` (spaces preserved) | `1210_-_Proposed.pdf` (spaces → underscores) |
| `agenda.html` retained | no | **yes** |

The two `meeting.json` schemas are produced by two different programs. The camelCase one
matches the TypeScript/Puppeteer scraper at
`~/workspace/projects/ksd_forensic/boarddocs-scraper/`. The snake_case one is the older
Python scraper whose output convention (`-630-pm` slugs, flat layout, `agenda.html`
retained) is the same convention used across the entire 2005–2017 corpus.

**What the onset did *not* coincide with:**

- **No scraper commit.** `boarddocs-scraper` git HEAD is `fc9d2ae "Fix scraping agendas"`;
  nothing in the log lands near 2026-02-22, and the structured scraper's output for
  2026-01-14 (2026-02-04 era) and 2026-03-11/03-25 is normal.
- **No BoardDocs markup change.** The retained `agenda.html` for 2026-02-11 uses the same
  `/wa/ksdwa/Board.nsf/files/<ID>/$file/<name>` href form found throughout the corpus. The
  markup is intact — see §Phase 1 `source_url`.
- **No key format change.** `meeting_id` and `agenda_item_id` values are the same 12-character
  uppercase alphanumeric form in 2026 as in 2019 (`DQHLJD5650A5` vs `BCX52982A00C`).
- **No rate-limit truncation.** Per-meeting counts are complete against the agenda; nothing
  is cut off mid-run.

**Why 2026-02-11 has zero linked attachments.** The structured scraper *did* run for that
meeting — on **2026-02-10, the day before the meeting**. At that point the agenda was
published but the item detail pages were not yet populated. That run produced only `.txt`
files and a single `archive_*` subdirectory containing no `item.json`, so it yielded zero
agenda items and zero attachments (command A8). The structured scraper then never revisited
the meeting. The only source of those 40 attachments is the 2026-02-22 flat scrape.

**Raw HTML retention** (command A9) — the material needed to date the onset *was* retained,
so no STOP condition applies:

| Era | Meeting dirs | With `agenda.html` |
|---|---|---|
| 2005–2017 (flat scraper) | ~1,000 | all |
| 2018 | 107 | 30 |
| 2019–2025 (structured scraper) | ~700 | **0** |
| 2026 | 27 | 9 |

`agenda.html` retention is a property of the *flat* scraper. The structured scraper discards
it. This matters for the `source_url` recovery below.

~~The era labels in the table above imply these files are survivors of the original
2005–2017 flat-scraper era, retained since the meetings were first scraped.~~ **That reading
is wrong — all 806 were written on 2026-02-22 by the run described above. See Correction
C1.**

---

### Correction C1 — 2026-09-13: the 2026-02-22 run was 4h36m across 806 meetings

*Added 2026-09-13, after the scraper's own run log was located. Published as a correction per
the append-only convention; the original text above is struck through, not deleted.*

**What was wrong.** §0.2 dated the run to **07:02–07:05** and characterised it as a
second scrape of a handful of 2026 meetings. Both were inferred from the mtimes of the four
affected 2026 meeting directories — the only evidence available at the time, because the
scraper and its log had not yet been found.

**What the run log shows.** The log was recovered from inside the backup corpus on
2026-09-13 and is now archived with the scraper at
`~/workspace/archive/legacy-flat-scraper/boarddocs_scraper.log`
(SHA-256 `6bb4e41a3f76123ec81e38e1d18338ed85715853811dc0786076ede7bf599760`,
mtime `2026-02-22 11:19:07.268286800 -0800`, 2,262,477 bytes):

| | Recorded in §0.2 | Actual |
|---|---|---|
| Window | 07:02–07:05 (~3 min) | **06:43:20 → 11:19:07 (~4h36m)** |
| Meetings touched | 4 | **806 distinct slugs** |
| Evidence | directory mtimes | 9,126 log lines |

Slug years in the log: 2005–2017 heavily (28–91/yr), 2018 × 30, 2026 × 9.

**Why the original inference was off — the script's mtime falls *inside* the run.** The
scraper file is stamped `2026-02-22 07:07:05`, which reads like a run boundary but is not;
the log's `11:19:07` is the true end. Dating the incident from directory and file mtimes
alone therefore compresses a 4.5-hour job into the few minutes during which the four
examined directories happened to be written.

**Independent corroboration from the corpus.** `agenda.html` retention is unique to this
scraper (table above). The corpus holds **806** `agenda.html` files; their year-by-year
distribution matches the log's slug distribution **exactly** — 2018 → 30, 2026 → 9, and so
on for every year — and **all 806 carry mtime `2026-02-22`**. Verify with:

```bash
B=/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data
ls $B/*/agenda.html | sed -E 's|.*/data/([0-9]{4})-.*|\1|' | sort | uniq -c
ls $B/*/agenda.html | xargs stat -c '%y' | cut -c1-10 | sort | uniq -c

cd ~/workspace/archive/legacy-flat-scraper
grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}-[a-z0-9-]+' boarddocs_scraper.log \
  | sort -u | cut -c1-4 | uniq -c
```

**What does NOT change.**

- **The 71-record damage figure stands.** 2005–2018 meetings were *already* flat in the
  corpus, so the flat `external_id` matched on re-scrape and `ON CONFLICT DO NOTHING` made
  those a genuine no-op. A collision required a meeting previously scraped *structured*,
  which in practice meant 2026 only. Wider blast radius, same record count.
- **The cause classification in §1.3 stands.** Scraper substitution, confirmed, offline
  reproduced.
- **The 31-duplicate / 40-orphan split stands.**

**What this does change.**

- **§0.2's framing** — "a second, older scraper was run against 2026 meetings" understates
  it. It was run against most of the corpus; 2026 is merely where it collided.
- **R1 and R4 Route 2 are not independently sourced.** Both parse `agenda.html` out of
  `documents.content_raw`, and every one of those 806 files came from this single run. The
  recovery remains sound — one consistent snapshot of intact BoardDocs markup — but a
  systematic flaw in this run's capture would be inherited by all 806 with nothing to check
  it against. **Recommend folding a sampled spot-check against live BoardDocs into the five
  URL confirmations already gating R4**, rather than treating the stored HTML as ground
  truth.
- **The retention table's era labels** are misleading and are struck through above.

**Provenance.** Scraper and log were found loose inside the authoritative 1,684-meeting
backup corpus and moved to `~/workspace/archive/legacy-flat-scraper/` on 2026-09-13
(same-filesystem renames; hashes, mtimes and inodes preserved and recorded in
`ksd-main/docs/data-paths.md`). Full account: `reports/hygiene-2026-09-13.md` §3.3 and
Addendum §A5.

### 0.3 Metadata diff — unlinked vs. correctly linked, same meeting

Same underlying file (`file_size_bytes` 160,414 and `char_count` 3,004 are identical — these
are two DB rows for one PDF), meeting DQHKYP542B0C, 2026-01-28 (query A3):

| Field | Correctly linked | Unlinked |
|---|---|---|
| `external_id` | `DQHKYP542B0C_DQHLJD5650A5_1210 - Proposed.pdf` | `DQHKYP542B0C_1210_-_Proposed.pdf` |
| `title` | `1210 - Proposed.pdf` | `1210_-_Proposed.pdf` |
| `file_path` | `.../2026-01-28-regular-meeting-6-30-p-m-/9-02-dqhljd5650a5-second-reading-and-approval-of-policy-1210-ann/1210 - Proposed.pdf` | `.../2026-01-28-regular-meeting-630-pm/1210_-_Proposed.pdf` |
| `meeting_id` | `DQHKYP542B0C` | `DQHKYP542B0C` *(same)* |
| **`agenda_item_id`** | **`DQHLJD5650A5`** | **NULL** |
| `committee_name` | `Regular Meeting` | `Regular Meeting` *(same)* |
| `source_url` | NULL | NULL |
| `processing_status` | `complete` | `complete` |
| `page_count` | 2 | 2 *(same)* |
| `metadata.item_name` | `Second Reading and Approval of Policy 1210 Annual Organization Meeting` | **NULL** |
| `metadata.item_order` | `9.02` | **NULL** |
| `metadata.meeting_slug` | `2026-01-28-regular-meeting-6-30-p-m-` | `2026-01-28-regular-meeting-630-pm` |
| `metadata.file_size_bytes` | 160,414 | 160,414 *(same)* |
| `metadata.char_count` | 3,004 | 3,004 *(same)* |
| `created_at` | 2026-02-23 23:21:07.230 | 2026-02-23 23:21:07.515 |

Three fields differ and they all differ for the same reason: `agenda_item_id`,
`metadata.item_name`, and `metadata.item_order` are populated only by the structured code
path. The flat path hardcodes the latter two to `None` and never sets the first.

**The 30 unlinked 2026-01-28 records are duplicates of already-correct records.** Matching on
meeting + underscore-normalized filename resolves 30 of 30 (query A2), e.g.:

```
DQHKYP542B0C_3418_-_Redline.pdf              -> DQHKYP542B0C_DQHLGV562A10_3418 - Redline.pdf
DQHKYP542B0C_Donations_Board_Review_1.28.2026.pdf -> DQHKYP542B0C_DQHKZM542B58_Donations Board Review 1.28.2026.pdf
DQHKYP542B0C_5000_-_Proposed_KSD.pdf         -> DQHKYP542B0C_DQHLGE561FC7_5000 - Proposed KSD.pdf
```

They survived the `ON CONFLICT DO NOTHING` idempotency guard because `external_id` — the
unique key — differs between the two layouts. Idempotency is keyed on a value that encodes
the scraper's layout, so it cannot detect cross-layout duplicates.

For 2026-02-04 the same holds (1 duplicate). For the two 2026-02-11 meetings, the 40 records
are **not** duplicates — they are the only copy in the corpus.

Net: of the 71, **31 are duplicate rows of correctly-linked records** and **40 are unique
content that exists in the corpus only in unlinked form**.

### 0.4 How "unprocessed" is defined

`documents.processing_status` — `character varying(32)`, default `'pending'`, indexed as
`idx_documents_status`. It is a status column, not a missing join or a null field.

Observed values corpus-wide (query A4):

| `document_type` | `complete` | `pending` | `deferred` | `failed` |
|---|---|---|---|---|
| agenda | 951 | — | — | — |
| agenda_item | 6,481 | **48** | — | — |
| attachment | 12,206 | — | 24 | 75 |
| email | 45 | — | — | — |
| transcript | 348 | — | — | — |

`boarddocs_loader` writes every new record as `'pending'`
(`loader.py:163`, `:199`, `:260`); `document_processor` is the stage that promotes it to
`complete` / `deferred` / `failed`.

**Verdict: never attempted.** Not stuck, not skipped, not failed.
- `processing_error` is NULL on all 48 — nothing raised.
- `content_raw` and `content_text` are non-null on all 48 — the loader did its job.
- `created_at` ≈ `updated_at` (2026-03-25 23:02:54 → 23:03:02) — the rows have never been
  touched since insert.
- No `deferred` and no `failed` records exist in the same batch, which is what a partial
  processor run would leave behind.

This is a pipeline run that stopped after stage 1. Per `CLAUDE.md`, the embedding cron is
deliberately disabled and `document_processor` is run by hand; the 2026-03-25 loader run was
simply never followed by stages 2 and 3.

---

## Phase 1 — Diagnosis

### 1.1 Where the link is made

**`boarddocs_loader/boarddocs_loader/loader.py:198`**

```python
att_doc = DocumentRecord(
    ...
    external_id=f"{meeting.meeting_id}_{item.item_id}_{att_name}",   # line 191
    ...
    agenda_item_id=item.item_id,                                      # line 198
```

**The key is the item subdirectory's own `item.json` → `itemId`.** There is no lookup, no
join, and no matching heuristic. `_process_structured` walks each subdirectory of the meeting
directory (`loader.py:124`), requires an `item.json` in it (`loader.py:128-132`), parses
`itemId` out of it (`parsers.py:136-147`), then attributes every attachment file found in
that same subdirectory (`loader.py:142`) to that item. The link is **positional** — an
attachment is linked to an item because the scraper placed the file inside that item's
folder.

**`boarddocs_loader/boarddocs_loader/loader.py:286-304`** — the flat path builds its
attachment record with **no `agenda_item_id` argument at all**, and explicitly writes
`"item_order": None, "item_name": None` (`loader.py:300-301`). This is deliberate: a flat
directory has no per-item folders, so there is no positional information to link on.

**`boarddocs_loader/boarddocs_loader/detector.py:13-44`** is the switch. It returns
`"structured"` if and only if the meeting directory contains at least one subdirectory
(`detector.py:24-27, 40-41`), otherwise `"flat"`. Which branch runs — and therefore whether
attachments get linked — is decided entirely by **directory shape on disk**.

**The linking code contains no defect.** Given a structured directory it links correctly;
given a flat directory it correctly declines to invent a link it has no basis for.

### 1.2 Offline reproduction

Both stored pages exist, so the failure reproduces without any network access. Running the
project's own `detect_format` and `parse_meeting_json` against one 2022 page and one 2026
page (command A7):

```
2022-05-11-regular-meeting-7-p-m-
  format=structured  meeting_id=CDMM32590DB4  date=2022-05-11
  categories=11  items=45
  -> loader branch: _process_structured (sets agenda_item_id)

2026-02-11-regular-meeting-630-pm
  format=flat  meeting_id=DQU45Y09F4B0  date=2026-02-11
  categories=0  items=0
  -> loader branch: _process_flat (agenda_item_id NEVER set)
```

**The divergence is at `detector.py:40` and nowhere else.** The 2022 page has 11 categories
and 45 items in its `meeting.json`; the 2026 flat page has zero of both, because the flat
schema does not carry a category/item tree. Everything downstream follows deterministically
from the branch taken. Feed the loader the *structured* 2026-01-28 directory and it produces
correctly linked records — it already did, on 2026-02-04.

### 1.3 Cause classification

**Primary — scraper substitution (71 records). Confirmed.**
The legacy Python flat-format scraper was run against 2026 meetings on 2026-02-22
07:02–07:05, producing flat output for meetings the structured scraper either had already
covered (2026-01-28, 2026-02-04) or had covered too early to be useful (2026-02-11, scraped
2026-02-10 pre-meeting). The loader ingested that flat output on 2026-02-23 23:21 alongside
the structured output. Ranked first because it is directly reproduced offline and accounts
for every one of the 71 records.

**Secondary — pipeline stage never run (48 records). Confirmed.**
Independent incident; see §0.4. Ranked second because it is confirmed but trivially
remediable and involves no defect.

**Tertiary — measurement error in the prior report (44 records). Confirmed.**
Email attachments counted as BoardDocs ingest failures, creating the appearance of a
degradation beginning in 2024. Ranked third because it changes the interpretation but not
the data.

**Contributing weakness — idempotency key encodes layout.** `external_id` is built as
`<meeting>_<item>_<file>` in the structured path and `<meeting>_<file>` in the flat path
(`loader.py:191` vs `loader.py:288`). The unique constraint
`documents_tenant_id_external_id_key` therefore cannot recognize that the two rows describe
one document. This is why re-scraping a meeting in a different layout silently doubles it
instead of being a no-op. This is a latent defect independent of the 2026 incident and will
recur on any future mixed-layout run.

**Ruled out, with evidence:**

| Hypothesis | Why ruled out |
|---|---|
| Site markup change | Retained 2026-02-11 `agenda.html` uses the same `Board.nsf/files/<ID>/$file/<name>` href form as the rest of the corpus; 21 such hrefs parse cleanly |
| Key format change | `meeting_id`/`agenda_item_id` are the same 12-char uppercase alphanumeric form in 2026 as 2019 |
| Scraper code regression | No `boarddocs-scraper` commit near 2026-02-22; the structured scraper produced normal output for 2026-01-14 and 2026-03-11/03-25 |
| Rate-limit truncation | Per-meeting attachment counts are complete against the agenda; no partial-run signature |
| Linking-code bug | Offline reproduction shows the linker works correctly on both inputs it is given |

### 1.4 `source_url` — derivability verdict

**Verdict: derivable for 12,207 of 12,305 attachments (99.2%) from data already in
PostgreSQL. No re-scrape required. No network access required to perform the derivation.**

**Correction to the task brief:** the pattern given —
`/wa/ksdwa/Board.nsf/pfiles/<FILEID>/$file/<original filename>` — does not match the stored
markup. Every href in the retained HTML and in the stored link metadata uses **`/files/`**,
not `/pfiles/`. The correct pattern is:

```
https://go.boarddocs.com/wa/ksdwa/Board.nsf/files/<FILEID>/$file/<url-encoded filename>
```

Using `/pfiles/` will produce 404s. This should be settled by the operator confirmation
requests listed below before any bulk backfill is written.

**Sample of 20 across years** (query A10) — two attachments per year, 2005–2026. The corpus
splits cleanly by era: 2005–2017 are all flat-format, 2018–2026 all structured-format. Each
era has a different but equally viable recovery route.

**Route 1 — structured attachments (5,637 records, 2018–2026).** The parent `agenda_item`
document's `metadata->'links'` already contains the full href, the BoardDocs file ID, and
the original filename. Example (query A11):

```json
[{"href": "https://go.boarddocs.com/wa/ksdwa/Board.nsf/files/DQHLKL569030/$file/1210%20-%20Proposed.pdf",
  "text": "1210 - Proposed.pdf (160 KB)", "order": "00001",
  "unique": "DQHLKL569030", "filename": "1210 - Proposed.pdf"}, ...]
```

Joining attachments to their parent item on `(meeting_id, agenda_item_id)` and matching
`documents.title = links->>'filename'` resolves **5,583 of 5,637 (99.0%)** (query A12). The
54 misses are filename-normalization edge cases, not missing data.

Five derived URLs (query A13):

| Meeting | Filename | File ID |
|---|---|---|
| 2019-06-12 | Executive Session 060519.pdf | `BCX52982A00C` |
| 2019-06-12 | 2b NO BusLane Bid Tab May 28 2019 Apparent Low.pdf | `BCX52X830513` |
| 2019-06-12 | 1b res #1560  NVES New in lieu of.pdf | `BD3TYQ790AEC` |
| 2019-06-12 | 4b Northwood Field Renovation Ph I Bid TAB 062018.pdf | `BCX5338332C8` |
| 2019-06-12 | Board Special Meeting Minutes 060519.pdf | `BCX526829E11` |

**Route 2 — flat attachments (6,668 records, 2005–2017 plus the 2026 incident).** The flat
loader path stores the meeting's entire `agenda.html` into the agenda document's
`content_raw` (`loader.py:227-231, 253`). Those hrefs carry the file IDs. `content_raw` is
present for **6,624 of 6,668** flat attachments' meetings (query A14) — the 44 misses are
exactly the email attachments, which have no BoardDocs URL by definition.

Extraction works (query A15) — 4 of 5 on a naive underscore→space normalization:

| Meeting | Stored filename | Recovered file ID |
|---|---|---|
| 2026-02-04 | Budget_Update_2.4.26.pdf | `DQVAHX27D3F1` |
| 2026-02-11 | 6630_-_Redline.pdf | `DR2UWZ7E158A` |
| 2026-02-11 | 1708CE~1.DOC.pdf | `DQU5A20F3A46` |
| 2026-02-11 | 6640_-_Proposed.pdf | `DR2UX37E1867` |
| 2026-02-11 | 1707_Certified_Signatures_Real_Estate_Transactions_2026.pdf | *(no match)* |

The one miss is instructive: the original filename is
`1707 Certified Signatures_Real Estate Transactions 2026.pdf` — it contains a genuine
underscore *and* spaces, so blanket underscore→space substitution produces the wrong string.
A backfill must match against the href's own decoded filename rather than reconstructing it
from the sanitized one. That is a matching refinement, not a data gap.

**Not derivable: 44 records** — the email attachments. Correct outcome; they have no
BoardDocs URL. `source_url` should stay NULL for these, or be set to a mailbox reference.

**Five requests for the operator to run by hand** to confirm the `/files/` pattern resolves
before any backfill is authorized. These are the only network operations this diagnostic
recommends, and none were performed:

```
curl -sSIL -o /dev/null -w '%{http_code} %{url_effective}\n' \
  'https://go.boarddocs.com/wa/ksdwa/Board.nsf/files/DQHLKL569030/$file/1210%20-%20Proposed.pdf'

curl -sSIL -o /dev/null -w '%{http_code} %{url_effective}\n' \
  'https://go.boarddocs.com/wa/ksdwa/Board.nsf/files/BCX52982A00C/$file/Executive%20Session%20060519.pdf'

curl -sSIL -o /dev/null -w '%{http_code} %{url_effective}\n' \
  'https://go.boarddocs.com/wa/ksdwa/Board.nsf/files/DQVAHX27D3F1/$file/Budget%20Update%202.4.26.pdf'

curl -sSIL -o /dev/null -w '%{http_code} %{url_effective}\n' \
  'https://go.boarddocs.com/wa/ksdwa/Board.nsf/files/DQU5A20F3A46/$file/1708CE~1.DOC.pdf'

curl -sSIL -o /dev/null -w '%{http_code} %{url_effective}\n' \
  'https://go.boarddocs.com/wa/ksdwa/Board.nsf/files/DR2UWZ7E158A/$file/6630%20-%20Redline.pdf'
```

Expected: `200`. If these return 404, retry one with `/pfiles/` substituted to test the
brief's pattern. **5 requests total.**

---

## Gaps in recorded provenance

Three things the corpus does not record, each of which made this diagnosis harder than it
needed to be:

1. **No scraper version or commit on any document row.** `documents` has no column for it and
   `metadata` does not carry it. The scraper identity had to be inferred from `meeting.json`
   key casing and directory-slug convention. Had both scrapers stamped their output, this
   would have been a one-query answer.
2. **No loader run ID.** Records from one run are identifiable only by clustering
   `created_at` to the second. Workable here; fragile in general.
3. **`file_path` prefixes are stale.** Every row points at `/home/donald/ksd_forensic/...`,
   which does not exist on this host. The tree the corpus was built from is now at
   `/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/...`. The working
   copy at `~/workspace/projects/ksd_forensic/boarddocs/data/` holds **729** meeting
   directories against the backup's **1,684**, and contains **no** `agenda.html` files at
   all. Anything that resolves `file_path` from the DB on this host will fail, and any
   re-run of the loader against the working copy would see less than half the corpus.

---

## Recommended changes (requires operator approval)

Nothing below has been executed. All of it is off by default until you say otherwise.

### R1 — Fix the linking gap for the 40 orphaned 2026-02-11 attachments *(highest value)*

These 40 records are the only copy of that content in the corpus, and 2026-02-11 is a regular
board meeting whose minutes, vouchers, personnel report, and policy readings are all in this
set. They are currently unreachable by any agenda-item-scoped query.

The retained `agenda.html` for both 2026-02-11 meetings contains the agenda structure,
including item anchors (`Board.nsf/goto?open&id=<ITEMID>`) adjacent to the file hrefs. A
backfill can recover `agenda_item_id` by parsing that stored HTML — **no network access
required**, since the HTML is already in `documents.content_raw`.

Effort: one script, roughly 100 lines, plus verification. **0 network requests.**

### R2 — Deduplicate the 31 cross-layout duplicate rows

The 31 flat-format rows for 2026-01-28 and 2026-02-04 duplicate correctly-linked structured
rows. They inflate retrieval results, waste embedding slots, and will surface twice in RAG
citations for the same document.

Recommended action: delete the 31 flat rows (cascade removes their chunks) and remove the
corresponding Qdrant points. **Verify chunk and point counts before and after.** Do not run
this until R1 is done and verified, since R1's HTML parsing uses the same records.

Effort: one query plus a Qdrant deletion. **0 network requests.**

### R3 — Make the loader layout-aware so this cannot recur

Two changes to `boarddocs_loader`:

- **Content-based idempotency.** Add a `content_hash` column (SHA-256 of the file bytes) and
  a unique constraint on `(tenant_id, meeting_id, content_hash)` alongside the existing
  `external_id` key. A re-scrape in a different layout then becomes a genuine no-op instead
  of a silent duplicate. This is the change that actually prevents recurrence.
- **Flat-path link recovery.** In `_process_flat` (`loader.py:210-304`), parse the
  `agenda.html` that is already being read at `loader.py:227-231` for item anchors and file
  hrefs, and populate `agenda_item_id` and `source_url` when they can be resolved. Today that
  HTML is stored and then ignored.

Also worth adding: a `scraper_version` field written from `meeting.json`, and a loader run ID.

Effort: moderate; schema migration plus loader changes plus tests. **0 network requests.**

### R4 — Backfill `source_url` for 12,207 attachments

Two passes as described in §1.4 — Route 1 for structured, Route 2 for flat. Gate this on the
5 confirmation requests above returning `200`; if they 404, the pattern is wrong and the
backfill would write 12,207 broken links into the citation layer, which is worse than NULL.

Match on the href's own decoded filename, not on a reconstruction of the sanitized one.

Effort: one script, two passes, plus a sampled verification. **5 network requests to confirm
the pattern; 0 for the backfill itself.** Optionally, a sampled HEAD check of ~100 backfilled
URLs afterward would validate the result — **~100 additional requests**, operator's call.

### R5 — Run `document_processor` and `embedding_pipeline` for the 48 pending agenda items

Standard pipeline stages 2 and 3, on records that already have their content loaded. Nothing
is broken; the run just needs to happen. Re-enable the embedding cron afterward, or add a
post-load check that flags `pending` records older than a day so a half-finished pipeline run
is visible rather than silent.

Effort: two existing commands. **0 network requests.**

### R6 — Correct the Phase 0 coverage report

`reports/boarddocs-coverage-2026-09-12.md` §0.6 attributes 26 (2024) and 16 (2025) unlinked
attachments to structured-scraper degradation. Those are email attachments. The corrected
table:

| Year | BoardDocs unlinked | Email attachments | Linked |
|---|---|---|---|
| 2024 | **0** | 26 | 657 |
| 2025 | **0** | 16 | 868 |
| 2026 | **71** | 2 | 155 |

The report's conclusion that "the structured scraper is degrading on recent meetings" and its
remediation item for "~115 records across 2024–2026" should be revised. Per the report's own
append-only convention, publish this as a correction rather than editing the original.

### Do 2024–2025 need the same treatment?

**No.** Zero BoardDocs attachments in 2024 or 2025 are unlinked. The 42 records in question
are email attachments that are correctly unlinked and require no action. The remediation
scope is **2026 only: 71 attachment records (40 to link, 31 to delete) and 48 agenda items to
process** — 119 records, not the 121 in the task framing, because 2 of the 73 are email.

### Suggested order

R5 → R1 → R2 → R4 (after the 5 confirmations) → R3 → R6.

R5 first because it is zero-risk and unblocks the current corpus. R3 last because it is the
largest change and the others give it better test fixtures.

---

## Stop-rule compliance

- No scraper was modified or executed.
- No records were patched, inserted, updated, or deleted. Every database statement was a
  `SELECT`.
- No network requests were made to BoardDocs or anywhere else. The 5 confirmation requests in
  §1.4 are listed for the operator, not executed.
- `.env` was not modified. Credentials were not printed.
- No hook was bypassed or worked around. No guard gap was found or used.

**One note on credentials, not a guard gap.** The `.env` `POSTGRES_PASSWORD` fails
authentication against the container from the host (`FATAL: password authentication failed
for user "boarddocs"`). All queries in this report were run via
`podman exec boarddocs-postgres psql -U boarddocs -d boarddocs`, which uses the container's
local trust/peer path. This is a legitimate read path, but it means the credential in `.env`
is stale or is expected to be injected at runtime from elsewhere. Worth resolving so tooling
on the host works without container exec.

**One security observation, unrelated to this diagnostic.** `boarddocs-postgres` publishes
`0.0.0.0:5432->5432/tcp` — the database is bound to all interfaces, not loopback. Every other
service on the host (`qdrant`, `vllm`, `open-webui`) correctly binds `127.0.0.1`. The project
`CLAUDE.md` requires loopback-only binding. If the host firewall is not blocking 5432, the
corpus is reachable from the LAN. Recommend checking `firewall-cmd --list-ports` and
correcting the quadlet's `PublishPort` to `127.0.0.1:5432:5432`. Flagged only — not changed.

---

## Appendix — every query and command

All `psql` invocations were run as:
`podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -P pager=off [flags] -c "<SQL>"`

Environment discovery (before the container-exec path was adopted):

```bash
podman ps --format '{{.Names}} | {{.Image}} | {{.Status}} | {{.Ports}}'
podman inspect boarddocs-postgres --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i -v password
# failed host-side attempt, recorded for completeness:
cd ~/workspace/projects/ksd-main && set -a && source .env && set +a && \
  PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" \
  -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt"
```

Schema:

```sql
\dt
\d documents
```

**A1 — unlinked attachments clustered by meeting, 2024+**

```sql
SELECT meeting_date, meeting_id, count(*) AS unlinked,
       min(created_at)::timestamp(0) AS first_ingest,
       max(created_at)::timestamp(0) AS last_ingest
FROM documents
WHERE document_type='attachment' AND agenda_item_id IS NULL AND meeting_date >= '2024-01-01'
GROUP BY 1,2 ORDER BY 1;
```

Full listing of the 73 (external_id, title, file_path, created_at):

```sql
SELECT meeting_date, external_id, title, file_path, created_at::timestamp(0)
FROM documents
WHERE document_type='attachment' AND agenda_item_id IS NULL AND meeting_date >= '2026-01-01'
ORDER BY meeting_date, external_id;
```

BoardDocs-only subset (excludes email):

```sql
SELECT meeting_date, meeting_id, external_id
FROM documents
WHERE document_type='attachment' AND agenda_item_id IS NULL
  AND meeting_date >= '2026-01-01' AND external_id NOT LIKE 'email_%'
ORDER BY meeting_date, external_id;
```

Per-meeting linked/unlinked split:

```sql
SELECT meeting_date, meeting_id,
       count(*) FILTER (WHERE agenda_item_id IS NOT NULL) linked,
       count(*) FILTER (WHERE agenda_item_id IS NULL) unlinked
FROM documents
WHERE document_type='attachment' AND meeting_date>='2026-01-01'
  AND external_id NOT LIKE 'email_%'
GROUP BY 1,2 ORDER BY 1;
```

Source split (email vs BoardDocs), 2024+:

```sql
SELECT extract(year from meeting_date)::int yr,
       count(*) FILTER (WHERE external_id LIKE 'email_%') email_src,
       count(*) FILTER (WHERE external_id NOT LIKE 'email_%') boarddocs_src
FROM documents
WHERE document_type='attachment' AND agenda_item_id IS NULL AND meeting_date>='2024-01-01'
GROUP BY 1 ORDER BY 1;
```

**A2 — duplicate detection, unlinked vs linked**

```sql
WITH u AS (
  SELECT id, external_id,
         regexp_replace(split_part(external_id,'_',1)||'|'||
           replace(substr(external_id, position('_' in external_id)+1),'_',' '),'  ',' ') k
  FROM documents
  WHERE document_type='attachment' AND agenda_item_id IS NULL
    AND meeting_date>='2026-01-01' AND external_id NOT LIKE 'email_%'),
 l AS (
  SELECT meeting_id||'|'||substr(external_id, length(meeting_id)+length(agenda_item_id)+3) k2,
         external_id
  FROM documents
  WHERE document_type='attachment' AND agenda_item_id IS NOT NULL
    AND meeting_date>='2026-01-01')
SELECT u.external_id, l.external_id FROM u LEFT JOIN l ON u.k=l.k2;
```

**A3 — full metadata dump, matched pair**

```sql
SELECT external_id, title, file_path, source_url, meeting_id, agenda_item_id,
       committee_name, processing_status, page_count, metadata, created_at, updated_at
FROM documents
WHERE external_id IN ('DQHKYP542B0C_1210_-_Proposed.pdf',
                      'DQHKYP542B0C_Board_Presentation_Winter_2026_.pdf',
                      'DQU45Y09F4B0_ASB_Vouchers_02-11-26.pdf')
   OR external_id LIKE 'DQHKYP542B0C_DQ%1210 - Proposed%';
```

**A4 — processing status distribution**

```sql
SELECT document_type, processing_status, count(*)
FROM documents GROUP BY 1,2 ORDER BY 1,2;

SELECT extract(year from meeting_date) yr,
       count(*) FILTER (WHERE agenda_item_id IS NULL) unlinked,
       count(*) FILTER (WHERE agenda_item_id IS NOT NULL) linked
FROM documents WHERE document_type='attachment' AND meeting_date >= '2023-01-01'
GROUP BY 1 ORDER BY 1;
```

**A5 — the 48 pending agenda items**

```sql
SELECT meeting_date, meeting_id, count(*),
       min(created_at)::timestamp(0), max(updated_at)::timestamp(0),
       count(*) FILTER (WHERE content_text IS NULL) null_text,
       count(*) FILTER (WHERE content_raw IS NULL) null_raw,
       coalesce(string_agg(DISTINCT coalesce(processing_error,'(none)'),'; '),'')
FROM documents WHERE document_type='agenda_item' AND processing_status='pending'
GROUP BY 1,2 ORDER BY 1;
```

**A6 — locating the two scrape trees**

```bash
ls -la ~/workspace/projects/ksd-main/
ls ~/workspace/projects/ksd_forensic/boarddocs/data/ | grep '^2026'
find /home/donald -maxdepth 8 -type d -name "2026-02-11-regular-meeting*" 2>/dev/null
find /home/donald -maxdepth 8 -type d -name "*-630-pm" 2>/dev/null | head -20

B=/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data
ls "$B/2026-02-11-regular-meeting-6-30-p-m-/"
ls "$B/2026-02-11-regular-meeting-630-pm/"
stat -c '%y %n' "$B/2026-02-11-regular-meeting-6-30-p-m-" \
                "$B/2026-02-11-regular-meeting-630-pm" \
                "$B/2026-01-28-regular-meeting-6-30-p-m-" \
                "$B/2026-01-28-regular-meeting-630-pm"

cd ~/workspace/projects/ksd_forensic/boarddocs-scraper && git log --oneline -12
```

**A7 — offline reproduction with project code**

```bash
cd ~/workspace/projects/ksd-main/boarddocs_loader
B=/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data
python3 -c "
import sys, json; sys.path.insert(0,'.')
from pathlib import Path
from boarddocs_loader.detector import detect_format
from boarddocs_loader.parsers import parse_meeting_json
for d in ['2026-02-11-regular-meeting-630-pm','2026-01-28-regular-meeting-630-pm',
          '2026-01-28-regular-meeting-6-30-p-m-','2022-05-11-regular-meeting-7-p-m-']:
    p=Path('$B')/d
    fmt=detect_format(p)
    raw=json.loads((p/'meeting.json').read_text())
    m=parse_meeting_json(p/'meeting.json', fmt)
    print(d, fmt, sorted(raw.keys()),
          raw.get('scrapedAt') or raw.get('scraped_at'),
          len(m.categories), sum(len(c.items) for c in m.categories))
"
```

**A8 — structured 2026-02-11 directory yields nothing**

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
from pathlib import Path
from boarddocs_loader.detector import detect_format
p=Path('\$B/2026-02-11-regular-meeting-6-30-p-m-')
print('fmt=',detect_format(p))
subs=[x for x in p.iterdir() if x.is_dir()]
print('subdirs=',[x.name for x in subs])
print('item.json present:',[(x/'item.json').exists() for x in subs])
print('pdfs at top level:',len([x for x in p.iterdir() if x.suffix.lower()=='.pdf']))
"
```

**A9 — agenda.html retention**

```bash
B=/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data
echo "total meeting dirs: $(ls -d $B/*/ | wc -l)"
echo "with agenda.html: $(ls $B/*/agenda.html 2>/dev/null | wc -l)"
for y in 2005 2010 2015 2018 2020 2022 2024 2025 2026; do
  echo "$y: dirs=$(ls -d $B/$y-*/ 2>/dev/null|wc -l) html=$(ls $B/$y-*/agenda.html 2>/dev/null|wc -l)"
done
L=/home/donald/workspace/projects/ksd_forensic/boarddocs/data
echo "live: total=$(ls -d $L/*/ | wc -l) html=$(ls $L/*/agenda.html 2>/dev/null | wc -l)"

# grep for pfiles vs files in retained HTML
F="$B/2026-02-11-regular-meeting-630-pm/agenda.html"
grep -o 'pfiles/[^"'"'"']*' "$F" | head        # 0 results
grep -oE 'href="[^"]{0,120}"' "$F" | head -20  # all use /files/
```

**A10 — sample of 20 attachments across years**

```sql
WITH s AS (
 SELECT d.id, d.title, d.meeting_date, d.agenda_item_id,
   CASE WHEN d.agenda_item_id IS NULL THEN 'flat' ELSE 'structured' END fmt,
   row_number() OVER (PARTITION BY extract(year from d.meeting_date) ORDER BY d.id) rn
 FROM documents d WHERE d.document_type='attachment')
SELECT extract(year from meeting_date)::int yr, fmt, count(*)
FROM s WHERE rn<=2 GROUP BY 1,2 ORDER BY 1;
```

**A11 — file IDs present in agenda_item link metadata**

```sql
SELECT external_id, metadata->'links' AS links,
  (SELECT count(*) FROM regexp_matches(coalesce(content_raw,''),
     'Board\.nsf/files/[A-Z0-9]+','g')) raw_file_refs
FROM documents
WHERE document_type='agenda_item' AND external_id='DQHKYP542B0C_DQHLJD5650A5';
```

**A12 — structured derivability, corpus-wide**

```sql
WITH att AS (
  SELECT id, title, meeting_id, agenda_item_id, meeting_date
  FROM documents WHERE document_type='attachment' AND agenda_item_id IS NOT NULL),
lnk AS (
  SELECT meeting_id, agenda_item_id, l->>'filename' fn, l->>'unique' uniq
  FROM documents ai, jsonb_array_elements(ai.metadata->'links') l
  WHERE ai.document_type='agenda_item')
SELECT count(*) AS structured_attachments,
       count(lnk.uniq) AS with_matching_file_id,
       round(100.0*count(lnk.uniq)/count(*),1) AS pct
FROM att LEFT JOIN lnk
  ON att.meeting_id=lnk.meeting_id AND att.agenda_item_id=lnk.agenda_item_id
 AND att.title=lnk.fn;
```

**A13 — five derived structured file IDs**

```sql
WITH lnk AS (
  SELECT meeting_id, agenda_item_id, l->>'filename' fn, l->>'unique' uniq
  FROM documents ai, jsonb_array_elements(ai.metadata->'links') l
  WHERE ai.document_type='agenda_item')
SELECT d.meeting_date, d.title, lnk.uniq AS file_id
FROM documents d
JOIN lnk ON d.meeting_id=lnk.meeting_id AND d.agenda_item_id=lnk.agenda_item_id
        AND d.title=lnk.fn
WHERE d.document_type='attachment'
  AND d.meeting_date IN ('2019-06-12','2021-04-28','2023-05-10','2025-06-11','2026-01-14')
ORDER BY d.meeting_date LIMIT 5;
```

**A14 — agenda.html availability for flat attachments**

```sql
WITH att AS (
  SELECT meeting_id, meeting_date FROM documents
  WHERE document_type='attachment' AND agenda_item_id IS NULL)
SELECT CASE WHEN ag.content_raw IS NULL THEN 'no agenda.html retained'
            ELSE 'agenda.html in DB' END AS state,
       count(*) FILTER (WHERE att.meeting_date < '2018-01-01') pre2018,
       count(*) FILTER (WHERE att.meeting_date >= '2018-01-01') post2018,
       count(*) total
FROM att LEFT JOIN documents ag
  ON ag.meeting_id=att.meeting_id AND ag.document_type='agenda'
GROUP BY 1;

-- file-ID reference counts in stored agenda HTML for the four affected meetings
SELECT external_id, length(content_raw),
  (SELECT count(*) FROM regexp_matches(content_raw,'Board\.nsf/files/[A-Z0-9]+','g')) AS file_id_refs
FROM documents
WHERE document_type='agenda'
  AND meeting_id IN ('DQU45Y09F4B0','DQU47R0A3706','DQHKYP542B0C','DQU2PT0331CA');
```

**A15 — flat file-ID recovery from stored agenda HTML**

```sql
WITH att AS (
  SELECT d.title, d.meeting_id, d.meeting_date,
         replace(regexp_replace(d.title,'\.pdf$','','i'),'_',' ') AS want,
         ag.content_raw
  FROM documents d
  JOIN documents ag ON ag.meeting_id=d.meeting_id AND ag.document_type='agenda'
  WHERE d.document_type='attachment' AND d.agenda_item_id IS NULL
    AND d.meeting_date>='2026-02-01'
  LIMIT 5)
SELECT meeting_date, title,
  (SELECT m[1] FROM regexp_matches(content_raw,
     'Board\.nsf/files/([A-Z0-9]+)/\$file/'||
     regexp_replace(replace(want,' ','%20'),'([().$*+?\[\]^|\\])','\\\1','g'),'g') m
   LIMIT 1) AS file_id
FROM att;
```

**Source files read (no modifications):**

- `boarddocs_loader/boarddocs_loader/loader.py` (353 lines)
- `boarddocs_loader/boarddocs_loader/detector.py` (52 lines)
- `boarddocs_loader/boarddocs_loader/parsers.py` (292 lines)
- `CLAUDE.md`
- `reports/boarddocs-coverage-2026-09-12.md`
