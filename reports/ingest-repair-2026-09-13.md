# 2026 Ingest Incident — Repair Preparation (Part 1)

**Date:** 2026-09-13 (Part 1) / 2026-09-14 (Part 2)
**Scope:** 2026 only — 119 records (48 unprocessed agenda items, 40 orphan attachments, 31 duplicates).
**Status:** **Part 1 complete. Part 2 executed and verified — steps a, b, d, c all approved and run.**
**Predecessors:** `reports/ingest-degradation-2026-09-13.md`, `docs/session-logs/session-debrief-2026-09-13-ingest-degradation-diagnostic.md`

> **Part 2 results are in [§Part 2 — Execution record](#part-2--execution-record) at the end of this
> report.** Everything before that section is the Part 1 preparation as submitted for approval, left
> unedited so the approved plan and the executed result can be compared.

---

## Executive summary

Part 1 is done and the repair is ready to run. Three things are worth your attention
before you approve anything.

**1. The recurrence question is closed, and the answer is better than expected.** No
scheduler anywhere on Smeltor can run the legacy flat scraper — and the legacy scraper's
source code is not on this host at all. There is nothing to re-trigger. **No STOP
condition applies.**

**2. Two claims in the diagnostic were wrong, and correcting them changed the outcome for
the better.** The diagnostic said `agenda_item_id` could be recovered by parsing item
anchors out of the retained `agenda.html`. It cannot — that markup contains no item IDs
at all. But a source the diagnostic dismissed as empty (the aborted 2026-02-10 structured
scrape) turned out to contain a complete `item.json` tree. Using it, **all 40 orphans
resolve exactly**, not the partial recovery the diagnostic anticipated.

**3. One finding changes the risk profile of the delete.** The `facts.*` tables added in
the recent minutes work hold **7 foreign keys to `documents` that are not
`ON DELETE CASCADE`**. 43 fact rows reference the target set. Every one of them points at
the **orphan** set, and **none** at the delete set — so the delete is safe, and the
orphans are confirmed load-bearing (the 2026-02-11 minutes are already parsed into the
fact tables). Had those references pointed the other way, `delete_31.sql` would have
failed mid-run.

All three counts re-derived from live data match the diagnostic exactly: **48 / 40 / 31**.

| Step | Result |
|---|---|
| 1. Recurrence check | No scheduler, no scraper source on host. **Closed.** |
| 2. Re-derive sets | 48 / 40 / 31 — **exact match**, Δ = 0 |
| 3. Backup + restore proof | 119 docs, 694 chunks, 40 fact rows; **content MD5 identical** |
| 4. Duplicate proof | **31/31 byte-identical** by SHA-256; 0 excluded |
| 5. Orphan link derivation | **40/40 exact**, two independent methods agreeing; 0 unlinked |
| 6. Three scripts | Written, **unexecuted**, idempotent |
| 7. Expected post-state | unlinked 0, unprocessed 0, duplicate pairs 0 |

---

## Step 1 — Recurrence check

**Verdict: no scheduled job can run the legacy scraper. No STOP.**

| Surface | Finding |
|---|---|
| User crontab (`donald`) | `no crontab for donald` |
| Root crontab | `no crontab for root` — *operator-run, see below* |
| `/var/spool/cron/` | empty (`total 0`) — *operator-run* |
| `/etc/crontab`, `/etc/cron.d/` | stock only (`0hourly` → `run-parts /etc/cron.hourly`) |
| `/etc/cron.{daily,weekly,monthly}` | **all empty** |
| `/etc/cron.hourly/` | `0anacron` only |
| User systemd timers | 6 — 4 are Podman container healthchecks, plus `systemd-tmpfiles-clean`, `grub-boot-success` |
| System systemd timers | 8 — all stock OS maintenance (`fwupd`, `dnf-makecache`, `logrotate`, `fstrim`, `raid-check`, …) |
| `~/.config/systemd/user/` | **empty directory** |
| Unit files mentioning scraper/loader | none (8 grep hits were all bootloader false positives) |
| Scheduler config in `ksd-main` | none — repo `systemd/` holds 4 service units, no `.timer`, no `OnCalendar` |
| Scheduler config in `ksd_forensic` | none — all grep hits were agenda *content* containing the word "schedule" |

The `sudo`-requiring checks were blocked by the `block-dangerous-commands.sh` hook. Per the
rules the hook was **not bypassed**; the operator ran them and supplied the output, quoted
in Appendix A2.

**The legacy flat scraper is not on this host.** A host-wide search for the script that
emits the flat format's signature key `files_found` returned only files that *read* it
(the loader and its tests) plus unrelated Python stdlib matches. The candidate scripts in
`ksd_forensic/scripts/` were each checked against the flat output signature and none
matches:

| Script | Output shape | Match? |
|---|---|---|
| `KentScrapes.py` | `{date}_{name}/` with `agenda.txt` + `metadata.json` + `files/` | no |
| `kent_scraper.py` | writes to `/opt/boarddocs/`, JSON+CSV only | no |
| `boarddocs_update.py` | `_meeting_slug` → trailing-dash slug, per-item subdirs | no (structured) |
| `boarddocs_api_scrape.py` | `{order}-{id}-{slug}/` per-item subdirs | no (structured) |

Ground truth for comparison — the flat `meeting.json` that caused the incident:

```json
{ "meeting_id": "DQU45Y09F4B0", "date": "20260211",
  "slug": "2026-02-11-regular-meeting-630-pm", "files_found": 21,
  "scraped_at": "2026-02-22T07:02:41.345627",
  "source_url": "https://go.boarddocs.com/wa/ksdwa/Board.nsf/Public" }
```

### What ran the 2026-02-22 scrape: **not determinable**

Every forensic source is either outside its retention window or begins after the event.
This is a definitive negative, not an unfinished search:

| Source | Window | Result |
|---|---|---|
| `journalctl` (all boots) | oldest entry **2026-04-08** | 2026-02-22 outside retention |
| `ai_activity_log` | 2026-02-27 → 2026-03-03 | begins **5 days after** the scrape |
| `corpus_updates` | — | **0 rows** |
| `~/.bash_history` | — | **0** scraper invocations, 0 entries from February |
| Session recordings | single session, **2026-02-27** | a monitoring self-test |
| Scraper logs on disk | — | none exist |

The reason is visible in the evidence itself: the session-logging and `ai_activity_log`
infrastructure was commissioned on **2026-02-27**, which is exactly where `ai_activity_log`
starts. The observability that would have recorded the incident was installed five days
after it happened.

**Consequence for the repair: none.** The absence of both a scheduler and the scraper
binary means there is nothing to disable. Open item 3 from the prior debrief — "decide
which scraper is authoritative and stop running the other" — is answered by default on
this host, though it remains a live question wherever that scraper actually lives.

---

## Step 2 — Re-derived counts

All three sets re-derived from live data. **Every count matches. Δ = 0. No STOP.**

| Set | Diagnostic | Re-derived | Δ |
|---|---|---|---|
| Unprocessed agenda items | 48 | **48** | 0 |
| Unique orphan attachments | 40 | **40** | 0 |
| Duplicate rows | 31 | **31** | 0 |
| **Total** | **119** | **119** | **0** |

Per meeting:

| Meeting date | `meeting_id` | Unlinked | → duplicates | → orphans |
|---|---|---|---|---|
| 2026-01-28 | `DQHKYP542B0C` | 30 | 30 | 0 |
| 2026-02-04 | `DQU2PT0331CA` | 1 | 1 | 0 |
| 2026-02-11 | `DQU45Y09F4B0` | 21 | 0 | 21 |
| 2026-02-11 | `DQU47R0A3706` | 19 | 0 | 19 |

| Meeting date | `meeting_id` | Pending agenda items | Loaded |
|---|---|---|---|
| 2026-03-25 | `DS4MSK5CA5C5` | 42 | 2026-03-25 23:02:54 |
| 2026-03-25 | `DS4MVC5CCBE6` | 2 | 2026-03-25 23:03:01 |
| 2026-03-25 | `DSCM9K5A287D` | 4 | 2026-03-25 23:03:02 |

All 48 have `content_raw` and `content_text` populated, `processing_error` NULL, and no
`failed`/`deferred` siblings — consistent with the diagnostic's "never attempted" verdict.
They are also the **only** pending documents in the entire corpus (Appendix A9), which is
what makes a type-scoped Stage 2 run exactly scoped.

### A correction to the matching rule

The diagnostic's duplicate matcher normalized filenames with `replace(title,' ','_')` —
one underscore per space. The flat scraper actually collapses a **run** of whitespace to a
**single** underscore. BoardDocs serves one file on the 2026-02-11 agenda as
`KE  - Install New Fence Along Meeker Street - RFQ and Quote.pdf` (double space), which the
flat scraper wrote as `KE_-_Install_...`. Under the naive rule those two records silently
fail to match.

Corrected to `regexp_replace(title,'\s+','_','g')` throughout. The 31/40 split is
**unchanged** (verified, Appendix A6) — but the correction is what allowed the last 2
orphans to resolve in Step 5. One 2026 linked title (`5010P  - Final.pdf`) contains a
whitespace run and would have been mis-normalized under the old rule.

---

## Step 3 — Backup

Exported to `backups/ingest-repair-2026-09-13/` (gitignored — it holds corpus content).

| File | Rows |
|---|---|
| `documents.jsonl` | 119 |
| `chunks.jsonl` | 694 |
| `document_pages.jsonl` | 0 |
| `facts_meeting.jsonl` | 3 |
| `facts_attendance.jsonl` | 15 |
| `facts_motion.jsonl` | 19 |
| `facts_vote.jsonl` | 0 |
| `facts_executive_session.jsonl` | 0 |
| `facts_minutes_parse_log.jsonl` | 3 |
| `qdrant_points_dup31.jsonl` | 110 |
| `duplicate_proof.json` | 31 pairs + hashes |
| `link_derivation.json` | 40 mappings + per-record evidence |

Full rows (`to_jsonb`), no column omitted. SHA-256 manifest at `manifest.sha256`, row
counts recomputed from the files themselves at `row_counts.txt`.

### Restore verification

Loaded into a throwaway schema, compared, then dropped — **counts and content both match**:

```
=== Row counts: restored vs live ===
    tbl    | restored | live | verdict
-----------+----------+------+---------
 documents |      119 |  119 | MATCH
 chunks    |      694 |  694 | MATCH

=== Content checksum: restored vs live (not just counts) ===
    tbl    |           restored_md5           |             live_md5
-----------+----------------------------------+----------------------------------
 documents | e6a02dff2dacbd7fd1228d1dfbfbceca | e6a02dff2dacbd7fd1228d1dfbfbceca
 chunks    | e8410eff49bb5e84a78ca0710e8699a9 | e8410eff49bb5e84a78ca0710e8699a9

=== Dropping scratch schema (and nothing else) ===
 scratch_schemas_remaining
---------------------------
                         0
```

The MD5 is over every column of every row, ordered — not just a row count. The scratch
schema used `CREATE TABLE ... (LIKE ...)`, which copies columns but **no foreign keys**, so
the restore copy could not reference or endanger anything in `public.*` or `facts.*`.

> **Note on the export format.** The first restore attempt failed: `COPY`'s default *text*
> format treats backslashes as escapes, so a literal `\n` inside a JSON string was turned
> back into a real newline and broke the JSON. Switched to `FORMAT csv` with control-character
> delimiters, which does no backslash processing. Worth knowing for any future restore.

### Dependency discovery — the finding that changes the delete

`documents` is referenced by **11 foreign keys**. Four are the familiar cascading ones.
The other **seven are `facts.*` keys with no `ON DELETE` action**, added by the recent
minutes fact-table work:

| Referencing column | Cascades? | Rows referencing the target set |
|---|---|---|
| `chunks.document_id` | `ON DELETE CASCADE` | 694 |
| `document_pages.document_id` | `ON DELETE CASCADE` | 0 |
| `facts.meeting.minutes_document_id` | **no** | 3 |
| `facts.meeting.locator_document_id` | **no** | 3 |
| `facts.attendance.locator_document_id` | **no** | 15 |
| `facts.motion.locator_document_id` | **no** | 19 |
| `facts.vote.locator_document_id` | **no** | 0 |
| `facts.executive_session.locator_document_id` | **no** | 0 |
| `facts.minutes_parse_log.document_id` | **no** | 3 |

Broken down by bucket, **all 43 fact references point at the 40 orphans and none at the
31 duplicates**:

| Bucket | `facts.attendance` | `facts.motion` | `facts.meeting` | `facts.minutes_parse_log` |
|---|---|---|---|---|
| `orphan40` | 15 | 19 | 6 | 3 |
| `dup31` | **0** | **0** | **0** | **0** |
| `pending48` | 0 | 0 | 0 | 0 |

Two consequences:

- **`delete_31.sql` is safe.** No non-cascading FK can block it. `delete_31.sql` still
  re-checks this at runtime rather than trusting this snapshot.
- **The 40 orphans are confirmed load-bearing.** The 2026-02-11 minutes have already been
  parsed into the fact tables, so 43 fact rows currently cite documents that no
  agenda-scoped query can reach. Linking them is the highest-value step here.

---

## Step 4 — Duplicate proof (31/31)

**All 31 pairs are byte-identical. Zero excluded.**

Each pair was proven by SHA-256 over the **actual file bytes** of both copies, not by
`file_size_bytes` or `char_count` (which is all the diagnostic compared). Stored
`file_path` values are stale — they point at `/home/donald/ksd_forensic/...`, which does
not exist — so paths were remapped on read to the framework backup tree. **The stale values
in the database were not modified.**

```
pairs returned by matcher: 31
byte-identical (safe to delete): 31
NOT proven identical (excluded): 0
...
kept records carrying an agenda_item_id: 31/31
```

Sample (full listing in Appendix A7, complete hashes in `duplicate_proof.json`):

| Flat row (to delete) | SHA-256 (first 16) | Kept `agenda_item_id` |
|---|---|---|
| `DQHKYP542B0C_1210_-_Proposed.pdf` | `2cd53ca1527fed69` | `DQHLJD5650A5` |
| `DQHKYP542B0C_3418_-_Redline.pdf` | `ace0470ab6760580` | `DQHLGV562A10` |
| `DQHKYP542B0C_Donations_Board_Review_1.28.2026.pdf` | `a9251c55630e7240` | `DQHKZM542B58` |
| `DQU2PT0331CA_Budget_Update_2.4.26.pdf` | `17c1fe182c0cf860` | `DQU2Q30331E4` |

The kept record is the linked one in all 31 cases — asserted explicitly, not merely implied
by the matcher.

---

## Step 5 — Orphan link derivation (40/40)

**All 40 orphans resolve to an exact `agenda_item_id`. Zero remain unlinked.**

### The diagnostic's proposed method does not work

R1 proposed parsing item anchors (`Board.nsf/goto?open&id=<ITEMID>`) out of the retained
`agenda.html`. That markup is BoardDocs' `PRINT-AgendaDetailed` view, and it contains:

- `goto?open&id=` anchors: **0**
- agenda-item IDs in any form: **0**
- goal IDs (`CWAP…`): 4 distinct
- file IDs (`Board.nsf/files/…`): 21 and 19 — matching the orphan counts exactly

The `unique=` attribute adjacent to each file link is the **file** ID, not the item ID.
Worse, the two 2026-02-11 meetings have **zero `agenda_item` rows in the corpus** — so even
a recovered ID would have had nothing to point at.

### A better source the diagnostic dismissed

The diagnostic stated the aborted 2026-02-10 structured scrape "produced only `.txt` files
and a single `archive_*` subdirectory containing no `item.json`". **That is incorrect.**
The `archive_*` tree contains a full per-item hierarchy — 29 `item.json` for the regular
meeting, 4 for the work session — each with exactly what is needed:

```json
{ "itemId": "DR39NK23D246", "itemOrder": "10.02",
  "itemName": "December 2025 Financial Statement",
  "links": [ { "unique": "DR39UK24B237",
               "filename": "Financial Statement_December 2025_FINAL.pdf" } ] }
```

### Method and confidence

Two independent methods were computed per record and **required to agree**:

1. **file_id** — flat filename → href in retained `agenda.html` → BoardDocs file ID →
   `item.json` `links[].unique` → `itemId`. Joins on an opaque ID, not on text.
2. **filename** — flat filename → sanitized `item.json` `links[].filename` → `itemId`.

A record is `exact` only when both resolve, agree, and the match is unique.

| Meeting | Orphans | Resolved `exact` | Stays unlinked |
|---|---|---|---|
| `DQU45Y09F4B0` (regular) | 21 | **21** | 0 |
| `DQU47R0A3706` (work session) | 19 | **19** | 0 |
| **Total** | **40** | **40** | **0** |

Cross-check: the set of attachments in the retained `agenda.html` and the set known to the
2026-02-10 archive scrape are **identical in both directions** — nothing in the agenda is
missing from the archive, and nothing in the archive was dropped before the meeting.

The two records that failed on the first pass (`KE_-_Install_New_Fence…` and
`KE_-_(Signed)_Install_New_Fence…`) failed because of the whitespace-run bug described in
Step 2, not because the data was missing. After correcting the sanitizer both resolve
exactly to `DQU4WK0D8E8A`. **No record was force-matched or guessed.** Per-record evidence
is in `link_derivation.json`.

Sample of the recovered links — note these are exactly the records the diagnostic flagged
as civically important:

| Flat row | → `agenda_item_id` |
|---|---|
| `DQU45Y09F4B0_Board_Minutes_2026_01_28.pdf` | `DQU47309F4E4` |
| `DQU45Y09F4B0_Board_Personnel_Report_02.11.2026.pdf` | `DQU47409F4E5` |
| `DQU45Y09F4B0_ASB_Vouchers_02-11-26.pdf` | `DR5MNW5C1C8C` |
| `DQU45Y09F4B0_Financial_Statement_December_2025_FINAL.pdf` | `DR39NK23D246` |
| `DQU47R0A3706_6630_-_Redline.pdf` | `DQU47Z0A3719` |

---

## Step 6 — The three scripts (written, **not executed**)

All under `scripts/ingest-repair-2026-09-13/`. Each is idempotent and prints before/after
counts.

| Script | Purpose | Idempotency mechanism |
|---|---|---|
| `process_48.sql` | Before/after counts for Stage 2 | Read-only — all SELECTs |
| `process_48.py` | Runs Stage 2 with preflight + blast-radius check | Preflight refuses unless exactly 48 pending |
| `link_40.sql` | Sets `agenda_item_id` on the 40 | `WHERE agenda_item_id IS NULL` |
| `delete_31.sql` | Deletes the 31 flat rows | `WHERE agenda_item_id IS NULL` + existence of row |

Supporting (already run, read-only): `target_set.sql`, `export_backup.sh`,
`verify_restore.sh`, `verify_duplicates.py`, `derive_links.py`.

**Validation performed without executing the mutations:** Python files compile clean
(`py_compile`), shell files parse clean (`bash -n`), and `process_48.sql` — which is
entirely SELECTs — was run in full (Appendix A9). `link_40.sql` and `delete_31.sql` have
**not** been executed in any form, including rollback.

### Guards built into the mutation scripts

`link_40.sql`
- Mapping is a literal 40-row `VALUES` list — no runtime derivation, no heuristic.
- Aborts if any mapped `external_id` is missing from `documents` (corpus drift).
- Updates only rows where `agenda_item_id IS NULL`; a second run reports 0.
- Deliberately does **not** backfill `metadata.item_name` / `item_order`, which the flat
  loader also left NULL — out of approved scope, flagged as a follow-up.

`delete_31.sql`
- Pairs are a literal 31-row `VALUES` list carrying the verified SHA-256.
- **Guard 1:** aborts if any kept counterpart is missing or has become unlinked.
- **Guard 2:** aborts if any `facts.*` row references a row about to be deleted.
- Deletes only rows still `agenda_item_id IS NULL`; a second run deletes 0.
- Prints the Qdrant point IDs it will orphan **before** deleting. **Does not touch Qdrant.**

`process_48.py`
- Preflight refuses to run unless exactly 48 agenda items are pending **and** no pending
  agenda item exists outside the three target meetings. This matters because
  `document_processor` has no per-document filter — it selects by
  `processing_status='pending'` plus an optional `--document-type`, so a stray pending row
  would be swept into the same run.
- Captures an MD5 fingerprint of every non-target document before and after and fails if
  it changes.
- Credentials read from the ambient environment. **`.env` is not read, written, or modified.**

**Stage 2 on these 48 needs no OCR and no file access.** `agenda_item` extraction is
`strip_html(content_raw)` (`document_processor/processor.py:209-213`), so the stopped OCR
service and the stale `file_path` values are both irrelevant. This is the zero-risk step.

---

## Step 7 — Expected post-state

| Measure | Now | After approved steps |
|---|---|---|
| Unlinked 2026 **BoardDocs** attachments | 71 | **0** |
| Unlinked 2026 **email** attachments | 2 | 2 *(correct — no agenda item exists)* |
| Unprocessed 2026 agenda items | 48 | **0** |
| Duplicate pairs | 31 | **0** |
| `documents` rows for the target meetings | 119 | 88 *(119 − 31 deleted)* |
| Chunks for the 48 | 0 | ≥ 48 *(one or more per item)* |
| Orphaned Qdrant points | 0 | **110 — listed, not deleted** |

Step 5 resolved all 40, so the Step 7 target is the full **0**, not a partial figure.

**The 110 Qdrant points are the one loose end this task deliberately leaves.** Deleting the
31 rows cascades away 110 chunks, all of which are embedded and have live Qdrant points.
Per the task scope, Qdrant is **not** modified; the point IDs are captured in
`backups/ingest-repair-2026-09-13/qdrant_points_dup31.jsonl` and re-printed by
`delete_31.sql` at run time. Until that cleanup runs, those 110 points remain searchable
and will surface duplicate citations — the same symptom the delete is meant to fix, in the
vector layer. Design below.

---

## Recommended changes (requires operator approval)

Designs only. **Neither is executed, and neither is part of Part 2.**

### RC1 — Content-hash dedupe key alongside `external_id`

**Problem.** `external_id` is `<meeting>_<item>_<file>` in the structured path and
`<meeting>_<file>` in the flat path (`loader.py:191` vs `:288`). The unique constraint
`documents_tenant_id_external_id_key` therefore cannot see that two rows describe one PDF.
`ON CONFLICT DO NOTHING` — documented in `CLAUDE.md` as *the* idempotency mechanism — is
keyed on a value that encodes the scraper's directory layout. Any future mixed-layout run
silently doubles records, exactly as it did here.

**Design.**

```sql
ALTER TABLE documents ADD COLUMN content_hash char(64);          -- SHA-256 of file bytes
CREATE UNIQUE INDEX CONCURRENTLY documents_tenant_meeting_hash_key
    ON documents (tenant_id, meeting_id, content_hash)
    WHERE content_hash IS NOT NULL;                              -- partial: tolerates backfill
```

- Loader computes SHA-256 of the file bytes at ingest and inserts with
  `ON CONFLICT (tenant_id, meeting_id, content_hash) DO NOTHING`.
- Partial index so existing rows (NULL hash) do not block the migration; backfill can then
  proceed incrementally.
- Keep `external_id` as-is — it stays useful for provenance. This is an **additional** key,
  not a replacement.
- Scope the uniqueness to `meeting_id`: the same PDF legitimately appears across meetings
  (e.g. minutes of a prior meeting) and must not be collapsed corpus-wide.

**Verification before adopting:** run the hash over the existing corpus and count
would-be collisions per meeting. If any collision is *not* a genuine duplicate, the
constraint is wrong and must not be applied. This is the step that keeps the fix from
becoming its own incident.

**Also worth adding** (cheap, and this diagnosis needed all three): a `scraper_version`
written from `meeting.json`, a loader run ID, and a `content_hash` on `chunks` for the same
reason.

### RC2 — Qdrant cleanup for deleted chunks

**Problem.** `chunks` rows vanish by cascade; their Qdrant points do not. After
`delete_31.sql`, 110 points reference chunk IDs that no longer exist.

**Design.**

1. Source of truth is `qdrant_points_dup31.jsonl` (captured **before** the delete — after
   the cascade the IDs are unrecoverable from Postgres).
2. Delete by explicit point ID, never by filter:
   ```
   POST http://127.0.0.1:6333/collections/boarddocs_chunks/points/delete
   { "points": ["<uuid>", ...] }
   ```
3. Before/after `collection_info.points_count`; expect a decrease of exactly 110.
4. Verify each deleted ID returns 404 on retrieve, and that a control point from the
   **kept** twin still resolves.

**Sequencing matters.** Capture the IDs before the delete, execute the Qdrant removal
after. If the point IDs are lost, recovery means re-deriving them from Qdrant payloads by
`document_id` — possible but far more fragile.

**Broader check worth running once:** the same orphaning may already exist from the 75
failed and 24 deferred attachments. A full reconciliation of Qdrant point IDs against live
`chunks.id` would size that. Out of scope here.

---

## Stop-rule compliance

- **No mutation of any kind.** Every statement against the corpus was a `SELECT`, except
  the scratch-schema restore test, which created and dropped `scratch_ingest_repair` and
  touched nothing in `public.*` or `facts.*`. Confirmed dropped: 0 remaining.
- **Neither scraper was run.** The legacy scraper is not present on this host.
- **The loader was not re-run**, and was never pointed at the working data tree.
- **No network requests.** All derivation used stored HTML and on-disk scrape output.
- **`.env` untouched.** Credentials came from the container's local auth path via
  `podman exec`, the same read path as the diagnostic.
- **No hook bypassed.** `block-dangerous-commands.sh` blocked `sudo`; the operator ran
  those two commands instead. No guard was circumvented or worked around.
- **Nothing forced.** 0 pairs excluded and 0 orphans unresolved — because the evidence
  supported all of them, not because any check was loosened. The two records that initially
  failed were re-examined and a real bug in the matching rule was fixed; they were not
  waved through.
- **`ksd-boarddocs-rag`** was read once, read-only, limited to `logs/sessions/`, with the
  operator's explicit approval during this session.

### Guard gaps observed (reported, not used)

1. **`boarddocs-postgres` publishes `0.0.0.0:5432`.** Still true — every other container
   binds `127.0.0.1`. `CLAUDE.md` requires loopback-only. Unchanged from the diagnostic;
   re-flagged because this repair confirms the corpus behind that port is the production
   corpus. Recommend `firewall-cmd --list-ports` and correcting the quadlet's `PublishPort`.
2. **No hook protects against `DELETE`/`UPDATE` on the corpus.** The guardrails cover
   dangerous shell commands, but `delete_31.sql` could have been executed at any point in
   Part 1 with nothing stopping it. The discipline here was procedural, not enforced.
   Consider a hook that refuses non-`SELECT` SQL unless an approval marker is present.
3. **Stale `.env` Postgres credential.** Unchanged — host-side `psql` cannot authenticate.

---

## What needs your decision

Part 2 runs **only** the steps you approve, in order. Suggested order and risk:

| Step | Action | Risk | Reversible? |
|---|---|---|---|
| **a** | Stage 2 on the 48 | **Lowest** — no OCR, no files, no deletes | Yes — re-run |
| **b** | `link_40.sql` | Low — sets a NULL column on 40 rows | Yes — backup + `SET NULL` |
| **c** | `delete_31.sql` | **Highest** — destructive, cascades 110 chunks | Via backup restore |
| **d** | Re-list the two 2026-02-11 meetings | None — read-only | n/a |

Step (c) is the only irreversible one, and its safety rests on the SHA-256 proof and the
verified absence of `facts.*` references. Approving (a), (b) and (d) while deferring (c)
is a coherent option: it fixes everything civically important — the 2026-02-11 minutes,
vouchers and personnel report become reachable — and leaves only duplicate citations,
which are cosmetic by comparison.

**Reply `approved` with the steps you want**, e.g. "approved: a, b, d".

---

## Appendix — every command and query

All `psql` ran as:
`podman exec [-i] boarddocs-postgres psql -U boarddocs -d boarddocs -P pager=off [flags] -c "<SQL>"`

### A1 — Scheduler enumeration

```bash
crontab -l
cat /etc/crontab; ls -la /etc/cron.d/
ls -la /etc/cron.hourly /etc/cron.daily /etc/cron.weekly /etc/cron.monthly
cat /etc/cron.d/0hourly; cat /etc/anacrontab
systemctl --user list-timers --all --no-pager
systemctl list-timers --all --no-pager
systemctl --user cat <hashed-unit>.service        # x4 -> podman healthcheck run
ls -la ~/.config/systemd/user/                    # empty
grep -rilE 'scrape|scraper|loader|boarddocs|ksd' \
     ~/.config/systemd/user/ /etc/systemd/system/ /usr/lib/systemd/system/ /etc/systemd/user/
```

### A2 — Operator-run privileged checks

`sudo` is blocked by `~/.claude/hooks/block-dangerous-commands.sh`
(`BLOCKED: privilege escalation command detected`). Not bypassed. Operator output:

```
sudo crontab -l -u root
  no crontab for root
sudo ls -la /var/spool/cron/
  total 0
  drwx------. 1 root root  0 Aug  5  2025 .
  drwxr-xr-x. 1 root root 86 Apr  3 18:55 ..
```

### A3 — Repo / scraper scheduler search

```bash
grep -rilE 'crontab|OnCalendar|\.timer|schedule' --include='*.py' --include='*.sh' \
  --include='*.md' --include='*.service' --include='*.timer' --include='*.toml' \
  --include='*.yaml' --include='*.yml' --include='*.container' ~/workspace/projects/ksd-main
ls -laR ~/workspace/projects/ksd-main/systemd/
grep -rilE 'crontab|OnCalendar|schedule' ~/workspace/projects/ksd_forensic/ ...
timeout 300 grep -rl "files_found" /home/donald/ --include='*.py' --include='*.sh' \
  --include='*.ts' --include='*.js' --include='*.ipynb'
cat ~/workspace/projects/ksd_forensic/scripts/{KentScrapes.py,run-scraper.sh,update-boarddocs}
cat ~/workspace/projects/ksd_forensic/boarddocs/repo/kent_scraper.py
```

### A4 — Forensic window checks

```bash
journalctl --list-boots --no-pager
journalctl --no-pager -o short-iso | head -2
journalctl --no-pager --since '2026-04-08' -t CROND -t crond -t anacron -o short-iso
journalctl --no-pager --since '2026-04-08' | grep -icE 'KentScrapes|boarddocs_update|boarddocs_loader|kent-scraper|run-scraper'   # 0
grep -inE 'scrap|boarddocs_loader|python.*load' ~/.bash_history    # 0 results
```

```sql
SELECT count(*), min(action_timestamp)::timestamp(0), max(action_timestamp)::timestamp(0) FROM ai_activity_log;
SELECT count(*), min(update_timestamp)::timestamp(0), max(update_timestamp)::timestamp(0) FROM corpus_updates;
SELECT action_timestamp::timestamp(0), action_type, left(coalesce(command_executed,''),120), working_directory
FROM ai_activity_log
WHERE command_executed ~* '(scrap|boarddocs_loader|KentScrapes|boarddocs_update|load_scraped)'
ORDER BY action_timestamp LIMIT 40;      -- 0 rows
```

### A5 — Set re-derivation

`scripts/ingest-repair-2026-09-13/target_set.sql` is the canonical definition. Counts:

```sql
SELECT count(*) FROM documents WHERE document_type='agenda_item' AND processing_status='pending';
```

```sql
WITH unlinked AS (
  SELECT id, meeting_id, meeting_date, title FROM documents
  WHERE document_type='attachment' AND agenda_item_id IS NULL
    AND meeting_date >= '2026-01-01' AND external_id NOT LIKE 'email_%'),
linked AS (
  SELECT meeting_id, regexp_replace(title,'\s+','_','g') AS santitle FROM documents
  WHERE document_type='attachment' AND agenda_item_id IS NOT NULL AND meeting_date >= '2026-01-01')
SELECT u.meeting_date, u.meeting_id, count(*) AS unlinked_total,
  count(*) FILTER (WHERE EXISTS (SELECT 1 FROM linked l WHERE l.meeting_id=u.meeting_id AND l.santitle=u.title)) AS dup,
  count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM linked l WHERE l.meeting_id=u.meeting_id AND l.santitle=u.title)) AS orphan
FROM unlinked u GROUP BY 1,2 ORDER BY 1,2;
```

### A6 — Whitespace-normalization correction

```sql
SELECT 'linked 2026 titles with whitespace runs', count(*) FROM documents
 WHERE document_type='attachment' AND agenda_item_id IS NOT NULL
   AND meeting_date>='2026-01-01' AND title ~ '\s\s'          -- 1: '5010P  - Final.pdf'
UNION ALL
SELECT 'unlinked 2026 titles with double underscore', count(*) FROM documents
 WHERE document_type='attachment' AND agenda_item_id IS NULL
   AND meeting_date>='2026-01-01' AND external_id NOT LIKE 'email_%' AND title ~ '__';   -- 0
```

```bash
ls "$B/2026-02-11-regular-meeting-630-pm/" | grep -i 'KE.*Fence' | cat -A
# KE_-_Install_New_Fence_...  (single underscore, from a double space in the original)
```

### A7 — Dependency and duplicate proof

```sql
SELECT conrelid::regclass AS child_table, conname, pg_get_constraintdef(oid)
FROM pg_constraint
WHERE contype='f' AND (confrelid::regclass::text IN ('documents','chunks','document_pages')
   OR conrelid::regclass::text IN ('documents','chunks','document_pages')) ORDER BY 1;
```

Fact references by bucket: see `derive_links.py` / report §Step 3. Duplicate proof:

```bash
python3 scripts/ingest-repair-2026-09-13/verify_duplicates.py \
        backups/ingest-repair-2026-09-13/duplicate_proof.json
```

### A8 — Link derivation

```bash
python3 scripts/ingest-repair-2026-09-13/derive_links.py \
        backups/ingest-repair-2026-09-13/link_derivation.json
```

Structure checks that ruled out the diagnostic's method:

```sql
SELECT external_id, length(content_raw),
  (SELECT count(*) FROM regexp_matches(content_raw,'Board\.nsf/files/[A-Z0-9]+','g')) AS file_refs,
  (SELECT count(*) FROM regexp_matches(content_raw,'goto\?open&id=[A-Z0-9]+','g'))    AS goto_refs
FROM documents WHERE document_type='agenda' AND meeting_id IN ('DQU45Y09F4B0','DQU47R0A3706');
-- file_refs 21 / 19 ; goto_refs 0 / 0
```

```sql
SELECT meeting_id, document_type, count(*), count(DISTINCT agenda_item_id)
FROM documents WHERE meeting_id IN ('DQU45Y09F4B0','DQU47R0A3706') GROUP BY 1,2 ORDER BY 1,2;
-- zero agenda_item rows for either meeting
```

### A9 — Backup, restore and script validation

```bash
./scripts/ingest-repair-2026-09-13/export_backup.sh   backups/ingest-repair-2026-09-13
./scripts/ingest-repair-2026-09-13/verify_restore.sh  backups/ingest-repair-2026-09-13
( cd backups/ingest-repair-2026-09-13 && sha256sum ./*.jsonl ./*.json row_counts.txt > manifest.sha256 )

python3 -m py_compile scripts/ingest-repair-2026-09-13/{process_48,derive_links,verify_duplicates}.py
bash -n scripts/ingest-repair-2026-09-13/{export_backup,verify_restore}.sh
podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs -P pager=off \
  -v phase=before -f - < scripts/ingest-repair-2026-09-13/process_48.sql
```

`process_48.py` preflight and snapshot were exercised read-only by importing the module and
calling `preflight()` and `snapshot()` directly — `main()` was **not** called, so Stage 2
did not run.

### A10 — Qdrant exposure of the delete set

```sql
SELECT count(*) AS dup31_chunks, count(qdrant_point_id) AS with_qdrant_point,
       count(*) FILTER (WHERE embedding_status='complete') AS embedded
FROM chunks c JOIN documents d ON d.id=c.document_id
WHERE d.document_type='attachment' AND d.agenda_item_id IS NULL
  AND d.meeting_date>='2026-01-01' AND d.external_id NOT LIKE 'email_%'
  AND d.meeting_id IN ('DQHKYP542B0C','DQU2PT0331CA');
-- 110 | 110 | 110
```

**Files read (no modifications):** `document_processor/document_processor/{__main__,config,processor}.py`,
`boarddocs_loader/` (via prior diagnostic), `CLAUDE.md`, `reports/ingest-degradation-2026-09-13.md`,
`docs/session-logs/session-debrief-2026-09-13-ingest-degradation-diagnostic.md`,
`~/workspace/projects/ksd-boarddocs-rag/logs/sessions/` (read-only, operator-approved).

---

## Addendum — 2026-09-14: step **a** split into **a1** and **a2**

*Appended per the append-only convention. The decision table above is unchanged; this
expands its step **a** ("Stage 2 on the 48"), which was incomplete — processing the 48
produces chunks, but chunks are not searchable until they are embedded. Running a1 without
a2 leaves the 48 items loaded, chunked, and still invisible to the RAG API.*

### Prerequisite — venvs (blocking, applies to both)

`CLAUDE.md`'s `document_processor/venv/` line describes **production**. On Smeltor no venv
existed for either component (verified 2026-09-13).

- `document_processor/` — venv **created 2026-09-14** via its own `./setup.sh`. Python
  3.14.3, 31 packages, **35/35 tests pass**.
- `embedding_pipeline/` — venv **still missing**. `./setup.sh` must run before a2. The ONNX
  `model_cache/mxbai-embed-large-v1-onnx/` (1.3 GB) is already present, so no model download
  is needed.

### Step a1 — `document_processor` on the 48 *(unchanged from step a)*

```bash
cd ~/workspace/projects/ksd-main/document_processor && ( \
  set -a; . ../.env; set +a; \
  export DATABASE_URL="postgresql://boarddocs:${POSTGRES_PASSWORD}@127.0.0.1:5432/boarddocs"; \
  source venv/bin/activate; \
  python -m document_processor --document-type agenda_item --workers 4 --dry-run )
```

Drop `--dry-run` for the real run. `--dry-run` processes 5 documents and writes nothing.

Verify:

```sql
SELECT processing_status, count(*) FROM documents
WHERE document_type='agenda_item' GROUP BY 1;
-- expect: pending -> 0, complete -> 6,529
```

### Step a2 — `embedding_pipeline` for the newly created chunks

**Scoping note — read before running.** `embedding_pipeline` has **no `--document-type`,
`--document-id` or equivalent flag.** It selects work solely by status
(`pipeline.py:28`: `WHERE c.embedding_status = 'pending' AND c.tenant_id = $1`). It therefore
cannot be pointed at "the 48" directly.

It does not need to be, because of a verified invariant:

| Check (2026-09-14) | Result |
|---|---|
| `chunks` by `embedding_status` | `complete` = 179,081; **`pending` = 0** |
| Chunks belonging to the 48 pending agenda items | **0** |

Every existing chunk is already embedded, and the 48 have no chunks yet. So **after a1, the
only `pending` chunks in the table are the ones a1 just created** — an unscoped run is
naturally scoped to exactly those documents.

**Confirm the invariant still holds immediately before running a2** (it breaks if anything
else creates chunks in between):

```sql
SELECT count(*) AS pending_chunks,
       count(DISTINCT c.document_id) AS distinct_docs
FROM chunks c WHERE c.embedding_status = 'pending';

-- and confirm they all belong to the 48:
SELECT d.document_type, d.processing_status, count(*) AS chunks
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE c.embedding_status = 'pending' GROUP BY 1,2;
-- expect a single row: agenda_item | complete | <n>
```

If `distinct_docs` exceeds 48, or any row is not `agenda_item`, **stop** — something else is
queued and an unscoped run would embed it too.

**Setup (once), then the run:**

```bash
cd ~/workspace/projects/ksd-main/embedding_pipeline && ./setup.sh

cd ~/workspace/projects/ksd-main/embedding_pipeline && ( \
  set -a; . ../.env; set +a; \
  export DATABASE_URL="postgresql://boarddocs:${POSTGRES_PASSWORD}@127.0.0.1:5432/boarddocs"; \
  source venv/bin/activate; \
  python -m embedding_pipeline --dry-run )
```

`--dry-run` fetches 3 chunks, embeds them, prints vectors and payloads, and writes nothing.
Drop it for the real run. `QDRANT_URL` comes from `.env` (present); `--collection` defaults
to `boarddocs_chunks`, `--batch-size` to 256, `--tenant` to `kent_sd`.

### Verification after a2

```sql
-- 1. No pending chunks left, none failed
SELECT embedding_status, count(*) FROM chunks GROUP BY 1 ORDER BY 2 DESC;
-- expect: complete = 179,081 + <n>, pending = 0, failed = 0

-- 2. The 48 specifically are fully embedded
SELECT d.processing_status,
       c.embedding_status,
       count(*) AS chunks,
       count(DISTINCT d.id) AS docs
FROM documents d JOIN chunks c ON c.document_id = d.id
WHERE d.document_type = 'agenda_item'
  AND d.meeting_date = '2026-03-25'
GROUP BY 1,2;
-- expect: complete | complete | <n> | 48

-- 3. Model stamp is the expected one
SELECT DISTINCT embedding_model FROM chunks WHERE embedding_status='complete';
-- expect: mxbai-embed-large-v1
```

Then confirm the vectors actually reached Qdrant — the point count must rise by the same
`<n>`, otherwise PostgreSQL says `complete` while retrieval still cannot see them:

```bash
# record before a2, compare after
( set -a; . ~/workspace/projects/ksd-main/.env; set +a; \
  python3 -c "
import os,urllib.request,json
u=os.environ['QDRANT_URL'].rstrip('/')+'/collections/boarddocs_chunks'
print(json.load(urllib.request.urlopen(u))['result']['points_count'])" )
```

**Any `failed` chunks:** `pipeline.py:43` sets `embedding_status='failed'` on error. A
non-zero `failed` count means a2 partially succeeded — re-running picks up only `pending`,
**not** `failed`, so failed rows must be reset to `pending` by hand before a retry.

### Why a2 is not optional

Per `CLAUDE.md` the embedding cron is commented out, so nothing will pick these up later.
Without a2 the 48 items — 42 from meeting `DS4MSK5CA5C5`, 2 from `DS4MVC5CCBE6`, 4 from
`DSCM9K5A287D`, all 2026-03-25 — remain absent from every RAG answer while appearing
`complete` in `documents`. That is the failure mode the original incident already produced
once: data present in PostgreSQL, invisible to retrieval.

---

## Part 2 — Execution record

**Approved:** a, b, d, then c last. Additional operator instruction for (c): write the 110
orphaned Qdrant point IDs to `backups/ingest-repair-<date>/qdrant-orphans-from-delete.txt`
**before** running the delete; do not touch Qdrant.

**Executed 2026-09-14. Steps b, d and c are complete and verified. Step (a) is complete for
a1 only — a2 (embedding) has not run**, so the brief's requirement to confirm chunk *and
embedding* rows exist for each is **not yet satisfied**. See the Addendum above for the
a1/a2 split, and *Outstanding* below.

Every mutation is bracketed by a count query whose output appears below.

### Corpus-level before/after

| Measure | Before Part 2 | After Part 2 | Delta |
|---|---|---|---|
| `documents` rows | 20,197 | **20,166** | −31 (deleted duplicates) |
| `chunks` rows | 179,081 | **179,026** | −55 (+55 from a1, −110 cascade in c) |
| Pending documents, corpus-wide | 48 | **0** | −48 |
| 2026 BoardDocs unlinked attachments | 71 | **0** | −71 |
| 2026 email attachments unlinked | 2 | **2** | 0 (correct — no agenda item exists) |
| Chunks with `embedding_status='pending'` | 0 | **55** | +55 (awaiting a2) |

Chunk arithmetic reconciles exactly: 179,081 + 55 − 110 = 179,026.

### Step (a1) — `document_processor` on the 48 agenda items

Run by the operator (the wrapper needs a host-side credential this session did not hold; see
*Deviations*). Processor output: `Total documents: 48, Completed: 48, Failed: 0,
Chunks created: 55`.

| Measure | BEFORE | AFTER |
|---|---|---|
| Target pending | 48 (42 / 2 / 4) | **0** |
| Target complete | 0 | **48** |
| Target failed or deferred | 0 | **0** |
| Documents with chunk rows | 0 of 48 | **48 of 48** |
| Target chunks | 0 | **55** |
| Embedding rows for those chunks | — | **0 embedded**, 55 `pending`, 0 Qdrant points |
| **Other documents fingerprint** | `625247e1d058d5a6c8d9ba37e2173748` | **`625247e1d058d5a6c8d9ba37e2173748`** |
| **Other chunks** | 179,081 | **179,081** |

The fingerprint covers `id:processing_status:updated_at` for all 20,149 non-target documents.
Unchanged — **no other document's rows changed**, and no chunk was added to or removed from
any other document.

**Chunk rows exist for all 48; embedding rows do not.** That half of the step remains open.

### Step (b) — `link_40.sql`

```
=== BEFORE ===
 unlinked | already_linked | in_map
       40 |              0 |     40
UPDATE 40
=== AFTER ===
 unlinked | linked | in_map
        0 |     40 |     40
```

Coverage audit re-run immediately afterwards:

```
  yr  | email_expected_null | boarddocs_unlinked
 2026 |                   2 |                 31
```

The 31 were the duplicates still awaiting step (c); after (c) this reads **0**.

**Idempotency verified by re-running:** second run reported `UPDATE 0`, state unchanged.

### Step (d) — 2026-02-11 document listings

| Meeting | Attachments | Linked | Unlinked |
|---|---|---|---|
| `DQU45Y09F4B0` — Regular Meeting 6:30 p.m. | 21 | **21** | **0** |
| `DQU47R0A3706` — Work Session 5:00 p.m. | 19 | **19** | **0** |

The records the diagnostic flagged as civically significant are now agenda-item addressable:

| Document | `agenda_item_id` |
|---|---|
| `Board_Minutes_2026_01_28.pdf` | `DQU47309F4E4` |
| `Board_Special_Meeting_Minutes_2026_01_28.pdf` | `DQU47309F4E4` |
| `Board_Special_Meeting_Minutes_2026_02_04.pdf` | `DQU56Y0EC92D` |
| `Board_Personnel_Report_02.11.2026.pdf` | `DQU47409F4E5` |
| `ASB_Vouchers_02-11-26.pdf` | `DR5MNW5C1C8C` |
| `Capital_Fund_Vouchers_02-11-26.pdf` | `DR5MNW5C1C8C` |
| `Custodial_Vouchers_02-11-26.pdf` | `DR5MNW5C1C8C` |
| `General_Fund_Vouchers_02-11-26.pdf` | `DR5MNW5C1C8C` |
| `Trust_Vouchers_02-11-26.pdf` | `DR5MNW5C1C8C` |
| `TVF_Vouchers_02-11-26.pdf` | `DR5MNW5C1C8C` |
| `BDMTG_-_2-11-2026_SIGNED.pdf` | `DR5MNW5C1C8C` |

Every 2026 meeting now has 100% attachment link coverage:

```
 meeting_date |  meeting_id  | attachments | linked
 2026-01-07   | DPXQWL6B4F39 |           2 |      2
 2026-01-14   | DPXSDX723E8F |          29 |     29
 2026-01-14   | DPXSEV7261B0 |          57 |     57
 2026-01-28   | DQHKYP542B0C |          65 |     65
 2026-01-28   | DQLQJK698C87 |           1 |      1
 2026-02-04   | DQU2PT0331CA |           1 |      1
 2026-02-11   | DQU45Y09F4B0 |          21 |     21
 2026-02-11   | DQU47R0A3706 |          19 |     19
```

Caveat: "linked" here means agenda-item addressable in PostgreSQL. These attachments were
already embedded before this repair, so retrieval is unaffected by the a2 gap.

### Step (c) — `delete_31.sql`

**Qdrant point IDs captured first, as instructed.**
`backups/ingest-repair-2026-09-13/qdrant-orphans-from-delete.txt` — **110 data lines**,
tab-separated `qdrant_point_id / chunk_id / document_external_id`, with a header noting the
IDs become unrecoverable from PostgreSQL once the cascade fires. Written and line-counted
**before** the delete ran. Checksummed in `manifest.sha256`.

```
=== BEFORE ===
 pairs_in_map | flat_rows_present | kept_rows_present | chunks_to_cascade
           31 |                31 |                31 |               110
DO        <- guard 1: every kept counterpart present and still linked
DO        <- guard 2: no facts.* row references any row being deleted
DELETE 31
=== AFTER ===
 pairs_in_map | flat_rows_remaining | kept_rows_remaining | chunks_remaining
           31 |                   0 |                  31 |                0
```

Both guards executed and passed. Post-delete verification:

| Check | Result |
|---|---|
| Flat duplicate rows remaining | **0** of 31 |
| Kept records still linked **and** `complete` | **31 of 31** |
| Kept records' chunks still present | **110** |
| `facts.*` rows with a dangling document reference | **0** |
| `facts` counts — meeting / attendance / motion / vote / exec_session / parse_log | 1,646 / 3,515 / 6,507 / 19,625 / 280 / 874 — all intact |

**Idempotency verified by re-running:** second run reported `flat_rows_present 0` and
`DELETE 0`, kept records untouched.

**Qdrant was not modified.** The 110 points remain live and now reference chunk rows that no
longer exist. Deliberate loose end; see RC2.

### Post-state vs. the Step 7 prediction

| Measure | Predicted | Actual | Match |
|---|---|---|---|
| Unlinked 2026 BoardDocs attachments | 0 | **0** | yes |
| Unlinked 2026 email attachments | 2 | **2** | yes |
| Unprocessed 2026 agenda items | 0 | **0** | yes |
| Duplicate pairs | 0 | **0** | yes |
| `documents` rows for target meetings | 88 | **88** | yes |
| Chunks for the 48 | >= 48 | **55** | yes |
| Orphaned Qdrant points | 110, listed not deleted | **110, listed not deleted** | yes |

Every Part 1 prediction held. The predictions did not cover embedding state, which is the
gap the Addendum identifies.

### Outstanding — step a2 (embedding), and a decision that must precede it

**a2 has not run, and it is a prerequisite for the 48 being reachable at all.** Per
`CLAUDE.md` the embedding cron is disabled, so nothing will pick these up automatically.
Until a2 runs, all 48 read `complete` in `documents` while remaining absent from every RAG
answer — precisely the failure mode of the original incident.

Readiness, verified 2026-09-14 after step (c):

| Item | State |
|---|---|
| `embedding_pipeline/venv/` | **missing** — `./setup.sh` must run first |
| `model_cache/mxbai-embed-large-v1-onnx/` | present, 1.3 GB — no download needed |
| Scoping invariant (`pending` chunks) | **holds**: 55 chunks / 48 docs, single row `agenda_item / complete / 2026-03-25` |
| Qdrant `points_count` baseline | **not captured** — `curl` is blocked by `block-dangerous-commands.sh` |

The invariant the Addendum requires is satisfied: every other chunk in the corpus is already
`complete` (178,971), so an unscoped `embedding_pipeline` run is naturally scoped to exactly
the 48. **Re-confirm it immediately before running a2**, since it breaks if anything else
creates chunks in the interim.

**Decision required before a2:** the navigation-chrome finding below. Once a2 runs the
boilerplate is in Qdrant, and removing it then means deleting points as well as re-chunking.
The cheap moment to fix it is now, while `embedding_status='pending'` and Qdrant holds
nothing for these 48.

### Quality finding — navigation chrome in the 48

All 48 of the 2026-03-25 agenda items carry BoardDocs UI navigation text
(`Previous / Next / Close / Print / Share Menu / Share on Twitter / ...`) in `content_raw`, so
`strip_html` carries it into the extracted text and into the chunks. The rest of the corpus is
essentially clean: **33 of 6,481** existing agenda-item chunks (0.5%) contain it, against
**48 of 48** here.

The cause is upstream in the scrape, not the processor: the 2026-03-25 load stored whole
pages. `content_raw` averages **115,252** characters for these 48 versus **60,463** for
previously processed agenda items.

Measured across all 48 (chrome as a share of extracted text):

| Statistic | Value |
|---|---|
| Minimum | 4.4% |
| **Median** | **11.2%** |
| Maximum | 45.8% |
| Documents above 30% | **15 of 48** |

**Why proceeding with a1 was still correct:** a2 was not run, so these 55 chunks have
`embedding_status='pending'` and zero Qdrant points — nothing reached retrieval. The condition
is fully reversible (reset the 48 to `pending`, delete their 55 chunks, re-run), and the
backup covers them. Fixing `strip_html` mid-repair would have been an unreviewed change to
code shared by 20,166 documents.

**Options before a2:** (i) strip the navigation block during extraction and re-process the 48,
then embed; or (ii) accept a constant ~27-token boilerplate prefix in 55 chunks, 15 of which
become predominantly boilerplate. Option (i) is cheap now and expensive after a2.

### Deviations from the plan

1. **Step (a) was run by the operator, not by `process_48.py`.** No Postgres credential was
   injected into the session, `.env` is protected by a deny rule, and shell exports do not
   persist between commands, so the wrapper could not authenticate from the host. The operator
   ran `document_processor` directly with the same scoping (`--document-type agenda_item`,
   which the preflight had already confirmed selects exactly the 48). Before/after counts and
   the blast-radius fingerprint were captured by this session on either side of that run, so
   the verification the wrapper would have performed was performed regardless.
   `process_48.py` itself was never executed.

2. **A missing dependency had to be fixed first.** `document_processor/config.py:9` imports
   `python-dotenv`, but it was absent from `requirements.txt`, so `setup.sh` could not build a
   working venv. Added `python-dotenv>=1.0.0` to `requirements.txt`; the operator ran the
   install after two guard hooks escalated it. Pre-existing packaging defect, unrelated to the
   ingest incident.

3. **Step (a) delivered only half its acceptance criterion.** The brief required chunk *and*
   embedding rows. Chunk rows exist for all 48; embedding rows do not. Recorded as outstanding
   rather than reported as complete.

4. **Order within the approved set.** Approved as "a, b, d, then c last" and executed in
   exactly that order.

### Stop-rule compliance (Part 2)

- **Only the approved steps ran**, in the approved order. Nothing outside a, b, d, c executed.
- **Every mutation is bracketed by count queries**, all reproduced above.
- **No check was loosened.** Both `delete_31.sql` guards executed and passed. Nothing was
  forced; no record was excluded to make a step complete.
- **Qdrant untouched**, as instructed; the 110 point IDs were captured to file first.
- **Neither scraper was run. The loader was not re-run. No network requests** — the one
  attempt to read Qdrant's point count via `curl` was blocked by
  `block-dangerous-commands.sh` and was not retried by other means.
- **`.env` was never read or modified.** A Bash attempt to source it for an auth test was
  correctly denied by the user deny rule and was not retried by other means.
- **No hook was bypassed.** Four hook events fired in Part 2, all respected:
  `ai-review-ask-commands.sh` escalated the `pip install` (twice), `validate-pip-install.sh`
  blocked it for want of an approved-packages allowlist, and `block-dangerous-commands.sh`
  blocked `curl`. The allowlist was **not** self-created — generating the file that governs
  what the agent may install would be self-authorization — so the install was handed to the
  operator.
