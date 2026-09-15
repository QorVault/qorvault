# Session Debrief — 2026-09-13 — 2026 ingest degradation diagnostic

**Session scope**: Read-only diagnostic. Why do 73 of 228 attachments on 2026 meetings have no agenda-item link, why are all 48 unprocessed agenda items in 2026, and is `source_url` (NULL on all 12,305 attachments) recoverable from stored metadata?
**Branch/project**: `ksd-main` on Smeltor, branch `claude/feat-facts-minutes`. PostgreSQL `boarddocs` container + on-disk scraper trees. Report at `reports/ingest-degradation-2026-09-13.md`. Diagnostic only; nothing modified, no network access, no re-ingestion.

## Decisions

**Read the corpus via `podman exec` rather than a host-side psql connection.** The `.env` `POSTGRES_PASSWORD` fails authentication against the container (`FATAL: password authentication failed for user "boarddocs"`). Rather than edit `.env` — explicitly out of scope — all queries ran as `podman exec boarddocs-postgres psql -U boarddocs -d boarddocs`, which uses the container's local auth path. Read-only either way. Flagged in the report as something to resolve so host tooling works.

**Reproduced the failure by importing the project's own `detector.py` and `parsers.py` against stored scrape directories**, rather than reasoning about the code statically. This is what turned "the linker looks like it might mis-key on something" into "the linker is correct and the input changed" — a materially different conclusion with a different fix.

**Did not treat the Phase 0 report's framing as given.** The task inherited "26 in 2024, 16 in 2025, 73 in 2026 — a degrading trend." Checking the `external_id` prefix on those records first, before diagnosing the trend, dissolved two thirds of it.

## What Changed

Nothing in the corpus, the code, or the scrapers. Two new files:

- `reports/ingest-degradation-2026-09-13.md`
- `docs/session-logs/session-debrief-2026-09-13-ingest-degradation-diagnostic.md`

Both uncommitted — see Open Items.

## Findings

**The "degradation" is three unrelated things, and one is not a defect.**

**1. 2024–2025 is a measurement error, not a regression (44 records).** All 42 of the 2024/2025 unlinked attachments — plus 2 of the 73 in 2026 — are *email* attachments from `input/Email/attachments/`, loaded 2026-02-25 from a separate source tree. `external_id` prefix `email_<Message-ID>_`, no `meeting_id`, no agenda item exists to link to. BoardDocs attachments unlinked in 2024: **0**. In 2025: **0**. The Phase 0 report's conclusion that "the structured scraper is degrading on recent meetings" and its ~115-record remediation item are both wrong.

**2. The real 2026 problem is 71 records, caused by scraper substitution — not a code regression.** Two different scrapers exist and both were run against 2026 meetings:

| | Structured (TS/Puppeteer) | Flat (legacy Python) |
|---|---|---|
| Slug | `2026-01-28-regular-meeting-6-30-p-m-` | `2026-01-28-regular-meeting-630-pm` |
| `meeting.json` keys | camelCase (`categories`, `meetingSlug`, `scrapedAt`) | snake_case (`meeting_id`, `slug`, `scraped_at`) |
| Item subdirs | 47 | 0 |
| Filenames | spaces preserved | spaces → underscores |
| `agenda.html` kept | no | **yes** |

The legacy flat scraper ran **2026-02-22 07:02–07:05** against meetings the structured scraper had already covered (2026-01-28, 2026-02-04) or had covered uselessly (2026-02-11, scraped 2026-02-10 *the day before the meeting*, producing only `.txt` and no `item.json`). The loader ingested the flat output on 2026-02-23 23:21.

`detector.py:40` returns `"structured"` iff the meeting directory has subdirectories. Flat → `_process_flat` (`loader.py:210-304`), which **never sets `agenda_item_id`** and hardcodes `item_order`/`item_name` to `None`. That is correct behaviour: a flat directory carries no positional information to link on. The link itself is made at `loader.py:198` off the item subdirectory's `item.json` → `itemId` — purely positional, no lookup, no defect.

Offline reproduction on stored pages:
```
2022-05-11-regular-meeting-7-p-m-   format=structured  categories=11  items=45  -> links
2026-02-11-regular-meeting-630-pm   format=flat        categories=0   items=0   -> does not link
```

**Split of the 71: 31 are duplicate rows of already-correct records; 40 are unique orphans.** The 2026-01-28 and 2026-02-04 flat rows duplicate structured rows for the same PDFs (identical `file_size_bytes` and `char_count`). The two 2026-02-11 meetings have **zero** linked attachments — those 40 records, including board minutes, vouchers, the personnel report and policy readings, exist in the corpus only in unlinked form.

**Ruled out with evidence**: site markup change (retained `agenda.html` uses the same href form as the whole corpus), key format change (same 12-char IDs in 2026 as 2019), scraper code regression (no `boarddocs-scraper` commit near 2026-02-22; structured output normal for 2026-01-14 and 2026-03-11/25), rate-limit truncation (no partial-run signature).

**3. The 48 unprocessed agenda items were never attempted.** All 48 are from **2026-03-25**, three meetings, one loader run at 23:02:54–23:03:02. `processing_error` NULL on all 48, `content_raw`/`content_text` populated on all 48, `created_at ≈ updated_at`, and no `deferred`/`failed` siblings — which is what a partial processor run would leave. The loader writes `'pending'` (`loader.py:163, 199, 260`) and `document_processor` promotes it. Stage 2 simply never ran after that ingest. Not stuck, not skipped, not failed.

**4. `source_url` is derivable for 12,207 of 12,305 attachments (99.2%) — from data already in PostgreSQL.** The Phase 0 report concluded this was unrecoverable, having looked only at `external_id`. Two other columns carry the file IDs:

- **Structured (5,637)** — the parent `agenda_item`'s `metadata->'links'` already holds `href`, `unique` (the file ID), and `filename`. Joining on `(meeting_id, agenda_item_id)` + `title = links->>'filename'` resolves **5,583 / 5,637 (99.0%)**.
- **Flat (6,668)** — `_process_flat` stores the entire `agenda.html` into the agenda document's `content_raw` (`loader.py:227-231, 253`) and then ignores it. Those hrefs carry the file IDs. `content_raw` present for **6,624 / 6,668**; the 44 misses are exactly the email attachments, which correctly have no URL.

No re-scrape and no network access are needed for the derivation.

**The URL pattern in the task brief is wrong.** It specifies `/wa/ksdwa/Board.nsf/pfiles/<FILEID>/$file/<name>`. Every href in the stored markup uses **`/files/`**, not `/pfiles/`; `grep -o 'pfiles/'` on a retained `agenda.html` returns zero. A backfill on the brief's pattern would write 12,207 broken links into the citation layer.

## Surprises

**`agenda.html` retention is a property of the *flat* scraper, not of era.** Retained for all of 2005–2017, 30 of 107 in 2018, **zero** for 2019–2025, and 9 in 2026 — the 9 being the 2026-02-22 flat re-scrape. The structured scraper discards raw HTML. The incident that caused the problem is also the only reason the raw HTML needed to diagnose and fix it exists.

**The idempotency guard cannot see cross-layout duplicates.** `external_id` is `<meeting>_<item>_<file>` structured vs `<meeting>_<file>` flat (`loader.py:191` vs `:288`), so `documents_tenant_id_external_id_key` treats one PDF as two documents. `ON CONFLICT DO NOTHING` is documented in `CLAUDE.md` as the idempotency mechanism, but it is keyed on a value that encodes the scraper's directory layout. Any future mixed-layout run silently doubles records. Latent defect, independent of this incident.

**`file_path` is stale on every row, and the working data tree is less than half the corpus.** All rows point at `/home/donald/ksd_forensic/...`, which does not exist on this host; the tree the corpus was built from is now `/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/...` (1,684 meeting dirs). The working copy at `~/workspace/projects/ksd_forensic/boarddocs/data/` has **729** dirs and **no** `agenda.html` at all. Re-running the loader against the working copy would see less than half the corpus and none of the recoverable HTML.

**No provenance is recorded anywhere.** No scraper version, no commit, no loader run ID on any document row. The scraper identity had to be inferred from `meeting.json` key casing and slug convention; loader runs are identifiable only by clustering `created_at` to the second. This should have been a one-query answer.

**`boarddocs-postgres` publishes `0.0.0.0:5432`.** Every other container on the host binds `127.0.0.1` (qdrant, vllm, open-webui), and `CLAUDE.md` requires loopback-only. If the host firewall is not blocking 5432 the corpus is LAN-reachable. Flagged in the report, not changed — unrelated to this diagnostic.

## Workarounds

`podman exec` in place of a host psql connection (see Decisions). Not a workaround around a guard — a different legitimate read path, taken because fixing the credential would have meant editing `.env`.

## Unfinished / Deferred

Deliberately, per the stop rules: no scraper modified or run, no records patched, no network requests issued. Every database statement was a `SELECT`.

Recommended changes are written up in the report for approval, none executed. Suggested order R5 → R1 → R2 → R4 → R3 → R6:

- **R5** — run `document_processor` + `embedding_pipeline` for the 48 pending items. Zero-risk, unblocks the current corpus. 0 network requests.
- **R1** — recover `agenda_item_id` for the 40 orphaned 2026-02-11 attachments by parsing the `agenda.html` already in `content_raw`. 0 network requests. Highest value: this is a regular board meeting currently unreachable by agenda-scoped queries.
- **R2** — delete the 31 cross-layout duplicate rows and their Qdrant points. After R1, since R1 reads the same records.
- **R4** — backfill `source_url` for 12,207 attachments. **Gated on 5 operator-run `curl` confirmations** that `/files/` resolves (listed verbatim in the report); optional ~100-request sampled validation afterward.
- **R3** — content-hash idempotency key + flat-path link recovery in the loader, so this cannot recur.
- **R6** — publish a correction to `reports/boarddocs-coverage-2026-09-12.md` §0.6. Append-only per project convention; do not edit the original.

**2024–2025 need no remediation.** Corrected scope is 2026 only: 40 records to link, 31 to delete, 48 to process — 119 records, not the 121 in the task framing, because 2 of the 73 are email attachments.

## Open Items

1. **Commit is pending operator action.** Per the session rules a blocked commit is left uncommitted with the commands recorded. Nothing was committed this session. To commit:
   ```
   cd ~/workspace/projects/ksd-main
   git add reports/ingest-degradation-2026-09-13.md \
           docs/session-logs/session-debrief-2026-09-13-ingest-degradation-diagnostic.md
   git commit -m "docs: diagnose 2026 ingest degradation and source_url derivability"
   ```
   Note the working tree also carries two untracked debriefs from 2026-09-08 (`session-debrief-2026-09-08-retrieval-smoke-test.md`, `session-debrief-2026-09-08-vllm-status-and-branch-conflict.md`) that predate this session and are still unstaged.

2. **The 5 URL-pattern confirmations must run before R4.** `/files/` vs the brief's `/pfiles/` is the difference between 12,207 working citations and 12,207 404s. Requests are listed verbatim in the report appendix.

3. **Decide which scraper is authoritative going forward, and stop running the other.** The 2026-02-22 flat run is what caused this. If the flat scraper is being used because the structured one is failing or is being run too early relative to meeting dates (as on 2026-02-10), that scheduling problem is the real upstream issue and R3 only limits the damage.

4. **Stale `.env` Postgres credential** — host-side psql cannot authenticate. Unblocks host tooling.

5. **`boarddocs-postgres` bound to `0.0.0.0:5432`** — verify `firewall-cmd --list-ports`, consider correcting the quadlet's `PublishPort` to `127.0.0.1:5432:5432`.
