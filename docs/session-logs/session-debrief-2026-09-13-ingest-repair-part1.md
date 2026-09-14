# Session Debrief — 2026-09-13 — 2026 ingest repair, Part 1 (prepare)

**Session scope**: Part 1 of the controlled two-part repair of the 2026 ingest incident — recurrence check, re-derive the 48/40/31 sets, back up and prove the backup restores, prove the 31 duplicates byte-identical, derive `agenda_item_id` for the 40 orphans, write three unexecuted scripts, report expected post-state, stop for approval. Part 2 not started.
**Branch/project**: `ksd-main` on Smeltor, branch `claude/feat-facts-minutes`. PostgreSQL `boarddocs` container via `podman exec`. Report at `reports/ingest-repair-2026-09-13.md`. No mutation of corpus data; no network access; neither scraper run; loader not re-run.

## Decisions

**Treated the diagnostic as a hypothesis, not as input.** Two of its load-bearing claims turned out to be wrong, and both mattered. It said `agenda_item_id` was recoverable by parsing item anchors from the retained `agenda.html` — that markup contains zero item IDs of any form. It also said the aborted 2026-02-10 structured scrape left an `archive_*` directory "containing no `item.json`" — it contains a complete per-item tree with `itemId` and per-attachment file IDs. Checking both directly rather than building on them is the difference between a partial recovery and 40/40 exact.

**Required two independent methods to agree before accepting any link.** Each orphan was resolved both by opaque file ID (flat filename → href in `agenda.html` → file ID → `item.json` `links[].unique` → `itemId`) and by sanitized filename against `item.json` `links[].filename`. A record is `exact` only if both resolve, agree, and are unique. Single-method matching on filenames alone is exactly the approach that produced the diagnostic's known miss.

**Proved duplicates on file bytes, not metadata.** The diagnostic identified the 31 by equal `file_size_bytes` and `char_count`. That is not proof — it is a strong hint. Re-proved all 31 by SHA-256 over the actual PDF bytes on both sides, which required remapping the stale `file_path` prefix on read (the DB values were not modified).

**Verified the backup by restoring it, and compared content rather than counts.** Row counts matching only proves the right number of rows moved. Compared an MD5 over every column of every row, ordered, restored-vs-live. The scratch schema used `CREATE TABLE ... (LIKE ...)`, which copies no foreign keys, so the restore copy could not reference anything real.

**Read `ksd-boarddocs-rag` only after asking.** The hard rules said "do not touch" it. Its `logs/sessions/` was the last source that could identify the 2026-02-22 invoker, so I stopped and asked rather than deciding unilaterally what "touch" meant. Operator approved read-only access to that directory alone.

## What Changed

No corpus data. No scraper, no loader, no `.env`, no CLAUDE.md. The only database write in the entire session was a scratch schema created and dropped for the restore test (`scratch_schemas_remaining` = 0).

New files:

- `reports/ingest-repair-2026-09-13.md`
- `docs/session-logs/session-debrief-2026-09-13-ingest-repair-part1.md`
- `scripts/ingest-repair-2026-09-13/` — `target_set.sql`, `export_backup.sh`, `verify_restore.sh`, `verify_duplicates.py`, `derive_links.py`, `process_48.sql`, `process_48.py`, `link_40.sql`, `delete_31.sql`
- `backups/ingest-repair-2026-09-13/` — 10 JSONL exports, 2 proof JSONs, `row_counts.txt`, `manifest.sha256` (gitignored; contains corpus content)

`link_40.sql` and `delete_31.sql` have **never been executed**, including under rollback.

## Findings

**The recurrence gate is closed, and the reason is stronger than "nothing is scheduled".** No user crontab, no root crontab, empty `/var/spool/cron`, empty `cron.{daily,weekly,monthly}`, no systemd timer on either bus that touches the project (the four hashed user timers are Podman healthchecks), and no scheduler config in either repo. Beyond that, **the legacy flat scraper's source is not on this host at all** — a host-wide search for the script emitting its signature key `files_found` found only code that *reads* it. Each candidate in `ksd_forensic/scripts/` was checked against the flat output signature and none matches. There is nothing to disable.

**What ran the 2026-02-22 scrape is not determinable, and that is a finished answer rather than an unfinished search.** `journalctl` starts 2026-04-08; `ai_activity_log` starts 2026-02-27; `corpus_updates` is empty; `~/.bash_history` has zero scraper invocations and nothing from February; no scraper logs exist. The single stored session recording is dated 2026-02-27 and is a monitoring self-test — which is also the explanation: the observability stack was commissioned five days *after* the incident it would have recorded.

**All three counts re-derived exactly: 48 / 40 / 31.** Δ = 0 on every set; no STOP.

**`facts.*` adds seven non-cascading foreign keys to `documents`, and their distribution decided the risk of the delete.** 43 fact rows reference the target set. **All 43 point at the 40 orphans; zero at the 31 duplicates.** So `delete_31.sql` cannot be blocked by an FK violation — and the orphans are confirmed load-bearing, since the 2026-02-11 minutes are already parsed into the fact tables and are currently cited by rows no agenda-scoped query can reach.

**All 40 orphans resolved exactly; all 31 duplicates proved byte-identical.** Zero excluded on either side — not because any check was loosened, but because the evidence supported every record. A cross-check confirmed the attachment set in the retained `agenda.html` and the set in the 2026-02-10 archive scrape are identical in both directions.

**Stage 2 on the 48 is genuinely zero-risk.** `agenda_item` extraction is `strip_html(content_raw)` (`processor.py:209-213`) — no OCR call, no file access. The stopped OCR service and the corpus-wide stale `file_path` values are both irrelevant to it. The 48 are also the only pending documents in the entire corpus, so a type-scoped run is exactly scoped.

## Surprises

**The diagnostic's filename normalization had a real bug, and it nearly cost two records.** It used `replace(title,' ','_')` — one underscore per space. The flat scraper collapses a *run* of whitespace to a *single* underscore. BoardDocs serves one 2026-02-11 file as `KE  - Install New Fence…` with a double space, stored as `KE_-_Install…`. Under the naive rule those two orphans looked like "added to the agenda after the archive scrape ran" — a plausible, entirely wrong conclusion I had already written down before checking it. Corrected to `regexp_replace(title,'\s+','_','g')` everywhere; both records then resolved exactly. The 31/40 split was unchanged, but one linked 2026 title (`5010P  - Final.pdf`) does contain a whitespace run and would have been mis-normalized.

**The incident's own damage is what makes it repairable — twice over.** The diagnostic noted that the flat scraper is the only reason the raw `agenda.html` exists. The same holds for the premature 2026-02-10 structured scrape: it was useless at the time (ran the day before the meeting, loaded nothing) but it left behind the `item.json` tree that is the *only* offline source of the agenda item IDs. Two separate operational mistakes are jointly the reason no network access is needed.

**`COPY`'s default text format silently corrupts JSONL.** The first restore attempt failed on `Character with value 0x0a must be escaped` — `COPY` text format treats backslash sequences as escapes, so a literal `\n` inside a JSON string became a real newline. `FORMAT csv` with control-character delimiters does no backslash processing. Anyone restoring these backups by hand needs this.

**The `facts` schema is invisible from `\d documents`'s usual reading.** The FK list shows the referencing tables, but it is easy to skim past seven non-cascading references in a repair that is mostly about `chunks`. Had they pointed at the delete set, `delete_31.sql` would have aborted partway.

## Workarounds

`podman exec` instead of a host-side `psql` connection, same as the diagnostic — the `.env` credential still fails authentication and editing `.env` is out of scope. A legitimate read path, not a circumvented guard.

Stale `file_path` prefixes remapped **on read only** (`/home/donald/ksd_forensic` → `/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic`) to hash the real files. The database values were not corrected — that is a separate change.

## Unfinished / Deferred

Deliberately, per the stop rules: **Part 2 was not started.** No processing, no linking, no deleting. `link_40.sql` and `delete_31.sql` are written and unexecuted.

Also deferred by design:

- **RC1 — content-hash dedupe key.** Designed in the report, not executed. Includes the pre-adoption check (hash the corpus, count would-be collisions per meeting) that keeps the fix from becoming its own incident.
- **RC2 — Qdrant cleanup for the 110 orphaned points.** Designed, not executed. Sequencing is the critical part: the point IDs must be captured *before* the delete cascades the chunks away. They are already in `qdrant_points_dup31.jsonl`.
- **`metadata.item_name` / `item_order`** remain NULL on the 40 — the flat loader never set them. `link_40.sql` deliberately does not backfill these; out of approved scope.

## Open Items

1. **Part 2 is gated on operator approval.** Reply `approved` with the steps wanted (a = Stage 2 on the 48, b = `link_40.sql`, c = `delete_31.sql`, d = re-list the two 2026-02-11 meetings). Approving a/b/d while deferring c is coherent — it restores everything civically significant and leaves only duplicate citations.

2. **Committed — not blocked.** `ae3a127` on `claude/feat-facts-minutes`, 11 files, 2,205 insertions. All pre-commit hooks passed and none was bypassed.

   The hooks did fail on the first attempt, and the failures were legitimate rather than spurious, so they were fixed rather than skipped: `ruff` raised 9 × `S608` (SQL built by f-string interpolation) in `process_48.py`, and `interrogate` measured 77.8% docstring coverage against an 80% minimum. `process_48.py` was rewritten so every SQL statement is a fixed literal with no interpolation — which is what `CLAUDE.md` requires anyway — and the missing docstrings were added. All three scripts were re-run afterwards and produce byte-identical results (31/31, 40/40, preflight OK), confirming the reformatting changed no behaviour.

   `backups/` is gitignored and must stay that way — it contains corpus content. Verified with `git check-ignore` before staging.

   The working tree still carries the prior session's uncommitted `reports/ingest-degradation-2026-09-13.md` and three untracked debriefs from 2026-09-08 and 2026-09-13. Left alone: not this session's work, and committing someone else's uncommitted output silently is worse than leaving it visible.

7. **Commit signing identity is a placeholder.** `git log --show-signature` returns a good ED25519 signature, but the identity reads `Good "git" signature for YOUR_EMAIL_HERE`. The signature is cryptographically valid; the allowed-signers entry was never filled in. Signed commits therefore verify without attributing to a real identity, which defeats much of the point. Worth correcting in `.gitconfig` / the allowed-signers file. Noticed in passing, not changed.

3. **Guard gap: no hook protects the corpus from `DELETE`/`UPDATE`.** The guardrails cover dangerous shell commands — `sudo` was correctly blocked this session — but nothing would have stopped `delete_31.sql` from being executed during a read-only phase. The discipline here was procedural, not enforced. Worth a hook that refuses non-`SELECT` SQL against `boarddocs-postgres` unless an approval marker is present. **Reported, not used.**

4. **`boarddocs-postgres` still publishes `0.0.0.0:5432`.** Unchanged from the diagnostic and re-flagged: this repair confirms the corpus behind that port is the production corpus. Check `firewall-cmd --list-ports`; correct the quadlet to `127.0.0.1:5432:5432`.

5. **Stale `.env` Postgres credential.** Unchanged; host tooling still cannot authenticate.

6. **The scraper question is unresolved where it matters.** Nothing on Smeltor can re-run the flat scraper, but the machine that ran it on 2026-02-22 has not been identified and may still have both the script and a schedule. Worth confirming wherever that scraper actually lives.
