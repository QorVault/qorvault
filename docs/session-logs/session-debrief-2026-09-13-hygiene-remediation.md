# Session Debrief — 2026-09-13 — hygiene remediation (four carried-over items)

**Session scope**: Close four hygiene items flagged by the ingest degradation diagnostic and the Part 1 repair report — (a) the `.env` `POSTGRES_PASSWORD` that fails from the host, (b) `boarddocs-postgres` publishing `0.0.0.0:5432`, (c) the half-corpus working data tree that could be loaded by mistake, (d) write the production bind-address check list. Setup/hygiene only, with one explicit operator credential grant.
**Branch/project**: `ksd-main` on Smeltor, branch `claude/feat-facts-minutes`. User-level only, no sudo. Report at `reports/hygiene-2026-09-13.md`.

## Decisions

**Compared the two credentials by SHA-256 and length rather than reading either one.** The grant allowed reading the password, but not printing it. Hashing both sides answered "are these the same secret?" without any value reaching the transcript — and it turned the item from "the credential is stale" into "these are two different credentials" (13 chars vs 32), which is a different finding with different consequences.

**Accepted the operator's denial of the `.env` backup rather than working around it.** A `cp -p .env .env.bak` was attempted out of habit and denied. That was the right call and the attempt was the wrong instinct: a `.env.bak` duplicates a live secret onto disk with no access control and no lifecycle. Proceeded with an in-place single-line rewrite guarded by an `assert hits == 1`, so a malformed file would abort rather than be silently mangled.

**Widened the pre-restart connection check beyond what the task specified.** The stop-rule asked for `WHERE datname = current_database()`. Ran that, then re-ran across all databases — a connection from another database would still have been killed by the restart. Both returned only this session's own backend.

**Verified the backup held the full corpus *before* `chmod -R a-w` on the working tree.** Write-protecting 5.2 GB is cheap to reverse, but confirming 1,684 directories and 806 `agenda.html` files existed elsewhere first meant the move could never be the thing that made a copy unrecoverable.

**Reported the hook coverage gap instead of quietly using it.** An early `sed`-based redacted listing of `.env` passed a hook that blocks `cat|head|tail|less|more|base64|xxd` on secret files. Values were redacted and nothing sensitive printed, but `sed`/`awk`/`python3` are not in that pattern list. Written into the report as a gap for the operator rather than relied on again.

## What Changed

Three files modified, two created. Nothing else on the host was touched.

| File | Change |
|---|---|
| `~/workspace/projects/ksd-main/.env` | One line — `POSTGRES_PASSWORD` |
| `~/.config/containers/systemd/civic-postgres.container` | One line — `PublishPort` |
| `~/workspace/projects/ksd_forensic/boarddocs/data` | Renamed to `data_DO_NOT_LOAD`, `chmod -R a-w` |
| `reports/hygiene-2026-09-13.md` | New |
| `docs/session-logs/session-debrief-2026-09-13-hygiene-remediation.md` | New |

The guarded tree (`ksd-boarddocs-rag`) was read only via `grep`. Its `.env` was not touched. Production was not contacted.

## Findings

**1. The `.env` password was not stale — it was a different secret entirely.** Container credential: 13 characters, `sha256=a6a775f61e77`. `.env` credential: 32 characters, `sha256=6f7f2f77cd2e`. That is not truncation or a half-landed rotation; the development `.env` was carrying a secret belonging to something else, most plausibly a generated production credential. Host-side `psql ... -c 'select 1'` now returns `1`, exit 0.

Two consequences worth separating. First, **the overwritten 32-character value is gone from this host** — no backup was made (correctly). It survives only wherever it was originally issued. Second, **the working credential is 13 characters and sits in plaintext in the quadlet unit file** (`Environment=POSTGRES_PASSWORD=`). Until this session it was also LAN-reachable. Fixing the bind address does not fix the credential.

**2. The bind address was exactly as reported, and the firewall question could not be answered.** `ss -ltnp` showed `*:5432` before the change and `127.0.0.1:5432` after, with an explicit wildcard test passing. The container is healthy, `select 1` still works, and `count(*) from documents` returns 20,197 — data intact. But `firewall-cmd --state` and `--list-ports` both returned `Authorization failed` (root required, no sudo this session), so **whether the LAN could actually reach 5432 is still unknown**. The fix is correct regardless; the size of the exposure window is not established.

The unit is named `civic-postgres.container` / `civic-postgres.service` while the container it creates is `boarddocs-postgres`. Grepping for a `boarddocs-postgres` unit finds nothing. Not changed, but it cost time here and will cost it again.

**3. The quarantine closed a theoretical risk, not a live one — and the real exposure is elsewhere.** The tree matched its description exactly (729 directories, zero `agenda.html`, no `chattr +i`) and is now read-only and rejecting writes. But the reference sweep found that **no loader default pointed at it**. Three path variants exist in code and docs, and *none currently resolves on this host*:

| Variant | References | Where |
|---|---|---|
| `/home/donald/workspace/projects/ksd_forensic/boarddocs/data` (the moved path) | 1 | Prior report's appendix only |
| `/home/donald/projects/ksd_forensic/boarddocs/data` | 7 | Guarded tree loader/README/CLAUDE.md; 3 `ksd_forensic` scripts |
| `/home/qorvault/projects/ksd_forensic/boarddocs/data` | 4 | `ksd-main` loader/README/CLAUDE.md |

A default-invocation loader run on Smeltor would have failed with a missing-directory error, not silently ingested half the corpus. The genuine exposure was a hand-typed `--data-dir`, or a future edit "fixing" a stale default to the working path without noticing it holds 729 of 1,684 meetings. The quarantine closes both and the `_DO_NOT_LOAD` suffix is self-documenting — defence in depth, not the removal of an active risk.

The larger issue the sweep exposed: **every recorded data path in this codebase is wrong**, on both trees, and both `CLAUDE.md` files assert a corpus of 1,682 meetings at a path that does not exist. A fourth stale variant (`/home/donald/ksd_forensic/...`) is stored in `documents.file_path` in PostgreSQL and is unaffected by any of this.

**4. The legacy flat scraper that caused the original degradation is still present and still executable.** The prior diagnostic inferred it from `meeting.json` key casing but never located the file. It is at:

```
/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data/boarddocs_scraper.py
-rwxr-xr-x   46,461 bytes   mtime 2026-02-22 07:07:05
```

Four independent signals identify it conclusively: snake_case keys exactly matching the predicted set (`meeting_id`, `slug`, `name`, `source_url`, `scraped_at`, `committee_id`, `files_found`) with zero camelCase keys; retains `agenda.html`; `re.sub(r'\s+', '_', name)` at line 598 producing the spaces-to-underscores filenames; and an mtime two minutes after the 07:02–07:05 scrape run that created the 71 unlinked records.

It is world-executable, it lives **inside a data directory** in the backup archive — the last place a code audit looks, which is why it was not found before — and that archive is the tree holding the authoritative 1,684-meeting corpus. A careless run there contaminates the good copy, not the quarantined one. Per the diagnostic's §1.3, a re-run would silently duplicate records again rather than being a no-op, because the idempotency key encodes layout. Nothing was changed; the backup tree was treated as read-only.

## Workarounds

None. No guard was bypassed and no hook blocked any command this session. The `firewall-cmd` authorization failure was left failed rather than escalated — the answer is deferred to the operator, not worked around.

## Unfinished / Deferred

**Step 4 was written, not executed**, as specified — `reports/hygiene-2026-09-13.md` §4 holds the production discovery commands covering: bind address (`ss`, `podman ps`), where the publish is configured (quadlet / compose / `podman run`), firewall state (the question that could not be answered here), historical off-host connections in `pg_stat_activity` and `podman logs`, the apply-and-verify sequence, and — added beyond the brief — checks for the same plaintext-password and stale-data-path issues on production.

Deliberately not done: no hard-coded path was edited (operator's, per instruction); the guarded tree's `.env` was not touched; the flat scraper was not moved, renamed, or chmod'd; nothing was committed.

## Open Items

1. **Commit is pending operator action.** Nothing was committed this session. To commit:
   ```
   cd ~/workspace/projects/ksd-main
   git add reports/hygiene-2026-09-13.md \
           docs/session-logs/session-debrief-2026-09-13-hygiene-remediation.md
   git commit -m "docs: close four hygiene items (credential, bind address, data tree quarantine)"
   ```
   Note the quadlet change (`~/.config/containers/systemd/civic-postgres.container`) is **outside any git repo** and will not be captured by that commit. The working tree also still carries the untracked files from prior sessions: `reports/ingest-degradation-2026-09-13.md` and three `docs/session-logs/` debriefs (2026-09-08 ×2, 2026-09-13 ingest diagnostic).

2. **Quarantine or remove the legacy flat scraper — highest priority.** `.../framework-backup/home/ksd_forensic/boarddocs/data/boarddocs_scraper.py`, `-rwxr-xr-x`, inside the authoritative backup corpus. Suggested minimum: `chmod a-x`. Better: move it out of the data directory into a clearly-marked `legacy/` path so it cannot be run by autocomplete or a stray `find -exec`. This is the tool that caused the 71-record incident and nothing currently prevents a repeat.

3. **Guarded-tree `.env` — operator edit required.** `~/workspace/projects/ksd-boarddocs-rag/.env` was not touched, per instruction. If it carries the same mismatched credential, host-side tooling there still cannot authenticate. The working value is in the container environment and in `civic-postgres.container`. Compare without printing:
   ```
   cd ~/workspace/projects/ksd-boarddocs-rag
   grep '^POSTGRES_PASSWORD=' .env | cut -d= -f2- | tr -d '\n' | sha256sum | cut -c1-12
   # working credential hashes to: a6a775f61e77
   ```

4. **Hard-coded path references — operator edits, 11 sites.** Listed with file:line in `reports/hygiene-2026-09-13.md` §3.1. Four are in the guarded tree (`boarddocs_loader/config.py:29`, `README.md:23` and `:35`, `CLAUDE.md:19`), four in `ksd-main` (same four files, `/home/qorvault/` variant), three in `ksd_forensic/scripts/` (`boarddocs_api_scrape.py:36`, `boarddocs_update.py:52`, `load_scraped_meetings.py:24`). **None currently resolves.** Decide the canonical data path first — most likely the 1,684-meeting backup — then correct all eleven together, and correct the "1,682 meetings" claim in both `CLAUDE.md` files at the same time.

5. **Answer the firewall question.** Needs root: `sudo firewall-cmd --list-ports` and `sudo firewall-cmd --list-all` on Smeltor. Determines whether the corpus was actually LAN-reachable before today's fix, which decides whether this was a latent misconfiguration or an actual exposure worth treating as an incident.

6. **Rotate the database credential, and get it out of the unit file.** 13 characters in plaintext in `civic-postgres.container`. Rotating means changing it in the container, the quadlet, and both `.env` files together. Consider `Environment=POSTGRES_PASSWORD` sourced from a root-only `EnvironmentFile=` rather than inline in the unit.

7. **LAN clients will now fail.** If anything off-host was connecting to 5432 — the Framework Desktop is the obvious candidate — it is disconnected by design as of this session. Nothing was connected at change time, so nothing in flight broke, but an intermittent or scheduled client will fail on its next run. The correct fix is an SSH tunnel, not reverting the bind. Rollback if genuinely needed: set `PublishPort=0.0.0.0:5432:5432`, `systemctl --user daemon-reload`, restart.

8. **Hook coverage gap.** `~/.claude/hooks/block-dangerous-commands.sh` blocks `cat|head|tail|less|more|base64|xxd` against `.env`/`.pem`/`.key`, but not `sed`, `awk`, `python3`, `sort`, or `grep`. A rule keyed on the target file extension regardless of reading tool would close it. Hook not modified.

9. **Rename the quadlet for clarity.** `civic-postgres.container` creates a container named `boarddocs-postgres`. Same mismatch likely exists for `civic-qdrant.container` → `boarddocs-qdrant`. Cosmetic, but it costs search time during an incident.
