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

Nothing else on the host was touched.

**Main session:**

| File | Change |
|---|---|
| `~/workspace/projects/ksd-main/.env` | One line — `POSTGRES_PASSWORD` |
| `~/.config/containers/systemd/civic-postgres.container` | One line — `PublishPort` |
| `~/workspace/projects/ksd_forensic/boarddocs/data` | Renamed to `data_DO_NOT_LOAD`, `chmod -R a-w` |
| `reports/hygiene-2026-09-13.md` | New |
| `docs/session-logs/session-debrief-2026-09-13-hygiene-remediation.md` | New |

**Follow-up, on separate operator approvals:**

| File | Change |
|---|---|
| `…/framework-backup/…/boarddocs/data/boarddocs_scraper.py` | `chmod a-x`, then **moved** to `~/workspace/archive/legacy-flat-scraper/` |
| `~/workspace/archive/legacy-flat-scraper/README.md` | New — provenance, hash, incident reference, do-not-run warning |
| `docs/data-paths.md` | New — canonical data locations and known-stale references |
| `reports/hygiene-2026-09-13.md` | Addendum A1–A6 appended (append-only; §2 given a `SUPERSEDED` pointer) |

The guarded tree (`ksd-boarddocs-rag`) was read only via `grep`. Its `.env` was not touched. Production was not contacted. The backup corpus received **no new file and no tombstone** — the only change to it was the removal of the scraper.

## Findings

**1. The `.env` password was not stale — it was a different secret entirely.** Container credential: 13 characters, `sha256=a6a775f61e77`. `.env` credential: 32 characters, `sha256=6f7f2f77cd2e`. That is not truncation or a half-landed rotation; the development `.env` was carrying a secret belonging to something else, most plausibly a generated production credential. Host-side `psql ... -c 'select 1'` now returns `1`, exit 0.

Two consequences worth separating. First, **the overwritten 32-character value is gone from this host** — no backup was made (correctly). It survives only wherever it was originally issued. Second, **the working credential is 13 characters and sits in plaintext in the quadlet unit file** (`Environment=POSTGRES_PASSWORD=`). Until this session it was also LAN-reachable. Fixing the bind address does not fix the credential.

**2. The bind address was exactly as reported. The firewall question was open at time of writing and has since been closed — there was no exposure window.** `ss -ltnp` showed `*:5432` before the change and `127.0.0.1:5432` after, with an explicit wildcard test passing. The container is healthy, `select 1` still works, and `count(*) from documents` returns 20,197 — data intact. During the session `firewall-cmd --state` and `--list-ports` both returned `Authorization failed` (root required, no sudo), so the exposure window could not be sized from inside the session.

**Resolved post-session:** the operator ran `firewall-cmd --list-all` in a terminal — zone `FedoraServer`, ports empty, services `ssh`/`cockpit`/`dhcpv6-client` only, rootless container. **5432/tcp was never permitted inbound**, so the `0.0.0.0` publish was a socket-layer exposure only. Classification: **latent misconfiguration corrected during routine hygiene, not a security incident.** Full evidence and the two independent corroborating facts (point-to-point /30 segment; firewalld continuously active) are in Open Item 5, along with the one thing this does not establish — that the verdict rests on firewall policy, not on connection records, because none exist.

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

1. **Commit succeeded — no operator action needed.** Committed as `9dff46c` on `claude/feat-facts-marks`* with only this session's two files staged:
   ```
   git add reports/hygiene-2026-09-13.md \
           docs/session-logs/session-debrief-2026-09-13-hygiene-remediation.md
   git commit -m "docs: close four hygiene items (credential, bind address, data tree quarantine)"
   ```
   All pre-commit hooks passed, including `detect private key` and `Detect hardcoded secrets` — independent confirmation that no credential leaked into the report or this debrief. **The signing-identity gap from the Part 1 session is confirmed fixed**: `git log --show-signature` reports `Good "git" signature for donald@qorvault.com with ED25519 key SHA256:BixbZ0qL...`, status `G`. Nothing was pushed (the hook blocks `git push`; pushing remains the operator's call).

   *Branch is `claude/feat-facts-minutes`.

   Two things this commit does **not** cover. The quadlet change (`~/.config/containers/systemd/civic-postgres.container`) is **outside any git repo** — the loopback fix is unversioned and will be lost by any process that regenerates that file. And the working tree still carries untracked files from prior sessions: `reports/ingest-degradation-2026-09-13.md` and three `docs/session-logs/` debriefs (2026-09-08 ×2, 2026-09-13 ingest diagnostic).

2. **Legacy flat scraper — CLOSED.** Relocated out of the authoritative corpus on operator approval.

   **New path:** `~/workspace/archive/legacy-flat-scraper/boarddocs_scraper.py`
   **Original path:** `…/framework-backup/home/ksd_forensic/boarddocs/data/boarddocs_scraper.py`

   Same-filesystem rename (btrfs, device `37`, subvol `/root`), so this is the same file rather than a copy. Verified identical before and after:

   | Property | Before | After |
   |---|---|---|
   | SHA-256 | `877529ae3e6f3e9fdb20681a4decee54f5854d482a4a42557e24460e7d20539f` | identical |
   | mtime | `2026-02-22 07:07:05.409386920 -0800` (epoch `1771772825`) | identical |
   | inode | `9672210` | identical |
   | size | 46,461 | identical |
   | perms | `-rw-r--r--` | identical (non-executable) |

   Backup corpus verified clean afterwards: **1,684** meeting directories, **806** `agenda.html`, scraper gone, **no tombstone and no new file** created inside it. `~/workspace/archive/legacy-flat-scraper/README.md` documents original path, hash, mtime, incident reference and the do-not-run warning. `docs/data-paths.md` records the new location.

   **Checksum manifest: no impact.** The only genuine manifest in the archive (`…/ksd-boarddocs-rag/backups/2026-04-03/manifest.json`) is a 228-character database backup summary — `timestamp`, `postgres_dump_size_bytes`, `qdrant_snapshot_size_bytes`, and document/chunk/vector counts. It contains no file paths and no reference to the scraper or the data tree, so **this move invalidates nothing.** Not edited. All other `*manifest*`/`*checksum*` hits under the backup tree are Chromium and pip cache artefacts, unrelated to the corpus.

   **Left in place deliberately: `boarddocs_scraper.log`**, still at the original location, now the only top-level file in that data directory. It is the run log of the incident, carries no execution risk, and moving it was outside the approval. Recommend deciding whether it should follow the scraper into the archive — see Open Item 11 for why it is now more valuable than it looked.

   Retained for the record: `chmod a-x` alone (applied earlier) was **not** sufficient — `./boarddocs_scraper.py` was blocked but `python3 boarddocs_scraper.py` still ran. Relocation, not the permission bit, is what closed this.

3. **Guarded-tree `.env` — operator edit required.** `~/workspace/projects/ksd-boarddocs-rag/.env` was not touched, per instruction. If it carries the same mismatched credential, host-side tooling there still cannot authenticate. The working value is in the container environment and in `civic-postgres.container`. Compare without printing:
   ```
   cd ~/workspace/projects/ksd-boarddocs-rag
   grep '^POSTGRES_PASSWORD=' .env | cut -d= -f2- | tr -d '\n' | sha256sum | cut -c1-12
   # working credential hashes to: a6a775f61e77
   ```

4. **Hard-coded path references — operator edits, 11 sites.** Listed with file:line in `reports/hygiene-2026-09-13.md` §3.1. Four are in the guarded tree (`boarddocs_loader/config.py:29`, `README.md:23` and `:35`, `CLAUDE.md:19`), four in `ksd-main` (same four files, `/home/qorvault/` variant), three in `ksd_forensic/scripts/` (`boarddocs_api_scrape.py:36`, `boarddocs_update.py:52`, `load_scraped_meetings.py:24`). **None currently resolves.** Decide the canonical data path first — most likely the 1,684-meeting backup — then correct all eleven together, and correct the "1,682 meetings" claim in both `CLAUDE.md` files at the same time.

5. **Firewall question — CLOSED. Verdict: latent misconfiguration, no exposure window.**

   Operator ran `firewall-cmd --list-all` in a terminal (root required; `/etc/firewalld` is `drwxr-x--- root root` and there is no non-root read path). Result:

   | Property | Value |
   |---|---|
   | Zone | `FedoraServer` |
   | Interfaces | `enp9s0`, `enp10s0` |
   | Ports | *(empty)* |
   | Services | `ssh`, `cockpit`, `dhcpv6-client` |
   | Container | rootless |

   **5432/tcp was never permitted inbound.** The `0.0.0.0` publish exposed the port at the socket layer only; firewalld dropped inbound traffic to it on both interfaces for the entire period. This is a **latent misconfiguration corrected during routine hygiene, not a security incident.** No exposure window, no scope question, no notification consideration.

   Two supporting facts, gathered without root, that independently narrow the same conclusion:

   - **The segment is point-to-point, not a LAN.** Smeltor is `10.10.3.2/30` on `enp10s0`, default route via `10.10.3.1`. A /30 carries exactly two usable addresses, so there is precisely **one** possible on-segment neighbour — the gateway. The phrasing "the corpus is reachable from the LAN" in `reports/ingest-degradation-2026-09-13.md`, repeated in `reports/hygiene-2026-09-13.md` §2, **overstates the topology** and should be read with this correction. Any reach would have required deliberate routing by `10.10.3.1`.
   - **firewalld ran continuously.** `active` and `enabled`, with a clean start at boot on 2026-09-08 in the journal. No window where the service was stopped.

   `enp9s0` is in the same zone but carried no address at audit time (`ip -br addr` showed only `enp10s0` up). Same policy applies if it is ever brought up — no action needed.

   **The one thing this does not close: there is no connection record, and there never was.** The verdict above rests on firewall policy, not on observed traffic. Postgres logging is configured such that a successful off-host connection would have left no trace at all:

   ```
   log_connections    | off
   log_disconnections | off
   log_line_prefix    | %m [%p]      <- no %h, so no client address
   listen_addresses   | *
   ssl                | off
   ```

   Four `password authentication failed for user "boarddocs"` entries exist in the journal (2026-09-08 ×2, 2026-09-13 19:20 and 20:30). They align exactly with the known `.env` credential mismatch — two in the Sep 8 smoke-test session, two in the Sep 13 diagnostic session, which documented its failed host-side attempt in its own appendix. That is the innocent explanation and it fits the timeline. **It is not proof**, because without `%h` a local failure is indistinguishable from a remote one. The firewall policy is what makes the verdict safe; the logs could not have supported it either way.

   Also noted from the same query, neither urgent given the corrected bind: `ssl = off` (traffic unencrypted in transit — low impact for public meeting records), and `listen_addresses = *` inside the container, meaning Postgres itself offers no second line of defence if the `PublishPort` line is ever reverted.

6. **Recommended change (NOT executed) — enable connection logging so this is answerable from logs next time.**

   The gap above is the real lesson from this item: the reachability question had to be settled by reading firewall policy because no connection record existed. Closing that gap costs one restart.

   In `~/.config/containers/systemd/civic-postgres.container`, pass the settings to the server process:

   ```ini
   [Container]
   Exec=postgres -c log_connections=on \
                 -c log_disconnections=on \
                 -c "log_line_prefix=%%m [%%p] %%h %%u@%%d "
   ```

   **The doubled `%%` is required and is the easy thing to get wrong.** systemd treats `%` as a specifier prefix in unit files, so a literal `%m` is consumed by systemd before Postgres ever sees it and the prefix silently comes out malformed. Quadlet files are unit files. If the escaping proves awkward, the alternative is a mounted `postgresql.conf` fragment via `Volume=` plus `-c config_file=`, which avoids systemd's parser entirely — slightly more moving parts, no escaping trap.

   `%h` is the field that was missing and the reason the four auth failures are uninterpretable. Consider `log_disconnections` optional; `log_connections` plus `%h` is the minimum that makes a future "who connected?" answerable.

   **Before applying:** this needs a container restart, so the same stop-rule from Step 2 applies — check `pg_stat_activity` for connections other than your own first, and do not restart while anything is writing. Verify afterward with `SELECT name, setting FROM pg_settings WHERE name IN ('log_connections','log_line_prefix');` rather than assuming the quadlet took effect.

   Not executed: modifying Postgres server configuration is outside the scope authorized for this session, which was `.env`, the `PublishPort` line, and the data tree.

7. **Rotate the database credential, and get it out of the unit file.** 13 characters in plaintext in `civic-postgres.container`. Rotating means changing it in the container, the quadlet, and both `.env` files together. Consider `Environment=POSTGRES_PASSWORD` sourced from a root-only `EnvironmentFile=` rather than inline in the unit.

   Priority note: this was the sharpest item while the bind looked exposed. With Open Item 5 closed — firewalld never permitted 5432 inbound — a short plaintext credential on a loopback-only, rootless container is a hygiene item, not an urgent one. Still worth doing on the next maintenance pass.

8. **~~LAN clients will now fail.~~ Resolved by the Open Item 5 evidence — no action needed.** This was raised on the assumption that something off-host might have been using 5432 and would break when the bind moved to loopback. The firewall verdict removes the premise: if 5432 was never permitted inbound, **no off-host client could have been connecting in the first place**, so the bind change cannot have broken one. Consistent with the pre-restart check, which found only this session's own backend. Retained rather than deleted so the reasoning is visible.

   If a LAN client is ever *wanted*, the correct mechanism is an SSH tunnel, not reverting `PublishPort`.

9. **Hook coverage gap.** `~/.claude/hooks/block-dangerous-commands.sh` blocks `cat|head|tail|less|more|base64|xxd` against `.env`/`.pem`/`.key`, but not `sed`, `awk`, `python3`, `sort`, or `grep`. A rule keyed on the target file extension regardless of reading tool would close it. Hook not modified.

10. **Rename the quadlet for clarity.** `civic-postgres.container` creates a container named `boarddocs-postgres`. Same mismatch likely exists for `civic-qdrant.container` → `boarddocs-qdrant`. Cosmetic, but it costs search time during an incident.

11. **NEW — the 2026-02-22 run was far larger than the diagnostic records. Recommend a correction to `reports/ingest-degradation-2026-09-13.md`.**

    Found while writing the archive README, from the scraper's own run log.

    §0.2 of the diagnostic dates the run to **2026-02-22 07:02–07:05** — inferred from the mtimes of the four affected 2026 meeting directories — and frames it as "a second, older scraper was run against 2026 meetings." The log shows otherwise:

    ```
    first: 2026-02-22 06:43:20   last: 2026-02-22 11:19:07
    9,126 lines / 2.2 MB / 806 distinct meeting slugs
    ```

    **A 4.5-hour run across 806 meetings, not a 3-minute run across four.** Slug years: 2005–2017 heavily, 2018 × 30, 2026 × 9.

    Corroborated independently by the corpus. `agenda.html` retention is unique to this scraper, and the corpus holds **806** `agenda.html` files whose year distribution matches the log **exactly, year for year** (2018 → 30, 2026 → 9), with **all 806 carrying mtime `2026-02-22`**. Conclusion: **every `agenda.html` in the corpus was written by this single run** — they are not survivors of a historic flat-scraper era, as §0.2's retention table implies.

    Two consequences:

    - **The 71-record damage figure still stands.** 2005–2018 meetings were *already* flat in the corpus, so the flat `external_id` matched and `ON CONFLICT DO NOTHING` made those a genuine no-op. Collision required a meeting previously scraped *structured*, which in practice meant 2026 only. Wider blast radius, same record count.
    - **R1 and R4 Route 2 depend on this run's output.** Both parse `agenda.html` from `documents.content_raw`, and that HTML came from this scraper on 2026-02-22. Recovery remains sound — one consistent snapshot of intact BoardDocs markup — but it is **not an independent source**, and any systematic flaw in this run's capture is inherited by all 806. Worth a sampled spot-check against live BoardDocs before R4 goes bulk, folded into the 5 URL confirmations already gated on R4.

    Per the project's append-only convention, publish as a correction rather than editing §0.2. Not written — this is a finding, not an authorised edit to a prior report.
