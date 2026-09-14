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

2. **Quarantine or remove the legacy flat scraper — still open.** `.../framework-backup/home/ksd_forensic/boarddocs/data/boarddocs_scraper.py`, inside the authoritative backup corpus. This is the tool that caused the 71-record incident.

   **`chmod a-x` was applied** at operator request after the main session (`-rwxr-xr-x` → `-rw-r--r--`; SHA-256 `877529ae3e6f3e9f` unchanged, mtime `2026-02-22 07:07:05` preserved, so the forensic timestamp evidence survives).

   **That is not sufficient on its own, and the caveat matters more than the fix.** For a Python script the execute bit only governs the shebang path. Both paths were tested after the change:

   | Invocation | Result |
   |---|---|
   | `./boarddocs_scraper.py` | blocked |
   | `python3 boarddocs_scraper.py` | **still runs** |

   The realistic way anyone re-runs this — deliberately or by copying a line out of an old shell history — is `python3 <file>`, which is completely unaffected. One vector of two is closed. **Do not read this item as neutralized.**

   The effective remedy is to get the file out of the data directory, since the hazard is that it sits *inside* the 1,684-meeting corpus where tab-completion or a stray `find -exec` reaches it:
   ```
   mkdir -p ~/qorvault-dev-archive/legacy-scrapers
   mv ~/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data/boarddocs_scraper.py \
      ~/qorvault-dev-archive/legacy-scrapers/boarddocs_scraper.py.DO_NOT_RUN
   ```
   Not executed — moving a file inside the backup archive is a larger step than the chmod authorized, and whether that archive stays byte-for-byte as captured is the operator's call.

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
