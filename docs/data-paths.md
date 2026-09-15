# Data paths — canonical reference

**Last updated:** 2026-09-13
**Host:** Smeltor

Created because every data path recorded in this codebase — in `config.py` defaults, in
`README.md` examples, in both `CLAUDE.md` files, and in `documents.file_path` in PostgreSQL —
points somewhere that does not exist on this host. This file is the single place to check
before running anything that reads or writes corpus data.

> **If a path here disagrees with a path in code or in `CLAUDE.md`, this file is correct and
> the code is stale.** See *Known-stale references* below. Nothing has been auto-corrected;
> those edits are pending an operator decision on the canonical path.

---

## Authoritative corpus

```
/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data
```

- **1,684 meeting directories**, 806 with `agenda.html`
- This is the complete corpus and the one the database was built from
- Treat as **read-only**. Nothing should write here

Caveat on the `agenda.html` files: all 806 were written by the legacy flat scraper in a single
run on 2026-02-22 and all carry that mtime. They are a consistent snapshot of intact BoardDocs
markup — the recovery routes R1 and R4 depend on them — but they are not an independent
source. See `~/workspace/archive/legacy-flat-scraper/README.md`.

## Quarantined — half corpus, DO NOT LOAD

```
/home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD
```

- **729 meeting directories** — less than half the corpus — and **zero** `agenda.html` files
- Renamed from `.../boarddocs/data` on 2026-09-13 and set read-only (`chmod -R a-w`)
- Loading this instead of the authoritative corpus would silently ingest a partial dataset
- To reverse if ever needed: `chmod -R u+w <path>` then rename back

## Quarantined — legacy flat scraper and its run log

```
/home/donald/workspace/archive/legacy-flat-scraper/
```

The scraper that caused the 2026-02-22 ingest degradation (71 unlinked attachment records),
and the log of the run that did it. Both moved here on **2026-09-13** from inside the
authoritative corpus, where the script was executable and both sat loose in a data directory.

**This directory is deliberately outside git.** The hashes below are the versioned record of
provenance — verify against them rather than trusting the files alone.

| File | SHA-256 | Size | mtime | inode |
|---|---|---|---|---|
| `boarddocs_scraper.py` | `877529ae3e6f3e9fdb20681a4decee54f5854d482a4a42557e24460e7d20539f` | 46,461 | `2026-02-22 07:07:05.409386920 -0800` | `9672210` |
| `boarddocs_scraper.log` | `6bb4e41a3f76123ec81e38e1d18338ed85715853811dc0786076ede7bf599760` | 2,262,477 | `2026-02-22 11:19:07.268286800 -0800` | `9672209` |

Verify with:

```bash
cd ~/workspace/archive/legacy-flat-scraper && sha256sum -c <<'EOF'
877529ae3e6f3e9fdb20681a4decee54f5854d482a4a42557e24460e7d20539f  boarddocs_scraper.py
6bb4e41a3f76123ec81e38e1d18338ed85715853811dc0786076ede7bf599760  boarddocs_scraper.log
EOF
```

Both moves were same-filesystem renames (btrfs, device `37`, subvol `/root`), so inodes
carried over — these are the original files, not copies. Both are stored non-executable.

**Run summary from the log** — 4h36m, `2026-02-22 06:43:20` → `11:19:07`, 9,126 lines,
**806 distinct meeting slugs** (2005–2017 heavily, 2018 × 30, 2026 × 9). This is materially
wider than the 07:02–07:05 / four-meeting window recorded in
`reports/ingest-degradation-2026-09-13.md` §0.2, and it is the reason all 806 `agenda.html`
files in the corpus share mtime `2026-02-22`.

- **Do not run the scraper against any corpus.** Superseded by the structured scraper
- Full context, including why `chmod a-x` alone was insufficient:
  `~/workspace/archive/legacy-flat-scraper/README.md`

**Original location, now empty of both files:**
`…/framework-backup/home/ksd_forensic/boarddocs/data/` — that directory holds **only** the
1,684 meeting directories, with zero top-level files and no tombstone.

## Supported scraper

```
/home/donald/workspace/projects/ksd_forensic/boarddocs-scraper/
```

TypeScript/Puppeteer, produces the structured per-agenda-item layout the loader needs in order
to set `agenda_item_id`. This is the tool to use. If it fails, fix it rather than falling back
to the legacy flat scraper — that fallback is what caused the incident.

## Services

| Service | Address | Notes |
|---|---|---|
| PostgreSQL | `127.0.0.1:5432` | db/user `boarddocs`; loopback-only since 2026-09-13 |
| Qdrant | `127.0.0.1:6333` | |
| RAG API | `127.0.0.1:8000` | |

Always use `127.0.0.1`, never `localhost` — Fedora resolves `localhost` to IPv6 first and the
containers bind IPv4 only.

---

## Known-stale references

None of the three path variants below resolves on this host. **Not corrected** — the canonical
path must be decided first (most likely the authoritative corpus above), then all of them
changed together, along with the "1,682 meetings" claim in both `CLAUDE.md` files.

**Variant A — `/home/donald/projects/ksd_forensic/boarddocs/data`** (missing `workspace/`):

| File:line |
|---|
| `ksd-boarddocs-rag/boarddocs_loader/boarddocs_loader/config.py:29` |
| `ksd-boarddocs-rag/boarddocs_loader/README.md:23`, `:35` |
| `ksd-boarddocs-rag/CLAUDE.md:19` |
| `ksd_forensic/scripts/boarddocs_api_scrape.py:36` |
| `ksd_forensic/scripts/boarddocs_update.py:52` |
| `ksd_forensic/scripts/load_scraped_meetings.py:24` |

**Variant B — `/home/qorvault/projects/ksd_forensic/boarddocs/data`** (production path on a
dev host):

| File:line |
|---|
| `ksd-main/boarddocs_loader/boarddocs_loader/config.py:29` |
| `ksd-main/boarddocs_loader/README.md:23`, `:35` |
| `ksd-main/CLAUDE.md:25` |

**Variant C — `/home/donald/ksd_forensic/…`** — stored in `documents.file_path` for every row
in PostgreSQL. Not a code reference and unaffected by any filesystem move. Any tooling that
resolves `file_path` from the database will fail on this host until it is rewritten or mapped.

### Non-data stale claims in `CLAUDE.md` (same correction batch)

**`document_processor/venv/` does not exist on Smeltor.** `ksd-main/CLAUDE.md` states *"Doc
processor venv path: `document_processor/venv/` (not `.venv`)"*. That describes the
**production** host. Verified 2026-09-13: neither `venv/` nor `.venv/` existed under
`document_processor/` **or** `embedding_pipeline/`, in `ksd-main` or in the guarded tree.
Running `python3 -m document_processor` with the system interpreter fails at
`ModuleNotFoundError: No module named 'bs4'`.

Current state after 2026-09-14 setup:

| Component | venv | Notes |
|---|---|---|
| `document_processor/` | **created** 2026-09-14 via its own `./setup.sh` | Python 3.14.3, 31 packages, 35/35 tests pass |
| `embedding_pipeline/` | **still missing** | `./setup.sh` not yet run; ONNX `model_cache/` (1.3 GB) is present |

Both components ship a `setup.sh` that creates the venv, installs `requirements.txt` and runs
the tests. Treat `CLAUDE.md`'s venv line as describing production only.

Because the loader's *defaults* are Variants A and B, a default-invocation run on Smeltor
fails immediately with a missing-directory error rather than silently ingesting the wrong
tree. The real risk is a hand-typed `--data-dir`, or someone "fixing" a stale default to the
729-directory tree without noticing it is half a corpus.

---

## Before running the loader

1. Confirm the `--data-dir` you pass is the **authoritative corpus** above, not a default
2. Confirm the target has ~1,684 meeting directories:
   `find <path> -mindepth 1 -maxdepth 1 -type d | wc -l`
3. Never point it at anything ending `_DO_NOT_LOAD`

---

## BoardDocs committee ID

The KSD BoardDocs committee ID for the **Main Governing Board** is `A94NQ8610101`.

It is passed as `current_committee_id` in the POST body of every BoardDocs lookup agent
(`BD-GetMeetingsList`, `BD-GetAgenda`, `BD-GetAgendaItem`, `PRINT-AgendaDetailed`). Omitting
it does **not** produce an error: the agents return HTTP 200 with an empty result set, so the
failure is silent and looks like "this district has no meetings."

| Source | Value | Status |
|---|---|---|
| `ksd_forensic/scripts/boarddocs_update.py:49` | `A94NQ8610101` | **Authoritative** — this scraper ran at scale (806 meetings) |
| `ksd_forensic/scripts/boarddocs_api_scrape.py:34` | `A94NQ8610101` | **Authoritative** — same value, independently |
| `ksd_forensic/scripts/KentScrapes.py:17` | `A4EP5J5A1F8D` | **Unverified — do not use.** Flagged by its own comment: `# Main Board - you may need to verify this` |

Confirmed independently on 2026-09-15 against the live public page, where `A94NQ8610101` is
the `committeeid` attribute on the "Main Governing Board" selector, and by a successful
`BD-GetMeetingsList` call returning all 1,736 meetings.
