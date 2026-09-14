# BoardDocs Coverage Audit — 2026-09-12

**Scope**: Read-only diagnostic. Does the ksd-main corpus match what is published on BoardDocs?
**Project**: `~/workspace/projects/ksd-main`
**Status**: **PHASE 0 COMPLETE — PHASE 1 NOT RUN (blocked, zero network requests made)**
**Request budget used**: 0 of 400.

---

## Executive summary

Phase 0 (corpus-side inventory) completed in full and produced a clear picture of what
the corpus contains. Phase 1 (site-side listing) **was not attempted**: every tool capable
of making the HTTP requests is blocked by a security hook, and the task's own rules forbid
bypassing hooks. No BoardDocs requests were made.

The headline result is that **the corpus cannot be verified against the site today**, but
Phase 0 alone surfaced five substantive coverage defects that do not require the network to
confirm. The most consequential of these is structural, not a matter of a few missing files:

> **The corpus has no agenda-item layer at all before 2018.** For 2005–2017 — thirteen years,
> roughly 40% of the corpus by document count — there are 725 meeting agendas and 6,163
> attachments, and **zero** agenda items. Item-level retrieval, and any attachment-to-item
> attribution, is impossible for that entire period.

The confirmed ESSER gap is real and is now characterised precisely: ESSER money is visible
throughout the corpus as a *funding source line* inside other items, but **no ESSER
allocation or acceptance instrument exists in the corpus** under any title.

---

## Phase 0 — Corpus-side inventory

### 0.1 Totals

| Metric | Value |
|---|---|
| Documents | 20,197 |
| Chunks | 179,081 |
| Distinct `meeting_id` | 1,626 |
| Distinct meeting dates | 1,061 |
| Documents with NULL `meeting_date` | 15 |
| Oldest meeting | **2005-03-23** |
| Newest meeting | **2026-03-25** |
| Tenants | `kent_sd` only (20,197) |

### 0.2 Documents by type

| Type | Docs | Complete | Not complete |
|---|---|---|---|
| attachment | 12,305 | 12,206 | 99 |
| agenda_item | 6,529 | 6,481 | 48 |
| agenda | 951 | 951 | 0 |
| transcript | 348 | 348 | 0 |
| email | 45 | 45 | 0 |
| ospi_data | 15 | 15 | 0 |
| research_analysis | 4 | 4 | 0 |

The last four types are not BoardDocs-sourced and are out of scope for a site-coverage
comparison; the BoardDocs-derived corpus is 19,785 documents.

### 0.3 Meetings and documents per year

| Year | Meetings | Distinct dates | Agendas | Agenda items | Attachments |
|---|---|---|---|---|---|
| 2005 | 28 | 27 | 28 | **0** | 444 |
| 2006 | 35 | 33 | 35 | **0** | 449 |
| 2007 | 34 | 32 | 34 | **0** | 481 |
| 2008 | 47 | 40 | 47 | **0** | 496 |
| 2009 | 57 | 43 | 57 | **0** | 383 |
| 2010 | 77 | 45 | 77 | **0** | 443 |
| 2011 | 60 | 34 | 60 | **0** | 509 |
| 2012 | 54 | 35 | 54 | **0** | 482 |
| 2013 | 63 | 39 | 63 | **0** | 587 |
| 2014 | 70 | 41 | 70 | **0** | 565 |
| 2015 | 75 | 48 | 75 | **0** | 511 |
| 2016 | 76 | 47 | 76 | **0** | 514 |
| 2017 | 91 | 58 | 91 | **0** | 465 |
| 2018 | 68 | 45 | 33 | 317 | 509 |
| 2019 | 89 | 45 | 3 | 903 | 672 |
| 2020 | 92 | 57 | 4 | 855 | 727 |
| 2021 | 109 | 67 | 29 | 858 | 778 |
| 2022 | 127 | 74 | 17 | 902 | 729 |
| 2023 | 134 | 80 | 33 | 918 | 766 |
| 2024 | 115 | 86 | 25 | 742 | 683 |
| 2025 | 100 | 68 | 27 | 847 | 884 |
| 2026 | 25 | 17 | 13 | 187 | 228 |

**2018 is the format changeover.** Before it, the scrape captured a flat `agenda.txt` per
meeting plus loose attachments. After it, the scrape captured per-item records. This is a
known design note in CLAUDE.md ("Data has two formats"), but its retrieval consequence has
not previously been quantified: **6,163 pre-2018 attachments have no parent item**, so a
question like "which agenda item authorised this contract in 2012" is unanswerable from the
corpus regardless of how good the retrieval is.

### 0.4 Years with fewer than 12 regular meetings

**None**, other than the partial current year. Regular-meeting counts run 13–21 per year for
2005–2025. 2026 shows 5, consistent with a corpus that ends 2026-03-25.

| Year | Regular | Special | Work session |
|---|---|---|---|
| 2018 | 13 | 34 | 9 |
| 2019 | 18 | 41 | 16 |
| 2020 | 17 | 48 | 25 |
| 2021 | 19 | 49 | 25 |
| 2022 | 17 | 53 | 31 |
| 2023 | 18 | 43 | 11 |
| 2024 | 18 | 41 | 5 |
| 2025 | 18 | 29 | 2 |
| 2026 | 5 | 5 | 2 |

Full per-year table in Appendix A, query A4. No year is flagged for the Phase 1 "every
meeting in a flagged year" sample.

### 0.5 Agenda items per meeting

| Metric | Value |
|---|---|
| Meetings with ≥1 item | 678 |
| Min / median / mean / max items | 1 / 3 / 9.6 / 65 |
| Meetings 2019+ with an agenda but **zero** items | **148** |

The median of 3 against a mean of 9.6 reflects the large number of short special/executive
sessions. The 148 zero-item meetings are the ones worth sampling against the site — some are
legitimately empty (executive sessions), but that assumption has never been tested.

### 0.6 Attachments per agenda item

| Metric | Value |
|---|---|
| Attachments linked to an item | 5,637 |
| Attachments with **no** item link | 6,668 |
| Attachments with no meeting at all | 44 |
| Items carrying ≥1 attachment | 3,137 of 6,529 (48%) |
| Mean / max attachments per item | 1.80 / 46 |
| **Orphans (item_id set but no such item)** | **0** |

Referential integrity is clean where linkage exists. The 6,668 unlinked attachments are
overwhelmingly the pre-2018 flat-format ones — but not entirely:

| Year | Unlinked | Linked |
|---|---|---|
| 2005–2017 | 6,329 | 0 |
| 2018 | 224 | 285 |
| 2019–2023 | 0 | 3,672 |
| 2024 | 26 | 657 |
| 2025 | 16 | 868 |
| 2026 | **73** | 155 |

**2024–2026 show flat-format attachments reappearing**, and 2026 is nearly a third unlinked
(73 of 228). The structured scraper is degrading on recent meetings. This is a live
regression, not a legacy artifact.

### 0.7 Documents by title-pattern category

| Category | Docs | Attachments | Agenda items | Agendas |
|---|---|---|---|---|
| other | 13,776 | 7,804 | 4,632 | 944 |
| policy | 1,704 | 1,085 | 615 | 3 |
| minutes | 1,167 | 874 | 293 | 0 |
| presentation | 820 | 774 | 44 | 2 |
| contract/agreement | 766 | 375 | 391 | 0 |
| voucher/warrant | 723 | 635 | 88 | 0 |
| resolution | 549 | 377 | 172 | 0 |
| agenda | 326 | 132 | 191 | 0 |
| budget | 201 | 157 | 30 | 2 |
| financial statement | 165 | 92 | 73 | 0 |

"other" at 68% is high. Attachment titles are raw filenames (`1708CE~1.DOC.pdf`,
`TVF_Vouchers_02-11-26.pdf`), so title-pattern classification is inherently weak here — this
table should be read as a floor on each category, not a measurement. Categorisation would be
substantially better done off extracted text than off filenames.

Split by era, to show the pre-2018 material is not categorically different — only
structurally flatter:

| Era | minutes | resolution | voucher | budget | contract |
|---|---|---|---|---|---|
| pre-2018 (flat) | 467 | 233 | 199 | 123 | 124 |
| 2018+ (structured) | 700 | 317 | 550 | 205 | 666 |

### 0.8 Dev fidelity block (for operator comparison against production)

| Metric | Dev value | Verified by |
|---|---|---|
| Documents by type | see §0.2 (20,197 total) | Postgres |
| Chunk count | **179,081** | Postgres |
| Chunks with a Qdrant point ID | 179,081 (100%, all distinct) | Postgres |
| Embedding status | `complete` for 100% | Postgres |
| Embedding model in metadata | **`mxbai-embed-large-v1`**, single value across all chunks | Postgres |
| Newest meeting date | **2026-03-25** | Postgres |
| Qdrant collection | `boarddocs_chunks`, **1024-dim, Cosine**, `on_disk_payload: true`, HNSW m=16 ef_construct=100 | Qdrant `config.json` on disk |
| Qdrant collection size on disk | 1.3 GB, 1 shard, 3 segments | filesystem |
| **Qdrant live point count** | **NOT PRODUCED — see below** | — |

**Per the task's stop rule, this is reported rather than worked around.** The live point
count requires an HTTP call to `127.0.0.1:6333`, and every available mechanism for that is
blocked (§ Blockers). The Postgres side asserts **179,081** points should exist, and every
chunk holds a distinct `qdrant_point_id`, so 179,081 is the expected figure — but it is an
expectation derived from the writer's own bookkeeping, not a reading of Qdrant. It will not
detect points lost on Qdrant's side. Treat it as unconfirmed.

**CLAUDE.md is stale on all of these.** It documents 19,775 records / 116,667 chunks /
168,950 Qdrant points, and Postgres credentials `qorvault`/`qorvault`. The live database is
`boarddocs`/`boarddocs` and the counts above are materially higher. See Open Items.

---

## Phase 1 — Site-side listing: NOT RUN

**Zero network requests were made. The full 400-request budget is unspent.**

Phase 1 requires `curl --http1.1` against BoardDocs. The PreToolUse hook
`~/.claude/hooks/block-dangerous-commands.sh` blocks it at line 34:

```
if echo "$CMD_LOWER" | grep -qP '(?<!\w)(curl|wget|nc|ncat)(?!\w)'; then
    echo "BLOCKED: network tool detected (curl/wget/nc/ncat). Use a dedicated tool or ask the user." >&2
```

and blocks the Python fallback at line 99:

```
if echo "$CMD_LOWER" | grep -qP 'python3?\s+-c\s' && \
   echo "$CMD_LOWER" | grep -qP 'import\s+(urllib|requests|socket|http\.client|ftplib|smtplib|paramiko)'; then
    echo "BLOCKED: python one-liner with network library import." >&2
```

The hook's own comment states it is written to catch evasion — *"including inside python -c
strings, subshells, pipes, quoted fragments like cu\"\"rl"*. A route that would technically
slip past it exists (a Python **script file** rather than `-c`), and it was deliberately not
used: the task rules say hooks are never bypassed, and the global CLAUDE.md says to explain
the block and ask the operator rather than work around it. Nothing was attempted against
BoardDocs.

**This is a permissions decision for the operator, not a technical failure.** Phase 1 needs
an explicit, narrow allowance — see Recommendations.

### Partial substitute performed with no network

The local scrape snapshot at `~/workspace/projects/ksd_forensic/boarddocs/data/` (729 meeting
directories, 2019-08-28 → 2026-03-25) was compared against the database. This is **not** the
site — it is the same ad-hoc scrape the corpus was built from, so it cannot detect anything
the scrape itself missed, which is precisely the question Phase 1 exists to answer. It can
only catch losses *between* scrape and load. It found two:

| Scraped meeting | Content on disk | In DB? |
|---|---|---|
| `2020-02-22-special-meeting-work-session-8-00-a-m-` | `meeting.json` only, no items | No |
| `2024-07-01-board-members-attending-ksd-levy-listening-session-monday-july-1-2024` | `meeting.json` + **item.json files** | **No** |

The 2020-02-22 entry is an empty shell and is probably a legitimate skip. **The 2024-07-01
levy listening session has real item content on disk that was never loaded** — a genuine
ingest loss.

38 database meeting dates in the same window are absent from the scrape directory, indicating
at least one other ingest path fed the corpus. That path has not been identified; the
pre-2019 material (897 meetings) has no local source directory at all.

---

## Targeted: ESSER / CARES / ARP / budget instruments

### Method note — a false positive corrected

An initial case-insensitive substring search for `esser` returned 467 documents dating back
to 2005, which would have contradicted the known gap. Those are matches inside the word
**"lesser"** (126 documents contain it). Re-run with a word boundary and case sensitivity
(`~ '\mESSER\M'`), the count is **277 documents, none before 2020** — which is what a real
ESSER footprint looks like. All figures below use the strict pattern.

### Title-level search (the instrument itself)

| Term | Documents with term **in title** |
|---|---|
| ESSER | **2** |
| CARES | **0** |
| ARP / American Rescue | 2 |
| budget extension | 6 |
| grant acceptance | 16 |
| budget amendment | 2 |

The only two ESSER-titled documents in the entire corpus:

| Date | Type | Title |
|---|---|---|
| 2023-12-13 | attachment | `KSD Drinking Fountain ESSER -- Bid Tab.pdf` |
| 2023-12-13 | attachment | `KSD Drinking Fountain ESSER  Kent Medium construction contract form 12-14-23.pdf` |

Both are project-level bid documents for a single drinking-fountain installation. **This
confirms the gap exactly as previously reported.**

### Body-text search (where the money actually appears)

277 documents mention ESSER by name, first appearing in 2020:

| Year | agenda_item | attachment | transcript | other |
|---|---|---|---|---|
| 2020 | 0 | 5 | 0 | 0 |
| 2021 | 11 | 49 | 18 | 0 |
| 2022 | 10 | 42 | 15 | 0 |
| 2023 | 5 | 39 | 11 | 0 |
| 2024 | 0 | 32 | 9 | 2 email |
| 2025 | 0 | 17 | 2 | 1 email |
| 2026 | 0 | 6 | 1 | 3 research |

The 26 agenda items that mention ESSER are all **spending** items where ESSER is named as the
funding source — laptop repairs, Verizon wireless, hybrid learning technology, air filters,
Hazel Health, library programs, technology refreshes, musical instruments. Not one is an
allocation instrument.

### Missing-instrument list

The following should exist for a district of Kent's size and **are not in the corpus in any
form**:

| Instrument | Expected window | In corpus? | Notes |
|---|---|---|---|
| ESSER I / CARES Act award acceptance | Spring–Summer 2020 | **No** | Zero CARES mentions in any title corpus-wide |
| ESSER II / CRRSA award acceptance | Winter–Spring 2021 | **No** | — |
| ESSER III / ARP award acceptance | Spring–Summer 2021 | **No** | Closest is an *application* item, not an acceptance |
| ESSER expenditure plan / use-of-funds plan | 2021 | **No** | Federally required and subject to public comment |
| ESSER quarterly or annual reporting | 2021–2024 | **No** | — |

The nearest surviving artifacts are an **application** and an unrelated city pass-through:

| Date | Type | Title |
|---|---|---|
| 2021-11-10 | agenda_item | Request to Apply for the American Rescue Plan Act Grant |
| 2024-09-25 | agenda_item | City of Kent American Rescue Plan Act (ARPA) Federal Grant Fund Acceptance 2024-2026 |

Adjacent budget instruments that *are* present, which show the corpus does capture this class
of document when it was scraped — making the ESSER absence more likely a real gap in the
scrape or on the site than a categorisation artifact:

| Date | Type | Title |
|---|---|---|
| 2020-05-13 | attachment | Resolution 1586 DSF Budget Extension May 13 2020.pdf |
| 2021-04-28 | attachment | Resolution 1604 TVF GF Budget Extension FY21.pdf |
| 2021-04-28 | agenda_item | Resolution No. 1604 Extension of 2020-2021 General Fund and Transportation Vehicle Fund |
| 2021-08-25 | agenda_item | 2021-2022 District Budget and 2020-2021 Budget Extension |
| 2022-05-11 | agenda_item | Resolution No. 1624 - Budget Extension for Fiscal Year 2021-22 |

**Note that Resolutions 1604 and 1624 are budget extensions in exactly the period ESSER money
was landing.** A budget extension is the mechanism a district uses to book unbudgeted federal
revenue. These are the most likely documents to name the ESSER award amounts, and they are in
the corpus — the text of those two resolutions is the highest-value place to look before
assuming the allocation record is absent from the site entirely.

### BoardDocs file IDs — mostly unavailable

The task asked for file IDs where visible. **`source_url` is NULL for all 12,305
attachments**, and for the 45 emails and 4 research documents:

| Type | Docs | With `source_url` |
|---|---|---|
| attachment | 12,305 | **0** |
| agenda_item | 6,529 | 6,529 |
| agenda | 951 | 951 |
| transcript | 348 | 348 |

Attachment BoardDocs IDs are partially recoverable from `external_id`, which encodes
`<meeting_id>_<item_id>_<filename>` — e.g. `DQU2PT0331CA_DQU2Q30331E4_Budget Update 2.4.26.pdf`
yields meeting `DQU2PT0331CA` and item `DQU2Q30331E4`. But the pre-2018 flat attachments carry
only `<meeting_id>_<filename>`, and no attachment has a resolvable direct URL. **Citations to
attachments cannot currently link back to the source document on BoardDocs** — worth fixing
independent of this audit, since it undercuts the transparency purpose of the tool.

---

## Coverage estimate

Stated honestly: **an internal-consistency measurement, not a site-coverage measurement.**
Phase 1 did not run, so nothing below is validated against BoardDocs. Sample sizes are given
as required, but every sample here is drawn from the corpus and its own scrape snapshot.

| Dimension | Finding | Basis / sample size |
|---|---|---|
| Meeting-level, 2019-08-28 → 2026-03-25 | 2 of 424 scraped meeting dates absent from DB (99.5% loaded) | full comparison, N=424 dates |
| Meeting-level, 2005 → 2019-08 | **Unmeasurable** | no local source; site required |
| Agenda-item layer, 2005–2017 | **0% coverage** — the layer does not exist | full corpus, N=725 meetings |
| Agenda-item layer, 2019–2023 | Structurally complete | full corpus |
| Agenda-item layer, 2024–2026 | Degrading — 115 of 1,680 attachments unlinked | full corpus |
| Attachment→item integrity | 100% where linkage exists, 0 orphans | full corpus, N=5,637 |
| Text extraction | 99.5% complete (99 failed/deferred of 19,785) | full corpus |
| Embedding | 100% of 179,081 chunks, single model | full corpus |
| ESSER allocation instruments | **0 found**; 5 expected instrument classes absent | full-corpus title + body search |

**Overall: the corpus is internally consistent and well-processed, but its coverage of the
site is unknown for the 2005–2019 period and unverified everywhere.**

---

## Blockers

**1. Network access to BoardDocs — blocks all of Phase 1.**
`~/.claude/hooks/block-dangerous-commands.sh` lines 34 and 99. Not bypassed. Operator
decision required.

**2. HTTP access to local Qdrant — blocks the live point count in the dev-fidelity block.**
Same hook, same lines. `127.0.0.1:6333` is a local container, not egress, but the hook does
not distinguish by destination.

**3. rag_api is not running.** Nothing is listening on 127.0.0.1:8000 or :8001; the only
uvicorn process is Open WebUI on :8080. The task authorised read-only checks against rag_api;
none were possible. Start command is in CLAUDE.md if the operator wants this covered.

**4. `.env` read denied** by the configured `Read(**/.env)` deny rule. Not worked around. The
database role was obtained from the container's own config (`POSTGRES_USER`/`POSTGRES_DB`),
with the password field filtered out of the output.

---

## Recommendations (all require operator approval)

### Immediate, no network needed

1. **Read Resolutions 1604 and 1624 out of the corpus.** They are budget extensions from the
   exact period ESSER funds arrived and are already ingested. If they name the ESSER award
   amounts, the allocation record may be present in substance under a different title, which
   would change the conclusion of this audit. Cost: zero requests, one query.
2. **Load the 2024-07-01 levy listening session.** Item content sits on disk unloaded.
3. **Resolve the 48 pending 2026 agenda items.** All 48 unprocessed documents are in 2026 —
   a recent ingest stalled and was never retried.
4. **Correct CLAUDE.md's stale figures and credentials** (see Open Items). Not a code change,
   but it actively misleads.

### Requires a hook allowance

5. **Qdrant point count.** Narrowest possible fix: allow HTTP to `127.0.0.1:6333` only. This
   closes the one missing dev-fidelity metric.
6. **Phase 1 proper.** Needs an allowance for `curl --http1.1` to the BoardDocs host.

### Re-scrape priority, if Phase 1 confirms gaps

Ordered by civic value per request, at the stated rate limit of ≥6 s between requests:

| Priority | Target | Est. requests | Est. wall time | Rationale |
|---|---|---|---|---|
| 1 | ESSER-window meeting listings, Mar 2020 – Dec 2021 | ~120 | ~12 min | Directly tests the known gap |
| 2 | 2024–2026 meetings with unlinked attachments | ~115 | ~12 min | Fixes a *live* regression, not history |
| 3 | 2018 transition-year re-scrape in structured mode | ~68 | ~7 min | Recovers the item layer for a partial year cheaply |
| 4 | 2005–2017 item-layer backfill | ~725 meetings + items | **many hours, multi-session** | Largest gap, but only if BoardDocs still serves item structure for that era — verify on a 5-meeting probe first |

**Priority 4 should not be committed to before a 5-request probe** establishes whether
BoardDocs exposes per-item structure for 2005-era meetings at all. If it does not, the
pre-2018 item layer is permanently unavailable and the corpus should document that limitation
rather than chase it.

A full Phase 1 as originally specified — meeting enumeration, 40-meeting stratified sample,
60 agenda-item attachment comparisons, plus targeted searches — fits inside the 400-request
budget at roughly 250–300 requests, about 30 minutes of wall time at the 6-second floor.

---

## Open items

**CLAUDE.md is materially stale — MEDIUM.** Documented Postgres credentials (`qorvault`/
`qorvault`) do not exist; the live role and database are both `boarddocs`. Documented corpus
figures (19,775 records, 116,667 chunks, 168,950 points) are all superseded (20,197 / 179,081
/ unverified). Documented paths are `/home/qorvault/projects/...` while the working tree is
`/home/donald/workspace/projects/...`. Anyone following CLAUDE.md's commands verbatim gets a
connection failure on the first step.

**Postgres is listening on all interfaces — SECURITY, MEDIUM.** `ss -lntp` shows
`*:5432`, and the container publishes `0.0.0.0:5432->5432/tcp`. Qdrant, vLLM and Open WebUI
are all correctly bound to `127.0.0.1`. CLAUDE.md's own rule states services must bind
127.0.0.1. The corpus is public-records material so disclosure risk is low, but the database
also holds `user_sessions`, `access_requests` and `session_queries` — who searched for what —
which is not public-records material. Recommend rebinding to `127.0.0.1:5432`.

**Attachment `source_url` is universally NULL — MEDIUM.** 12,305 of 12,305. Attachment
citations cannot link back to BoardDocs. Undercuts the transparency goal of the project and
should be captured on any future scrape.

**The pre-2019 ingest path is unidentified — LOW, but blocks future re-scrapes.** 897
meetings predate the local scrape snapshot and no source directory for them was found under
`ksd_forensic`. Whoever re-scrapes will need to know how that material arrived.

**The cited basis document does not exist — LOW.** The task cites
`session-debrief-2026-09-08-retrieval-smoke-test.md`. No file of that name exists anywhere
under `~/`; the newest session log in `~/meta-forge/session-logs/` is dated 2026-04-05, and
there are no 2026-09 logs. The ESSER gap it described is independently confirmed by this
audit, so the substance holds, but the record itself is missing.

---

## Appendix A — every command run

All Postgres queries were read-only `SELECT`. No writes, no ingestion, no file fetches, no
network requests.

### Environment

```bash
# A1 — project orientation
cd ~/workspace/projects/ksd-main && pwd && ls -la
ls ~/workspace/projects/

# A2 — container state
podman ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
podman ps -a --format '{{.Names}}\t{{.Status}}'

# A3 — database role discovery (password field excluded from output)
podman inspect boarddocs-postgres --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^POSTGRES_(USER|DB)='

# A4 — listening sockets / rag_api presence
ss -lntp | grep -E ':(8000|8001|6333|5432|8080|3000)'
pgrep -af "uvicorn|rag_api"

# A5 — schema
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\dt"
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\d documents"
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -c "\d chunks"
```

### Phase 0 queries

All run as:
`podman exec -i boarddocs-postgres psql -U boarddocs -d boarddocs <<'SQL'`

```sql
-- A6 — documents by type and processing status
SELECT document_type, count(*) AS docs,
       count(*) FILTER (WHERE processing_status='complete') AS complete,
       count(*) FILTER (WHERE processing_status<>'complete') AS not_complete
FROM documents WHERE tenant_id='kent_sd' GROUP BY 1 ORDER BY 2 DESC;

-- A7 — tenants
SELECT tenant_id, count(*) FROM documents GROUP BY 1;

-- A8 — totals
SELECT (SELECT count(*) FROM documents) AS documents,
       (SELECT count(*) FROM chunks) AS chunks,
       (SELECT count(*) FROM document_pages) AS pages;

-- A9 — meeting date range
SELECT min(meeting_date) AS oldest, max(meeting_date) AS newest,
       count(DISTINCT meeting_id) AS distinct_meeting_ids,
       count(DISTINCT meeting_date) AS distinct_meeting_dates,
       count(*) FILTER (WHERE meeting_date IS NULL) AS null_dates
FROM documents;

-- A10 — meeting types
SELECT coalesce(committee_name,'(null)') AS committee,
       count(DISTINCT meeting_id) AS meetings, count(*) AS docs
FROM documents GROUP BY 1 ORDER BY 2 DESC NULLS LAST LIMIT 40;

-- A11 — meetings and docs per year
SELECT extract(year FROM meeting_date)::int AS yr,
       count(DISTINCT meeting_id) AS meetings,
       count(DISTINCT meeting_date) AS distinct_dates,
       count(*) FILTER (WHERE document_type='agenda') AS agendas,
       count(*) FILTER (WHERE document_type='agenda_item') AS agenda_items,
       count(*) FILTER (WHERE document_type='attachment') AS attachments
FROM documents WHERE meeting_date IS NOT NULL GROUP BY 1 ORDER BY 1;

-- A12 — regular meetings per year (the <12 flag)
SELECT extract(year FROM meeting_date)::int AS yr,
       count(DISTINCT meeting_date) FILTER (WHERE committee_name='Regular Meeting') AS regular_dates,
       count(DISTINCT meeting_id)   FILTER (WHERE committee_name='Regular Meeting') AS regular_mtgs,
       count(DISTINCT meeting_date) FILTER (WHERE committee_name='Special Meeting') AS special_dates,
       count(DISTINCT meeting_date) FILTER (WHERE committee_name='Work Session')    AS work_dates
FROM documents WHERE meeting_date IS NOT NULL GROUP BY 1 ORDER BY 1;

-- A13 — attachments per agenda item
WITH att AS (
  SELECT agenda_item_id, count(*) c FROM documents
  WHERE document_type='attachment' AND agenda_item_id IS NOT NULL GROUP BY 1)
SELECT count(*) AS items_with_attachments, sum(c) AS total_attachments,
       round(avg(c),2) AS avg_per_item, max(c) AS max_per_item FROM att;

-- A14 — attachment linkage
SELECT count(*) FILTER (WHERE agenda_item_id IS NOT NULL) AS att_linked_to_item,
       count(*) FILTER (WHERE agenda_item_id IS NULL)     AS att_orphan_no_item,
       count(*) FILTER (WHERE meeting_id IS NULL)         AS att_no_meeting
FROM documents WHERE document_type='attachment';

-- A15 — identifier format inspection (explains the corrected join below)
SELECT external_id, meeting_id, agenda_item_id, left(title,50) AS title
FROM documents WHERE document_type='agenda_item' ORDER BY meeting_date DESC LIMIT 5;
SELECT external_id, meeting_id, agenda_item_id, left(title,50) AS title
FROM documents WHERE document_type='attachment' AND agenda_item_id IS NOT NULL
ORDER BY meeting_date DESC LIMIT 5;
SELECT external_id, meeting_id, agenda_item_id, left(title,50) AS title
FROM documents WHERE document_type='attachment' AND agenda_item_id IS NULL
ORDER BY meeting_date DESC LIMIT 5;

-- A16 — CORRECTED items-with-attachments join.
-- NOTE: an earlier version joined a.agenda_item_id = i.external_id and returned 0,
-- which looked like a total integrity failure. It was a bad join: agenda_item.external_id
-- is prefixed 'boarddocs:kent_sd:item:<ID>' while attachments carry the bare <ID>.
-- The correct key is agenda_item_id on both sides.
SELECT count(*) AS total_items,
  count(*) FILTER (WHERE EXISTS (SELECT 1 FROM documents a
      WHERE a.document_type='attachment' AND a.agenda_item_id=i.agenda_item_id)) AS items_with_att
FROM documents i WHERE i.document_type='agenda_item';

-- A17 — true orphan check
SELECT count(*) FROM documents a
WHERE a.document_type='attachment' AND a.agenda_item_id IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM documents i
      WHERE i.document_type='agenda_item' AND i.agenda_item_id=a.agenda_item_id);

-- A18 — flat vs linked attachments by year
SELECT extract(year FROM meeting_date)::int AS yr,
       count(*) FILTER (WHERE agenda_item_id IS NULL) AS flat_att,
       count(*) FILTER (WHERE agenda_item_id IS NOT NULL) AS linked_att
FROM documents WHERE document_type='attachment' AND meeting_date IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- A19 — title-pattern categories
WITH cat AS (
  SELECT CASE
    WHEN title ~* '(minutes)'                                   THEN 'minutes'
    WHEN title ~* '(agenda)'                                    THEN 'agenda'
    WHEN title ~* '(resolution)'                                THEN 'resolution'
    WHEN title ~* '(policy|procedure|^[0-9]{4}(P|F)? *-)'       THEN 'policy'
    WHEN title ~* '(contract|agreement|mou|interlocal|lease|amendment to)' THEN 'contract/agreement'
    WHEN title ~* '(voucher|warrant|payroll)'                   THEN 'voucher/warrant'
    WHEN title ~* '(financial (statement|report)|budget status|monthly financial|fiscal report)' THEN 'financial statement'
    WHEN title ~* '(budget)'                                    THEN 'budget'
    WHEN title ~* '(presentation|slide|powerpoint|\.pptx?$|\.ppsx$)' THEN 'presentation'
    ELSE 'other' END AS category, document_type
  FROM documents WHERE title IS NOT NULL)
SELECT category, count(*) AS docs,
  count(*) FILTER (WHERE document_type='attachment') AS attachments,
  count(*) FILTER (WHERE document_type='agenda_item') AS agenda_items,
  count(*) FILTER (WHERE document_type='agenda') AS agendas
FROM cat GROUP BY 1 ORDER BY 2 DESC;

-- A20 — category by era
SELECT CASE WHEN meeting_date < '2018-01-01' THEN 'pre-2018 (flat)'
            ELSE '2018+ (structured)' END AS era,
  count(*) FILTER (WHERE title ~* 'minutes')            AS minutes,
  count(*) FILTER (WHERE title ~* 'resolution')         AS resolution,
  count(*) FILTER (WHERE title ~* 'voucher|warrant')    AS vouchers,
  count(*) FILTER (WHERE title ~* 'budget')             AS budget,
  count(*) FILTER (WHERE title ~* 'contract|agreement') AS contract
FROM documents WHERE meeting_date IS NOT NULL GROUP BY 1;

-- A21 — dev fidelity: embedding status and model
SELECT embedding_status, coalesce(embedding_model,'(null)') AS model,
       count(*) AS chunks, count(DISTINCT qdrant_point_id) AS distinct_point_ids
FROM chunks GROUP BY 1,2 ORDER BY 3 DESC;

SELECT count(*) AS chunks_total, count(qdrant_point_id) AS chunks_with_point_id,
       count(DISTINCT document_id) AS docs_with_chunks FROM chunks;

-- A22 — documents with no chunks
SELECT d.document_type, d.processing_status, count(*)
FROM documents d WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id=d.id)
GROUP BY 1,2 ORDER BY 3 DESC;

-- A23 — agenda items per meeting
WITH per AS (SELECT meeting_id, count(*) c FROM documents
  WHERE document_type='agenda_item' GROUP BY 1)
SELECT count(*) AS meetings_with_items, min(c) AS min_items, round(avg(c),1) AS avg_items,
  max(c) AS max_items, percentile_cont(0.5) WITHIN GROUP (ORDER BY c) AS median FROM per;

-- A24 — 2019+ meetings with an agenda but zero items
SELECT count(DISTINCT d.meeting_id) FROM documents d
WHERE d.meeting_date >= '2019-01-01'
AND NOT EXISTS (SELECT 1 FROM documents i
    WHERE i.document_type='agenda_item' AND i.meeting_id=d.meeting_id);

-- A25 — source_url coverage
SELECT document_type, count(*) AS docs, count(source_url) AS with_url
FROM documents GROUP BY 1 ORDER BY 2 DESC;

-- A26 — pending agenda items by year
SELECT extract(year FROM meeting_date)::int AS yr, count(*)
FROM documents WHERE document_type='agenda_item' AND processing_status='pending'
GROUP BY 1 ORDER BY 1;
```

### Targeted ESSER queries

```sql
-- A27 — title-level instrument search
SELECT
  count(*) FILTER (WHERE title ~* 'esser')            AS esser,
  count(*) FILTER (WHERE title ~* 'cares act|cares')  AS cares,
  count(*) FILTER (WHERE title ~* '\marp\M|american rescue') AS arp,
  count(*) FILTER (WHERE title ~* 'budget extension') AS budget_extension,
  count(*) FILTER (WHERE title ~* 'grant acceptance|accept.*grant') AS grant_acceptance,
  count(*) FILTER (WHERE title ~* 'budget amendment')  AS budget_amendment
FROM documents;

SELECT document_type, meeting_date, left(title,90) AS title
FROM documents WHERE title ~* 'esser|cares act|american rescue|\marp\M'
ORDER BY meeting_date LIMIT 60;

-- A28 — FIRST BODY SEARCH: substring match, produced FALSE POSITIVES.
-- Returned 467 docs back to 2005 by matching 'esser' inside 'lesser'. Superseded by A29.
SELECT
  count(*) FILTER (WHERE content_text ~* 'esser')           AS esser_body,
  count(*) FILTER (WHERE content_text ~* 'cares act')       AS cares_body,
  count(*) FILTER (WHERE content_text ~* 'american rescue') AS arp_body,
  count(*) FILTER (WHERE content_text ~* 'elementary and secondary school emergency relief')
                                                            AS esser_longform
FROM documents;

-- A29 — CORRECTED body search: word boundary + case sensitive
SELECT extract(year FROM meeting_date)::int AS yr, document_type, count(*)
FROM documents
WHERE content_text ~ '\mESSER\M'
   OR content_text ~* '\melementary and secondary school emergency relief\M'
GROUP BY 1,2 ORDER BY 1,2;

SELECT count(*) FILTER (WHERE content_text ~ '\mESSER\M')                  AS esser_strict,
       count(*) FILTER (WHERE content_text ~* '\mESSER *(I{1,3}|1|2|3)\M') AS esser_numbered,
       count(*) FILTER (WHERE content_text ~* 'lesser')                    AS lesser_falsepos
FROM documents;

-- A30 — budget instruments present in corpus
SELECT meeting_date, document_type, left(title,80) AS title
FROM documents
WHERE title ~* 'budget extension|budget amendment|grant acceptance|accept.*grant'
ORDER BY meeting_date;

-- A31 — agenda items that mention ESSER in body
SELECT meeting_date, document_type, left(title,85) AS title
FROM documents WHERE content_text ~ '\mESSER\M' AND document_type IN ('agenda_item')
ORDER BY meeting_date LIMIT 30;
```

### Qdrant — on-disk inspection only (no HTTP)

```bash
# A32
podman inspect boarddocs-qdrant --format '{{range .Mounts}}{{println .Source " -> " .Destination}}{{end}}'
Q=/home/donald/.local/share/containers/storage/volumes/boarddocs_qdrant_storage/_data
ls "$Q"; ls "$Q/collections"
find "$Q/collections" -maxdepth 2 -name "*.json"
du -sh "$Q/collections/boarddocs_chunks"
ls "$Q/collections/boarddocs_chunks"/0/segments
# config.json read with the Read tool
```

### Local scrape snapshot comparison (no network)

```bash
# A33
ls ~/workspace/projects/ksd_forensic/boarddocs/data | wc -l
ls ~/workspace/projects/ksd_forensic/boarddocs/data | cut -d- -f1 | sort | uniq -c

# A34 — scrape vs DB date comparison
cd ~/workspace/projects/ksd_forensic/boarddocs/data
ls | sed -E 's/^([0-9]{4}-[0-9]{2}-[0-9]{2}).*/\1/' | sort -u > /tmp/scrape_dates.txt
podman exec boarddocs-postgres psql -U boarddocs -d boarddocs -tAc \
  "SELECT DISTINCT meeting_date::text FROM documents WHERE meeting_date >= '2019-08-28' ORDER BY 1;" \
  > /tmp/db_dates.txt
comm -23 /tmp/scrape_dates.txt /tmp/db_dates.txt   # scraped, not in DB
comm -13 /tmp/scrape_dates.txt /tmp/db_dates.txt   # in DB, not scraped

# A35 — inspect the two unloaded meetings
D=~/workspace/projects/ksd_forensic/boarddocs/data
for d in 2020-02-22 2024-07-01; do ls -d $D/${d}*; find $D/${d}* -type f; done
```

### Blocked — attempted, refused, not worked around

```bash
# A36 — BLOCKED by block-dangerous-commands.sh line 34
curl -sf http://127.0.0.1:8000/health
curl -sf http://127.0.0.1:6333/collections
# → "BLOCKED: network tool detected (curl/wget/nc/ncat)."

# A37 — BLOCKED by the Read(**/.env) deny rule
cat .env    # attempted as: sed 's/=.*/=<redacted>/' .env
# → denied; role obtained from container config instead (A3)

# A38 — NOT ATTEMPTED, by rule
#   All BoardDocs HTTP requests for Phase 1.
#   Zero of the 400-request budget consumed.
```
