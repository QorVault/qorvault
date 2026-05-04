# OSPI Authoritative Data-Layer Roadmap

**Author:** Claude Code review (Opus 4.7, 1M context)
**Date:** 2026-05-04
**Repo HEAD reference:** `8f08baa900732da6cb7e10bc941a5c87c78629cd` (origin/main, post-PR #33)
**Status:** DRAFT for Donald's review. Prepared for a dedicated docs-only PR.
No code written.

This memo establishes the architecture for turning the OSPI corpus —
586,204,574 bytes of state-reported district JSON across 10 datasets and
698,222 records, as confirmed by the companion STC profile report — into a
trusted, citeable, queryable fact layer for QorVault.

**Strategic frame:** OSPI is the most authoritative structured dataset
QorVault has access to. It should be made reliable as a fact layer **before**
weaker sources (live agenda packets, transcripts, generated summaries) build
analyses that depend on OSPI numbers. This memo is the planning artifact
that makes the OSPI work explicit, gated, and dev-only first.

---

## Table of contents

- 1. Existing implementation and current limits
- 2. Corpus inventory and dataset roles
- 3. Dataset-by-dataset analytical value
- 4. Record identity strategy
- 5. Provenance model
- 6. Versioning model
- 7. Manifest dependency
- 8. Schema / profile strategy
- 9. SQL vs RAG split
- 10. Query / citation requirements
- 11. Dev-only ingestion design
- 12. Validation gates
- 13. Test strategy
- 14. Risks and open decisions
- 15. Staged issue backlog
- 16. Relationship to Issue #27
- 17. Plain-English owner decisions

---

## 1. Existing implementation and current limits

**What exists today:**

- `scripts/validate_ospi_manifest.py` — read-only manifest validator. Produces
  a deterministic JSON manifest describing every `.json` file under a
  corpus-root path: SHA-256, top-level shape, record count, schema sample,
  `record_id_preview`, deterministic `corpus_file_id`, schema_summary.
  Refuses output paths inside the corpus. Does not mutate inputs.
- `tests/test_validate_ospi_manifest.py` — 11 synthetic-data tests including
  the just-merged trailing-bytes guard (PR #33).
- `.github/workflows/stage1-no-secrets-ci.yml` — Stage 1 CI runs Ruff + the
  validator's tests on PR. Read-only; no secrets; non-required.
- The 10 OSPI JSON files live developer-side at
  `/mnt/qorvault-dev/repo/research/ospi_data/` (gitignored via
  `research/ospi_data/*.json`) and a parallel copy at
  `/mnt/qorvault-dev/repo/docs/ospi_data/` (also gitignored as of PR #28).
- First-pass discovery profile completed by STC on 2026-05-04:
  `docs/ai/ospi/ospi_corpus_profile_20260504T203334Z.md`. This was read-only,
  did not ingest data, did not write to a database, and did not modify corpus
  files.

**What does not exist:**

- No schema-locked, versioned Pydantic per-dataset profile yet. The STC
  Markdown profile is discovery evidence, not the future executable contract.
- No schema models, no Pydantic contracts.
- No record-identity strategy beyond the validator's `corpus_file_id` and
  preview record fingerprints.
- No staging tables, no normalized tables, no migrations.
- No query/citation layer.
- No SQL facts library.
- No tests for ingestion or query because the upstream layers don't exist.
- No mapping from OSPI facts to BoardDocs agenda items.

**Current limits:**

The validator was scoped explicitly as "describe what's in the corpus
without touching it." It tells us the shape; it does not give us a fact
layer. Building the fact layer is the work this memo plans.

---

## 2. Corpus inventory and dataset roles

| Dataset | File | Role |
|---|---|---|
| Assessment | `assessment.json` | Standardized test results: SBA (ELA, math), WCAS (science). The headline academic-outcome dataset. |
| Attendance | `attendance.json` | Regular attendance rates. Tracks a federal accountability indicator. |
| Discipline | `discipline.json` | Disciplinary incident rates, especially exclusionary discipline. Equity-sensitive. |
| Enrollment | `enrollment.json` | Student head counts by school, grade, and student group. The denominator behind most rate calculations. |
| Graduation | `graduation.json` | 4-year and (where reported) 5-year graduation rates by student group. |
| Growth | `growth.json` | Student Growth Percentile (SGP) – measures year-over-year academic growth, separate from absolute achievement. |
| SQSS | `sqss.json` | School Quality and Student Success indicators – Washington's federal-accountability composite. Identifies schools needing support. |
| Teacher demographics | `teacher_demographics.json` | Workforce demographic distribution. Useful for representation analysis. |
| Teacher experience | `teacher_experience.json` | Years-of-experience distribution; novice-teacher rate. Useful for retention analysis. |
| WaKIDS | `wakids.json` | Kindergarten readiness across six developmental domains. Earliest-grade indicator. |

**Confirmed STC profile totals**
(`docs/ai/ospi/ospi_corpus_profile_20260504T203334Z.md`):
586,204,574 bytes across 10/10 expected files; 698,222 records; 0 byte
delta; 0 record delta; no missing expected files; no unexpected JSON files;
all 10 top-level shapes are arrays, and all dataset schema summaries report
0 nested paths.

**Roles in QorVault analyses:**

- **Outcome metrics:** assessment, growth, graduation, WaKIDS.
- **Behavior metrics:** attendance, discipline.
- **Structural metrics:** enrollment, SQSS.
- **Workforce metrics:** teacher demographics, teacher experience.

---

## 3. Dataset-by-dataset analytical value

For each dataset, the *kind* of question it can answer and the *quality* of
that answer when grounded in OSPI data rather than scraped agenda summaries.

- **assessment** — "What share of KSD students met standard in math at the
  elementary level in 2024-25?" Authoritative for cohort/year/subject. Watch
  for suppressed cells in small subgroups.
- **attendance** — "Has regular attendance recovered since 2021-22?" Strong
  for trend analysis at school and district level. Definition changes over
  time; year-on-year comparisons need definitional check.
- **discipline** — "Are exclusionary discipline events declining and is the
  decline equitable across student groups?" Equity questions often hinge on
  small-cell suppression; some answers will be "OSPI suppressed."
- **enrollment** — "How has KSD's overall enrollment changed over five
  years, and how does that vary by grade?" Cleanest dataset; few suppression
  issues at school level.
- **graduation** — "Is graduation rate improving for student groups?"
  Subject to small-cohort suppression; needs careful group selection.
- **growth** — "Is academic growth keeping pace with achievement levels?"
  SGP is a percentile rank; values do not aggregate the way percentages do.
  Need careful interpretation in any answer.
- **sqss** — "Which KSD schools are flagged for support and what tier?"
  Tier labels change format across years; year-aware decoding is required.
- **teacher_demographics** — "Does the workforce reflect the student body's
  demographics?" Side-by-side comparison with enrollment is the natural
  question.
- **teacher_experience** — "Has the share of novice teachers increased?"
  Useful for retention storylines.
- **wakids** — "Are entering kindergartners meeting the readiness benchmark
  across six domains?" Six-domain structure means a single dataset can
  produce six near-independent answers per cohort/year/group.

These analytical roles inform Section 9 (what belongs in SQL vs. RAG).

---

## 4. Record identity strategy

Every OSPI fact must be reachable by a stable, deterministic identity that
survives corpus snapshots and schema evolution.

**Composite identity:**

```
ospi_record_id = stable_hash(
    dataset,                  # e.g. "assessment"
    reporting_year,           # canonicalized e.g. "2024-2025"
    district_code,            # OSPI district code, zero-padded
    school_code | NULL,       # OSPI school code, zero-padded; NULL for district-level rows
    student_group | NULL,     # canonicalized student-group code; NULL for "All Students"
    measure_dims,             # dataset-specific dims (subject, grade band, indicator type)
    record_payload_hash       # SHA-256 of the canonicalized record JSON
)
```

**Why each component:**

- `dataset` keeps records from the same dimensions but different files
  distinguishable.
- `reporting_year` is canonicalized to a single format (`YYYY-YYYY`) with
  validation against acceptable historical formats (decision O3).
- `district_code` and `school_code` use OSPI's stable codes, zero-padded
  for sort-friendly storage.
- `student_group` is canonicalized to a controlled vocabulary (decision O4).
- `measure_dims` is a dataset-specific tuple recording the secondary
  dimensions that distinguish two records of the same year/school/group
  (e.g., subject + grade-band for assessment).
- `record_payload_hash` detects changes to a record's contents under the
  same identity (revision detection).

**Stable across reruns:** the identity is a deterministic function of
inputs. Rerunning the profile or ingestion produces the same `ospi_record_id`
for the same source row.

**Linkage to validator output:** the manifest's `record_id_preview` and
`corpus_file_id` are content-addressed by file and record-index plus
canonicalized JSON. Profile and ingestion layers consume those as inputs
to the composite identity above.

---

## 5. Provenance model

Every fact carries the full provenance chain. Anything missing a chain
element is treated as untrusted and cannot ship in a citation.

| Field | Where it comes from | Why it matters |
|---|---|---|
| `source_file_path` | manifest `relative_path` | Reproducibility — "where on disk." |
| `source_file_sha256` | manifest `sha256` | Bit-exact integrity check. |
| `manifest_id` | manifest `corpus_file_id` (or new manifest-level ID) | Ties fact to a frozen manifest snapshot. |
| `manifest_sha256` | hash of the manifest itself | Detects manifest drift. |
| `profile_run_id` | profile job UUID | Which profile run produced the schema/typing. |
| `ingestion_run_id` | ingestion job UUID | Which run wrote the fact to dev DB. |
| `schema_version` | Pydantic model version (e.g. `ospi.assessment/v1`) | Enables schema migration safely. |
| `record_payload_hash` | SHA-256 of canonical record JSON | Revision detection. |
| `extraction_timestamp` | UTC instant the fact was derived | Audit log. |
| `corpus_snapshot_version` | versioned snapshot of the dataset (Section 6) | Year/release awareness. |

**Citation surface:** when an answer cites an OSPI fact, the JSON returned
to a caller must include a minimum subset of these fields — see Section 10.

---

## 6. Versioning model

Five versioned things, each independent:

| What | Why versioned | Lifecycle |
|---|---|---|
| **Corpus snapshot** | OSPI publishes new annual data; old data may be revised. | Each download produces a snapshot id (date + manifest hash). Old snapshots remain queryable. |
| **Manifest** | The validator output is the gate to profiling and ingestion. | One per snapshot. Pinned by hash. |
| **Schema version** | Field sets evolve; codes get retired. | Per dataset (`ospi.assessment/v1` etc.). Bump when meaning changes. |
| **Ingestion run version** | Each run stamps a version on the rows it writes. | New version per run; old rows preserved. |
| **Query/citation contract version** | Public surface of the citation API. | Bump when fields change shape. Old contract supported until callers migrate. |

**Snapshot lifecycle policy** (decision O5): keep all historical snapshots
in dev DB unless storage forces eviction. A historical answer cited
against snapshot `2026-04-30T00:00Z` should remain re-derivable indefinitely.

---

## 7. Manifest dependency

The validator's manifest is the **gate** to everything downstream.

**Future schema-locked profiling will refuse to run without a manifest.** The
profile job reads the manifest hash from a config or argument; if the on-disk
corpus's manifest-of-record disagrees, the profile aborts. The completed STC
discovery profile is a read-only companion artifact; it does not yet create
this enforcement contract.

**Ingestion refuses to run without a manifest reconciliation.** Before
writing a single row to dev DB, ingestion verifies:

1. The manifest file exists at the expected path.
2. The manifest's `corpus_path` matches the configured corpus path.
3. Re-deriving the manifest's SHA-256 over each file matches the manifest's
   stored hash. (Spot-check vs full re-derive is decision O6.)
4. No file in the manifest is missing on disk; no file on disk is missing
   from the manifest.

**Query layer refuses unrecognized snapshots.** When a caller asks for
"2024-25 KSD assessment in math," the query layer maps that to a specific
snapshot id and refuses if no ingestion run has loaded that snapshot.

This makes the manifest the contract between layers: profiling, ingestion,
and querying all bind to the same frozen description of what the corpus is.

---

## 8. Schema / profile strategy

Per dataset, the completed STC profile is the first-pass evidence artifact:
counts, field coverage, suppression patterns, year ranges, Kent School
District signal checks, and numeric-string patterns. The future
schema-locked **profile** is a deterministic, versioned JSON document
containing:

- **Top-level shape** — verified to be array (the manifest already confirms
  this for current corpus).
- **Record count** — exact match against manifest.
- **Field set** — the union of keys observed across all records, with
  per-key occurrence count.
- **Type inference per field** — string vs numeric vs boolean, with a
  histogram of representations (e.g., `"82.4"` vs `82.4` vs `null` vs `"*"`).
- **Null / suppression patterns** — the set of values OSPI uses to indicate
  suppressed or missing data: `null`, `""`, `"*"`, `"<10"`, `"Suppressed"`,
  `"N<10"`, etc. Each gets a normalized canonical form.
- **Year ranges** — distinct values of `schoolyear` (or its dataset-specific
  equivalent), validated against an allow-list of formats.
- **District / school identifiers** — distinct codes; cross-referenced
  against a canonical district/school registry (decision O7).
- **Candidate primary keys** — minimum-cardinality field combinations that
  uniquely identify a row. Used to validate the record-identity strategy
  in Section 4.
- **Schema version stamp** — the profile is a versioned artifact; rerunning
  with new code produces a new version.

**Tooling:** the profile job is Python stdlib + Pydantic v2. It reads only
the manifest plus the JSON files in the corpus path. It writes a single
profile JSON per dataset to a directory **outside** the corpus (refused if
inside, same rule as the validator).

**Profiles are committed to git** (per dataset, under
`docs/ai/ospi/profiles/`) once they are produced. They are small and
human-reviewable; the underlying data is not.

---

## 9. SQL vs RAG split

**Hard rule: numeric facts never depend on embeddings.** A claim like "KSD
4-year graduation rate was 91.0% in 2023-24" must come from a deterministic
SQL query against typed columns, never from an LLM summarizing retrieved
chunks of a JSON record.

**Belongs in SQL:**

- Every numeric, categorical, and identifier field from every OSPI dataset.
- Suppression flags as enum columns.
- Year ranges, codes, controlled-vocabulary values.
- Aggregation views (district totals, group breakdowns, year-over-year
  deltas).

**Optionally RAG-able later (not V1):**

- Per-dataset documentation (what each field means, suppression conventions,
  definitional changes across years). Embedding this is potentially useful
  for "what does WaKIDS measure?" questions but is not required to answer
  data questions.
- Cross-dataset disambiguation notes (e.g., "growth percentile is not the
  same as percent meeting standard").
- Mapping notes for agenda-to-OSPI evidence retrieval (later).

**Never RAG-able:**

- Numeric values.
- Suppression decisions.
- Identifier resolution.
- Aggregations.
- Anything that requires arithmetic or set logic.

**Operational rule:** the agenda-analysis worker, when it needs an OSPI
fact, calls the OSPI query/citation layer (deterministic SQL → JSON facts)
and treats the response as a tool result. The LLM may generate prose around
the fact but cannot generate the fact.

---

## 10. Query / citation requirements

An answer that cites an OSPI fact must carry **all** of the following or
it is not trustworthy:

```json
{
  "fact_id": "ospi:assessment:2024-2025:17415:0123:5_meeting_standard_math:5",
  "value": "62.4",
  "value_kind": "percent",
  "value_unit": "percent_students_meeting_standard",
  "suppression_status": "reported",
  "suppression_code": null,
  "dimensions": {
    "dataset": "assessment",
    "reporting_year": "2024-2025",
    "district_code": "17415",
    "district_name": "Kent School District",
    "school_code": "0123",
    "school_name": "Example Elementary",
    "student_group": "All Students",
    "subject": "math",
    "grade_band": "5"
  },
  "provenance": {
    "source_file_path": "research/ospi_data/assessment.json",
    "source_file_sha256": "...",
    "manifest_id": "ospi-manifest-validator/v1:...",
    "manifest_sha256": "...",
    "corpus_snapshot_version": "2026-04-30T00:00Z",
    "schema_version": "ospi.assessment/v1",
    "ingestion_run_id": "uuid",
    "extraction_timestamp": "2026-05-04T12:00:00Z"
  },
  "record_payload_hash": "..."
}
```

**Suppressed cells must surface as suppression, not as missing.** A
suppressed value carries `value: null`, `suppression_status: "suppressed"`,
and the suppression code. The renderer presents "OSPI suppressed
(small group)" rather than silently dropping the row.

**Citations must round-trip.** Given a fact_id, the query layer must be
able to re-derive the exact citation. This forces deterministic IDs.

**Citation contract is versioned.** Bumps go through a PR with a migration
window for callers (the agenda-analysis renderer will be the first caller).

---

## 11. Dev-only ingestion design

**Three-tier storage** in dev Postgres (same `qorvault` database, new
`ospi_facts` schema for namespace isolation):

| Tier | Table family | Purpose |
|---|---|---|
| Raw | `ospi_facts.raw_<dataset>` | One row per input JSON record, with full original JSON in `payload_jsonb`, plus identity columns and provenance. Source of truth for everything below. |
| Normalized | `ospi_facts.<dataset>` | Typed columns per dataset's schema model; one row per record; FK to raw. The query layer hits these. |
| Aggregated | `ospi_facts.v_<dataset>_<measure>` views | Pre-computed aggregations (district totals, year deltas). All views derive from normalized; no parallel ETL. |

**Ingestion flow** (per dataset, per snapshot):

1. Manifest reconciliation gate (Section 7).
2. Stream JSON file with the validator's `JsonStream` (already proven; no
   new parser).
3. For each record: compute `record_payload_hash`, stamp provenance,
   `INSERT … ON CONFLICT DO NOTHING` into raw table keyed by
   `(snapshot_version, dataset, record_index_in_file)`.
4. From raw, project into normalized table via a deterministic transform
   defined by the dataset's Pydantic model. Failures emit a warning row
   into `ospi_facts.ingestion_warnings` and skip; they do not abort the run.
5. After full ingest, run reconciliation queries: row counts per
   normalized table must match manifest record counts (minus skipped/warned
   rows).

**No production writes ever.** The dev DB is separate. Production schema
changes require their own migration design and approval.

**No migrations applied until separately approved.** This memo specifies
the schema design; the actual `CREATE TABLE` DDL ships as a follow-up
issue with its own review.

**Idempotency:** rerunning the same snapshot's ingestion is a no-op for
unchanged rows (`ON CONFLICT DO NOTHING`) and adds new rows for any new
records the manifest covers. Snapshots are immutable post-ingest.

**Resumability:** ingestion checkpoints by `(dataset, file_offset)` in
`ospi_facts.ingestion_checkpoints`; a crash mid-run resumes from the last
flushed offset on restart.

---

## 12. Validation gates

Four gates, each blocking the next layer until passed.

1. **Before profiling** — manifest exists, manifest passes
   `validate_ospi_manifest.py` against the on-disk corpus, and Donald has
   approved a corpus snapshot version.
2. **Before dev ingestion** — every dataset has a committed profile
   (Section 8) reviewed and approved; every dataset has a committed Pydantic
   schema (Section 8); manifest reconciles bit-for-bit.
3. **Before query layer goes live (in dev)** — ingestion completed for at
   least one full snapshot; reconciliation queries pass; citation
   round-trips for a sampled set of records.
4. **Before agenda automation can consume OSPI facts** — query/citation
   contract version is locked; failure modes (suppressed, missing, schema
   mismatch) are documented; the agenda renderer's "OSPI fact" tool has its
   own contract test.

Each gate is a checklist item in the corresponding issue. None is
"automatic" — each is owner-approved.

---

## 13. Test strategy

**In CI (Stage 1.x as it expands):**

- **Synthetic unit tests** — parser tests for type coercion, suppression
  handling, year-format canonicalization, record identity.
- **Fixture-based profile tests** — small JSON fixtures for each dataset
  shape (a few records each); profile job runs against fixtures and
  asserts the produced profile JSON.
- **Schema contract tests** — Pydantic models validate against fixtures;
  invalid records fail closed.
- **Citation formatting tests** — given a synthetic ingestion result, the
  citation builder produces a JSON conforming to the contract; required
  fields present; missing-provenance cases fail closed.
- **Ingestion against test DB** — pytest with a disposable Postgres test
  schema (created per session, dropped after), fixture data only.

**Not in CI by default:**

- Real-corpus profile runs (large, slow, and the corpus isn't in git).
- Real-corpus ingestion runs.
- Performance/scaling tests.

**Optional integration tests** runnable locally with explicit
`OSPI_RUN_REAL_CORPUS=1` env gate. Not required for merge. Tracks corpus
freshness when Donald wants to validate against the real files.

---

## 14. Risks and open decisions

| # | Risk / open question |
|---|---|
| R1 | **Suppression conventions vary across files.** assessment uses one set of codes; SQSS may use another. A naïve normalizer could conflate "suppressed for privacy" with "suppressed because not yet reported." Mitigation: use the completed STC profile's Null, Missing, And Suppression Patterns table, which confirms material variance across assessment, attendance, graduation, growth, SQSS, and WaKIDS; then build a canonical taxonomy with provenance before normalization. |
| R2 | **Year format is inconsistent.** Some datasets store `"2024-25"`, others `"2024-2025"`, others a numeric end-year. Mitigation: canonicalize on ingest and validate during profile; refuse unknown formats. |
| R3 | **District/school code stability across years.** Codes can be retired, merged, renamed (e.g., school closures). A code valid in 2018-19 may not exist in 2024-25. Mitigation: build a registry table that captures code-lifetime windows; reject ingestion of facts whose codes don't validate against the registry for the reporting year. |
| R4 | **Numeric values stored as strings.** The STC profile confirms numeric-like values are string-encoded across the current 10 datasets, with percentages and placeholders mixed in. Coercion errors silently corrupt data. Mitigation: strict Pydantic models with explicit string→number coercion + suppression-code allowlist. Anything else fails ingestion. |
| R5 | **Multi-grain rows.** Some datasets emit district-level, school-level, and grade-level rows in the same file. A naïve aggregate sum would double-count. Mitigation: profile detects grain via key cardinality; normalized table includes a `grain` enum column; aggregation views explicitly filter by grain. |
| R6 | **Volume in dev DB.** ~700k rows × 10 datasets, normalized + raw + warnings ≈ 5-10 million rows + raw JSONB blobs. Manageable on Smeltor but warrants disk planning. Mitigation: estimate disk per dataset during profiling; storage check is a gate before first ingestion. |
| R7 | **Cross-dataset analyses require reconciled keys.** "Compare attendance and discipline at the same school" depends on consistent school-code semantics. Mitigation: registry-based canonical IDs, plus a `dataset_join_compatibility` matrix in docs. |
| R8 | **Corpus comes from `download_ospi.py`, which we have not audited.** The corpus on disk is presumed to match what OSPI publishes, but we haven't verified the download script doesn't transform values en route. Mitigation: audit `download_ospi.py` as a Stage-0 task before any ingestion. |
| R9 | **OSPI may publish corrections.** A "corrected" file with the same name and different bytes is detected by manifest hash drift. Mitigation: `corpus_snapshot_version` carries the manifest hash; downstream ingestion treats new manifest as a new snapshot. |
| R10 | **Trust boundary.** Downstream callers (agenda renderer) might over-claim by treating OSPI facts as ground truth. Mitigation: every citation includes "Source: OSPI Report Card, [year]" framing; the renderer must surface OSPI as a source, not as omniscient truth. |

---

## 15. Staged issue backlog

Sequencing maximizes safe verifiability and aligns with the gates in
Section 12.

| Backlog item | Description | Gate it serves |
|---|---|---|
| **OSPI corpus profile — schema-lock (Stage 0 design)** | Convert STC's discovery profile into a Pydantic-versioned, committed per-dataset profile artifact with inputs, outputs, fixtures, tests, and gate criteria. | Gate 1 (before schema-lock) |
| **OSPI authority and provenance model — OspiCitation contract** | Lock the citation field schema (Section 5 + Section 10). Pydantic model for `OspiCitation`. Tests for required-fields-present. | Gate 3 (before query layer) |
| **OSPI per-dataset schema and record-identity design** | Per-dataset Pydantic schemas + record-identity composite. The schema-locked profile feeds this. Tests against fixtures. | Gate 2 (before dev ingestion) |
| **OSPI dev-only ingestion — table design and migration plan (docs only)** | Schema for `ospi_facts.raw_*` and `ospi_facts.<dataset>` tables, plus migration plan, plus checkpoint design. Docs only. | Gate 2 (before dev ingestion) |
| **OSPI dev-only ingestion DDL + ingest job — first dataset (enrollment)** | The actual `CREATE TABLE` migration applied to dev DB only, plus the streaming ingest job for one dataset. | Gate 2 (before dev ingestion) |
| **OSPI query / citation layer — deterministic SQL → JSON facts** | Deterministic SQL → JSON facts API. Versioned contract. Tests for citation round-trip. | Gate 3 (before query layer) |
| **OSPI SQL facts library — parameterized common analyses** | Parameterized queries for the most common analyses (district totals, year deltas, group breakdowns). Lives as views or as Python helpers; same contract. | Gate 3 (before query layer) |
| **OSPI-to-agenda evidence mapping** | Spec for how the agenda-analysis worker fetches OSPI facts as evidence for an item. Depends on the agenda-analysis worker shipping first. | Gate 4 (before agenda automation consumes OSPI) |
| **Stage 1.1 CI hardening for OSPI-related code** | Expand Ruff scope to OSPI ingestion code; add the OSPI-specific synthetic test scopes to the CI workflow. Pin Ruff/pytest versions. | Cross-cutting |

**Per-dataset rollout for the first ingestion job:** the first ingestion
target is the cleanest dataset (enrollment in my preliminary view) so we
exercise the full path without fighting suppression complexity. After that
lands and reconciles, add datasets in increasing complexity.

---

## 16. Relationship to Issue #27

Issue #27 ("Clarify route semantics: hybrid is retrieval-only after PR
#26") is small and **unrelated** to the OSPI workstream's data path.

- #27 affects how the `rag_api` answers ad-hoc queries about agenda
  documents from Qdrant.
- The OSPI work doesn't go through `rag_api`'s hybrid route at all — it
  hits dedicated SQL queries against `ospi_facts.*` tables that don't yet
  exist.
- The agenda-analysis worker (a future consumer of both layers) will use
  the **retrieval-only** path for evidence retrieval in its analysis loop.
  That's exactly what #27 codifies.

**Recommendation: run #27 in parallel.** It's a small clarification fix
that helps documentation, helps the agenda-analysis design, and helps any
caller building on the retrieval layer. The OSPI workstream is much
larger; #27 should not be allowed to block it, and OSPI should not block
#27. Do them concurrently.

---

## 17. Plain-English owner decisions

These are the explicit choices that need Donald's input. None block this
memo. They block the issues they correspond to.

### O1 — Memo location

This memo is at `docs/ai/architecture/`. Should I move it to
`docs/ai/ospi/` to group all OSPI material together?
**Recommendation: leave at `architecture/`.** Workstream-level architecture
memos go in `architecture/` (see also the agenda-analysis memo from PR
#32); operational OSPI docs (validator plan/review) live in
`docs/ai/ospi/`.

### O2 — Schema namespace

OSPI tables go in a new `ospi_facts` Postgres schema in the same
`qorvault` database (parallel to `agenda_runs`).
**Recommendation: yes.** Keeps namespace isolation; same DB simplifies
backups and credentials.

### O3 — Year-format canonical form

Canonical year format for storage and query.
**Recommendation: full form `YYYY-YYYY` (e.g., `"2024-2025"`).** More
human-readable; trivially convertible to short form for display.

### O4 — Student-group taxonomy

Should QorVault adopt OSPI's student-group taxonomy as-is, or canonicalize
to a project-defined controlled vocabulary?
**Recommendation: adopt OSPI's taxonomy as-is**, store the original
string, and add a `student_group_canonical` column populated from a
mapping table that lives in version control. Aliases don't get inferred by
LLM.

### O5 — Snapshot lifecycle

Keep all historical snapshots forever, or evict after N years?
**Recommendation: keep all, no eviction in V1.** Disk on Smeltor is
adequate; historical answers must remain re-derivable.

### O6 — Manifest reconciliation depth

Spot-check (N random files) vs. full re-derive (all files) on every
ingestion run.
**Recommendation: full re-derive on every ingestion run.** Cost is
bounded (stream-hash 586 MB; ~5–10 seconds); avoids any chance of silent
drift.

### O7 — District / school registry source

Build the canonical registry from (a) profiling distinct codes across the
corpus, (b) downloading OSPI's published district/school list, or (c) a
manual hand-maintained file.
**Recommendation: (a) primary, (b) optional cross-check.** Profile-derived
gives us only codes present in the data; OSPI's published list adds codes
that exist but appear with no facts. Avoid (c) — manual maintenance ages
poorly.

### O8 — First-ingestion target dataset

Which dataset is the cleanest first target for end-to-end ingestion?
**Recommendation: enrollment**, with attendance second. Enrollment has
fewer suppression complications and is the denominator for many later
metrics.

### O9 — `download_ospi.py` audit

Should we audit the existing download script before any ingestion, or
trust the on-disk corpus as-is?
**Recommendation: audit before ingestion.** A read-only review of the
script is small and surfaces any pre-ingest transformations we'd
otherwise be unable to detect after the fact.

### O10 — Trust framing

How do we present OSPI facts to readers downstream of the agenda renderer?
**Recommendation: always present as "OSPI Report Card, [year]"** with a
link or attribution. Never imply that a number originated with QorVault
analysis.

---

## Top 5 risks (ranked)

1. **R3 — district/school code stability across years.** The single
   biggest source of silent join errors if not handled by a code-lifetime
   registry. Mitigation: registry table is a Stage-0 deliverable in the
   schema and record-identity design work.
2. **R1 — suppression-convention divergence.** Conflating
   "privacy-suppressed" with "not yet reported" produces wrong
   percentages. Mitigation: use STC's suppression-pattern evidence before
   any normalization, then lock the taxonomy in the schema design work.
3. **R5 — multi-grain rows.** Aggregations that don't filter by grain
   double-count. Mitigation: explicit grain enum column; aggregation
   views filter by grain in the schema and ingestion design work.
4. **R8 — `download_ospi.py` is unaudited.** Anything wrong upstream of
   our corpus is invisible to manifest validation. Mitigation: audit
   before any ingestion (decision O9).
5. **R10 — trust boundary.** OSPI facts get treated as omniscient by
   downstream consumers. Mitigation: every citation carries OSPI
   attribution; renderer is required to surface it (decision O10).

## Top 5 next implementation steps

1. **Open: OSPI corpus profile — schema-lock (Stage 0 design).** Convert
   STC's discovery profile into a Pydantic-versioned per-dataset profile
   artifact with fixture format, profile-output schema, and gate criteria.
2. **Open: OSPI authority and provenance model — OspiCitation contract.** Lock the
   `OspiCitation` schema (Section 5/10). Pydantic + JSON schema.
3. **Open: OSPI per-dataset schema and record-identity design.** Per-dataset
   Pydantic schemas + the composite identity. Depends on the schema-locked
   profile outputs.
4. **Open: OSPI dev-only ingestion — table design and migration plan (docs only).**
   Specifies the three-tier table layout and the migration plan.
5. **Audit `download_ospi.py`** as a small standalone task before any
   ingestion lands (decision O9). Read-only review; no execution.

## Proposed staged issue backlog

See Section 15 for the full table. Sequence: schema-lock profile, citation
contract, schema/identity design, dev-only ingestion design, first-dataset
ingestion, query/citation layer, SQL facts library, agenda evidence mapping,
then CI hardening.
Issue #27 runs in parallel with this stack at any point.

## Plain-English owner decisions for Donald

See Section 17 for the full set (O1–O10). The most consequential are O3
(year format), O4 (student-group taxonomy), O7 (registry source), and O9
(audit `download_ospi.py` first). The rest are routine.

## Assumptions requiring owner approval

- **A1.** The validator manifest is the single source of truth for "what
  is in the corpus" and gates everything downstream. Confirm.
- **A2.** OSPI tables live in the same `qorvault` Postgres database as
  existing schema, under a new `ospi_facts` schema. (Decision O2.)
- **A3.** Citations always include "OSPI Report Card, [year]" framing.
  (Decision O10.)
- **A4.** Dev DB is separate from any production DB (today, only dev
  exists; production DB is theoretical). Any future production schema is
  out of scope for this roadmap and requires its own design.
- **A5.** Stage 1.1 CI expansion is best done after the first
  OSPI ingestion lands so the CI scope reflects real code, not hypothetical
  modules.
- **A6.** The agenda-analysis worker (per the prior architecture memo) is
  the first consumer of the OSPI query/citation layer. The OSPI roadmap
  must therefore lock its citation contract before OSPI-to-agenda mapping
  can ship.

---

## Safety attestation

- **Docs-only artifact:** this memo records architecture decisions and
  does not add runtime code, schema migrations, or ingestion paths.
- **No real OSPI corpus access in this memo-authoring task:** no
  `download_ospi.py` invocation, no validation against the real corpus, no
  reading of any `research/ospi_data/*.json` or `docs/ospi_data/*.json`
  file. Corpus totals quoted in this memo come from the companion STC
  profile report, not direct corpus access by this memo.
- **Companion profile boundary:** the separate STC profiling pass was
  read-only and is included here as a companion evidence artifact. It did not
  run `download_ospi.py`, did not write to any database, and did not modify
  corpus files.
- **No production access. No Framework. No database — no DDL applied,
  no migrations, no queries.**
- **No ingestion, indexing, embedding, deployment, promotion, or sync
  triggered.**
- **No live BoardDocs / Kent School District access — no scraping, no
  downloads.**
- **No staging, commit, push, branch creation, tag, merge, or PR
  creation.** This memo file is uncommitted; Donald must explicitly
  authorize a commit on a feature branch.
- **No implementation authorization** — every issue listed in Section 15
  requires its own approval, scope review, and PR.
