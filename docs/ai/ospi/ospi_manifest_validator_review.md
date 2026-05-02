# OSPI Manifest Validator Review

Timestamp: `20260430T190612Z`

## Validator Safety Review

- Reviewed `scripts/validate_ospi_manifest.py` as a standalone stdlib validator.
- No network libraries, database clients, ingestion imports, subprocess calls, migration calls, embedding calls, indexing calls, sync/deploy/promote calls, or download calls are present.
- Input corpus files are opened only for read-only text parsing and binary hashing.
- The validator writes only the explicit manifest output file after hardening; it no longer creates missing parent directories.
- Output paths resolving inside the corpus directory are rejected.
- JSON discovery is deterministic: `corpus.rglob("*.json")` sorted by corpus-relative POSIX path.
- File IDs are based on relative path, size, and file SHA-256, not timestamps.
- Record fingerprints are based on file ID, record index, and canonical record JSON, not timestamps.
- Samples are bounded: 5 record fingerprints, 250 schema sample records, 200 representative keys, and 100 likely fields per group.
- Manifest entries include schema summaries and fingerprints only; no bulk raw records are emitted.
- No input mutation path was found.

## Test Coverage Review

- Existing tests covered missing/non-directory corpus paths, rejecting output inside corpus, deterministic ordering, `download_ospi.py` exclusion, array/object schema summaries, malformed/empty JSON warnings, stable checksums, stable IDs, no input mutation, and CLI output.
- Added a hardening test for missing output parent directories so the validator cannot create extra filesystem state.
- Test suite now has 10 focused tests and passes.

## Smoke Manifest Review

- Inspected `docs/ai/ospi/ospi_manifest_dry_run_20260430T185258Z.json`.
- Manifest shape is useful for ingestion design: it gives corpus totals, per-file byte size, SHA-256, record count, top-level shape, record container path, likely entity/year/metric fields, schema key summaries, bounded record fingerprint previews, warnings, and observed safety constraints.
- Real-corpus summary: 10 files, 586,204,574 bytes, 698,222 records, 0 warnings.
- All 10 manifest files are top-level arrays at `$`: assessment, attendance, discipline, enrollment, graduation, growth, sqss, teacher demographics, teacher experience, and WaKIDS.
- `download_ospi.py` is absent from the manifest.
- Record fingerprints are deterministic enough for later dry-run ingestion design because they derive from stable file identity, record index, and canonical record JSON.

## Gaps/Risks

- `generated_at` and per-file `modified_time_utc` are informational timestamps, so full manifest files are not byte-for-byte identical across runs even though file and record IDs are deterministic.
- Schema inference is intentionally bounded and heuristic; it is not a normalized OSPI fact model.
- No chunk IDs are emitted because no chunking strategy has been defined.
- Numeric/entity/year field detection is useful for planning but should not become ingestion authority without explicit source-schema mapping tests.
- Object-array handling chooses the first dict-record array in a top-level object; current corpus files are arrays, so this is acceptable for now.

## Recommended Hardening

- Applied: require the output parent directory to already exist and remove implicit directory creation.
- Recommended later: add a repeat-smoke stability check that compares file IDs and record fingerprint previews while ignoring `generated_at`.
- Recommended later: factor the fingerprint strategy into a shared helper if dry-run ingestion begins consuming the IDs directly.

## Recommended Branch/Commit Plan

- Move this work to a new branch instead of leaving it on `fix/hybrid-route-retrieval-only`; the current branch name describes the earlier hybrid-route fix, not this OSPI validator.
- Recommended branch name: `feat/ospi-manifest-validator`.
- Do not commit repo-local OSPI JSON files under `research/ospi_data/`.
- Do not commit unrelated untracked files or the untracked `docs/ospi_data/` corpus copy.
- Use explicit paths only; do not use `git add .`.
- Recommended commit message: `feat: add read-only OSPI manifest validator`.

## Recommended Next Task

Design a normalized, read-only OSPI fact/query contract with deterministic tests for canonical entity, year, metric, source authority, and suppressed/missing value handling before any ingestion, embedding, indexing, or router integration work.
