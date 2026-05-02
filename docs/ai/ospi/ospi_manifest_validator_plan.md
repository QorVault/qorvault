# OSPI Manifest Validator Plan

## Implemented

- Added `scripts/validate_ospi_manifest.py`, a standalone stdlib CLI for deterministic, read-only OSPI corpus manifest generation.
- Requires an explicit `--corpus-path` and explicit `--output`.
- Refuses missing corpus paths, non-directory corpus paths, output paths inside the input corpus directory, and output paths whose parent directory does not already exist.
- Discovers `*.json` files deterministically by repo-relative path and ignores `download_ospi.py`.
- Streams SHA-256 hashing in 1 MiB blocks.
- Streams top-level JSON arrays one record at a time for counting, schema summaries, and deterministic dry-run record fingerprints.
- Supports small/top-level object manifests and streams top-level object array values, such as `$.records`, without importing ingestion code.

## CLI Usage

```bash
python -B scripts/validate_ospi_manifest.py \
  --corpus-path /mnt/qorvault-dev/repo/research/ospi_data \
  --output docs/ai/ospi/ospi_manifest_dry_run_YYYYMMDDTHHMMSSZ.json
```

## Manifest Schema

Top-level fields:

- `schema_version`
- `generated_at`
- `corpus_path`
- `file_count`
- `total_size_bytes`
- `total_record_count`
- `files`
- `warnings`
- `constraints_observed`

Per-file fields include:

- `name`, `relative_path`, `size_bytes`, `sha256`, `modified_time_utc`
- `corpus_file_id` and `corpus_file_id_strategy`
- `top_level_shape`, `record_container_path`, `record_count`
- `major_keys`, `representative_nested_keys`, and likely field groups
- `record_id_strategy` and bounded `record_id_preview`
- `schema_summary` and file-local `warnings`

No bulk raw records are included in the manifest. Record previews contain deterministic fingerprints only.

## Safety Guarantees

- No imports from `scripts/ingest_ospi_data.py` or `scripts/ingest_new_ospi_data.py`.
- No network, database, ingestion, embedding, indexing, migration, sync, deploy, or promotion calls.
- No credentials required.
- No writes to the corpus directory; output is refused if it resolves inside the corpus path.
- No implicit directory creation; the output parent directory must already exist so the manifest file is the validator's only write.
- Safe to rerun. File and record IDs are derived from stable file paths, bytes, indexes, and canonical record JSON, not timestamps.

## Tests Added

`tests/test_validate_ospi_manifest.py` covers:

- Missing corpus path.
- Non-directory corpus path.
- Output path inside corpus rejection.
- Missing output parent directory rejection.
- Deterministic JSON file ordering.
- Top-level array schema/count/ID handling.
- Top-level object schema/count/ID handling.
- Malformed JSON warning behavior.
- Empty JSON warning behavior.
- Checksum stability.
- Deterministic IDs.
- Output manifest shape.
- No mutation of input files.
- `download_ospi.py` ignored.
- CLI writes only to an explicit output path.

## Real-Corpus Smoke Result

Command:

```bash
python -B scripts/validate_ospi_manifest.py \
  --corpus-path /mnt/qorvault-dev/repo/research/ospi_data \
  --output docs/ai/ospi/ospi_manifest_dry_run_20260430T185258Z.json
```

Result:

- `file_count`: 10
- `total_size_bytes`: 586,204,574
- `total_record_count`: 698,222
- `warnings`: 0
- `download_ospi.py`: ignored and absent from `files`

## Known Limitations

- The validator does not define a chunking strategy, so it intentionally emits no chunk IDs.
- Schema inference is bounded to representative records and fields; it is an inventory/safety manifest, not a normalized OSPI fact model.
- The stdlib streaming parser is optimized for top-level arrays and top-level object arrays. A future deeply nested, very large non-array object value may need a dedicated token parser before full schema extraction.

## Recommended Next Task

Design the normalized OSPI fact/query contract: choose canonical entity/year/metric dimensions, define source authority rules, then add deterministic read-only query tests before any ingestion or indexing work.
