#!/usr/bin/env python3
"""Build a deterministic read-only manifest for repo-local OSPI JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "ospi-manifest-validator/v1"
READ_CHARS = 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
PREVIEW_RECORD_LIMIT = 5
SCHEMA_SAMPLE_LIMIT = 250
MAX_REPRESENTATIVE_KEYS = 200
MAX_LIKELY_FIELDS = 100


class ManifestValidationError(Exception):
    """Raised when manifest validation cannot proceed safely."""


class JsonStream:
    """Small streaming JSON reader for top-level arrays and object members."""

    def __init__(self, path: Path) -> None:
        """Open path for incremental JSON decoding."""
        self.path = path
        self.handle = path.open("r", encoding="utf-8")
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0
        self.eof = False

    def close(self) -> None:
        """Close the underlying file handle."""
        self.handle.close()

    def _read_more(self) -> bool:
        """Append another chunk to the decode buffer when input remains."""
        if self.eof:
            return False
        chunk = self.handle.read(READ_CHARS)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _compact(self) -> None:
        """Drop already-consumed buffer content to keep memory bounded."""
        if self.position > READ_CHARS:
            self.buffer = self.buffer[self.position :]
            self.position = 0

    def _ensure_available(self) -> bool:
        """Ensure at least one unread character is buffered, if possible."""
        while self.position >= len(self.buffer):
            if not self._read_more():
                return False
        return True

    def skip_whitespace(self) -> None:
        """Advance the stream position past JSON whitespace."""
        while True:
            if not self._ensure_available():
                return
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                self._compact()
                return

    def peek_non_whitespace(self) -> str | None:
        """Return the next non-whitespace character without consuming it."""
        self.skip_whitespace()
        if not self._ensure_available():
            return None
        return self.buffer[self.position]

    def consume(self, expected: str) -> None:
        """Consume an expected delimiter after optional whitespace."""
        actual = self.peek_non_whitespace()
        if actual != expected:
            raise json.JSONDecodeError(f"Expected {expected!r}", self.buffer, self.position)
        self.position += 1
        self._compact()

    def decode_value(self) -> Any:
        """Decode one complete JSON value from the current stream position."""
        self.skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self._read_more():
                    continue
                raise
            self.position = end
            self._compact()
            return value

    def require_no_trailing_non_whitespace(self) -> None:
        """Raise if non-whitespace content remains after a top-level value."""
        self.skip_whitespace()
        if self._ensure_available():
            raise json.JSONDecodeError(
                "Trailing non-whitespace content after top-level JSON value",
                self.buffer,
                self.position,
            )


def build_manifest(corpus_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate an OSPI corpus directory and write a JSON manifest to output_path."""
    corpus = _validate_corpus_path(Path(corpus_path).expanduser())
    output = _validate_output_path(Path(output_path).expanduser(), corpus)
    json_files = _discover_json_files(corpus)

    files: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    total_size_bytes = 0
    total_record_count = 0

    for path in json_files:
        file_entry, file_warnings = _inspect_json_file(corpus, path)
        files.append(file_entry)
        warnings.extend(file_warnings)
        total_size_bytes += file_entry["size_bytes"]
        total_record_count += file_entry["record_count"]

    if not json_files:
        warnings.append({"level": "warning", "path": str(corpus), "message": "No JSON files discovered."})

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "corpus_path": str(corpus),
        "file_count": len(files),
        "total_size_bytes": total_size_bytes,
        "total_record_count": total_record_count,
        "files": files,
        "warnings": warnings,
        "constraints_observed": {
            "read_only_validator": True,
            "explicit_corpus_path_required": True,
            "explicit_output_path_required": True,
            "output_path_must_be_outside_corpus": True,
            "download_ospi_py_ignored": True,
            "corpus_files_written": False,
            "network_calls": False,
            "database_connections": False,
            "ingestion_scripts_imported": False,
            "embedding_jobs_started": False,
            "indexing_jobs_started": False,
            "migrations_run": False,
            "downloads_started": False,
        },
    }

    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _validate_corpus_path(path: Path) -> Path:
    """Resolve and validate the user-provided corpus directory path."""
    if not path.exists():
        raise ManifestValidationError(f"Corpus path does not exist: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ManifestValidationError(f"Corpus path is not a directory: {resolved}")
    return resolved


def _validate_output_path(path: Path, corpus: Path) -> Path:
    """Resolve and validate that manifest output stays outside the corpus tree."""
    resolved = path.resolve(strict=False)
    if resolved == corpus or _is_relative_to(resolved, corpus):
        raise ManifestValidationError(f"Manifest output path must be outside the corpus directory: {resolved}")
    if not resolved.parent.exists():
        raise ManifestValidationError(f"Manifest output parent directory does not exist: {resolved.parent}")
    if not resolved.parent.is_dir():
        raise ManifestValidationError(f"Manifest output parent path is not a directory: {resolved.parent}")
    return resolved


def _is_relative_to(child: Path, parent: Path) -> bool:
    """Return whether child is nested within parent."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _discover_json_files(corpus: Path) -> list[Path]:
    """Return corpus JSON files in deterministic relative-path order."""
    return sorted(
        (path for path in corpus.rglob("*.json") if path.is_file() and path.name != "download_ospi.py"),
        key=lambda path: path.relative_to(corpus).as_posix(),
    )


def _inspect_json_file(corpus: Path, path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Inspect one JSON file and summarize its schema, IDs, and warnings."""
    stat = path.stat()
    relative_path = path.relative_to(corpus).as_posix()
    sha256 = _sha256_file(path)
    file_id = _short_id("ospi_file", f"{relative_path}\0{stat.st_size}\0{sha256}")
    warnings: list[dict[str, str]] = []
    entry = _base_file_entry(path, relative_path, stat, sha256, file_id)

    if stat.st_size == 0:
        _mark_empty(entry, warnings, relative_path)
        _finalize_schema_fields(entry)
        return entry, warnings

    reader = JsonStream(path)
    try:
        first_char = reader.peek_non_whitespace()
        if first_char is None:
            _mark_empty(entry, warnings, relative_path)
        elif first_char == "[":
            _inspect_top_level_array(reader, entry, file_id)
            reader.require_no_trailing_non_whitespace()
        elif first_char == "{":
            _inspect_top_level_object(reader, entry, file_id)
            reader.require_no_trailing_non_whitespace()
        else:
            value = reader.decode_value()
            entry["top_level_shape"] = type(value).__name__
            entry["record_count"] = 1
            warnings.append(
                {
                    "level": "warning",
                    "path": relative_path,
                    "message": f"Unexpected top-level JSON shape: {entry['top_level_shape']}.",
                }
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        entry["top_level_shape"] = "malformed"
        entry["record_count"] = 0
        entry["warnings"].append("Malformed JSON; schema and record counts were not inferred.")
        warnings.append({"level": "warning", "path": relative_path, "message": f"Malformed JSON: {exc}"})
    finally:
        reader.close()

    _finalize_schema_fields(entry)
    return entry, warnings


def _base_file_entry(path: Path, relative_path: str, stat_result: Any, sha256: str, file_id: str) -> dict[str, Any]:
    """Build the manifest entry scaffold for a single corpus file."""
    return {
        "name": path.name,
        "relative_path": relative_path,
        "size_bytes": stat_result.st_size,
        "sha256": sha256,
        "modified_time_utc": _format_timestamp(stat_result.st_mtime),
        "corpus_file_id": file_id,
        "corpus_file_id_strategy": "sha256(relative_path + size_bytes + file_sha256)",
        "top_level_shape": "unknown",
        "record_container_path": None,
        "record_count": 0,
        "major_keys": [],
        "representative_nested_keys": [],
        "likely_id_fields": [],
        "likely_school_entity_fields": [],
        "likely_district_fields": [],
        "likely_year_fields": [],
        "likely_numeric_metric_fields": [],
        "record_id_strategy": "sha256(corpus_file_id + record_index + canonical_record_json)",
        "record_id_preview": [],
        "schema_summary": {
            "sampled_record_count": 0,
            "major_key_count": 0,
            "representative_nested_key_count": 0,
        },
        "warnings": [],
        "_major_keys": set(),
        "_nested_keys": set(),
        "_likely_id_fields": set(),
        "_likely_school_entity_fields": set(),
        "_likely_district_fields": set(),
        "_likely_year_fields": set(),
        "_likely_numeric_metric_fields": set(),
    }


def _mark_empty(entry: dict[str, Any], warnings: list[dict[str, str]], relative_path: str) -> None:
    """Record an empty-file warning on both the file entry and manifest summary."""
    entry["top_level_shape"] = "empty"
    entry["record_count"] = 0
    entry["warnings"].append("Empty JSON file.")
    warnings.append({"level": "warning", "path": relative_path, "message": "Empty JSON file."})


def _inspect_top_level_array(reader: JsonStream, entry: dict[str, Any], file_id: str) -> None:
    """Scan a top-level array file and merge sampled record statistics."""
    entry["top_level_shape"] = "array"
    entry["record_container_path"] = "$"
    stats = _new_record_stats()
    count = _scan_array(reader, stats, file_id, "$", "")
    entry["record_count"] = count
    entry["record_id_preview"] = stats["record_id_preview"]
    entry["schema_summary"]["sampled_record_count"] = stats["sampled_record_count"]
    _merge_stats(entry, stats)


def _inspect_top_level_object(reader: JsonStream, entry: dict[str, Any], file_id: str) -> None:
    """Inspect a top-level object and prefer the first array-of-records member."""
    entry["top_level_shape"] = "object"
    reader.consume("{")
    selected_stats: dict[str, Any] | None = None
    selected_count = 0
    selected_path: str | None = None

    next_char = reader.peek_non_whitespace()
    if next_char == "}":
        reader.consume("}")
        entry["record_count"] = 1
        entry["record_container_path"] = "$"
        return

    while True:
        key = reader.decode_value()
        if not isinstance(key, str):
            raise json.JSONDecodeError("Expected object key string", reader.buffer, reader.position)
        entry["_major_keys"].add(key)
        reader.consume(":")

        value_start = reader.peek_non_whitespace()
        if value_start == "[":
            stats = _new_record_stats()
            count = _scan_array(reader, stats, file_id, f"$.{key}", f"{key}[]")
            _add_nested_key(entry, f"{key}[]")
            if selected_stats is None and stats["saw_dict_record"]:
                selected_stats = stats
                selected_count = count
                selected_path = f"$.{key}"
        else:
            value = reader.decode_value()
            _record_object_value_schema(entry, key, value)

        delimiter = reader.peek_non_whitespace()
        if delimiter == ",":
            reader.consume(",")
            continue
        if delimiter == "}":
            reader.consume("}")
            break
        raise json.JSONDecodeError("Expected ',' or '}'", reader.buffer, reader.position)

    if selected_stats is not None:
        entry["record_count"] = selected_count
        entry["record_container_path"] = selected_path
        entry["record_id_preview"] = selected_stats["record_id_preview"]
        entry["schema_summary"]["sampled_record_count"] = selected_stats["sampled_record_count"]
        _merge_stats(entry, selected_stats)
    else:
        entry["record_count"] = 1
        entry["record_container_path"] = "$"
        entry["record_id_preview"] = [
            {
                "record_index": 0,
                "record_fingerprint": _short_id("ospi_rec", f"{file_id}\0object"),
            }
        ]


def _scan_array(
    reader: JsonStream,
    stats: dict[str, Any],
    file_id: str,
    container_path: str,
    field_prefix: str,
) -> int:
    """Consume an array and summarize each record without mutating input files."""
    del container_path
    reader.consume("[")
    count = 0

    next_char = reader.peek_non_whitespace()
    if next_char == "]":
        reader.consume("]")
        return count

    while True:
        record = reader.decode_value()
        if isinstance(record, dict):
            stats["saw_dict_record"] = True
        _record_schema(stats, record, count, file_id, field_prefix)
        count += 1

        delimiter = reader.peek_non_whitespace()
        if delimiter == ",":
            reader.consume(",")
            continue
        if delimiter == "]":
            reader.consume("]")
            return count
        raise json.JSONDecodeError("Expected ',' or ']'", reader.buffer, reader.position)


def _new_record_stats() -> dict[str, Any]:
    """Create the mutable accumulator used while sampling record schemas."""
    return {
        "sampled_record_count": 0,
        "saw_dict_record": False,
        "major_keys": set(),
        "nested_keys": set(),
        "likely_id_fields": set(),
        "likely_school_entity_fields": set(),
        "likely_district_fields": set(),
        "likely_year_fields": set(),
        "likely_numeric_metric_fields": set(),
        "record_id_preview": [],
    }


def _record_schema(
    stats: dict[str, Any],
    record: Any,
    record_index: int,
    file_id: str,
    field_prefix: str,
) -> None:
    """Collect schema hints and deterministic preview IDs for one record."""
    if stats["sampled_record_count"] < SCHEMA_SAMPLE_LIMIT:
        stats["sampled_record_count"] += 1
        if isinstance(record, dict):
            for key, value in record.items():
                stats["major_keys"].add(key)
                field_path = f"{field_prefix}.{key}" if field_prefix else key
                _add_schema_field(stats, field_path, key, value)
                _flatten_nested(value, field_path, stats["nested_keys"])

    if len(stats["record_id_preview"]) < PREVIEW_RECORD_LIMIT:
        stats["record_id_preview"].append(
            {
                "record_index": record_index,
                "record_fingerprint": _record_fingerprint(file_id, record_index, record),
            }
        )


def _record_object_value_schema(entry: dict[str, Any], key: str, value: Any) -> None:
    """Fold top-level object member structure into the manifest schema summary."""
    _add_nested_key(entry, key)
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            field_path = f"{key}.{child_key}"
            _add_nested_key(entry, field_path)
            _classify_likely_field(entry, field_path, child_key, child_value)
            _flatten_nested(child_value, field_path, entry["_nested_keys"])
    else:
        _classify_likely_field(entry, key, key, value)


def _add_schema_field(stats: dict[str, Any], field_path: str, field_name: str, value: Any) -> None:
    """Track a field path and classify it into likely semantic buckets."""
    if len(stats["nested_keys"]) < MAX_REPRESENTATIVE_KEYS:
        stats["nested_keys"].add(field_path)
    _classify_likely_field(stats, field_path, field_name, value)


def _add_nested_key(entry: dict[str, Any], field_path: str) -> None:
    """Store one representative nested key path up to the configured limit."""
    if len(entry["_nested_keys"]) < MAX_REPRESENTATIVE_KEYS:
        entry["_nested_keys"].add(field_path)


def _flatten_nested(value: Any, prefix: str, nested_keys: set[str]) -> None:
    """Recursively record representative nested object and array paths."""
    if len(nested_keys) >= MAX_REPRESENTATIVE_KEYS:
        return
    if isinstance(value, dict):
        for key, child_value in value.items():
            child_path = f"{prefix}.{key}"
            nested_keys.add(child_path)
            _flatten_nested(child_value, child_path, nested_keys)
            if len(nested_keys) >= MAX_REPRESENTATIVE_KEYS:
                return
    elif isinstance(value, list):
        array_path = f"{prefix}[]"
        nested_keys.add(array_path)
        for item in value[:3]:
            _flatten_nested(item, array_path, nested_keys)
            if len(nested_keys) >= MAX_REPRESENTATIVE_KEYS:
                return


def _classify_likely_field(stats: dict[str, Any], field_path: str, field_name: str, value: Any) -> None:
    """Add a field path to any matching semantic hint collections."""
    lowered = _normalize_field_name(field_name)
    path_name = field_path
    if _looks_like_year_field(lowered):
        _bounded_add(_field_set(stats, "likely_year_fields"), path_name)
    if _looks_like_school_entity_field(lowered):
        _bounded_add(_field_set(stats, "likely_school_entity_fields"), path_name)
    if _looks_like_district_field(lowered):
        _bounded_add(_field_set(stats, "likely_district_fields"), path_name)
    if _looks_like_id_field(lowered):
        _bounded_add(_field_set(stats, "likely_id_fields"), path_name)
    if _looks_like_numeric_metric_field(lowered, value):
        _bounded_add(_field_set(stats, "likely_numeric_metric_fields"), path_name)


def _field_set(stats: dict[str, Any], public_name: str) -> set[str]:
    """Return a schema hint accumulator set by public or private backing name."""
    candidate = stats.get(public_name)
    if isinstance(candidate, set):
        return candidate
    private_name = f"_{public_name}"
    private_candidate = stats.get(private_name)
    if isinstance(private_candidate, set):
        return private_candidate
    raise TypeError(f"Schema accumulator is missing a set for {public_name}.")


def _bounded_add(values: set[str], value: str) -> None:
    """Add a value to a hint set while respecting the configured cap."""
    if len(values) < MAX_LIKELY_FIELDS:
        values.add(value)


def _looks_like_year_field(lowered: str) -> bool:
    """Return whether a normalized field name likely denotes a year value."""
    return "year" in lowered or "schoolyr" in lowered or lowered in {"yr", "fiscalyr"}


def _looks_like_school_entity_field(lowered: str) -> bool:
    """Return whether a normalized field name likely names a school-like entity."""
    if _looks_like_year_field(lowered):
        return False
    return any(token in lowered for token in ("school", "organization", "orgname", "entity", "building"))


def _looks_like_district_field(lowered: str) -> bool:
    """Return whether a normalized field name likely refers to a district."""
    return "district" in lowered or lowered.startswith("lea") or "localeducationagency" in lowered


def _looks_like_id_field(lowered: str) -> bool:
    """Return whether a normalized field name likely carries an identifier."""
    id_tokens = ("identifier", "organizationid", "schoolid", "districtid", "recordid")
    return (
        lowered == "id"
        or lowered.endswith("id")
        or lowered.endswith("code")
        or any(token in lowered for token in id_tokens)
    )


def _looks_like_numeric_metric_field(lowered: str, value: Any) -> bool:
    """Return whether a field looks like a numeric metric instead of an ID-like code."""
    metric_tokens = (
        "rate",
        "percent",
        "pct",
        "score",
        "count",
        "number",
        "total",
        "value",
        "metric",
        "index",
        "ratio",
        "amount",
        "average",
        "avg",
        "median",
        "growth",
        "attendance",
        "enrollment",
        "graduation",
        "discipline",
        "absence",
    )
    if not _looks_numeric(value):
        return False
    if any(token in lowered for token in ("year", "code", "id", "zip", "phone")) and not any(
        token in lowered for token in metric_tokens
    ):
        return False
    return any(token in lowered for token in metric_tokens) or isinstance(value, int | float)


def _looks_numeric(value: Any) -> bool:
    """Return whether a value can be treated as a numeric measurement."""
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int | float):
        return True
    if not isinstance(value, str):
        return False
    stripped = value.strip().replace(",", "").removesuffix("%")
    if stripped.startswith(("<", ">")):
        stripped = stripped[1:]
    if not stripped or stripped.upper() in {"N/A", "NULL", "SUPPRESS"} or stripped.upper().startswith("N<"):
        return False
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def _normalize_field_name(field_name: str) -> str:
    """Normalize a field name to lowercase alphanumerics for heuristic matching."""
    return "".join(character for character in field_name.lower() if character.isalnum())


def _merge_stats(entry: dict[str, Any], stats: dict[str, Any]) -> None:
    """Merge sampled schema-hint sets into the file-level manifest entry."""
    entry["_major_keys"].update(stats["major_keys"])
    entry["_nested_keys"].update(stats["nested_keys"])
    entry["_likely_id_fields"].update(stats["likely_id_fields"])
    entry["_likely_school_entity_fields"].update(stats["likely_school_entity_fields"])
    entry["_likely_district_fields"].update(stats["likely_district_fields"])
    entry["_likely_year_fields"].update(stats["likely_year_fields"])
    entry["_likely_numeric_metric_fields"].update(stats["likely_numeric_metric_fields"])


def _finalize_schema_fields(entry: dict[str, Any]) -> None:
    """Sort public schema lists and discard private accumulator sets."""
    for public_key, private_key in (
        ("major_keys", "_major_keys"),
        ("representative_nested_keys", "_nested_keys"),
        ("likely_id_fields", "_likely_id_fields"),
        ("likely_school_entity_fields", "_likely_school_entity_fields"),
        ("likely_district_fields", "_likely_district_fields"),
        ("likely_year_fields", "_likely_year_fields"),
        ("likely_numeric_metric_fields", "_likely_numeric_metric_fields"),
    ):
        entry[public_key] = sorted(entry[private_key])
        del entry[private_key]
    entry["schema_summary"]["major_key_count"] = len(entry["major_keys"])
    entry["schema_summary"]["representative_nested_key_count"] = len(entry["representative_nested_keys"])


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for a file in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_fingerprint(file_id: str, record_index: int, record: Any) -> str:
    """Build a deterministic preview identifier for one logical record."""
    canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _short_id("ospi_rec", f"{file_id}\0{record_index}\0{canonical}")


def _short_id(prefix: str, stable_input: str) -> str:
    """Return a stable shortened SHA-256-based identifier with a fixed prefix."""
    digest = hashlib.sha256(stable_input.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _format_timestamp(timestamp: float) -> str:
    """Render a POSIX timestamp as a UTC ISO 8601 string."""
    return datetime.fromtimestamp(timestamp, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _utc_now() -> str:
    """Return the current UTC time in normalized ISO 8601 form."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for corpus and output paths."""
    parser = argparse.ArgumentParser(description="Validate a repo-local OSPI JSON corpus without mutating inputs.")
    parser.add_argument("--corpus-path", required=True, type=Path, help="Explicit path to the OSPI corpus directory.")
    parser.add_argument("--output", required=True, type=Path, help="Explicit output path for the JSON manifest.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point and report success or validation errors."""
    args = parse_args(argv)
    try:
        manifest = build_manifest(args.corpus_path, args.output)
    except ManifestValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        "OSPI manifest written: "
        f"{Path(args.output).resolve(strict=False)} "
        f"({manifest['file_count']} files, {manifest['total_record_count']} records)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
