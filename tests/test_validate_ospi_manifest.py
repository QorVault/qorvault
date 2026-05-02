"""Synthetic tests for the read-only OSPI manifest validator."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_ospi_manifest.py"


def load_validator():
    """Import the validator module directly from the repo-local script path."""
    spec = importlib.util.spec_from_file_location("validate_ospi_manifest", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    """Return a small-file SHA-256 digest for fixture integrity assertions."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Write deterministic JSON fixture content to a synthetic test path."""
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_missing_corpus_path_fails(tmp_path: Path) -> None:
    """Missing corpus paths should fail fast with a validation error."""
    validator = load_validator()

    with pytest.raises(validator.ManifestValidationError, match="does not exist"):
        validator.build_manifest(tmp_path / "missing", tmp_path / "manifest.json")


def test_non_directory_corpus_path_fails(tmp_path: Path) -> None:
    """A file path must not be accepted where a corpus directory is required."""
    validator = load_validator()
    not_a_directory = tmp_path / "corpus.json"
    write_json(not_a_directory, [])

    with pytest.raises(validator.ManifestValidationError, match="not a directory"):
        validator.build_manifest(not_a_directory, tmp_path / "manifest.json")


def test_output_path_inside_corpus_is_rejected(tmp_path: Path) -> None:
    """Manifest output must stay outside the corpus tree."""
    validator = load_validator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_json(corpus / "assessment.json", [])

    with pytest.raises(validator.ManifestValidationError, match="outside the corpus"):
        validator.build_manifest(corpus, corpus / "manifest.json")


def test_output_parent_directory_must_exist(tmp_path: Path) -> None:
    """The validator should reject outputs whose parent directory is absent."""
    validator = load_validator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_json(corpus / "assessment.json", [])
    output_path = tmp_path / "missing" / "manifest.json"

    with pytest.raises(validator.ManifestValidationError, match="parent directory does not exist"):
        validator.build_manifest(corpus, output_path)

    assert not output_path.parent.exists()


def test_manifest_shape_deterministic_ordering_and_download_script_ignored(tmp_path: Path) -> None:
    """Manifest output should be deterministic and ignore the download helper script."""
    validator = load_validator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_json(corpus / "zeta.json", [{"schoolyear": "2024-25", "districtname": "Kent"}])
    write_json(corpus / "alpha.json", [{"schoolyear": "2023-24", "districtname": "Kent"}])
    (corpus / "download_ospi.py").write_text("raise SystemExit('do not run')\n", encoding="utf-8")

    manifest = validator.build_manifest(corpus, tmp_path / "manifest.json")

    assert manifest["schema_version"] == "ospi-manifest-validator/v1"
    assert manifest["corpus_path"] == str(corpus.resolve())
    assert manifest["file_count"] == 2
    assert [file_entry["name"] for file_entry in manifest["files"]] == ["alpha.json", "zeta.json"]
    assert "download_ospi.py" not in {file_entry["name"] for file_entry in manifest["files"]}
    assert manifest["total_record_count"] == 2
    assert manifest["constraints_observed"]["download_ospi_py_ignored"] is True
    assert manifest["constraints_observed"]["read_only_validator"] is True


def test_top_level_array_file_schema_counts_and_ids(tmp_path: Path) -> None:
    """Top-level arrays should produce stable schema hints and record previews."""
    validator = load_validator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_json(
        corpus / "attendance.json",
        [
            {
                "countyname": "King",
                "districtcode": "17415",
                "districtname": "Kent School District",
                "regularattendance_rate": "82.4",
                "schoolcode": "1234",
                "schoolname": "Example High",
                "schoolyear": "2024-25",
                "studentgroup": "All Students",
            },
            {
                "districtcode": "17415",
                "districtname": "Kent School District",
                "regularattendance_rate": "81.2",
                "schoolcode": "5678",
                "schoolname": "Example Middle",
                "schoolyear": "2024-25",
            },
        ],
    )

    first = validator.build_manifest(corpus, tmp_path / "manifest-one.json")
    second = validator.build_manifest(corpus, tmp_path / "manifest-two.json")
    file_entry = first["files"][0]

    assert file_entry["top_level_shape"] == "array"
    assert file_entry["record_count"] == 2
    assert file_entry["major_keys"][:3] == ["countyname", "districtcode", "districtname"]
    assert "schoolyear" in file_entry["likely_year_fields"]
    assert "schoolname" in file_entry["likely_school_entity_fields"]
    assert "districtname" in file_entry["likely_district_fields"]
    assert "regularattendance_rate" in file_entry["likely_numeric_metric_fields"]
    assert file_entry["corpus_file_id"] == second["files"][0]["corpus_file_id"]
    assert file_entry["record_id_preview"] == second["files"][0]["record_id_preview"]
    assert file_entry["record_id_preview"][0]["record_index"] == 0
    assert file_entry["record_id_preview"][0]["record_fingerprint"].startswith("ospi_rec_")


def test_top_level_object_file_schema_counts_nested_keys_and_ids(tmp_path: Path) -> None:
    """Top-level objects with record arrays should expose nested schema details."""
    validator = load_validator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_json(
        corpus / "summary.json",
        {
            "metadata": {"source": "synthetic", "schoolyear": "2024-25"},
            "records": [
                {"districtname": "Kent", "metricvalue": 42, "schoolname": "Example A"},
                {"districtname": "Kent", "metricvalue": 43, "schoolname": "Example B"},
            ],
        },
    )

    manifest = validator.build_manifest(corpus, tmp_path / "manifest.json")
    file_entry = manifest["files"][0]

    assert file_entry["top_level_shape"] == "object"
    assert file_entry["record_count"] == 2
    assert file_entry["record_container_path"] == "$.records"
    assert "records[].metricvalue" in file_entry["representative_nested_keys"]
    assert file_entry["corpus_file_id"].startswith("ospi_file_")
    assert len(file_entry["record_id_preview"]) == 2


def test_malformed_and_empty_json_files_warn_without_aborting(tmp_path: Path) -> None:
    """Malformed or empty files should warn without aborting the manifest build."""
    validator = load_validator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "broken.json").write_text('{"missing": ', encoding="utf-8")
    (corpus / "empty.json").write_text("", encoding="utf-8")

    manifest = validator.build_manifest(corpus, tmp_path / "manifest.json")

    assert manifest["file_count"] == 2
    assert manifest["total_record_count"] == 0
    assert len(manifest["warnings"]) == 2
    assert {entry["top_level_shape"] for entry in manifest["files"]} == {"malformed", "empty"}


def test_checksum_stability_and_no_input_mutation(tmp_path: Path) -> None:
    """Running the validator must not mutate input files or their checksums."""
    validator = load_validator()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / "growth.json"
    write_json(source, [{"districtname": "Kent", "growthpercentile": 55, "schoolyear": "2024-25"}])
    before = (source.stat().st_size, source.stat().st_mtime_ns, sha256_file(source))

    first = validator.build_manifest(corpus, tmp_path / "manifest-one.json")
    second = validator.build_manifest(corpus, tmp_path / "manifest-two.json")
    after = (source.stat().st_size, source.stat().st_mtime_ns, sha256_file(source))

    assert before == after
    assert first["files"][0]["sha256"] == second["files"][0]["sha256"] == before[2]
    assert first["files"][0]["size_bytes"] == before[0]


def test_cli_writes_manifest_to_explicit_output_path(tmp_path: Path) -> None:
    """The CLI should write the manifest only to the explicit output path."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_json(corpus / "graduation.json", [{"districtname": "Kent", "graduationrate": "91.0"}])
    output_path = tmp_path / "out" / "manifest.json"
    output_path.parent.mkdir()

    result = subprocess.run(  # noqa: S603 - test runs this repo-local script with synthetic tmp_path fixtures.
        [sys.executable, str(VALIDATOR_PATH), "--corpus-path", str(corpus), "--output", str(output_path)],
        check=False,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["file_count"] == 1
    assert manifest["files"][0]["name"] == "graduation.json"
