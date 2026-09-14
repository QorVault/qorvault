#!/usr/bin/env python3
"""Prove the 31 flat-layout attachment rows duplicate a correctly linked row.

A pair qualifies for deletion only if the two rows point at files whose bytes
hash identically under SHA-256. Equal ``file_size_bytes`` is not sufficient --
two different PDFs can share a byte count -- and the diagnostic that produced
the 31 matched on size and char_count alone.

Any pair that fails for any reason (hash mismatch, missing file, unreadable
file) is reported and drops out of the delete set. This script never widens the
set and never deletes anything; it only classifies.

Read-only: SELECTs against the corpus, reads of on-disk PDFs, no writes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

# documents.file_path was written on a host where the scrape tree lived at
# /home/donald/ksd_forensic. That path does not exist here; the tree the corpus
# was actually built from is the framework backup. Remap on read only -- the
# stale values in the database are not modified by this script.
STALE_PREFIX = "/home/donald/ksd_forensic"
REAL_PREFIX = "/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic"

PAIR_SQL = r"""
WITH unlinked AS (
    SELECT id, meeting_id, meeting_date, title, external_id, file_path
    FROM documents
    WHERE document_type='attachment' AND agenda_item_id IS NULL
      AND meeting_date>='2026-01-01' AND external_id NOT LIKE 'email_%'
),
linked AS (
    SELECT id, meeting_id, agenda_item_id, title, external_id, file_path,
           regexp_replace(title,'\s+','_','g') AS santitle
    FROM documents
    WHERE document_type='attachment' AND agenda_item_id IS NOT NULL
      AND meeting_date>='2026-01-01'
)
SELECT json_build_object(
    'flat_id', u.id, 'flat_external_id', u.external_id, 'flat_path', u.file_path,
    'kept_id', l.id, 'kept_external_id', l.external_id, 'kept_path', l.file_path,
    'kept_agenda_item_id', l.agenda_item_id,
    'meeting_id', u.meeting_id, 'meeting_date', u.meeting_date
)::text
FROM unlinked u
JOIN linked l ON l.meeting_id = u.meeting_id AND l.santitle = u.title
ORDER BY u.external_id;
"""


def query(sql: str) -> list[str]:
    """Run SQL in the corpus container and return non-empty output lines."""
    out = subprocess.run(
        [
            "podman",
            "exec",
            "-i",
            "boarddocs-postgres",
            "psql",
            "-U",
            "boarddocs",
            "-d",
            "boarddocs",
            "-P",
            "pager=off",
            "-At",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def resolve(path: str) -> Path:
    """Map a stored (stale) file_path onto this host's actual scrape tree."""
    if path.startswith(STALE_PREFIX):
        path = REAL_PREFIX + path[len(STALE_PREFIX) :]
    return Path(path)


def sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed so large PDFs stay off the heap."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    """Hash both sides of every candidate pair and classify it.

    Returns:
        Process exit code -- 0 normally, 1 only if a kept record turns out to
        have no agenda_item_id, which would mean the matcher is wrong and no
        deletion should proceed.
    """
    pairs = [json.loads(ln) for ln in query(PAIR_SQL)]
    print(f"pairs returned by matcher: {len(pairs)}")

    identical, failed = [], []

    for p in pairs:
        flat, kept = resolve(p["flat_path"]), resolve(p["kept_path"])
        record = dict(p)

        for role, path in (("flat", flat), ("kept", kept)):
            record[f"{role}_resolved"] = str(path)
            record[f"{role}_exists"] = path.is_file()

        if not (record["flat_exists"] and record["kept_exists"]):
            record["reason"] = "file missing on disk"
            failed.append(record)
            continue

        try:
            record["flat_sha256"] = sha256(flat)
            record["kept_sha256"] = sha256(kept)
        except OSError as exc:  # unreadable/corrupt -- cannot prove identity
            record["reason"] = f"read error: {exc}"
            failed.append(record)
            continue

        record["flat_bytes"] = flat.stat().st_size
        record["kept_bytes"] = kept.stat().st_size

        if record["flat_sha256"] == record["kept_sha256"]:
            identical.append(record)
        else:
            record["reason"] = "SHA-256 mismatch"
            failed.append(record)

    print(f"byte-identical (safe to delete): {len(identical)}")
    print(f"NOT proven identical (excluded): {len(failed)}")
    print()

    print("=== Confirmed byte-identical pairs ===")
    print(f"{'flat external_id':<52} {'sha256[:16]':<18} {'kept agenda_item_id'}")
    for r in identical:
        print(f"{r['flat_external_id']:<52} {r['flat_sha256'][:16]:<18} " f"{r['kept_agenda_item_id']}")

    if failed:
        print()
        print("=== EXCLUDED from the delete set ===")
        for r in failed:
            print(f"  {r['flat_external_id']}")
            print(f"    reason: {r['reason']}")
            if "flat_sha256" in r:
                print(f"    flat: {r['flat_sha256']}  ({r.get('flat_bytes')} bytes)")
                print(f"    kept: {r['kept_sha256']}  ({r.get('kept_bytes')} bytes)")

    # Every kept record must be the linked one. This is a property of the
    # matcher (the kept side is selected with agenda_item_id IS NOT NULL), but
    # assert it explicitly so the claim in the report is backed by a check.
    unlinked_keeps = [r for r in identical if not r["kept_agenda_item_id"]]
    print()
    print(f"kept records carrying an agenda_item_id: " f"{len(identical) - len(unlinked_keeps)}/{len(identical)}")
    if unlinked_keeps:
        print("ERROR: a kept record has no agenda_item_id -- do not proceed")
        return 1

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps({"identical": identical, "failed": failed}, indent=2, default=str))
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
