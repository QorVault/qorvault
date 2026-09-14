#!/usr/bin/env python3
"""Derive agenda_item_id for the 40 orphaned 2026-02-11 attachments.

The diagnostic proposed recovering the link by parsing item anchors out of the
retained ``agenda.html``. That is not possible: the stored HTML is BoardDocs'
PRINT-AgendaDetailed view, which emits goal IDs and file IDs but no agenda-item
IDs, and ``goto?open&id=`` anchors appear zero times in it.

A better source exists on disk. The structured scraper ran against both
2026-02-11 meetings on 2026-02-10 and left an ``archive_*`` subdirectory of
per-item folders, each holding an ``item.json`` with ``itemId`` and a ``links``
array carrying each attachment's BoardDocs file ID and original filename. That
gives an exact, ID-level join rather than a filename guess.

Two independent methods are computed per record and must agree:

  file_id  -- the flat filename is matched to an href in the retained
              agenda.html to recover the BoardDocs file ID, which is then
              looked up in item.json ``links[].unique``. Joins on an opaque ID.
  filename -- the flat filename is matched directly against a sanitized
              item.json ``links[].filename``.

Confidence is ``exact`` only when both methods resolve, agree, and the match is
unique. Anything else is reported and stays unlinked -- this script never
guesses and never widens the link set.

Read-only: SELECTs plus reads of on-disk scrape output. Writes no database rows.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path("/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data")

# Flat meeting_id -> the structured scrape directory holding the archive_* tree.
MEETING_DIRS = {
    "DQU45Y09F4B0": "2026-02-11-regular-meeting-6-30-p-m-",
    "DQU47R0A3706": "2026-02-11-special-meeting-work-session-5-00-p-m-",
}

ORPHAN_SQL = r"""
WITH unlinked AS (
    SELECT id, meeting_id, meeting_date, title, external_id
    FROM documents
    WHERE document_type='attachment' AND agenda_item_id IS NULL
      AND meeting_date>='2026-01-01' AND external_id NOT LIKE 'email_%'
),
linked AS (
    SELECT meeting_id, regexp_replace(title,'\s+','_','g') AS santitle
    FROM documents
    WHERE document_type='attachment' AND agenda_item_id IS NOT NULL
      AND meeting_date>='2026-01-01'
)
SELECT json_build_object('id', u.id, 'meeting_id', u.meeting_id,
                         'title', u.title, 'external_id', u.external_id)::text
FROM unlinked u
WHERE NOT EXISTS (SELECT 1 FROM linked l
                  WHERE l.meeting_id = u.meeting_id AND l.santitle = u.title)
ORDER BY u.meeting_id, u.external_id;
"""

AGENDA_SQL = """
SELECT meeting_id || E'\\t' || replace(replace(content_raw, E'\\n', ' '), E'\\r', ' ')
FROM documents
WHERE document_type='agenda' AND meeting_id IN ('DQU45Y09F4B0','DQU47R0A3706');
"""

# href="/wa/ksdwa/Board.nsf/files/<FILEID>/$file/<url-encoded filename>"
HREF_RE = re.compile(r'Board\.nsf/files/([A-Z0-9]+)/\$file/([^"\'>\s]+)')


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


def sanitize(name: str) -> str:
    """Apply the flat scraper's filename rule: runs of whitespace -> one '_'.

    A *run* of whitespace collapses to a single underscore, not one underscore
    per space. Confirmed against the scrape output on disk: BoardDocs serves
    "KE  - Install New Fence Along Meeker Street - RFQ and Quote.pdf" with a
    double space, and the flat scraper wrote
    "KE_-_Install_New_Fence_Along_Meeker_Street_-_RFQ_and_Quote.pdf".
    A naive space->underscore replacement yields a doubled underscore and
    silently fails to match those two records.

    Always applied in this direction (original -> sanitized). The reverse is
    ambiguous, because a filename may legitimately contain an underscore, so
    reconstructing the original from the sanitized form can produce a wrong
    string -- the failure mode the diagnostic hit on
    "1707 Certified Signatures_Real Estate Transactions 2026.pdf".
    """
    return re.sub(r"\s+", "_", name)


def load_items(meeting_id: str) -> list[dict]:
    """Load every item.json under a meeting's archive_* tree."""
    root = DATA_ROOT / MEETING_DIRS[meeting_id]
    return [json.loads(p.read_text(encoding="utf-8", errors="replace")) for p in sorted(root.rglob("item.json"))]


def build_indexes(items: list[dict]) -> tuple[dict, dict]:
    """Index items by attachment file ID and by sanitized attachment filename."""
    by_file_id: dict[str, set[str]] = defaultdict(set)
    by_filename: dict[str, set[str]] = defaultdict(set)
    for it in items:
        item_id = it.get("itemId")
        for link in it.get("links") or []:
            if link.get("unique"):
                by_file_id[link["unique"]].add(item_id)
            if link.get("filename"):
                by_filename[sanitize(link["filename"])].add(item_id)
    return by_file_id, by_filename


def agenda_file_ids(html: str) -> dict[str, set[str]]:
    """Map sanitized filename -> file IDs, from hrefs in the retained agenda."""
    out: dict[str, set[str]] = defaultdict(set)
    for file_id, enc_name in HREF_RE.findall(html):
        name = urllib.parse.unquote(enc_name)
        out[sanitize(name)].add(file_id)
    return out


def main() -> int:
    """Resolve every orphan to an agenda item, or report why it cannot be.

    Returns:
        Process exit code -- always 0; unresolved records are reported, not
        treated as an error, because leaving a record unlinked is the correct
        outcome when the evidence does not support a link.
    """
    orphans = [json.loads(ln) for ln in query(ORPHAN_SQL)]
    print(f"orphans returned by matcher: {len(orphans)}")

    agendas = {}
    for line in query(AGENDA_SQL):
        mid, _, html = line.partition("\t")
        agendas[mid] = html

    caches: dict[str, tuple] = {}
    for mid in MEETING_DIRS:
        items = load_items(mid)
        by_file_id, by_filename = build_indexes(items)
        caches[mid] = (by_file_id, by_filename, agenda_file_ids(agendas.get(mid, "")))
        print(f"  {mid}: {len(items)} item.json, " f"{len(by_file_id)} indexed attachments")
    print()

    resolved, unresolved = [], []

    for o in orphans:
        mid, title = o["meeting_id"], o["title"]
        rec = dict(o)

        if mid not in caches:
            rec["reason"] = f"no archive tree mapped for meeting {mid}"
            unresolved.append(rec)
            continue

        by_file_id, by_filename, agenda_ids = caches[mid]

        # Method 1: filename -> file ID (from agenda.html) -> itemId.
        file_ids = agenda_ids.get(title, set())
        rec["file_ids_from_agenda"] = sorted(file_ids)
        m1 = set()
        for fid in file_ids:
            m1 |= by_file_id.get(fid, set())

        # Method 2: filename -> itemId directly from item.json links.
        m2 = by_filename.get(title, set())

        rec["item_ids_by_file_id"] = sorted(m1)
        rec["item_ids_by_filename"] = sorted(m2)

        if len(m1) == 1 and m1 == m2:
            rec["agenda_item_id"] = next(iter(m1))
            rec["method"] = "file_id+filename (both agree)"
            rec["confidence"] = "exact"
            resolved.append(rec)
        elif not m1 and not m2:
            rec["reason"] = (
                "attachment absent from the 2026-02-10 archive scrape " "(added to the agenda after that scrape ran)"
            )
            unresolved.append(rec)
        elif m1 != m2:
            rec["reason"] = f"methods disagree: by_file_id={sorted(m1)} by_filename={sorted(m2)}"
            unresolved.append(rec)
        else:
            rec["reason"] = f"ambiguous: resolves to {len(m1 or m2)} items"
            unresolved.append(rec)

    print(f"resolved to an exact agenda_item_id: {len(resolved)}")
    print(f"remaining unlinked (listed, not guessed): {len(unresolved)}")
    print()

    print("=== RESOLVED ===")
    print(f"{'external_id':<62} {'agenda_item_id':<14} confidence")
    for r in resolved:
        print(f"{r['external_id']:<62} {r['agenda_item_id']:<14} {r['confidence']}")

    if unresolved:
        print()
        print("=== STAYS UNLINKED ===")
        for r in unresolved:
            print(f"  {r['external_id']}")
            print(f"    {r['reason']}")

    by_meeting: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in resolved:
        by_meeting[r["meeting_id"]][0] += 1
    for r in unresolved:
        by_meeting[r["meeting_id"]][1] += 1
    print()
    print("=== per meeting ===")
    for mid, (ok, no) in sorted(by_meeting.items()):
        print(f"  {mid}: resolved={ok} unlinked={no}")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps({"resolved": resolved, "unresolved": unresolved}, indent=2, default=str))
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
