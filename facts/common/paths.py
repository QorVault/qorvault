"""Resolution of stored document paths onto files that actually exist.

``documents.file_path`` is stale corpus-wide, and it is stale under **more
than one root**:

* the bulk 2005-2026 corpus is stored as ``/home/donald/ksd_forensic/...``
  and survives only inside the backup archive;
* a later 2026 re-scrape is stored as
  ``/home/donald/workspace/projects/ksd_forensic/boarddocs/data/...`` and
  still exists on the live filesystem, but its ``data`` directory was
  renamed ``data_DO_NOT_LOAD`` after ingest.

Both stale prefixes begin ``/home/donald/``, so the rewrites must be tried
**most specific first**. A generic rewrite applied first produces a path
that does not exist, and a missing rewrite is indistinguishable from a
missing file -- the failure mode is silence, not an error.

Beyond the scraped corpus there are two further sources of voucher PDFs:

``MEETING_FILES_ROOT``
    A meeting directory placed by hand rather than by the scraper.

``STAGING_ROOT``
    Packets staged by the operator for a meeting that posts after the last
    scrape. This is a first-class input, not a workaround: a voucher packet
    is published on meeting night, and the scrape that would collect it
    runs later. Directories here are named by bare ISO date
    (``2026-06-24``) rather than by scraper slug.

This module is deliberately free of any package-local import so that every
fact package can share one answer.
"""

from __future__ import annotations

import os
import re

# Ordered (stale prefix, replacement) pairs. First existing hit wins, and
# the order is load-bearing -- see the module docstring.
PATH_REWRITES: tuple[tuple[str, str], ...] = (
    (
        "/home/donald/workspace/projects/ksd_forensic/boarddocs/data/",
        "/home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD/",
    ),
    (
        "/home/donald/ksd_forensic/",
        "/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/",
    ),
    (
        "/home/donald/",
        "/home/donald/qorvault-dev-archive/framework-backup/home/",
    ),
)

ARCHIVE_CORPUS_ROOT = "/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data"
LIVE_CORPUS_ROOT = "/home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD"
MEETING_FILES_ROOT = "/home/donald/workspace/meeting_files"
STAGING_ROOT = "/home/donald/workspace/staging/vouchers-2026"

# Every root that may hold a voucher PDF, whether or not ingest has seen it.
CORPUS_ROOTS: tuple[str, ...] = (
    ARCHIVE_CORPUS_ROOT,
    LIVE_CORPUS_ROOT,
    MEETING_FILES_ROOT,
    STAGING_ROOT,
)

# Roots whose files are staged by the operator rather than scraped. A row
# sourced from one of these carries source='staged_pdf' until ingest gives
# it a document id.
STAGED_ROOTS: tuple[str, ...] = (STAGING_ROOT,)

# A meeting directory is named either by scraper slug
# ("2026-03-25-regular-meeting-6-30-p-m-") or, under the staging root, by
# bare ISO date ("2026-06-24").
MEETING_SLUG_RX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:-(.*))?$")


def resolve_pdf_path(file_path: str | None) -> str | None:
    """Map a stored ``documents.file_path`` onto a file that actually exists.

    Args:
        file_path: Value of ``documents.file_path``, possibly stale or None.

    Returns:
        An existing filesystem path to a file, or None if nothing resolves.
    """
    if not file_path:
        return None
    if os.path.isfile(file_path):
        return file_path
    for stale, replacement in PATH_REWRITES:
        if file_path.startswith(stale):
            candidate = file_path.replace(stale, replacement, 1)
            if os.path.isfile(candidate):
                return candidate
    return None


def resolve_any_path(file_path: str | None) -> str | None:
    """Resolve a stored path to a file **or** directory that exists.

    BoardDocs agenda items are scraped as directories, so a voucher agenda
    item's ``file_path`` names a directory rather than a file.

    Args:
        file_path: Value of ``documents.file_path``, possibly stale or None.

    Returns:
        An existing filesystem path, or None if nothing resolves.
    """
    if not file_path:
        return None
    if os.path.exists(file_path):
        return file_path
    for stale, replacement in PATH_REWRITES:
        if file_path.startswith(stale):
            candidate = file_path.replace(stale, replacement, 1)
            if os.path.exists(candidate):
                return candidate
    return None


def meeting_date_from_path(path: str) -> str | None:
    """Extract the meeting date encoded in a path's meeting directory.

    The directory name carries the meeting date, which is the only date
    available for a PDF that has no ``documents`` row.

    Args:
        path: Any path beneath a corpus or staging root.

    Returns:
        ISO date string, or None when no meeting directory is present.
    """
    for part in os.path.normpath(path).split(os.sep):
        match = MEETING_SLUG_RX.match(part)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def is_staged(path: str | None) -> bool:
    """Whether a resolved path came from an operator-staged directory.

    Args:
        path: A resolved filesystem path.

    Returns:
        True when the file was staged rather than scraped.
    """
    if not path:
        return False
    real = os.path.realpath(path)
    return any(real.startswith(os.path.realpath(root)) for root in STAGED_ROOTS)


def source_kind(path: str | None) -> str:
    """Classify where a resolved file came from.

    Args:
        path: A resolved filesystem path.

    Returns:
        ``staged_pdf`` or ``corpus_pdf``.
    """
    return "staged_pdf" if is_staged(path) else "corpus_pdf"
