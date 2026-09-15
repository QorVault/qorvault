"""Census of voucher artifacts across Postgres and the corpus on disk.

The two inventories do not agree, and the difference is the point.

* ``documents`` holds a row for every artifact the ingest pipeline saw. It is
  the only place a stable ``document_id`` exists, and locators need one.
* The corpus directories hold every artifact that was ever **scraped**,
  including voucher sets that were never ingested. Phase 0 found the
  2026-03-25 and 2026-05-27 sets in exactly this category: the agenda item
  is in ``documents``, the PDFs beneath it are not.

Taking either inventory alone would be wrong in a way that matters. The
database alone silently drops the two most recent voucher nights; the disk
alone has no document ids to anchor locators to.

The same file is reachable under more than one corpus root, and the scraper
stored some meetings twice under differently punctuated slugs. Deduplication
is therefore by SHA-256 of the file content, not by path or by name -- a
name-based rule would keep ``GF Vouchers 3-25-26.pdf`` and
``General Fund Vouchers 03-25-26.pdf`` as two sets of the same money.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from classify import (
    doc_class_from_title,
    excluded_reason,
    fiscal_year,
    fund_from_title,
)
from locators import CORPUS_ROOTS, meeting_date_from_path, resolve_pdf_path, sha256_of

# A candidate is anything whose title or file name uses the voucher /
# warrant vocabulary. Deliberately broad: Phase 0's job is to see everything
# and then say what it excluded, not to filter first and count later.
CANDIDATE_TITLE_RX = re.compile(
    r"voucher|warrant|BD[_ ]?MTG|Board\s+M(?:ee)?t(?:in)?g",
    re.I,
)

# Directory name the scraper gives a voucher agenda item.
VOUCHER_ITEM_DIR_RX = re.compile(r"voucher", re.I)

CANDIDATE_SQL = """
    SELECT id::text            AS document_id,
           document_type,
           title,
           meeting_date::text  AS meeting_date,
           file_path,
           page_count,
           processing_status,
           length(content_text) AS content_len
    FROM documents
    WHERE title ILIKE ANY (%s)
    ORDER BY meeting_date, title
"""

# Text-side sweep: catches listings whose title names no fund and no
# voucher word but whose first page carries the accounting system's header.
PHRASE_SQL = """
    SELECT id::text            AS document_id,
           document_type,
           title,
           meeting_date::text  AS meeting_date,
           file_path,
           page_count,
           processing_status,
           length(content_text) AS content_len
    FROM documents
    WHERE document_type = 'attachment'
      AND content_text IS NOT NULL
      AND left(content_text, 4000) ~* %s
    ORDER BY meeting_date, title
"""

# Voucher agenda items -- the board's own index of which meetings were
# voucher nights. Needed to answer "meetings with a voucher agenda item but
# no PDF", which no file inventory can answer on its own.
AGENDA_ITEM_SQL = """
    SELECT id::text            AS document_id,
           title,
           meeting_date::text  AS meeting_date,
           file_path,
           length(content_text) AS content_len
    FROM documents
    WHERE document_type = 'agenda_item'
      AND title ~* '(^|[^a-z])vouchers?([^a-z]|$)'
    ORDER BY meeting_date
"""

TITLE_PATTERNS = [
    "%voucher%",
    "%warrant%",
    "%bdmtg%",
    "%bd mtg%",
    "%board mtg%",
]

PHRASE_PATTERN = (
    r"(General Fund|ACH|Capital Projects? Fund|Associated Student Body|ASB Fund"
    r"|Trust Fund|Transportation Vehicle Fund|Custodial Fund|Permanent Funds?)"
    r"\s+(Warrants?|Payments?)"
)


@dataclass
class Artifact:
    """One voucher-shaped artifact, from the database, the disk, or both.

    Attributes:
        title: Title as stored, or the file name when disk-only.
        meeting_date: ISO meeting date.
        document_id: ``documents.id`` when a row exists, else None.
        document_type: ``documents.document_type`` when a row exists.
        file_path: Stored path, when a row exists.
        resolved_path: Existing path on disk, when one was found.
        fund: Fund code from the title.
        doc_class: Artifact class from the title.
        page_count: Page count as recorded by ingest, when known.
        content_len: Length of ``content_text``, when a row exists.
        processing_status: Ingest status, when a row exists.
        origin: ``db``, ``disk``, or ``both``.
        source_root: Corpus root the disk copy was found under.
        exclusion: Why the artifact is not a voucher set, when it is not.
        sha256: Content digest, when the file could be read.
        duplicate_of: Title of the artifact this one duplicates.
        notes: Free-text observations carried into the report.
    """

    title: str
    meeting_date: str | None
    document_id: str | None = None
    document_type: str | None = None
    file_path: str | None = None
    resolved_path: str | None = None
    fund: str | None = None
    doc_class: str | None = None
    page_count: int | None = None
    content_len: int | None = None
    processing_status: str | None = None
    origin: str = "db"
    source_root: str | None = None
    exclusion: str | None = None
    sha256: str | None = None
    duplicate_of: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def fiscal_year(self) -> str | None:
        """Washington school fiscal year of the meeting."""
        return fiscal_year(self.meeting_date)

    @property
    def is_pdf(self) -> bool:
        """Whether the artifact is a PDF and therefore parseable at all."""
        return self.title.lower().endswith(".pdf")


def _classify(art: Artifact) -> Artifact:
    """Attach exclusion reason, fund and document class to an artifact.

    Args:
        art: Artifact with a title already set.

    Returns:
        The same artifact, classified in place.
    """
    # A BoardDocs agenda item is the board's index entry for a voucher
    # night, not a listing of money. Classifying it as a detail listing
    # would create a set with no rows and no total.
    if art.document_type == "agenda_item":
        art.doc_class = "voucher_agenda_item"
        return art
    art.exclusion = excluded_reason(art.title)
    if art.exclusion is None:
        art.fund = fund_from_title(art.title)
        art.doc_class = doc_class_from_title(art.title)
        if art.doc_class is None:
            art.exclusion = "unclassified"
    return art


def db_artifacts(query_dicts) -> list[Artifact]:
    """Collect voucher-shaped artifacts recorded in ``documents``.

    Args:
        query_dicts: Read-only dict-returning query callable.

    Returns:
        Classified artifacts, one per matching ``documents`` row.
    """
    rows = list(query_dicts(CANDIDATE_SQL, (TITLE_PATTERNS,)))
    seen = {r["document_id"] for r in rows}
    for row in query_dicts(PHRASE_SQL, (PHRASE_PATTERN,)):
        if row["document_id"] not in seen:
            row = dict(row)
            row["_phrase_only"] = True
            rows.append(row)

    out: list[Artifact] = []
    for row in rows:
        art = Artifact(
            title=row["title"] or "",
            meeting_date=row["meeting_date"],
            document_id=row["document_id"],
            document_type=row["document_type"],
            file_path=row["file_path"],
            resolved_path=resolve_pdf_path(row["file_path"]),
            page_count=row["page_count"],
            content_len=row["content_len"],
            processing_status=row["processing_status"],
            origin="db",
        )
        if row.get("_phrase_only"):
            art.notes.append("found by first-page phrase, not by title")
        out.append(_classify(art))
    return out


def voucher_agenda_items(query_dicts) -> list[dict]:
    """Return the voucher agenda items, one per voucher night on the agenda.

    Args:
        query_dicts: Read-only dict-returning query callable.

    Returns:
        Raw rows, ordered by meeting date.
    """
    return list(query_dicts(AGENDA_ITEM_SQL, None))


def disk_artifacts() -> list[Artifact]:
    """Walk the corpus roots for voucher PDFs, ingested or not.

    Returns:
        Classified artifacts, one per PDF found on disk.
    """
    out: list[Artifact] = []
    for root in CORPUS_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            in_voucher_dir = bool(VOUCHER_ITEM_DIR_RX.search(os.path.basename(dirpath)))
            for name in filenames:
                if not name.lower().endswith(".pdf"):
                    continue
                if not (in_voucher_dir or CANDIDATE_TITLE_RX.search(name)):
                    continue
                path = os.path.join(dirpath, name)
                art = Artifact(
                    title=name,
                    meeting_date=meeting_date_from_path(path),
                    resolved_path=path,
                    origin="disk",
                    source_root=root,
                )
                art = _classify(art)
                # A PDF sitting inside a "...-vouchers" agenda item directory
                # is a voucher artifact even when its name says nothing --
                # the board filed it under the voucher item.
                if art.exclusion == "unclassified" and in_voucher_dir:
                    art.exclusion = None
                    art.doc_class = "unnamed_in_voucher_item"
                    art.notes.append("named nothing; filed under a voucher agenda item")
                out.append(art)
    return out


def merge(db: list[Artifact], disk: list[Artifact]) -> list[Artifact]:
    """Merge the two inventories, deduplicating by file content.

    Args:
        db: Artifacts from ``documents``.
        disk: Artifacts found by walking the corpus.

    Returns:
        One artifact per distinct file content, database rows preferred so
        that a document id survives wherever one exists. Duplicates are
        dropped from the result and counted on the survivor's ``notes``.
    """
    by_path: dict[str, Artifact] = {}
    orphans: list[Artifact] = []
    for art in db:
        if art.resolved_path:
            by_path.setdefault(os.path.realpath(art.resolved_path), art)
        else:
            orphans.append(art)

    candidates: list[Artifact] = list(by_path.values())
    for art in disk:
        key = os.path.realpath(art.resolved_path or "")
        existing = by_path.get(key)
        if existing is not None:
            existing.origin = "both"
            existing.source_root = art.source_root
        else:
            candidates.append(art)

    # Content dedupe. Database-backed artifacts sort first so the surviving
    # copy is the one carrying a document id.
    candidates.sort(key=lambda a: (a.document_id is None, a.meeting_date or "", a.title))
    by_digest: dict[str, Artifact] = {}
    merged: list[Artifact] = []
    for art in candidates:
        if art.resolved_path:
            art.sha256 = sha256_of(art.resolved_path)
        if art.sha256:
            first = by_digest.get(art.sha256)
            if first is not None:
                first.notes.append(f"duplicate copy: {art.title}")
                continue
            by_digest[art.sha256] = art
        merged.append(art)

    merged.extend(orphans)
    merged.sort(key=lambda a: (a.meeting_date or "", a.fund or "", a.title))
    return merged
