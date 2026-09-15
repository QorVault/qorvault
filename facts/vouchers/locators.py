"""Locator resolution for voucher fact rows.

Every voucher fact row must carry the source document, a page number, a
character offset and a verbatim quote. Chunk ids are never a locator.

``facts/minutes/locators.py`` established that Postgres holds no page-level
text: ``document_pages`` is empty and ``chunks.source_page`` is NULL corpus
wide. Page numbers therefore come from re-reading the source PDF with
pdfplumber, and so does the text the parser runs on -- the voucher row regex
depends on column geometry that ``documents.content_text`` does not preserve.

Two differences from the minutes package, both established by Phase 0 recon:

1. ``documents.file_path`` is stale under **two** different roots, not one.
   The 2005-2026 bulk corpus is rooted at ``/home/donald/ksd_forensic/``; a
   later 2026 re-scrape is rooted at
   ``/home/donald/workspace/projects/ksd_forensic/`` and its ``data``
   directory has since been renamed ``data_DO_NOT_LOAD``. The minutes
   package's single prefix rewrite resolves the first and silently misses
   the second.

2. Voucher PDFs exist on disk that have **no ``documents`` row at all** (the
   2026-03-25 and 2026-05-27 sets). Those are addressed by path, so this
   module resolves in both directions: document -> path, and path -> the
   meeting it belongs to.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

LOG = logging.getLogger(__name__)

# Ordered (stale prefix, replacement) pairs. First existing hit wins.
#
# Order matters: the workspace rewrite must be tried before the bare
# /home/donald/ rewrite, because the workspace paths also start with
# /home/donald/ and the archive has no workspace/ subtree -- an unordered
# match would resolve nothing and look like a missing file.
PATH_REWRITES: tuple[tuple[str, str], ...] = (
    # Later 2026 re-scrape. The live tree still holds the files; only the
    # leaf directory was renamed after ingest.
    (
        "/home/donald/workspace/projects/ksd_forensic/boarddocs/data/",
        "/home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD/",
    ),
    # Bulk 2005-2026 corpus. Its only surviving copy is the backup archive.
    (
        "/home/donald/ksd_forensic/",
        "/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/",
    ),
    # Generic archive fallback, as used by facts/minutes.
    (
        "/home/donald/",
        "/home/donald/qorvault-dev-archive/framework-backup/home/",
    ),
)

# Corpus roots that hold voucher PDFs, whether or not a documents row exists.
CORPUS_ROOTS: tuple[str, ...] = (
    "/home/donald/qorvault-dev-archive/framework-backup/home/ksd_forensic/boarddocs/data",
    "/home/donald/workspace/projects/ksd_forensic/boarddocs/data_DO_NOT_LOAD",
    "/home/donald/workspace/meeting_files",
)

# Meeting directory slugs begin with an ISO date: "2026-03-25-regular-...".
MEETING_SLUG_RX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.*)$")


def resolve_pdf_path(file_path: str | None) -> str | None:
    """Map a stored ``documents.file_path`` onto a file that actually exists.

    Args:
        file_path: Value of ``documents.file_path``, possibly stale or None.

    Returns:
        An existing filesystem path, or None if nothing resolves.
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
    """Extract the meeting date encoded in a corpus path's meeting directory.

    The scraped directory name carries the meeting date, which is the only
    date available for a PDF that has no ``documents`` row.

    Args:
        path: Any path beneath a corpus root.

    Returns:
        ISO date string, or None when no meeting slug is present.
    """
    for part in os.path.normpath(path).split(os.sep):
        m = MEETING_SLUG_RX.match(part)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


@dataclass(frozen=True)
class PageText:
    """Text of one PDF page with its offset into the concatenated document.

    Attributes:
        page_number: 1-indexed page number as printed by the PDF itself.
        text: Layout-preserved text of the page.
        start: Character offset of this page within the joined document text.
        end: Character offset just past the end of this page.
    """

    page_number: int
    text: str
    start: int
    end: int


class PdfText:
    """Layout-preserved text of a voucher PDF, with page boundaries kept.

    The voucher parser needs three things at once: the text of each row,
    the page that row is printed on, and a character offset that lets the
    operator find it again. Extracting page by page and remembering the
    boundaries gives all three without a second pass or a proportional
    estimate -- offsets here are exact, not scaled.
    """

    def __init__(self, path: str) -> None:
        """Extract per-page text from a PDF.

        Args:
            path: Filesystem path to an existing PDF.

        Raises:
            ImportError: If pdfplumber is not installed.
        """
        import pdfplumber  # imported lazily: optional at import time

        self.path = path
        self.pages: list[PageText] = []
        chunks: list[str] = []
        total = 0
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                # layout=True preserves the column spacing the row regex
                # keys on. Without it the vendor name and the date collapse
                # into a single space and the columns become ambiguous.
                page_text = page.extract_text(layout=True) or ""
                start = total
                total += len(page_text)
                self.pages.append(PageText(index, page_text, start, total))
                chunks.append(page_text)
        self.text = "".join(chunks)

    @property
    def page_count(self) -> int:
        """Number of pages read from the PDF."""
        return len(self.pages)

    @property
    def has_text_layer(self) -> bool:
        """Whether the PDF carries extractable text rather than only images."""
        return any(p.text.strip() for p in self.pages)

    def page_for_offset(self, offset: int) -> int | None:
        """Return the page number containing a character offset.

        Args:
            offset: Character offset into ``self.text``.

        Returns:
            1-indexed page number, or None when the PDF had no pages.
        """
        if not self.pages:
            return None
        for page in self.pages:
            if offset < page.end:
                return page.page_number
        return self.pages[-1].page_number


def make_quote(text: str, start: int, end: int, max_len: int = 300) -> str:
    """Build a normalized verbatim quote for a locator.

    Args:
        text: Full document text.
        start: Start character offset of the span.
        end: End character offset of the span.
        max_len: Maximum quote length in characters.

    Returns:
        Whitespace-normalized excerpt of the source text.
    """
    snippet = text[start : min(end, start + max_len)]
    return re.sub(r"\s+", " ", snippet).strip()
