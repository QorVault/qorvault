"""Locator resolution for voucher fact rows.

Every voucher fact row must carry the source file, a page number, a
character offset and a verbatim quote. Chunk ids are never a locator, and a
``document_id`` is never guessed: where ingest has not produced one, the row
is anchored by file path and SHA-256 instead and relinked later.

Phase 0 established that Postgres holds no page-level text: ``document_pages``
is empty and ``chunks.source_page`` is NULL corpus wide. Page numbers
therefore come from re-reading the source PDF with pdfplumber, and so does
the text the parser runs on -- the voucher row regex depends on column
geometry that ``documents.content_text`` does not preserve.

Path resolution lives in ``facts/common/paths.py`` because more than one
fact package needs the same answer to "where does this file actually live".
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
from dataclasses import dataclass

# facts/ is the parent of this package; facts/common is shared code.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import (  # noqa: E402  - path bootstrap must precede the import
    CORPUS_ROOTS,
    MEETING_SLUG_RX,
    PATH_REWRITES,
    STAGED_ROOTS,
    STAGING_ROOT,
    is_staged,
    meeting_date_from_path,
    resolve_any_path,
    resolve_pdf_path,
    source_kind,
)

__all__ = [
    "CORPUS_ROOTS",
    "MEETING_SLUG_RX",
    "PATH_REWRITES",
    "STAGED_ROOTS",
    "STAGING_ROOT",
    "PageText",
    "PdfText",
    "is_staged",
    "make_quote",
    "meeting_date_from_path",
    "resolve_any_path",
    "resolve_pdf_path",
    "sha256_of",
    "source_kind",
]

LOG = logging.getLogger(__name__)


def sha256_of(path: str) -> str | None:
    """Return the SHA-256 digest of a file, or None if unreadable.

    The digest is the durable half of a locator. A path can change when a
    directory is renamed -- which has already happened once in this corpus --
    but the digest identifies the exact bytes a quote was read from, and it
    is the key the relink step uses to attach a document id later.

    Args:
        path: Filesystem path.

    Returns:
        Hex digest, or None when the file cannot be read.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
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

    The voucher parser needs three things at once: the text of each row, the
    page that row is printed on, and a character offset that lets the
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
        return any(page.text.strip() for page in self.pages)

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
