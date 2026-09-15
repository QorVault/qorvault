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
class PlacedRow:
    """One printed row, with both its geometry and its place in the text.

    Attributes:
        row: The row's characters and their x coordinates.
        line: The row as ``extract_text(layout=True)`` rendered it.
        char_offset: Offset of ``line`` within the joined document text.
    """

    row: object
    line: str
    char_offset: int


@dataclass(frozen=True)
class PageText:
    """Text of one PDF page with its offset into the concatenated document.

    Attributes:
        page_number: 1-indexed page number as printed by the PDF itself.
        text: Layout-preserved text of the page.
        start: Character offset of this page within the joined document text.
        end: Character offset just past the end of this page.
        rows: Printed rows paired with their layout lines and offsets.
        aligned: Whether every geometry row paired with a layout line. False
            means the two views of the page disagreed and the rows on it
            cannot be trusted to carry the right offsets.
        header_tops: ``top`` of each row belonging to a column header.
    """

    page_number: int
    text: str
    start: int
    end: int
    rows: tuple[PlacedRow, ...] = ()
    aligned: bool = True
    header_tops: frozenset[float] = frozenset()


class PdfText:
    """Layout-preserved text of a voucher PDF, with page boundaries kept.

    The voucher parser needs three things at once: the text of each row, the
    page that row is printed on, and a character offset that lets the
    operator find it again. Extracting page by page and remembering the
    boundaries gives all three without a second pass or a proportional
    estimate -- offsets here are exact, not scaled.
    """

    def __init__(self, path: str, geometry_too: bool = True) -> None:
        """Extract per-page text, and optionally geometry, from a PDF.

        The two views of a page are kept together on purpose. Column values
        come from the geometry; the character offset, the quote, the TOTAL
        line and the era header all come from the layout text, unchanged
        from before this module grew coordinates. Pairing them by order is
        sound because ``extract_text(layout=True)`` renders one text line
        per printed row, in printed order -- verified on 244 pages across
        all four format eras with zero mismatches -- and every page where
        the two disagree is flagged rather than guessed at.

        Args:
            path: Filesystem path to an existing PDF.
            geometry_too: Read character coordinates as well as text.

        Raises:
            ImportError: If pdfplumber is not installed.
        """
        import pdfplumber  # imported lazily: optional at import time

        self.path = path
        self.pages: list[PageText] = []
        self.grid = None
        self.grid_note: str | None = None
        self.grid_drift: list[str] = []
        self.misaligned_pages: list[int] = []
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
                rows, aligned, tops = self._place(page, page_text, start) if geometry_too else ((), True, frozenset())
                if not aligned:
                    self.misaligned_pages.append(index)
                self.pages.append(PageText(index, page_text, start, total, rows, aligned, tops))
                chunks.append(page_text)
        self.text = "".join(chunks)
        if geometry_too:
            self._build_grid()

    @staticmethod
    def _place(page, page_text: str, start: int) -> tuple[tuple[PlacedRow, ...], bool, frozenset[float]]:
        """Pair a page's printed rows with its layout text lines.

        Args:
            page: A ``pdfplumber`` page.
            page_text: The page's layout-preserved text.
            start: Offset of the page within the joined document text.

        Returns:
            ``(rows, aligned, header_tops)``.
        """
        import geometry as geo

        rows = geo.page_rows(page)
        found = geo.find_header(rows)
        tops = frozenset({found[2]} if found else set())

        offsets: list[tuple[str, int]] = []
        cursor = start
        for line in page_text.split("\n"):
            if line.strip():
                offsets.append((line, cursor))
            cursor += len(line) + 1

        solid = [row for row in rows if row.text.strip()]
        if len(solid) != len(offsets):
            return (
                tuple(PlacedRow(row, row.text, start) for row in solid),
                False,
                tops,
            )
        placed = tuple(PlacedRow(row, line, offset) for row, (line, offset) in zip(solid, offsets, strict=True))
        return placed, True, tops

    def _build_grid(self) -> None:
        """Derive the document's column grid and check every printed header."""
        import geometry as geo

        first: tuple[str, list, int] | None = None
        tops: set[float] = set()
        for page in self.pages:
            if not page.header_tops:
                continue
            tops |= set(page.header_tops)
            header = geo.find_header([placed.row for placed in page.rows])
            if header is None:
                continue
            schema, labels, _ = header
            if first is None:
                first = (schema.name, labels, page.page_number)
            else:
                self.grid_drift.extend(_drift(first[1], labels, page.page_number))
        if first is None:
            self.grid_note = "no header matched any known listing format"
            return
        all_rows = [placed.row for page in self.pages for placed in page.rows]
        self.grid, self.grid_note = geo.build_grid(first[0], first[1], all_rows, tops, first[2])

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


def _drift(reference: list, other: list, page: int) -> list[str]:
    """Report header labels that moved between two pages of one document.

    Args:
        reference: Labels from the document's first header.
        other: Labels from a later page's header.
        page: The later page's number.

    Returns:
        One message per label that moved more than the tolerance allows.
    """
    import geometry as geo

    out: list[str] = []
    by_key = {label.key: label for label in other}
    for label in reference:
        twin = by_key.get(label.key)
        if twin is None:
            out.append(f"page {page}: column {label.key!r} is absent from this page's header")
            continue
        for edge in ("x0", "x1"):
            delta = abs(getattr(label, edge) - getattr(twin, edge))
            if delta > geo.DRIFT_TOLERANCE:
                out.append(f"page {page}: {label.key}.{edge} moved {delta:.2f} pt")
    return out


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
