"""Locator resolution for minutes fact rows.

Every fact row must carry the source ``document_id``, a page number and a
verbatim quote or character offset. Chunk ids are never a locator.

Phase 0 established that Postgres holds no page-level text: ``document_pages``
is empty and ``chunks.source_page`` is NULL for all 179,081 chunks. Page
numbers therefore have to come from the document text itself, or from
re-reading the source PDF.

Two page maps are provided:

``FooterPageMap``
    Derives page boundaries from the running footer the minutes carry
    ("Regular Board Meeting / Page 2 / 08 June 2011"). Available today, but
    only about 45% of minutes documents print such a footer.

``PdfPageMap``
    Reads true page boundaries from the source PDF with pdfplumber. Preferred,
    and used automatically when pdfplumber is importable and the PDF resolves.

Both expose the same ``page_for_offset`` interface so the parser does not care
which one it was given.
"""

from __future__ import annotations

import logging
import os
import re

LOG = logging.getLogger(__name__)

# documents.file_path values are stale: they are rooted at a path that no
# longer exists. The complete 2005-2026 corpus lives under the archive root.
STALE_PREFIX = "/home/donald/"
ARCHIVE_PREFIX = "/home/donald/qorvault-dev-archive/framework-backup/home/"

# "Page 2", "Page 2 of 7", optionally preceded by a meeting-name line.
FOOTER_RX = re.compile(r"\bPage\s+(\d{1,3})\b(?:\s+of\s+\d{1,3})?", re.I)


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
    if file_path.startswith(STALE_PREFIX):
        candidate = file_path.replace(STALE_PREFIX, ARCHIVE_PREFIX, 1)
        if os.path.isfile(candidate):
            return candidate
    return None


class PageMap:
    """Base page map: a single page covering the whole document."""

    kind = "none"

    def __init__(self, text: str) -> None:
        """Store the document text.

        Args:
            text: Full extracted text of the document.
        """
        self._text = text

    def page_for_offset(self, offset: int) -> int | None:
        """Return the page number containing a character offset.

        Args:
            offset: Character offset into the document text.

        Returns:
            1-indexed page number, or None when unknown.
        """
        return None


class FooterPageMap(PageMap):
    """Page map derived from printed "Page N" running footers."""

    kind = "footer"

    def __init__(self, text: str) -> None:
        """Build page boundaries from footer markers.

        Args:
            text: Full extracted text of the document.
        """
        super().__init__(text)
        # Each footer marks the END of the page whose number it prints, so
        # text before the first marker is page 1, between marker N and N+1 is
        # page N+1, and so on.
        self._marks: list[tuple[int, int]] = []
        for m in FOOTER_RX.finditer(text):
            try:
                self._marks.append((m.start(), int(m.group(1))))
            except ValueError:
                continue

    @property
    def usable(self) -> bool:
        """Whether enough footers were found to place offsets."""
        return len(self._marks) >= 1

    def page_for_offset(self, offset: int) -> int | None:
        """Return the page number containing a character offset.

        Args:
            offset: Character offset into the document text.

        Returns:
            1-indexed page number, or None when no footers were found.
        """
        if not self._marks:
            return None
        page = 1
        for pos, num in self._marks:
            if offset <= pos:
                return page
            page = num + 1
        return page


class PdfPageMap(PageMap):
    """Page map read from the source PDF via pdfplumber.

    The PDF text will not be byte-identical to ``documents.content_text``
    (which came through a different extractor), so offsets are mapped by
    proportional position across the concatenated per-page text. That is exact
    at page granularity for well-behaved documents and degrades gracefully.
    """

    kind = "pdf"

    def __init__(self, text: str, pdf_path: str) -> None:
        """Extract per-page text and build cumulative boundaries.

        Args:
            text: Full extracted text of the document (``content_text``).
            pdf_path: Resolved path to the source PDF.

        Raises:
            ImportError: If pdfplumber is not installed.
        """
        super().__init__(text)
        import pdfplumber  # imported lazily: optional dependency

        self._bounds: list[int] = []
        total = 0
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True) or ""
                total += len(page_text)
                self._bounds.append(total)
        self._pdf_len = total or 1
        self._txt_len = len(text) or 1

    @property
    def usable(self) -> bool:
        """Whether any pages were read."""
        return bool(self._bounds)

    def page_for_offset(self, offset: int) -> int | None:
        """Return the page number containing a character offset.

        Args:
            offset: Character offset into ``content_text``.

        Returns:
            1-indexed page number, or None when the PDF had no pages.
        """
        if not self._bounds:
            return None
        scaled = offset * self._pdf_len / self._txt_len
        for i, edge in enumerate(self._bounds):
            if scaled <= edge:
                return i + 1
        return len(self._bounds)


class SinglePageMap(PageMap):
    """Page map for a document that has exactly one page.

    BoardDocs agenda items are single web pages, not paginated PDFs. Page 1 is
    the truthful page number for such a document, not a guess: there is nowhere
    else in it for a quote to be.
    """

    kind = "single_page"

    def page_for_offset(self, offset: int) -> int | None:
        """Return page 1 for any offset.

        Args:
            offset: Character offset into the document text.

        Returns:
            Always 1.
        """
        return 1


HTML_SUFFIXES = (".html", ".htm", ".xhtml")


def is_single_page_source(file_path: str | None) -> bool:
    """Whether a document is a single web page rather than a paginated file.

    BoardDocs agenda items are scraped web pages. Their stored ``file_path``
    points at the scrape DIRECTORY rather than a ``.html`` file, so an
    extension check alone misses them.

    Args:
        file_path: Value of ``documents.file_path``.

    Returns:
        True when the document is a single web page.
    """
    if not file_path:
        return False
    if file_path.lower().endswith(HTML_SUFFIXES):
        return True
    for candidate in (file_path, file_path.replace(STALE_PREFIX, ARCHIVE_PREFIX, 1)):
        if os.path.isdir(candidate):
            return True
    return False


def build_page_map(text: str, file_path: str | None) -> PageMap:
    """Choose the best available page map for a document.

    Prefers true PDF pages, falls back to printed footers, then to no page
    information at all. A missing page number is recorded as NULL rather than
    guessed.

    Args:
        text: Full extracted text of the document.
        file_path: Value of ``documents.file_path``.

    Returns:
        A page map instance.
    """
    pdf_path = resolve_pdf_path(file_path)
    if pdf_path:
        try:
            pdf_map = PdfPageMap(text, pdf_path)
            if pdf_map.usable:
                return pdf_map
        except Exception as exc:
            # pdfplumber missing, or a malformed PDF. Falling back to footers
            # is correct, but silence would hide a corpus-wide regression (for
            # example the dependency going missing), so record why.
            LOG.debug("PDF page map unavailable for %s: %s", pdf_path, exc)
    footer_map = FooterPageMap(text)
    if footer_map.usable:
        return footer_map
    # A single-page web document: page 1 is a fact about it, not an assumption.
    if is_single_page_source(file_path):
        return SinglePageMap(text)
    return PageMap(text)


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
