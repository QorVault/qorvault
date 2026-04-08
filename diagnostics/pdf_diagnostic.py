#!/usr/bin/env python3
"""PDF corpus diagnostic for BoardDocs RAG pipeline.

Analyzes unprocessed PDF attachment documents to classify them as
native digital, mixed-content, fully scanned, empty, or corrupt.
Produces a JSON report and a human-readable summary to inform
OCR pipeline design decisions.

Usage:
    python3 pdf_diagnostic.py                  # full corpus
    python3 pdf_diagnostic.py --sample 400     # random 400 files
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from pathlib import Path as _Path

import fitz  # pymupdf
import psycopg2

# ---------------------------------------------------------------------------
# Constants and thresholds
# ---------------------------------------------------------------------------
# Database connection — built from env vars, 127.0.0.1 (never localhost)
# because Podman containers only bind IPv4.
from dotenv import load_dotenv as _load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

_load_dotenv(_Path(__file__).resolve().parent.parent / ".env")


def _default_dsn() -> str:
    """Build PostgreSQL URL from POSTGRES_* env vars. No hardcoded credentials."""
    _host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    _port = os.environ.get("POSTGRES_PORT", "5432")
    _db = os.environ.get("POSTGRES_DB", "qorvault")
    _user = os.environ.get("POSTGRES_USER", "qorvault")
    _pw = os.environ.get("POSTGRES_PASSWORD")
    if not _pw:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set. " "Copy .env.example to .env and fill in credentials."
        )
    return f"postgresql://{_user}:{_pw}@{_host}:{_port}/{_db}"


DEFAULT_DSN = _default_dsn()

# Page classification thresholds.
# TEXT_RICH: extractable text > 100 chars, density > 0.001 chars/pt²,
# and at least 3 word-like tokens (3+ chars). This eliminates pages
# with only page numbers, headers, or watermarks.
#
# Density calibration: a US Letter page is 612×792 = 484,704 pt².
# A typical text page has ~2000 chars → density ~0.004.
# A sparse page (short memo) has ~300 chars → density ~0.0006.
# Threshold 0.001 catches pages with at least ~500 chars worth of
# text density, filtering out near-blank pages while accepting
# any page with meaningful text content.
TEXT_RICH_MIN_CHARS = 100
TEXT_RICH_MIN_DENSITY = 0.001
TEXT_RICH_MIN_WORD_TOKENS = 3

# TEXT_SPARSE: some text but not enough to be TEXT_RICH.
# Captures pages with minimal embedded text — possible OCR overlays
# or pages with only figures and captions.
TEXT_SPARSE_MIN_CHARS = 10

# IMAGE_ONLY: < 10 chars — practically no text.
IMAGE_ONLY_MAX_CHARS = 10

# OCR overlay detection: if fewer than 40% of tokens are recognizable
# English words, the text may be garbled OCR output rather than real text.
OCR_OVERLAY_WORD_RATIO_THRESHOLD = 0.40
OCR_OVERLAY_MIN_TOKENS = 5

# Document-level classification thresholds (fraction of TEXT_RICH pages).
NATIVE_DIGITAL_THRESHOLD = 0.90  # 90%+ text-rich → native digital
MIXED_CONTENT_THRESHOLD = 0.10  # 10-90% text-rich → mixed
# Below 10% → fully scanned

# Processing speed benchmarks (pages per minute).
NATIVE_PAGES_PER_MIN = 200.0  # pymupdf text extraction
SCANNED_PAGES_PER_MIN = 3.0  # Surya OCR on a typical local GPU

# Scanner-origin keywords in PDF creator/producer metadata.
SCANNER_KEYWORDS = [
    "scan",
    "fujitsu",
    "canon",
    "epson",
    "xerox",
    "ricoh",
    "konica",
    "minolta",
    "sharp",
    "kyocera",
    "brother",
    "hp scanjet",
    "twain",
    "paperstream",
    "scansnap",
    "naps2",
    "wia",
    "iscan",
]

# Report output directory.
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# ---------------------------------------------------------------------------
# Common English words for OCR overlay detection.
# Top ~200 most frequent English words — enough to detect garbled OCR
# without requiring nltk or external word lists.
# ---------------------------------------------------------------------------

COMMON_ENGLISH_WORDS = frozenset(
    {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "are",
        "is",
        "was",
        "were",
        "been",
        "has",
        "had",
        "did",
        "does",
        "may",
        "might",
        "shall",
        "should",
        "must",
        "need",
        "each",
        "every",
        "both",
        "few",
        "more",
        "many",
        "much",
        "own",
        "same",
        "such",
        "very",
        "being",
        "here",
        "where",
        "between",
        "under",
        "again",
        "once",
        "during",
        "before",
        "through",
        "too",
        "down",
        "off",
        "above",
        "below",
        "while",
        "until",
        # Common school-district domain words:
        "board",
        "district",
        "school",
        "student",
        "students",
        "policy",
        "meeting",
        "motion",
        "vote",
        "approved",
        "superintendent",
        "education",
        "program",
        "budget",
        "report",
        "public",
        "action",
        "item",
        "resolution",
        "contract",
        "personnel",
        "fund",
        "services",
        "information",
        "number",
        "date",
        "page",
        "total",
        "amount",
        "name",
        "address",
        "state",
        "city",
        "county",
        "washington",
        "kent",
        "federal",
        "section",
        "chapter",
        "title",
        "order",
        "financial",
        "fiscal",
        "building",
        "staff",
        "teacher",
        "principal",
        "department",
        "office",
        "community",
        "family",
        "support",
        "learning",
        "instruction",
        "assessment",
        "grade",
        "class",
        "high",
        "middle",
        "elementary",
        "special",
        "general",
    }
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class PageType(str, Enum):
    TEXT_RICH = "TEXT_RICH"
    TEXT_SPARSE = "TEXT_SPARSE"
    IMAGE_ONLY = "IMAGE_ONLY"
    UNREADABLE = "UNREADABLE"


class DocClassification(str, Enum):
    NATIVE_DIGITAL = "NATIVE_DIGITAL"
    MIXED_CONTENT = "MIXED_CONTENT"
    FULLY_SCANNED = "FULLY_SCANNED"
    EMPTY = "EMPTY"
    CORRUPT = "CORRUPT"
    MISSING_FILE = "MISSING_FILE"


@dataclass
class PageAnalysis:
    page_number: int
    page_type: str
    char_count: int
    char_density: float
    word_token_count: int
    english_word_ratio: float | None = None
    possible_ocr_overlay: bool = False
    error: str | None = None


@dataclass
class DocAnalysis:
    document_id: str
    file_path: str
    classification: str
    page_count: int = 0
    text_rich_pages: int = 0
    text_sparse_pages: int = 0
    image_only_pages: int = 0
    unreadable_pages: int = 0
    possible_ocr_overlay_pages: int = 0
    scanner_origin: bool = False
    pdf_creator: str | None = None
    pdf_producer: str | None = None
    pdf_creation_date: str | None = None
    file_size_bytes: int = 0
    estimated_ocr_minutes: float = 0.0
    error: str | None = None
    pages: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def classify_page(page: fitz.Page, page_num: int) -> PageAnalysis:
    """Classify a single PDF page by its text content characteristics."""
    try:
        text = page.get_text("text")
    except Exception as exc:
        return PageAnalysis(
            page_number=page_num,
            page_type=PageType.UNREADABLE.value,
            char_count=0,
            char_density=0.0,
            word_token_count=0,
            error=str(exc),
        )

    char_count = len(text.strip())

    # Calculate character density relative to page area (in points²).
    rect = page.rect
    page_area = rect.width * rect.height
    char_density = char_count / page_area if page_area > 0 else 0.0

    # Count word-like tokens (3+ characters, split on whitespace).
    tokens = text.split()
    word_tokens = [t for t in tokens if len(t) >= 3]
    word_token_count = len(word_tokens)

    # Classify the page.
    if (
        char_count > TEXT_RICH_MIN_CHARS
        and char_density > TEXT_RICH_MIN_DENSITY
        and word_token_count >= TEXT_RICH_MIN_WORD_TOKENS
    ):
        page_type = PageType.TEXT_RICH.value
    elif char_count >= TEXT_SPARSE_MIN_CHARS:
        page_type = PageType.TEXT_SPARSE.value
    elif char_count < IMAGE_ONLY_MAX_CHARS:
        page_type = PageType.IMAGE_ONLY.value
    else:
        # Between IMAGE_ONLY_MAX_CHARS and TEXT_SPARSE_MIN_CHARS shouldn't
        # happen with current thresholds (both are 10), but handle edge case.
        page_type = PageType.TEXT_SPARSE.value

    # OCR overlay detection for TEXT_SPARSE pages.
    # Check whether extracted text contains recognizable English words.
    english_word_ratio = None
    possible_ocr_overlay = False

    if page_type == PageType.TEXT_SPARSE.value and len(tokens) >= OCR_OVERLAY_MIN_TOKENS:
        lower_tokens = [t.lower().strip(".,;:!?()[]{}\"'") for t in tokens]
        matches = sum(1 for t in lower_tokens if t in COMMON_ENGLISH_WORDS)
        english_word_ratio = matches / len(tokens) if tokens else 0.0
        if english_word_ratio < OCR_OVERLAY_WORD_RATIO_THRESHOLD:
            possible_ocr_overlay = True

    return PageAnalysis(
        page_number=page_num,
        page_type=page_type,
        char_count=char_count,
        char_density=round(char_density, 6),
        word_token_count=word_token_count,
        english_word_ratio=round(english_word_ratio, 4) if english_word_ratio is not None else None,
        possible_ocr_overlay=possible_ocr_overlay,
    )


def classify_document(page_analyses: list[PageAnalysis]) -> str:
    """Classify a whole document based on its page composition."""
    total = len(page_analyses)
    if total == 0:
        return DocClassification.EMPTY.value

    text_rich = sum(1 for p in page_analyses if p.page_type == PageType.TEXT_RICH.value)
    unreadable = sum(1 for p in page_analyses if p.page_type == PageType.UNREADABLE.value)

    if unreadable == total:
        return DocClassification.EMPTY.value

    ratio = text_rich / total
    if ratio >= NATIVE_DIGITAL_THRESHOLD:
        return DocClassification.NATIVE_DIGITAL.value
    elif ratio >= MIXED_CONTENT_THRESHOLD:
        return DocClassification.MIXED_CONTENT.value
    else:
        return DocClassification.FULLY_SCANNED.value


def is_scanner_origin(creator: str | None, producer: str | None) -> bool:
    """Check if PDF metadata suggests a scanner origin."""
    combined = ((creator or "") + " " + (producer or "")).lower()
    return any(kw in combined for kw in SCANNER_KEYWORDS)


def estimate_ocr_minutes(page_analyses: list[PageAnalysis]) -> float:
    """Estimate OCR processing time based on page classifications."""
    native_pages = sum(1 for p in page_analyses if p.page_type == PageType.TEXT_RICH.value)
    ocr_pages = sum(1 for p in page_analyses if p.page_type in (PageType.IMAGE_ONLY.value, PageType.TEXT_SPARSE.value))
    # Unreadable pages are skipped (zero time).

    native_time = native_pages / NATIVE_PAGES_PER_MIN if NATIVE_PAGES_PER_MIN > 0 else 0
    ocr_time = ocr_pages / SCANNED_PAGES_PER_MIN if SCANNED_PAGES_PER_MIN > 0 else 0

    return round(native_time + ocr_time, 3)


def analyze_pdf(document_id: str, file_path: str) -> DocAnalysis:
    """Analyze a single PDF file and return its classification."""
    # Check if file exists.
    if not os.path.isfile(file_path):
        return DocAnalysis(
            document_id=document_id,
            file_path=file_path,
            classification=DocClassification.MISSING_FILE.value,
            error="File not found on disk",
        )

    file_size = os.path.getsize(file_path)

    # Try to open the PDF.
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        return DocAnalysis(
            document_id=document_id,
            file_path=file_path,
            classification=DocClassification.CORRUPT.value,
            file_size_bytes=file_size,
            error=str(exc),
        )

    try:
        # Extract PDF metadata.
        metadata = doc.metadata or {}
        pdf_creator = metadata.get("creator") or None
        pdf_producer = metadata.get("producer") or None
        pdf_creation_date = metadata.get("creationDate") or None
        scanner = is_scanner_origin(pdf_creator, pdf_producer)

        # Analyze each page.
        page_analyses = []
        for i in range(len(doc)):
            try:
                page = doc[i]
                pa = classify_page(page, i)
            except Exception as exc:
                pa = PageAnalysis(
                    page_number=i,
                    page_type=PageType.UNREADABLE.value,
                    char_count=0,
                    char_density=0.0,
                    word_token_count=0,
                    error=str(exc),
                )
            page_analyses.append(pa)

        # Classify the document as a whole.
        classification = classify_document(page_analyses)

        # Counts.
        text_rich = sum(1 for p in page_analyses if p.page_type == PageType.TEXT_RICH.value)
        text_sparse = sum(1 for p in page_analyses if p.page_type == PageType.TEXT_SPARSE.value)
        image_only = sum(1 for p in page_analyses if p.page_type == PageType.IMAGE_ONLY.value)
        unreadable = sum(1 for p in page_analyses if p.page_type == PageType.UNREADABLE.value)
        ocr_overlay = sum(1 for p in page_analyses if p.possible_ocr_overlay)

        est_minutes = estimate_ocr_minutes(page_analyses)

        return DocAnalysis(
            document_id=document_id,
            file_path=file_path,
            classification=classification,
            page_count=len(doc),
            text_rich_pages=text_rich,
            text_sparse_pages=text_sparse,
            image_only_pages=image_only,
            unreadable_pages=unreadable,
            possible_ocr_overlay_pages=ocr_overlay,
            scanner_origin=scanner,
            pdf_creator=pdf_creator,
            pdf_producer=pdf_producer,
            pdf_creation_date=pdf_creation_date,
            file_size_bytes=file_size,
            estimated_ocr_minutes=est_minutes,
            pages=[asdict(p) for p in page_analyses],
        )
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Database query
# ---------------------------------------------------------------------------


def fetch_documents(dsn: str, sample: int | None = None) -> list[tuple[str, str]]:
    """Fetch (document_id, file_path) pairs for unprocessed PDF attachments."""
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            # Get all attachment PDFs that aren't fully processed.
            # LOWER(file_path) LIKE '%.pdf' catches both .pdf and .PDF.
            cur.execute("""
                SELECT id::text, file_path
                FROM documents
                WHERE document_type = 'attachment'
                  AND processing_status != 'complete'
                  AND file_path IS NOT NULL
                  AND LOWER(file_path) LIKE '%%.pdf'
                ORDER BY meeting_date DESC NULLS LAST
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if sample is not None and sample < len(rows):
        rows = random.sample(rows, sample)

    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_summary(
    results: list[DocAnalysis],
    elapsed_seconds: float,
    sample_size: int | None,
    total_corpus_size: int,
) -> str:
    """Generate the human-readable summary report."""
    lines: list[str] = []
    w = lines.append

    w("=" * 72)
    w("  PDF CORPUS DIAGNOSTIC REPORT")
    w(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 72)
    w("")

    total_analyzed = len(results)
    is_sample = sample_size is not None

    if is_sample:
        w(f"  Mode: SAMPLE ({total_analyzed} of {total_corpus_size} total PDF attachments)")
        w("  Percentages below are estimates based on this sample.")
    else:
        w(f"  Mode: FULL CORPUS ({total_analyzed} PDF attachments)")
    w(f"  Analysis time: {elapsed_seconds:.1f} seconds")
    w("")

    # Classification breakdown.
    counts = Counter(r.classification for r in results)
    w("-" * 72)
    w("  DOCUMENT CLASSIFICATION BREAKDOWN")
    w("-" * 72)
    w("")
    w(f"  {'Classification':<20} {'Count':>8} {'Percent':>10}")
    w(f"  {'-'*20:<20} {'-'*8:>8} {'-'*10:>10}")
    for cls in [
        DocClassification.NATIVE_DIGITAL.value,
        DocClassification.MIXED_CONTENT.value,
        DocClassification.FULLY_SCANNED.value,
        DocClassification.EMPTY.value,
        DocClassification.CORRUPT.value,
        DocClassification.MISSING_FILE.value,
    ]:
        count = counts.get(cls, 0)
        pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0
        w(f"  {cls:<20} {count:>8} {pct:>9.1f}%")
    w(f"  {'-'*20:<20} {'-'*8:>8} {'-'*10:>10}")
    w(f"  {'TOTAL':<20} {total_analyzed:>8} {'100.0%':>10}")
    w("")

    # Extrapolation to full corpus.
    if is_sample and total_analyzed > 0:
        w("-" * 72)
        w("  EXTRAPOLATED FULL CORPUS ESTIMATES")
        w("-" * 72)
        w("")
        scale = total_corpus_size / total_analyzed
        for cls in [
            DocClassification.NATIVE_DIGITAL.value,
            DocClassification.MIXED_CONTENT.value,
            DocClassification.FULLY_SCANNED.value,
            DocClassification.EMPTY.value,
            DocClassification.CORRUPT.value,
            DocClassification.MISSING_FILE.value,
        ]:
            count = counts.get(cls, 0)
            estimated = round(count * scale)
            w(f"  {cls:<20} ~{estimated:>7} files")
        w("")

    # Page-level statistics.
    total_pages = sum(r.page_count for r in results)
    total_text_rich = sum(r.text_rich_pages for r in results)
    total_text_sparse = sum(r.text_sparse_pages for r in results)
    total_image_only = sum(r.image_only_pages for r in results)
    total_unreadable = sum(r.unreadable_pages for r in results)
    total_ocr_overlay = sum(r.possible_ocr_overlay_pages for r in results)

    w("-" * 72)
    w("  PAGE-LEVEL STATISTICS")
    w("-" * 72)
    w("")
    w(f"  Total pages analyzed:         {total_pages:>10}")
    w(f"  TEXT_RICH pages:              {total_text_rich:>10} ({total_text_rich/max(total_pages,1)*100:.1f}%)")
    w(f"  TEXT_SPARSE pages:            {total_text_sparse:>10} ({total_text_sparse/max(total_pages,1)*100:.1f}%)")
    w(f"  IMAGE_ONLY pages:             {total_image_only:>10} ({total_image_only/max(total_pages,1)*100:.1f}%)")
    w(f"  UNREADABLE pages:             {total_unreadable:>10} ({total_unreadable/max(total_pages,1)*100:.1f}%)")
    w(f"  Possible OCR overlay pages:   {total_ocr_overlay:>10}")
    w("")

    if total_analyzed > 0:
        avg_pages = total_pages / total_analyzed
        w(f"  Average pages per document:   {avg_pages:>10.1f}")
        w("")

    # File size statistics.
    sizes = [r.file_size_bytes for r in results if r.file_size_bytes > 0]
    if sizes:
        total_mb = sum(sizes) / (1024 * 1024)
        avg_mb = total_mb / len(sizes)
        max_mb = max(sizes) / (1024 * 1024)
        w(f"  Total file size (analyzed):   {total_mb:>10.1f} MB")
        w(f"  Average file size:            {avg_mb:>10.2f} MB")
        w(f"  Largest file:                 {max_mb:>10.2f} MB")
        w("")

    # Scanner origin analysis.
    scanner_count = sum(1 for r in results if r.scanner_origin)
    w("-" * 72)
    w("  SCANNER ORIGIN DETECTION")
    w("-" * 72)
    w("")
    w(f"  Documents with scanner-origin metadata: {scanner_count}")
    w(f"  Percentage: {scanner_count/max(total_analyzed,1)*100:.1f}%")
    w("")

    # Top PDF creator applications.
    creators = Counter()
    for r in results:
        creator = r.pdf_creator or "(none/empty)"
        creators[creator] += 1

    w("-" * 72)
    w("  TOP 10 PDF CREATOR APPLICATIONS")
    w("-" * 72)
    w("")
    for creator, count in creators.most_common(10):
        pct = count / total_analyzed * 100 if total_analyzed > 0 else 0
        # Truncate long creator strings.
        display = creator[:50] + "..." if len(creator) > 50 else creator
        w(f"  {display:<55} {count:>5} ({pct:.1f}%)")
    w("")

    # OCR processing time estimates.
    sample_ocr_minutes = sum(r.estimated_ocr_minutes for r in results)

    w("-" * 72)
    w("  ESTIMATED OCR PROCESSING TIME")
    w("-" * 72)
    w("")
    w(f"  Analyzed documents: {sample_ocr_minutes:.1f} minutes ({sample_ocr_minutes/60:.1f} hours)")

    if is_sample and total_analyzed > 0:
        scale = total_corpus_size / total_analyzed
        est_total_minutes = sample_ocr_minutes * scale
        # Pessimistic: 1.5x (retries, overhead, I/O waits).
        # Optimistic: 0.8x (some parallelism possible).
        optimistic = est_total_minutes * 0.8
        pessimistic = est_total_minutes * 1.5
        w("  Extrapolated full corpus:")
        w(f"    Optimistic:  {optimistic:.0f} minutes ({optimistic/60:.1f} hours)")
        w(f"    Expected:    {est_total_minutes:.0f} minutes ({est_total_minutes/60:.1f} hours)")
        w(f"    Pessimistic: {pessimistic:.0f} minutes ({pessimistic/60:.1f} hours)")
    else:
        optimistic = sample_ocr_minutes * 0.8
        pessimistic = sample_ocr_minutes * 1.5
        w(f"    Optimistic:  {optimistic:.0f} minutes ({optimistic/60:.1f} hours)")
        w(f"    Expected:    {sample_ocr_minutes:.0f} minutes ({sample_ocr_minutes/60:.1f} hours)")
        w(f"    Pessimistic: {pessimistic:.0f} minutes ({pessimistic/60:.1f} hours)")
    w("")

    # Missing / corrupt file count.
    missing = counts.get(DocClassification.MISSING_FILE.value, 0)
    corrupt = counts.get(DocClassification.CORRUPT.value, 0)
    w("-" * 72)
    w("  FILE INTEGRITY")
    w("-" * 72)
    w("")
    w(f"  Missing files (not on disk): {missing}")
    w(f"  Corrupt files (unreadable):  {corrupt}")
    w("")

    # Recommendations.
    native_pct = counts.get(DocClassification.NATIVE_DIGITAL.value, 0) / max(total_analyzed, 1) * 100
    scanned_pct = counts.get(DocClassification.FULLY_SCANNED.value, 0) / max(total_analyzed, 1) * 100
    mixed_pct = counts.get(DocClassification.MIXED_CONTENT.value, 0) / max(total_analyzed, 1) * 100
    ocr_overlay_pct = total_ocr_overlay / max(total_pages, 1) * 100

    w("-" * 72)
    w("  RECOMMENDATIONS")
    w("-" * 72)
    w("")

    if native_pct > 60:
        w(f"  1. MAJORITY NATIVE DIGITAL: Over {native_pct:.0f}% of documents are native")
        w("     digital PDFs. The OCR pipeline should prioritize fast text")
        w("     extraction via pymupdf for the majority of documents, with")
        w("     OCR as a fallback for the scanned minority.")
    elif scanned_pct > 60:
        w(f"  1. MAJORITY SCANNED: Over {scanned_pct:.0f}% of documents are scanned")
        w("     images. GPU OCR capacity is critical. Budget for substantial")
        w("     processing time and prioritize GPU-accelerated OCR (Surya).")
    else:
        w("  1. MIXED CORPUS: The corpus contains a significant mix of native")
        w("     digital and scanned documents. The pipeline needs a two-tier")
        w("     approach: fast pymupdf extraction for digital pages, GPU OCR")
        w("     for scanned pages.")

    w("")

    if scanned_pct + mixed_pct > 20:
        w(f"  2. GPU OCR CAPACITY: {scanned_pct + mixed_pct:.0f}% of documents need OCR processing.")
        w("     At ~3 pages/minute on a typical local GPU, budget")
        if is_sample:
            pages_needing_ocr = round((total_image_only + total_text_sparse) * (total_corpus_size / total_analyzed))
            est_hours = pages_needing_ocr / SCANNED_PAGES_PER_MIN / 60
            w(f"     ~{est_hours:.0f} GPU-hours for the full corpus ({pages_needing_ocr:,} pages).")
        else:
            est_hours = (total_image_only + total_text_sparse) / SCANNED_PAGES_PER_MIN / 60
            w(f"     ~{est_hours:.0f} GPU-hours for the corpus ({total_image_only + total_text_sparse:,} pages).")
    else:
        w(f"  2. GPU OCR CAPACITY: Only {scanned_pct + mixed_pct:.0f}% of documents need OCR.")
        w("     GPU OCR capacity is not a bottleneck for this corpus.")

    w("")

    if ocr_overlay_pct > 2:
        w(f"  3. OCR OVERLAY DETECTION: {ocr_overlay_pct:.1f}% of pages show signs of OCR")
        w("     overlay (garbled text). The pipeline should detect these and")
        w("     re-OCR from the image layer rather than trusting the embedded text.")
    else:
        w(f"  3. OCR OVERLAY: Minimal OCR overlay detected ({ocr_overlay_pct:.1f}% of pages).")
        w("     No special handling needed in the pipeline.")

    w("")

    if scanner_count > total_analyzed * 0.05:
        w(
            f"  4. SCANNER METADATA: {scanner_count / max(total_analyzed, 1) * 100:.0f}% of documents have scanner-origin"
        )
        w("     metadata. This confirms a significant portion of the corpus")
        w("     was digitized from paper originals.")
    else:
        w(f"  4. SCANNER METADATA: Few documents ({scanner_count}) have scanner-origin")
        w("     metadata. Most PDFs were likely created digitally.")

    w("")

    if missing + corrupt > total_analyzed * 0.05:
        w(
            f"  5. FILE INTEGRITY: {(missing + corrupt) / max(total_analyzed, 1) * 100:.0f}% of files are missing or corrupt."
        )
        w("     Investigate and potentially re-scrape these documents.")
    else:
        w(f"  5. FILE INTEGRITY: Good — only {missing + corrupt} missing/corrupt files")
        w(f"     ({(missing + corrupt) / max(total_analyzed, 1) * 100:.1f}% of analyzed).")

    w("")
    w("=" * 72)
    w("  END OF REPORT")
    w("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Analyze unprocessed PDF attachments for OCR pipeline planning.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N documents instead of processing all.",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=DEFAULT_DSN,
        help="PostgreSQL DSN (default: project standard).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    console = Console()

    console.print("[bold]PDF Corpus Diagnostic[/bold]")
    console.print("Connecting to database...")

    # Fetch document list.
    try:
        all_docs = fetch_documents(args.dsn, sample=None)  # fetch all first for count
    except Exception as exc:
        console.print(f"[red]Database error: {exc}[/red]")
        sys.exit(1)

    total_corpus_size = len(all_docs)
    console.print(f"Found {total_corpus_size} unprocessed PDF attachments in database.")

    # Apply sampling if requested.
    if args.sample is not None and args.sample < total_corpus_size:
        docs = random.sample(all_docs, args.sample)
        console.print(f"Sampling {args.sample} documents (seed={args.seed}).")
    else:
        docs = all_docs

    if not docs:
        console.print("[yellow]No documents to analyze.[/yellow]")
        sys.exit(0)

    # Analyze each PDF with progress bar.
    results: list[DocAnalysis] = []
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing PDFs", total=len(docs))

        for doc_id, file_path in docs:
            # Show current file in progress description.
            short_name = Path(file_path).name if file_path else "(no path)"
            if len(short_name) > 40:
                short_name = "..." + short_name[-37:]
            progress.update(task, description=f"[cyan]{short_name}[/cyan]")

            result = analyze_pdf(doc_id, file_path)
            results.append(result)
            progress.advance(task)

    elapsed = time.time() - t0

    # Generate reports.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report (full detail, but omit per-page data for brevity in
    # sample runs — include page-level data only in full runs).
    json_path = REPORTS_DIR / f"pdf_analysis_{timestamp}.json"
    json_results = []
    for r in results:
        d = asdict(r)
        # Omit page-level detail to keep JSON manageable.
        # The per-page data is still in the DocAnalysis objects in memory.
        d.pop("pages", None)
        json_results.append(d)

    report_data = {
        "generated_at": datetime.now().isoformat(),
        "mode": f"sample_{args.sample}" if args.sample else "full",
        "total_corpus_pdf_attachments": total_corpus_size,
        "documents_analyzed": len(results),
        "analysis_time_seconds": round(elapsed, 1),
        "random_seed": args.seed if args.sample else None,
        "documents": json_results,
    }

    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    # Human-readable summary.
    summary = generate_summary(results, elapsed, args.sample, total_corpus_size)
    txt_path = REPORTS_DIR / f"pdf_analysis_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(summary)

    # Also print summary to console.
    console.print()
    console.print(summary)
    console.print()
    console.print("[bold green]Reports saved:[/bold green]")
    console.print(f"  JSON: {json_path}")
    console.print(f"  Text: {txt_path}")


if __name__ == "__main__":
    main()
