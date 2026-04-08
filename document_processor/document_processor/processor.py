"""Main processing loop: fetch pending docs, extract text, chunk, store."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import asyncpg
from bs4 import BeautifulSoup

from .chunker import chunk_text
from .libreoffice_converter import (
    CONVERTIBLE_EXTENSIONS,
    convert_to_pdf,
)
from .ocr_client import OCRClient, OCRUnavailableError

logger = logging.getLogger(__name__)

FETCH_PENDING_SQL = """
SELECT id, document_type, title, content_raw, file_path,
       meeting_date, committee_name, meeting_id, agenda_item_id,
       source_url, metadata
FROM documents
WHERE tenant_id = $1 AND processing_status = 'pending'
  AND ($2::varchar IS NULL OR document_type = $2)
ORDER BY meeting_date DESC NULLS LAST
"""

INSERT_CHUNK_SQL = """
INSERT INTO chunks (tenant_id, document_id, chunk_index, content, token_count,
                    embedding_status, metadata)
VALUES ($1, $2, $3, $4, $5, 'pending', $6)
ON CONFLICT (document_id, chunk_index) DO NOTHING
"""

UPDATE_DOC_COMPLETE_SQL = """
UPDATE documents
SET processing_status = 'complete',
    content_text = $2,
    ocr_applied = $3,
    ocr_method = $4,
    page_count = $5,
    metadata = metadata || $6::jsonb,
    updated_at = NOW()
WHERE id = $1
"""

UPDATE_DOC_FAILED_SQL = """
UPDATE documents
SET processing_status = 'failed',
    processing_error = $2,
    updated_at = NOW()
WHERE id = $1
"""


@dataclass
class ProcessingStats:
    total: int = 0
    completed: int = 0
    failed: int = 0
    chunks_created: int = 0
    start_time: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)


def strip_html(raw: str) -> str:
    """Strip HTML tags and return plain text."""
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _build_chunk_metadata(
    row: asyncpg.Record,
    extra_meta: dict | None = None,
) -> str:
    """Build JSONB metadata for chunks from document row."""
    meta: dict = {}
    if row["meeting_date"]:
        meta["meeting_date"] = str(row["meeting_date"])
    if row["committee_name"]:
        meta["committee_name"] = row["committee_name"]
    if row["meeting_id"]:
        meta["meeting_id"] = row["meeting_id"]
    if row["agenda_item_id"]:
        meta["agenda_item_id"] = row["agenda_item_id"]
    if row["title"]:
        meta["title"] = row["title"]
    if row["source_url"]:
        meta["source_url"] = row["source_url"]
    if extra_meta:
        meta.update(extra_meta)
    return json.dumps(meta)


async def process_document(
    row: asyncpg.Record,
    pool: asyncpg.Pool,
    ocr: OCRClient,
    tenant: str,
    dry_run: bool = False,
) -> tuple[bool, int]:
    """Process a single document. Returns (success, chunk_count)."""
    doc_id = row["id"]
    doc_type = row["document_type"]
    title = row["title"] or "(untitled)"

    try:
        text, ocr_applied, ocr_method, page_count, raw_image_count = await _extract_text(row, ocr)
        # Zero-trust: validate image_count at this boundary regardless of source.
        # _extract_text may return data from OCR HTTP response, local extractors,
        # or future code paths — never trust upstream to have validated.
        try:
            image_count = max(int(raw_image_count), 0)
        except (TypeError, ValueError):
            image_count = 0
    except OCRUnavailableError as e:
        # OCR service not running — leave document as 'pending' for retry
        logger.warning("OCR unavailable, skipping [%s] %s: %s", doc_type, title, e)
        return False, 0
    except Exception as e:
        logger.error("Extract failed [%s] %s: %s", doc_type, title, e)
        if not dry_run:
            async with pool.acquire() as conn:
                await conn.execute(UPDATE_DOC_FAILED_SQL, doc_id, str(e)[:1000])
        return False, 0

    if not text or not text.strip():
        msg = "No text extracted"
        logger.warning("Empty text [%s] %s", doc_type, title)
        if not dry_run:
            async with pool.acquire() as conn:
                await conn.execute(UPDATE_DOC_FAILED_SQL, doc_id, msg)
        return False, 0

    chunks = chunk_text(text)
    extra_meta = {"extraction_tier": 0} if ocr_method == "pptx_native" else None
    chunk_meta = _build_chunk_metadata(row, extra_meta=extra_meta)
    char_count = len(text)

    if dry_run:
        logger.info(
            "DRY RUN [%s] %s — %d chars, %d chunks",
            doc_type,
            title,
            char_count,
            len(chunks),
        )
        for i, (ct, tc) in enumerate(chunks):
            logger.info("  chunk %d: %d tokens, preview: %.80s…", i, tc, ct)
        return True, len(chunks)

    # Atomic transaction: insert chunks + update document
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for idx, (chunk_content, token_count) in enumerate(chunks):
                    await conn.execute(
                        INSERT_CHUNK_SQL,
                        tenant,
                        doc_id,
                        idx,
                        chunk_content,
                        token_count,
                        chunk_meta,
                    )

                doc_meta = json.dumps({"char_count": char_count, "image_count": image_count})
                await conn.execute(
                    UPDATE_DOC_COMPLETE_SQL,
                    doc_id,
                    text,
                    ocr_applied,
                    ocr_method,
                    page_count,
                    doc_meta,
                )
    except Exception as e:
        logger.error("Transaction failed [%s] %s: %s", doc_type, title, e)
        try:
            async with pool.acquire() as conn:
                await conn.execute(UPDATE_DOC_FAILED_SQL, doc_id, str(e)[:1000])
        except Exception:
            pass  # Already logging the primary error
        return False, 0

    logger.info(
        "OK [%s] %s — %d chars, %d chunks",
        doc_type,
        title,
        char_count,
        len(chunks),
    )
    return True, len(chunks)


PPTX_EXTENSIONS = {".pptx", ".ppsx"}


async def _extract_text(row: asyncpg.Record, ocr: OCRClient) -> tuple[str, bool, str | None, int | None, int]:
    """Extract text from a document. Returns (text, ocr_applied, ocr_method, page_count, image_count)."""
    doc_type = row["document_type"]

    if doc_type == "agenda_item":
        raw = row["content_raw"]
        if not raw:
            return "", False, None, None, 0
        return strip_html(raw), False, None, None, 0

    # agenda or attachment — check file path
    file_path = row["file_path"]
    if not file_path:
        raise ValueError(f"No file_path for {doc_type} document")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # PowerPoint files: extract directly, no OCR needed
    if ext in PPTX_EXTENSIONS:
        from .pptx_extractor import extract_pptx

        text, slide_count = extract_pptx(file_path)
        return text, False, "pptx_native", slide_count, 0

    # Legacy Office formats: convert to PDF via LibreOffice, then OCR
    if ext in CONVERTIBLE_EXTENSIONS:
        return await _extract_via_libreoffice(file_path, ocr)

    # PDF and other attachments — use OCR service
    result = await ocr.extract(file_path)
    return result.text, True, result.source, result.page_count, result.image_count


async def _extract_via_libreoffice(file_path: str, ocr: OCRClient) -> tuple[str, bool, str | None, int | None, int]:
    """Convert a legacy Office file to PDF, then extract text via OCR service."""
    loop = asyncio.get_running_loop()
    pdf_path = await loop.run_in_executor(None, convert_to_pdf, file_path)
    try:
        result = await ocr.extract(pdf_path)
        method = f"libreoffice_convert+{result.source}"
        return result.text, True, method, result.page_count, result.image_count
    finally:
        # Clean up the temp PDF and its directory
        try:
            parent = os.path.dirname(pdf_path)
            os.unlink(pdf_path)
            os.rmdir(parent)
        except OSError:
            pass


async def _progress_reporter(stats: ProcessingStats, interval: float = 10.0) -> None:
    """Background task that logs progress every interval seconds."""
    while True:
        await asyncio.sleep(interval)
        elapsed = time.time() - stats.start_time
        rate = stats.completed / (elapsed / 60) if elapsed > 0 else 0
        remaining = stats.total - stats.completed - stats.failed
        eta_min = remaining / rate if rate > 0 else 0
        logger.info(
            "Progress: %d/%d complete, %d failed, %.1f docs/min, ETA %.1f min",
            stats.completed,
            stats.total,
            stats.failed,
            rate,
            eta_min,
        )


async def _log_document_result(
    log_path: str,
    doc_id: str,
    doc_type: str,
    title: str,
    success: bool,
    chunk_count: int,
    error: str | None = None,
) -> None:
    """Append a JSON log line for this document."""
    entry = {
        "document_id": str(doc_id),
        "document_type": doc_type,
        "title": title,
        "success": success,
        "chunk_count": chunk_count,
        "timestamp": time.time(),
    }
    if error:
        entry["error"] = error
    line = json.dumps(entry) + "\n"

    try:
        with open(log_path, "a") as f:
            f.write(line)
    except OSError:
        pass  # Don't fail processing due to log write errors


async def run(
    dsn: str,
    ocr_url: str,
    tenant: str,
    workers: int = 4,
    dry_run: bool = False,
    document_type: str | None = None,
) -> ProcessingStats:
    """Main entry point: fetch pending docs and process them."""
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=workers + 1)
    ocr = OCRClient(ocr_url)
    sem = asyncio.Semaphore(workers)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(FETCH_PENDING_SQL, tenant, document_type)

        if dry_run:
            rows = rows[:5]

        stats = ProcessingStats(total=len(rows))
        logger.info(
            "Found %d pending documents (type=%s, dry_run=%s)",
            len(rows),
            document_type or "all",
            dry_run,
        )

        if not rows:
            return stats

        # Start progress reporter
        progress_task = asyncio.create_task(_progress_reporter(stats))

        log_path = "processing.log"

        async def _process_one(row: asyncpg.Record) -> None:
            async with sem:
                try:
                    success, chunk_count = await process_document(row, pool, ocr, tenant, dry_run)
                except Exception as e:
                    success, chunk_count = False, 0
                    logger.error("Unexpected error doc %s: %s", row["id"], e)

                if success:
                    stats.completed += 1
                    stats.chunks_created += chunk_count
                else:
                    stats.failed += 1
                    stats.errors.append(f"{row['id']}: {row.get('title', '')}")

                await _log_document_result(
                    log_path,
                    row["id"],
                    row["document_type"],
                    row["title"] or "",
                    success,
                    chunk_count,
                    error=None if success else "processing failed",
                )

        tasks = [asyncio.create_task(_process_one(row)) for row in rows]
        await asyncio.gather(*tasks)

        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    finally:
        await ocr.close()
        await pool.close()

    elapsed = time.time() - stats.start_time
    logger.info(
        "Done: %d complete, %d failed, %d chunks in %.1fs",
        stats.completed,
        stats.failed,
        stats.chunks_created,
        elapsed,
    )
    return stats
