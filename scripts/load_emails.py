#!/usr/bin/env python3
"""Load .eml files into the BoardDocs RAG pipeline.

Parses each .eml file, inserts email body as a document + chunks it,
and extracts PDF/DOCX attachments as pending documents for the
document_processor to handle via OCR.

Usage:
    cd /home/qorvault/projects/ksd-boarddocs-rag
    document_processor/venv/bin/python scripts/load_emails.py
    document_processor/venv/bin/python scripts/load_emails.py --dry-run
    document_processor/venv/bin/python scripts/load_emails.py --input-dir /path/to/emails
"""

from __future__ import annotations

import argparse
import asyncio
import email
import email.policy
import email.utils
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

# Add document_processor to sys.path so we can import chunk_text
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "document_processor"))

import asyncpg
from dotenv import load_dotenv

from document_processor.chunker import chunk_text

load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

TENANT_ID = "kent_sd"


def _build_dsn() -> str:
    """Build PostgreSQL URL from POSTGRES_* env vars. No hardcoded credentials."""
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "qorvault")
    user = os.environ.get("POSTGRES_USER", "qorvault")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set. " "Copy .env.example to .env and fill in credentials."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


DSN = _build_dsn()

INSERT_DOC_SQL = """
INSERT INTO documents (
    tenant_id, external_id, document_type, title,
    content_raw, content_text, file_path,
    meeting_date, processing_status, metadata
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (tenant_id, external_id) DO NOTHING
RETURNING id
"""

INSERT_CHUNK_SQL = """
INSERT INTO chunks (tenant_id, document_id, chunk_index, content, token_count,
                    embedding_status, metadata)
VALUES ($1, $2, $3, $4, $5, 'pending', $6)
ON CONFLICT (document_id, chunk_index) DO NOTHING
"""

ATTACHMENT_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".ppsx", ".xlsx"}


def parse_date(msg: email.message.Message, eml_path: Path) -> date:
    """Extract a date from the Date header, falling back to file mtime."""
    date_str = msg.get("Date")
    if date_str:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.date()
    mtime = eml_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=UTC).date()


def slugify(text: str, max_len: int = 60) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    text = text.strip("-")
    return text[:max_len].rstrip("-")


def content_hash(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def get_body_text(msg: email.message.Message) -> tuple[str, str | None]:
    """Extract plain text body and optional raw HTML.

    Returns (plain_text, html_or_none).
    """
    body = msg.get_body(preferencelist=("plain",))
    html_body = msg.get_body(preferencelist=("html",))

    plain_text = ""
    raw_html = None

    if body:
        content = body.get_content()
        if isinstance(content, str):
            plain_text = content.strip()

    if html_body:
        content = html_body.get_content()
        if isinstance(content, str):
            raw_html = content.strip()

    # If no plain text but we have HTML, extract text from HTML
    if not plain_text and raw_html:
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []

            def handle_data(self, data: str) -> None:
                self.parts.append(data)

        extractor = TextExtractor()
        extractor.feed(raw_html)
        plain_text = " ".join(extractor.parts).strip()

    return plain_text, raw_html


def extract_attachments(
    msg: email.message.Message,
    output_dir: Path,
    email_date: date,
    subject: str,
) -> list[dict]:
    """Extract file attachments to disk. Returns list of attachment info dicts."""
    attachments = []
    slug = slugify(subject) or "no-subject"
    subdir = output_dir / f"{email_date}_{slug}"

    for part in msg.walk():
        content_disposition = part.get("Content-Disposition", "")
        if "attachment" not in content_disposition.lower():
            continue

        filename = part.get_filename()
        if not filename:
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in ATTACHMENT_EXTENSIONS:
            logger.debug("Skipping non-document attachment: %s", filename)
            continue

        payload = part.get_payload(decode=True)
        if not payload:
            continue

        subdir.mkdir(parents=True, exist_ok=True)
        # Sanitize filename
        safe_name = re.sub(r"[^\w.\-()]", "_", filename)
        dest = subdir / safe_name

        # Handle duplicates
        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = subdir / f"{stem}_{counter}{suffix}"
                counter += 1

        dest.write_bytes(payload)

        attachments.append(
            {
                "filename": filename,
                "safe_filename": safe_name,
                "file_path": str(dest),
                "file_extension": ext,
                "file_size_bytes": len(payload),
            }
        )

    return attachments


async def load_email(
    eml_path: Path,
    pool: asyncpg.Pool,
    attachment_output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Parse and load a single .eml file. Returns stats dict."""
    stats = {"email_inserted": False, "chunks_created": 0, "attachments_extracted": 0, "skipped": False}

    raw = eml_path.read_bytes()
    msg = email.message_from_bytes(raw, policy=email.policy.default)

    # Message-ID for dedup
    message_id = (msg.get("Message-ID") or "").strip()
    if not message_id:
        message_id = f"content-hash:{content_hash(raw)}"

    external_id = f"email_{message_id}"
    subject = (msg.get("Subject") or "(no subject)").strip()
    email_date = parse_date(msg, eml_path)

    # From/To
    from_addr = (msg.get("From") or "").strip()
    to_addr = (msg.get("To") or "").strip()

    # Body text
    plain_text, raw_html = get_body_text(msg)

    # Extract attachments from message
    attachments = extract_attachments(msg, attachment_output_dir, email_date, subject)
    stats["attachments_extracted"] = len(attachments)

    # Metadata
    doc_metadata = {
        "from_address": from_addr,
        "to_address": to_addr,
        "message_id": message_id,
        "has_attachments": len(attachments) > 0,
        "attachment_count": len(attachments),
        "filename": eml_path.name,
    }

    if dry_run:
        chunks = chunk_text(plain_text) if plain_text else []
        logger.info(
            "DRY RUN: %s — %d chars, %d chunks, %d attachments",
            subject[:60],
            len(plain_text),
            len(chunks),
            len(attachments),
        )
        stats["email_inserted"] = True
        stats["chunks_created"] = len(chunks)
        return stats

    async with pool.acquire() as conn:
        async with conn.transaction():
            # Insert email document
            row = await conn.fetchrow(
                INSERT_DOC_SQL,
                TENANT_ID,
                external_id,
                "email",
                subject,
                raw_html,  # content_raw = HTML version
                plain_text,  # content_text = plain text
                str(eml_path),  # file_path = source .eml
                email_date,
                "complete",  # no OCR needed for email body
                json.dumps(doc_metadata),
            )

            if row is None:
                # ON CONFLICT — already loaded
                logger.info("Already loaded: %s", subject[:60])
                stats["skipped"] = True
                return stats

            doc_id = row["id"]
            stats["email_inserted"] = True

            # Chunk and insert
            if plain_text:
                chunks = chunk_text(plain_text)
                chunk_meta = json.dumps(
                    {
                        "meeting_date": str(email_date),
                        "title": subject,
                        "committee_name": None,
                        "source_url": None,
                        "document_type": "email",
                        "from_address": from_addr,
                    }
                )

                for idx, (chunk_content, token_count) in enumerate(chunks):
                    await conn.execute(
                        INSERT_CHUNK_SQL,
                        TENANT_ID,
                        doc_id,
                        idx,
                        chunk_content,
                        token_count,
                        chunk_meta,
                    )
                stats["chunks_created"] = len(chunks)

            # Insert attachment documents (pending for document_processor)
            for att in attachments:
                att_external_id = f"email_{message_id}_{att['filename']}"
                att_metadata = {
                    "from_email": from_addr,
                    "email_subject": subject,
                    "email_date": str(email_date),
                    "file_extension": att["file_extension"],
                    "file_size_bytes": att["file_size_bytes"],
                    "source_email_external_id": external_id,
                }
                await conn.fetchrow(
                    INSERT_DOC_SQL,
                    TENANT_ID,
                    att_external_id,
                    "attachment",
                    f"{subject} — {att['filename']}",
                    None,  # content_raw
                    None,  # content_text
                    att["file_path"],  # file_path for OCR
                    email_date,
                    "pending",  # needs document_processor
                    json.dumps(att_metadata),
                )

    logger.info(
        "Loaded: %s — %d chunks, %d attachments",
        subject[:60],
        stats["chunks_created"],
        stats["attachments_extracted"],
    )
    return stats


async def main() -> None:
    parser = argparse.ArgumentParser(description="Load .eml files into BoardDocs RAG pipeline")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/qorvault/projects/ksd_forensic/input/Email"),
        help="Directory containing .eml files",
    )
    parser.add_argument(
        "--attachment-dir",
        type=Path,
        default=Path("/home/qorvault/projects/ksd_forensic/input/Email/attachments"),
        help="Directory to extract attachments into",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse and show stats without writing to DB")
    parser.add_argument("--skip-embed", action="store_true", help="Skip running embedding pipeline after loading")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-8s %(message)s", stream=sys.stderr)

    input_dir = args.input_dir.expanduser().resolve()
    attachment_dir = args.attachment_dir.expanduser().resolve()

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    eml_files = sorted(input_dir.glob("*.eml"))
    if not eml_files:
        logger.error("No .eml files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d .eml files in %s", len(eml_files), input_dir)

    if not args.dry_run:
        attachment_dir.mkdir(parents=True, exist_ok=True)

    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=2)

    t_start = time.time()
    totals = {"emails_inserted": 0, "chunks_created": 0, "attachments_extracted": 0, "skipped": 0, "errors": 0}

    try:
        for eml_path in eml_files:
            try:
                stats = await load_email(eml_path, pool, attachment_dir, dry_run=args.dry_run)
                if stats["email_inserted"]:
                    totals["emails_inserted"] += 1
                if stats["skipped"]:
                    totals["skipped"] += 1
                totals["chunks_created"] += stats["chunks_created"]
                totals["attachments_extracted"] += stats["attachments_extracted"]
            except Exception as exc:
                logger.error("Failed to load %s: %s", eml_path.name, exc)
                totals["errors"] += 1
    finally:
        await pool.close()

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"EMAIL LOADING {'(DRY RUN) ' if args.dry_run else ''}SUMMARY")
    print(f"{'='*60}")
    print(f"Input directory:      {input_dir}")
    print(f"Total .eml files:     {len(eml_files)}")
    print(f"Emails inserted:      {totals['emails_inserted']}")
    print(f"Already loaded:       {totals['skipped']}")
    print(f"Chunks created:       {totals['chunks_created']}")
    print(f"Attachments extracted: {totals['attachments_extracted']}")
    print(f"Errors:               {totals['errors']}")
    print(f"Time:                 {elapsed:.1f}s")
    print()

    # Run embedding pipeline for new email chunks
    if not args.dry_run and not args.skip_embed and totals["chunks_created"] > 0:
        print("Running embedding pipeline for new email chunks...")
        embed_venv = PROJECT_ROOT / "embedding_pipeline" / "venv"
        embed_python = embed_venv / "bin" / "python"
        if embed_python.exists():
            result = subprocess.run(
                [str(embed_python), "-m", "embedding_pipeline"],
                cwd=str(PROJECT_ROOT / "embedding_pipeline"),
                capture_output=False,
            )
            if result.returncode == 0:
                print("Embedding pipeline completed successfully.")
            else:
                print(f"Embedding pipeline exited with code {result.returncode}")
        else:
            print(f"Embedding pipeline venv not found at {embed_venv}")
            print("Run the embedding pipeline manually: cd embedding_pipeline && venv/bin/python -m embedding_pipeline")


if __name__ == "__main__":
    asyncio.run(main())
