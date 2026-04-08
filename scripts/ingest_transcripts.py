#!/usr/bin/env python3
"""Ingest 7 priority transcript files into PostgreSQL as documents + chunks,
then run the embedding pipeline for just those documents.
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import date
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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
TENANT_ID = "kent_sd"
TRANSCRIPT_DIR = Path(os.path.expanduser("~/ksd_forensic/output/transcripts"))
MAX_CHUNK_WORDS = 150  # ~200 tokens, consistent with document_processor chunking

# Map filenames to meeting dates
FILENAME_TO_DATE = {
    "25JUN25Meeting.json": date(2025, 6, 25),
    "26FEB2025Meeting.json": date(2025, 2, 26),
    "10JUL2024Meeting.json": date(2024, 7, 10),
    "26JUN2024Meeting.json": date(2024, 6, 26),
    "21AUG2024Meeting.json": date(2024, 8, 21),
    "24JAN2024Meeting.json": date(2024, 1, 24),
    "13DEC2023Meeting.json": date(2023, 12, 13),
}


def group_words_to_utterances(words: list[dict]) -> list[dict]:
    """Group consecutive same-speaker words into utterances."""
    if not words:
        return []
    utterances = []
    current_speaker = words[0].get("speaker", "Unknown")
    current_words = [words[0]["text"]]
    current_start = words[0]["start"]

    for w in words[1:]:
        speaker = w.get("speaker", "Unknown")
        if speaker == current_speaker:
            current_words.append(w["text"])
        else:
            utterances.append(
                {
                    "speaker": current_speaker,
                    "text": " ".join(current_words),
                    "start": current_start,
                }
            )
            current_speaker = speaker
            current_words = [w["text"]]
            current_start = w["start"]

    utterances.append(
        {
            "speaker": current_speaker,
            "text": " ".join(current_words),
            "start": current_start,
        }
    )
    return utterances


def chunk_utterances(utterances: list[dict], max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Chunk utterances into segments of roughly max_words words."""
    chunks = []
    current_chunk_parts = []
    current_word_count = 0

    for utt in utterances:
        speaker = utt["speaker"]
        text = utt["text"]
        word_count = len(text.split())
        line = f"[{speaker}]: {text}"

        # If single utterance exceeds max, split it
        if word_count > max_words:
            # Flush current chunk first
            if current_chunk_parts:
                chunks.append("\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_word_count = 0
            # Split long utterance
            words = text.split()
            for i in range(0, len(words), max_words):
                segment = " ".join(words[i : i + max_words])
                chunks.append(f"[{speaker}]: {segment}")
        elif current_word_count + word_count > max_words:
            # Flush and start new chunk
            chunks.append("\n".join(current_chunk_parts))
            current_chunk_parts = [line]
            current_word_count = word_count
        else:
            current_chunk_parts.append(line)
            current_word_count += word_count

    if current_chunk_parts:
        chunks.append("\n".join(current_chunk_parts))

    return chunks


async def main():
    json_files = sorted(TRANSCRIPT_DIR.glob("*.json"))
    if not json_files:
        print("No JSON transcript files found!")
        sys.exit(1)

    print(f"Found {len(json_files)} transcript files")

    conn = await asyncpg.connect(DSN)

    doc_ids = []
    total_chunks = 0

    for fpath in json_files:
        fname = fpath.name
        meeting_date = FILENAME_TO_DATE.get(fname)
        if meeting_date is None:
            print(f"  SKIP {fname} — no date mapping")
            continue

        print(f"\nProcessing {fname} (meeting: {meeting_date})...")

        with open(fpath) as f:
            words = json.load(f)

        print(f"  {len(words)} word tokens")

        # Group into utterances
        utterances = group_words_to_utterances(words)
        print(f"  {len(utterances)} utterances")

        # Build full text for content_text
        full_text = "\n".join(f"[{u['speaker']}]: {u['text']}" for u in utterances)

        # Chunk
        chunks = chunk_utterances(utterances)
        print(f"  {len(chunks)} chunks")

        # Insert document
        doc_id = uuid.uuid4()
        title = f"Board Meeting Transcript — {meeting_date.strftime('%B %d, %Y')}"

        await conn.execute(
            """
            INSERT INTO documents (
                id, tenant_id, external_id, document_type, title,
                content_text, meeting_date, committee_name,
                processing_status, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (tenant_id, external_id) DO NOTHING
        """,
            doc_id,
            TENANT_ID,
            f"transcript-{fname}",
            "transcript",
            title,
            full_text,
            meeting_date,
            "Regular Meeting",
            "complete",  # text extraction is already done
            json.dumps({"source": "assemblyai", "word_count": len(words)}),
        )

        # Insert chunks with embedding_status='pending'
        for idx, chunk_text in enumerate(chunks):
            chunk_id = uuid.uuid4()
            word_count = len(chunk_text.split())
            await conn.execute(
                """
                INSERT INTO chunks (
                    id, tenant_id, document_id, chunk_index, content,
                    token_count, embedding_status, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (document_id, chunk_index) DO NOTHING
            """,
                chunk_id,
                TENANT_ID,
                doc_id,
                idx,
                chunk_text,
                int(word_count * 1.3),  # approximate token count
                "pending",
                json.dumps({"document_type": "transcript"}),
            )

        doc_ids.append(doc_id)
        total_chunks += len(chunks)
        print(f"  Inserted doc {doc_id} with {len(chunks)} chunks")

    await conn.close()

    print(f"\n{'='*60}")
    print(f"Inserted {len(doc_ids)} documents, {total_chunks} chunks total")
    print("\nDocument IDs:")
    for did in doc_ids:
        print(f"  {did}")

    # Write doc IDs to file for embedding step
    with open("/tmp/transcript_doc_ids.txt", "w") as f:
        for did in doc_ids:
            f.write(f"{did}\n")

    print("\nDoc IDs saved to /tmp/transcript_doc_ids.txt")


if __name__ == "__main__":
    asyncio.run(main())
