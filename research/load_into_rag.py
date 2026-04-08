#!/usr/bin/env python3
"""Load research documents into the RAG pipeline (documents + chunks tables)."""

import asyncio
import json
import os
import re
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


DB_DSN = _build_dsn()
TENANT_ID = "kent_sd"
TODAY = date.today().isoformat()

# Files to load
RESEARCH_DIR = Path(__file__).parent
FILES = [
    {
        "path": RESEARCH_DIR / f"meeting_intelligence_{TODAY}.md",
        "title": f"Meeting Intelligence Brief — {TODAY}",
        "doc_type": "research_analysis",
    },
    {
        "path": RESEARCH_DIR / f"cheat_sheet_{TODAY}.md",
        "title": f"Budget Meeting Cheat Sheet — {TODAY}",
        "doc_type": "research_analysis",
    },
    {
        "path": RESEARCH_DIR / "contradictions.json",
        "title": "Budget Claim Contradictions Analysis (Dec 2023 – Jun 2025)",
        "doc_type": "research_analysis",
    },
    {
        "path": RESEARCH_DIR / "extracted_claims.json",
        "title": "Extracted Factual Claims from Board Meeting Transcripts (Dec 2023 – Jun 2025)",
        "doc_type": "research_analysis",
    },
]


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            # Keep overlap from end of current chunk
            if len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


async def main():
    conn = await asyncpg.connect(DB_DSN)

    total_docs = 0
    total_chunks = 0

    for file_info in FILES:
        path = file_info["path"]
        if not path.exists():
            print(f"  SKIP: {path.name} (not found)")
            continue

        content = path.read_text()
        title = file_info["title"]
        doc_type = file_info["doc_type"]
        doc_id = str(uuid.uuid4())

        # For JSON files, create a readable text version for chunking
        if path.suffix == ".json":
            data = json.loads(content)
            text_parts = []

            if "contradictions" in data:
                text_parts.append(
                    f"# Budget Claim Contradictions Analysis\n\nTotal contradictions: {data['total_contradictions']}\nHIGH: {data['by_confidence']['HIGH']}, MEDIUM: {data['by_confidence']['MEDIUM']}, LOW: {data['by_confidence']['LOW']}\n"
                )
                for i, c in enumerate(data["contradictions"], 1):
                    claim = c["claim"]
                    cont = c["contradiction"]
                    text_parts.append(
                        f"## Contradiction #{i} [{cont['confidence']}]\n"
                        f"Speaker: {claim['speaker_name']} ({claim.get('speaker_role', 'Unknown')})\n"
                        f"Meeting: {claim['meeting_date']}\n"
                        f"Claim: \"{claim['quote']}\"\n"
                        f"Contradiction: {cont['summary']}\n"
                        f"Evidence: {cont['evidence_details']}\n"
                        f"Source: {cont['evidence_source']}\n"
                    )
                chunk_content = "\n\n".join(text_parts)
            elif "transcripts" in data:
                text_parts.append(
                    f"# Extracted Claims from Board Meeting Transcripts\n\nTotal claims: {data['total_claims']}\n"
                )
                for t in data["transcripts"]:
                    text_parts.append(f"## {t['title']} ({t['meeting_date']})\n")
                    for claim in t.get("claims", []):
                        text_parts.append(
                            f"- {claim.get('speaker_name', 'Unknown')} ({claim.get('category', 'Unknown')}): "
                            f"\"{claim.get('quote', '')[:300]}\"\n"
                            f"  Verifiable: {claim.get('verifiable_element', 'N/A')}\n"
                        )
                chunk_content = "\n\n".join(text_parts)
            else:
                chunk_content = json.dumps(data, indent=2)
        else:
            chunk_content = content

        # Insert document
        await conn.execute(
            """
            INSERT INTO documents (id, tenant_id, document_type, title, content_text,
                                   processing_status, meeting_date, metadata)
            VALUES ($1, $2, $3, $4, $5, 'complete', $6, $7)
            ON CONFLICT DO NOTHING
        """,
            uuid.UUID(doc_id),
            TENANT_ID,
            doc_type,
            title,
            chunk_content,
            date.fromisoformat(TODAY),
            json.dumps({"source": "research_analysis", "generated_date": TODAY}),
        )

        # Chunk and insert
        chunks = chunk_text(chunk_content)
        for i, chunk in enumerate(chunks):
            await conn.execute(
                """
                INSERT INTO chunks (tenant_id, document_id, chunk_index, content,
                                    token_count, embedding_status, metadata)
                VALUES ($1, $2, $3, $4, $5, 'pending', $6)
                ON CONFLICT DO NOTHING
            """,
                TENANT_ID,
                uuid.UUID(doc_id),
                i,
                chunk,
                estimate_tokens(chunk),
                json.dumps({"document_title": title, "document_type": doc_type}),
            )

        total_docs += 1
        total_chunks += len(chunks)
        print(f"  OK: {title} — {len(chunk_content):,} chars → {len(chunks)} chunks")

    await conn.close()

    print(f"\nLoaded {total_docs} documents, {total_chunks} chunks (pending embedding)")
    print("Embedding pipeline will pick these up on next cron run (*/30 * * * *)")
    print("Or run manually: cd embedding_pipeline && venv/bin/python -m embedding_pipeline")


if __name__ == "__main__":
    asyncio.run(main())
