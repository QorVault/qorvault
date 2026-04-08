"""Main embedding pipeline: fetch chunks, embed, upsert to Qdrant, update PostgreSQL."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field

import asyncpg
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from .embedder import Embedder

logger = logging.getLogger(__name__)

FETCH_PENDING_SQL = """
SELECT c.id, c.tenant_id, c.document_id, c.chunk_index, c.content,
       c.token_count, c.contains_table, c.metadata,
       d.document_type
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE c.embedding_status = 'pending' AND c.tenant_id = $1
ORDER BY c.document_id, c.chunk_index
LIMIT $2
"""

UPDATE_CHUNK_SQL = """
UPDATE chunks
SET embedding_status = 'complete',
    qdrant_point_id = $2,
    embedding_model = $3
WHERE id = $1
"""

MARK_FAILED_SQL = """
UPDATE chunks
SET embedding_status = 'failed'
WHERE id = $1
"""


@dataclass
class PipelineStats:
    total_embedded: int = 0
    total_failed: int = 0
    batches_processed: int = 0
    start_time: float = field(default_factory=time.time)


def build_payload(row: asyncpg.Record) -> dict:
    """Build Qdrant payload from a chunk row + joined document data."""
    raw = row["metadata"]
    if isinstance(raw, str):
        meta = json.loads(raw)
    elif isinstance(raw, dict):
        meta = raw
    else:
        meta = {}
    return {
        "chunk_id": str(row["id"]),
        "document_id": str(row["document_id"]),
        "tenant_id": row["tenant_id"],
        "content": row["content"],
        "document_type": row["document_type"],
        "meeting_date": meta.get("meeting_date"),
        "committee_name": meta.get("committee_name"),
        "meeting_id": meta.get("meeting_id"),
        "agenda_item_id": meta.get("agenda_item_id"),
        "title": meta.get("title"),
        "source_url": meta.get("source_url"),
        "chunk_index": row["chunk_index"],
        "token_count": row["token_count"],
        "contains_table": row["contains_table"] or False,
    }


REQUIRED_PAYLOAD_FIELDS = {
    "chunk_id",
    "document_id",
    "tenant_id",
    "content",
    "document_type",
    "meeting_date",
    "committee_name",
    "meeting_id",
    "agenda_item_id",
    "title",
    "source_url",
    "chunk_index",
    "token_count",
    "contains_table",
}


def _upsert_with_retry(
    qdrant: QdrantClient,
    collection: str,
    points: list[PointStruct],
    max_retries: int = 3,
    backoff: float = 10.0,
) -> None:
    """Upsert to Qdrant with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            qdrant.upsert(collection_name=collection, points=points)
            return
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "Qdrant upsert attempt %d/%d failed: %s, retrying in %.0fs",
                    attempt,
                    max_retries,
                    e,
                    backoff * attempt,
                )
                time.sleep(backoff * attempt)
            else:
                raise


def _log_batch(log_path: str, batch_num: int, count: int, success: bool, error: str | None = None) -> None:
    """Append a JSON log line for a batch."""
    entry = {
        "batch": batch_num,
        "count": count,
        "success": success,
        "timestamp": time.time(),
    }
    if error:
        entry["error"] = error
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


async def _progress_reporter(stats: PipelineStats, interval: float = 15.0) -> None:
    """Background task that logs progress every interval seconds."""
    while True:
        await asyncio.sleep(interval)
        elapsed = time.time() - stats.start_time
        rate = stats.total_embedded / (elapsed / 60) if elapsed > 0 else 0
        logger.info(
            "Progress: %d embedded, %d failed, %.1f chunks/min, %d batches",
            stats.total_embedded,
            stats.total_failed,
            rate,
            stats.batches_processed,
        )


async def run(
    dsn: str,
    qdrant_url: str,
    collection: str,
    tenant: str,
    batch_size: int = 256,
    dry_run: bool = False,
    limit: int = 0,
) -> PipelineStats:
    """Main entry point: fetch pending chunks, embed, upsert."""
    embedder = Embedder()
    embedder.load()

    qdrant = QdrantClient(url=qdrant_url, check_compatibility=False)
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
    stats = PipelineStats()
    log_path = "embedding.log"

    # Start progress reporter
    progress_task = asyncio.create_task(_progress_reporter(stats))

    try:
        chunks_processed = 0

        while True:
            # Determine how many to fetch this batch
            fetch_limit = batch_size
            if limit > 0:
                remaining = limit - chunks_processed
                if remaining <= 0:
                    break
                fetch_limit = min(batch_size, remaining)

            if dry_run:
                fetch_limit = 3

            # Fetch batch
            async with pool.acquire() as conn:
                rows = await conn.fetch(FETCH_PENDING_SQL, tenant, fetch_limit)

            if not rows:
                logger.info("No more pending chunks.")
                break

            logger.info("Batch %d: %d chunks", stats.batches_processed + 1, len(rows))

            # Extract texts
            texts = [row["content"] for row in rows]

            # Encode with retry
            embeddings = None
            for encode_attempt in range(2):
                try:
                    embeddings = embedder.encode(texts)
                    break
                except Exception as e:
                    if encode_attempt == 0:
                        logger.warning("Encode failed, retrying in 5s: %s", e)
                        await asyncio.sleep(5)
                    else:
                        logger.error("Encode failed after retry: %s", e)

            if embeddings is None:
                # Mark all chunks in batch as failed
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        for row in rows:
                            await conn.execute(MARK_FAILED_SQL, row["id"])
                stats.total_failed += len(rows)
                _log_batch(log_path, stats.batches_processed + 1, len(rows), False, "encode failed")
                stats.batches_processed += 1
                chunks_processed += len(rows)
                if dry_run:
                    break
                continue

            # Build Qdrant points
            point_ids: list[uuid.UUID] = []
            points: list[PointStruct] = []
            payloads: list[dict] = []

            for i, row in enumerate(rows):
                pid = uuid.uuid4()
                point_ids.append(pid)
                payload = build_payload(row)
                payloads.append(payload)
                points.append(
                    PointStruct(
                        id=str(pid),
                        vector=embeddings[i].tolist(),
                        payload=payload,
                    )
                )

            if dry_run:
                print("\n=== DRY RUN ===", file=sys.stderr)
                for i, (row, payload) in enumerate(zip(rows, payloads)):
                    vec = embeddings[i]
                    print(f"\nChunk {i}: {row['id']}", file=sys.stderr)
                    print(f"  Dimensions: {vec.shape[0]}", file=sys.stderr)
                    print(f"  First 8 dims: {vec[:8].tolist()}", file=sys.stderr)
                    norm = float(np.linalg.norm(vec))
                    print(f"  L2 norm: {norm:.6f}", file=sys.stderr)
                    print(f"  Payload: {json.dumps(payload, indent=2, default=str)}", file=sys.stderr)
                print("\nDry run complete — nothing written to Qdrant or PostgreSQL.", file=sys.stderr)
                stats.total_embedded = len(rows)
                stats.batches_processed = 1
                break

            # Upsert to Qdrant with retry
            try:
                _upsert_with_retry(qdrant, collection, points)
            except Exception as e:
                logger.error("Qdrant upsert failed after retries: %s", e)
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        for row in rows:
                            await conn.execute(MARK_FAILED_SQL, row["id"])
                stats.total_failed += len(rows)
                _log_batch(log_path, stats.batches_processed + 1, len(rows), False, str(e))
                stats.batches_processed += 1
                chunks_processed += len(rows)
                continue

            # Update PostgreSQL — if this fails, chunks will be re-embedded next run
            try:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        for row, pid in zip(rows, point_ids):
                            await conn.execute(
                                UPDATE_CHUNK_SQL,
                                row["id"],
                                pid,
                                "mxbai-embed-large-v1",
                            )
            except Exception as e:
                logger.warning("PostgreSQL update failed after Qdrant upsert (safe to re-run): %s", e)

            stats.total_embedded += len(rows)
            stats.batches_processed += 1
            chunks_processed += len(rows)
            _log_batch(log_path, stats.batches_processed, len(rows), True)

            logger.info(
                "Batch %d complete: %d chunks embedded",
                stats.batches_processed,
                len(rows),
            )

    finally:
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass
        await pool.close()
        qdrant.close()

    elapsed = time.time() - stats.start_time
    logger.info(
        "Done: %d embedded, %d failed in %.1fs (%.1f chunks/min)",
        stats.total_embedded,
        stats.total_failed,
        elapsed,
        stats.total_embedded / (elapsed / 60) if elapsed > 0 else 0,
    )
    return stats
