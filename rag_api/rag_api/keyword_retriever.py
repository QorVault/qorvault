"""PostgreSQL full-text keyword search for document chunks.

Uses the tsvector column on the chunks table with ts_rank_cd scoring.
Joins to the documents table for metadata required by the RAG pipeline.
All queries are tenant-scoped and parameterized.
"""

from __future__ import annotations

import logging

import asyncpg

from .retriever import RetrievedChunk, build_boarddocs_url

logger = logging.getLogger(__name__)

# plainto_tsquery splits on whitespace and ANDs the terms together.
# This works well for multi-word name queries like "Onorati retirement".
_KEYWORD_SEARCH_SQL = """\
SELECT
    c.id AS chunk_id,
    c.document_id,
    c.content,
    c.chunk_index,
    d.title,
    d.meeting_date,
    d.committee_name,
    d.document_type,
    d.meeting_id,
    d.agenda_item_id,
    ts_rank_cd(c.search_vector, plainto_tsquery('english', $2)) AS rank
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.tenant_id = $1
  AND c.search_vector @@ plainto_tsquery('english', $2)
ORDER BY rank DESC
LIMIT $3
"""

# phraseto_tsquery requires terms to appear adjacent and in order.
# Better for exact name lookups like "Karen Onorati".
_PHRASE_SEARCH_SQL = """\
SELECT
    c.id AS chunk_id,
    c.document_id,
    c.content,
    c.chunk_index,
    d.title,
    d.meeting_date,
    d.committee_name,
    d.document_type,
    d.meeting_id,
    d.agenda_item_id,
    ts_rank_cd(c.search_vector, phraseto_tsquery('english', $2)) AS rank
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.tenant_id = $1
  AND c.search_vector @@ phraseto_tsquery('english', $2)
ORDER BY rank DESC
LIMIT $3
"""


class KeywordRetriever:
    """Search chunks using PostgreSQL full-text search."""

    def __init__(self, pool: asyncpg.Pool, boarddocs_base_url: str = "") -> None:
        """Initialize with an asyncpg connection pool."""
        self._pool = pool
        self._boarddocs_base_url = boarddocs_base_url

    async def search(
        self,
        query: str,
        top_k: int = 20,
        tenant_id: str = "kent_sd",
        phrase_mode: bool = False,
        excluded_doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run a keyword search and return ranked chunks.

        Args:
            query: Search terms (whitespace-separated).
            top_k: Maximum results to return.
            tenant_id: Tenant scope.
            phrase_mode: If True, use phraseto_tsquery for exact
                phrase matching (adjacent terms in order).
            excluded_doc_types: Document types to exclude from results.

        Returns:
            List of RetrievedChunk sorted by ts_rank_cd descending.
        """
        sql = _PHRASE_SEARCH_SQL if phrase_mode else _KEYWORD_SEARCH_SQL

        # Inject exclusion clause before ORDER BY when doc types are excluded.
        params: list = [tenant_id, query, top_k]
        if excluded_doc_types:
            sql = sql.replace(
                "ORDER BY rank DESC",
                "  AND d.document_type != ALL($4)\nORDER BY rank DESC",
            )
            params.append(excluded_doc_types)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        chunks = []
        for row in rows:
            meeting_id = row["meeting_id"]
            agenda_item_id = row["agenda_item_id"]
            meeting_date = str(row["meeting_date"]) if row["meeting_date"] else None

            source_url = (
                build_boarddocs_url(self._boarddocs_base_url, meeting_id, agenda_item_id)
                if self._boarddocs_base_url
                else None
            )

            chunks.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=row["content"],
                    score=float(row["rank"]),
                    title=row["title"],
                    meeting_date=meeting_date,
                    committee_name=row["committee_name"],
                    document_type=row["document_type"] or "",
                    source_url=source_url,
                    chunk_index=row["chunk_index"],
                    meeting_id=meeting_id,
                    agenda_item_id=agenda_item_id,
                )
            )

        logger.info(
            "Keyword search (%s mode): %d results for %r",
            "phrase" if phrase_mode else "plain",
            len(chunks),
            query[:60],
        )
        return chunks
