"""Hybrid retriever combining Qdrant vector search with PostgreSQL keyword search.

Runs both retrievers concurrently and fuses their ranked result lists
using Reciprocal Rank Fusion (RRF).  Supports three modes for A/B
comparison: "hybrid" (default), "vector", "keyword".

When person names are detected in the query, an additional ILIKE retrieval
leg searches for name mentions in chunk content and contributes to RRF
fusion as a third ranked list.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import asyncpg

from .keyword_retriever import KeywordRetriever
from .retriever import RetrievedChunk, Retriever, build_boarddocs_url

if TYPE_CHECKING:
    from .reranker import Reranker

logger = logging.getLogger(__name__)

# Standard RRF constant.  Higher k dampens rank differences;
# k=60 is the value from the original Cormack et al. paper.
RRF_K = 60


def recency_multiplier(
    meeting_date_str: str | None,
    reference_date: datetime.date | None = None,
    half_life_days: int = 365,
) -> float:
    """Compute a recency boost multiplier using half-life decay.

    Documents lose relevance gradually, not abruptly.  A 1-year half-life
    means a document from 1 year ago gets 50% of the boost, 2 years ago
    gets 25%, etc.  This keeps older documents findable while giving
    recent ones an edge when relevance scores are otherwise similar.

    Args:
        meeting_date_str: ISO-format date string (YYYY-MM-DD) or None.
        reference_date: The "now" date for age calculation (defaults to
            today at query time, not server start time).
        half_life_days: Days until the boost decays to 50%.

    Returns:
        Multiplier between 0.0 and 1.0.  1.0 for today, 0.5 at one
        half-life ago, 0.25 at two half-lives, etc.  Documents with
        unknown dates get 0.5 (neutral).
    """
    if not meeting_date_str:
        return 0.5

    try:
        meeting_date = datetime.date.fromisoformat(meeting_date_str)
    except (ValueError, TypeError):
        return 0.5

    if reference_date is None:
        reference_date = datetime.date.today()

    age_days = (reference_date - meeting_date).days
    if age_days < 0:
        return 1.0

    return math.pow(0.5, age_days / half_life_days)


# Personnel action verbs.  When person names are detected AND the query
# contains one of these, the FTS keyword search is modified to search
# for the person name only (dropping the action verb).  This prevents
# plainto_tsquery from AND-joining "Kalberg" AND "hired" when the actual
# chunk says "retirement".
ACTION_VERBS: set[str] = {
    "hired",
    "retired",
    "resigned",
    "terminated",
    "appointed",
    "promoted",
    "transferred",
    "contract",
    "retirement",
    "resignation",
    "hiring",
    "appointment",
}

# ILIKE query for name-based chunk retrieval.  Uses asyncpg parameterized
# placeholders ($1, $2, $3) — the '%' || $2 || '%' pattern is SQL-side
# string concatenation, NOT Python string interpolation.  This is the
# same pattern used in database_handler.py and main.py.
_NAME_SEARCH_SQL = """\
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
    1.0 AS rank
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.tenant_id = $1
  AND c.content ILIKE '%' || $2 || '%'
ORDER BY d.meeting_date DESC NULLS LAST
LIMIT $3
"""


@dataclass
class FusedChunk:
    """A chunk annotated with per-source ranks and fused RRF score.

    Uses generic source_ranks / source_scores dicts so the fusion
    function handles any number of retrieval legs without dataclass
    changes.  The legacy vector_rank / keyword_rank accessors are
    preserved as properties for backward compatibility with logging.
    """

    chunk: RetrievedChunk
    rrf_score: float
    source_ranks: dict[str, int] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)

    @property
    def vector_rank(self) -> int | None:
        """Return rank from vector search, or None if not in that list."""
        return self.source_ranks.get("vector")

    @property
    def vector_score(self) -> float | None:
        """Return score from vector search, or None if not in that list."""
        return self.source_scores.get("vector")

    @property
    def keyword_rank(self) -> int | None:
        """Return rank from keyword search, or None if not in that list."""
        return self.source_ranks.get("keyword")

    @property
    def keyword_score(self) -> float | None:
        """Return score from keyword search, or None if not in that list."""
        return self.source_scores.get("keyword")


def reciprocal_rank_fusion(
    result_lists: list[tuple[str, list[RetrievedChunk]]],
    k: int = RRF_K,
) -> list[FusedChunk]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    For each chunk, the RRF score is the sum of 1/(k + rank) across
    all lists in which it appears.  Chunks appearing in multiple lists
    receive contributions from each, naturally ranking highest.

    Args:
        result_lists: Pairs of (source_name, ranked_chunks).
            Example: [("vector", vec_results), ("keyword", kw_results)]
            or with a third ILIKE leg:
            [("vector", vec), ("keyword", kw), ("ilike", name_results)]
        k: Smoothing constant (default 60).

    Returns:
        List of FusedChunk sorted by rrf_score descending.
    """
    # Map chunk_id -> FusedChunk for merging.
    fused: dict[str, FusedChunk] = {}

    for source_name, chunks in result_lists:
        for rank, chunk in enumerate(chunks, start=1):
            score = 1.0 / (k + rank)
            if chunk.chunk_id in fused:
                fused[chunk.chunk_id].rrf_score += score
                fused[chunk.chunk_id].source_ranks[source_name] = rank
                fused[chunk.chunk_id].source_scores[source_name] = chunk.score
            else:
                fused[chunk.chunk_id] = FusedChunk(
                    chunk=chunk,
                    rrf_score=score,
                    source_ranks={source_name: rank},
                    source_scores={source_name: chunk.score},
                )

    result = sorted(fused.values(), key=lambda f: f.rrf_score, reverse=True)

    # Log overlap statistics for debugging.
    source_names = [name for name, _ in result_lists]
    source_counts = {name: len(chunks) for name, chunks in result_lists}
    multi = sum(1 for f in result if len(f.source_ranks) > 1)
    parts = " + ".join(f"{source_counts[n]} {n}" for n in source_names)
    logger.info(
        "RRF fusion: %s -> %d fused (%d in multiple lists)",
        parts,
        len(result),
        multi,
    )

    return result


class HybridRetriever:
    """Orchestrates vector + keyword search with RRF fusion."""

    def __init__(
        self,
        vector_retriever: Retriever,
        keyword_retriever: KeywordRetriever,
        embedder,
        db_pool: asyncpg.Pool | None = None,
        reranker: Reranker | None = None,
        recency_weight: float = 0.3,
        recency_half_life_days: int = 365,
    ) -> None:
        """Initialize with both retrievers and the query embedder.

        Args:
            vector_retriever: Qdrant semantic search.
            keyword_retriever: PostgreSQL FTS search.
            embedder: Query embedding model.
            db_pool: asyncpg pool for ILIKE name searches.  If None,
                the pool is obtained from the keyword retriever.
            reranker: Optional cross-encoder reranker for post-RRF
                rescoring.  When available and ready, replaces RRF
                scores with cross-encoder relevance scores.
            recency_weight: How much recency affects final ranking.
                0.0 disables the boost; 1.0 means recency can add
                up to 100% to a result's RRF score.
            recency_half_life_days: Half-life for the decay curve.
                365 means a 1-year-old document gets 50% of the boost.
        """
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._embedder = embedder
        self._pool = db_pool if db_pool is not None else keyword_retriever._pool
        self._boarddocs_base_url = keyword_retriever._boarddocs_base_url
        self._reranker = reranker
        self._recency_weight = recency_weight
        self._recency_half_life_days = recency_half_life_days

    async def _name_search(
        self,
        person_names: list[str],
        top_k: int = 20,
        tenant_id: str = "kent_sd",
        excluded_doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Search for person names in chunk content using ILIKE.

        Runs one parameterized ILIKE query per detected name, then
        merges results (deduplicated by chunk_id, first occurrence wins).

        Args:
            person_names: List of person name strings from NER.
            top_k: Max results per name query.
            tenant_id: Tenant scope.
            excluded_doc_types: Document types to exclude from results.

        Returns:
            Combined list of RetrievedChunk from all name queries.
        """
        if not person_names:
            return []

        # Inject exclusion clause before ORDER BY when doc types are excluded.
        sql = _NAME_SEARCH_SQL
        extra_params: list = []
        if excluded_doc_types:
            sql = sql.replace(
                "ORDER BY d.meeting_date DESC",
                "  AND d.document_type != ALL($4)\nORDER BY d.meeting_date DESC",
            )
            extra_params = [excluded_doc_types]

        seen_ids: set[str] = set()
        combined: list[RetrievedChunk] = []

        async with self._pool.acquire() as conn:
            for name in person_names:
                try:
                    rows = await conn.fetch(
                        sql,
                        tenant_id,
                        name,
                        top_k,
                        *extra_params,
                    )
                except Exception as exc:
                    logger.warning(
                        "ILIKE name search failed for %r: %s",
                        name,
                        exc,
                    )
                    continue

                for row in rows:
                    chunk_id = str(row["chunk_id"])
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)

                    meeting_id = row["meeting_id"]
                    agenda_item_id = row["agenda_item_id"]
                    meeting_date = str(row["meeting_date"]) if row["meeting_date"] else None

                    source_url = (
                        build_boarddocs_url(
                            self._boarddocs_base_url,
                            meeting_id,
                            agenda_item_id,
                        )
                        if self._boarddocs_base_url
                        else None
                    )

                    combined.append(
                        RetrievedChunk(
                            chunk_id=chunk_id,
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
            "ILIKE name search: %d results for %d names %r",
            len(combined),
            len(person_names),
            person_names,
        )
        return combined

    @staticmethod
    def _build_keyword_query(
        query: str,
        person_names: list[str] | None,
    ) -> str:
        """Optionally strip action verbs from the keyword query.

        When person names are detected AND the query contains personnel
        action verbs (hired, retired, resigned, etc.), the keyword search
        is modified to search for the person name tokens only.  This
        prevents plainto_tsquery from AND-joining name + action verb when
        the actual chunk uses a different verb (e.g., query says "hired"
        but chunk says "retirement").

        When person names are detected but NO action verbs are present,
        or when no person names are detected at all, the original query
        is returned unchanged.
        """
        if not person_names:
            return query

        words = query.split()
        has_action_verb = any(w.lower() in ACTION_VERBS for w in words)

        if not has_action_verb:
            return query

        # Build a query from only the person name tokens (dropping
        # action verbs and filler words).
        name_tokens: set[str] = set()
        for name in person_names:
            for token in name.split():
                name_tokens.add(token.lower())

        # Keep only words that are part of a detected person name.
        filtered = [w for w in words if w.lower() in name_tokens]

        if not filtered:
            # Safety: if stripping removed everything, fall back to
            # original to avoid an empty tsquery.
            return query

        modified = " ".join(filtered)
        logger.info(
            "Keyword query modified for name search: %r -> %r",
            query,
            modified,
        )
        return modified

    async def search(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: str = "kent_sd",
        mode: str = "hybrid",
        date_from: str | None = None,
        date_to: str | None = None,
        document_type: str | None = None,
        committee_name: str | None = None,
        person_names: list[str] | None = None,
        excluded_doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Run retrieval in the specified mode and return top_k chunks.

        Args:
            query: The user's natural language question.
            top_k: Number of results to return after fusion.
            tenant_id: Tenant scope.
            mode: "hybrid" (default), "vector", or "keyword".
            date_from: Optional ISO date lower bound.
            date_to: Optional ISO date upper bound.
            document_type: Optional filter.
            committee_name: Optional filter.
            person_names: Person names detected by NER (from router).
                When provided, an additional ILIKE retrieval leg is
                added to RRF fusion and FTS keyword-stripping is applied.
            excluded_doc_types: Document types to exclude from all
                retrieval legs (e.g. types without verifiable source URLs).

        Returns:
            List of RetrievedChunk ordered by relevance.
        """
        # Fetch more from each source than we need post-fusion so that
        # proper-noun chunks that rank low in one list can still enter
        # the fusion pool.
        fetch_k = max(top_k, 20)

        if mode == "keyword":
            kw_query = self._build_keyword_query(query, person_names)
            kw_results = await self._keyword.search(
                kw_query,
                top_k=fetch_k,
                tenant_id=tenant_id,
                excluded_doc_types=excluded_doc_types,
            )
            return kw_results[:top_k]

        # Embed the query (needed for vector and hybrid modes).
        query_vector = await asyncio.to_thread(self._embedder.embed_query, query)

        if mode == "vector":
            vec_results = await asyncio.to_thread(
                self._vector.search,
                query_vector=query_vector,
                top_k=fetch_k,
                tenant_id=tenant_id,
                date_from=date_from,
                date_to=date_to,
                document_type=document_type,
                committee_name=committee_name,
                excluded_doc_types=excluded_doc_types,
            )
            return vec_results[:top_k]

        # Hybrid: run vector + keyword concurrently, optionally add
        # ILIKE name search as a third retrieval leg.
        kw_query = self._build_keyword_query(query, person_names)

        vec_task = asyncio.to_thread(
            self._vector.search,
            query_vector=query_vector,
            top_k=fetch_k,
            tenant_id=tenant_id,
            date_from=date_from,
            date_to=date_to,
            document_type=document_type,
            committee_name=committee_name,
            excluded_doc_types=excluded_doc_types,
        )
        kw_task = self._keyword.search(
            kw_query,
            top_k=fetch_k,
            tenant_id=tenant_id,
            excluded_doc_types=excluded_doc_types,
        )

        # If person names were detected, run ILIKE search concurrently.
        if person_names:
            name_task = self._name_search(
                person_names,
                top_k=fetch_k,
                tenant_id=tenant_id,
                excluded_doc_types=excluded_doc_types,
            )
            vec_results, kw_results, name_results = await asyncio.gather(
                vec_task,
                kw_task,
                name_task,
            )

            fused = reciprocal_rank_fusion(
                [
                    ("vector", vec_results),
                    ("keyword", kw_results),
                    ("ilike", name_results),
                ]
            )
        else:
            vec_results, kw_results = await asyncio.gather(vec_task, kw_task)

            fused = reciprocal_rank_fusion(
                [
                    ("vector", vec_results),
                    ("keyword", kw_results),
                ]
            )

        # Apply recency boost when enabled and the user has not already
        # scoped their search temporally with date filters.  This runs
        # before reranker pool selection so newer documents are more
        # likely to enter the cross-encoder's candidate set.
        apply_recency = self._recency_weight > 0 and date_from is None and date_to is None

        if apply_recency:
            ref_date = datetime.date.today()
            for f in fused:
                mult = recency_multiplier(
                    f.chunk.meeting_date,
                    reference_date=ref_date,
                    half_life_days=self._recency_half_life_days,
                )
                f.rrf_score = f.rrf_score * (1.0 + self._recency_weight * mult)
            fused.sort(key=lambda f: f.rrf_score, reverse=True)

        # Cross-encoder reranking: the single highest-impact retrieval
        # quality improvement.  Runs after RRF fusion so the cross-encoder
        # only scores the most promising candidates (typically 20),
        # keeping latency bounded at ~3-4s on CPU.  The cross-encoder
        # jointly attends to query + passage text, catching nuanced
        # semantic relationships that bi-encoder similarity misses.
        if self._reranker is not None and self._reranker.ready:
            rerank_pool = fused[: max(top_k * 2, 20)]
            passages = [f.chunk.content for f in rerank_pool]

            t0 = datetime.datetime.now()
            reranked = await asyncio.to_thread(self._reranker.rerank, query, passages, top_k=top_k)
            elapsed_ms = (datetime.datetime.now() - t0).total_seconds() * 1000

            logger.info(
                "Cross-encoder reranked %d candidates to %d in %.0fms",
                len(rerank_pool),
                len(reranked),
                elapsed_ms,
            )

            results = []
            for orig_idx, ce_score in reranked:
                chunk = rerank_pool[orig_idx].chunk
                # Cross-encoder score replaces RRF score as the
                # authoritative relevance signal.
                chunk.score = ce_score
                results.append(chunk)

            return results

        # Fallback when reranker is not available: use RRF scores directly.
        results = []
        for f in fused[:top_k]:
            chunk = f.chunk
            chunk.score = f.rrf_score
            results.append(chunk)

        return results
