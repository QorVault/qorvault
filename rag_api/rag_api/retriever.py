"""Qdrant vector search for document retrieval.

Uses the REST API directly for compatibility with Qdrant server v1.9,
since qdrant-client >= 1.12 dropped the search() method in favor of
query_points() which requires Qdrant server >= 1.10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned from Qdrant search."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    title: str | None
    meeting_date: str | None
    committee_name: str | None
    document_type: str
    source_url: str | None
    chunk_index: int
    meeting_id: str | None
    agenda_item_id: str | None


def build_boarddocs_url(
    base_url: str,
    meeting_id: str | None = None,
    agenda_item_id: str | None = None,
) -> str | None:
    """Construct a BoardDocs goto permalink from payload IDs.

    Prefers agenda_item_id (links to the specific item) over meeting_id
    (links to the full meeting agenda).  Returns None if both are missing.
    """
    doc_id = agenda_item_id or meeting_id
    if not doc_id:
        return None
    return f"{base_url.rstrip('/')}/goto?open&id={doc_id}"


_ALLOWED_URL_SCHEMES = ("https://", "http://")


def validate_url_scheme(url: str | None) -> str | None:
    """Return the URL only if it uses an allowed HTTP(S) scheme.

    Rejects javascript:, data:, vbscript:, and any other non-HTTP scheme
    to prevent stored XSS from untrusted payload URLs.  Returns None and
    logs a warning for rejected URLs.
    """
    if not url:
        return None
    if url.lower().startswith(_ALLOWED_URL_SCHEMES):
        return url
    logger.warning("Rejected URL with disallowed scheme: %.120s", url)
    return None


class Retriever:
    """Search Qdrant for semantically similar chunks."""

    def __init__(self, qdrant_url: str, collection: str, boarddocs_base_url: str = "") -> None:
        """Initialize with Qdrant connection and optional BoardDocs base URL."""
        self._base_url = qdrant_url.rstrip("/")
        self._collection = collection
        self._boarddocs_base_url = boarddocs_base_url
        self._http = httpx.Client(timeout=30.0)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        tenant_id: str = "kent_sd",
        date_from: str | None = None,
        date_to: str | None = None,
        document_type: str | None = None,
        committee_name: str | None = None,
        excluded_doc_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Search Qdrant and return ranked chunks."""
        must_conditions: list[dict] = [
            {"key": "tenant_id", "match": {"value": tenant_id}},
        ]

        if document_type:
            must_conditions.append(
                {"key": "document_type", "match": {"value": document_type}},
            )

        if committee_name:
            must_conditions.append(
                {"key": "committee_name", "match": {"value": committee_name}},
            )

        # Exclude unverifiable document types (e.g. transcripts without source URLs).
        must_not_conditions: list[dict] = []
        if excluded_doc_types:
            for doc_type in excluded_doc_types:
                must_not_conditions.append(
                    {"key": "document_type", "match": {"value": doc_type}},
                )

        # When date filtering is active, fetch extra results for post-filtering.
        # Qdrant Range only accepts numeric values; meeting_date is stored as
        # an ISO string in the payload, so we filter in Python.
        has_date_filter = date_from is not None or date_to is not None
        fetch_limit = top_k * 3 if has_date_filter else top_k

        qdrant_filter: dict = {"must": must_conditions}
        if must_not_conditions:
            qdrant_filter["must_not"] = must_not_conditions

        body = {
            "vector": query_vector,
            "filter": qdrant_filter,
            "limit": fetch_limit,
            "with_payload": True,
        }

        url = f"{self._base_url}/collections/{self._collection}/points/search"
        resp = self._http.post(url, json=body)
        resp.raise_for_status()
        results = resp.json().get("result", [])

        chunks = []
        for point in results:
            p = point.get("payload") or {}
            meeting_date = p.get("meeting_date")

            # Post-retrieval date filtering on ISO date strings.
            if date_from and (not meeting_date or meeting_date < date_from):
                continue
            if date_to and (not meeting_date or meeting_date > date_to):
                continue

            meeting_id = p.get("meeting_id")
            agenda_item_id = p.get("agenda_item_id")

            # Construct BoardDocs permalink from IDs when possible;
            # fall back to the URL stored in the Qdrant payload (e.g.
            # YouTube URLs on transcript chunks).
            source_url = None
            if self._boarddocs_base_url:
                source_url = build_boarddocs_url(
                    self._boarddocs_base_url,
                    meeting_id,
                    agenda_item_id,
                )
            if source_url is None:
                source_url = validate_url_scheme(p.get("source_url"))

            chunks.append(
                RetrievedChunk(
                    chunk_id=p.get("chunk_id", ""),
                    document_id=p.get("document_id", ""),
                    content=p.get("content", ""),
                    score=point.get("score", 0.0),
                    title=p.get("title"),
                    meeting_date=meeting_date,
                    committee_name=p.get("committee_name"),
                    document_type=p.get("document_type", ""),
                    source_url=source_url,
                    chunk_index=p.get("chunk_index", 0),
                    meeting_id=meeting_id,
                    agenda_item_id=agenda_item_id,
                )
            )

            if len(chunks) >= top_k:
                break

        logger.info(
            "Retrieved %d chunks (top score: %.4f)",
            len(chunks),
            chunks[0].score if chunks else 0.0,
        )
        return chunks

    def get_collection_info(self) -> dict:
        """Get collection info for health checks."""
        url = f"{self._base_url}/collections/{self._collection}"
        resp = self._http.get(url)
        resp.raise_for_status()
        return resp.json().get("result", {})

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
