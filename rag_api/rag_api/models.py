"""Pydantic v2 request/response models for the RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Query endpoint ---


class QueryRequest(BaseModel):
    """Incoming query with optional filters and retrieval settings."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=25,
        description="Number of chunks to retrieve",
    )
    date_from: str | None = Field(
        default=None,
        description="Filter: earliest meeting date (YYYY-MM-DD)",
    )
    date_to: str | None = Field(
        default=None,
        description="Filter: latest meeting date (YYYY-MM-DD)",
    )
    document_type: str | None = Field(
        default=None,
        description="Filter: agenda_item, agenda, attachment",
    )
    committee_name: str | None = Field(
        default=None,
        description="Filter: committee name",
    )
    rewrite_query: bool = Field(
        default=False,
        description="Rewrite query using technical vocabulary and decompose into sub-queries",
    )
    enable_routing: bool = Field(
        default=False,
        description=(
            "Enable intelligent query routing (rag/database/hybrid).  "
            "Disabled by default: the 'database' route runs LLM-generated "
            "SQL against the database and should only be enabled behind an "
            "authenticated admin surface with a read-only database role and "
            "an AST-based SQL allow-list.  See database_handler.py."
        ),
    )
    retrieval_mode: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'hybrid' (vector+keyword RRF), 'vector', or 'keyword'",
    )


class Citation(BaseModel):
    """A single source citation referencing a retrieved chunk."""

    source_number: int = Field(
        description="Source number as referenced in the answer [Source N]",
    )
    chunk_id: str
    document_id: str
    title: str | None = None
    meeting_date: str | None = None
    committee_name: str | None = None
    document_type: str | None = None
    source_url: str | None = None
    relevance_score: float = Field(
        description="Relevance score (RRF in hybrid mode, cosine in vector, ts_rank in keyword)",
    )


class QueryResponse(BaseModel):
    """RAG query response with answer, citations, and timing metadata."""

    answer: str = Field(
        description="The LLM-generated answer with [Source N] citations",
    )
    citations: list[Citation] = Field(
        description="Ordered list of source documents",
    )
    query: str = Field(description="The original query")
    chunks_retrieved: int
    model: str = Field(description="LLM model used")
    input_tokens: int
    output_tokens: int
    latency_seconds: float = Field(description="Total end-to-end latency")
    embedding_latency_seconds: float
    retrieval_latency_seconds: float
    llm_latency_seconds: float
    sub_queries: list[str] | None = Field(
        default=None,
        description="Sub-queries used if rewrite enabled",
    )
    rewrite_reasoning: str | None = Field(
        default=None,
        description="Rewriting reasoning if rewrite enabled",
    )
    routing_decision: dict | None = Field(
        default=None,
        description="Routing classification result if routing enabled: {route, confidence, reasoning}",
    )


# --- Documents endpoint ---


class DocumentSummary(BaseModel):
    """Summary of a document for list endpoints."""

    document_id: str
    title: str | None
    document_type: str
    meeting_date: str | None
    committee_name: str | None
    source_url: str | None
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    documents: list[DocumentSummary]
    total: int
    offset: int
    limit: int


class DocumentDetail(BaseModel):
    """Full document detail including content text."""

    document_id: str
    title: str | None
    document_type: str
    meeting_date: str | None
    committee_name: str | None
    source_url: str | None
    content_text: str | None
    chunk_count: int
    created_at: str


# --- Health endpoint ---


class HealthResponse(BaseModel):
    """Health check response for all system dependencies."""

    status: str = Field(description="healthy or degraded")
    database: bool = Field(description="PostgreSQL connection OK")
    qdrant: bool = Field(description="Qdrant connection OK")
    embedder: bool = Field(description="ONNX embedder loaded")
    reranker: bool = Field(default=False, description="Cross-encoder reranker loaded")
    qdrant_collection_count: int = Field(
        default=0,
        description="Points in Qdrant collection",
    )
    version: str = "0.1.0"
