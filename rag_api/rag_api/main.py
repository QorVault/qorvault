"""FastAPI application for the BoardDocs RAG API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date as date_type
from pathlib import Path
from uuid import UUID

import asyncpg
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings
from .database_handler import execute_database_query, execute_scoped_database_query
from .embedder import Embedder
from .hybrid_retriever import HybridRetriever
from .keyword_retriever import KeywordRetriever
from .llm import LLMClient, LLMResponse
from .models import (
    Citation,
    DocumentDetail,
    DocumentListResponse,
    DocumentSummary,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from .prompts import SYSTEM_PROMPT, build_context_block, build_user_message
from .reranker import Reranker
from .retriever import Retriever
from .rewriter import rewrite_query as _rewrite_query
from .router import classify_query

logger = logging.getLogger(__name__)

# Ensure application loggers have a handler when started via uvicorn CLI
# (uvicorn only configures its own loggers, not application-level ones).
# basicConfig is a no-op if the root logger already has handlers (e.g.,
# when started via `python -m rag_api` which calls basicConfig first).
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Module-level state (single worker only).
settings = Settings()
embedder = Embedder(cache_dir=settings.model_cache_dir)
reranker: Reranker | None = None
retriever: Retriever | None = None
hybrid_retriever: HybridRetriever | None = None
llm_client: LLMClient | None = None
db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    global retriever, hybrid_retriever, llm_client, db_pool  # noqa: PLW0603

    # Resolve API key: check RAG_ANTHROPIC_API_KEY, then ANTHROPIC_API_KEY
    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # Load embedder (blocks briefly on startup — ~2s for ONNX model)
    logger.info("Loading embedding model...")
    embedder.load()

    # Load cross-encoder reranker (Phase 1 retrieval improvement).
    # Adds ~3-4s latency per query but significantly improves
    # precision by jointly scoring query+passage pairs.  Degrades
    # gracefully: if loading fails, retrieval continues without reranking.
    global reranker  # noqa: PLW0603
    if settings.reranker_enabled:
        try:
            reranker = Reranker(
                cache_dir=settings.reranker_cache_dir,
                model_filename=settings.reranker_model_filename,
                expected_sha256=settings.reranker_model_sha256,
            )
            reranker.load()
            logger.info("Cross-encoder reranker loaded successfully")
        except Exception as exc:
            logger.warning("Reranker failed to load, continuing without it: %s", exc)
            reranker = None

    # Initialize Qdrant retriever
    retriever = Retriever(
        settings.qdrant_url,
        settings.qdrant_collection,
        boarddocs_base_url=settings.boarddocs_base_url,
    )

    # Initialize Anthropic client
    llm_client = LLMClient(api_key=api_key, model=settings.anthropic_model)

    # Initialize database pool
    db_pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=1,
        max_size=5,
    )

    # Initialize hybrid retriever (vector + keyword search with RRF)
    keyword_ret = KeywordRetriever(db_pool, boarddocs_base_url=settings.boarddocs_base_url)
    hybrid_retriever = HybridRetriever(
        retriever,
        keyword_ret,
        embedder,
        db_pool=db_pool,
        reranker=reranker,
        recency_weight=settings.recency_weight,
        recency_half_life_days=settings.recency_half_life_days,
    )

    logger.info(
        "RAG API ready (model=%s, collection=%s, hybrid_search=enabled)",
        settings.anthropic_model,
        settings.qdrant_collection,
    )
    yield

    # Shutdown
    logger.info("Shutting down RAG API...")
    if db_pool:
        await db_pool.close()
    if retriever:
        retriever.close()


app = FastAPI(
    title="BoardDocs RAG API",
    description="Retrieval-augmented generation for Kent School District public records",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Static files (query UI)
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
async def root():
    """Serve the query interface."""
    return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# POST /api/v1/query
# ---------------------------------------------------------------------------


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using RAG over BoardDocs documents."""
    t0 = time.time()
    sub_queries: list[str] | None = None
    rewrite_reasoning: str | None = None
    routing_decision: dict | None = None
    route = "rag"
    person_names: list[str] | None = None

    # 0a. Optional query routing
    if request.enable_routing:
        try:
            routing_result = await asyncio.to_thread(
                classify_query,
                llm_client._client,
                request.query,
            )
            route = routing_result["route"]
            confidence = routing_result["confidence"]

            # Fall back to RAG if confidence below threshold
            if confidence < 0.7:
                route = "rag"
                routing_result["reasoning"] += f" (confidence {confidence} < 0.7, falling back to rag)"

            routing_decision = routing_result

            # Extract person names from routing for name-aware retrieval
            extracted_filters = routing_result.get("extracted_filters", {})
            detected_names = extracted_filters.get("person_names", [])
            if detected_names:
                person_names = detected_names
                logger.info("Detected person names for retrieval: %r", person_names)
        except Exception as exc:
            logger.warning("Query routing failed, falling back to rag: %s", exc)
            routing_decision = {
                "route": "rag",
                "confidence": 0.0,
                "reasoning": f"routing_error: {exc}",
            }
            route = "rag"
            # Routing failed but we still want NER-based name search.
            # Run spaCy extraction directly so the ILIKE leg can fire.
            try:
                from .router import _extract_person_names

                fallback_names = _extract_person_names(request.query)
                if fallback_names:
                    person_names = fallback_names
                    logger.info(
                        "Extracted person names despite routing failure: %r",
                        person_names,
                    )
            except Exception as ner_exc:
                logger.debug("Fallback NER also failed: %s", ner_exc)

    if route == "database":
        # Direct SQL path — skip embedding entirely
        db_result = await execute_database_query(
            db_pool,
            request.query,
            llm_client._client,
            routing_result.get("extracted_filters", {}),
        )
        if db_result.get("error"):
            # SQL generation/validation failed — fall back to RAG
            route = "rag"
            routing_decision["reasoning"] += f" -> SQL failed: {db_result['error']}, falling back to rag"
        else:
            return QueryResponse(
                answer=db_result["answer"],
                citations=[],
                query=request.query,
                chunks_retrieved=0,
                model=settings.anthropic_model,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=round(time.time() - t0, 2),
                embedding_latency_seconds=0.0,
                retrieval_latency_seconds=0.0,
                llm_latency_seconds=0.0,
                routing_decision=routing_decision,
            )

    # 0b. Optional query rewriting and decomposition
    if request.rewrite_query:
        try:
            rewrite_result = await asyncio.to_thread(
                _rewrite_query,
                llm_client._client,
                request.query,
            )
        except Exception as exc:
            logger.warning("Query rewrite failed, using original: %s", exc)
            rewrite_result = {
                "rewritten_query": request.query,
                "sub_queries": [],
                "reasoning": f"rewrite_error: {exc}",
            }

        all_queries = [rewrite_result["rewritten_query"]] + rewrite_result["sub_queries"]
        sub_queries = rewrite_result["sub_queries"]
        rewrite_reasoning = rewrite_result["reasoning"]

        logger.info(
            "Query rewrite: %d total queries — %s",
            len(all_queries),
            all_queries,
        )

        # Run hybrid retrieve for each query, dedup by chunk_id
        t_retrieve = time.time()
        all_chunks = []
        seen_ids: set[str] = set()

        for q in all_queries:
            try:
                results = await hybrid_retriever.search(
                    query=q,
                    top_k=request.top_k,
                    tenant_id=settings.tenant_id,
                    mode=request.retrieval_mode,
                    date_from=request.date_from,
                    date_to=request.date_to,
                    document_type=request.document_type,
                    committee_name=request.committee_name,
                    person_names=person_names,
                    excluded_doc_types=settings.excluded_document_types,
                )
            except Exception as exc:
                logger.warning("Retrieval failed for sub-query '%s': %s", q[:50], exc)
                continue

            for chunk in results:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    all_chunks.append(chunk)

        embedding_latency = 0.0  # Handled inside hybrid_retriever
        retrieval_latency = time.time() - t_retrieve

        # Sort by score descending
        chunks = sorted(all_chunks, key=lambda c: c.score, reverse=True)

    else:
        # Standard single-query path using hybrid retriever
        t_retrieve = time.time()
        try:
            chunks = await hybrid_retriever.search(
                query=request.query,
                top_k=request.top_k,
                tenant_id=settings.tenant_id,
                mode=request.retrieval_mode,
                date_from=request.date_from,
                date_to=request.date_to,
                document_type=request.document_type,
                committee_name=request.committee_name,
                person_names=person_names,
                excluded_doc_types=settings.excluded_document_types,
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")
        embedding_latency = 0.0  # Handled inside hybrid_retriever
        retrieval_latency = time.time() - t_retrieve

    if not chunks:
        return QueryResponse(
            answer=("I don't have enough information in the available documents to answer this question."),
            citations=[],
            query=request.query,
            chunks_retrieved=0,
            model=settings.anthropic_model,
            input_tokens=0,
            output_tokens=0,
            latency_seconds=round(time.time() - t0, 2),
            embedding_latency_seconds=round(embedding_latency, 2),
            retrieval_latency_seconds=round(retrieval_latency, 2),
            llm_latency_seconds=0.0,
            sub_queries=sub_queries,
            rewrite_reasoning=rewrite_reasoning,
            routing_decision=routing_decision,
        )

    # 3. Build prompt
    context = build_context_block(chunks)

    # 3a. Hybrid route: add scoped database results to context
    if route == "hybrid":
        doc_ids = list({chunk.document_id for chunk in chunks})
        try:
            db_extra = await execute_scoped_database_query(
                db_pool,
                request.query,
                llm_client._client,
                routing_decision.get("extracted_filters", {}),
                doc_ids,
            )
            if db_extra:
                context += db_extra
        except Exception as exc:
            logger.warning("Hybrid database query failed: %s", exc)
    user_message = build_user_message(request.query, context)

    # 4. Call LLM
    t_llm = time.time()
    try:
        llm_response: LLMResponse = await asyncio.to_thread(
            llm_client.generate,
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=settings.max_response_tokens,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")
    llm_latency = time.time() - t_llm

    # 5. Build citations list
    citations = [
        Citation(
            source_number=i + 1,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            meeting_date=chunk.meeting_date,
            committee_name=chunk.committee_name,
            document_type=chunk.document_type or None,
            source_url=chunk.source_url,
            relevance_score=round(chunk.score, 4),
        )
        for i, chunk in enumerate(chunks)
    ]

    total_latency = time.time() - t0

    return QueryResponse(
        answer=llm_response.content,
        citations=citations,
        query=request.query,
        chunks_retrieved=len(chunks),
        model=llm_response.model,
        input_tokens=llm_response.input_tokens,
        output_tokens=llm_response.output_tokens,
        latency_seconds=round(total_latency, 2),
        embedding_latency_seconds=round(embedding_latency, 2),
        retrieval_latency_seconds=round(retrieval_latency, 2),
        llm_latency_seconds=round(llm_latency, 2),
        sub_queries=sub_queries,
        rewrite_reasoning=rewrite_reasoning,
        routing_decision=routing_decision,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------

DOCUMENTS_LIST_SQL = """
SELECT d.id, d.title, d.document_type, d.meeting_date,
       d.committee_name, d.source_url,
       COUNT(c.id) AS chunk_count
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
WHERE d.tenant_id = $1
  AND ($2::varchar IS NULL OR d.document_type = $2)
  AND ($3::date IS NULL OR d.meeting_date >= $3)
  AND ($4::date IS NULL OR d.meeting_date <= $4)
  AND ($5::varchar IS NULL OR d.committee_name ILIKE '%' || $5 || '%')
GROUP BY d.id
ORDER BY d.meeting_date DESC NULLS LAST, d.title
LIMIT $6 OFFSET $7
"""

DOCUMENTS_COUNT_SQL = """
SELECT COUNT(*) FROM documents d
WHERE d.tenant_id = $1
  AND ($2::varchar IS NULL OR d.document_type = $2)
  AND ($3::date IS NULL OR d.meeting_date >= $3)
  AND ($4::date IS NULL OR d.meeting_date <= $4)
  AND ($5::varchar IS NULL OR d.committee_name ILIKE '%' || $5 || '%')
"""


@app.get("/api/v1/documents", response_model=DocumentListResponse)
async def list_documents(
    document_type: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    committee: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List documents with optional filtering and pagination."""
    date_from_parsed = date_type.fromisoformat(date_from) if date_from else None
    date_to_parsed = date_type.fromisoformat(date_to) if date_to else None

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            DOCUMENTS_LIST_SQL,
            settings.tenant_id,
            document_type,
            date_from_parsed,
            date_to_parsed,
            committee,
            limit,
            offset,
        )
        total = await conn.fetchval(
            DOCUMENTS_COUNT_SQL,
            settings.tenant_id,
            document_type,
            date_from_parsed,
            date_to_parsed,
            committee,
        )

    documents = [
        DocumentSummary(
            document_id=str(row["id"]),
            title=row["title"],
            document_type=row["document_type"],
            meeting_date=str(row["meeting_date"]) if row["meeting_date"] else None,
            committee_name=row["committee_name"],
            source_url=row["source_url"],
            chunk_count=row["chunk_count"],
        )
        for row in rows
    ]

    return DocumentListResponse(
        documents=documents,
        total=total,
        offset=offset,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{document_id}
# ---------------------------------------------------------------------------

DOCUMENT_DETAIL_SQL = """
SELECT d.id, d.title, d.document_type, d.meeting_date,
       d.committee_name, d.source_url, d.content_text, d.created_at,
       COUNT(c.id) AS chunk_count
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
WHERE d.id = $1 AND d.tenant_id = $2
GROUP BY d.id
"""


@app.get("/api/v1/documents/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str):
    """Get a specific document by ID."""
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            DOCUMENT_DETAIL_SQL,
            doc_uuid,
            settings.tenant_id,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetail(
        document_id=str(row["id"]),
        title=row["title"],
        document_type=row["document_type"],
        meeting_date=str(row["meeting_date"]) if row["meeting_date"] else None,
        committee_name=row["committee_name"],
        source_url=row["source_url"],
        content_text=row["content_text"],
        chunk_count=row["chunk_count"],
        created_at=row["created_at"].isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """Check health of all dependencies."""
    db_ok = False
    qdrant_ok = False
    qdrant_count = 0

    # Check PostgreSQL
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        db_ok = True
    except Exception as exc:
        logger.warning("Health check: DB unreachable: %s", exc)

    # Check Qdrant
    try:
        info = retriever.get_collection_info()
        qdrant_ok = True
        qdrant_count = info.get("points_count", 0)
    except Exception as exc:
        logger.warning("Health check: Qdrant unreachable: %s", exc)

    status = "healthy" if (db_ok and qdrant_ok and embedder.ready) else "degraded"

    return HealthResponse(
        status=status,
        database=db_ok,
        qdrant=qdrant_ok,
        embedder=embedder.ready,
        reranker=reranker.ready if reranker else False,
        qdrant_collection_count=qdrant_count,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible Chat Completions API (for Open WebUI)
# ---------------------------------------------------------------------------


class _ChatMessage(BaseModel):
    role: str
    content: str = ""


class _ChatCompletionRequest(BaseModel):
    model: str = "boarddocs-rag"
    messages: list[_ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def _format_citations(chunks) -> str:
    """Format retrieved chunks as a markdown sources block."""
    if not chunks:
        return ""
    lines = ["\n\n---\n**Sources:**\n"]
    for i, chunk in enumerate(chunks, 1):
        parts = []
        if chunk.title:
            parts.append(chunk.title)
        meta = []
        if chunk.committee_name:
            meta.append(chunk.committee_name)
        if chunk.meeting_date:
            meta.append(chunk.meeting_date)
        if meta:
            parts.append(f"({', '.join(meta)})")
        if chunk.source_url:
            parts.append(f"— {chunk.source_url}")
        lines.append(f"- [Source {i}] {' '.join(parts)}")
    return "\n".join(lines)


async def _run_rag_pipeline(query: str) -> tuple[str, list]:
    """Run the full RAG pipeline, returning (answer_with_citations, chunks)."""
    # 1+2. Hybrid retrieve (embeds query internally)
    # Note: OpenAI-compatible endpoint does NOT use the router, so
    # person_names stays None (default).
    chunks = await hybrid_retriever.search(
        query=query,
        top_k=settings.default_top_k,
        tenant_id=settings.tenant_id,
        mode="hybrid",
        excluded_doc_types=settings.excluded_document_types,
    )

    if not chunks:
        return (
            "I don't have enough information in the available documents to answer this question.",
            [],
        )

    # 3. Build prompt
    context = build_context_block(chunks)
    user_message = build_user_message(query, context)

    # 4. Call LLM
    llm_response = await asyncio.to_thread(
        llm_client.generate,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=settings.max_response_tokens,
    )

    answer = llm_response.content + _format_citations(chunks)
    return answer, chunks


@app.post("/v1/chat/completions")
async def chat_completions(request: _ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint for Open WebUI."""
    # Extract last user message
    user_content = None
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content.strip():
            user_content = msg.content.strip()
            break

    if not user_content:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "No user message found in messages array",
                    "type": "invalid_request_error",
                    "code": 400,
                }
            },
        )

    chat_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    if not request.stream:
        # --- Non-streaming response ---
        try:
            answer, _chunks = await _run_rag_pipeline(user_content)
        except Exception as exc:
            logger.exception("RAG pipeline error in chat completions")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": str(exc),
                        "type": "internal_error",
                        "code": 500,
                    }
                },
            )

        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": "boarddocs-rag",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    # --- Streaming response (pseudo-stream) ---
    async def generate_stream():
        try:
            answer, _chunks = await _run_rag_pipeline(user_content)
        except Exception as exc:
            logger.exception("RAG pipeline error in chat completions (stream)")
            error_payload = {
                "error": {
                    "message": str(exc),
                    "type": "internal_error",
                    "code": 500,
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # First chunk: include role
        words = answer.split(" ")
        for i, word in enumerate(words):
            token = word if i == len(words) - 1 else word + " "
            delta = {"content": token}
            if i == 0:
                delta["role"] = "assistant"
            chunk_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "boarddocs-rag",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            await asyncio.sleep(0.01)

        # Final chunk: finish_reason stop
        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "boarddocs-rag",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/models")
async def list_models():
    """Return available models (for Open WebUI connection verification)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "boarddocs-rag",
                "object": "model",
                "created": 1700000000,
                "owned_by": "kent-school-district",
            }
        ],
    }


@app.get("/health")
async def health_simple():
    """Simple health check."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /ping — Cloudflare Tunnel health check target
# ---------------------------------------------------------------------------


@app.get("/ping")
async def ping():
    """Minimal liveness probe for Cloudflare Tunnel health checks.

    Returns HTTP 200 to confirm the Python process is alive. Intentionally
    does NOT query PostgreSQL, Qdrant, or any backend service — if those
    are down, the /api/v1/health endpoint reports the details. This
    endpoint exists solely so cloudflared can verify the origin is
    reachable without adding load to downstream services.
    """
    return {"status": "ok"}
