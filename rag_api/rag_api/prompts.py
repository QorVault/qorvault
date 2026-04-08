"""Prompt templates for the RAG pipeline."""

from __future__ import annotations

from .retriever import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a research assistant specializing in Kent School District public records. "
    "You answer questions about school board meetings, policies, budgets, and district "
    "operations using ONLY the provided source documents.\n\n"
    "Rules:\n"
    "1. Answer ONLY based on the provided context documents. Do not use prior knowledge.\n"
    "2. Cite every claim using [Source N] notation, where N corresponds to the source number.\n"
    "3. If the context does not contain enough information to answer, say: "
    '"I don\'t have enough information in the available documents to answer this question."\n'
    "4. When multiple sources support a point, cite all of them: [Source 1][Source 3].\n"
    '5. Present dates in a human-readable format (e.g., "December 10, 2025").\n'
    "6. If information conflicts between sources, note the discrepancy and cite both.\n"
    "7. Be concise but thorough. Prefer direct quotes from the documents when appropriate."
)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Build the context block for the LLM prompt.

    Each chunk is labeled [Source N] with metadata and content.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header_parts = [f"[Source {i}]"]
        if chunk.title:
            header_parts.append(f"Title: {chunk.title}")
        if chunk.meeting_date:
            header_parts.append(f"Date: {chunk.meeting_date}")
        if chunk.committee_name:
            header_parts.append(f"Committee: {chunk.committee_name}")
        if chunk.source_url:
            header_parts.append(f"URL: {chunk.source_url}")

        header = "\n".join(header_parts)
        parts.append(f"{header}\n\n{chunk.content}")

    return "\n\n---\n\n".join(parts)


def build_user_message(query: str, context: str) -> str:
    """Build the user message containing context + query."""
    return f"Context documents:\n\n{context}\n\n" f"---\n\n" f"Question: {query}"
