"""Tests for prompt template construction."""

from __future__ import annotations

from rag_api.prompts import SYSTEM_PROMPT, build_context_block, build_user_message
from rag_api.retriever import RetrievedChunk


def test_system_prompt_contains_citation_instruction():
    assert "[Source N]" in SYSTEM_PROMPT
    assert "ONLY" in SYSTEM_PROMPT


def test_system_prompt_contains_insufficient_info_instruction():
    assert "I don't have enough information" in SYSTEM_PROMPT


def test_build_context_block_format(sample_chunks):
    result = build_context_block(sample_chunks)

    assert "[Source 1]" in result
    assert "[Source 2]" in result
    assert "[Source 3]" in result
    assert "Title: Second Reading and Approval of Policy 3210" in result
    assert "Date: 2025-12-10" in result
    assert "Committee: Regular Meeting" in result
    assert "---" in result


def test_build_context_block_includes_content(sample_chunks):
    result = build_context_block(sample_chunks)

    assert "Policy 3210" in result
    assert "budget" in result
    assert "enrollment" in result


def test_build_context_block_includes_urls(sample_chunks):
    result = build_context_block(sample_chunks)

    assert "URL: https://go.boarddocs.com/wa/ksdwa/Board.nsf/goto?open&id=AAA" in result


def test_build_context_block_handles_missing_metadata():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            content="Some content here.",
            score=0.9,
            title=None,
            meeting_date=None,
            committee_name=None,
            document_type="agenda_item",
            source_url=None,
            chunk_index=0,
            meeting_id=None,
            agenda_item_id=None,
        ),
    ]
    result = build_context_block(chunks)

    assert "[Source 1]" in result
    assert "Some content here." in result
    # No metadata lines beyond the source number
    assert "Title:" not in result
    assert "Date:" not in result


def test_build_context_block_empty():
    result = build_context_block([])
    assert result == ""


def test_build_user_message_includes_query_and_context():
    context = "[Source 1]\nSome context"
    result = build_user_message("What is the policy?", context)

    assert "Context documents:" in result
    assert "Some context" in result
    assert "Question: What is the policy?" in result
