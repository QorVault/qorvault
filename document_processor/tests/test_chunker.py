"""Tests for tiktoken sentence-aware chunking."""

from __future__ import annotations

import pytest
import tiktoken

from document_processor.chunker import chunk_text


@pytest.fixture
def enc():
    return tiktoken.get_encoding("cl100k_base")


class TestChunkText:
    def test_basic_chunking_token_counts(self, long_text, enc):
        """Chunks should have correct token counts."""
        chunks = chunk_text(long_text)
        assert len(chunks) > 1
        for text, token_count in chunks:
            actual = len(enc.encode(text))
            assert token_count == actual

    def test_overlap_between_chunks(self, long_text, enc):
        """Consecutive chunks should share approximately 38 tokens of overlap."""
        chunks = chunk_text(long_text)
        assert len(chunks) >= 2

        # Check that end of chunk N overlaps with start of chunk N+1
        for i in range(len(chunks) - 1):
            tokens_a = set(enc.encode(chunks[i][0])[-50:])
            tokens_b = set(enc.encode(chunks[i + 1][0])[:50])
            # There should be some overlap (not exact due to sentence boundaries)
            overlap = tokens_a & tokens_b
            assert len(overlap) > 0, f"No overlap between chunk {i} and {i+1}"

    def test_small_chunks_discarded(self, enc):
        """Chunks below min_tokens are discarded when there are multiple."""
        # Two sentences: one long enough, one too short
        long_sentence = "Word " * 200 + "end."
        short_sentence = "Hi."
        text = long_sentence + " " + short_sentence
        chunks = chunk_text(text, target_tokens=300, min_tokens=100)
        for _, tc in chunks:
            assert tc >= 100

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == []

    def test_single_short_sentence(self, enc):
        """Single sentence below min_tokens still returns one chunk."""
        text = "Hello world."
        chunks = chunk_text(text, min_tokens=100)
        assert len(chunks) == 1
        assert chunks[0][0] == "Hello world."
        assert chunks[0][1] == len(enc.encode("Hello world."))

    def test_long_text_multiple_chunks(self, long_text):
        """Long text produces multiple chunks."""
        chunks = chunk_text(long_text)
        assert len(chunks) >= 3

    def test_sentence_boundaries_respected(self, enc):
        """Chunks should end at sentence boundaries, not mid-sentence."""
        text = (
            "First sentence here. Second sentence follows. "
            "Third sentence now. Fourth sentence added. "
            "Fifth sentence more. Sixth sentence final."
        )
        chunks = chunk_text(text, target_tokens=20, overlap_tokens=5, min_tokens=1)
        for chunk_text_str, _ in chunks:
            # Each chunk should end with a period (sentence boundary)
            assert chunk_text_str.rstrip().endswith(".")

    def test_chunk_sizes_near_target(self, long_text, enc):
        """Most chunks should be near the target token count."""
        chunks = chunk_text(long_text, target_tokens=384)
        # All chunks except possibly the last should be >= 200 tokens
        for text, tc in chunks[:-1]:
            assert tc >= 200, f"Chunk too small: {tc} tokens"
