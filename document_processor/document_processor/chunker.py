"""Sentence-aware text chunking with tiktoken."""

from __future__ import annotations

import re

import tiktoken

# Compile once at module level
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_ENCODING = None


def _get_encoding() -> tiktoken.Encoding:
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def chunk_text(
    text: str,
    target_tokens: int = 384,
    overlap_tokens: int = 38,
    min_tokens: int = 100,
) -> list[tuple[str, int]]:
    """Split text into chunks respecting sentence boundaries.

    Returns list of (chunk_text, token_count) tuples.
    """
    if not text or not text.strip():
        return []

    enc = _get_encoding()
    sentences = _split_sentences(text)

    if not sentences:
        return []

    chunks: list[tuple[str, int]] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tokens = len(enc.encode(sentence))

        # If a single sentence exceeds target, handle it
        if s_tokens > target_tokens and not current_sentences:
            # Force-add oversized sentence as its own chunk
            chunks.append((sentence, s_tokens))
            continue

        if current_tokens + s_tokens > target_tokens and current_sentences:
            # Emit current chunk
            chunk_text_str = " ".join(current_sentences)
            chunk_token_count = len(enc.encode(chunk_text_str))
            chunks.append((chunk_text_str, chunk_token_count))

            # Build overlap from end of current sentences
            current_sentences, current_tokens = _build_overlap(current_sentences, enc, overlap_tokens)

        current_sentences.append(sentence)
        current_tokens += s_tokens

    # Emit final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunk_token_count = len(enc.encode(chunk_text_str))
        chunks.append((chunk_text_str, chunk_token_count))

    # Filter out small chunks (unless it's the only one)
    if len(chunks) > 1:
        chunks = [(t, c) for t, c in chunks if c >= min_tokens]

    # Edge case: all chunks filtered out (shouldn't happen, but be safe)
    if not chunks and text.strip():
        full_tokens = len(enc.encode(text.strip()))
        chunks = [(text.strip(), full_tokens)]

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving non-empty parts."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _build_overlap(
    sentences: list[str],
    enc: tiktoken.Encoding,
    overlap_tokens: int,
) -> tuple[list[str], int]:
    """Take sentences from the end of the list until we reach overlap_tokens."""
    overlap_sents: list[str] = []
    total = 0
    for s in reversed(sentences):
        s_tok = len(enc.encode(s))
        if total + s_tok > overlap_tokens and overlap_sents:
            break
        overlap_sents.insert(0, s)
        total += s_tok
    return overlap_sents, total
