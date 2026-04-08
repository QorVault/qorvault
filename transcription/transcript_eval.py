#!/usr/bin/env python3
"""Transcript Evaluation Tool

Compare AssemblyAI and Deepgram transcripts side-by-side using semantic
search over chunked and embedded utterances.  Everything stays in memory —
no writes to PostgreSQL or Qdrant.

Usage:
    python transcript_eval.py <assemblyai.json> <deepgram.json>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Embedding model — reuse existing ONNX pipeline
# ---------------------------------------------------------------------------

PIPELINE_ROOT = Path(__file__).resolve().parent.parent / "embedding_pipeline"


def load_embedder():
    """Import and load the ONNX embedder from the embedding pipeline."""
    sys.path.insert(0, str(PIPELINE_ROOT))
    from embedding_pipeline.embedder import Embedder  # noqa: E402

    embedder = Embedder()
    embedder.load()
    return embedder


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    speaker: str
    start_secs: float
    text: str
    service: str


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_assemblyai(path: str) -> tuple[list[dict], int]:
    """Parse AssemblyAI JSON → list of utterances + raw word count.

    The file is a bare JSON array of word-level tokens, each with:
      text (str), start (int, ms), end (int, ms), speaker (str), confidence (float)

    Words are grouped into utterances by consecutive same-speaker runs.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "utterances" in data:
        # Standard AssemblyAI format with pre-grouped utterances
        utts = []
        for u in data["utterances"]:
            utts.append(
                {
                    "speaker": str(u["speaker"]),
                    "start": u["start"] / 1000.0,
                    "text": u["text"],
                }
            )
        word_count = sum(len(u["text"].split()) for u in utts)
        return utts, word_count

    if not isinstance(data, list):
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        print(
            f"ERROR: AssemblyAI JSON has unexpected structure. "
            f"Expected a list of word tokens or dict with 'utterances'. Got: {keys}"
        )
        sys.exit(1)

    if not data:
        print("ERROR: AssemblyAI JSON is an empty array.")
        sys.exit(1)

    # Validate first element has required keys
    sample = data[0]
    required = {"text", "start", "speaker"}
    missing = required - set(sample.keys())
    if missing:
        print(f"ERROR: AssemblyAI word tokens missing keys: {missing}. " f"Found keys: {list(sample.keys())}")
        sys.exit(1)

    # Group consecutive same-speaker word tokens into utterances
    utterances: list[dict] = []
    cur_speaker = None
    cur_words: list[str] = []
    cur_start: int | None = None

    for w in data:
        speaker = w["speaker"]
        if speaker != cur_speaker and cur_words:
            utterances.append(
                {
                    "speaker": str(cur_speaker),
                    "start": cur_start / 1000.0,  # ms → seconds
                    "text": " ".join(cur_words),
                }
            )
            cur_words = []
            cur_start = None
        cur_speaker = speaker
        if cur_start is None:
            cur_start = w["start"]
        cur_words.append(w["text"])

    if cur_words:
        utterances.append(
            {
                "speaker": str(cur_speaker),
                "start": cur_start / 1000.0,
                "text": " ".join(cur_words),
            }
        )

    return utterances, len(data)


def parse_deepgram(path: str) -> tuple[list[dict], int]:
    """Parse Deepgram JSON → list of utterances + raw word count.

    Words live at results.channels[0].alternatives[0].words, each with:
      word (str), punctuated_word (str), start (float, s), speaker (int), confidence (float)
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "results" not in data:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        print(f"ERROR: Deepgram JSON missing 'results' key. " f"Top-level keys: {keys}")
        sys.exit(1)

    # Navigate to words array
    try:
        words = data["results"]["channels"][0]["alternatives"][0]["words"]
    except (KeyError, IndexError) as exc:
        print(f"ERROR: Deepgram JSON missing expected path " f"results.channels[0].alternatives[0].words: {exc}")
        sys.exit(1)

    if not words:
        print("ERROR: Deepgram words array is empty.")
        sys.exit(1)

    # Group consecutive same-speaker words into utterances
    utterances: list[dict] = []
    cur_speaker: str | None = None
    cur_words: list[str] = []
    cur_start: float | None = None

    for w in words:
        speaker = f"Speaker {w['speaker']}"
        if speaker != cur_speaker and cur_words:
            utterances.append(
                {
                    "speaker": cur_speaker,
                    "start": cur_start,  # already seconds
                    "text": " ".join(cur_words),
                }
            )
            cur_words = []
            cur_start = None
        cur_speaker = speaker
        if cur_start is None:
            cur_start = w["start"]
        cur_words.append(w.get("punctuated_word", w["word"]))

    if cur_words:
        utterances.append(
            {
                "speaker": cur_speaker,
                "start": cur_start,
                "text": " ".join(cur_words),
            }
        )

    return utterances, len(words)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Word-count proxies for token counts (~1.3 tokens/word in English).
# 200 tokens ≈ 150 words; 50 tokens ≈ 38 words.
TARGET_WORDS = 150
SHORT_THRESHOLD = 38


def _wc(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def chunk_utterances(utterances: list[dict], service: str) -> list[Chunk]:
    """Merge short same-speaker utterances; split long ones at sentence boundaries."""
    chunks: list[Chunk] = []

    buf_speaker: str | None = None
    buf_start: float | None = None
    buf_text = ""

    def flush() -> None:
        nonlocal buf_speaker, buf_start, buf_text
        if not buf_text.strip():
            buf_speaker = buf_start = None
            buf_text = ""
            return

        wc = _wc(buf_text)
        if wc > int(TARGET_WORDS * 1.5):
            # Split at sentence boundaries
            sentences = _split_sentences(buf_text)
            sub = ""
            for sent in sentences:
                if _wc(sub) + _wc(sent) > TARGET_WORDS and sub.strip():
                    chunks.append(
                        Chunk(
                            speaker=buf_speaker,
                            start_secs=buf_start,
                            text=sub.strip(),
                            service=service,
                        )
                    )
                    sub = sent
                else:
                    sub = f"{sub} {sent}".strip()
            if sub.strip():
                chunks.append(
                    Chunk(
                        speaker=buf_speaker,
                        start_secs=buf_start,
                        text=sub.strip(),
                        service=service,
                    )
                )
        else:
            chunks.append(
                Chunk(
                    speaker=buf_speaker,
                    start_secs=buf_start,
                    text=buf_text.strip(),
                    service=service,
                )
            )

        buf_speaker = None
        buf_start = None
        buf_text = ""

    for utt in utterances:
        speaker = utt["speaker"]
        start = utt["start"]
        text = utt["text"]
        wc = _wc(text)

        # Speaker change → flush
        if speaker != buf_speaker and buf_text:
            flush()

        # Buffer would exceed target → flush first
        if buf_text and _wc(buf_text) + wc > TARGET_WORDS:
            flush()

        # Start or extend buffer
        if not buf_text:
            buf_speaker = speaker
            buf_start = start
            buf_text = text
        else:
            buf_text = f"{buf_text} {text}"

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------


def search_top_k(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Cosine similarity via dot product on L2-normalized vectors."""
    scores = embeddings @ query_vec  # (N,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def fmt_time(secs: float) -> str:
    """Seconds → HH:MM:SS or MM:SS."""
    total = int(secs)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _format_one(rank: int, chunk: Chunk, score: float, width: int = 60) -> list[str]:
    header = f"  #{rank}  [{chunk.speaker}]  {fmt_time(chunk.start_secs)}  sim={score:.3f}"
    wrapped = textwrap.fill(
        chunk.text,
        width=width,
        initial_indent="      ",
        subsequent_indent="      ",
    )
    return [header, wrapped, ""]


def display_results(
    aai_results: list[tuple[Chunk, float]],
    dg_results: list[tuple[Chunk, float]],
    term_width: int,
) -> None:
    if term_width >= 140:
        _display_side_by_side(aai_results, dg_results, term_width)
    else:
        _display_sequential(aai_results, dg_results)


def _display_side_by_side(aai_results, dg_results, term_width):
    col_w = (term_width - 3) // 2
    text_w = max(col_w - 10, 40)

    left_lines = [f"{'─' * col_w}", "  ASSEMBLYAI", ""]
    right_lines = [f"{'─' * col_w}", "  DEEPGRAM", ""]

    for i in range(max(len(aai_results), len(dg_results))):
        if i < len(aai_results):
            left_lines.extend(_format_one(i + 1, *aai_results[i], text_w))
        if i < len(dg_results):
            right_lines.extend(_format_one(i + 1, *dg_results[i], text_w))

    n = max(len(left_lines), len(right_lines))
    print()
    for i in range(n):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        print(f"{left:<{col_w}} │ {right}")
    print()


def _display_sequential(aai_results, dg_results):
    print(f"\n{'═' * 60}")
    print("  ASSEMBLYAI RESULTS")
    print(f"{'─' * 60}")
    for i, (chunk, score) in enumerate(aai_results):
        for line in _format_one(i + 1, chunk, score):
            print(line)

    print(f"\n{'═' * 60}")
    print("  DEEPGRAM RESULTS")
    print(f"{'─' * 60}")
    for i, (chunk, score) in enumerate(dg_results):
        for line in _format_one(i + 1, chunk, score):
            print(line)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AssemblyAI and Deepgram transcripts via semantic search",
    )
    parser.add_argument("assemblyai_json", help="Path to AssemblyAI transcript JSON")
    parser.add_argument("deepgram_json", help="Path to Deepgram transcript JSON")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Parse transcripts
    # ------------------------------------------------------------------
    print("Loading transcripts...")
    aai_utts, aai_wc = parse_assemblyai(args.assemblyai_json)
    dg_utts, dg_wc = parse_deepgram(args.deepgram_json)

    print(f"  AssemblyAI: {len(aai_utts):,} utterances from {aai_wc:,} word tokens")
    print(f"  Deepgram:   {len(dg_utts):,} utterances from {dg_wc:,} word tokens")

    ratio = max(aai_wc, dg_wc) / max(min(aai_wc, dg_wc), 1)
    if ratio > 1.2:
        print(f"  WARNING: word count ratio is {ratio:.1f}x — coverage may differ")

    # ------------------------------------------------------------------
    # 2. Chunk
    # ------------------------------------------------------------------
    aai_chunks = chunk_utterances(aai_utts, "AssemblyAI")
    dg_chunks = chunk_utterances(dg_utts, "Deepgram")

    print("\nChunking:")
    print(f"  AssemblyAI: {len(aai_chunks):,} chunks")
    print(f"  Deepgram:   {len(dg_chunks):,} chunks")

    # ------------------------------------------------------------------
    # 3. Load embedder
    # ------------------------------------------------------------------
    print("\nLoading embedding model (ONNX Runtime CPU)...")
    embedder = load_embedder()

    # ------------------------------------------------------------------
    # 4. Embed
    # ------------------------------------------------------------------
    all_chunks = aai_chunks + dg_chunks
    all_texts = [c.text for c in all_chunks]

    print(f"Embedding {len(all_texts):,} chunks...")
    t0 = time.time()
    all_vecs = embedder.encode(all_texts)
    embed_time = time.time() - t0

    n_aai = len(aai_chunks)
    aai_vecs = all_vecs[:n_aai]
    dg_vecs = all_vecs[n_aai:]

    print("\nEmbedding complete:")
    print(f"  Dimension:  {all_vecs.shape[1]}")
    print(f"  AssemblyAI: {aai_vecs.shape[0]:,} vectors")
    print(f"  Deepgram:   {dg_vecs.shape[0]:,} vectors")
    print(f"  Time:       {embed_time:.1f}s")

    norm = float(np.linalg.norm(aai_vecs[0]))
    print(f"  L2 norm check (first vector): {norm:.6f}")

    # ------------------------------------------------------------------
    # 5. Interactive REPL
    # ------------------------------------------------------------------
    print(f"\n{'═' * 60}")
    print("  Transcript Comparison REPL")
    print("  Type a query to search both transcripts.")
    print("  'quit' or 'exit' to leave.  Ctrl+C also works.")
    print(f"{'═' * 60}\n")

    try:
        tw = os.get_terminal_size().columns
    except OSError:
        tw = 80

    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            print("Please enter a query.\n")
            continue

        if query.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        q_vec = embedder.encode([query])[0]

        aai_hits = search_top_k(q_vec, aai_vecs, aai_chunks)
        dg_hits = search_top_k(q_vec, dg_vecs, dg_chunks)

        display_results(aai_hits, dg_hits, tw)


if __name__ == "__main__":
    main()
