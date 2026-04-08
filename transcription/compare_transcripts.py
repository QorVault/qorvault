#!/usr/bin/env python3
"""Compare WhisperX transcripts with YouTube auto-generated transcripts.

Computes basic quality metrics and shows side-by-side text samples
at matching timestamps.

Usage:
    python compare_transcripts.py
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

WHISPERX_DIR = Path("/home/qorvault/projects/ksd_forensic/output/whisperx_transcripts")
YOUTUBE_DIR = Path("/home/qorvault/projects/ksd_forensic/output/transcripts")

# Map audio dates to YouTube transcript filenames
PAIRS = [
    {
        "label": "Executive Session 08/17/2021 (short, ~2hr)",
        "whisperx": "20210818_-_KSD_Board_Executive_Session_-_08_17_2021.json",
        "youtube": "17AUG2021Meeting.json",
    },
    {
        "label": "Special Board Meeting 04/19/2023 (medium, ~2hr)",
        "whisperx": "20230420_-_KSD_Special_Board_Meeting__Work_Session_-_04_19_2023.json",
        "youtube": "19APR2023Meeting.json",
    },
    {
        "label": "Regular Board Meeting 09/11/2024 (long, ~9hr)",
        "whisperx": "20240912_-_KSD_Regular_Board_Meeting_-_09_11_24.json",
        "youtube": "11SEP2024Meeting.json",
    },
]

# Timestamps to sample for side-by-side comparison (seconds from start)
SAMPLE_TIMES = [300, 900, 1800, 3600]


def load_whisperx(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_youtube(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_whisperx_text_at(segments: list[dict], t: float, window: float = 30) -> str:
    """Get WhisperX text within a time window around t."""
    texts = []
    for seg in segments:
        if seg["start"] >= t - 5 and seg["start"] <= t + window:
            texts.append(seg.get("text", "").strip())
    return " ".join(texts)


def get_youtube_text_at(segments: list[dict], t: float, window: float = 30) -> str:
    """Get YouTube text within a time window around t."""
    texts = []
    for seg in segments:
        seg_start = seg["start"]
        if seg_start >= t - 5 and seg_start <= t + window:
            texts.append(seg.get("text", "").strip())
    return " ".join(texts)


def word_error_rate(ref: str, hyp: str) -> float:
    """Simplified WER: edit distance on word sequences."""
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    sm = SequenceMatcher(None, ref_words, hyp_words)
    edits = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        edits += max(i2 - i1, j2 - j1)
    return edits / len(ref_words)


def fmt_time(secs: float) -> str:
    total = int(secs)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def compare_pair(pair: dict) -> None:
    wx_path = WHISPERX_DIR / pair["whisperx"]
    yt_path = YOUTUBE_DIR / pair["youtube"]

    if not wx_path.exists():
        print(f"  SKIP: WhisperX file not found: {wx_path}")
        return
    if not yt_path.exists():
        print(f"  SKIP: YouTube file not found: {yt_path}")
        return

    wx = load_whisperx(wx_path)
    yt = load_youtube(yt_path)

    wx_segs = wx.get("segments", [])
    yt_segs = yt.get("segments", [])

    # Basic metrics
    wx_words = wx.get("word_count", sum(len(s.get("text", "").split()) for s in wx_segs))
    yt_text = yt.get("full_text", "")
    yt_words = len(yt_text.split()) if yt_text else sum(len(s.get("text", "").split()) for s in yt_segs)

    wx_duration = wx.get("duration_seconds", 0) / 60
    wx_speakers = wx.get("speaker_count", 0)
    wx_flags = wx.get("hallucination_flags", [])

    print(f"\n{'=' * 78}")
    print(f"  {pair['label']}")
    print(f"{'=' * 78}")
    print(f"  {'Metric':<25} {'WhisperX':>15} {'YouTube':>15}")
    print(f"  {'-' * 55}")
    print(f"  {'Duration (min)':<25} {wx_duration:>15.1f} {'N/A':>15}")
    print(f"  {'Word count':<25} {wx_words:>15,} {yt_words:>15,}")
    print(f"  {'Segments':<25} {len(wx_segs):>15,} {len(yt_segs):>15,}")
    print(f"  {'Speakers':<25} {wx_speakers:>15} {'N/A (no diarize)':>15}")
    if wx_flags:
        print(f"  {'Hallucination flags':<25} {', '.join(wx_flags)}")
    else:
        print(f"  {'Hallucination flags':<25} {'none':>15}")

    # Word density (words per minute)
    if wx_duration > 0:
        wx_wpm = wx_words / wx_duration
        yt_wpm = yt_words / wx_duration  # use same duration reference
        print(f"  {'Words/minute':<25} {wx_wpm:>15.0f} {yt_wpm:>15.0f}")

    # Side-by-side text comparison at sample timestamps
    print(f"\n  {'─' * 74}")
    print("  Side-by-side text comparison (30s windows)")
    print(f"  {'─' * 74}")

    wer_scores = []
    for t in SAMPLE_TIMES:
        if t > wx.get("duration_seconds", 0):
            continue

        wx_text = get_whisperx_text_at(wx_segs, t)
        yt_text_chunk = get_youtube_text_at(yt_segs, t)

        if not wx_text and not yt_text_chunk:
            continue

        wer = word_error_rate(wx_text, yt_text_chunk)
        wer_scores.append(wer)

        print(f"\n  @ {fmt_time(t)} (WER: {wer:.0%})")
        print(f"  WhisperX: {wx_text[:200]}")
        print(f"  YouTube:  {yt_text_chunk[:200]}")

    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        print(f"\n  Average WER across {len(wer_scores)} samples: {avg_wer:.0%}")
        print("  (Lower WER = more similar. Note: neither transcript is 'ground truth')")

    # Show WhisperX-only features: speaker diarization sample
    if wx_speakers > 0:
        print(f"\n  {'─' * 74}")
        print("  Speaker diarization sample (WhisperX only)")
        print(f"  {'─' * 74}")
        seen_speakers = set()
        for seg in wx_segs:
            spk = seg.get("speaker", "")
            if spk and spk not in seen_speakers and len(seen_speakers) < 5:
                seen_speakers.add(spk)
                t = seg.get("start", 0)
                text = seg.get("text", "")[:100]
                print(f"  [{spk}] @ {fmt_time(t)}: {text}")

    # Show some proper nouns/terms WhisperX might get better
    print(f"\n  {'─' * 74}")
    print("  Sample content (first 5 non-trivial WhisperX segments)")
    print(f"  {'─' * 74}")
    shown = 0
    for seg in wx_segs:
        text = seg.get("text", "").strip()
        if len(text) > 30 and shown < 5:
            spk = seg.get("speaker", "?")
            t = seg.get("start", 0)
            print(f"  [{spk}] {fmt_time(t)}: {text[:120]}")
            shown += 1


def main() -> int:
    print("WhisperX vs YouTube Auto-Caption Transcript Comparison")
    print("=" * 78)

    for pair in PAIRS:
        compare_pair(pair)

    print(f"\n{'=' * 78}")
    print("Summary")
    print("=" * 78)
    print("WhisperX advantages over YouTube auto-captions:")
    print("  - Speaker diarization (who said what)")
    print("  - Word-level timestamps with confidence scores")
    print("  - Better handling of proper nouns and domain terminology")
    print("  - Forced alignment for precise timing")
    print("  - No 'foreign' / '[Music]' filler segments")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
