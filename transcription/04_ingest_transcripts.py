#!/usr/bin/env python3
"""Ingest WhisperX transcripts into the BoardDocs RAG system.

Downloads completed transcripts from Cloudflare R2, resolves anonymous
speaker labels to real names via Claude API, chunks with speaker-aware
boundaries, computes per-turn acoustic/interaction metrics, and inserts
into PostgreSQL.  The existing embedding pipeline then picks up the
pending chunks automatically.

Usage:
    python 04_ingest_transcripts.py              # one-shot
    python 04_ingest_transcripts.py --watch       # poll every 60s
    python 04_ingest_transcripts.py --dry-run     # preview without DB writes
    python 04_ingest_transcripts.py --force        # re-ingest existing docs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import asyncpg
import tiktoken
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRANSCRIPT_DIR = Path("/home/qorvault/projects/ksd_forensic/output/whisperx_transcripts")
MANIFEST_KEY = "manifest.json"
TENANT = "kent_sd"
COLLECTION = "boarddocs_chunks"

# Chunking parameters — match existing pipeline
TARGET_TOKENS = 384
OVERLAP_TOKENS = 38
MIN_TOKENS = 100

# Merge turns separated by < this many seconds
TURN_MERGE_GAP = 1.0

# Speaker resolution — segments from opening minutes sent to LLM
OPENING_SEGMENTS = 150

# Local LLM via llama.cpp
LLAMA_CLI = Path("/opt/llama.cpp-vulkan/build/bin/llama-cli")
LLAMA_MODEL = Path("/home/qorvault/models/chat--Mistral-Small-3.1-24B-Instruct-2503-Q8_0.gguf")
LLAMA_GPU_LAYERS = 99  # offload all layers to GPU

logger = logging.getLogger(__name__)

# tiktoken encoding — lazy singleton
_ENCODING: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding:
    """Return cached tiktoken cl100k_base encoding."""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Segment:
    """A single WhisperX segment with speaker and word-level detail."""

    start: float
    end: float
    text: str
    speaker: str | None
    words: list[dict] = field(default_factory=list)


@dataclass
class Turn:
    """Consecutive segments from the same speaker, merged into one turn."""

    speaker: str
    start: float
    end: float
    text: str
    segments: list[Segment] = field(default_factory=list)

    # Acoustic / interaction metrics computed after merging
    words_per_minute: float = 0.0
    mean_confidence: float = 0.0
    pause_before: float = 0.0
    overlap_with_previous: bool = False


@dataclass
class Chunk:
    """A chunk ready for database insertion."""

    index: int
    content: str
    token_count: int
    speakers: list[str]
    time_start: float
    time_end: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1: Download from R2
# ---------------------------------------------------------------------------


def get_s3_client():
    """Create a boto3 S3 client for Cloudflare R2."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def download_new_transcripts() -> list[Path]:
    """Download completed transcripts from R2 that aren't already local.

    Returns list of newly downloaded file paths.
    """
    s3 = get_s3_client()
    bucket = os.environ["R2_BUCKET_NAME"]

    resp = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)
    manifest = json.loads(resp["Body"].read().decode("utf-8"))

    complete = [f for f in manifest["files"] if f["status"] == "complete"]
    logger.info("Manifest: %d total, %d complete", manifest["total_files"], len(complete))

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    new_files: list[Path] = []

    for entry in complete:
        transcript_key = entry.get("transcript_key")
        if not transcript_key:
            continue

        filename = transcript_key.split("/", 1)[-1]
        local_path = TRANSCRIPT_DIR / filename

        if local_path.exists():
            continue

        try:
            s3.download_file(bucket, transcript_key, str(local_path))
            size_kb = local_path.stat().st_size / 1024
            logger.info("Downloaded: %s (%.0f KB)", filename, size_kb)
            new_files.append(local_path)
        except Exception as e:
            logger.error("Download failed: %s — %s", filename, e)

    return new_files


# ---------------------------------------------------------------------------
# Step 2: Parse WhisperX JSON
# ---------------------------------------------------------------------------

# Filename patterns:
#   20240912_-_KSD_Regular_Board_Meeting_-_09_11_24.json
#   Date is at the end: _MM_DD_YY or _MM_DD_YYYY

_DATE_RE = re.compile(r"(\d{1,2})_(\d{1,2})_(\d{2,4})\s*$")

# Committee name extraction from filename
_COMMITTEE_PATTERNS = [
    (re.compile(r"Regular_Board_Meeting", re.I), "Regular Board Meeting"),
    (re.compile(r"Special_Board_Meeting", re.I), "Special Board Meeting"),
    (re.compile(r"Executive_Session", re.I), "Executive Session"),
    (re.compile(r"Work_Session", re.I), "Work Session"),
    (re.compile(r"Board_Retreat", re.I), "Board Retreat"),
]


def parse_meeting_date(filename: str) -> date | None:
    """Extract meeting date from audio/transcript filename."""
    stem = filename.rsplit(".", 1)[0]
    m = _DATE_RE.search(stem)
    if not m:
        return None
    month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if year < 100:
        year += 2000
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_committee_name(filename: str) -> str:
    """Extract committee/meeting type from filename."""
    for pattern, name in _COMMITTEE_PATTERNS:
        if pattern.search(filename):
            return name
    return "Board Meeting"


def parse_transcript(path: Path) -> dict:
    """Load and return parsed WhisperX transcript JSON."""
    with open(path) as f:
        return json.load(f)


def extract_segments(data: dict) -> list[Segment]:
    """Convert raw transcript JSON segments into Segment objects."""
    segments = []
    for seg in data.get("segments", []):
        segments.append(
            Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                speaker=seg.get("speaker"),
                words=seg.get("words", []),
            )
        )
    return segments


# ---------------------------------------------------------------------------
# Step 3: Speaker name resolution via LLM
# ---------------------------------------------------------------------------


async def get_board_members_for_date(pool: asyncpg.Pool, meeting_date: date) -> list[str]:
    """Query PostgreSQL for board member names active near a meeting date.

    Looks at voting record documents within ±6 months for mentioned names.
    Falls back to known composition if no data found.
    """
    # Known board compositions by era
    compositions = [
        # (start, end, members, president, vp, superintendent)
        (
            date(2023, 12, 1),
            date(2025, 12, 31),
            [
                "Meghin Margel",
                "Maya Vengadasalam",
                "Donald Cook",
                "Joe Farah",
                "Awale Farah",
                "Leslie Hamada",
                "Tim Clark",
                "Hyun-Jin Song",
            ],
            "Meghin Margel",
            "Joe Farah",
            "Israel Vela",
        ),
        (
            date(2025, 12, 1),
            date(2030, 12, 31),
            ["Meghin Margel", "Donald Cook", "Hyun-Jin Song", "Denise Gregory", "Nyema Williams"],
            "Meghin Margel",
            "Denise Gregory",
            "Israel Vela",
        ),
        (
            date(2021, 1, 1),
            date(2023, 12, 1),
            ["Meghin Margel", "Maya Vengadasalam", "Donald Cook", "Joe Farah", "Leslie Hamada", "Tim Clark"],
            "Maya Vengadasalam",
            "Meghin Margel",
            "Israel Vela",
        ),
    ]

    for start, end, members, president, vp, supt in compositions:
        if start <= meeting_date <= end:
            return members

    # Fallback: return the oldest known composition
    return compositions[-1][2]


def get_board_president(meeting_date: date) -> str:
    """Return the board president for a given date."""
    if meeting_date >= date(2023, 12, 1):
        return "Meghin Margel"
    return "Maya Vengadasalam"


async def resolve_speakers_llm(
    segments: list[Segment],
    meeting_date: date,
    board_members: list[str],
) -> dict[str, str]:
    """Use Claude API to map SPEAKER_XX labels to real names.

    Sends the opening segments plus any roll call vote segments
    to Claude Haiku for name resolution.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed; skipping LLM speaker resolution")
        return {}

    prompt = _build_speaker_prompt(segments, meeting_date, board_members)
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if response_text.startswith("```"):
            response_text = re.sub(r"```(?:json)?\s*", "", response_text)
            response_text = response_text.rstrip("`").strip()

        speaker_map = json.loads(response_text)
        logger.info(
            "LLM resolved %d speakers: %s",
            len(speaker_map),
            speaker_map,
        )
        return speaker_map

    except Exception as e:
        logger.warning("LLM speaker resolution failed: %s", e)
        return {}


def _build_speaker_prompt(
    segments: list[Segment],
    meeting_date: date,
    board_members: list[str],
) -> str:
    """Build the speaker-resolution prompt shared by all LLM backends."""
    opening_text_parts = []
    for seg in segments[:OPENING_SEGMENTS]:
        speaker = seg.speaker or "UNKNOWN"
        timestamp = f"{seg.start:.1f}s"
        opening_text_parts.append(f"[{speaker}] ({timestamp}): {seg.text}")

    roll_call_parts = []
    for seg in segments[OPENING_SEGMENTS:]:
        text_lower = seg.text.lower()
        if any(kw in text_lower for kw in ["roll call", "aye", "nay", "motion", "second", "all in favor"]):
            speaker = seg.speaker or "UNKNOWN"
            timestamp = f"{seg.start:.1f}s"
            roll_call_parts.append(f"[{speaker}] ({timestamp}): {seg.text}")
            if len(roll_call_parts) >= 50:
                break

    opening_text = "\n".join(opening_text_parts)
    roll_call_text = "\n".join(roll_call_parts) if roll_call_parts else "(no roll call votes found)"

    president = get_board_president(meeting_date)
    superintendent = "Israel Vela"

    return (
        f"Here is the opening of a Kent School District board meeting"
        f" on {meeting_date.isoformat()}.\n"
        f"\n"
        f"The board members at this time were: {', '.join(board_members)}\n"
        f"The board president is: {president}\n"
        f"The superintendent is: {superintendent}\n"
        f"\n"
        f"The president typically calls the meeting to order and conducts roll call.\n"
        f'During roll call, the president calls each director by name ("Director Cook?")'
        f' and they respond "Present" or "Here".\n'
        f'Roll call votes later have longer "Aye"/"Nay" responses that are more'
        f" reliably diarized.\n"
        f'Other speakers may say "Thank you, President {president.split()[-1]}"'
        f" which confirms the president's label.\n"
        f"\n"
        f"=== OPENING OF MEETING ===\n"
        f"{opening_text}\n"
        f"\n"
        f"=== ROLL CALL VOTES (later in meeting) ===\n"
        f"{roll_call_text}\n"
        f"\n"
        f"Map SPEAKER_XX labels to real names. Only map speakers you are confident about.\n"
        f"Return a JSON object mapping speaker labels to names, e.g.:\n"
        f'{{"SPEAKER_14": "Meghin Margel", "SPEAKER_20": "Israel Vela"}}\n'
        f"\n"
        f"Return ONLY the JSON object, no other text."
    )


def resolve_speakers_local(
    segments: list[Segment],
    meeting_date: date,
    board_members: list[str],
) -> dict[str, str]:
    """Use local llama.cpp to map SPEAKER_XX labels to real names.

    Runs llama-cli with Mistral Small 3.1 24B in single-turn mode.
    """
    if not LLAMA_CLI.exists():
        logger.warning("llama-cli not found at %s; skipping local resolution", LLAMA_CLI)
        return {}
    if not LLAMA_MODEL.exists():
        logger.warning("Model not found at %s; skipping local resolution", LLAMA_MODEL)
        return {}

    prompt = _build_speaker_prompt(segments, meeting_date, board_members)

    # Write prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name

    try:
        cmd = [
            str(LLAMA_CLI),
            "-m",
            str(LLAMA_MODEL),
            "-ngl",
            str(LLAMA_GPU_LAYERS),
            "-f",
            prompt_path,
            "-n",
            "1024",
            "--temp",
            "0.1",
            "-c",
            "8192",
            "--no-display-prompt",
            "--single-turn",
            "--jinja",
        ]

        logger.info("Running local LLM for speaker resolution (meeting %s)...", meeting_date)
        result = subprocess.run(  # noqa: S603 — hardcoded binary path, no user input
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.warning(
                "llama-cli exited with code %d: %s",
                result.returncode,
                result.stderr[-500:] if result.stderr else "(no stderr)",
            )
            return {}

        response_text = result.stdout.strip()

        # The model output may contain leaked prompt text — find the LAST
        # JSON object which is the actual model response
        json_matches = re.findall(r"\{[^}]+\}", response_text)
        if not json_matches:
            logger.warning("No JSON found in local LLM output: %s", response_text[-300:])
            return {}

        speaker_map = json.loads(json_matches[-1])
        logger.info(
            "Local LLM resolved %d speakers: %s",
            len(speaker_map),
            speaker_map,
        )
        return speaker_map

    except subprocess.TimeoutExpired:
        logger.warning("Local LLM timed out after 300s for meeting %s", meeting_date)
        return {}
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse local LLM JSON: %s", e)
        return {}
    except Exception as e:
        logger.warning("Local LLM speaker resolution failed: %s", e)
        return {}
    finally:
        Path(prompt_path).unlink(missing_ok=True)


def apply_speaker_map(
    segments: list[Segment],
    speaker_map: dict[str, str],
    meeting_date: date,
) -> dict[str, str]:
    """Apply speaker map to segments, numbering unresolved speakers.

    Unresolved speakers get meeting-specific labels like
    "Speaker 3 (2024-09-11)" to be unique across meetings.

    Returns the final complete speaker map (including unresolved labels).
    """
    # Track unresolved speaker labels and assign sequential numbers
    unresolved_counter = 0
    full_map: dict[str, str] = dict(speaker_map)
    date_str = meeting_date.isoformat()

    for seg in segments:
        if seg.speaker and seg.speaker not in full_map:
            unresolved_counter += 1
            full_map[seg.speaker] = f"Speaker {unresolved_counter} ({date_str})"

    # Now apply the full map to all segments
    for seg in segments:
        if seg.speaker and seg.speaker in full_map:
            seg.speaker = full_map[seg.speaker]

    return full_map


# ---------------------------------------------------------------------------
# Step 4: Merge segments into turns and compute interaction metrics
# ---------------------------------------------------------------------------


def merge_segments_to_turns(segments: list[Segment]) -> list[Turn]:
    """Group consecutive same-speaker segments into turns.

    Segments from the same speaker separated by < TURN_MERGE_GAP seconds
    are merged into a single turn.
    """
    if not segments:
        return []

    turns: list[Turn] = []
    current_speaker = segments[0].speaker
    current_start = segments[0].start
    current_end = segments[0].end
    current_texts: list[str] = [segments[0].text]
    current_segs: list[Segment] = [segments[0]]

    for seg in segments[1:]:
        # Same speaker and close in time → merge
        if seg.speaker == current_speaker and seg.start - current_end < TURN_MERGE_GAP:
            current_end = seg.end
            current_texts.append(seg.text)
            current_segs.append(seg)
        else:
            # Emit turn
            turns.append(
                Turn(
                    speaker=current_speaker or "UNKNOWN",
                    start=current_start,
                    end=current_end,
                    text=" ".join(current_texts),
                    segments=current_segs,
                )
            )
            current_speaker = seg.speaker
            current_start = seg.start
            current_end = seg.end
            current_texts = [seg.text]
            current_segs = [seg]

    # Final turn
    turns.append(
        Turn(
            speaker=current_speaker or "UNKNOWN",
            start=current_start,
            end=current_end,
            text=" ".join(current_texts),
            segments=current_segs,
        )
    )

    return turns


def compute_turn_metrics(turns: list[Turn]) -> None:
    """Compute acoustic and interaction metrics for each turn in-place.

    Metrics derived from WhisperX word-level data:
    - words_per_minute: speaking pace (urgency vs deliberateness)
    - mean_confidence: average word recognition confidence (clarity proxy)
    - pause_before: seconds of silence before this turn (hesitation/deference)
    - overlap_with_previous: True if this turn starts before the previous ends
    """
    for i, turn in enumerate(turns):
        duration = turn.end - turn.start
        if duration > 0:
            word_count = sum(len(seg.words) for seg in turn.segments)
            turn.words_per_minute = (word_count / duration) * 60.0
        else:
            turn.words_per_minute = 0.0

        # Mean word confidence
        all_scores = [w.get("score", 0.0) for seg in turn.segments for w in seg.words if w.get("score", 0) > 0]
        turn.mean_confidence = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Pause before this turn / overlap detection
        # WhisperX produces non-overlapping segments, so true overlap
        # is rare.  We use a short-gap heuristic: if a different speaker
        # starts within 0.3s of the previous speaker ending, count it
        # as a "rapid takeover" (proxy for interruption/crosstalk).
        if i > 0:
            prev = turns[i - 1]
            gap = turn.start - prev.end
            if gap < 0:
                turn.overlap_with_previous = True
                turn.pause_before = 0.0
            elif gap < 0.3 and turn.speaker != prev.speaker:
                # Rapid speaker change — proxy for interruption
                turn.overlap_with_previous = True
                turn.pause_before = gap
            else:
                turn.pause_before = gap
        else:
            turn.pause_before = turn.start  # silence at start of recording


def compute_meeting_dynamics(turns: list[Turn]) -> dict:
    """Compute meeting-level interaction and power dynamics metrics.

    Returns a dict with per-speaker stats and interaction patterns
    that enable bias and power-dynamics analysis across meetings.
    """
    speaker_stats: dict[str, dict] = defaultdict(
        lambda: {
            "speaking_time_s": 0.0,
            "turn_count": 0,
            "total_words": 0,
            "interruptions_suffered": 0,
            "interruptions_initiated": 0,
            "mean_confidence": [],
            "words_per_minute": [],
            "response_latencies": [],
        }
    )

    # Adjacency: who speaks after whom
    adjacency: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for i, turn in enumerate(turns):
        spk = turn.speaker
        duration = turn.end - turn.start
        word_count = sum(len(seg.words) for seg in turn.segments)

        speaker_stats[spk]["speaking_time_s"] += duration
        speaker_stats[spk]["turn_count"] += 1
        speaker_stats[spk]["total_words"] += word_count
        speaker_stats[spk]["words_per_minute"].append(turn.words_per_minute)
        if turn.mean_confidence > 0:
            speaker_stats[spk]["mean_confidence"].append(turn.mean_confidence)

        # Track adjacency pairs and interruptions
        if i > 0:
            prev_speaker = turns[i - 1].speaker
            if prev_speaker != spk:
                adjacency[prev_speaker][spk] += 1

                # Response latency (how long before prev_speaker's turn
                # gets a reply)
                latency = turn.start - turns[i - 1].end
                speaker_stats[prev_speaker]["response_latencies"].append(latency)

                # Overlap = interruption
                if turn.overlap_with_previous:
                    speaker_stats[prev_speaker]["interruptions_suffered"] += 1
                    speaker_stats[spk]["interruptions_initiated"] += 1

    # Aggregate per-speaker metrics
    total_speaking_time = sum(s["speaking_time_s"] for s in speaker_stats.values())
    dynamics: dict[str, dict] = {}

    for spk, stats in speaker_stats.items():
        confs = stats["mean_confidence"]
        wpms = stats["words_per_minute"]
        latencies = stats["response_latencies"]

        dynamics[spk] = {
            "speaking_time_s": round(stats["speaking_time_s"], 1),
            "speaking_share_pct": round(100.0 * stats["speaking_time_s"] / total_speaking_time, 1)
            if total_speaking_time > 0
            else 0.0,
            "turn_count": stats["turn_count"],
            "total_words": stats["total_words"],
            "mean_turn_length_words": round(stats["total_words"] / stats["turn_count"], 1)
            if stats["turn_count"] > 0
            else 0.0,
            "mean_wpm": round(sum(wpms) / len(wpms), 1) if wpms else 0.0,
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
            "interruptions_suffered": stats["interruptions_suffered"],
            "interruptions_initiated": stats["interruptions_initiated"],
            "mean_response_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
        }

    # Build adjacency matrix (top pairs only to keep metadata manageable)
    top_adjacency = []
    for from_spk, targets in adjacency.items():
        for to_spk, count in targets.items():
            if count >= 3:  # only significant interaction pairs
                top_adjacency.append({"from": from_spk, "to": to_spk, "count": count})
    top_adjacency.sort(key=lambda x: x["count"], reverse=True)

    return {
        "speaker_dynamics": dynamics,
        "interaction_pairs": top_adjacency[:30],
        "total_speaking_time_s": round(total_speaking_time, 1),
        "total_turns": len(turns),
    }


def format_turn_for_chunk(turn: Turn) -> str:
    """Format a turn with speaker label, timestamps, and acoustic annotation.

    Includes speaking pace and confidence as inline metadata to enable
    tone/demeanor analysis at query time.
    """
    time_start = _format_timestamp(turn.start)
    time_end = _format_timestamp(turn.end)

    # Build acoustic annotation string
    annotations = []
    if turn.words_per_minute > 0:
        annotations.append(f"{turn.words_per_minute:.0f} wpm")
    if turn.mean_confidence > 0:
        annotations.append(f"conf:{turn.mean_confidence:.2f}")
    if turn.overlap_with_previous:
        annotations.append("overlapping")
    if turn.pause_before > 3.0:
        annotations.append(f"pause:{turn.pause_before:.1f}s")

    annotation_str = f" ({', '.join(annotations)})" if annotations else ""

    return f"[{turn.speaker}] ({time_start}-{time_end}){annotation_str}: {turn.text}"


def _format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


# ---------------------------------------------------------------------------
# Step 5: Speaker-aware chunking
# ---------------------------------------------------------------------------


def chunk_turns(turns: list[Turn]) -> list[Chunk]:
    """Chunk turns into ~384-token segments, respecting speaker boundaries.

    Each chunk preserves who said what.  Overlap is implemented by
    repeating the last turn of the previous chunk at the start of the next.
    """
    enc = _enc()
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_tokens = 0
    current_speakers: set[str] = set()
    current_start: float | None = None
    current_end: float = 0.0
    last_turn_line: str | None = None
    last_turn_tokens = 0
    chunk_index = 0

    for turn in turns:
        line = format_turn_for_chunk(turn)
        line_tokens = len(enc.encode(line))

        # Would this turn exceed the target?
        if current_tokens + line_tokens > TARGET_TOKENS and current_lines:
            # Emit chunk
            content = "\n".join(current_lines)
            token_count = len(enc.encode(content))

            if token_count >= MIN_TOKENS or chunk_index == 0:
                chunks.append(
                    Chunk(
                        index=chunk_index,
                        content=content,
                        token_count=token_count,
                        speakers=sorted(current_speakers),
                        time_start=current_start or 0.0,
                        time_end=current_end,
                    )
                )
                chunk_index += 1

            # Start new chunk with overlap (last turn of previous chunk)
            if last_turn_line:
                current_lines = [last_turn_line]
                current_tokens = last_turn_tokens
            else:
                current_lines = []
                current_tokens = 0
            current_speakers = set()
            current_start = None

        current_lines.append(line)
        current_tokens += line_tokens
        current_speakers.add(turn.speaker)
        if current_start is None:
            current_start = turn.start
        current_end = turn.end
        last_turn_line = line
        last_turn_tokens = line_tokens

    # Emit final chunk
    if current_lines:
        content = "\n".join(current_lines)
        token_count = len(enc.encode(content))
        if token_count >= MIN_TOKENS or chunk_index == 0:
            chunks.append(
                Chunk(
                    index=chunk_index,
                    content=content,
                    token_count=token_count,
                    speakers=sorted(current_speakers),
                    time_start=current_start or 0.0,
                    time_end=current_end,
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Step 6: Database insertion
# ---------------------------------------------------------------------------

INSERT_DOC_SQL = """
INSERT INTO documents (
    tenant_id, external_id, document_type, title,
    content_text, meeting_date, committee_name,
    processing_status, metadata
)
VALUES ($1, $2, 'transcript', $3, $4, $5, $6, 'complete', $7::jsonb)
ON CONFLICT (tenant_id, external_id) DO NOTHING
RETURNING id
"""

INSERT_CHUNK_SQL = """
INSERT INTO chunks (
    tenant_id, document_id, chunk_index, content,
    token_count, embedding_status, metadata
)
VALUES ($1, $2, $3, $4, $5, 'pending', $6::jsonb)
ON CONFLICT (document_id, chunk_index) DO NOTHING
"""


async def insert_transcript(
    pool: asyncpg.Pool,
    audio_key: str,
    title: str,
    meeting_date: date,
    committee_name: str,
    full_text: str,
    chunks: list[Chunk],
    doc_metadata: dict,
    dry_run: bool = False,
) -> str | None:
    """Insert document and chunks into PostgreSQL.

    Returns the document UUID, or None if already exists (idempotent).
    """
    external_id = f"transcript_{audio_key}"

    if dry_run:
        logger.info("DRY RUN: would insert doc '%s' with %d chunks", title, len(chunks))
        for c in chunks[:3]:
            logger.info(
                "  chunk %d: %d tokens, speakers=%s, preview=%.80s...",
                c.index,
                c.token_count,
                c.speakers,
                c.content,
            )
        if len(chunks) > 3:
            logger.info("  ... and %d more chunks", len(chunks) - 3)
        return None

    doc_metadata_json = json.dumps(doc_metadata)

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                INSERT_DOC_SQL,
                TENANT,
                external_id,
                title,
                full_text,
                meeting_date,
                committee_name,
                doc_metadata_json,
            )

            if row is None:
                logger.info("Document already exists: %s (skipped)", title)
                return None

            doc_id = row["id"]

            for chunk in chunks:
                chunk_meta = json.dumps(
                    {
                        "meeting_date": meeting_date.isoformat(),
                        "committee_name": committee_name,
                        "title": title,
                        "speakers": chunk.speakers,
                        "time_start": chunk.time_start,
                        "time_end": chunk.time_end,
                        "audio_key": audio_key,
                    }
                )
                await conn.execute(
                    INSERT_CHUNK_SQL,
                    TENANT,
                    doc_id,
                    chunk.index,
                    chunk.content,
                    chunk.token_count,
                    chunk_meta,
                )

    logger.info(
        "Inserted: %s — %d chunks (doc_id=%s)",
        title,
        len(chunks),
        doc_id,
    )
    return str(doc_id)


# ---------------------------------------------------------------------------
# Step 7: Run embedding pipeline
# ---------------------------------------------------------------------------


async def run_embedding_pipeline() -> None:
    """Invoke the existing embedding pipeline for pending transcript chunks."""
    logger.info("Running embedding pipeline for pending chunks...")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "embedding_pipeline",
        "--tenant",
        TENANT,
        cwd=str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 0:
        logger.info("Embedding pipeline completed successfully")
    else:
        logger.error(
            "Embedding pipeline failed (exit %d): %s",
            proc.returncode,
            stderr.decode()[-500:] if stderr else "(no stderr)",
        )
    if stdout:
        # Log final stats from pipeline
        for line in stdout.decode().strip().split("\n")[-5:]:
            logger.info("  embed: %s", line)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def process_transcript(
    path: Path,
    pool: asyncpg.Pool,
    dry_run: bool = False,
    force: bool = False,
    use_local_llm: bool = False,
) -> bool:
    """Process a single transcript file end-to-end.

    Returns True if the transcript was ingested (or would be in dry-run).
    """
    filename = path.name
    meeting_date = parse_meeting_date(filename)
    if not meeting_date:
        logger.warning("Cannot parse date from: %s (skipping)", filename)
        return False

    committee_name = parse_committee_name(filename)
    logger.info(
        "Processing: %s (date=%s, committee=%s)",
        filename,
        meeting_date,
        committee_name,
    )

    # Parse transcript
    data = parse_transcript(path)
    audio_key = data.get("audio_key", filename)
    segments = extract_segments(data)

    if not segments:
        logger.warning("No segments in: %s (skipping)", filename)
        return False

    # Check if already ingested (unless --force)
    if not force and not dry_run:
        external_id = f"transcript_{audio_key}"
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM documents WHERE tenant_id = $1 AND external_id = $2",
                TENANT,
                external_id,
            )
            if exists:
                logger.info("Already ingested: %s (use --force to re-ingest)", filename)
                return False

    # Resolve speakers via LLM (local or API)
    board_members = await get_board_members_for_date(pool, meeting_date)
    if use_local_llm:
        speaker_map = resolve_speakers_local(segments, meeting_date, board_members)
    else:
        speaker_map = await resolve_speakers_llm(segments, meeting_date, board_members)

    # Apply speaker names (modifies segments in-place, assigns numbered
    # labels to unresolved speakers like "Speaker 3 (2024-09-11)")
    full_speaker_map = apply_speaker_map(segments, speaker_map, meeting_date)

    # Merge segments into turns and compute metrics
    turns = merge_segments_to_turns(segments)
    compute_turn_metrics(turns)

    # Compute meeting-level dynamics for bias/power analysis
    meeting_dynamics = compute_meeting_dynamics(turns)

    # Build full text (for content_text column)
    full_text = "\n".join(f"[{t.speaker}] ({_format_timestamp(t.start)}): {t.text}" for t in turns)

    # Chunk turns
    chunks = chunk_turns(turns)

    if not chunks:
        logger.warning("No chunks produced for: %s", filename)
        return False

    # Build document-level metadata
    doc_metadata = {
        "duration_seconds": data.get("duration_seconds", 0),
        "speaker_count": data.get("speaker_count", 0),
        "word_count": data.get("word_count", 0),
        "audio_key": audio_key,
        "speaker_map": full_speaker_map,
        "model": data.get("model", ""),
        "meeting_dynamics": meeting_dynamics,
        "hallucination_flags": data.get("hallucination_flags", []),
    }

    title = f"Board Meeting Transcript - {meeting_date.isoformat()}"
    if committee_name != "Board Meeting":
        title = f"{committee_name} Transcript - {meeting_date.isoformat()}"

    # If force mode, delete existing document first
    if force and not dry_run:
        external_id = f"transcript_{audio_key}"
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM documents WHERE tenant_id = $1 AND external_id = $2",
                TENANT,
                external_id,
            )

    doc_id = await insert_transcript(
        pool=pool,
        audio_key=audio_key,
        title=title,
        meeting_date=meeting_date,
        committee_name=committee_name,
        full_text=full_text,
        chunks=chunks,
        doc_metadata=doc_metadata,
        dry_run=dry_run,
    )

    if dry_run:
        # Print dynamics summary
        logger.info("Meeting dynamics for %s:", title)
        for spk, stats in meeting_dynamics["speaker_dynamics"].items():
            logger.info(
                "  %-30s  %5.1f%% time, %3d turns, %4d words, " "%.0f wpm, %d interruptions suffered",
                spk,
                stats["speaking_share_pct"],
                stats["turn_count"],
                stats["total_words"],
                stats["mean_wpm"],
                stats["interruptions_suffered"],
            )

    return doc_id is not None or dry_run


def get_dsn() -> str:
    """Build PostgreSQL DSN from environment variables."""
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "qorvault")
    user = os.environ.get("POSTGRES_USER", "qorvault")
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD not set. Copy .env.example to .env and fill in.")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


async def run_once(
    dry_run: bool = False,
    force: bool = False,
    skip_download: bool = False,
    skip_embed: bool = False,
    use_local_llm: bool = False,
) -> int:
    """Run the full pipeline once: download → parse → chunk → insert → embed.

    Returns count of transcripts ingested.
    """
    # Download new transcripts from R2
    if not skip_download:
        try:
            new_files = download_new_transcripts()
            logger.info("Downloaded %d new transcripts", len(new_files))
        except Exception as e:
            logger.error("R2 download failed: %s", e)
            logger.info("Continuing with locally available transcripts")

    # Find all local transcript files
    if not TRANSCRIPT_DIR.exists():
        logger.error("Transcript directory not found: %s", TRANSCRIPT_DIR)
        return 0

    transcript_files = sorted(TRANSCRIPT_DIR.glob("*.json"))
    if not transcript_files:
        logger.info("No transcript files found in %s", TRANSCRIPT_DIR)
        return 0

    logger.info("Found %d transcript files", len(transcript_files))

    # Connect to PostgreSQL
    dsn = get_dsn()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)

    ingested = 0
    try:
        for path in transcript_files:
            try:
                result = await process_transcript(path, pool, dry_run, force, use_local_llm=use_local_llm)
                if result:
                    ingested += 1
            except Exception as e:
                logger.error("Failed to process %s: %s", path.name, e, exc_info=True)
    finally:
        await pool.close()

    logger.info("Ingested %d / %d transcripts", ingested, len(transcript_files))

    # Run embedding pipeline (unless dry-run or skip)
    if ingested > 0 and not dry_run and not skip_embed:
        await run_embedding_pipeline()

    return ingested


async def watch_loop(
    interval: int = 60,
    dry_run: bool = False,
    force: bool = False,
    use_local_llm: bool = False,
) -> None:
    """Continuously poll for new transcripts and ingest them."""
    logger.info("Watch mode: polling every %ds (Ctrl+C to stop)", interval)
    while True:
        try:
            ingested = await run_once(dry_run=dry_run, force=force, use_local_llm=use_local_llm)
            if ingested > 0:
                logger.info("Watch: ingested %d transcripts this cycle", ingested)
            else:
                logger.debug("Watch: no new transcripts")
        except Exception as e:
            logger.error("Watch cycle error: %s", e, exc_info=True)

        await asyncio.sleep(interval)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest WhisperX transcripts into BoardDocs RAG",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll for new transcripts",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Watch polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview parsing and chunking without DB writes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest transcripts even if already in DB",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip R2 download, use local files only",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip running the embedding pipeline after insertion",
    )
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Use local llama.cpp instead of Anthropic API for speaker resolution",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Load environment from both .env files
    load_dotenv(SCRIPT_DIR / ".env")
    load_dotenv(PROJECT_ROOT / ".env")

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stderr,
    )

    if args.watch:
        try:
            asyncio.run(
                watch_loop(
                    interval=args.interval,
                    dry_run=args.dry_run,
                    force=args.force,
                    use_local_llm=args.local_llm,
                )
            )
        except KeyboardInterrupt:
            logger.info("Watch mode stopped by user")
        return 0
    else:
        ingested = asyncio.run(
            run_once(
                dry_run=args.dry_run,
                force=args.force,
                skip_download=args.skip_download,
                skip_embed=args.skip_embed,
                use_local_llm=args.local_llm,
            )
        )
        return 0 if ingested >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
