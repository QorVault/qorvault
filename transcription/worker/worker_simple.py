#!/usr/bin/env python3
"""Single-GPU WhisperX transcription worker.

Sequential processing: one file at a time on a single GPU.
Designed for SSH deployment via 03_provision_simple.py.

Environment variables (set by provisioner):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
    HF_TOKEN (pyannote speaker diarization model access)
    RUNPOD_POD_ID (auto-set by RunPod, optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_KEY = "manifest.json"
TRANSCRIPT_PREFIX = "transcripts/"
WHISPER_MODEL = "large-v3"
WHISPER_BATCH_SIZE = 6  # safe for 24GB VRAM
WHISPER_COMPUTE_TYPE = "float16"
LANGUAGE = "en"
SAMPLE_RATE = 16000

# Hallucination detection thresholds
MIN_WORDS_PER_MINUTE = 20
MAX_SEGMENT_REPEAT_COUNT = 3
MAX_REPEAT_RATIO = 0.3

# Temp directories on the pod
AUDIO_DIR = Path("/tmp/audio")
WAV_DIR = Path("/tmp/wav")
HEARTBEAT_PATH = Path("/tmp/worker_alive")

logger = logging.getLogger("worker_simple")

# Shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    sig_name = signal.Signals(signum).name
    logger.warning("%s received — will stop after current file", sig_name)
    _shutdown = True


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------


def create_s3_client():
    """Create boto3 S3 client for Cloudflare R2."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


# ---------------------------------------------------------------------------
# Manifest manager
# ---------------------------------------------------------------------------


class ManifestManager:
    """Single-writer manifest coordinator."""

    def __init__(self, s3, bucket: str, worker_id: str):
        self._s3 = s3
        self._bucket = bucket
        self._worker_id = worker_id
        self._manifest: dict = {}
        self._by_key: dict[str, dict] = {}

    def load(self) -> None:
        resp = self._s3.get_object(Bucket=self._bucket, Key=MANIFEST_KEY)
        self._manifest = json.loads(resp["Body"].read().decode("utf-8"))
        self._by_key = {f["key"]: f for f in self._manifest["files"]}
        total = self._manifest["total_files"]
        pending = sum(1 for f in self._manifest["files"] if f["status"] == "pending")
        logger.info("Manifest loaded: %d total, %d pending", total, pending)

    def get_pending_keys(self) -> list[str]:
        return [f["key"] for f in self._manifest["files"] if f["status"] == "pending"]

    def reset_stale_processing(self, timeout_minutes: int = 90) -> None:
        """Reset entries stuck in 'processing' for longer than timeout.

        Only resets files that have been processing for > timeout_minutes,
        so other active workers' files are not disturbed.
        """
        now = datetime.now(UTC)
        reset_count = 0
        for f in self._manifest["files"]:
            if f["status"] != "processing":
                continue
            started = f.get("started_at")
            if not started:
                # No timestamp — probably very old, reset it
                f["status"] = "pending"
                f["worker_id"] = None
                reset_count += 1
                continue
            try:
                started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                elapsed = (now - started_dt).total_seconds() / 60
                if elapsed > timeout_minutes:
                    f["status"] = "pending"
                    f["worker_id"] = None
                    f["started_at"] = None
                    reset_count += 1
                    logger.info("Reset stale file (%.0f min): %s", elapsed, f["key"])
            except (ValueError, TypeError):
                f["status"] = "pending"
                f["worker_id"] = None
                reset_count += 1
        if reset_count:
            logger.warning("Reset %d stale processing files to pending", reset_count)
            self._upload()

    def claim_file(self, key: str) -> bool:
        """Re-read manifest from R2 and claim a file. Returns True if claimed.

        Re-reads to minimize race conditions with other workers.
        """
        self.load()  # Re-read from R2
        entry = self._by_key.get(key)
        if not entry or entry["status"] != "pending":
            logger.warning(
                "File %s no longer pending (status=%s) — skipping", key, entry["status"] if entry else "missing"
            )
            return False
        entry["status"] = "processing"
        entry["worker_id"] = self._worker_id
        entry["started_at"] = datetime.now(UTC).isoformat()
        self._upload()
        return True

    def complete_file(
        self,
        key: str,
        transcript_key: str,
        duration: float,
        processing_time: float,
        word_count: int,
        segment_count: int,
        speaker_count: int,
        hallucination_flags: list[str],
    ) -> None:
        entry = self._by_key[key]
        entry["status"] = "complete"
        entry["completed_at"] = datetime.now(UTC).isoformat()
        entry["transcript_key"] = transcript_key
        entry["error"] = None
        if hallucination_flags:
            entry["hallucination_flags"] = hallucination_flags
        self._upload()

    def fail_file(self, key: str, error: str) -> None:
        entry = self._by_key[key]
        entry["status"] = "failed"
        entry["completed_at"] = datetime.now(UTC).isoformat()
        entry["error"] = error[:500]
        self._upload()

    def get_progress(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self._manifest["files"]:
            s = f["status"]
            counts[s] = counts.get(s, 0) + 1
        return counts

    def _upload(self, max_retries: int = 3) -> None:
        body = json.dumps(self._manifest, indent=2).encode("utf-8")
        for attempt in range(1, max_retries + 1):
            try:
                self._s3.put_object(
                    Bucket=self._bucket,
                    Key=MANIFEST_KEY,
                    Body=body,
                    ContentType="application/json",
                )
                return
            except Exception as e:
                if attempt < max_retries:
                    logger.warning("Manifest upload attempt %d/%d failed: %s", attempt, max_retries, e)
                    time.sleep(2 * attempt)
                else:
                    logger.error("Manifest upload FAILED after %d attempts: %s", max_retries, e)


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert audio file to 16kHz mono PCM WAV using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg exit {result.returncode}: {result.stderr[:500]}")


def audio_key_to_transcript_key(audio_key: str) -> str:
    """Convert audio/foo.opus -> transcripts/foo.json"""
    filename = audio_key.split("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    return f"{TRANSCRIPT_PREFIX}{stem}.json"


# ---------------------------------------------------------------------------
# Hallucination detection
# ---------------------------------------------------------------------------


def detect_hallucinations(segments: list[dict], audio_duration: float) -> list[str]:
    """Detect common WhisperX hallucination patterns. Returns flag strings."""
    flags = []

    if not segments:
        if audio_duration > 30:
            flags.append("no_segments_for_long_audio")
        return flags

    # Low word density
    total_words = sum(len(s.get("text", "").split()) for s in segments)
    duration_minutes = audio_duration / 60
    if duration_minutes > 1 and total_words / duration_minutes < MIN_WORDS_PER_MINUTE:
        flags.append(f"low_word_density:{total_words / duration_minutes:.0f}_wpm")

    # Repeated consecutive segments
    if len(segments) >= 3:
        texts = [s.get("text", "").strip().lower() for s in segments]
        max_repeat = 1
        current = 1
        for i in range(1, len(texts)):
            if texts[i] == texts[i - 1] and texts[i]:
                current += 1
                max_repeat = max(max_repeat, current)
            else:
                current = 1
        if max_repeat >= MAX_SEGMENT_REPEAT_COUNT:
            flags.append(f"repeated_segments:{max_repeat}_consecutive")

    # High duplicate ratio
    if len(segments) >= 5:
        texts = [s.get("text", "").strip().lower() for s in segments if s.get("text", "").strip()]
        text_counts: dict[str, int] = {}
        for t in texts:
            text_counts[t] = text_counts.get(t, 0) + 1
        duplicated = sum(c - 1 for c in text_counts.values() if c > 1)
        if texts and duplicated / len(texts) > MAX_REPEAT_RATIO:
            flags.append(f"high_duplicate_ratio:{duplicated / len(texts):.2f}")

    # Timestamp overflow
    for i, seg in enumerate(segments):
        if seg.get("end", 0) > audio_duration + 1.0:
            flags.append(f"timestamp_overflow:seg_{i}")
            break

    return flags


# ---------------------------------------------------------------------------
# Transcript builder
# ---------------------------------------------------------------------------


def build_transcript(audio_key: str, segments: list[dict], duration: float, hallucination_flags: list[str]) -> dict:
    """Build transcript JSON for upload to R2."""
    speakers = set()
    word_count = 0
    for seg in segments:
        if seg.get("speaker"):
            speakers.add(seg["speaker"])
        word_count += len(seg.get("text", "").split())

    return {
        "version": 1,
        "audio_key": audio_key,
        "created_at": datetime.now(UTC).isoformat(),
        "model": f"whisperx/{WHISPER_MODEL}",
        "language": LANGUAGE,
        "duration_seconds": round(duration, 2),
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "word_count": word_count,
        "segment_count": len(segments),
        "hallucination_flags": hallucination_flags,
        "segments": [
            {
                "start": round(s.get("start", 0), 3),
                "end": round(s.get("end", 0), 3),
                "text": s.get("text", "").strip(),
                "speaker": s.get("speaker"),
                "words": [
                    {
                        "word": w.get("word", ""),
                        "start": round(w.get("start", 0), 3),
                        "end": round(w.get("end", 0), 3),
                        "score": round(w.get("score", 0), 3),
                        "speaker": w.get("speaker"),
                    }
                    for w in s.get("words", [])
                ],
            }
            for s in segments
        ],
    }


# ---------------------------------------------------------------------------
# cuDNN 8 pre-loading for CTranslate2 compatibility
# ---------------------------------------------------------------------------

_cudnn8_loaded = False


def preload_cudnn8():
    """Pre-load cuDNN 8 libraries via ctypes for CTranslate2 compatibility.

    CTranslate2 (used by faster-whisper/whisperx) requires cuDNN 8, while
    PyTorch uses cuDNN 9. We must NOT put cuDNN 8 in LD_LIBRARY_PATH because
    on Blackwell GPUs (B200), PyTorch would pick up cuDNN 8 (no Blackwell
    kernels) and crash. Instead, we load cuDNN 8 .so files explicitly via
    ctypes.CDLL with RTLD_GLOBAL so CTranslate2 finds them in memory.
    """
    global _cudnn8_loaded
    if _cudnn8_loaded:
        return

    cudnn8_dir = Path("/opt/cudnn8/lib")
    if not cudnn8_dir.exists():
        logger.debug("No /opt/cudnn8/lib — cuDNN 8 preload not needed")
        _cudnn8_loaded = True
        return

    import ctypes

    # Load in dependency order: main lib first, then sub-libs
    load_order = [
        "libcudnn.so.8",
        "libcudnn_ops_infer.so.8",
        "libcudnn_ops_train.so.8",
        "libcudnn_cnn_infer.so.8",
        "libcudnn_cnn_train.so.8",
        "libcudnn_adv_infer.so.8",
        "libcudnn_adv_train.so.8",
    ]

    loaded = 0
    for lib_name in load_order:
        lib_path = cudnn8_dir / lib_name
        if lib_path.exists():
            try:
                ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError as e:
                logger.warning("Failed to preload %s: %s", lib_name, e)

    if loaded:
        logger.info("Pre-loaded %d cuDNN 8 libraries from %s", loaded, cudnn8_dir)
    _cudnn8_loaded = True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_models(hf_token: str, do_diarize: bool):
    """Load all WhisperX models once. Returns (whisper_model, align_model, align_meta, diarize_pipeline)."""
    # Pre-load cuDNN 8 for CTranslate2 before importing torch/whisperx
    preload_cudnn8()

    import torch
    import whisperx

    # PyTorch 2.6+ defaults weights_only=True in torch.load(), which breaks
    # pyannote.audio model loading (uses omegaconf globals not in safe list).
    # lightning_fabric also passes weights_only=True explicitly, so we must
    # force it to False.
    _original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading WhisperX model (%s) on %s...", WHISPER_MODEL, device)
    t0 = time.time()

    whisper_model = whisperx.load_model(
        WHISPER_MODEL,
        device,
        compute_type=WHISPER_COMPUTE_TYPE,
        language=LANGUAGE,
    )
    logger.info("Whisper model loaded in %.0fs", time.time() - t0)

    t1 = time.time()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=device,
    )
    logger.info("Alignment model loaded in %.0fs", time.time() - t1)

    diarize_pipeline = None
    if do_diarize:
        t2 = time.time()
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )
        logger.info("Diarization pipeline loaded in %.0fs", time.time() - t2)

    # Log GPU memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
        logger.info("GPU memory: %.1fGB allocated, %.1fGB reserved, %.1fGB total", allocated, reserved, total)

    return whisper_model, align_model, align_metadata, diarize_pipeline


# ---------------------------------------------------------------------------
# Single-file transcription
# ---------------------------------------------------------------------------


def transcribe_file(
    wav_path: Path,
    audio_key: str,
    whisper_model,
    align_model,
    align_metadata,
    diarize_pipeline,
    batch_size: int,
) -> dict:
    """Transcribe a single WAV file. Returns result dict."""
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    result = {
        "key": audio_key,
        "success": False,
        "transcript_data": None,
        "error": None,
        "duration_seconds": 0.0,
        "processing_seconds": 0.0,
        "word_count": 0,
        "segment_count": 0,
        "speaker_count": 0,
        "hallucination_flags": [],
    }

    try:
        # Load audio
        audio = whisperx.load_audio(str(wav_path))
        duration = len(audio) / SAMPLE_RATE
        result["duration_seconds"] = duration
        logger.info("  Audio loaded: %.0fs (%.1f min)", duration, duration / 60)

        # Step 1: Transcribe
        raw = whisper_model.transcribe(
            audio,
            batch_size=batch_size,
        )
        logger.info("  Transcription done (%d raw segments)", len(raw.get("segments", [])))

        # Step 2: Forced alignment
        aligned = whisperx.align(
            raw["segments"],
            align_model,
            align_metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )
        logger.info("  Alignment done")

        # Step 3: Speaker diarization (optional)
        if diarize_pipeline is not None:
            diarize_result = diarize_pipeline(audio)
            final = whisperx.assign_word_speakers(diarize_result, aligned)
            logger.info("  Diarization done")
        else:
            final = aligned

        segments = final.get("segments", [])

        # Step 4: Hallucination detection
        h_flags = detect_hallucinations(segments, duration)
        if h_flags:
            logger.warning("  Hallucination flags: %s", h_flags)

        # Step 5: Build transcript
        transcript = build_transcript(audio_key, segments, duration, h_flags)

        # Collect metrics
        result["success"] = True
        result["transcript_data"] = transcript
        result["word_count"] = transcript["word_count"]
        result["segment_count"] = transcript["segment_count"]
        result["speaker_count"] = transcript["speaker_count"]
        result["hallucination_flags"] = h_flags

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error("Transcription failed for %s: %s", audio_key, e, exc_info=True)

    result["processing_seconds"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU WhisperX transcription worker",
    )
    parser.add_argument(
        "--test-keys",
        nargs="+",
        default=None,
        help="Process only these R2 keys (for proof-of-concept)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=WHISPER_BATCH_SIZE,
        help=f"WhisperX batch size (default: {WHISPER_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Skip speaker diarization",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        stream=sys.stdout,
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    worker_id = os.environ.get("RUNPOD_POD_ID", f"worker-{os.getpid()}")
    bucket = os.environ["R2_BUCKET_NAME"]
    hf_token = os.environ.get("HF_TOKEN", "")

    logger.info("Worker %s starting (single-GPU sequential mode)", worker_id)
    logger.info("Batch size: %d, diarize: %s", args.batch_size, not args.no_diarize)

    # Validate GPU and auto-scale batch size
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
            logger.info("GPU: %s (%.1fGB VRAM)", gpu_name, gpu_mem)
            # Auto-scale batch size if user didn't override
            if args.batch_size == WHISPER_BATCH_SIZE:
                if gpu_mem >= 140:
                    args.batch_size = 32  # B200 180GB
                elif gpu_mem >= 70:
                    args.batch_size = 16  # H100/A100 80GB
                elif gpu_mem >= 40:
                    args.batch_size = 12  # L40S 48GB
                if args.batch_size != WHISPER_BATCH_SIZE:
                    logger.info("Auto-scaled batch size to %d for %.0fGB VRAM", args.batch_size, gpu_mem)
        else:
            logger.warning("No CUDA GPU detected — transcription will be very slow")
    except ImportError:
        logger.error("PyTorch not available")
        return 1

    # Load manifest
    s3 = create_s3_client()
    manifest = ManifestManager(s3, bucket, worker_id)
    manifest.load()
    manifest.reset_stale_processing()

    # Determine which files to process
    if args.test_keys:
        pending_keys = [
            k for k in args.test_keys if k in manifest._by_key and manifest._by_key[k]["status"] == "pending"
        ]
        if not pending_keys:
            logger.info("All test keys already processed or not found in manifest")
            return 0
        logger.info("Test mode: %d files to process", len(pending_keys))
    else:
        pending_keys = manifest.get_pending_keys()

    if not pending_keys:
        logger.info("Nothing to do — all files already processed")
        return 0

    logger.info("%d files pending transcription", len(pending_keys))

    # Load models
    do_diarize = not args.no_diarize and bool(hf_token)
    if not args.no_diarize and not hf_token:
        logger.warning("HF_TOKEN not set — skipping diarization")

    whisper_model, align_model, align_metadata, diarize_pipeline = load_models(
        hf_token,
        do_diarize,
    )

    # Process files one at a time
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    failed = 0
    t_start = time.time()

    for i, key in enumerate(pending_keys):
        if _shutdown:
            logger.warning("Shutdown requested — stopping")
            break

        # Heartbeat
        HEARTBEAT_PATH.write_text(datetime.now(UTC).isoformat())

        filename = key.split("/", 1)[-1]
        logger.info("[%d/%d] Processing: %s", i + 1, len(pending_keys), filename)

        # Claim file in manifest (re-reads from R2 to avoid races)
        if not manifest.claim_file(key):
            continue  # Another worker got it first

        try:
            # Download from R2
            audio_path = AUDIO_DIR / filename
            if not audio_path.exists():
                logger.info("  Downloading from R2...")
                t_dl = time.time()
                s3.download_file(bucket, key, str(audio_path))
                logger.info("  Downloaded in %.0fs (%.1fMB)", time.time() - t_dl, audio_path.stat().st_size / 1e6)

            # Convert to WAV
            wav_path = WAV_DIR / f"{audio_path.stem}.wav"
            if not wav_path.exists():
                logger.info("  Converting to WAV...")
                t_conv = time.time()
                convert_to_wav(audio_path, wav_path)
                logger.info("  Converted in %.0fs", time.time() - t_conv)

            # Transcribe
            result = transcribe_file(
                wav_path,
                key,
                whisper_model,
                align_model,
                align_metadata,
                diarize_pipeline,
                args.batch_size,
            )

            transcript_key = audio_key_to_transcript_key(key)

            if result["success"]:
                # Upload transcript to R2
                body = json.dumps(
                    result["transcript_data"],
                    indent=2,
                    ensure_ascii=False,
                ).encode("utf-8")
                s3.put_object(
                    Bucket=bucket,
                    Key=transcript_key,
                    Body=body,
                    ContentType="application/json",
                )
                logger.info("  Transcript uploaded: %s (%.0fKB)", transcript_key, len(body) / 1024)

                manifest.complete_file(
                    key,
                    transcript_key,
                    result["duration_seconds"],
                    result["processing_seconds"],
                    result["word_count"],
                    result["segment_count"],
                    result["speaker_count"],
                    result["hallucination_flags"],
                )
                completed += 1
            else:
                manifest.fail_file(key, result.get("error", "unknown"))
                failed += 1

        except Exception as e:
            logger.error("Unexpected error processing %s: %s", key, e, exc_info=True)
            manifest.fail_file(key, f"{type(e).__name__}: {e}")
            failed += 1

        # Clean up local files to save disk space
        for p in [AUDIO_DIR / filename, WAV_DIR / f"{Path(filename).stem}.wav"]:
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

        # Progress summary
        elapsed = time.time() - t_start
        total_done = completed + failed
        rate_str = (
            f"{result['duration_seconds'] / result['processing_seconds']:.1f}x"
            if result.get("processing_seconds", 0) > 0
            else "N/A"
        )
        logger.info(
            "[%d/%d] %s — %s in %.0fs (%s realtime) | %d ok, %d fail",
            total_done,
            len(pending_keys),
            filename,
            "OK" if result.get("success") else "FAIL",
            result.get("processing_seconds", 0),
            rate_str,
            completed,
            failed,
        )

    # Final summary
    elapsed = time.time() - t_start
    progress = manifest.get_progress()
    logger.info("=" * 70)
    logger.info("TRANSCRIPTION COMPLETE")
    logger.info("  Files processed: %d (%d ok, %d failed)", completed + failed, completed, failed)
    logger.info("  Wall time: %.1f minutes", elapsed / 60)
    if completed + failed > 0:
        logger.info("  Avg time per file: %.1f minutes", elapsed / 60 / (completed + failed))
    logger.info("  Manifest status: %s", json.dumps(progress))
    logger.info("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
