#!/usr/bin/env python3
"""WhisperX transcription worker for RunPod 8x B200 GPU pods.

Self-contained script downloaded from R2 by the bootstrap in 02_provision.py.
Installs its own dependencies, processes all pending audio files from the
manifest using 8 GPU workers in parallel, writes transcripts back to R2,
and updates manifest.json after each file completion (per-file checkpointing).

Environment variables (set by RunPod pod config):
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
    HF_TOKEN (pyannote speaker diarization model access)
    RUNPOD_POD_ID (auto-set by RunPod)
    RUNPOD_API_KEY (for self-termination)
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
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
TARGET_GPU_COUNT = 8
WHISPER_MODEL = "large-v3"
WHISPER_BATCH_SIZE = 16
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

logger = logging.getLogger("worker")

# Shutdown flag shared across processes
_shutdown_requested = mp.Event()

# Globals set per GPU worker process (after fork)
_whisper_model = None
_align_model = None
_align_metadata = None
_diarize_pipeline = None


# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------


def install_dependencies() -> None:
    """Install WhisperX, ffmpeg, and boto3. Called once before forking."""
    print("[worker] Installing dependencies...", flush=True)
    t0 = time.time()

    # ffmpeg for audio conversion
    subprocess.run(
        ["apt-get", "update", "-qq"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "ffmpeg"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # WhisperX + boto3
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "whisperx",
            "boto3",
        ]
    )

    print(f"[worker] Dependencies installed in {time.time() - t0:.0f}s", flush=True)


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
# Manifest manager (main process only)
# ---------------------------------------------------------------------------


class ManifestManager:
    """Single-writer manifest coordinator. Only the main process uses this."""

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

    def reset_processing_files(self) -> None:
        """Reset stale 'processing' entries back to 'pending' (handles prior eviction)."""
        reset_count = 0
        for f in self._manifest["files"]:
            if f["status"] == "processing":
                f["status"] = "pending"
                f["worker_id"] = None
                f["started_at"] = None
                reset_count += 1
        if reset_count:
            logger.warning("Reset %d stale processing files to pending", reset_count)
            self._upload()

    def claim_file(self, key: str) -> None:
        entry = self._by_key[key]
        entry["status"] = "processing"
        entry["worker_id"] = self._worker_id
        entry["started_at"] = datetime.now(UTC).isoformat()

    def claim_batch(self, keys: list[str]) -> None:
        """Claim multiple files and upload manifest once."""
        for key in keys:
            self.claim_file(key)
        self._upload()

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
# GPU worker process
# ---------------------------------------------------------------------------

_gpu_counter = mp.Value("i", 0)


def _pool_initializer(hf_token: str) -> None:
    """Called once per worker process. Loads WhisperX models on assigned GPU."""
    global _whisper_model, _align_model, _align_metadata, _diarize_pipeline

    with _gpu_counter.get_lock():
        gpu_id = _gpu_counter.value
        _gpu_counter.value += 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import whisperx

    logger.info("GPU %d: loading models...", gpu_id)
    t0 = time.time()

    _whisper_model = whisperx.load_model(
        WHISPER_MODEL,
        "cuda",
        compute_type=WHISPER_COMPUTE_TYPE,
        language=LANGUAGE,
    )

    _align_model, _align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device="cuda",
    )

    _diarize_pipeline = whisperx.DiarizationPipeline(
        use_auth_token=hf_token,
        device="cuda",
    )

    logger.info("GPU %d: models loaded in %.0fs", gpu_id, time.time() - t0)


def _transcribe_file(args: tuple[str, str]) -> dict:
    """Transcribe a single WAV file. Runs inside GPU worker process.

    Args: (wav_path, audio_key)
    Returns: result dict with success, transcript_data, error, metrics
    """
    wav_path, audio_key = args
    import whisperx

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
        audio = whisperx.load_audio(wav_path)
        duration = len(audio) / SAMPLE_RATE
        result["duration_seconds"] = duration

        # Step 1: Transcribe with anti-hallucination settings
        raw = _whisper_model.transcribe(
            audio,
            batch_size=WHISPER_BATCH_SIZE,
            condition_on_previous_text=False,
        )

        # Step 2: Forced alignment for word-level timestamps
        aligned = whisperx.align(
            raw["segments"],
            _align_model,
            _align_metadata,
            audio,
            device="cuda",
            return_char_alignments=False,
        )

        # Step 3: Speaker diarization
        diarize_result = _diarize_pipeline(audio)
        final = whisperx.assign_word_speakers(diarize_result, aligned)

        segments = final.get("segments", [])

        # Step 4: Hallucination detection
        h_flags = detect_hallucinations(segments, duration)

        # Step 5: Build transcript
        transcript = build_transcript(audio_key, segments, duration, h_flags)

        # Collect metrics
        speakers = set()
        word_count = 0
        for seg in segments:
            if seg.get("speaker"):
                speakers.add(seg["speaker"])
            word_count += len(seg.get("text", "").split())

        result["success"] = True
        result["transcript_data"] = transcript
        result["word_count"] = word_count
        result["segment_count"] = len(segments)
        result["speaker_count"] = len(speakers)
        result["hallucination_flags"] = h_flags

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        logger.error("Transcription failed for %s: %s", audio_key, e, exc_info=True)

    result["processing_seconds"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# SIGTERM handling
# ---------------------------------------------------------------------------


def _handle_signal(signum, frame):
    """Set shutdown flag on SIGTERM/SIGINT."""
    sig_name = signal.Signals(signum).name
    logger.warning("%s received — initiating graceful shutdown", sig_name)
    _shutdown_requested.set()


# ---------------------------------------------------------------------------
# Self-termination
# ---------------------------------------------------------------------------


def _self_terminate_pod() -> None:
    """Terminate this pod via RunPod API to stop billing."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not api_key or not pod_id:
        logger.info("No RUNPOD_API_KEY/POD_ID — skipping self-termination")
        return

    try:
        import runpod

        runpod.api_key = api_key
        runpod.terminate_pod(pod_id)
        logger.info("Self-terminated pod %s", pod_id)
    except Exception as e:
        logger.warning("Self-termination failed (monitor will handle): %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    # ---- Phase 0: Install dependencies ----
    install_dependencies()

    import torch

    # ---- Setup logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(processName)s] %(message)s",
        stream=sys.stdout,
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    worker_id = os.environ.get("RUNPOD_POD_ID", f"worker-{os.getpid()}")
    bucket = os.environ["R2_BUCKET_NAME"]
    hf_token = os.environ["HF_TOKEN"]

    gpu_count = torch.cuda.device_count()
    num_workers = min(TARGET_GPU_COUNT, gpu_count) if gpu_count > 0 else 1
    logger.info("Worker %s starting — %d GPUs detected, using %d workers", worker_id, gpu_count, num_workers)

    # ---- Phase 1: Load manifest ----
    s3 = create_s3_client()
    manifest = ManifestManager(s3, bucket, worker_id)
    manifest.load()
    manifest.reset_processing_files()

    pending_keys = manifest.get_pending_keys()
    logger.info("%d files pending transcription", len(pending_keys))

    if not pending_keys:
        logger.info("Nothing to do — all files already processed")
        _self_terminate_pod()
        return 0

    # ---- Phase 2: Download audio files ----
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %d audio files from R2...", len(pending_keys))
    t_download = time.time()
    local_paths: dict[str, Path] = {}

    for i, key in enumerate(pending_keys):
        if _shutdown_requested.is_set():
            logger.warning("Shutdown during download — exiting")
            return 0

        filename = key.split("/", 1)[-1]
        local_path = AUDIO_DIR / filename

        if not local_path.exists():
            s3.download_file(bucket, key, str(local_path))

        local_paths[key] = local_path

        if (i + 1) % 50 == 0 or (i + 1) == len(pending_keys):
            logger.info("Downloaded %d/%d files", i + 1, len(pending_keys))

    logger.info("Download complete in %.0fs", time.time() - t_download)

    # ---- Phase 3: Convert to WAV ----
    logger.info("Converting audio to 16kHz mono WAV...")
    t_convert = time.time()
    wav_paths: dict[str, Path] = {}

    for key, audio_path in local_paths.items():
        if _shutdown_requested.is_set():
            break

        wav_path = WAV_DIR / f"{audio_path.stem}.wav"
        try:
            convert_to_wav(audio_path, wav_path)
            wav_paths[key] = wav_path
        except Exception as e:
            logger.error("ffmpeg failed for %s: %s", key, e)
            manifest.fail_file(key, f"ffmpeg: {e}")

    logger.info("Conversion complete in %.0fs (%d files)", time.time() - t_convert, len(wav_paths))

    # ---- Phase 4: Claim files and dispatch to GPU workers ----
    work_items = [(str(wav_paths[key]), key) for key in pending_keys if key in wav_paths]

    if not work_items:
        logger.warning("No files to process after conversion")
        _self_terminate_pod()
        return 1

    # Claim all files as "processing" in manifest
    manifest.claim_batch([key for _, key in work_items])
    logger.info("Claimed %d files, starting %d GPU workers", len(work_items), num_workers)

    completed = 0
    failed = 0
    t_start = time.time()

    with mp.Pool(
        processes=num_workers,
        initializer=_pool_initializer,
        initargs=(hf_token,),
    ) as pool:
        results_iter = pool.imap_unordered(_transcribe_file, work_items)

        for result in results_iter:
            if _shutdown_requested.is_set():
                logger.warning("Shutdown requested — terminating pool")
                pool.terminate()
                break

            key = result["key"]
            transcript_key = audio_key_to_transcript_key(key)

            if result["success"]:
                # Upload transcript to R2
                try:
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
                except Exception as e:
                    logger.error("Transcript upload failed for %s: %s", key, e)
                    result["success"] = False
                    result["error"] = f"transcript_upload: {e}"

            # Update manifest (per-file checkpoint)
            if result["success"]:
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

            # Clean up local files
            for paths in (local_paths, wav_paths):
                p = paths.pop(key, None)
                if p and p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass

            # Progress
            elapsed = time.time() - t_start
            total_done = completed + failed
            rate = total_done / (elapsed / 60) if elapsed > 0 else 0
            h_flags = result.get("hallucination_flags", [])
            flag_str = f" [HALLU: {', '.join(h_flags)}]" if h_flags else ""
            logger.info(
                "[%d/%d] %s -> %s (%.0fs)%s  |  %.1f files/min",
                total_done,
                len(work_items),
                key.split("/")[-1],
                "OK" if result["success"] else f"FAIL: {result.get('error', '')[:80]}",
                result["processing_seconds"],
                flag_str,
                rate,
            )

    # ---- Phase 5: Summary ----
    elapsed = time.time() - t_start
    progress = manifest.get_progress()
    logger.info("=" * 70)
    logger.info("TRANSCRIPTION COMPLETE")
    logger.info("  Files processed: %d (%d ok, %d failed)", completed + failed, completed, failed)
    logger.info("  Wall time: %.1f minutes", elapsed / 60)
    if completed + failed > 0:
        logger.info("  Rate: %.1f files/min", (completed + failed) / (elapsed / 60))
    logger.info("  Manifest status: %s", json.dumps(progress))
    logger.info("=" * 70)

    # ---- Phase 6: Self-terminate ----
    _self_terminate_pod()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
