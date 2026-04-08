#!/usr/bin/env python3
"""Batch speaker diarization using pyannote.audio 4.x (CPU-only).

Processes a directory of 16kHz mono WAV files and outputs per-file
diarization JSON with speaker segments. Supports checkpointing via
file-based locking and multi-process parallelism.

Usage:
    python3 diarize.py --input-dir PATH --output-dir PATH
    python3 diarize.py --input-dir PATH --output-dir PATH --workers 4
    python3 diarize.py --input-dir PATH --output-dir PATH --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
import wave
from datetime import UTC, datetime
from pathlib import Path

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"
MIN_SPEAKERS = 2
MAX_SPEAKERS = 15
STALE_LOCK_SECONDS = 2 * 3600  # 2 hours

# HF_TOKEN search paths (first match wins)
ENV_CANDIDATES = [
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent / "transcription" / ".env",
]


def load_hf_token() -> str:
    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            load_dotenv(env_path)
            break
    token = os.environ.get("HF_TOKEN")
    if not token:
        console.print("[red]HF_TOKEN not found in any .env file[/red]")
        console.print("Searched:", [str(p) for p in ENV_CANDIDATES])
        sys.exit(1)
    return token


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch speaker diarization (CPU-only, pyannote 4.x)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing 16kHz mono WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for diarization JSON output",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). " "4-8 is reasonable for a 16-core CPU-only machine.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed, then exit",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------


def validate_wav_format(wav_path: Path) -> None:
    """Check that a WAV file is 16kHz mono. Exits on mismatch."""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()

    errors = []
    if rate != 16000:
        errors.append(f"sample rate is {rate}Hz, expected 16000Hz")
    if channels != 1:
        errors.append(f"channels is {channels}, expected 1 (mono)")

    if errors:
        console.print(f"[red]WAV format error in {wav_path.name}:[/red]")
        for e in errors:
            console.print(f"  {e}")
        console.print("All input files must be 16kHz mono PCM WAV.")
        sys.exit(1)


def get_wav_duration(wav_path: Path) -> float:
    """Return duration in seconds."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


# ---------------------------------------------------------------------------
# File discovery and locking
# ---------------------------------------------------------------------------


def discover_files(
    input_dir: Path,
    output_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Return (to_process, skipped) lists of WAV files."""
    all_wavs = sorted(input_dir.glob("*.wav"))
    to_process = []
    skipped = []

    for wav in all_wavs:
        out_json = output_dir / f"{wav.stem}_diarization.json"
        if out_json.exists():
            skipped.append(wav)
        else:
            to_process.append(wav)

    return to_process, skipped


def cleanup_stale_locks(output_dir: Path) -> int:
    """Remove lockfiles older than STALE_LOCK_SECONDS. Returns count removed."""
    now = time.time()
    removed = 0
    for lock in output_dir.glob("*.lock"):
        try:
            age = now - lock.stat().st_mtime
            if age > STALE_LOCK_SECONDS:
                lock.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def try_claim(output_dir: Path, wav_path: Path) -> bool:
    """Attempt to atomically claim a file via O_CREAT|O_EXCL lockfile.

    Returns True if this process now owns the file.
    """
    lock_path = output_dir / f"{wav_path.stem}.lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        # Write our PID for debugging
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_lock(output_dir: Path, wav_path: Path) -> None:
    """Remove the lockfile for a completed/failed file."""
    lock_path = output_dir / f"{wav_path.stem}.lock"
    try:
        lock_path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Logging (per-worker)
# ---------------------------------------------------------------------------


def setup_worker_logger(name: str, output_dir: Path) -> logging.Logger:
    wlogger = logging.getLogger(name)
    wlogger.setLevel(logging.DEBUG)
    wlogger.propagate = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{name}_{timestamp}.log"

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    wlogger.addHandler(fh)

    wlogger.info("Log file: %s  PID: %d", log_file, os.getpid())
    return wlogger


# ---------------------------------------------------------------------------
# Diarization core
# ---------------------------------------------------------------------------


def diarize_file(pipeline, wav_path: Path) -> dict:
    """Run diarization on a single WAV file. Returns the output dict."""
    duration = get_wav_duration(wav_path)

    t0 = time.perf_counter()
    result = pipeline(
        str(wav_path),
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS,
    )
    processing_time = time.perf_counter() - t0

    annotation = result.exclusive_speaker_diarization

    # Build ordered speaker map: first-seen order -> SPEAKER_01, SPEAKER_02, ...
    speaker_map: dict[str, str] = {}
    raw_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{len(speaker_map) + 1:02d}"
        raw_segments.append(
            {
                "speaker": speaker_map[speaker],
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
            }
        )

    return {
        "metadata": {
            "source_file": wav_path.name,
            "processed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round(duration, 2),
            "num_speakers_detected": len(speaker_map),
            "pyannote_version": "4.0.4",
            "processing_seconds": round(processing_time, 2),
        },
        "segments": raw_segments,
    }


def write_failure(output_dir: Path, wav_path: Path, error: Exception) -> None:
    """Write a failure marker JSON so downstream scripts know this file failed."""
    fail_path = output_dir / f"{wav_path.stem}_diarization_failed.json"
    fail_data = {
        "source_file": wav_path.name,
        "failed_at": datetime.now(UTC).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    fail_path.write_text(json.dumps(fail_data, indent=2))


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def worker_main(
    worker_id: int,
    input_dir: Path,
    output_dir: Path,
    file_list: list[str],  # filenames (not Paths) for pickling
    status_dict: dict,  # mp.Manager dict for progress reporting
    completed_counter,  # mp.Manager.Value (not mp.Value)
    audio_processed_counter,  # mp.Manager.Value
    shutdown_flag: dict,  # mp.Manager dict used as boolean flag
) -> int:
    """Entry point for each worker process. Returns number of failures."""
    wlogger = setup_worker_logger(f"diarize_worker_{worker_id}", output_dir)
    wlogger.info("Worker %d starting, %d candidate files", worker_id, len(file_list))

    # Ignore SIGINT in workers — main process handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Load pipeline in this process
    status_dict[worker_id] = "loading pipeline..."
    hf_token = load_hf_token()

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(PIPELINE_MODEL, token=hf_token)
    pipeline.to(torch.device("cpu"))
    wlogger.info("Pipeline loaded on CPU")

    failures = 0

    for filename in file_list:
        if shutdown_flag.get("stop", False):
            wlogger.info("Shutdown requested, stopping")
            break

        wav_path = input_dir / filename

        # Skip if already done
        out_json = output_dir / f"{wav_path.stem}_diarization.json"
        if out_json.exists():
            completed_counter.value += 1
            continue

        # Try to claim
        if not try_claim(output_dir, wav_path):
            continue  # another worker got it

        status_dict[worker_id] = filename[:55]
        wlogger.info("Processing %s", filename)

        try:
            result = diarize_file(pipeline, wav_path)

            out_json.write_text(json.dumps(result, indent=2))

            duration = result["metadata"]["duration_seconds"]
            proc_time = result["metadata"]["processing_seconds"]
            n_speakers = result["metadata"]["num_speakers_detected"]

            wlogger.info(
                "OK  %-55s  dur=%.0fs  speakers=%d  proc=%.1fs",
                filename,
                duration,
                n_speakers,
                proc_time,
            )

            audio_processed_counter.value += duration

        except Exception as e:
            wlogger.error("FAIL %s: %s\n%s", filename, e, traceback.format_exc())
            write_failure(output_dir, wav_path, e)
            failures += 1

        finally:
            release_lock(output_dir, wav_path)
            completed_counter.value += 1

    status_dict[worker_id] = "done"
    wlogger.info("Worker %d finished, %d failures", worker_id, failures)
    return failures


# ---------------------------------------------------------------------------
# Progress display (multi-worker)
# ---------------------------------------------------------------------------


def build_progress_table(
    num_workers: int,
    status_dict: dict,
    completed: int,
    total: int,
    elapsed: float,
    audio_processed_s: float,
) -> Table:
    """Build a rich Table for the live display."""
    remaining = total - completed
    pct = completed / total * 100 if total else 0
    bar_len = 30
    filled = int(pct / 100 * bar_len)
    bar = "#" * filled + "." * (bar_len - filled)

    if elapsed > 0 and audio_processed_s > 0:
        speed = (audio_processed_s / 60) / (elapsed / 60)
        speed_str = f"{speed:.2f}x realtime"
    else:
        speed_str = "calculating..."

    if completed > 0 and remaining > 0:
        avg_wall = elapsed / completed
        eta_s = avg_wall * remaining
        eta_h = int(eta_s // 3600)
        eta_m = int((eta_s % 3600) // 60)
        if eta_h > 0:
            eta_str = f"{eta_h}h {eta_m:02d}m"
        else:
            eta_str = f"{eta_m}m {int(eta_s % 60):02d}s"
    else:
        eta_str = "calculating..."

    elapsed_m = int(elapsed // 60)
    elapsed_s = int(elapsed % 60)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold", width=14)
    table.add_column("Value")

    table.add_row("Progress", f"[green]{bar}[/green]  {completed}/{total}  ({pct:.0f}%)")
    table.add_row("Elapsed", f"{elapsed_m}m {elapsed_s:02d}s")
    table.add_row("Speed", speed_str)
    table.add_row("ETA", eta_str)
    table.add_row("", "")

    for wid in range(num_workers):
        status = status_dict.get(wid, "idle")
        label = f"Worker {wid}"
        if status == "done":
            table.add_row(label, "[green]done[/green]")
        elif status.startswith("loading"):
            table.add_row(label, f"[yellow]{status}[/yellow]")
        else:
            table.add_row(label, status)

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        console.print(f"[red]Input directory not found: {args.input_dir}[/red]")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Validate WAV format on first file
    first_wav = next(args.input_dir.glob("*.wav"), None)
    if first_wav is None:
        console.print(f"[red]No .wav files found in {args.input_dir}[/red]")
        return 1
    validate_wav_format(first_wav)
    console.print(f"  [green]WAV format OK:[/green] 16kHz mono ({first_wav.name})")

    # Cleanup stale locks
    removed = cleanup_stale_locks(args.output_dir)
    if removed:
        console.print(f"  [yellow]Cleaned up {removed} stale lockfile(s)[/yellow]")

    # Discover files
    to_process, skipped = discover_files(args.input_dir, args.output_dir)
    total = len(to_process) + len(skipped)
    console.print(f"  Files: {total} total, {len(skipped)} already done, {len(to_process)} to process")
    console.print(f"  Workers: {args.workers}")

    if len(to_process) == 0:
        console.print("[green]Nothing to do — all files already processed.[/green]")
        return 0

    if args.dry_run:
        console.print(f"\n[yellow]DRY RUN — {len(to_process)} files across {args.workers} worker(s):[/yellow]")

        # Show round-robin assignment preview
        for wid in range(args.workers):
            chunk = [to_process[i] for i in range(wid, len(to_process), args.workers)]
            console.print(f"\n  [bold]Worker {wid}[/bold] ({len(chunk)} files):")
            for wav in chunk[:5]:
                dur = get_wav_duration(wav)
                console.print(f"    {wav.name}  ({dur / 60:.1f} min)")
            if len(chunk) > 5:
                console.print(f"    ... and {len(chunk) - 5} more")

        total_dur = sum(get_wav_duration(w) for w in to_process)
        console.print(f"\n  Total audio: {total_dur / 3600:.1f} hours across {len(to_process)} files")
        effective_speed = 1.5 * args.workers
        est_hours = (total_dur / 3600) / effective_speed
        console.print(f"  Estimated wall time: ~{est_hours:.1f} hours ({args.workers} workers at ~1.5x realtime each)")
        return 0

    # -----------------------------------------------------------------------
    # Single-worker fast path (no multiprocessing overhead)
    # -----------------------------------------------------------------------
    if args.workers == 1:
        return _run_single_worker(args, to_process, skipped, total)

    # -----------------------------------------------------------------------
    # Multi-worker path
    # -----------------------------------------------------------------------
    return _run_multi_worker(args, to_process, skipped, total)


def _run_single_worker(
    args: argparse.Namespace,
    to_process: list[Path],
    skipped: list[Path],
    total: int,
) -> int:
    """Original single-process loop — no locking needed."""
    log_file = _setup_main_logger(args.output_dir)

    logger = logging.getLogger("diarize")
    logger.info("Single-worker mode, %d files to process", len(to_process))

    console.print("\n  Loading pyannote pipeline...")
    hf_token = load_hf_token()

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(PIPELINE_MODEL, token=hf_token)
    pipeline.to(torch.device("cpu"))
    console.print("  [green]Pipeline loaded (CPU)[/green]\n")
    logger.info("Pipeline loaded: %s on CPU", PIPELINE_MODEL)

    completed = len(skipped)
    failed = 0
    audio_processed_s = 0.0
    start_time = time.perf_counter()

    for i, wav_path in enumerate(to_process):
        remaining = len(to_process) - i
        elapsed = time.perf_counter() - start_time

        _render_single_progress(
            wav_path.name,
            completed,
            total,
            remaining,
            elapsed,
            audio_processed_s,
        )

        try:
            result = diarize_file(pipeline, wav_path)
            out_path = args.output_dir / f"{wav_path.stem}_diarization.json"
            out_path.write_text(json.dumps(result, indent=2))

            duration = result["metadata"]["duration_seconds"]
            proc_time = result["metadata"]["processing_seconds"]
            n_speakers = result["metadata"]["num_speakers_detected"]
            audio_processed_s += duration

            logger.info(
                "OK  %-60s  dur=%.0fs  speakers=%d  proc=%.1fs",
                wav_path.name,
                duration,
                n_speakers,
                proc_time,
            )
            completed += 1

        except Exception as e:
            logger.error("FAIL %s: %s\n%s", wav_path.name, e, traceback.format_exc())
            write_failure(args.output_dir, wav_path, e)
            failed += 1
            completed += 1

    elapsed = time.perf_counter() - start_time
    _render_single_progress("done", completed, total, 0, elapsed, audio_processed_s)

    console.print()
    if failed:
        console.print(f"[yellow]Completed with {failed} failure(s)[/yellow]")
    else:
        console.print("[bold green]All files processed successfully[/bold green]")
    console.print(f"  Log: {log_file}")
    logger.info("Finished: %d processed, %d failed, %.0fs elapsed", completed, failed, elapsed)
    return 1 if failed else 0


def _run_multi_worker(
    args: argparse.Namespace,
    to_process: list[Path],
    skipped: list[Path],
    total: int,
) -> int:
    """Multi-process path with file-based locking and aggregated progress."""
    main_logger = _setup_main_logger(args.output_dir)
    logger = logging.getLogger("diarize")
    logger.info("Multi-worker mode: %d workers, %d files", args.workers, len(to_process))

    num_workers = min(args.workers, len(to_process))
    filenames = [p.name for p in to_process]

    # Round-robin partition into per-worker file lists
    worker_files: list[list[str]] = [[] for _ in range(num_workers)]
    for i, fname in enumerate(filenames):
        worker_files[i % num_workers].append(fname)

    # Shared state via multiprocessing Manager (all proxy objects, safe for Pool)
    manager = mp.Manager()
    status_dict = manager.dict()
    completed_counter = manager.Value("i", len(skipped))
    audio_processed_counter = manager.Value("d", 0.0)
    shutdown_flag = manager.dict()
    shutdown_flag["stop"] = False

    for wid in range(num_workers):
        status_dict[wid] = "starting..."

    console.print(f"\n  Launching {num_workers} worker(s)...\n")

    # Launch workers
    pool_args = [
        (
            wid,
            args.input_dir,
            args.output_dir,
            worker_files[wid],
            status_dict,
            completed_counter,
            audio_processed_counter,
            shutdown_flag,
        )
        for wid in range(num_workers)
    ]

    pool = mp.Pool(num_workers, maxtasksperchild=1)
    async_results = [pool.apply_async(worker_main, a) for a in pool_args]
    pool.close()  # no more tasks

    start_time = time.perf_counter()

    # Live progress display in main process
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                elapsed = time.perf_counter() - start_time
                done = completed_counter.value
                audio_s = audio_processed_counter.value

                table = build_progress_table(
                    num_workers,
                    status_dict,
                    done,
                    total,
                    elapsed,
                    audio_s,
                )
                live.update(table)

                # Check if all workers finished
                if all(r.ready() for r in async_results):
                    break

                time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — signaling workers to stop...[/yellow]")
        shutdown_flag["stop"] = True
        pool.terminate()
        pool.join()

        done = completed_counter.value
        elapsed = time.perf_counter() - start_time
        console.print(f"  Completed {done}/{total} files in {elapsed:.0f}s before interruption")
        console.print(f"  Log: {main_logger}")
        return 1

    pool.join()

    # Collect failure counts
    total_failures = 0
    for r in async_results:
        try:
            total_failures += r.get()
        except Exception as e:
            logger.error("Worker raised: %s", e)
            total_failures += 1

    elapsed = time.perf_counter() - start_time
    done = completed_counter.value
    audio_s = audio_processed_counter.value

    console.print()
    if total_failures:
        console.print(f"[yellow]Completed with {total_failures} failure(s)[/yellow]")
    else:
        console.print("[bold green]All files processed successfully[/bold green]")

    speed = (audio_s / 60) / (elapsed / 60) if elapsed > 0 and audio_s > 0 else 0
    console.print(f"  {done} files, {audio_s / 3600:.1f}h audio, {elapsed:.0f}s elapsed, {speed:.1f}x realtime")
    console.print(f"  Log: {main_logger}")
    logger.info(
        "Finished: %d files, %d failures, %.0fs elapsed, %.1fx realtime",
        done,
        total_failures,
        elapsed,
        speed,
    )
    return 1 if total_failures else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_main_logger(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"diarize_{timestamp}.log"

    logger = logging.getLogger("diarize")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    logger.addHandler(fh)

    logger.info("Log file: %s", log_file)
    return log_file


def _render_single_progress(
    current_file: str,
    completed: int,
    total: int,
    remaining: int,
    elapsed: float,
    audio_processed_s: float,
) -> None:
    """Single-worker progress display (no Live context needed)."""
    pct = completed / total * 100 if total else 0
    bar_len = 30
    filled = int(pct / 100 * bar_len)
    bar = "#" * filled + "." * (bar_len - filled)

    if elapsed > 0 and audio_processed_s > 0:
        speed = (audio_processed_s / 60) / (elapsed / 60)
        speed_str = f"{speed:.2f}x realtime"
    else:
        speed_str = "calculating..."

    if completed > 0 and remaining > 0:
        avg_wall = elapsed / completed
        eta_s = avg_wall * remaining
        eta_m = int(eta_s // 60)
        eta_str = f"{eta_m}m {int(eta_s % 60):02d}s"
    else:
        eta_str = "calculating..."

    elapsed_m = int(elapsed // 60)
    elapsed_s = int(elapsed % 60)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold", width=14)
    table.add_column("Value")

    table.add_row("Processing", current_file[:60])
    table.add_row("Progress", f"[green]{bar}[/green]  {completed}/{total}  ({pct:.0f}%)")
    table.add_row("Remaining", str(remaining))
    table.add_row("Elapsed", f"{elapsed_m}m {elapsed_s:02d}s")
    table.add_row("Speed", speed_str)
    table.add_row("ETA", eta_str)

    console.clear()
    console.print(table)


if __name__ == "__main__":
    sys.exit(main())
