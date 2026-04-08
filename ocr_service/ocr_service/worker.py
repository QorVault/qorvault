"""Subprocess-based PDF extraction with hard timeout.

Runs Docling/OCR in a separate process that can be killed when it hangs
on problematic PDFs, preventing zombie threads and memory leaks.

Architecture:
  - Worker subprocess loads the Docling converter once at startup
  - Main process sends requests via Queue, receives results via Queue
  - On timeout: kill the subprocess (SIGKILL), start a new one
  - Stats tracked in the main process for the health/stats endpoints
"""

import logging
import multiprocessing as mp
import os
import queue
import time
from threading import Thread

logger = logging.getLogger(__name__)


def _worker_main(request_q: mp.Queue, result_q: mp.Queue) -> None:
    """Subprocess entry point. Loads converter once, processes requests forever."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""

    from .extractor import Extractor

    extractor = Extractor()
    extractor._initialize()

    result_q.put(
        {
            "type": "ready",
            "docling_ready": extractor.docling_ready,
        }
    )

    while True:
        try:
            request = request_q.get()
        except (EOFError, OSError):
            break

        try:
            result = extractor.extract_pdf(
                request["file_path"],
                force_ocr=request.get("force_ocr", False),
                use_surya=request.get("use_surya", True),
            )
            result_q.put({"type": "result", **result})
        except Exception as e:
            result_q.put(
                {
                    "type": "result",
                    "status": "error",
                    "text": "",
                    "source": "failed",
                    "page_count": 0,
                    "char_count": 0,
                    "processing_time_seconds": 0.0,
                    "warnings": [],
                    "error": str(e),
                }
            )


class ExtractionWorker:
    """Manages extraction in a killable subprocess.

    When extraction hangs (e.g., Docling loops on a problematic scanned PDF),
    the worker process is killed and restarted instead of leaving zombie
    threads that leak memory (previously grew to 42 GB).
    """

    def __init__(self) -> None:
        self._worker: mp.Process | None = None
        self._request_q: mp.Queue | None = None
        self._result_q: mp.Queue | None = None
        self._starting = False

        # Public attributes matching Extractor interface for health/stats
        self.docling_ready = False
        self.surya_ready = False  # Always False — API compat
        self.total_processed = 0
        self.documents_since_last_reset = 0
        self.memory_resets = 0
        self.source_counts: dict[str, int] = {}
        self.source_times: dict[str, list[float]] = {}
        self._start_time = time.time()
        self.worker_restarts = 0

    def initialize_background(self) -> None:
        """Start the worker subprocess in a background thread."""
        thread = Thread(target=self._start_worker, daemon=True)
        thread.start()

    def _start_worker(self) -> None:
        """Create and start the worker subprocess."""
        self._starting = True
        self.docling_ready = False

        ctx = mp.get_context("spawn")
        self._request_q = ctx.Queue()
        self._result_q = ctx.Queue()
        self._worker = ctx.Process(
            target=_worker_main,
            args=(self._request_q, self._result_q),
            daemon=True,
        )
        self._worker.start()
        logger.info("Started extraction worker (PID %d)", self._worker.pid)

        try:
            msg = self._result_q.get(timeout=180)
            if msg.get("type") == "ready":
                self.docling_ready = msg.get("docling_ready", False)
                logger.info("Extraction worker ready (docling=%s)", self.docling_ready)
        except queue.Empty:
            logger.error("Worker failed to initialize within 180s")
            self._kill_worker()
        finally:
            self._starting = False

    def _kill_worker(self) -> None:
        """Kill the worker subprocess and clean up queues."""
        if self._worker and self._worker.is_alive():
            pid = self._worker.pid
            logger.warning("Killing extraction worker (PID %d)", pid)
            self._worker.kill()
            self._worker.join(timeout=10)
            if self._worker.is_alive():
                logger.error("Worker PID %d did not die after SIGKILL", pid)

        self._worker = None
        self.docling_ready = False

        # Close old queues to release file descriptors
        for q in (self._request_q, self._result_q):
            if q is not None:
                try:
                    q.close()
                except Exception:
                    pass
        self._request_q = None
        self._result_q = None

    def shutdown(self) -> None:
        """Clean shutdown of the worker subprocess."""
        self._kill_worker()

    @property
    def alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    @property
    def starting(self) -> bool:
        return self._starting

    def extract_pdf(self, file_path: str, timeout: int = 300, force_ocr: bool = False, use_surya: bool = True) -> dict:
        """Send extraction to worker subprocess with hard timeout.

        If the worker hangs, kills it and starts a replacement.
        Returns a result dict (never raises).
        """
        if not self.alive:
            return {
                "status": "error",
                "text": "",
                "source": "failed",
                "page_count": 0,
                "char_count": 0,
                "processing_time_seconds": 0.0,
                "warnings": ["Worker not running"],
                "error": "Extraction worker not running",
            }

        start = time.time()
        self._request_q.put(
            {
                "file_path": file_path,
                "force_ocr": force_ocr,
                "use_surya": use_surya,
            }
        )

        try:
            msg = self._result_q.get(timeout=timeout)
            result = {k: v for k, v in msg.items() if k != "type"}
            source = result.get("source", "failed")
            elapsed = time.time() - start
            self._record(source, elapsed)
            return result

        except (queue.Empty, EOFError, OSError):
            # Timeout or worker crashed — kill and restart
            elapsed = time.time() - start
            basename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
            logger.error(
                "Extraction timed out after %.0fs for %s — killing worker",
                elapsed,
                basename,
            )
            self._kill_worker()
            self.worker_restarts += 1
            self._record("timeout", elapsed)

            # Restart worker in background
            Thread(target=self._start_worker, daemon=True).start()

            return {
                "status": "error",
                "text": "",
                "source": "timeout",
                "page_count": 0,
                "char_count": 0,
                "processing_time_seconds": round(elapsed, 2),
                "warnings": [f"Extraction killed after {timeout}s timeout"],
                "error": f"Extraction timed out after {timeout}s",
            }

    def _record(self, source: str, elapsed: float) -> None:
        self.total_processed += 1
        self.documents_since_last_reset += 1
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        self.source_times.setdefault(source, []).append(elapsed)

    def get_gpu_info(self) -> dict:
        return {
            "gpu_available": False,
            "gpu_name": "",
            "gpu_memory_used_gb": 0,
            "gpu_memory_total_gb": 0,
        }

    def get_stats(self) -> dict:
        avg_times = {}
        for source, times in self.source_times.items():
            avg_times[source] = round(sum(times) / len(times), 1) if times else 0.0
        return {
            "total_processed": self.total_processed,
            "by_source": dict(self.source_counts),
            "avg_processing_time_seconds": avg_times,
            "memory_resets": self.memory_resets,
            "worker_restarts": self.worker_restarts,
        }
