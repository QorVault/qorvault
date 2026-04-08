"""FastAPI application for the OCR extraction service."""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .models import ExtractionRequest, ExtractionResponse, HealthResponse, StatsResponse
from .office_extractor import extract_office, is_office_file
from .worker import ExtractionWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

worker = ExtractionWorker()
extraction_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start worker subprocess in the background on startup."""
    logger.info("Starting extraction worker subprocess...")
    worker.initialize_background()
    yield
    logger.info("OCR service shutting down")
    worker.shutdown()


app = FastAPI(
    title="BoardDocs OCR Service",
    description="2-tier PDF extraction microservice with subprocess isolation",
    version="0.2.0",
    lifespan=lifespan,
)


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """Extract text from a PDF or Office file."""
    file_path = request.file_path

    # Validate file exists
    if not Path(file_path).is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Handle Office files directly (no lock needed, no subprocess)
    if is_office_file(file_path):
        start = time.time()
        try:
            text = extract_office(file_path)
            elapsed = time.time() - start
            stripped = text.strip()
            return ExtractionResponse(
                status="success",
                file_path=file_path,
                text=text,
                source="office",
                char_count=len(stripped),
                processing_time_seconds=round(elapsed, 2),
            )
        except Exception as e:
            logger.error("Office extraction failed for %s: %s", file_path, e)
            return ExtractionResponse(
                status="error",
                file_path=file_path,
                error=str(e),
            )

    # PDF extraction — one at a time through the worker subprocess
    if not worker.alive:
        if worker.starting:
            raise HTTPException(
                status_code=503,
                detail="Extraction worker initializing. Retry later.",
            )
        # Worker died unexpectedly — auto-restart
        logger.warning("Worker died unexpectedly, restarting...")
        worker.initialize_background()
        raise HTTPException(
            status_code=503,
            detail="Extraction worker restarting. Retry later.",
        )

    if extraction_lock.locked():
        raise HTTPException(
            status_code=503,
            detail="Service busy processing another document. Retry later.",
        )

    async with extraction_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: worker.extract_pdf(
                file_path,
                timeout=request.timeout_seconds,
                force_ocr=request.force_ocr,
                use_surya=request.use_surya,
            ),
        )

    return ExtractionResponse(file_path=file_path, **result)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Return service health and worker status."""
    gpu_info = worker.get_gpu_info()
    uptime = time.time() - worker._start_time
    return HealthResponse(
        status="healthy" if worker.alive else "degraded",
        gpu_available=gpu_info["gpu_available"],
        gpu_name=gpu_info.get("gpu_name", ""),
        gpu_memory_used_gb=gpu_info.get("gpu_memory_used_gb", 0.0),
        gpu_memory_total_gb=gpu_info.get("gpu_memory_total_gb", 0.0),
        docling_ready=worker.docling_ready,
        surya_ready=worker.surya_ready,
        documents_processed=worker.total_processed,
        documents_since_last_reset=worker.documents_since_last_reset,
        uptime_seconds=round(uptime, 1),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return extraction statistics."""
    return StatsResponse(**worker.get_stats())
