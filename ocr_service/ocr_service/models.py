"""Pydantic request/response models for the OCR service API."""

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    file_path: str
    force_ocr: bool = False
    use_surya: bool = True
    timeout_seconds: int = 300


class ExtractionResponse(BaseModel):
    status: str
    file_path: str
    text: str = ""
    source: str = ""
    page_count: int = 0
    char_count: int = 0
    image_count: int = Field(default=0, ge=0)
    processing_time_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    error: str = ""


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: str = ""
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    docling_ready: bool = False
    surya_ready: bool = False
    documents_processed: int = 0
    documents_since_last_reset: int = 0
    uptime_seconds: float = 0.0


class StatsResponse(BaseModel):
    total_processed: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    avg_processing_time_seconds: dict[str, float] = Field(default_factory=dict)
    memory_resets: int = 0
    worker_restarts: int = 0
