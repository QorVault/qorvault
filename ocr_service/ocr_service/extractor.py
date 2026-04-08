"""2-tier PDF extraction: digital (pypdf) then Docling + RapidOCR ONNX.

Tier 1 extracts embedded text with pypdf (fast, no OCR).
Tier 2 uses Docling with RapidOCR ONNX backend (CPU-only OCR for scans).

Earlier Surya/PyTorch GPU tiers were removed because some integrated GPU
drivers crash when PyTorch loads BERT-family models.  RapidOCR ONNX handles
scanned PDFs on CPU reliably and is the safe default.

GPU is hidden from the process entirely (CUDA_VISIBLE_DEVICES="") to
prevent Docling's layout models from attempting GPU inference.
"""

import gc
import logging
import os
import time
from pathlib import Path
from threading import Thread

# Hide GPU before any torch/docling import — prevents Docling/PyTorch from
# touching GPUs whose driver may crash on BERT-family layout/OCR models.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

logger = logging.getLogger(__name__)

RESET_INTERVAL = 50
MIN_DIGITAL_CHARS = 100


class Extractor:
    """Manages 2-tier PDF extraction with memory-safe converter lifecycle."""

    def __init__(self):
        self.converter = None
        self.docling_ready = False
        self.surya_ready = False  # Always False — kept for API compat
        self.documents_since_last_reset = 0
        self.memory_resets = 0
        self.total_processed = 0
        self.source_counts: dict[str, int] = {}
        self.source_times: dict[str, list[float]] = {}
        self._start_time = time.time()

    def initialize_background(self):
        """Start model loading in a background thread."""
        thread = Thread(target=self._initialize, daemon=True)
        thread.start()

    def _initialize(self):
        """Load the Docling converter. Called from background thread."""
        try:
            self._create_converter()
            self.docling_ready = True
            logger.info("Docling converter initialized (RapidOCR ONNX backend)")
        except Exception as e:
            logger.error("Failed to initialize converter: %s", e)

    def _create_converter(self):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=RapidOcrOptions(backend="onnxruntime"),
        )
        self.converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)})

    def _maybe_reset_memory(self):
        """Recreate converter every RESET_INTERVAL documents to avoid leaks."""
        if self.documents_since_last_reset < RESET_INTERVAL:
            return
        logger.info("Performing memory reset after %d documents", RESET_INTERVAL)
        self.converter = None
        gc.collect()
        try:
            self._create_converter()
            self.docling_ready = True
        except Exception as e:
            logger.warning("Failed to recreate converter: %s", e)
            self.docling_ready = False
        self.memory_resets += 1
        self.documents_since_last_reset = 0

    def _record(self, source: str, elapsed: float):
        self.total_processed += 1
        self.documents_since_last_reset += 1
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        self.source_times.setdefault(source, []).append(elapsed)

    def extract_pdf(self, file_path: str, force_ocr: bool = False, use_surya: bool = True) -> dict:
        """Run 2-tier extraction on a PDF file.

        use_surya is accepted but ignored (Surya removed due to GPU crashes).
        """
        start = time.time()
        basename = Path(file_path).name
        warnings: list[str] = []
        page_count = 0
        image_count = 0

        # TIER 1: Digital extraction with pypdf
        if not force_ocr:
            try:
                text, page_count, image_count = self._tier1_digital(file_path)
                stripped = text.strip()
                if len(stripped) >= MIN_DIGITAL_CHARS:
                    elapsed = time.time() - start
                    logger.info(
                        "Extracted %s: source=digital, %d chars, %d images, %.1fs",
                        basename,
                        len(stripped),
                        image_count,
                        elapsed,
                    )
                    self._record("digital", elapsed)
                    self._maybe_reset_memory()
                    return {
                        "status": "success",
                        "text": text,
                        "source": "digital",
                        "page_count": page_count,
                        "char_count": len(stripped),
                        "image_count": image_count,
                        "processing_time_seconds": round(elapsed, 2),
                        "warnings": warnings,
                    }
                else:
                    logger.debug("Tier 1 insufficient (%d chars), trying OCR", len(stripped))
            except Exception as e:
                logger.warning("Tier 1 error on %s: %s", basename, e)
                warnings.append(f"Tier 1 failed: {e}")

        # TIER 2: Docling + RapidOCR ONNX (CPU)
        if self.converter is not None:
            try:
                text, page_count, image_count = self._tier2_docling(file_path)
                stripped = text.strip()
                elapsed = time.time() - start
                if stripped:
                    logger.info(
                        "Extracted %s: source=docling_ocr, %d chars, %d images, %.1fs",
                        basename,
                        len(stripped),
                        image_count,
                        elapsed,
                    )
                    self._record("docling_ocr", elapsed)
                    self._maybe_reset_memory()
                    return {
                        "status": "success",
                        "text": text,
                        "source": "docling_ocr",
                        "page_count": page_count,
                        "char_count": len(stripped),
                        "image_count": image_count,
                        "processing_time_seconds": round(elapsed, 2),
                        "warnings": warnings,
                    }
                else:
                    warnings.append("Tier 2 produced no text")
            except Exception as e:
                logger.warning("Tier 2 error on %s: %s", basename, e)
                warnings.append(f"Tier 2 failed: {e}")
        else:
            warnings.append("Tier 2 skipped: converter not ready")

        # All tiers failed
        elapsed = time.time() - start
        self._record("failed", elapsed)
        self._maybe_reset_memory()
        return {
            "status": "error",
            "text": "",
            "source": "failed",
            "page_count": page_count,
            "char_count": 0,
            "image_count": image_count,
            "processing_time_seconds": round(elapsed, 2),
            "warnings": warnings,
            "error": "All extraction tiers failed",
        }

    def _tier1_digital(self, file_path: str) -> tuple[str, int, int]:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        image_count = 0
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
            try:
                image_count += len(page.images)
            except Exception:
                pass  # Don't fail extraction over image counting
        return "\n".join(pages), len(reader.pages), image_count

    def _tier2_docling(self, file_path: str) -> tuple[str, int, int]:
        result = self.converter.convert(file_path)
        text = result.document.export_to_markdown()
        num_pages = getattr(result.document, "num_pages", 0)
        page_count = num_pages() if callable(num_pages) else (num_pages or 0)
        image_count = len(getattr(result.document, "pictures", []))
        return text, page_count, image_count

    def get_gpu_info(self) -> dict:
        """Get GPU status information."""
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "gpu_available": True,
                    "gpu_name": name,
                    "gpu_memory_used_gb": 0.0,
                    "gpu_memory_total_gb": round(mem_total, 1),
                }
        except Exception:
            pass
        return {"gpu_available": False, "gpu_name": "", "gpu_memory_used_gb": 0, "gpu_memory_total_gb": 0}

    def get_stats(self) -> dict:
        avg_times = {}
        for source, times in self.source_times.items():
            avg_times[source] = round(sum(times) / len(times), 1) if times else 0.0
        return {
            "total_processed": self.total_processed,
            "by_source": dict(self.source_counts),
            "avg_processing_time_seconds": avg_times,
            "memory_resets": self.memory_resets,
        }
