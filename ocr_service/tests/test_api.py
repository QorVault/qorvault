"""Tests for the FastAPI endpoints."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ocr_service.main as main_mod
from ocr_service.extractor import Extractor
from ocr_service.main import app


class _FakeWorker:
    """In-process worker for testing (no subprocess needed)."""

    def __init__(self):
        self._extractor = Extractor()
        self.docling_ready = False
        self.surya_ready = False
        self.total_processed = 0
        self.documents_since_last_reset = 0
        self.memory_resets = 0
        self.source_counts: dict[str, int] = {}
        self.source_times: dict[str, list[float]] = {}
        self._start_time = time.time()
        self.worker_restarts = 0

    @property
    def alive(self):
        return True

    @property
    def starting(self):
        return False

    def initialize_background(self):
        pass

    def shutdown(self):
        pass

    def extract_pdf(self, file_path, timeout=300, force_ocr=False, use_surya=True):
        result = self._extractor.extract_pdf(
            file_path,
            force_ocr=force_ocr,
            use_surya=use_surya,
        )
        source = result.get("source", "failed")
        self.total_processed += 1
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        return result

    def get_gpu_info(self):
        return {
            "gpu_available": False,
            "gpu_name": "",
            "gpu_memory_used_gb": 0,
            "gpu_memory_total_gb": 0,
        }

    def get_stats(self):
        return {
            "total_processed": self.total_processed,
            "by_source": dict(self.source_counts),
            "avg_processing_time_seconds": {},
            "memory_resets": 0,
            "worker_restarts": 0,
        }


@pytest.fixture
def client(monkeypatch):
    fake = _FakeWorker()
    monkeypatch.setattr(main_mod, "worker", fake)
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("healthy", "degraded")

    def test_health_includes_gpu_info(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "gpu_available" in data
        assert "gpu_name" in data

    def test_health_includes_readiness(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "docling_ready" in data
        assert "surya_ready" in data

    def test_health_includes_uptime(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] >= 0


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_processed" in data
        assert "by_source" in data

    def test_stats_has_expected_fields(self, client):
        resp = client.get("/stats")
        data = resp.json()
        assert "avg_processing_time_seconds" in data
        assert "memory_resets" in data
        assert "worker_restarts" in data


class TestExtractEndpoint:
    def test_extract_file_not_found(self, client):
        resp = client.post(
            "/extract",
            json={
                "file_path": "/nonexistent/file.pdf",
            },
        )
        assert resp.status_code == 404

    def test_extract_digital_pdf(self, client, digital_pdf: Path):
        resp = client.post(
            "/extract",
            json={
                "file_path": str(digital_pdf),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["source"] == "digital"
        assert data["char_count"] > 0

    def test_extract_docx(self, client, sample_docx: Path):
        resp = client.post(
            "/extract",
            json={
                "file_path": str(sample_docx),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["source"] == "office"
        assert "paragraph one" in data["text"]

    def test_extract_xlsx(self, client, sample_xlsx: Path):
        resp = client.post(
            "/extract",
            json={
                "file_path": str(sample_xlsx),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "office"
        assert "Alpha" in data["text"]

    def test_extract_returns_processing_time(self, client, digital_pdf: Path):
        resp = client.post(
            "/extract",
            json={
                "file_path": str(digital_pdf),
            },
        )
        data = resp.json()
        assert data["processing_time_seconds"] >= 0

    def test_extract_force_ocr_without_converters(self, client, digital_pdf: Path):
        """force_ocr with no converters loaded should fail gracefully."""
        resp = client.post(
            "/extract",
            json={
                "file_path": str(digital_pdf),
                "force_ocr": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
