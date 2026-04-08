"""Tests for the Extractor class and tier logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from ocr_service.extractor import MIN_DIGITAL_CHARS, RESET_INTERVAL, Extractor
from ocr_service.office_extractor import extract_office, is_office_file


class TestTier1Digital:
    """Tests for digital PDF extraction (pypdf)."""

    def test_digital_pdf_extracted(self, digital_pdf: Path):
        ext = Extractor()
        result = ext.extract_pdf(str(digital_pdf))
        assert result["status"] == "success"
        assert result["source"] == "digital"
        assert result["char_count"] >= MIN_DIGITAL_CHARS
        assert result["page_count"] == 1

    def test_minimal_pdf_falls_through(self, minimal_pdf: Path):
        """PDF with too little text should not return digital source."""
        ext = Extractor()
        result = ext.extract_pdf(str(minimal_pdf))
        assert result["source"] != "digital"

    def test_force_ocr_skips_digital(self, digital_pdf: Path):
        """force_ocr=True should skip Tier 1 entirely."""
        ext = Extractor()
        result = ext.extract_pdf(str(digital_pdf), force_ocr=True)
        assert result["source"] != "digital"

    def test_page_count_matches(self, digital_pdf: Path):
        ext = Extractor()
        result = ext.extract_pdf(str(digital_pdf))
        assert result["page_count"] == 1

    def test_processing_time_recorded(self, digital_pdf: Path):
        ext = Extractor()
        result = ext.extract_pdf(str(digital_pdf))
        assert result["processing_time_seconds"] >= 0


class TestTier2Docling:
    """Tests for Docling RapidOCR ONNX fallback."""

    def test_tier2_called_when_digital_insufficient(self, minimal_pdf: Path):
        ext = Extractor()
        mock_converter = MagicMock()
        mock_doc = MagicMock()
        mock_doc.export_to_markdown.return_value = "OCR extracted text " * 20
        mock_doc.num_pages = 1
        mock_result = MagicMock()
        mock_result.document = mock_doc
        mock_converter.convert.return_value = mock_result
        ext.converter = mock_converter

        result = ext.extract_pdf(str(minimal_pdf))
        assert result["source"] == "docling_ocr"
        mock_converter.convert.assert_called_once()

    def test_tier2_skipped_when_converter_none(self, minimal_pdf: Path):
        ext = Extractor()
        ext.converter = None
        result = ext.extract_pdf(str(minimal_pdf))
        assert "Tier 2 skipped" in str(result["warnings"])


class TestMemoryManagement:
    """Tests for converter reset logic."""

    def test_reset_interval_triggers(self):
        ext = Extractor()
        ext.documents_since_last_reset = RESET_INTERVAL
        ext.docling_ready = True
        with patch.object(ext, "_create_converter"):
            ext._maybe_reset_memory()
        assert ext.documents_since_last_reset == 0
        assert ext.memory_resets == 1

    def test_no_reset_below_interval(self):
        ext = Extractor()
        ext.documents_since_last_reset = RESET_INTERVAL - 1
        ext._maybe_reset_memory()
        assert ext.memory_resets == 0


class TestStats:
    """Tests for stats tracking."""

    def test_stats_after_extraction(self, digital_pdf: Path):
        ext = Extractor()
        ext.extract_pdf(str(digital_pdf))
        stats = ext.get_stats()
        assert stats["total_processed"] == 1
        assert "digital" in stats["by_source"]

    def test_empty_stats(self):
        ext = Extractor()
        stats = ext.get_stats()
        assert stats["total_processed"] == 0
        assert stats["by_source"] == {}


class TestOfficeExtractor:
    """Tests for Office file extraction."""

    def test_is_office_file(self):
        assert is_office_file("test.docx") is True
        assert is_office_file("test.pdf") is False
        assert is_office_file("test.xlsx") is True
        assert is_office_file("test.pptx") is True

    def test_extract_docx(self, sample_docx: Path):
        text = extract_office(str(sample_docx))
        assert "paragraph one" in text
        assert "paragraph two" in text

    def test_extract_xlsx(self, sample_xlsx: Path):
        text = extract_office(str(sample_xlsx))
        assert "Name" in text
        assert "Alpha" in text

    def test_extract_pptx(self, sample_pptx: Path):
        text = extract_office(str(sample_pptx))
        assert "Test Slide Title" in text
