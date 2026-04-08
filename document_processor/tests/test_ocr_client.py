"""Tests for OCR client with mocked httpx."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from document_processor.ocr_client import OCRClient, OCRError, OCRResult


@pytest.fixture
def ocr_client():
    return OCRClient("http://localhost:8001")


def _mock_response(status_code: int = 200, json_data: dict | None = None, text: str = "") -> httpx.Response:
    """Create a mock httpx Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    return resp


class TestOCRClient:
    @pytest.mark.asyncio
    async def test_successful_extraction(self, ocr_client):
        """Successful OCR returns text and metadata."""
        mock_resp = _mock_response(
            200,
            {
                "status": "success",
                "text": "Extracted document text",
                "source": "digital",
                "page_count": 3,
                "char_count": 22,
            },
        )

        with patch.object(ocr_client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get.return_value = mock_client

            result = await ocr_client.extract("/path/to/doc.pdf")

        assert isinstance(result, OCRResult)
        assert result.text == "Extracted document text"
        assert result.source == "digital"
        assert result.page_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_503(self, ocr_client):
        """Client retries on HTTP 503 (service busy)."""
        resp_503 = _mock_response(503, text="Service busy")
        resp_503.raise_for_status = MagicMock()  # 503 doesn't raise, we check status_code
        resp_ok = _mock_response(
            200,
            {
                "status": "success",
                "text": "Got it",
                "source": "docling_easyocr",
                "page_count": 1,
                "char_count": 6,
            },
        )

        with patch.object(ocr_client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [resp_503, resp_ok]
            mock_get.return_value = mock_client

            with patch("document_processor.ocr_client.asyncio.sleep", new_callable=AsyncMock):
                result = await ocr_client.extract("/path/to/doc.pdf")

        assert result.text == "Got it"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_500(self, ocr_client):
        """Client retries on HTTP 500 (server error)."""
        resp_500 = _mock_response(500, text="Internal error")
        resp_500.raise_for_status = MagicMock()
        resp_ok = _mock_response(
            200,
            {
                "status": "success",
                "text": "Recovered",
                "source": "digital",
                "page_count": 1,
                "char_count": 9,
            },
        )

        with patch.object(ocr_client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.side_effect = [resp_500, resp_ok]
            mock_get.return_value = mock_client

            with patch("document_processor.ocr_client.asyncio.sleep", new_callable=AsyncMock):
                result = await ocr_client.extract("/path/to/doc.pdf")

        assert result.text == "Recovered"

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self, ocr_client):
        """Timeout raises OCRError."""
        with patch.object(ocr_client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ReadTimeout("timeout")
            mock_get.return_value = mock_client

            with pytest.raises(OCRError, match="timed out"):
                await ocr_client.extract("/path/to/doc.pdf")

    @pytest.mark.asyncio
    async def test_file_not_found_404(self, ocr_client):
        """404 response raises OCRError immediately (no retry)."""
        resp_404 = _mock_response(404)

        with patch.object(ocr_client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = resp_404
            mock_get.return_value = mock_client

            with pytest.raises(OCRError, match="File not found"):
                await ocr_client.extract("/nonexistent.pdf")

        # Should not retry on 404
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, ocr_client):
        """After max retries, raises OCRError."""
        resp_500 = _mock_response(500, text="Server error")
        resp_500.raise_for_status = MagicMock()

        with patch.object(ocr_client, "_get_client") as mock_get:
            mock_client = AsyncMock()
            mock_client.post.return_value = resp_500
            mock_get.return_value = mock_client

            with patch("document_processor.ocr_client.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(OCRError, match="OCR service error 500"):
                    await ocr_client.extract("/path/to/doc.pdf")

        assert mock_client.post.call_count == 3
