"""Async HTTP client for the OCR extraction service."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
REQUEST_TIMEOUT = 300.0
CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive failures to trip
CIRCUIT_BREAKER_RESET_SECS = 60  # seconds before retrying after trip


@dataclass
class OCRResult:
    text: str
    source: str
    page_count: int
    char_count: int
    image_count: int = 0


class OCRError(Exception):
    """Raised when OCR extraction fails after retries."""


class OCRUnavailableError(OCRError):
    """Raised when the OCR service is unreachable (connection refused, etc.)."""


class OCRClient:
    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0))
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def extract(self, file_path: str) -> OCRResult:
        """POST to /extract and return extracted text.

        Retries on 5xx errors with linear backoff.  Uses a circuit breaker
        to avoid slow retries when the service is known to be down.
        """
        # Circuit breaker: fail fast if service was recently unreachable
        if self._consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            if time.monotonic() < self._circuit_open_until:
                raise OCRUnavailableError("OCR service unavailable (circuit open)")
            # Reset and allow a probe attempt
            logger.info("Circuit breaker: probing OCR service availability...")
            self._consecutive_failures = 0

        client = await self._get_client()
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.post(
                    f"{self.base_url}/extract",
                    json={"file_path": file_path},
                )

                if resp.status_code == 404:
                    raise OCRError(f"File not found: {file_path}")

                if resp.status_code >= 500:
                    last_error = (
                        OCRUnavailableError(f"OCR request failed: {resp.text}")
                        if resp.status_code == 503
                        else OCRError(f"OCR service error {resp.status_code}: {resp.text}")
                    )
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF_SECONDS * attempt
                        logger.warning(
                            "OCR %d/%d failed (HTTP %d), retry in %ds",
                            attempt,
                            MAX_RETRIES,
                            resp.status_code,
                            wait,
                        )
                        await asyncio.sleep(wait)
                        continue
                    # Trip circuit breaker for 503 (service busy/crashing)
                    if resp.status_code == 503:
                        self._consecutive_failures += 1
                        self._circuit_open_until = time.monotonic() + CIRCUIT_BREAKER_RESET_SECS
                    raise last_error

                resp.raise_for_status()
                data = resp.json()

                if data.get("status") != "success":
                    raise OCRError(f"OCR extraction failed: {data.get('error', 'unknown')}")

                # Success — reset circuit breaker
                self._consecutive_failures = 0
                # Coerce numeric fields to int — OCRResult is a plain
                # dataclass so won't reject bad types from the response.
                try:
                    img_count = int(data.get("image_count", 0))
                except (TypeError, ValueError):
                    img_count = 0

                return OCRResult(
                    text=data.get("text", ""),
                    source=data.get("source", ""),
                    page_count=data.get("page_count", 0),
                    char_count=data.get("char_count", 0),
                    image_count=max(img_count, 0),
                )

            except httpx.TimeoutException as e:
                self._consecutive_failures += 1
                self._circuit_open_until = time.monotonic() + CIRCUIT_BREAKER_RESET_SECS
                raise OCRUnavailableError(f"OCR request timed out after {REQUEST_TIMEOUT}s") from e
            except httpx.HTTPError as e:
                if isinstance(e, httpx.HTTPStatusError):
                    raise
                last_error = OCRUnavailableError(f"OCR service unreachable: {e}")
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "OCR %d/%d connection error, retry in %ds",
                        attempt,
                        MAX_RETRIES,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue
                # Trip the circuit breaker
                self._consecutive_failures += 1
                self._circuit_open_until = time.monotonic() + CIRCUIT_BREAKER_RESET_SECS
                raise last_error from e

        raise last_error or OCRError("OCR extraction failed after retries")
