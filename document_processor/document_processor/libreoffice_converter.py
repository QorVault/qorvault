"""Convert legacy Office formats (.doc, .ppt, .pps, .xls, etc.) to PDF via LibreOffice."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

CONVERTIBLE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pps", ".xls", ".xlsx"}


class ConversionError(Exception):
    """Raised when LibreOffice conversion fails."""


def convert_to_pdf(file_path: str, timeout: int = 60) -> str:
    """Convert an Office document to PDF using LibreOffice headless.

    Returns the path to the converted PDF.  The caller is responsible for
    cleaning up the returned file (and its parent temp directory).

    Setting HOME to the temp directory gives each invocation its own
    LibreOffice user profile, avoiding lock-file conflicts when multiple
    conversions run in parallel.
    """
    tmp_dir = tempfile.mkdtemp(prefix="lo_convert_")
    basename = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
    expected_pdf = os.path.join(tmp_dir, basename)

    env = os.environ.copy()
    env["HOME"] = tmp_dir  # isolated profile per conversion

    cmd = [
        "libreoffice",
        "--headless",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        tmp_dir,
        file_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise ConversionError(f"LibreOffice conversion timed out after {timeout}s: {file_path}")

    if result.returncode != 0:
        raise ConversionError(f"LibreOffice exited {result.returncode}: {result.stderr[:500]}")

    if not os.path.exists(expected_pdf):
        raise ConversionError(f"Conversion produced no PDF (expected {expected_pdf}): " f"stdout={result.stdout[:300]}")

    logger.debug("Converted %s → %s", file_path, expected_pdf)
    return expected_pdf
