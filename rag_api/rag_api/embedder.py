"""Query embedding via ONNX Runtime (mxbai-embed-large-v1)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_DIM = 1024


class Embedder:
    """Loads mxbai-embed-large-v1 ONNX model for query embedding."""

    def __init__(self, cache_dir: str = "") -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._tokenizer = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Load the ONNX model and tokenizer from cache."""
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        cache = self._cache_dir
        if cache is None:
            raise RuntimeError("No model cache directory configured")

        if not (cache / "model.onnx").exists():
            raise FileNotFoundError(
                f"ONNX model not found at {cache}. " f"Run embedding_pipeline first to export the model."
            )

        logger.info("Loading ONNX model from %s ...", cache)
        t0 = time.time()
        self._model = ORTModelForFeatureExtraction.from_pretrained(
            str(cache),
            provider="CPUExecutionProvider",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(str(cache))
        elapsed = time.time() - t0
        logger.info("Embedder ready in %.1fs (ONNX Runtime CPU)", elapsed)
        self._ready = True

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns 1024-dim float list."""
        if not self._ready:
            raise RuntimeError("Embedder not loaded — call load() first")

        inputs = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        outputs = self._model(**inputs)
        token_embeddings = outputs.last_hidden_state  # (1, seq_len, 1024)

        # Mean pooling weighted by attention mask
        mask = inputs["attention_mask"]  # (1, seq_len)
        mask_expanded = np.expand_dims(mask, axis=-1)  # (1, seq_len, 1)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)  # (1, 1024)
        counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_pooled = summed / counts  # (1, 1024)

        # L2 normalize to unit length
        norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        normalized = (mean_pooled / norms).astype(np.float32)

        vec = normalized[0]
        if vec.shape[0] != EXPECTED_DIM:
            raise ValueError(f"Expected {EXPECTED_DIM} dimensions, got {vec.shape[0]}")

        return vec.tolist()
