"""Model loading and embedding generation via ONNX Runtime."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
EXPECTED_DIM = 1024
ENCODE_BATCH_SIZE = 64

# ONNX model is exported once, then reused from this directory.
MODEL_CACHE_DIR = Path(__file__).resolve().parent.parent / "model_cache" / "mxbai-embed-large-v1-onnx"


class DimensionError(Exception):
    """Raised when embedding dimensions don't match expected value."""


class Embedder:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load or export the ONNX model and tokenizer."""
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        cache = MODEL_CACHE_DIR

        if (cache / "model.onnx").exists():
            logger.info("Loading cached ONNX model from %s ...", cache)
            t0 = time.time()
            self._model = ORTModelForFeatureExtraction.from_pretrained(
                str(cache),
                provider="CPUExecutionProvider",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(cache))
        else:
            logger.info(
                "Exporting %s to ONNX (first run, may take a minute) ...",
                MODEL_NAME,
            )
            t0 = time.time()
            self._model = ORTModelForFeatureExtraction.from_pretrained(
                MODEL_NAME,
                export=True,
                provider="CPUExecutionProvider",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            cache.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(str(cache))
            self._tokenizer.save_pretrained(str(cache))
            logger.info("ONNX model saved to %s", cache)

        elapsed = time.time() - t0
        logger.info("Model loaded in %.1fs (ONNX Runtime CPU)", elapsed)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts and return (N, 1024) normalized embeddings.

        Raises DimensionError if output dimensions don't match.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded — call load() first")

        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), ENCODE_BATCH_SIZE):
            batch_texts = texts[start : start + ENCODE_BATCH_SIZE]
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state  # (B, seq_len, dim)

            # Mean pooling weighted by attention mask
            mask = inputs["attention_mask"]  # (B, seq_len)
            mask_expanded = np.expand_dims(mask, axis=-1)  # (B, seq_len, 1)
            summed = np.sum(token_embeddings * mask_expanded, axis=1)  # (B, dim)
            counts = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)  # (B, 1)
            mean_pooled = summed / counts  # (B, dim)

            # L2 normalize to unit length
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            normalized = mean_pooled / norms

            all_embeddings.append(normalized.astype(np.float32))

        embeddings = np.concatenate(all_embeddings, axis=0)

        if embeddings.shape[1] != EXPECTED_DIM:
            raise DimensionError(
                f"FATAL: Expected {EXPECTED_DIM} dimensions, got {embeddings.shape[1]}. "
                f"Aborting to prevent Qdrant collection corruption."
            )

        return embeddings
