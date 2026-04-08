"""Tests for embedding model loading and vector generation."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_pipeline.embedder import EXPECTED_DIM, DimensionError, Embedder


class TestEmbedder:
    def test_model_produces_1024_dimensions(self, embedder):
        """Model must produce exactly 1024-dimension vectors."""
        vecs = embedder.encode(["Hello world"])
        assert vecs.shape == (1, EXPECTED_DIM)

    def test_normalized_embeddings_unit_norm(self, embedder):
        """Normalized embeddings should have L2 norm within 0.001 of 1.0."""
        texts = [
            "The Kent School District Board of Directors met in regular session.",
            "Budget allocations for fiscal year 2024-2025 were approved.",
            "A short text.",
        ]
        vecs = embedder.encode(texts)
        for i in range(len(texts)):
            norm = float(np.linalg.norm(vecs[i]))
            assert abs(norm - 1.0) < 0.001, f"Vector {i} L2 norm is {norm}, expected ~1.0"

    def test_dimension_assertion_catches_wrong_dims(self):
        """If dimensions are wrong, DimensionError must be raised immediately."""
        embedder = Embedder()

        # Monkey-patch model and tokenizer to simulate wrong output dimensions
        class FakeOutput:
            def __init__(self, batch_size):
                self.last_hidden_state = np.random.randn(batch_size, 5, 512).astype(np.float32)

        class FakeModel:
            def __call__(self, **kwargs):
                batch_size = kwargs["input_ids"].shape[0]
                return FakeOutput(batch_size)

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                n = len(texts)
                return {
                    "input_ids": np.ones((n, 5), dtype=np.int64),
                    "attention_mask": np.ones((n, 5), dtype=np.int64),
                }

        embedder._model = FakeModel()
        embedder._tokenizer = FakeTokenizer()

        with pytest.raises(DimensionError, match="Expected 1024.*got 512"):
            embedder.encode(["test"])

    def test_batch_encoding(self, embedder):
        """Multiple texts produce correct batch shape."""
        texts = [f"Sentence number {i}" for i in range(10)]
        vecs = embedder.encode(texts)
        assert vecs.shape == (10, EXPECTED_DIM)
