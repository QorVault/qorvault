"""Tests for the query embedder."""

from __future__ import annotations

import pytest

from rag_api.embedder import Embedder


def test_embed_query_raises_when_not_loaded():
    emb = Embedder(cache_dir="/nonexistent")
    with pytest.raises(RuntimeError, match="not loaded"):
        emb.embed_query("test query")


def test_embedder_ready_false_before_load():
    emb = Embedder(cache_dir="/nonexistent")
    assert emb.ready is False


def test_embedder_missing_cache_raises():
    emb = Embedder(cache_dir="/tmp/nonexistent_model_cache_12345")
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        emb.load()


def test_embedder_no_cache_dir_raises():
    emb = Embedder()
    with pytest.raises(RuntimeError, match="No model cache directory"):
        emb.load()
