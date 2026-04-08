"""Cross-encoder reranker using bge-reranker-v2-m3 via ONNX Runtime.

Reranks candidate chunks after RRF fusion by scoring each (query, passage)
pair through a cross-encoder.  Cross-encoders jointly attend to the query
and document, producing far more accurate relevance scores than the
independent embeddings used in bi-encoder retrieval — at the cost of
being O(n) in forward passes rather than a single query embed.

Architecture decision: We use ONNX Runtime on CPU rather than PyTorch
because (a) PyTorch can be unstable on integrated GPUs for BERT-family
models, (b) ONNX int8 quantized inference is fast enough for reranking
20-50 candidates, and (c) it matches the existing embedding pipeline's
approach.

Model: BAAI/bge-reranker-v2-m3 (278M params, MIT license)
- 51.8 nDCG@10 on BEIR benchmark
- ~130ms for 16 query-document pairs on CPU (short passages)
- ~3.5s for 20 candidates with 400+ token passages on CPU
- Chosen over the larger mxbai-rerank-base-v2 (0.5B params) for
  faster CPU inference; can be swapped later if quality needs improve.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Converts raw logits to 0-1 relevance probabilities.  Uses the
    standard split formula to avoid overflow: for x >= 0 use
    1/(1+exp(-x)), for x < 0 use exp(x)/(1+exp(x)).
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class Reranker:
    """Loads bge-reranker-v2-m3 ONNX model for cross-encoder reranking.

    Usage:
        reranker = Reranker(cache_dir="/path/to/bge-reranker-v2-m3-onnx")
        reranker.load()
        scores = reranker.score("What is the budget?", ["chunk1...", "chunk2..."])
    """

    def __init__(
        self,
        cache_dir: str = "",
        model_filename: str = "model_quantized.onnx",
        max_length: int = 512,
        expected_sha256: str = "",
    ) -> None:
        """Initialize with model location.

        Args:
            cache_dir: Path to the directory containing ONNX model + tokenizer.
            model_filename: Which ONNX variant to load.  model_quantized.onnx
                (INT8, 545MB) balances quality and CPU speed.  model.onnx +
                model.onnx_data is the full FP32 (2.2GB) for maximum quality.
            max_length: Maximum token length for query+passage pairs.  512 is
                the model's trained context window.  Longer passages are
                truncated, which is fine since our chunks are already <=512
                tokens from the embedding pipeline.
            expected_sha256: SHA-256 hex digest of the ONNX model file.
                Verified at load time to detect tampering.  If empty, a
                warning is logged but loading proceeds.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._model_filename = model_filename
        self._max_length = max_length
        self._expected_sha256 = expected_sha256
        self._session: ort.InferenceSession | None = None
        self._tokenizer = None
        self._ready = False

    @property
    def ready(self) -> bool:
        """Whether the model has been loaded and is ready for inference."""
        return self._ready

    def _verify_model_integrity(self, model_path: Path) -> None:
        """Verify the ONNX model file matches the expected SHA-256 digest.

        A tampered model file is the equivalent of a poisoned DNS resolver:
        the system returns answers, but they're the wrong answers.  This
        check catches both supply-chain attacks (tampered download) and
        local tampering (compromised filesystem).

        Args:
            model_path: Absolute path to the ONNX model file.

        Raises:
            RuntimeError: If the computed hash does not match the expected hash.
        """
        if not self._expected_sha256:
            logger.warning(
                "No expected SHA-256 configured for %s — skipping integrity check. "
                "Set RAG_RERANKER_MODEL_SHA256 to enable verification.",
                model_path.name,
            )
            return

        logger.info("Verifying model integrity for %s ...", model_path.name)
        sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
                sha256.update(chunk)
        computed = sha256.hexdigest()

        if computed != self._expected_sha256:
            raise RuntimeError(
                f"Model integrity check failed for {model_path.name}. "
                f"Expected SHA-256: {self._expected_sha256}, "
                f"computed: {computed}. "
                f"The model file may have been tampered with."
            )
        logger.info("Model integrity verified: SHA-256 matches expected digest.")

    def load(self) -> None:
        """Load the ONNX model and tokenizer from cache.

        Called once at application startup.  The model stays in memory
        for the lifetime of the process (~545MB for INT8 quantized).
        """
        from transformers import AutoTokenizer

        cache = self._cache_dir
        if cache is None:
            raise RuntimeError("No reranker model cache directory configured")

        # The ONNX files live in an 'onnx' subdirectory for this model.
        model_path = cache / "onnx" / self._model_filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"Reranker ONNX model not found at {model_path}. Download bge-reranker-v2-m3-ONNX from HuggingFace."
            )

        # Verify model file integrity before loading.  A tampered model
        # could silently manipulate relevance scores on every query.
        self._verify_model_integrity(model_path)

        logger.info("Loading reranker ONNX model from %s ...", model_path)
        t0 = time.time()

        # Configure ONNX Runtime for CPU with intra-op parallelism.
        # 8 intra-op threads is a sensible default for an 8+ core host;
        # tune lower on smaller machines or if the event loop starves.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer from the local cache directory only.
        # trust_remote_code=False prevents execution of custom tokenizer
        # classes.  local_files_only=True prevents network fallback.
        # B615 is a false positive here: revision pinning is for Hub
        # downloads; we load from a verified local directory with
        # model integrity already confirmed by _verify_model_integrity.
        self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            str(cache),
            trust_remote_code=False,
            local_files_only=True,
        )

        elapsed = time.time() - t0
        logger.info("Reranker ready in %.1fs (ONNX Runtime CPU, %s)", elapsed, self._model_filename)
        self._ready = True

    def score(
        self,
        query: str,
        passages: list[str],
        batch_size: int = 16,
    ) -> list[float]:
        """Score query-passage pairs and return relevance probabilities.

        Args:
            query: The user's search query.
            passages: List of passage texts to score against the query.
            batch_size: Number of pairs per inference batch.  16 is the
                sweet spot for CPU — large enough to amortize overhead,
                small enough to keep memory bounded.

        Returns:
            List of float scores in [0, 1] corresponding to each passage.
            Higher means more relevant.
        """
        if not self._ready:
            raise RuntimeError("Reranker not loaded — call load() first")

        if not passages:
            return []

        all_scores: list[float] = []

        # Process in batches to bound memory usage.  For typical
        # reranking (20 candidates), this is 1-2 batches.
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i : i + batch_size]

            # Tokenize as sentence pairs: (query, passage).  The cross-
            # encoder sees both texts jointly with [SEP] in between,
            # enabling full cross-attention — this is why it's more
            # accurate than bi-encoder similarity.
            inputs = self._tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="np",
            )

            # Run inference.  The model outputs a single logit per pair.
            ort_inputs = {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            }

            # Some ONNX exports include token_type_ids, some don't.
            # Check what the model expects and provide if needed.
            model_input_names = {inp.name for inp in self._session.get_inputs()}
            if "token_type_ids" in model_input_names and "token_type_ids" in inputs:
                ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

            outputs = self._session.run(None, ort_inputs)

            # Output shape is (batch_size, 1) — squeeze and sigmoid.
            logits = outputs[0].squeeze(-1)
            if logits.ndim == 0:
                logits = logits.reshape(1)
            scores = _sigmoid(logits)

            all_scores.extend(scores.tolist())

        return all_scores

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Score and sort passages by relevance to the query.

        Convenience method that calls score() and returns sorted indices.

        Args:
            query: The user's search query.
            passages: List of passage texts.
            top_k: If set, return only the top K results.

        Returns:
            List of (original_index, score) tuples sorted by score descending.
        """
        scores = self.score(query, passages)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores
