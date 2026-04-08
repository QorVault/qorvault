"""Integration test against the running Qdrant instance."""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from embedding_pipeline.pipeline import REQUIRED_PAYLOAD_FIELDS

COLLECTION = "boarddocs_chunks"
QDRANT_URL = "http://localhost:6333"
EXPECTED_DIM = 1024


@pytest.fixture
def qdrant():
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    yield client
    client.close()


class TestQdrantIntegration:
    def test_upsert_retrieve_delete(self, qdrant):
        """Upsert a test point, retrieve it, verify payload, then delete."""
        point_id = str(uuid.uuid4())

        # Generate a normalized 1024-dim test vector
        vec = np.random.randn(EXPECTED_DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        payload = {
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(uuid.uuid4()),
            "tenant_id": "kent_sd",
            "content": "Integration test chunk content",
            "document_type": "agenda_item",
            "meeting_date": "2024-01-15",
            "committee_name": "Board of Directors",
            "meeting_id": "mtg_test",
            "agenda_item_id": "ai_test",
            "title": "Integration Test",
            "source_url": "https://example.com/test",
            "chunk_index": 0,
            "token_count": 10,
            "contains_table": False,
        }

        # Upsert
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=point_id, vector=vec.tolist(), payload=payload)],
        )

        # Retrieve
        results = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        assert len(results) == 1
        point = results[0]

        # Verify all required fields present in payload
        missing = REQUIRED_PAYLOAD_FIELDS - set(point.payload.keys())
        assert not missing, f"Missing payload fields: {missing}"

        # Verify vector dimension
        assert len(point.vector) == EXPECTED_DIM

        # Verify payload values
        assert point.payload["tenant_id"] == "kent_sd"
        assert point.payload["content"] == "Integration test chunk content"

        # Clean up
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=[point_id],
        )

        # Verify deletion
        results = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=[point_id],
        )
        assert len(results) == 0
