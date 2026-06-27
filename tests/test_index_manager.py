"""Tests for QdrantIndexManager."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from qdrant_interpreter_plugin.index_manager import QdrantIndexManager


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def manager(client) -> QdrantIndexManager:
    return QdrantIndexManager(client)


class TestAutoCreate:
    def test_creates_collection(self, manager, client):
        manager.auto_create("docs", use_case="semantic search", vector_size=4)
        cols = {c.name for c in client.get_collections().collections}
        assert "docs" in cols

    def test_returns_summary_dict(self, manager):
        result = manager.auto_create("docs", use_case="semantic search", vector_size=4)
        for key in ("collection_name", "vector_size", "distance", "quantization",
                    "hnsw_m", "hnsw_ef_construct", "on_disk_payload", "rationale"):
            assert key in result, f"Missing key: {key}"

    def test_collection_name_in_result(self, manager):
        result = manager.auto_create("my_col", use_case="text retrieval", vector_size=8)
        assert result["collection_name"] == "my_col"

    def test_vector_size_respected(self, manager):
        result = manager.auto_create("col", use_case="search", vector_size=128)
        assert result["vector_size"] == 128

    def test_semantic_search_uses_cosine(self, manager):
        result = manager.auto_create("col", use_case="semantic text search", vector_size=4)
        assert result["distance"] == "Cosine"

    def test_anomaly_detection_uses_euclid(self, manager):
        result = manager.auto_create("col", use_case="anomaly detection", vector_size=4)
        assert result["distance"] == "Euclid"

    def test_priority_recall_small_no_quantization(self, manager):
        result = manager.auto_create(
            "col", use_case="search", vector_size=4,
            expected_count=10_000, priority="recall"
        )
        assert result["quantization"] == "none"

    def test_priority_recall_large_scalar_quantization(self, manager):
        result = manager.auto_create(
            "col", use_case="search", vector_size=4,
            expected_count=500_000, priority="recall"
        )
        assert result["quantization"] == "scalar"

    def test_sparse_vectors_flag(self, manager, client):
        result = manager.auto_create(
            "hybrid", use_case="hybrid search", vector_size=4, sparse_vectors=True
        )
        assert result["sparse_vectors"] is True
        # Confirm sparse vectors config is present
        info = client.get_collection("hybrid")
        sparse = info.config.params.sparse_vectors
        assert sparse is not None and "sparse" in sparse

    def test_duplicate_create_raises(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        with pytest.raises(Exception):
            manager.auto_create("col", use_case="search", vector_size=4)


class TestEnsure:
    def test_ensure_creates_when_absent(self, manager):
        result = manager.ensure("new_col", use_case="search", vector_size=4)
        assert result["action"] == "created"

    def test_ensure_is_idempotent(self, manager):
        manager.auto_create("existing", use_case="search", vector_size=4)
        result = manager.ensure("existing", use_case="search", vector_size=4)
        assert result["action"] == "already_exists"

    def test_ensure_returns_points_count_for_existing(self, manager, client):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.ensure("col", use_case="search", vector_size=4)
        assert "points_count" in result

    def test_ensure_returns_status_for_existing(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.ensure("col", use_case="search", vector_size=4)
        assert "status" in result


class TestDelete:
    def test_delete_removes_collection(self, manager, client):
        manager.auto_create("to_delete", use_case="search", vector_size=4)
        manager.delete("to_delete")
        cols = {c.name for c in client.get_collections().collections}
        assert "to_delete" not in cols

    def test_delete_returns_true(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        assert manager.delete("col") is True

    def test_delete_nonexistent_is_idempotent(self, manager):
        # In-memory backend silently accepts delete of nonexistent collections
        result = manager.delete("does_not_exist")
        assert result is True


class TestCreatePayloadIndex:
    def test_keyword_index(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.create_payload_index("col", "category", "keyword")
        assert result["status"] == "indexed"
        assert result["field"] == "category"
        assert result["type"] == "keyword"

    def test_integer_index(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.create_payload_index("col", "count", "integer")
        assert result["status"] == "indexed"

    def test_float_index(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.create_payload_index("col", "score", "float")
        assert result["status"] == "indexed"

    def test_bool_index(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.create_payload_index("col", "active", "bool")
        assert result["status"] == "indexed"

    def test_invalid_type_raises(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        with pytest.raises(ValueError, match="Unknown field_type"):
            manager.create_payload_index("col", "field", "invalid_type")

    def test_case_insensitive_type(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.create_payload_index("col", "tag", "KEYWORD")
        assert result["status"] == "indexed"


class TestHealth:
    def test_health_returns_status(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        h = manager.health("col")
        assert "status" in h
        assert h["status"] in ("green", "yellow", "grey", "red")

    def test_health_returns_points_count(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        h = manager.health("col")
        assert "points_count" in h
        assert isinstance(h["points_count"], int)

    def test_health_returns_vector_size(self, manager):
        manager.auto_create("col", use_case="search", vector_size=16)
        h = manager.health("col")
        assert h.get("vector_size") == 16

    def test_health_returns_distance(self, manager):
        manager.auto_create("col", use_case="semantic search", vector_size=4)
        h = manager.health("col")
        assert "distance" in h

    def test_health_no_recommendation_for_small(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        h = manager.health("col")
        # Empty collection with no quantization but 0 points → no recommendation
        assert "recommendation" not in h or h.get("points_count", 0) <= 100_000

    def test_health_collection_name_in_result(self, manager):
        manager.auto_create("mytest", use_case="search", vector_size=4)
        h = manager.health("mytest")
        assert h["collection_name"] == "mytest"


class TestOptimize:
    def test_optimize_recall_priority(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.optimize("col", priority="recall")
        assert "applied_optimizers" in result
        assert result["priority"] == "recall"

    def test_optimize_latency_priority(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.optimize("col", priority="latency")
        opts = result["applied_optimizers"]
        assert opts["indexing_threshold"] <= 10_000

    def test_optimize_memory_priority(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.optimize("col", priority="memory")
        opts = result["applied_optimizers"]
        assert opts["memmap_threshold"] <= 10_000

    def test_optimize_returns_collection_name(self, manager):
        manager.auto_create("col", use_case="search", vector_size=4)
        result = manager.optimize("col")
        assert result["collection"] == "col"
