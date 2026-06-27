"""Tests for the structured LangChain Qdrant tools."""

import json

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Skip entire module if langchain-core is not installed
pytest.importorskip("langchain_core")

from qdrant_interpreter_plugin.index_manager import QdrantIndexManager
from qdrant_interpreter_plugin.tools import make_qdrant_tools


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def manager(client) -> QdrantIndexManager:
    return QdrantIndexManager(client)


@pytest.fixture
def tools(manager) -> dict:
    """Return tools keyed by name for easy lookup."""
    tool_list = make_qdrant_tools(manager)
    return {t.name: t for t in tool_list}


@pytest.fixture
def populated_client(client) -> QdrantClient:
    """A client with a 'products' collection and 5 vectors already loaded."""
    client.create_collection(
        "products",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=i, vector=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
                    payload={"category": "shoes", "price": 9.99 * i})
        for i in range(1, 6)
    ]
    client.upsert("products", points=points)
    return client


@pytest.fixture
def populated_manager(populated_client) -> QdrantIndexManager:
    return QdrantIndexManager(populated_client)


@pytest.fixture
def populated_tools(populated_manager) -> dict:
    return {t.name: t for t in make_qdrant_tools(populated_manager)}


class TestToolRegistration:
    def test_returns_ten_tools(self, manager):
        tool_list = make_qdrant_tools(manager)
        assert len(tool_list) == 10

    def test_all_tool_names_present(self, tools):
        expected = {
            "create_qdrant_collection",
            "ensure_qdrant_collection",
            "upsert_qdrant_points",
            "search_qdrant_collection",
            "list_qdrant_collections",
            "describe_qdrant_collection",
            "create_qdrant_payload_index",
            "recommend_qdrant_params",
            "delete_qdrant_collection",
            "optimize_qdrant_collection",
        }
        assert set(tools.keys()) == expected

    def test_tools_have_invoke_method(self, tools):
        for name, t in tools.items():
            assert hasattr(t, "invoke"), f"Tool '{name}' missing .invoke()"


class TestCreateQdrantCollection:
    def test_success_message(self, tools):
        result = tools["create_qdrant_collection"].invoke({
            "collection_name": "test_col",
            "use_case": "semantic search with bert embeddings",
            "vector_size": 4,
        })
        assert "test_col" in result
        assert "ERROR" not in result

    def test_reports_vector_size(self, tools):
        result = tools["create_qdrant_collection"].invoke({
            "collection_name": "col",
            "use_case": "text search",
            "vector_size": 8,
        })
        assert "8" in result

    def test_reports_distance(self, tools):
        result = tools["create_qdrant_collection"].invoke({
            "collection_name": "col",
            "use_case": "semantic text search",
            "vector_size": 4,
        })
        assert "Cosine" in result

    def test_reports_rationale(self, tools):
        result = tools["create_qdrant_collection"].invoke({
            "collection_name": "col",
            "use_case": "anomaly detection",
            "vector_size": 4,
        })
        assert "ationale" in result  # "Rationale:" or "[rationale]"

    def test_duplicate_returns_error(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "dup",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["create_qdrant_collection"].invoke({
            "collection_name": "dup",
            "use_case": "search",
            "vector_size": 4,
        })
        assert "ERROR" in result

    def test_latency_priority(self, tools):
        result = tools["create_qdrant_collection"].invoke({
            "collection_name": "fast",
            "use_case": "fast retrieval",
            "vector_size": 4,
            "priority": "latency",
        })
        assert "ERROR" not in result

    def test_with_sparse_vectors(self, tools):
        result = tools["create_qdrant_collection"].invoke({
            "collection_name": "hybrid",
            "use_case": "hybrid search",
            "vector_size": 4,
            "sparse_vectors": True,
        })
        assert "ERROR" not in result
        assert "True" in result  # sparse_vectors: True in output


class TestEnsureQdrantCollection:
    def test_creates_when_absent(self, tools):
        result = tools["ensure_qdrant_collection"].invoke({
            "collection_name": "new",
            "use_case": "semantic search",
            "vector_size": 4,
        })
        assert "created" in result.lower()

    def test_reports_existing(self, tools):
        tools["ensure_qdrant_collection"].invoke({
            "collection_name": "existing",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["ensure_qdrant_collection"].invoke({
            "collection_name": "existing",
            "use_case": "search",
            "vector_size": 4,
        })
        assert "already exists" in result.lower()


class TestUpsertQdrantPoints:
    def test_upsert_success(self, tools):
        # First create the collection
        tools["create_qdrant_collection"].invoke({
            "collection_name": "docs",
            "use_case": "search",
            "vector_size": 4,
        })
        points = json.dumps([
            {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "payload": {"title": "doc1"}},
            {"id": 2, "vector": [0.5, 0.6, 0.7, 0.8], "payload": {"title": "doc2"}},
        ])
        result = tools["upsert_qdrant_points"].invoke({
            "collection_name": "docs",
            "points": points,
        })
        assert "2" in result
        assert "ERROR" not in result

    def test_upsert_invalid_json_returns_error(self, tools):
        result = tools["upsert_qdrant_points"].invoke({
            "collection_name": "docs",
            "points": "NOT VALID JSON",
        })
        assert "ERROR" in result

    def test_upsert_missing_id_returns_error(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "docs2",
            "use_case": "search",
            "vector_size": 4,
        })
        points = json.dumps([{"vector": [0.1, 0.2, 0.3, 0.4]}])
        result = tools["upsert_qdrant_points"].invoke({
            "collection_name": "docs2",
            "points": points,
        })
        assert "ERROR" in result


class TestSearchQdrantCollection:
    def test_search_returns_results(self, populated_tools):
        result = populated_tools["search_qdrant_collection"].invoke({
            "collection_name": "products",
            "query_vector": json.dumps([0.1, 0.2, 0.3, 0.4]),
            "limit": 3,
        })
        assert "Found" in result
        assert "score" in result

    def test_search_respects_limit(self, populated_tools):
        result = populated_tools["search_qdrant_collection"].invoke({
            "collection_name": "products",
            "query_vector": json.dumps([0.1, 0.2, 0.3, 0.4]),
            "limit": 2,
        })
        # At most 2 results in the output
        lines = [l for l in result.splitlines() if "id=" in l]
        assert len(lines) <= 2

    def test_search_invalid_json_returns_error(self, populated_tools):
        result = populated_tools["search_qdrant_collection"].invoke({
            "collection_name": "products",
            "query_vector": "not json",
        })
        assert "ERROR" in result

    def test_search_nonexistent_collection_returns_error(self, tools):
        result = tools["search_qdrant_collection"].invoke({
            "collection_name": "ghost",
            "query_vector": json.dumps([0.1, 0.2, 0.3, 0.4]),
        })
        assert "ERROR" in result


class TestListQdrantCollections:
    def test_empty_returns_message(self, tools):
        result = tools["list_qdrant_collections"].invoke({})
        assert "No collections" in result or "0 collection" in result

    def test_lists_existing_collections(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "alpha",
            "use_case": "search",
            "vector_size": 4,
        })
        tools["create_qdrant_collection"].invoke({
            "collection_name": "beta",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["list_qdrant_collections"].invoke({})
        assert "alpha" in result
        assert "beta" in result


class TestDescribeQdrantCollection:
    def test_describe_existing(self, populated_tools):
        result = populated_tools["describe_qdrant_collection"].invoke(
            {"collection_name": "products"}
        )
        assert "products" in result
        assert "status" in result.lower() or "Status" in result

    def test_describe_shows_points_count(self, populated_tools):
        result = populated_tools["describe_qdrant_collection"].invoke(
            {"collection_name": "products"}
        )
        assert "5" in result or "points" in result.lower()

    def test_describe_nonexistent_returns_error(self, tools):
        result = tools["describe_qdrant_collection"].invoke(
            {"collection_name": "ghost"}
        )
        assert "ERROR" in result


class TestCreateQdrantPayloadIndex:
    def test_keyword_index(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "col",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["create_qdrant_payload_index"].invoke({
            "collection_name": "col",
            "field_name": "category",
            "field_type": "keyword",
        })
        assert "category" in result
        assert "ERROR" not in result

    def test_integer_index(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "col2",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["create_qdrant_payload_index"].invoke({
            "collection_name": "col2",
            "field_name": "rank",
            "field_type": "integer",
        })
        assert "ERROR" not in result

    def test_invalid_type_returns_error(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "col3",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["create_qdrant_payload_index"].invoke({
            "collection_name": "col3",
            "field_name": "x",
            "field_type": "bogus_type",
        })
        assert "ERROR" in result


class TestRecommendQdrantParams:
    def test_returns_recommendation(self, tools):
        result = tools["recommend_qdrant_params"].invoke({
            "use_case": "semantic search over product descriptions using OpenAI ada-002",
            "expected_count": 500_000,
            "priority": "recall",
        })
        assert "Distance" in result or "distance" in result
        assert "Vector size" in result or "vector_size" in result

    def test_no_collection_created(self, tools, client):
        tools["recommend_qdrant_params"].invoke({
            "use_case": "semantic search",
            "vector_size": 4,
        })
        cols = client.get_collections().collections
        assert len(cols) == 0

    def test_includes_rationale(self, tools):
        result = tools["recommend_qdrant_params"].invoke({
            "use_case": "anomaly detection clustering",
        })
        assert "ationale" in result  # "Rationale:" heading


class TestDeleteQdrantCollection:
    def test_delete_existing(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "to_remove",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["delete_qdrant_collection"].invoke(
            {"collection_name": "to_remove"}
        )
        assert "to_remove" in result
        assert "ERROR" not in result

    def test_delete_nonexistent_is_idempotent(self, tools):
        # In-memory backend silently accepts delete of nonexistent collections
        result = tools["delete_qdrant_collection"].invoke(
            {"collection_name": "phantom"}
        )
        assert "ERROR" not in result


class TestOptimizeQdrantCollection:
    def test_optimize_success(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "opt",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["optimize_qdrant_collection"].invoke({
            "collection_name": "opt",
            "priority": "recall",
        })
        assert "opt" in result
        assert "ERROR" not in result

    def test_optimize_latency_priority(self, tools):
        tools["create_qdrant_collection"].invoke({
            "collection_name": "fast",
            "use_case": "search",
            "vector_size": 4,
        })
        result = tools["optimize_qdrant_collection"].invoke({
            "collection_name": "fast",
            "priority": "latency",
        })
        assert "ERROR" not in result
        assert "indexing_threshold" in result

    def test_optimize_nonexistent_returns_error(self, tools):
        result = tools["optimize_qdrant_collection"].invoke({
            "collection_name": "ghost",
        })
        assert "ERROR" in result
