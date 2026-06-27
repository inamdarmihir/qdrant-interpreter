"""Tests for UseCaseParamAdvisor."""

import pytest
from qdrant_client.models import Distance

from qdrant_interpreter_plugin.param_advisor import (
    CollectionParamAdvice,
    UseCaseParamAdvisor,
)


@pytest.fixture
def advisor() -> UseCaseParamAdvisor:
    return UseCaseParamAdvisor()


class TestDistanceSelection:
    def test_semantic_text_yields_cosine(self, advisor):
        advice = advisor.advise("semantic search over documents")
        assert advice.distance == Distance.COSINE

    def test_openai_embedding_yields_cosine(self, advisor):
        advice = advisor.advise("vector search using OpenAI ada-002 embeddings")
        assert advice.distance == Distance.COSINE

    def test_anomaly_detection_yields_euclid(self, advisor):
        advice = advisor.advise("anomaly detection on sensor data")
        assert advice.distance == Distance.EUCLID

    def test_clustering_yields_euclid(self, advisor):
        advice = advisor.advise("k-means clustering of tabular features")
        assert advice.distance == Distance.EUCLID

    def test_collaborative_filtering_yields_dot(self, advisor):
        advice = advisor.advise("collaborative filtering with matrix factorization")
        assert advice.distance == Distance.DOT

    def test_sparse_tfidf_yields_manhattan(self, advisor):
        advice = advisor.advise("sparse tfidf bag of words retrieval")
        assert advice.distance == Distance.MANHATTAN

    def test_bm25_yields_manhattan(self, advisor):
        advice = advisor.advise("BM25 sparse vector search")
        assert advice.distance == Distance.MANHATTAN

    def test_default_is_cosine(self, advisor):
        """Unrecognised use case defaults to COSINE."""
        advice = advisor.advise("some completely unrelated description")
        assert advice.distance == Distance.COSINE


class TestVectorSizeInference:
    def test_explicit_size_respected(self, advisor):
        advice = advisor.advise("semantic search", vector_size=512)
        assert advice.vector_size == 512

    def test_openai_keyword_gives_1536(self, advisor):
        advice = advisor.advise("search using OpenAI text-embedding-3-small")
        assert advice.vector_size == 1536

    def test_ada002_keyword_gives_1536(self, advisor):
        advice = advisor.advise("embeddings from ada-002 model")
        assert advice.vector_size == 1536

    def test_minilm_gives_384(self, advisor):
        advice = advisor.advise("sentence similarity using all-minilm-l6")
        assert advice.vector_size == 384

    def test_bert_base_gives_768(self, advisor):
        advice = advisor.advise("bert-base sentence encodings")
        assert advice.vector_size == 768

    def test_bare_dimension_number_extracted(self, advisor):
        advice = advisor.advise("custom embeddings with size 1024")
        assert advice.vector_size == 1024

    def test_dimension_keyword_extracted(self, advisor):
        advice = advisor.advise("embedding dimension=384 from MiniLM")
        assert advice.vector_size == 384

    def test_default_is_768(self, advisor):
        advice = advisor.advise("some unknown model without dimension hints")
        assert advice.vector_size == 768


class TestHnswConfig:
    def test_recall_small_gets_standard_defaults(self, advisor):
        advice = advisor.advise("text search", expected_count=50_000, priority="recall")
        assert advice.hnsw_config.m == 16
        assert advice.hnsw_config.ef_construct == 100

    def test_recall_medium_gets_higher_ef(self, advisor):
        advice = advisor.advise("text search", expected_count=500_000, priority="recall")
        assert advice.hnsw_config.m == 16
        assert advice.hnsw_config.ef_construct == 200

    def test_recall_large_gets_high_m_and_ef(self, advisor):
        advice = advisor.advise("text search", expected_count=2_000_000, priority="recall")
        assert advice.hnsw_config.m == 32
        assert advice.hnsw_config.ef_construct == 200

    def test_latency_priority_gets_lower_settings(self, advisor):
        advice = advisor.advise("fast search", expected_count=500_000, priority="latency")
        assert advice.hnsw_config.m == 16
        assert advice.hnsw_config.ef_construct == 100

    def test_memory_priority_gets_small_m(self, advisor):
        advice = advisor.advise("memory constrained", expected_count=500_000, priority="memory")
        assert advice.hnsw_config.m == 8
        assert advice.hnsw_config.ef_construct == 100


class TestQuantization:
    def test_small_recall_no_quantization(self, advisor):
        advice = advisor.advise("search", expected_count=50_000, priority="recall")
        assert advice.quantization_mode == "none"
        assert advice.quantization_config is None

    def test_large_recall_scalar_quantization(self, advisor):
        advice = advisor.advise("search", expected_count=500_000, priority="recall")
        assert advice.quantization_mode == "scalar"
        assert advice.quantization_config is not None

    def test_memory_priority_product_quantization(self, advisor):
        advice = advisor.advise("search", expected_count=500_000, priority="memory")
        assert advice.quantization_mode == "product"

    def test_latency_large_binary_quantization(self, advisor):
        advice = advisor.advise("search", expected_count=500_000, priority="latency")
        assert advice.quantization_mode == "binary"

    def test_low_memory_triggers_product_quantization(self, advisor):
        advice = advisor.advise("search", expected_count=200_000, memory_gb_available=2.0)
        assert advice.quantization_mode == "product"

    def test_high_memory_recall_large_uses_scalar(self, advisor):
        advice = advisor.advise("search", expected_count=200_000, memory_gb_available=32.0, priority="recall")
        assert advice.quantization_mode == "scalar"


class TestOnDiskPayload:
    def test_small_collection_payload_in_ram(self, advisor):
        advice = advisor.advise("search", expected_count=50_000, priority="recall")
        assert advice.on_disk_payload is False

    def test_large_collection_payload_on_disk(self, advisor):
        advice = advisor.advise("search", expected_count=2_000_000, priority="recall")
        assert advice.on_disk_payload is True

    def test_memory_priority_payload_on_disk(self, advisor):
        advice = advisor.advise("search", expected_count=50_000, priority="memory")
        assert advice.on_disk_payload is True


class TestRationale:
    def test_rationale_has_required_keys(self, advisor):
        advice = advisor.advise("semantic search with OpenAI embeddings", expected_count=100_000)
        for key in ("distance", "vector_size", "hnsw", "quantization", "on_disk_payload"):
            assert key in advice.rationale, f"Missing rationale key: {key}"

    def test_rationale_values_are_strings(self, advisor):
        advice = advisor.advise("anomaly detection", expected_count=500_000)
        for key, value in advice.rationale.items():
            assert isinstance(value, str), f"Rationale[{key}] is not a string"

    def test_rationale_not_empty(self, advisor):
        advice = advisor.advise("clustering", expected_count=200_000)
        for key, value in advice.rationale.items():
            assert len(value) > 0, f"Rationale[{key}] is empty"


class TestReturnType:
    def test_returns_collection_param_advice(self, advisor):
        advice = advisor.advise("semantic search", expected_count=100_000)
        assert isinstance(advice, CollectionParamAdvice)

    def test_hnsw_config_is_not_none(self, advisor):
        advice = advisor.advise("text search", expected_count=100_000)
        assert advice.hnsw_config is not None

    def test_optimizers_config_is_not_none(self, advisor):
        advice = advisor.advise("text search", expected_count=100_000)
        assert advice.optimizers_config is not None
