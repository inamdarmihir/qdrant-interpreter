"""
Qdrant Index Manager
====================

High-level collection management built on top of :class:`UseCaseParamAdvisor`.
The manager wraps a :class:`qdrant_client.QdrantClient` and provides
auto-parameterised CRUD, payload indexing, health checks, and optimisation
in a single, easy-to-use API.

Typical agent usage::

    from qdrant_client import QdrantClient
    from qdrant_interpreter_plugin.index_manager import QdrantIndexManager

    manager = QdrantIndexManager(QdrantClient(":memory:"))

    # Create with auto-selected parameters
    result = manager.auto_create(
        collection_name="products",
        use_case="semantic search over product descriptions using OpenAI ada-002",
        expected_count=500_000,
        priority="recall",
    )
    # result["distance"] == "Cosine", result["quantization"] == "scalar", ...

    # Add a payload index for fast filtered queries
    manager.create_payload_index("products", "category", "keyword")

    # Inspect health and get recommendations
    health = manager.health("products")
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from .param_advisor import PriorityHint, UseCaseParamAdvisor


class QdrantIndexManager:
    """Manages Qdrant collections with automatic parameter selection.

    Wraps a :class:`qdrant_client.QdrantClient` and adds a layer of
    use-case-aware intelligence via :class:`UseCaseParamAdvisor`. All
    operations return plain dicts so they integrate cleanly with LangChain
    tools and agent loops.

    Parameters
    ----------
    client:
        An already-constructed ``QdrantClient`` instance. The manager does
        not take ownership of the client's lifetime.
    """

    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        self._advisor = UseCaseParamAdvisor()

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def auto_create(
        self,
        collection_name: str,
        use_case: str,
        vector_size: int | None = None,
        expected_count: int = 100_000,
        priority: PriorityHint = "recall",
        memory_gb_available: float | None = None,
        sparse_vectors: bool = False,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a collection using automatically recommended parameters.

        Parameters
        ----------
        collection_name:
            Name for the new collection.
        use_case:
            Natural-language description of the intended use, e.g.
            ``"semantic search over product descriptions using OpenAI ada-002"``.
        vector_size:
            Vector dimension. Inferred from ``use_case`` keywords if not given.
        expected_count:
            Approximate number of vectors; influences HNSW and quantization.
        priority:
            Optimisation goal: ``"recall"``, ``"latency"``, or ``"memory"``.
        memory_gb_available:
            Available RAM in GB; drives quantization aggressiveness.
        sparse_vectors:
            When ``True``, also configure a ``"sparse"`` named sparse-vector
            field for hybrid (dense + sparse) search.
        overrides:
            Dict of parameter overrides merged on top of the advisor
            recommendations. Recognised keys:

            * ``"vector_size"`` — override inferred dimension
            * ``"distance"`` — a :class:`qdrant_client.models.Distance` value
            * ``"on_disk_vectors"`` — bool, store vectors on disk
            * ``"on_disk_payload"`` — bool
            * ``"quantization_config"`` — a quantization config object

        Returns
        -------
        dict
            Summary of the applied parameters and the advisor's rationale.
            Keys: ``collection_name``, ``vector_size``, ``distance``,
            ``quantization``, ``hnsw_m``, ``hnsw_ef_construct``,
            ``on_disk_payload``, ``sparse_vectors``, ``rationale``.
        """
        advice = self._advisor.advise(
            use_case=use_case,
            vector_size=vector_size,
            expected_count=expected_count,
            priority=priority,
            memory_gb_available=memory_gb_available,
        )
        overrides = overrides or {}

        # Sparse vector configuration
        sparse_cfg: dict[str, SparseVectorParams] | None = None
        if sparse_vectors:
            sparse_cfg = {
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=overrides.get("vector_size", advice.vector_size),
                distance=overrides.get("distance", advice.distance),
                on_disk=overrides.get("on_disk_vectors", False),
            ),
            sparse_vectors_config=sparse_cfg,
            hnsw_config=advice.hnsw_config,
            optimizers_config=advice.optimizers_config,
            quantization_config=overrides.get(
                "quantization_config", advice.quantization_config
            ),
            on_disk_payload=overrides.get("on_disk_payload", advice.on_disk_payload),
        )

        return {
            "collection_name": collection_name,
            "vector_size": advice.vector_size,
            "distance": advice.distance.value,
            "quantization": advice.quantization_mode,
            "hnsw_m": advice.hnsw_config.m,
            "hnsw_ef_construct": advice.hnsw_config.ef_construct,
            "on_disk_payload": advice.on_disk_payload,
            "sparse_vectors": sparse_vectors,
            "rationale": advice.rationale,
        }

    def ensure(
        self,
        collection_name: str,
        use_case: str,
        vector_size: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create the collection only if it does not already exist.

        Returns a dict with an ``"action"`` key set to either ``"created"``
        or ``"already_exists"``.
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if collection_name in existing:
            info = self.client.get_collection(collection_name)
            return {
                "action": "already_exists",
                "collection_name": collection_name,
                "points_count": info.points_count or 0,
                "status": info.status.value,
            }
        result = self.auto_create(collection_name, use_case, vector_size, **kwargs)
        result["action"] = "created"
        return result

    def delete(self, collection_name: str) -> bool:
        """Delete a collection. Returns ``True`` on success."""
        self.client.delete_collection(collection_name)
        return True

    # ------------------------------------------------------------------
    # Payload indexes
    # ------------------------------------------------------------------

    _SCHEMA_TYPE_MAP: dict[str, PayloadSchemaType] = {
        "keyword": PayloadSchemaType.KEYWORD,
        "integer": PayloadSchemaType.INTEGER,
        "float": PayloadSchemaType.FLOAT,
        "geo": PayloadSchemaType.GEO,
        "text": PayloadSchemaType.TEXT,
        "bool": PayloadSchemaType.BOOL,
        "datetime": PayloadSchemaType.DATETIME,
    }

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_type: str = "keyword",
    ) -> dict[str, str]:
        """Create a payload index for fast filtered search.

        Indexed payload fields enable WHERE-style filtering with near-zero
        extra latency, even on collections with millions of vectors.

        Parameters
        ----------
        collection_name:
            Target collection.
        field_name:
            Payload key to index (e.g. ``"category"``, ``"user_id"``).
        field_type:
            Schema type: ``keyword`` | ``integer`` | ``float`` | ``geo`` |
            ``text`` | ``bool`` | ``datetime``.

        Returns
        -------
        dict
            ``{"collection", "field", "type", "status"}``.

        Raises
        ------
        ValueError
            When ``field_type`` is not a recognised schema type.
        """
        schema_type = self._SCHEMA_TYPE_MAP.get(field_type.lower())
        if schema_type is None:
            valid = list(self._SCHEMA_TYPE_MAP)
            raise ValueError(
                f"Unknown field_type '{field_type}'. Valid options: {valid}"
            )
        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_type,
        )
        return {
            "collection": collection_name,
            "field": field_name,
            "type": field_type,
            "status": "indexed",
        }

    # ------------------------------------------------------------------
    # Health and optimisation
    # ------------------------------------------------------------------

    def health(self, collection_name: str) -> dict[str, Any]:
        """Return runtime stats and parameter recommendations for a collection.

        Returns
        -------
        dict
            Keys: ``collection_name``, ``points_count``,
            ``indexed_vectors_count``, ``status``, ``optimizer_status``,
            ``vector_size`` (if detectable), ``distance`` (if detectable),
            ``recommendation`` (optional advisory message).
        """
        info = self.client.get_collection(collection_name)

        result: dict[str, Any] = {
            "collection_name": collection_name,
            "points_count": info.points_count or 0,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "status": info.status.value,
            "optimizer_status": str(info.optimizer_status),
        }

        # Decode vector config — can be VectorParams or dict[str, VectorParams]
        cfg = info.config.params.vectors
        if isinstance(cfg, VectorParams):
            result["vector_size"] = cfg.size
            result["distance"] = cfg.distance.value
        elif isinstance(cfg, dict) and cfg:
            first = next(iter(cfg.values()))
            result["vector_size"] = first.size
            result["distance"] = first.distance.value

        # Advisory message for collections that would benefit from quantization
        quant = info.config.quantization_config
        count = info.points_count or 0
        if quant is None and count > 100_000:
            result["recommendation"] = (
                f"Collection has {count:,} vectors but no quantization configured. "
                "Consider scalar INT8 quantization for ~4× memory reduction with "
                "minimal recall impact."
            )

        return result

    def optimize(
        self, collection_name: str, priority: PriorityHint = "recall"
    ) -> dict[str, Any]:
        """Apply optimised indexing/memmap thresholds to an existing collection.

        Does not modify vector data or quantization settings. Safe to call
        on a live collection.

        Returns
        -------
        dict
            Applied optimizer settings.
        """
        count = (
            self.client.get_collection(collection_name).points_count or 100_000
        )
        optimizers = self._advisor._pick_optimizers(count, priority)
        self.client.update_collection(
            collection_name=collection_name,
            optimizer_config=optimizers,
        )
        return {
            "collection": collection_name,
            "priority": priority,
            "applied_optimizers": {
                "indexing_threshold": optimizers.indexing_threshold,
                "memmap_threshold": optimizers.memmap_threshold,
            },
        }
