"""
Structured LangChain Tools for Qdrant
======================================

Ten ``@tool``-decorated callables that expose every Qdrant management
operation to a LangChain / LangGraph agent. All tools are bound to a shared
:class:`~qdrant_interpreter_plugin.index_manager.QdrantIndexManager` instance
via the :func:`make_qdrant_tools` factory so they share one Qdrant client
and one advisor context.

Tool list
---------
1. ``create_qdrant_collection`` — auto-parameterised collection creation
2. ``ensure_qdrant_collection`` — idempotent create-if-absent
3. ``upsert_qdrant_points`` — batch insert / update
4. ``search_qdrant_collection`` — ANN search with optional payload filter
5. ``list_qdrant_collections`` — enumerate all collections
6. ``describe_qdrant_collection`` — detailed stats and recommendations
7. ``create_qdrant_payload_index`` — fast payload field indexing
8. ``recommend_qdrant_params`` — dry-run parameter advisor
9. ``delete_qdrant_collection`` — permanent delete
10. ``optimize_qdrant_collection`` — update optimizer settings

Usage::

    from qdrant_client import QdrantClient
    from qdrant_interpreter_plugin.index_manager import QdrantIndexManager
    from qdrant_interpreter_plugin.tools import make_qdrant_tools

    manager = QdrantIndexManager(QdrantClient(":memory:"))
    tools = make_qdrant_tools(manager)
    # Pass `tools` to create_react_agent() or AgentExecutor
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, PointStruct

from .index_manager import QdrantIndexManager
from .param_advisor import UseCaseParamAdvisor


def make_qdrant_tools(manager: QdrantIndexManager) -> list:
    """Build a list of LangChain tools bound to *manager*.

    Parameters
    ----------
    manager:
        A :class:`QdrantIndexManager` instance. All tools share this manager
        (and its underlying ``QdrantClient``) so operations on the same
        collection are consistent across tool calls within a single agent run.

    Returns
    -------
    list
        Ten LangChain tool callables ready to pass to ``create_react_agent()``
        or ``AgentExecutor``.
    """

    # ==================================================================
    # 1. create_qdrant_collection
    # ==================================================================

    class CreateCollectionInput(BaseModel):
        collection_name: str = Field(description="Name for the new Qdrant collection")
        use_case: str = Field(
            description=(
                "Natural-language description of what this collection stores and "
                "how it will be queried, e.g. "
                "'semantic search over product descriptions using OpenAI ada-002 "
                "embeddings, ~500k products'. The more context you provide the "
                "better the parameter selection."
            )
        )
        vector_size: Optional[int] = Field(
            default=None,
            description=(
                "Embedding dimension. Leave None to infer from the use_case "
                "(e.g. 'OpenAI ada-002' → 1536)."
            ),
        )
        expected_count: int = Field(
            default=100_000,
            description="Approximate number of vectors the collection will hold",
        )
        priority: Literal["recall", "latency", "memory"] = Field(
            default="recall",
            description=(
                "Optimisation goal. "
                "'recall' = highest accuracy, "
                "'latency' = fastest queries, "
                "'memory' = smallest RAM footprint"
            ),
        )
        sparse_vectors: bool = Field(
            default=False,
            description=(
                "Set True to also configure a sparse vector field for "
                "hybrid (dense + sparse) search with BM25 or SPLADE."
            ),
        )
        memory_gb_available: Optional[float] = Field(
            default=None,
            description=(
                "Available RAM in GB. Drives quantization aggressiveness. "
                "Values below 4 GB trigger Product quantization."
            ),
        )

    @tool("create_qdrant_collection", args_schema=CreateCollectionInput)
    def create_qdrant_collection(
        collection_name: str,
        use_case: str,
        vector_size: Optional[int] = None,
        expected_count: int = 100_000,
        priority: str = "recall",
        sparse_vectors: bool = False,
        memory_gb_available: Optional[float] = None,
    ) -> str:
        """Create a Qdrant collection with automatically recommended parameters.

        Selects the best distance metric, HNSW graph settings, quantization
        strategy, and storage options based on the use-case description and
        scale. Always prefer this over raw ``client.create_collection()`` so
        parameters are tuned for the stated goal.
        """
        try:
            result = manager.auto_create(
                collection_name=collection_name,
                use_case=use_case,
                vector_size=vector_size,
                expected_count=expected_count,
                priority=priority,
                memory_gb_available=memory_gb_available,
                sparse_vectors=sparse_vectors,
            )
            lines = [
                f"✓ Collection '{collection_name}' created successfully.",
                f"  Vector size       : {result['vector_size']}",
                f"  Distance metric   : {result['distance']}",
                f"  Quantization      : {result['quantization']}",
                f"  HNSW m / ef_construct : {result['hnsw_m']} / {result['hnsw_ef_construct']}",
                f"  On-disk payload   : {result['on_disk_payload']}",
                f"  Sparse vectors    : {result['sparse_vectors']}",
                "",
                "Parameter rationale:",
            ]
            for key, reason in result["rationale"].items():
                lines.append(f"  [{key}] {reason}")
            return "\n".join(lines)
        except Exception as exc:
            return f"ERROR creating collection '{collection_name}': {exc}"

    # ==================================================================
    # 2. ensure_qdrant_collection
    # ==================================================================

    class EnsureCollectionInput(BaseModel):
        collection_name: str = Field(description="Collection name")
        use_case: str = Field(
            description="Use-case description used for parameter selection if the collection must be created"
        )
        vector_size: Optional[int] = Field(
            default=None, description="Vector dimension (inferred if None)"
        )

    @tool("ensure_qdrant_collection", args_schema=EnsureCollectionInput)
    def ensure_qdrant_collection(
        collection_name: str,
        use_case: str,
        vector_size: Optional[int] = None,
    ) -> str:
        """Create a collection only if it does not already exist.

        Safe to call multiple times. Returns a status indicating whether the
        collection was newly created or was already present.
        """
        try:
            result = manager.ensure(collection_name, use_case, vector_size)
            action = result.get("action", "unknown")
            if action == "created":
                return (
                    f"✓ Collection '{collection_name}' created.\n"
                    f"  Vector size: {result.get('vector_size')}, "
                    f"Distance: {result.get('distance')}, "
                    f"Quantization: {result.get('quantization')}"
                )
            return (
                f"→ Collection '{collection_name}' already exists — no action taken.\n"
                f"  Points: {result.get('points_count', 0):,}, "
                f"Status: {result.get('status', 'unknown')}"
            )
        except Exception as exc:
            return f"ERROR in ensure_qdrant_collection: {exc}"

    # ==================================================================
    # 3. upsert_qdrant_points
    # ==================================================================

    class UpsertPointsInput(BaseModel):
        collection_name: str = Field(description="Target collection name")
        points: str = Field(
            description=(
                "JSON list of point objects. Each object MUST have:\n"
                "  - 'id': integer or UUID string\n"
                "  - 'vector': list of floats (same dimension as the collection)\n"
                "  - 'payload': (optional) dict of metadata fields\n"
                "Example: "
                '[{"id": 1, "vector": [0.1, 0.2], "payload": {"title": "doc"}}]'
            )
        )

    @tool("upsert_qdrant_points", args_schema=UpsertPointsInput)
    def upsert_qdrant_points(collection_name: str, points: str) -> str:
        """Insert or update vectors in a Qdrant collection.

        Accepts a JSON-encoded list of points. Existing points with the same
        ``id`` are overwritten; new ids are inserted. This is idempotent.
        """
        try:
            raw: list[dict] = json.loads(points)
            structs = [
                PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in raw
            ]
            op_info = manager.client.upsert(
                collection_name=collection_name,
                points=structs,
            )
            return (
                f"✓ Upserted {len(structs)} point(s) into '{collection_name}'. "
                f"Status: {op_info.status.value}"
            )
        except json.JSONDecodeError as exc:
            return f"ERROR: Could not parse 'points' as JSON — {exc}"
        except Exception as exc:
            return f"ERROR upserting into '{collection_name}': {exc}"

    # ==================================================================
    # 4. search_qdrant_collection
    # ==================================================================

    class SearchInput(BaseModel):
        collection_name: str = Field(description="Collection to search")
        query_vector: str = Field(
            description="JSON list of floats representing the query embedding"
        )
        limit: int = Field(default=5, description="Maximum number of results")
        score_threshold: Optional[float] = Field(
            default=None,
            description=(
                "Minimum similarity score. Points below this threshold are excluded. "
                "For COSINE distance this is typically 0.5–0.9."
            ),
        )
        filter_conditions: Optional[str] = Field(
            default=None,
            description=(
                "Optional JSON payload filter. Example (must match 'sports' category):\n"
                '{"must": [{"key": "category", "match": {"value": "sports"}}]}\n'
                "Only works on indexed payload fields "
                "(use create_qdrant_payload_index first)."
            ),
        )

    @tool("search_qdrant_collection", args_schema=SearchInput)
    def search_qdrant_collection(
        collection_name: str,
        query_vector: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[str] = None,
    ) -> str:
        """Search a Qdrant collection for the most similar vectors.

        Returns the top-``limit`` results with IDs, similarity scores, and
        stored payload fields. Optionally filters by payload metadata.
        """
        try:
            vec: list[float] = json.loads(query_vector)
            qfilter: Filter | None = None
            if filter_conditions:
                qfilter = Filter(**json.loads(filter_conditions))

            results = manager.client.search(
                collection_name=collection_name,
                query_vector=vec,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=qfilter,
                with_payload=True,
            )
            if not results:
                return "No results found (try lowering score_threshold or broadening the filter)."
            lines = [f"Found {len(results)} result(s):"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"  {i}. id={r.id}  score={r.score:.4f}  payload={r.payload}"
                )
            return "\n".join(lines)
        except json.JSONDecodeError as exc:
            return f"ERROR: Could not parse query_vector or filter_conditions as JSON — {exc}"
        except Exception as exc:
            return f"ERROR searching '{collection_name}': {exc}"

    # ==================================================================
    # 5. list_qdrant_collections
    # ==================================================================

    class ListCollectionsInput(BaseModel):
        pass

    @tool("list_qdrant_collections", args_schema=ListCollectionsInput)
    def list_qdrant_collections() -> str:
        """List all Qdrant collections with basic metadata.

        Call this first to discover which collections already exist before
        creating a new one with the same name.
        """
        try:
            cols = manager.client.get_collections().collections
            if not cols:
                return "No collections found. Create one with create_qdrant_collection."
            lines = [f"Found {len(cols)} collection(s):"]
            for c in cols:
                lines.append(f"  • {c.name}")
            return "\n".join(lines)
        except Exception as exc:
            return f"ERROR listing collections: {exc}"

    # ==================================================================
    # 6. describe_qdrant_collection
    # ==================================================================

    class DescribeCollectionInput(BaseModel):
        collection_name: str = Field(description="Collection to inspect")

    @tool("describe_qdrant_collection", args_schema=DescribeCollectionInput)
    def describe_qdrant_collection(collection_name: str) -> str:
        """Return detailed stats and optimisation recommendations for a collection.

        Useful for diagnosing performance issues or deciding whether to add
        quantization, payload indexes, or adjust optimizer settings.
        """
        try:
            h = manager.health(collection_name)
            lines = [
                f"Collection: {collection_name}",
                f"  Status              : {h.get('status')}",
                f"  Points count        : {h.get('points_count', 0):,}",
                f"  Indexed vectors     : {h.get('indexed_vectors_count', 0):,}",
                f"  Vector size         : {h.get('vector_size', 'n/a')}",
                f"  Distance            : {h.get('distance', 'n/a')}",
                f"  Optimizer status    : {h.get('optimizer_status', 'n/a')}",
            ]
            rec = h.get("recommendation")
            if rec:
                lines.append(f"\n⚠  Recommendation: {rec}")
            return "\n".join(lines)
        except Exception as exc:
            return f"ERROR describing '{collection_name}': {exc}"

    # ==================================================================
    # 7. create_qdrant_payload_index
    # ==================================================================

    class CreatePayloadIndexInput(BaseModel):
        collection_name: str = Field(description="Target collection")
        field_name: str = Field(
            description=(
                "Payload field to index. Must match the key used when upserting "
                "points, e.g. 'category', 'user_id', 'published_at'."
            )
        )
        field_type: str = Field(
            default="keyword",
            description=(
                "Schema type for the field. One of:\n"
                "  keyword  — exact-match string (categories, tags, IDs)\n"
                "  integer  — whole numbers (counts, timestamps as epoch)\n"
                "  float    — decimal numbers (prices, scores)\n"
                "  text     — full-text tokenised search\n"
                "  bool     — boolean flags\n"
                "  geo      — lat/lon geographic coordinates\n"
                "  datetime — ISO-8601 date/time strings"
            ),
        )

    @tool("create_qdrant_payload_index", args_schema=CreatePayloadIndexInput)
    def create_qdrant_payload_index(
        collection_name: str,
        field_name: str,
        field_type: str = "keyword",
    ) -> str:
        """Create a payload index to enable fast filtered vector search.

        Indexed fields let the agent push WHERE-style conditions directly into
        the ANN search, making filtered queries up to 100× faster on large
        collections. Always index any field you intend to filter on.
        """
        try:
            manager.create_payload_index(collection_name, field_name, field_type)
            return (
                f"✓ Payload index created on '{field_name}' ({field_type}) "
                f"in collection '{collection_name}'."
            )
        except ValueError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR creating payload index: {exc}"

    # ==================================================================
    # 8. recommend_qdrant_params
    # ==================================================================

    class RecommendParamsInput(BaseModel):
        use_case: str = Field(
            description="Natural-language description of the intended use case"
        )
        vector_size: Optional[int] = Field(
            default=None, description="Embedding dimension if known"
        )
        expected_count: int = Field(
            default=100_000, description="Expected number of vectors"
        )
        priority: Literal["recall", "latency", "memory"] = Field(
            default="recall", description="Optimisation priority"
        )

    @tool("recommend_qdrant_params", args_schema=RecommendParamsInput)
    def recommend_qdrant_params(
        use_case: str,
        vector_size: Optional[int] = None,
        expected_count: int = 100_000,
        priority: str = "recall",
    ) -> str:
        """Dry-run the parameter advisor without creating a collection.

        Useful when you want to explain parameter choices to the user or
        validate that the advisor would pick the right settings before
        committing to a create operation.
        """
        try:
            advisor = UseCaseParamAdvisor()
            advice = advisor.advise(
                use_case=use_case,
                vector_size=vector_size,
                expected_count=expected_count,
                priority=priority,
            )
            lines = [
                "Recommended parameters (not yet created):",
                f"  Distance          : {advice.distance.value}",
                f"  Vector size       : {advice.vector_size}",
                f"  HNSW m            : {advice.hnsw_config.m}",
                f"  HNSW ef_construct : {advice.hnsw_config.ef_construct}",
                f"  Quantization      : {advice.quantization_mode}",
                f"  On-disk payload   : {advice.on_disk_payload}",
                "",
                "Rationale:",
            ]
            for key, reason in advice.rationale.items():
                lines.append(f"  [{key}] {reason}")
            return "\n".join(lines)
        except Exception as exc:
            return f"ERROR generating recommendations: {exc}"

    # ==================================================================
    # 9. delete_qdrant_collection
    # ==================================================================

    class DeleteCollectionInput(BaseModel):
        collection_name: str = Field(description="Name of the collection to delete")

    @tool("delete_qdrant_collection", args_schema=DeleteCollectionInput)
    def delete_qdrant_collection(collection_name: str) -> str:
        """Permanently delete a Qdrant collection and all its vectors.

        This operation is irreversible. Double-check the collection name
        before calling this tool.
        """
        try:
            manager.delete(collection_name)
            return f"✓ Collection '{collection_name}' deleted."
        except Exception as exc:
            return f"ERROR deleting '{collection_name}': {exc}"

    # ==================================================================
    # 10. optimize_qdrant_collection
    # ==================================================================

    class OptimizeCollectionInput(BaseModel):
        collection_name: str = Field(description="Collection to optimise")
        priority: Literal["recall", "latency", "memory"] = Field(
            default="recall",
            description="Optimisation goal that drives indexing and memmap thresholds",
        )

    @tool("optimize_qdrant_collection", args_schema=OptimizeCollectionInput)
    def optimize_qdrant_collection(
        collection_name: str, priority: str = "recall"
    ) -> str:
        """Update optimizer settings on an existing collection.

        Adjusts ``indexing_threshold`` and ``memmap_threshold`` without
        modifying any vector data. Call this after bulk-loading data to
        tune the collection for query workloads.
        """
        try:
            result = manager.optimize(collection_name, priority)
            return (
                f"✓ Optimizer settings updated for '{collection_name}' "
                f"(priority={priority}).\n"
                f"  indexing_threshold : {result['applied_optimizers']['indexing_threshold']:,}\n"
                f"  memmap_threshold   : {result['applied_optimizers']['memmap_threshold']:,}"
            )
        except Exception as exc:
            return f"ERROR optimizing '{collection_name}': {exc}"

    return [
        create_qdrant_collection,
        ensure_qdrant_collection,
        upsert_qdrant_points,
        search_qdrant_collection,
        list_qdrant_collections,
        describe_qdrant_collection,
        create_qdrant_payload_index,
        recommend_qdrant_params,
        delete_qdrant_collection,
        optimize_qdrant_collection,
    ]
