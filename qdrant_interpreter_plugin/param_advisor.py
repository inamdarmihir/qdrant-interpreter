"""
Qdrant Collection Parameter Advisor
====================================

Maps natural-language use-case descriptions to the optimal Qdrant collection
parameters. The advisor is entirely rule-based, so it can run without an LLM
or any network access. It covers:

- **Distance metric** — COSINE / DOT / EUCLID / MANHATTAN, inferred from
  use-case keywords (e.g. "semantic search" → COSINE, "anomaly detection" →
  EUCLID).
- **Vector size** — inferred from model-family keywords or explicit dimension
  numbers found in the description.
- **HNSW graph parameters** — ``m`` and ``ef_construct`` scaled to dataset
  size and recall/latency/memory priority.
- **Quantization** — selects None, Scalar INT8, Product, or Binary
  quantization based on collection scale and memory constraints.
- **Optimizer settings** — ``indexing_threshold`` and ``memmap_threshold``
  tuned to the priority hint.
- **On-disk payload** — enabled automatically for large collections or when
  memory efficiency is the goal.

Usage::

    from qdrant_interpreter_plugin.param_advisor import UseCaseParamAdvisor

    advisor = UseCaseParamAdvisor()
    advice = advisor.advise(
        use_case="semantic search over product descriptions using OpenAI ada-002",
        expected_count=500_000,
        priority="recall",
    )
    print(advice.distance)        # Distance.COSINE
    print(advice.vector_size)     # 1536
    print(advice.quantization_mode)  # "scalar"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    CompressionRatio,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ProductQuantization,
    ProductQuantizationConfig,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)

QuantizationMode = Literal["none", "scalar", "product", "binary"]
PriorityHint = Literal["recall", "latency", "memory"]


@dataclass
class CollectionParamAdvice:
    """Fully resolved parameter recommendation for a Qdrant collection."""

    # Core vector params
    distance: Distance
    vector_size: int
    # HNSW graph config
    hnsw_config: HnswConfigDiff
    # Optimizer config
    optimizers_config: OptimizersConfigDiff
    # Quantization
    quantization_mode: QuantizationMode
    quantization_config: ScalarQuantization | ProductQuantization | BinaryQuantization | None
    # Storage
    on_disk_payload: bool
    # Human-readable rationale for each decision
    rationale: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Keyword sets for distance-metric inference
# ---------------------------------------------------------------------------

_EUCLID_KEYWORDS: frozenset[str] = frozenset({
    "anomaly", "outlier", "fraud", "detection", "euclidean", "l2",
    "knn", "nearest neighbor", "cluster", "clustering", "k-means",
    "genomic", "biological", "sequence", "dna", "numeric", "tabular",
    "time series", "timeseries", "signal processing", "raw feature",
})

_DOT_KEYWORDS: frozenset[str] = frozenset({
    "dot product", "inner product", "collaborative filtering",
    "matrix factorization", "als", "user-item", "user item",
    "mf model", "latent factor",
})

_MANHATTAN_KEYWORDS: frozenset[str] = frozenset({
    "sparse", "bag of words", "bow", "tfidf", "tf-idf", "bm25",
    "l1 norm", "manhattan", "taxicab", "count vector",
})

# Everything else defaults to COSINE (the safe choice for normalized embeddings)
_COSINE_KEYWORDS: frozenset[str] = frozenset({
    "semantic", "text", "nlp", "language", "document", "sentence",
    "paragraph", "embedding", "bert", "gpt", "openai", "cohere",
    "huggingface", "llm", "transformer", "rag", "retrieval-augmented",
    "qa", "question answering", "chatbot", "similarity search",
    "recommendation", "product", "ecommerce", "e-commerce",
    "clip", "image", "visual", "multimodal", "cross-modal",
    "news", "article", "blog", "legal", "medical", "scientific",
})

# ---------------------------------------------------------------------------
# Model-family → canonical vector dimension
# ---------------------------------------------------------------------------

_EMBEDDING_SIZES: dict[str, int] = {
    # Sentence-Transformers family (longest match first within lookup loop)
    "all-minilm-l6": 384,
    "all-minilm-l12": 384,
    "all-minilm": 384,
    "paraphrase-minilm": 384,
    "minilm": 384,
    "e5-small": 384,
    "bge-small": 384,
    "gte-small": 384,
    "all-mpnet-base": 768,
    "mpnet": 768,
    "e5-base": 768,
    "e5-large": 1024,
    "bge-base": 768,
    "bge-large": 1024,
    "gte-base": 768,
    "gte-large": 1024,
    "bert-base": 768,
    "bert-large": 1024,
    "roberta-base": 768,
    "roberta-large": 1024,
    "distilbert": 768,
    "sentence-transformers": 768,
    # OpenAI
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "ada-002": 1536,
    "openai": 1536,
    # Cohere
    "embed-english-v3": 1024,
    "embed-multilingual-v3": 1024,
    "cohere": 1024,
    # Anthropic / Google / Amazon
    "gecko": 768,
    "palm": 768,
    "titan-embed-text-v1": 1536,
    "titan": 1536,
    # Vision / multi-modal
    "clip-vit-base": 512,
    "clip-vit-large": 768,
    "clip": 512,
    "imagenet": 2048,
    "resnet50": 2048,
    "resnet": 2048,
    "vit-base": 768,
    "vit-large": 1024,
    "vit": 768,
    # Generic fallback (must come last)
    "bert": 768,
    "gpt": 1536,
}


class UseCaseParamAdvisor:
    """Rule-based advisor: use-case text + scale → optimal Qdrant parameters.

    All decisions are deterministic and explainable. The ``rationale`` dict
    on the returned :class:`CollectionParamAdvice` explains each choice in
    plain English, making it easy to surface to users or override selectively.
    """

    def advise(
        self,
        use_case: str,
        vector_size: int | None = None,
        expected_count: int = 100_000,
        priority: PriorityHint = "recall",
        memory_gb_available: float | None = None,
    ) -> CollectionParamAdvice:
        """Return fully resolved collection parameters for a use case.

        Parameters
        ----------
        use_case:
            Free-text description, e.g. ``"semantic search over product
            descriptions using OpenAI embeddings, ~500k products"``.
        vector_size:
            Embedding dimension. Inferred from keywords if ``None``.
        expected_count:
            Approximate number of vectors; drives HNSW tuning and quantization.
        priority:
            Optimisation goal: ``"recall"`` (highest accuracy), ``"latency"``
            (lowest query time), or ``"memory"`` (smallest RAM footprint).
        memory_gb_available:
            Available RAM in GB. ``None`` → conservative medium assumption.
            Values < 4 trigger aggressive quantization.

        Returns
        -------
        CollectionParamAdvice
            Fully resolved parameters with a ``rationale`` dict explaining
            every decision.
        """
        normalized = use_case.lower()
        rationale: dict[str, str] = {}

        distance, dist_reason = self._pick_distance(normalized)
        rationale["distance"] = dist_reason

        if vector_size is None:
            vector_size, size_reason = self._infer_size(normalized)
            rationale["vector_size"] = size_reason
        else:
            rationale["vector_size"] = f"Provided explicitly: {vector_size}"

        hnsw_config, hnsw_reason = self._pick_hnsw(expected_count, priority)
        rationale["hnsw"] = hnsw_reason

        quant_mode, quant_cfg, quant_reason = self._pick_quantization(
            expected_count, priority, memory_gb_available
        )
        rationale["quantization"] = quant_reason

        optimizers_config = self._pick_optimizers(expected_count, priority)
        rationale["optimizers"] = (
            f"indexing_threshold={optimizers_config.indexing_threshold}, "
            f"memmap_threshold={optimizers_config.memmap_threshold}"
        )

        on_disk = expected_count > 1_000_000 or priority == "memory"
        rationale["on_disk_payload"] = (
            "Enabled: large collection or memory-priority."
            if on_disk
            else "Disabled: collection fits in RAM."
        )

        return CollectionParamAdvice(
            distance=distance,
            vector_size=vector_size,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            quantization_mode=quant_mode,
            quantization_config=quant_cfg,
            on_disk_payload=on_disk,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_distance(self, text: str) -> tuple[Distance, str]:
        # Check more-specific metrics first; COSINE is the safe default
        for kw in _EUCLID_KEYWORDS:
            if kw in text:
                return (
                    Distance.EUCLID,
                    f"Euclidean (L2) selected — keyword '{kw}' signals raw feature space.",
                )
        for kw in _DOT_KEYWORDS:
            if kw in text:
                return (
                    Distance.DOT,
                    f"Dot product selected — keyword '{kw}' signals matrix-factorization style.",
                )
        for kw in _MANHATTAN_KEYWORDS:
            if kw in text:
                return (
                    Distance.MANHATTAN,
                    f"Manhattan (L1) selected — keyword '{kw}' signals sparse/count vectors.",
                )
        return (
            Distance.COSINE,
            "COSINE selected (default) — optimal for normalized text/semantic embeddings.",
        )

    def _infer_size(self, text: str) -> tuple[int, str]:
        # Explicit "dimension N" / "dim=N" / "size N" patterns
        m = re.search(
            r"\b(?:dimension|dim|size)[s]?\s*[=:of]*\s*(\d{3,4})\b", text
        )
        if m:
            return int(m.group(1)), f"Dimension extracted from description: {m.group(1)}"

        # Bare common dimension integers
        m = re.search(r"\b(384|512|768|1024|1536|2048|3072)\b", text)
        if m:
            return int(m.group(1)), f"Dimension extracted from description: {m.group(1)}"

        # Model-family keyword lookup (sorted by descending keyword length to
        # prefer more-specific matches, e.g. "bert-large" before "bert")
        for kw, size in sorted(
            _EMBEDDING_SIZES.items(), key=lambda x: -len(x[0])
        ):
            if kw in text:
                return size, f"Inferred from model keyword '{kw}': size={size}"

        return 768, "Defaulted to 768 — compatible with BERT-base / sentence-transformers."

    def _pick_hnsw(
        self, count: int, priority: PriorityHint
    ) -> tuple[HnswConfigDiff, str]:
        if priority == "latency":
            return (
                HnswConfigDiff(m=16, ef_construct=100),
                "HNSW m=16, ef_construct=100: latency-optimised (lower graph density).",
            )
        if priority == "memory":
            return (
                HnswConfigDiff(m=8, ef_construct=100),
                "HNSW m=8, ef_construct=100: memory-efficient (fewer graph edges).",
            )
        # recall priority
        if count > 1_000_000:
            return (
                HnswConfigDiff(m=32, ef_construct=200),
                "HNSW m=32, ef_construct=200: high-recall for very large collection (>1M vectors).",
            )
        if count > 100_000:
            return (
                HnswConfigDiff(m=16, ef_construct=200),
                "HNSW m=16, ef_construct=200: balanced recall for medium collection.",
            )
        return (
            HnswConfigDiff(m=16, ef_construct=100),
            "HNSW m=16, ef_construct=100: standard defaults for small collection.",
        )

    def _pick_quantization(
        self,
        count: int,
        priority: PriorityHint,
        memory_gb: float | None,
    ) -> tuple[QuantizationMode, ScalarQuantization | ProductQuantization | BinaryQuantization | None, str]:
        # Small high-recall collections: no quantization (accuracy over compression)
        if priority == "recall" and count < 100_000:
            return (
                "none",
                None,
                "No quantization: small collection with recall priority — accuracy over compression.",
            )

        # Aggressive memory saving
        if priority == "memory" or (memory_gb is not None and memory_gb < 4):
            cfg = ProductQuantization(
                product=ProductQuantizationConfig(
                    compression=CompressionRatio.X16,
                    always_ram=True,
                )
            )
            return (
                "product",
                cfg,
                "Product quantization (16× compression): aggressive memory savings; "
                "re-scoring preserves accuracy.",
            )

        # Speed-first on large collection
        if priority == "latency" and count > 100_000:
            cfg = BinaryQuantization(
                binary=BinaryQuantizationConfig(always_ram=True)
            )
            return (
                "binary",
                cfg,
                "Binary quantization: fastest ANN search; re-scoring step restores recall.",
            )

        # Default: scalar INT8 for large-enough collections
        if count > 100_000:
            cfg = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )
            return (
                "scalar",
                cfg,
                "Scalar INT8 quantization: ~4× memory reduction, <1% recall loss, "
                "vectors kept in RAM for speed.",
            )

        return (
            "none",
            None,
            "No quantization: collection is small — accuracy takes priority.",
        )

    def _pick_optimizers(
        self, count: int, priority: PriorityHint
    ) -> OptimizersConfigDiff:
        if priority == "latency":
            # Build HNSW index sooner → lower query latency after first inserts
            return OptimizersConfigDiff(
                indexing_threshold=10_000,
                memmap_threshold=50_000,
            )
        if priority == "memory":
            # Use memmap aggressively to reduce RAM footprint
            return OptimizersConfigDiff(
                indexing_threshold=20_000,
                memmap_threshold=10_000,
            )
        # Balanced default
        return OptimizersConfigDiff(
            indexing_threshold=20_000,
            memmap_threshold=100_000,
        )
