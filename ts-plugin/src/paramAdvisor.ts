import { z } from "zod";

// ---------------------------------------------------------------------------
// Zod Schemas for Enums and Validation
// ---------------------------------------------------------------------------

export const DistanceSchema = z.enum(["Cosine", "Dot", "Euclid", "Manhattan"]);
export type Distance = z.infer<typeof DistanceSchema>;
export const Distance = DistanceSchema.enum;

export const QuantizationModeSchema = z.enum(["none", "scalar", "product", "binary"]);
export type QuantizationMode = z.infer<typeof QuantizationModeSchema>;

export const PriorityHintSchema = z.enum(["recall", "latency", "memory"]);
export type PriorityHint = z.infer<typeof PriorityHintSchema>;

export const ScalarTypeSchema = z.enum(["int8"]);
export type ScalarType = z.infer<typeof ScalarTypeSchema>;
export const ScalarType = ScalarTypeSchema.enum;

export const CompressionRatioSchema = z.enum(["x4", "x8", "x16", "x32", "x64"]);
export type CompressionRatio = z.infer<typeof CompressionRatioSchema>;
export const CompressionRatio = CompressionRatioSchema.enum;

// Input validation schema
export const AdviseInputSchema = z.object({
  useCase: z.string(),
  vectorSize: z.number().int().positive().optional(),
  expectedCount: z.number().int().positive().default(100_000),
  priority: PriorityHintSchema.default("recall"),
  memoryGbAvailable: z.number().positive().optional(),
});
export type AdviseInput = z.infer<typeof AdviseInputSchema>;

// ---------------------------------------------------------------------------
// Qdrant Configuration Types
// ---------------------------------------------------------------------------

export interface HnswConfigDiff {
  m?: number;
  ef_construct?: number;
}

export interface OptimizersConfigDiff {
  indexing_threshold?: number;
  memmap_threshold?: number;
}

export interface BinaryQuantizationConfig {
  always_ram?: boolean;
}

export interface BinaryQuantization {
  binary: BinaryQuantizationConfig;
}

export interface ProductQuantizationConfig {
  compression: CompressionRatio;
  always_ram?: boolean;
}

export interface ProductQuantization {
  product: ProductQuantizationConfig;
}

export interface ScalarQuantizationConfig {
  type: ScalarType;
  quantile?: number;
  always_ram?: boolean;
}

export interface ScalarQuantization {
  scalar: ScalarQuantizationConfig;
}

export interface CollectionParamAdvice {
  /** Core vector params */
  distance: Distance;
  vector_size: number;
  /** HNSW graph config */
  hnsw_config: HnswConfigDiff;
  /** Optimizer config */
  optimizers_config: OptimizersConfigDiff;
  /** Quantization */
  quantization_mode: QuantizationMode;
  quantization_config: ScalarQuantization | ProductQuantization | BinaryQuantization | null;
  /** Storage */
  on_disk_payload: boolean;
  /** Human-readable rationale for each decision */
  rationale: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Keyword sets for distance-metric inference
// ---------------------------------------------------------------------------

const _EUCLID_KEYWORDS = new Set([
  "anomaly", "outlier", "fraud", "detection", "euclidean", "l2",
  "knn", "nearest neighbor", "cluster", "clustering", "k-means",
  "genomic", "biological", "sequence", "dna", "numeric", "tabular",
  "time series", "timeseries", "signal processing", "raw feature",
]);

const _DOT_KEYWORDS = new Set([
  "dot product", "inner product", "collaborative filtering",
  "matrix factorization", "als", "user-item", "user item",
  "mf model", "latent factor",
]);

const _MANHATTAN_KEYWORDS = new Set([
  "sparse", "bag of words", "bow", "tfidf", "tf-idf", "bm25",
  "l1 norm", "manhattan", "taxicab", "count vector",
]);

// Everything else defaults to COSINE (the safe choice for normalized embeddings)
const _COSINE_KEYWORDS = new Set([
  "semantic", "text", "nlp", "language", "document", "sentence",
  "paragraph", "embedding", "bert", "gpt", "openai", "cohere",
  "huggingface", "llm", "transformer", "rag", "retrieval-augmented",
  "qa", "question answering", "chatbot", "similarity search",
  "recommendation", "product", "ecommerce", "e-commerce",
  "clip", "image", "visual", "multimodal", "cross-modal",
  "news", "article", "blog", "legal", "medical", "scientific",
]);

// ---------------------------------------------------------------------------
// Model-family -> canonical vector dimension
// ---------------------------------------------------------------------------

const _EMBEDDING_SIZES: Record<string, number> = {
  // Sentence-Transformers family
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
  // OpenAI
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
  "text-embedding-ada-002": 1536,
  "ada-002": 1536,
  "openai": 1536,
  // Cohere
  "embed-english-v3": 1024,
  "embed-multilingual-v3": 1024,
  "cohere": 1024,
  // Anthropic / Google / Amazon
  "gecko": 768,
  "palm": 768,
  "titan-embed-text-v1": 1536,
  "titan": 1536,
  // Vision / multi-modal
  "clip-vit-base": 512,
  "clip-vit-large": 768,
  "clip": 512,
  "imagenet": 2048,
  "resnet50": 2048,
  "resnet": 2048,
  "vit-base": 768,
  "vit-large": 1024,
  "vit": 768,
  // Generic fallback (must come last in priority)
  "bert": 768,
  "gpt": 1536,
};

/**
 * Rule-based advisor: use-case text + scale -> optimal Qdrant parameters.
 *
 * All decisions are deterministic and explainable. The `rationale` dict
 * on the returned `CollectionParamAdvice` explains each choice in
 * plain English, making it easy to surface to users or override selectively.
 */
export class UseCaseParamAdvisor {
  /**
   * Return fully resolved collection parameters for a use case.
   */
  public advise(input: AdviseInput): CollectionParamAdvice {
    // Validate input using Zod
    const validated = AdviseInputSchema.parse(input);
    const { useCase, priority, memoryGbAvailable } = validated;
    let { vectorSize, expectedCount } = validated;

    const normalized = useCase.toLowerCase();
    const rationale: Record<string, string> = {};

    const [distance, distReason] = this._pickDistance(normalized);
    rationale["distance"] = distReason;

    if (vectorSize === undefined) {
      const [inferredSize, sizeReason] = this._inferSize(normalized);
      vectorSize = inferredSize;
      rationale["vector_size"] = sizeReason;
    } else {
      rationale["vector_size"] = `Provided explicitly: ${vectorSize}`;
    }

    const [hnswConfig, hnswReason] = this._pickHnsw(expectedCount, priority);
    rationale["hnsw"] = hnswReason;

    const [quantMode, quantCfg, quantReason] = this._pickQuantization(
      expectedCount,
      priority,
      memoryGbAvailable
    );
    rationale["quantization"] = quantReason;

    const optimizersConfig = this.pickOptimizers(expectedCount, priority);
    rationale["optimizers"] =
      `indexing_threshold=${optimizersConfig.indexing_threshold}, ` +
      `memmap_threshold=${optimizersConfig.memmap_threshold}`;

    const onDisk = expectedCount > 1_000_000 || priority === "memory";
    rationale["on_disk_payload"] = onDisk
      ? "Enabled: large collection or memory-priority."
      : "Disabled: collection fits in RAM.";

    return {
      distance,
      vector_size: vectorSize,
      hnsw_config: hnswConfig,
      optimizers_config: optimizersConfig,
      quantization_mode: quantMode,
      quantization_config: quantCfg,
      on_disk_payload: onDisk,
      rationale,
    };
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  private _pickDistance(text: string): [Distance, string] {
    // Check more-specific metrics first; COSINE is the safe default
    for (const kw of _EUCLID_KEYWORDS) {
      if (text.includes(kw)) {
        return [
          Distance.Euclid,
          `Euclidean (L2) selected — keyword '${kw}' signals raw feature space.`,
        ];
      }
    }
    for (const kw of _DOT_KEYWORDS) {
      if (text.includes(kw)) {
        return [
          Distance.Dot,
          `Dot product selected — keyword '${kw}' signals matrix-factorization style.`,
        ];
      }
    }
    for (const kw of _MANHATTAN_KEYWORDS) {
      if (text.includes(kw)) {
        return [
          Distance.Manhattan,
          `Manhattan (L1) selected — keyword '${kw}' signals sparse/count vectors.`,
        ];
      }
    }
    return [
      Distance.Cosine,
      "COSINE selected (default) — optimal for normalized text/semantic embeddings.",
    ];
  }

  private _inferSize(text: string): [number, string] {
    // Explicit "dimension N" / "dim=N" / "size N" patterns
    let m = text.match(/\b(?:dimension|dim|size)[s]?\s*[=:of]*\s*(\d{3,4})\b/);
    if (m && m[1]) {
      return [parseInt(m[1], 10), `Dimension extracted from description: ${m[1]}`];
    }

    // Bare common dimension integers
    m = text.match(/\b(384|512|768|1024|1536|2048|3072)\b/);
    if (m && m[1]) {
      return [parseInt(m[1], 10), `Dimension extracted from description: ${m[1]}`];
    }

    // Model-family keyword lookup (sorted by descending keyword length to
    // prefer more-specific matches, e.g. "bert-large" before "bert")
    const sortedKeywords = Object.entries(_EMBEDDING_SIZES).sort(
      (a, b) => b[0].length - a[0].length
    );

    for (const [kw, size] of sortedKeywords) {
      if (text.includes(kw)) {
        return [size, `Inferred from model keyword '${kw}': size=${size}`];
      }
    }

    return [768, "Defaulted to 768 — compatible with BERT-base / sentence-transformers."];
  }

  private _pickHnsw(count: number, priority: PriorityHint): [HnswConfigDiff, string] {
    if (priority === "latency") {
      return [
        { m: 16, ef_construct: 100 },
        "HNSW m=16, ef_construct=100: latency-optimised (lower graph density).",
      ];
    }
    if (priority === "memory") {
      return [
        { m: 8, ef_construct: 100 },
        "HNSW m=8, ef_construct=100: memory-efficient (fewer graph edges).",
      ];
    }
    // recall priority
    if (count > 1_000_000) {
      return [
        { m: 32, ef_construct: 200 },
        "HNSW m=32, ef_construct=200: high-recall for very large collection (>1M vectors).",
      ];
    }
    if (count > 100_000) {
      return [
        { m: 16, ef_construct: 200 },
        "HNSW m=16, ef_construct=200: balanced recall for medium collection.",
      ];
    }
    return [
      { m: 16, ef_construct: 100 },
      "HNSW m=16, ef_construct=100: standard defaults for small collection.",
    ];
  }

  private _pickQuantization(
    count: number,
    priority: PriorityHint,
    memoryGb: number | undefined
  ): [
    QuantizationMode,
    ScalarQuantization | ProductQuantization | BinaryQuantization | null,
    string
  ] {
    // Small high-recall collections: no quantization (accuracy over compression)
    if (priority === "recall" && count < 100_000) {
      return [
        "none",
        null,
        "No quantization: small collection with recall priority — accuracy over compression.",
      ];
    }

    // Aggressive memory saving
    if (priority === "memory" || (memoryGb !== undefined && memoryGb < 4)) {
      const cfg: ProductQuantization = {
        product: {
          compression: CompressionRatio.x16,
          always_ram: true,
        },
      };
      return [
        "product",
        cfg,
        "Product quantization (16× compression): aggressive memory savings; re-scoring preserves accuracy.",
      ];
    }

    // Speed-first on large collection
    if (priority === "latency" && count > 100_000) {
      const cfg: BinaryQuantization = {
        binary: { always_ram: true },
      };
      return [
        "binary",
        cfg,
        "Binary quantization: fastest ANN search; re-scoring step restores recall.",
      ];
    }

    // Default: scalar INT8 for large-enough collections
    if (count > 100_000) {
      const cfg: ScalarQuantization = {
        scalar: {
          type: ScalarType.int8,
          quantile: 0.99,
          always_ram: true,
        },
      };
      return [
        "scalar",
        cfg,
        "Scalar INT8 quantization: ~4× memory reduction, <1% recall loss, vectors kept in RAM for speed.",
      ];
    }

    return [
      "none",
      null,
      "No quantization: collection is small — accuracy takes priority.",
    ];
  }

  public pickOptimizers(count: number, priority: PriorityHint): OptimizersConfigDiff {
    if (priority === "latency") {
      // Build HNSW index sooner -> lower query latency after first inserts
      return {
        indexing_threshold: 10_000,
        memmap_threshold: 50_000,
      };
    }
    if (priority === "memory") {
      // Use memmap aggressively to reduce RAM footprint
      return {
        indexing_threshold: 20_000,
        memmap_threshold: 10_000,
      };
    }
    // Balanced default
    return {
      indexing_threshold: 20_000,
      memmap_threshold: 100_000,
    };
  }
}

/** Standalone convenience — same logic as UseCaseParamAdvisor.pickOptimizers. */
export function pickOptimizers(
  count: number,
  priority: PriorityHint,
): OptimizersConfigDiff {
  return new UseCaseParamAdvisor().pickOptimizers(count, priority);
}