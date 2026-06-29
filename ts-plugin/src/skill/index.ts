/**
 * Qdrant Manager — Interpreter Skill Module
 * ==========================================
 * This module runs inside the QuickJS interpreter. It wraps the PTC-bridged
 * tools.* namespace behind a clean camelCase API so interpreter code reads
 * naturally without knowing the underlying tool names.
 *
 * Usage from the interpreter:
 *
 *   const qdrant = await import("@/skills/qdrant-manager");
 *   const col = await qdrant.createCollection("products", {
 *     useCase: "semantic search using OpenAI ada-002",
 *     expectedCount: 500_000,
 *   });
 *
 * The `tools` global is injected by the CodeInterpreterMiddleware PTC bridge.
 * Every function here is a thin wrapper that normalises camelCase options to
 * the snake_case fields the underlying LangChain tools expect.
 */

// ---------------------------------------------------------------------------
// PTC bridge declaration (globalThis.tools injected by the middleware)
// ---------------------------------------------------------------------------

declare const tools: {
  createQdrantCollection(args: {
    collection_name: string;
    use_case: string;
    vector_size?: number;
    expected_count?: number;
    priority?: "recall" | "latency" | "memory";
    sparse_vectors?: boolean;
    memory_gb_available?: number;
  }): Promise<string>;

  ensureQdrantCollection(args: {
    collection_name: string;
    use_case: string;
    vector_size?: number;
  }): Promise<string>;

  listQdrantCollections(): Promise<string>;

  describeQdrantCollection(args: { collection_name: string }): Promise<string>;

  deleteQdrantCollection(args: { collection_name: string }): Promise<string>;

  upsertQdrantPoints(args: { collection_name: string; points: string }): Promise<string>;

  searchQdrantCollection(args: {
    collection_name: string;
    query_vector: string;
    limit?: number;
    score_threshold?: number;
    filter_conditions?: string;
  }): Promise<string>;

  createQdrantPayloadIndex(args: {
    collection_name: string;
    field_name: string;
    field_type?: string;
  }): Promise<string>;

  recommendQdrantParams(args: {
    use_case: string;
    vector_size?: number;
    expected_count?: number;
    priority?: "recall" | "latency" | "memory";
  }): Promise<string>;

  optimizeQdrantCollection(args: {
    collection_name: string;
    priority?: "recall" | "latency" | "memory";
  }): Promise<string>;
};

// ---------------------------------------------------------------------------
// Option types
// ---------------------------------------------------------------------------

export interface CreateCollectionOptions {
  /** Free-text workload description — drives all parameter decisions. */
  useCase: string;
  /** Embedding dimension; inferred from useCase keywords when omitted. */
  vectorSize?: number;
  /** Approximate vector count (default: 100 000). */
  expectedCount?: number;
  /** Optimisation goal (default: "recall"). */
  priority?: "recall" | "latency" | "memory";
  /** Add a sparse vector slot for hybrid dense+sparse search. */
  sparseVectors?: boolean;
  /** Available RAM in GB; values < 4 trigger product quantization. */
  memoryGbAvailable?: number;
}

export interface SearchOptions {
  limit?: number;
  scoreThreshold?: number;
  /** JSON-encoded Qdrant filter, e.g. '{"must":[{"key":"cat","match":{"value":"books"}}]}' */
  filterConditions?: string;
}

// ---------------------------------------------------------------------------
// Collection lifecycle
// ---------------------------------------------------------------------------

/**
 * Create a Qdrant collection with automatically recommended parameters.
 * Selects distance metric, HNSW settings, quantization, and storage options
 * based on the useCase description.
 */
export async function createCollection(
  name: string,
  options: CreateCollectionOptions,
): Promise<string> {
  return tools.createQdrantCollection({
    collection_name: name,
    use_case: options.useCase,
    ...(options.vectorSize !== undefined ? { vector_size: options.vectorSize } : {}),
    expected_count: options.expectedCount ?? 100_000,
    priority: options.priority ?? "recall",
    sparse_vectors: options.sparseVectors ?? false,
    ...(options.memoryGbAvailable !== undefined
      ? { memory_gb_available: options.memoryGbAvailable }
      : {}),
  });
}

/**
 * Create a collection only if it does not already exist. Safe to call
 * multiple times — returns a status indicating "created" or "already_exists".
 */
export async function ensureCollection(
  name: string,
  options: CreateCollectionOptions,
): Promise<string> {
  return tools.ensureQdrantCollection({
    collection_name: name,
    use_case: options.useCase,
    ...(options.vectorSize !== undefined ? { vector_size: options.vectorSize } : {}),
  });
}

/**
 * Permanently delete a collection and all its vectors. Irreversible.
 */
export async function deleteCollection(name: string): Promise<string> {
  return tools.deleteQdrantCollection({ collection_name: name });
}

/**
 * List all Qdrant collections. Call this first to discover what exists.
 */
export async function listCollections(): Promise<string> {
  return tools.listQdrantCollections();
}

/**
 * Return runtime stats and optimisation recommendations for a collection.
 */
export async function describeCollection(name: string): Promise<string> {
  return tools.describeQdrantCollection({ collection_name: name });
}

// ---------------------------------------------------------------------------
// Data operations
// ---------------------------------------------------------------------------

/**
 * Insert or update vectors. Existing IDs are overwritten; new IDs are inserted.
 *
 * @param points - Array of `{ id, vector, payload? }` objects
 */
export async function upsertPoints(
  name: string,
  points: Array<{ id: number | string; vector: number[]; payload?: Record<string, unknown> }>,
): Promise<string> {
  return tools.upsertQdrantPoints({
    collection_name: name,
    points: JSON.stringify(points),
  });
}

/**
 * Search for the most similar vectors.
 *
 * @param queryVector - Float array (same dimension as the collection)
 * @param options     - Optional limit, score threshold, and payload filter
 */
export async function searchCollection(
  name: string,
  queryVector: number[],
  options?: SearchOptions,
): Promise<string> {
  return tools.searchQdrantCollection({
    collection_name: name,
    query_vector: JSON.stringify(queryVector),
    ...(options?.limit !== undefined ? { limit: options.limit } : {}),
    ...(options?.scoreThreshold !== undefined ? { score_threshold: options.scoreThreshold } : {}),
    ...(options?.filterConditions !== undefined
      ? { filter_conditions: options.filterConditions }
      : {}),
  });
}

// ---------------------------------------------------------------------------
// Indexes and optimisation
// ---------------------------------------------------------------------------

/**
 * Create a payload index for fast filtered search.
 * Always index any field you intend to use in filter_conditions.
 *
 * @param fieldType - keyword · integer · float · geo · text · bool · datetime
 */
export async function createPayloadIndex(
  name: string,
  fieldName: string,
  fieldType = "keyword",
): Promise<string> {
  return tools.createQdrantPayloadIndex({
    collection_name: name,
    field_name: fieldName,
    field_type: fieldType,
  });
}

/**
 * Dry-run the parameter advisor without creating a collection.
 * Use this to preview and explain parameter choices.
 */
export async function recommendParams(options: CreateCollectionOptions): Promise<string> {
  return tools.recommendQdrantParams({
    use_case: options.useCase,
    ...(options.vectorSize !== undefined ? { vector_size: options.vectorSize } : {}),
    expected_count: options.expectedCount ?? 100_000,
    priority: options.priority ?? "recall",
  });
}

/**
 * Update optimizer settings on an existing collection.
 * Call after bulk-loading data to tune for query workloads.
 */
export async function optimizeCollection(
  name: string,
  priority: "recall" | "latency" | "memory" = "recall",
): Promise<string> {
  return tools.optimizeQdrantCollection({ collection_name: name, priority });
}
