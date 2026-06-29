/**
 * QdrantManager
 * =============
 * High-level Qdrant collection management powered by UseCaseParamAdvisor.
 * All methods return plain objects so they compose cleanly with LangChain
 * tools and interpreter code.
 */

import { QdrantClient } from "@qdrant/js-client-rest";
import { UseCaseParamAdvisor, pickOptimizers } from "./paramAdvisor.js";
import type { PriorityHint, CollectionParamAdvice } from "./paramAdvisor.js";

export type { PriorityHint };

// ---------------------------------------------------------------------------
// Public option types
// ---------------------------------------------------------------------------

export interface AutoCreateOptions {
  /** Free-text description of the workload — drives parameter selection. */
  useCase: string;
  /** Embedding dimension. Inferred from useCase keywords when omitted. */
  vectorSize?: number;
  /** Approximate vector count (default 100 000). */
  expectedCount?: number;
  /** Optimisation goal (default "recall"). */
  priority?: PriorityHint;
  /** Available RAM in GB; triggers aggressive quantization below 4 GB. */
  memoryGbAvailable?: number;
  /** Add a sparse vector slot for hybrid (dense + sparse) search. */
  sparseVectors?: boolean;
}

export interface CollectionSummary {
  collection_name: string;
  vector_size: number;
  distance: string;
  quantization: string;
  hnsw_m: number | undefined;
  hnsw_ef_construct: number | undefined;
  on_disk_payload: boolean;
  sparse_vectors: boolean;
  rationale: Record<string, string>;
  action?: string;
  points_count?: number;
  status?: string;
}

export interface HealthInfo {
  collection_name: string;
  points_count: number;
  indexed_vectors_count: number;
  status: string;
  optimizer_status: string;
  vector_size?: number;
  distance?: string;
  recommendation?: string;
}

export interface PayloadIndexResult {
  collection: string;
  field: string;
  type: string;
  status: string;
}

export interface OptimizeResult {
  collection: string;
  priority: string;
  applied_optimizers: { indexing_threshold: number; memmap_threshold: number };
}

// ---------------------------------------------------------------------------
// Field-type map (validated at call-site)
// ---------------------------------------------------------------------------

const FIELD_TYPES: Record<string, true> = {
  keyword: true,
  integer: true,
  float: true,
  geo: true,
  text: true,
  bool: true,
  datetime: true,
};

const VALID_FIELD_TYPES = Object.keys(FIELD_TYPES).join(", ");

type PayloadFieldType = "keyword" | "integer" | "float" | "geo" | "text" | "bool" | "datetime";

function assertFieldType(raw: string): PayloadFieldType {
  if (!FIELD_TYPES[raw]) {
    throw new Error(`Unknown field_type '${raw}'. Valid options: ${VALID_FIELD_TYPES}`);
  }
  return raw as PayloadFieldType;
}

// ---------------------------------------------------------------------------
// QdrantManager
// ---------------------------------------------------------------------------

export class QdrantManager {
  readonly client: QdrantClient;
  private readonly advisor = new UseCaseParamAdvisor();

  constructor(client: QdrantClient) {
    this.client = client;
  }

  // ------------------------------------------------------------------ //
  // Collection lifecycle                                                //
  // ------------------------------------------------------------------ //

  async autoCreate(name: string, options: AutoCreateOptions): Promise<CollectionSummary> {
    const {
      useCase,
      vectorSize,
      expectedCount = 100_000,
      priority = "recall",
      memoryGbAvailable,
      sparseVectors = false,
    } = options;

    const advice = this.advisor.advise({
      useCase,
      vectorSize,
      expectedCount,
      priority,
      memoryGbAvailable,
    });

    await this.client.createCollection(name, {
      vectors: {
        size: advice.vector_size,
        distance: advice.distance,
        on_disk: false,
      },
      hnsw_config: advice.hnsw_config,
      optimizers_config: advice.optimizers_config,
      on_disk_payload: advice.on_disk_payload,
      ...(advice.quantization_config !== null
        ? { quantization_config: advice.quantization_config }
        : {}),
      ...(sparseVectors
        ? { sparse_vectors: { sparse: { index: { on_disk: false } } } }
        : {}),
    });

    return summaryFromAdvice(name, advice, sparseVectors, "created");
  }

  async ensure(name: string, options: AutoCreateOptions): Promise<CollectionSummary> {
    const collections = await this.client.getCollections();
    const exists = collections.collections.some((c) => c.name === name);

    if (exists) {
      const info = await this.client.getCollection(name);
      return {
        action: "already_exists",
        collection_name: name,
        vector_size: 0,
        distance: "",
        quantization: "",
        hnsw_m: undefined,
        hnsw_ef_construct: undefined,
        on_disk_payload: false,
        sparse_vectors: false,
        rationale: {},
        points_count: info.points_count ?? 0,
        status: info.status,
      };
    }

    return this.autoCreate(name, options);
  }

  async delete(name: string): Promise<boolean> {
    await this.client.deleteCollection(name);
    return true;
  }

  // ------------------------------------------------------------------ //
  // Payload indexes                                                     //
  // ------------------------------------------------------------------ //

  async createPayloadIndex(
    name: string,
    fieldName: string,
    fieldType: string = "keyword",
  ): Promise<PayloadIndexResult> {
    const validated = assertFieldType(fieldType);
    await this.client.createPayloadIndex(name, {
      field_name: fieldName,
      field_schema: validated,
    });
    return { collection: name, field: fieldName, type: fieldType, status: "indexed" };
  }

  // ------------------------------------------------------------------ //
  // Health and stats                                                    //
  // ------------------------------------------------------------------ //

  async health(name: string): Promise<HealthInfo> {
    const info = await this.client.getCollection(name);

    const result: HealthInfo = {
      collection_name: name,
      points_count: info.points_count ?? 0,
      indexed_vectors_count: info.indexed_vectors_count ?? 0,
      status: info.status,
      optimizer_status:
        typeof info.optimizer_status === "object" &&
        info.optimizer_status !== null &&
        "status" in info.optimizer_status
          ? String(info.optimizer_status.status)
          : String(info.optimizer_status),
    };

    // Decode VectorParams | Record<string, VectorParams>
    const vectors = info.config.params.vectors;
    if (vectors !== null && vectors !== undefined && typeof vectors === "object") {
      if ("size" in vectors && typeof vectors.size === "number") {
        result.vector_size = vectors.size;
        result.distance = String(vectors.distance);
      } else {
        // Named-vectors map — take first entry
        const first = Object.values(vectors)[0];
        if (
          first !== undefined &&
          typeof first === "object" &&
          first !== null &&
          "size" in first &&
          typeof first.size === "number"
        ) {
          result.vector_size = first.size;
          result.distance = String(first.distance);
        }
      }
    }

    const count = info.points_count ?? 0;
    if (!info.config.quantization_config && count > 100_000) {
      result.recommendation =
        `Collection has ${count.toLocaleString()} vectors but no quantization configured. ` +
        "Consider scalar INT8 quantization for ~4× memory reduction with minimal recall impact.";
    }

    return result;
  }

  // ------------------------------------------------------------------ //
  // Optimisation                                                        //
  // ------------------------------------------------------------------ //

  async optimize(name: string, priority: PriorityHint = "recall"): Promise<OptimizeResult> {
    const info = await this.client.getCollection(name);
    const count = info.points_count ?? 100_000;
    const optimizers = pickOptimizers(count, priority);

    await this.client.updateCollection(name, {
      optimizers_config: {
        indexing_threshold: optimizers.indexing_threshold,
        memmap_threshold: optimizers.memmap_threshold,
      },
    });

    return {
      collection: name,
      priority,
      applied_optimizers: {
        indexing_threshold: optimizers.indexing_threshold ?? 20_000,
        memmap_threshold: optimizers.memmap_threshold ?? 100_000,
      },
    };
  }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function summaryFromAdvice(
  name: string,
  advice: CollectionParamAdvice,
  sparseVectors: boolean,
  action: string,
): CollectionSummary {
  return {
    collection_name: name,
    vector_size: advice.vector_size,
    distance: advice.distance,
    quantization: advice.quantization_mode,
    hnsw_m: advice.hnsw_config.m,
    hnsw_ef_construct: advice.hnsw_config.ef_construct,
    on_disk_payload: advice.on_disk_payload,
    sparse_vectors: sparseVectors,
    rationale: advice.rationale,
    action,
  };
}
