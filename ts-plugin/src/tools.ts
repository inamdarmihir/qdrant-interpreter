/**
 * Qdrant LangChain Tools
 * ======================
 * Ten StructuredTool instances that wrap QdrantManager. Pass them to any
 * LangChain / deepagents agent and optionally expose them via PTC inside the
 * QuickJS interpreter.
 */

import { tool } from "@langchain/core/tools";
import type { StructuredToolInterface } from "@langchain/core/tools";
import { z } from "zod";
import { QdrantManager } from "./indexManager.js";
import { UseCaseParamAdvisor } from "./paramAdvisor.js";

// ---------------------------------------------------------------------------
// Schemas shared across tools
// ---------------------------------------------------------------------------

const PointSchema = z.object({
  id: z.union([z.number().int(), z.string()]),
  vector: z.array(z.number()),
  payload: z.record(z.string(), z.unknown()).optional(),
});

const PointsSchema = z.array(PointSchema);

const FilterSchema = z.record(z.string(), z.unknown());

// ---------------------------------------------------------------------------
// makeQdrantTools
// ---------------------------------------------------------------------------

/**
 * Build the ten Qdrant tools bound to a shared manager instance.
 * All tools share one QdrantClient so cross-tool state is consistent.
 */
export function makeQdrantTools(manager: QdrantManager): StructuredToolInterface[] {
  // ── 1. create_qdrant_collection ────────────────────────────────────────
  const createQdrantCollection = tool(
    async (input) => {
      try {
        const result = await manager.autoCreate(input.collection_name, {
          useCase: input.use_case,
          ...(input.vector_size != null ? { vectorSize: input.vector_size } : {}),
          expectedCount: input.expected_count,
          priority: input.priority,
          sparseVectors: input.sparse_vectors,
          ...(input.memory_gb_available != null
            ? { memoryGbAvailable: input.memory_gb_available }
            : {}),
        });
        return [
          `✓ Collection '${input.collection_name}' created.`,
          `  Vector size         : ${result.vector_size}`,
          `  Distance metric     : ${result.distance}`,
          `  Quantization        : ${result.quantization}`,
          `  HNSW m/ef_construct : ${result.hnsw_m} / ${result.hnsw_ef_construct}`,
          `  On-disk payload     : ${result.on_disk_payload}`,
          `  Sparse vectors      : ${result.sparse_vectors}`,
          "",
          "Parameter rationale:",
          ...Object.entries(result.rationale).map(([k, v]) => `  [${k}] ${v}`),
        ].join("\n");
      } catch (err) {
        return `ERROR creating '${input.collection_name}': ${String(err)}`;
      }
    },
    {
      name: "create_qdrant_collection",
      description:
        "Create a Qdrant collection with automatically recommended parameters " +
        "(distance, HNSW, quantization, storage) based on the use-case description.",
      schema: z.object({
        collection_name: z.string().describe("Name for the new Qdrant collection"),
        use_case: z
          .string()
          .describe(
            "Natural-language description of the workload — the more detail the better, " +
              "e.g. 'semantic search over product descriptions using OpenAI ada-002, ~500k products'",
          ),
        vector_size: z
          .number()
          .int()
          .positive()
          .nullish()
          .describe("Embedding dimension. Inferred from use_case keywords when omitted."),
        expected_count: z
          .number()
          .int()
          .positive()
          .default(100_000)
          .describe("Approximate number of vectors"),
        priority: z
          .enum(["recall", "latency", "memory"])
          .default("recall")
          .describe("recall = highest accuracy · latency = fastest queries · memory = smallest RAM"),
        sparse_vectors: z
          .boolean()
          .default(false)
          .describe("Add sparse vector slot for hybrid (dense + sparse) search"),
        memory_gb_available: z
          .number()
          .positive()
          .nullish()
          .describe("Available RAM in GB. Values below 4 trigger product quantization."),
      }),
    },
  );

  // ── 2. ensure_qdrant_collection ────────────────────────────────────────
  const ensureQdrantCollection = tool(
    async (input) => {
      try {
        const result = await manager.ensure(input.collection_name, {
          useCase: input.use_case,
          ...(input.vector_size != null ? { vectorSize: input.vector_size } : {}),
        });
        if (result.action === "already_exists") {
          return (
            `→ '${input.collection_name}' already exists — no action taken.\n` +
            `  Points: ${(result.points_count ?? 0).toLocaleString()}, Status: ${result.status ?? "unknown"}`
          );
        }
        return (
          `✓ '${input.collection_name}' created.\n` +
          `  Vector size: ${result.vector_size}, Distance: ${result.distance}, Quantization: ${result.quantization}`
        );
      } catch (err) {
        return `ERROR in ensure_qdrant_collection: ${String(err)}`;
      }
    },
    {
      name: "ensure_qdrant_collection",
      description:
        "Idempotent create: creates the collection only if it does not already exist.",
      schema: z.object({
        collection_name: z.string().describe("Collection name"),
        use_case: z
          .string()
          .describe("Use-case description for parameter selection if created"),
        vector_size: z
          .number()
          .int()
          .positive()
          .nullish()
          .describe("Vector dimension (inferred if omitted)"),
      }),
    },
  );

  // ── 3. upsert_qdrant_points ────────────────────────────────────────────
  const upsertQdrantPoints = tool(
    async (input) => {
      try {
        const raw: unknown = JSON.parse(input.points);
        const points = PointsSchema.parse(raw);
        const result = await manager.client.upsert(input.collection_name, {
          wait: true,
          points: points.map((p) => ({
            id: p.id,
            vector: p.vector,
            ...(p.payload != null ? { payload: p.payload } : {}),
          })),
        });
        return `✓ Upserted ${points.length} point(s) into '${input.collection_name}'. Status: ${result.status}`;
      } catch (err) {
        return `ERROR upserting into '${input.collection_name}': ${String(err)}`;
      }
    },
    {
      name: "upsert_qdrant_points",
      description:
        "Insert or update vectors. Existing IDs are overwritten; new IDs are inserted.",
      schema: z.object({
        collection_name: z.string().describe("Target collection"),
        points: z
          .string()
          .describe(
            'JSON array of points: [{"id": 1, "vector": [0.1, ...], "payload": {"key": "val"}}]',
          ),
      }),
    },
  );

  // ── 4. search_qdrant_collection ────────────────────────────────────────
  const searchQdrantCollection = tool(
    async (input) => {
      try {
        const queryVec: unknown = JSON.parse(input.query_vector);
        if (!Array.isArray(queryVec) || !queryVec.every((v) => typeof v === "number")) {
          throw new Error("query_vector must be a JSON array of numbers");
        }

        const filterRaw =
          input.filter_conditions != null
            ? FilterSchema.parse(JSON.parse(input.filter_conditions))
            : undefined;

        const results = await manager.client.search(input.collection_name, {
          vector: queryVec as number[],
          limit: input.limit,
          ...(input.score_threshold != null
            ? { score_threshold: input.score_threshold }
            : {}),
          ...(filterRaw !== undefined ? { filter: filterRaw } : {}),
          with_payload: true,
        });

        if (results.length === 0) {
          return "No results (try lowering score_threshold or broadening the filter).";
        }
        return [
          `Found ${results.length} result(s):`,
          ...results.map(
            (r, i) =>
              `  ${i + 1}. id=${r.id}  score=${r.score.toFixed(4)}  payload=${JSON.stringify(r.payload)}`,
          ),
        ].join("\n");
      } catch (err) {
        return `ERROR searching '${input.collection_name}': ${String(err)}`;
      }
    },
    {
      name: "search_qdrant_collection",
      description:
        "ANN search. Returns top-N results with IDs, scores, and payload.",
      schema: z.object({
        collection_name: z.string().describe("Collection to search"),
        query_vector: z
          .string()
          .describe("JSON float array representing the query embedding"),
        limit: z
          .number()
          .int()
          .positive()
          .default(5)
          .describe("Max results to return"),
        score_threshold: z
          .number()
          .nullish()
          .describe("Minimum similarity score (COSINE typically 0.5–0.9)"),
        filter_conditions: z
          .string()
          .nullish()
          .describe(
            'Optional JSON payload filter, e.g. {"must":[{"key":"category","match":{"value":"sports"}}]}',
          ),
      }),
    },
  );

  // ── 5. list_qdrant_collections ─────────────────────────────────────────
  const listQdrantCollections = tool(
    async () => {
      try {
        const { collections } = await manager.client.getCollections();
        if (collections.length === 0) {
          return "No collections found. Create one with create_qdrant_collection.";
        }
        return [
          `Found ${collections.length} collection(s):`,
          ...collections.map((c) => `  • ${c.name}`),
        ].join("\n");
      } catch (err) {
        return `ERROR listing collections: ${String(err)}`;
      }
    },
    {
      name: "list_qdrant_collections",
      description:
        "List all Qdrant collections. Call this first to avoid name collisions.",
      schema: z.object({}),
    },
  );

  // ── 6. describe_qdrant_collection ──────────────────────────────────────
  const describeQdrantCollection = tool(
    async (input) => {
      try {
        const h = await manager.health(input.collection_name);
        const lines = [
          `Collection: ${input.collection_name}`,
          `  Status           : ${h.status}`,
          `  Points           : ${h.points_count.toLocaleString()}`,
          `  Indexed vectors  : ${h.indexed_vectors_count.toLocaleString()}`,
          `  Vector size      : ${h.vector_size ?? "n/a"}`,
          `  Distance         : ${h.distance ?? "n/a"}`,
          `  Optimizer status : ${h.optimizer_status}`,
        ];
        if (h.recommendation) lines.push(`\n⚠  Recommendation: ${h.recommendation}`);
        return lines.join("\n");
      } catch (err) {
        return `ERROR describing '${input.collection_name}': ${String(err)}`;
      }
    },
    {
      name: "describe_qdrant_collection",
      description:
        "Return stats and optimisation recommendations for a collection.",
      schema: z.object({
        collection_name: z.string().describe("Collection to inspect"),
      }),
    },
  );

  // ── 7. create_qdrant_payload_index ─────────────────────────────────────
  const createQdrantPayloadIndex = tool(
    async (input) => {
      try {
        await manager.createPayloadIndex(
          input.collection_name,
          input.field_name,
          input.field_type,
        );
        return (
          `✓ Payload index on '${input.field_name}' (${input.field_type}) ` +
          `created in '${input.collection_name}'.`
        );
      } catch (err) {
        return `ERROR creating payload index: ${String(err)}`;
      }
    },
    {
      name: "create_qdrant_payload_index",
      description:
        "Index a payload field for fast filtered search. Always index fields you filter on.",
      schema: z.object({
        collection_name: z.string().describe("Target collection"),
        field_name: z
          .string()
          .describe("Payload key to index, e.g. 'category' or 'price'"),
        field_type: z
          .enum(["keyword", "integer", "float", "geo", "text", "bool", "datetime"])
          .default("keyword")
          .describe("keyword · integer · float · geo · text · bool · datetime"),
      }),
    },
  );

  // ── 8. recommend_qdrant_params ─────────────────────────────────────────
  const recommendQdrantParams = tool(
    async (input) => {
      try {
        const advice = new UseCaseParamAdvisor().advise({
          useCase: input.use_case,
          ...(input.vector_size != null ? { vectorSize: input.vector_size } : {}),
          expectedCount: input.expected_count,
          priority: input.priority,
        });
        return [
          "Recommended parameters (dry-run — collection NOT created):",
          `  Distance          : ${advice.distance}`,
          `  Vector size       : ${advice.vector_size}`,
          `  HNSW m            : ${advice.hnsw_config.m}`,
          `  HNSW ef_construct : ${advice.hnsw_config.ef_construct}`,
          `  Quantization      : ${advice.quantization_mode}`,
          `  On-disk payload   : ${advice.on_disk_payload}`,
          "",
          "Rationale:",
          ...Object.entries(advice.rationale).map(([k, v]) => `  [${k}] ${v}`),
        ].join("\n");
      } catch (err) {
        return `ERROR generating recommendations: ${String(err)}`;
      }
    },
    {
      name: "recommend_qdrant_params",
      description: "Dry-run the parameter advisor without creating a collection.",
      schema: z.object({
        use_case: z
          .string()
          .describe("Natural-language description of the intended use case"),
        vector_size: z
          .number()
          .int()
          .positive()
          .nullish()
          .describe("Embedding dimension if known"),
        expected_count: z
          .number()
          .int()
          .positive()
          .default(100_000)
          .describe("Expected vector count"),
        priority: z
          .enum(["recall", "latency", "memory"])
          .default("recall")
          .describe("Optimisation priority"),
      }),
    },
  );

  // ── 9. delete_qdrant_collection ────────────────────────────────────────
  const deleteQdrantCollection = tool(
    async (input) => {
      try {
        await manager.delete(input.collection_name);
        return `✓ Collection '${input.collection_name}' deleted.`;
      } catch (err) {
        return `ERROR deleting '${input.collection_name}': ${String(err)}`;
      }
    },
    {
      name: "delete_qdrant_collection",
      description:
        "Permanently delete a collection and all its vectors. This operation is irreversible.",
      schema: z.object({
        collection_name: z
          .string()
          .describe("Name of the collection to delete"),
      }),
    },
  );

  // ── 10. optimize_qdrant_collection ─────────────────────────────────────
  const optimizeQdrantCollection = tool(
    async (input) => {
      try {
        const result = await manager.optimize(input.collection_name, input.priority);
        return [
          `✓ Optimizer updated for '${input.collection_name}' (priority=${input.priority}).`,
          `  indexing_threshold : ${result.applied_optimizers.indexing_threshold.toLocaleString()}`,
          `  memmap_threshold   : ${result.applied_optimizers.memmap_threshold.toLocaleString()}`,
        ].join("\n");
      } catch (err) {
        return `ERROR optimizing '${input.collection_name}': ${String(err)}`;
      }
    },
    {
      name: "optimize_qdrant_collection",
      description:
        "Update indexing/memmap thresholds on an existing collection without touching vector data.",
      schema: z.object({
        collection_name: z.string().describe("Collection to optimise"),
        priority: z
          .enum(["recall", "latency", "memory"])
          .default("recall")
          .describe("Optimisation goal"),
      }),
    },
  );

  return [
    createQdrantCollection,
    ensureQdrantCollection,
    upsertQdrantPoints,
    searchQdrantCollection,
    listQdrantCollections,
    describeQdrantCollection,
    createQdrantPayloadIndex,
    recommendQdrantParams,
    deleteQdrantCollection,
    optimizeQdrantCollection,
  ];
}
