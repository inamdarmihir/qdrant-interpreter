---
name: qdrant-manager
description: >
  Use this skill to create, query, and manage Qdrant vector database
  collections. Handles collection creation with automatic parameter
  selection, payload indexing, vector search, upsert, and optimisation.
metadata:
  module: ./index.ts
---

Use this skill whenever the user needs to work with a Qdrant vector database.

Import the module from the interpreter and call the exported functions:

```ts
const qdrant = await import("@/skills/qdrant-manager");

// Create a collection with automatically selected parameters
const col = await qdrant.createCollection("products", {
  useCase: "semantic search over product descriptions using OpenAI ada-002",
  expectedCount: 500_000,
  priority: "recall",
});

// Add payload indexes for fast filtered search
await qdrant.createPayloadIndex("products", "category", "keyword");
await qdrant.createPayloadIndex("products", "price", "float");
```

Batch multiple operations with `Promise.all` for efficiency:

```ts
const qdrant = await import("@/skills/qdrant-manager");

const [col, catIdx, priceIdx] = await Promise.all([
  qdrant.createCollection("products", {
    useCase: "semantic search using OpenAI ada-002, 500k products",
    expectedCount: 500_000,
    priority: "recall",
  }),
  qdrant.createPayloadIndex("products", "category", "keyword"),
  qdrant.createPayloadIndex("products", "price", "float"),
]);

col   // ← returned to the model
```

## Available functions

| Function | Description |
|---|---|
| `createCollection(name, options)` | Create with auto-tuned parameters |
| `ensureCollection(name, options)` | Idempotent create-if-absent |
| `deleteCollection(name)` | Permanent delete |
| `listCollections()` | Enumerate existing collections |
| `describeCollection(name)` | Stats and recommendations |
| `upsertPoints(name, points)` | Insert/update vectors |
| `searchCollection(name, queryVector, options?)` | ANN search |
| `createPayloadIndex(name, field, type?)` | Index for fast filtering |
| `recommendParams(options)` | Dry-run advisor (no collection created) |
| `optimizeCollection(name, priority?)` | Tune optimizer settings |

## Priority options

- **`"recall"`** (default) — highest accuracy, good for most workloads
- **`"latency"`** — fastest queries, trades some memory for speed
- **`"memory"`** — smallest RAM footprint, aggressive quantization

## Payload field types

`keyword` · `integer` · `float` · `geo` · `text` · `bool` · `datetime`

## Constraints

- Always call `ensureCollection` before assuming a collection exists.
- Create payload indexes for every field you intend to filter on.
- `upsertPoints` points must include `id` (number or UUID string) and `vector` (float array).
