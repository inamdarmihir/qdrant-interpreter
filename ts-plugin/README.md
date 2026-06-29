# qdrant-deepagent

**Qdrant vector-database plugin for Deep Agents** — a TypeScript package that gives any agent automatic, intelligent Qdrant collection management through a QuickJS interpreter and a [programmatic tool-calling (PTC)](https://www.langchain.com/blog/give-your-agents-an-interpreter) bridge.

The agent writes JavaScript once, batches all Qdrant operations in one `eval` call, and returns only the final result to the model context. No serial round-trips. No hand-tuned parameters. No boilerplate.

---

## Quick install

```bash
npm install qdrant-deepagent
# peer deps for the full agent
npm install deepagents @langchain/quickjs @langchain/core
```

```bash
pnpm add qdrant-deepagent deepagents @langchain/quickjs @langchain/core
```

---

## Three ways to use

### 1 — Full agent (one line)

```typescript
import { createQdrantAgent } from "qdrant-deepagent";

const { invoke } = createQdrantAgent({
  model: "openai:gpt-4o",          // any deepagents model string
  url: "http://localhost:6333",     // default
});

const reply = await invoke(
  "Create a semantic search collection for 500k product descriptions " +
  "using OpenAI ada-002. Add indexes for category and price."
);

console.log(reply);
```

The agent writes interpreter code like this and executes it in one shot:

```javascript
const [col, catIdx, priceIdx] = await Promise.all([
  tools.createQdrantCollection({
    collection_name: "products",
    use_case: "semantic search using OpenAI ada-002, 500k products",
    expected_count: 500_000,
    priority: "recall",
  }),
  tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "category", field_type: "keyword" }),
  tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "price",    field_type: "float"   }),
]);
col
```

---

### 2 — Tools only (plug into any LangChain agent)

Use the tools without the interpreter — pass them directly to any
`createReactAgent`, `AgentExecutor`, or similar:

```typescript
import { QdrantClient } from "@qdrant/js-client-rest";
import { QdrantManager, makeQdrantTools } from "qdrant-deepagent";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";

const client  = new QdrantClient({ url: "http://localhost:6333" });
const manager = new QdrantManager(client);
const tools   = makeQdrantTools(manager);

const agent = createReactAgent({
  llm: new ChatOpenAI({ model: "gpt-4o" }),
  tools,
});
```

---

### 3 — Interpreter skill

Drop the skill directory into your agent's skill folder. The agent discovers it
through progressive disclosure and imports the module when the task matches.

`skills/qdrant-manager/SKILL.md` — copy from `node_modules/qdrant-deepagent/src/skill/SKILL.md`

```typescript
// Inside the interpreter (the agent writes this):
const qdrant = await import("@/skills/qdrant-manager");

const col = await qdrant.createCollection("products", {
  useCase: "semantic search over product descriptions using OpenAI ada-002",
  expectedCount: 500_000,
  priority: "recall",
});

await Promise.all([
  qdrant.createPayloadIndex("products", "category", "keyword"),
  qdrant.createPayloadIndex("products", "price", "float"),
]);

col
```

---

## Configuration

```typescript
createQdrantAgent({
  model:               "openai:gpt-4o",   // required
  url:                 "http://localhost:6333",
  apiKey:              undefined,          // Qdrant Cloud
  systemPrompt:        undefined,          // override built-in prompt
  ptc:                 undefined,          // string[] allowlist; undefined = all 10 tools
  executionTimeoutMs:  10_000,             // per-eval QuickJS timeout
  memoryLimitBytes:    64 * 1024 * 1024,  // QuickJS heap cap
  maxPtcCalls:         256,               // max tools.* calls per eval
  maxResultChars:      4_000,             // truncation limit for result/stdout
  captureConsole:      true,              // collect console.log output
  subagents:           true,              // expose task() for dynamic subagents
});
```

---

## Available tools

All tools are callable from agent code directly **or** via `tools.*` inside the interpreter:

| Tool | Description |
|---|---|
| `create_qdrant_collection` | Create with auto-tuned parameters |
| `ensure_qdrant_collection` | Idempotent create-if-absent |
| `list_qdrant_collections` | Enumerate existing collections |
| `describe_qdrant_collection` | Stats + optimisation recommendations |
| `delete_qdrant_collection` | Permanent delete |
| `upsert_qdrant_points` | Insert/update vectors (JSON array) |
| `search_qdrant_collection` | ANN search with optional payload filter |
| `create_qdrant_payload_index` | Index a field for fast filtered search |
| `recommend_qdrant_params` | Dry-run the advisor without creating |
| `optimize_qdrant_collection` | Tune indexing/memmap thresholds |

---

## Auto-parameter selection

`create_qdrant_collection` and `recommend_qdrant_params` delegate to a fully
rule-based `UseCaseParamAdvisor` — deterministic, no LLM, no network calls.

### Distance metric

| Signal in `use_case` | Metric |
|---|---|
| `semantic`, `text`, `openai`, `bert`, `rag` | **Cosine** (default) |
| `anomaly`, `cluster`, `euclidean`, `tabular` | **Euclid** |
| `collaborative filtering`, `dot product` | **Dot** |
| `sparse`, `tfidf`, `bm25` | **Manhattan** |

### Vector size inference

| Keyword | Size |
|---|---|
| `all-minilm`, `e5-small`, `bge-small` | 384 |
| `bert-base`, `mpnet`, `e5-base`, `bge-base` | 768 |
| `e5-large`, `bge-large`, `cohere` | 1 024 |
| `openai`, `ada-002`, `text-embedding-3-small` | 1 536 |
| `text-embedding-3-large` | 3 072 |
| `clip` | 512 |

### HNSW + quantization

| Priority | Scale | HNSW m/ef | Quantization |
|---|---|---|---|
| `recall` | < 100k | 16 / 100 | None |
| `recall` | 100k–1M | 16 / 200 | Scalar INT8 |
| `recall` | > 1M | 32 / 200 | Scalar INT8 |
| `latency` | any | 16 / 100 | Binary |
| `memory` | any | 8 / 100 | Product 16× |

---

## Use the advisor directly

```typescript
import { UseCaseParamAdvisor } from "qdrant-deepagent";

const advisor = new UseCaseParamAdvisor();
const advice  = advisor.advise({
  useCase:       "semantic search using OpenAI ada-002",
  expectedCount: 500_000,
  priority:      "recall",
});

console.log(advice.distance);           // "Cosine"
console.log(advice.quantization_mode);  // "scalar"
console.log(advice.rationale);          // plain-English explanation for each decision
```

---

## Security

The interpreter runs code in an embedded QuickJS context — not a subprocess or
container. By default it has no filesystem, network, shell, or clock access.
Only the tools you allowlist via `ptc` are reachable.

| Capability | Default | How to enable |
|---|---|---|
| JavaScript execution | ✓ | Add interpreter middleware |
| `console.log` capture | ✓ | Disable with `captureConsole: false` |
| Qdrant tools | ✗ | Add tool names to `ptc` (or omit = all) |
| Filesystem / network / shell | ✗ | Use a sandbox backend |

---

## Package structure

```
src/
├── paramAdvisor.ts   Rule-based parameter selection (no LLM)
├── indexManager.ts   QdrantManager — high-level collection CRUD
├── tools.ts          makeQdrantTools() — 10 LangChain StructuredTools
├── agent.ts          createQdrantAgent() — one-liner factory
├── skill/
│   ├── SKILL.md      Interpreter skill descriptor (copy to your skills dir)
│   └── index.ts      Skill module — camelCase API over the PTC bridge
└── index.ts          Public package exports
```

---

## License

MIT
