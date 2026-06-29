# Qdrant Interpreter Plugin

An implementation of the **interpreter pattern** from the LangChain blog post
[*"Give Your Agents an Interpreter"*](https://www.langchain.com/blog/give-your-agents-an-interpreter)
applied to Qdrant vector database management.

Instead of calling Qdrant tools one-at-a-time (one model round-trip each), the
agent writes a short JavaScript program and calls the `eval` tool **once**.
Inside a sandboxed **QuickJS** VM, `await tools.createQdrantCollection({...})`
crosses a PTC bridge and invokes the real Python Qdrant tool — no extra
round-trip. Multiple operations batch with `Promise.all()`.

---

## How the interpreter pattern works

```
User query
   │
   ▼
Model reasons → writes JavaScript
   │
   ▼
  eval tool (one call)
   │
   ▼
 QuickJS sandbox  ──── no fs / net / shell ────
   │
   ├── await tools.createQdrantCollection({...})
   │       └── PTC bridge → Python Qdrant tool → QdrantIndexManager → param_advisor
   │
   ├── await Promise.all([tools.createQdrantPayloadIndex(...), ...])
   │       └── parallel bridge calls, all in one eval
   │
   └── final expression  ──────────────────────────────── returned to model
```

**Key properties (matching the blog spec):**

| Property | Value |
|---|---|
| Runtime | QuickJS VM via `quickjs-rs` — not the host Python process |
| Language | JavaScript / TypeScript |
| State | Persists across `eval` calls (REPL-style; `const` → `var` transform) |
| Isolation | No filesystem, network, or shell by default |
| Tool access | Explicit PTC bridge — allowlisted tools only |
| Memory limit | Configurable (default 64 MB) |
| Timeout | Per-eval (default 10 s) |
| Max tool calls | Per-eval budget (default 256) |
| `console.log` | Captured in `<stdout>` blocks |
| Snapshots | Serialize / restore full JS heap across turns |

---

## Installation

```bash
# Core — QuickJS interpreter without the full agent
pip install qdrant-client quickjs-rs

# Full agent — adds CodeInterpreterMiddleware + create_deep_agent
pip install "qdrant-client" "deepagents[quickjs]" "langchain-quickjs" \
            "langchain-anthropic"   # or langchain-openai

# Clone and install
git clone https://github.com/inamdarmihir/qdrant-interpreter.git
cd qdrant-interpreter
pip install -e ".[agent]"
```

---

## Quick Start

### Full agent

```python
from langchain_anthropic import ChatAnthropic
from qdrant_interpreter_plugin import QdrantAgentInterpreter

agent = QdrantAgentInterpreter(
    model=ChatAnthropic(model_name="claude-sonnet-4-6"),
)

# Agent writes JS, batches operations, returns only the final result
print(agent.run(
    "Create a collection for semantic search over product descriptions "
    "using OpenAI ada-002 embeddings. Expect 500k products. "
    "We need to filter by category and price range."
))
```

The model writes something like:

```javascript
// Batch create + indexes in one eval call
const [col, catIdx, priceIdx] = await Promise.all([
  tools.createQdrantCollection({
      collection_name: "products",
      use_case: "semantic search with OpenAI ada-002, 500k products",
      expected_count: 500000,
      priority: "recall",
  }),
  tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "category", field_type: "keyword" }),
  tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "price",    field_type: "float" }),
]);
col   // ← final expression returned to model context
```

### Standalone QuickJS interpreter (no LLM)

Drive the sandbox directly — useful for scripting or testing:

```python
from qdrant_client import QdrantClient
from qdrant_interpreter_plugin import QdrantIndexManager, QdrantInterpreter

manager = QdrantIndexManager(QdrantClient(":memory:"))
interp = QdrantInterpreter(manager, timeout=15.0)

result = interp.eval("""
    const col = await tools.createQdrantCollection({
        collection_name: "docs",
        use_case: "semantic search using OpenAI ada-002",
        expected_count: 500_000,
        priority: "recall",
    });
    console.log("Created:", col);

    const [a, b] = await Promise.all([
        tools.createQdrantPayloadIndex({ collection_name: "docs", field_name: "tag",  field_type: "keyword" }),
        tools.createQdrantPayloadIndex({ collection_name: "docs", field_name: "date", field_type: "datetime" }),
    ]);

    await tools.describeQdrantCollection({ collection_name: "docs" })
""")

print(result["stdout"])   # console.log output
print(result["result"])   # final expression value
```

### REPL state persists across calls

```python
interp.eval("const counter = 0")   # const → var, survives across evals
interp.eval("counter += 1")
print(interp.eval("counter")["result"])   # "1"
```

### Snapshot / restore

```python
snap = interp.snapshot()       # serialize JS heap to bytes
interp.restore(snap)           # restore (bridges re-registered automatically)
```

---

## Qdrant Tools (PTC namespace)

All tools are available as `async` functions inside the QuickJS sandbox:

```javascript
await tools.createQdrantCollection({ collection_name, use_case, vector_size?,
                                     expected_count?, priority?, sparse_vectors?,
                                     memory_gb_available? })
await tools.ensureQdrantCollection({ collection_name, use_case, vector_size? })
await tools.upsertQdrantPoints({ collection_name, points })   // points = JSON string
await tools.searchQdrantCollection({ collection_name, query_vector, limit?,
                                     score_threshold?, filter_conditions? })
await tools.listQdrantCollections()
await tools.describeQdrantCollection({ collection_name })
await tools.createQdrantPayloadIndex({ collection_name, field_name, field_type? })
await tools.recommendQdrantParams({ use_case, vector_size?, expected_count?, priority? })
await tools.deleteQdrantCollection({ collection_name })
await tools.optimizeQdrantCollection({ collection_name, priority? })
```

---

## Auto-Parameter Selection

`createQdrantCollection` delegates parameter selection to `UseCaseParamAdvisor`,
a fully rule-based engine — no LLM, no network, deterministic.

### Distance metric

| Signal in `use_case` | Metric |
|---|---|
| `semantic`, `text`, `openai`, `bert`, `rag` | **COSINE** (default) |
| `anomaly`, `cluster`, `euclidean`, `tabular` | **EUCLID** |
| `collaborative filtering`, `dot product` | **DOT** |
| `sparse`, `tfidf`, `bm25` | **MANHATTAN** |

### Vector size inference

Extracted from explicit dim numbers (`"size 1536"`) or model keywords:

| Keyword | Size |
|---|---|
| `all-minilm`, `e5-small` | 384 |
| `bert-base`, `mpnet`, `e5-base` | 768 |
| `e5-large`, `bge-large`, `cohere` | 1024 |
| `openai`, `ada-002`, `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
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

## Configuration

```python
agent = QdrantAgentInterpreter(
    model="openai:gpt-4o",          # string or BaseChatModel
    location=":memory:",            # or "https://xyz.qdrant.tech"
    api_key=None,                   # Qdrant Cloud API key
    timeout=10.0,                   # per-eval wall-clock limit (s)
    memory_limit_mb=64,             # QuickJS heap cap
    max_ptc_calls=256,              # max tools.* calls per eval
    max_result_chars=4_000,         # truncation limit for result/stdout
    capture_console=True,           # collect console.log output
    ptc=None,                       # tool allowlist; None = all 10 tools
    snapshot_between_turns=True,    # persist interpreter state across turns
    subagents=True,                 # expose task() for dynamic subagents
)
```

---

## Package Structure

```
qdrant_interpreter_plugin/
├── __init__.py            Public API surface
├── agent.py               QdrantAgentInterpreter — create_deep_agent + CodeInterpreterMiddleware
├── index_manager.py       QdrantIndexManager — smart collection CRUD
├── param_advisor.py       UseCaseParamAdvisor — rule-based parameter selection
├── qdrant_interpreter.py  QdrantInterpreter — QuickJS REPL + PTC bridges + LangChain tool
├── tools.py               make_qdrant_tools() — 10 structured LangChain tools
└── examples/
    └── agent_demo.py      Runnable demo (4 sections; --no-llm skips the agent)
```

---

## License

Educational use. Adapt freely.
