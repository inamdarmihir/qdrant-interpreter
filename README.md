# Qdrant Interpreter Plugin

A LangChain / LangGraph agent that **auto-creates and manages Qdrant collections** with the best parameters for your use case — no manual tuning required.

Give it a plain-English description of what you're building and it selects the right distance metric, HNSW graph settings, quantization strategy, and payload indexes automatically. An `eval_interpreter` escape hatch lets the agent fall back to executing arbitrary Python against the same Qdrant client when structured tools are not enough.

---

## Features

| Capability | Description |
|---|---|
| **Auto-parameter selection** | Distance metric, HNSW m/ef, quantization, on-disk settings — inferred from use-case text |
| **10 structured tools** | Create, upsert, search, index, optimize, inspect — no raw client calls needed |
| **Payload index management** | One call to enable fast filtered queries on any metadata field |
| **eval_interpreter** | Exec-based escape hatch for arbitrary multi-step Python in the same Qdrant context |
| **LangGraph ReAct agent** | Full reasoning loop — the agent explains every parameter choice it makes |
| **In-memory or remote** | Works with `:memory:`, local disk, or Qdrant Cloud |

---

## Installation

```bash
pip install qdrant-client

# Full agent (LangChain + LangGraph + OpenAI)
pip install langchain-core langchain-openai langgraph pydantic
```

Clone and install the package:

```bash
git clone https://github.com/inamdarmihir/qdrant-interpreter.git
cd qdrant-interpreter
pip install -e .
```

---

## Quick Start

### Full agent — natural-language interface

```python
from langchain_openai import ChatOpenAI
from qdrant_interpreter_plugin import QdrantAgentInterpreter

agent = QdrantAgentInterpreter(llm=ChatOpenAI(model="gpt-4o-mini"))

# Agent creates the collection, picks parameters, and explains the choices
print(agent.run(
    "Create a collection for semantic search over product descriptions "
    "using OpenAI ada-002 embeddings. Expect 500k products. "
    "We'll filter by category and price range."
))
```

**Example output:**
```
✓ Collection 'products' created successfully.
  Vector size       : 1536
  Distance metric   : Cosine
  Quantization      : scalar
  HNSW m / ef_construct : 16 / 200
  On-disk payload   : False

Parameter rationale:
  [distance] COSINE selected (default) — optimal for normalized text/semantic embeddings.
  [vector_size] Inferred from model keyword 'ada': size=1536
  [hnsw] HNSW m=16, ef_construct=200: balanced recall for medium collection.
  [quantization] Scalar INT8 quantization: ~4× memory reduction, <1% recall loss.
```

### No LLM — direct index manager

```python
from qdrant_client import QdrantClient
from qdrant_interpreter_plugin import QdrantIndexManager

manager = QdrantIndexManager(QdrantClient(":memory:"))

# Create collection with auto-recommended parameters
result = manager.auto_create(
    collection_name="products",
    use_case="semantic search over product descriptions using OpenAI ada-002",
    expected_count=500_000,
    priority="recall",        # "recall" | "latency" | "memory"
)
print(result["distance"])     # Cosine
print(result["quantization"]) # scalar
print(result["hnsw_m"])       # 16

# Add payload indexes for fast filtered queries
manager.create_payload_index("products", "category", "keyword")
manager.create_payload_index("products", "price",    "float")

# Health check with recommendations
health = manager.health("products")
print(health["status"])           # green
print(health.get("recommendation"))  # advisory if tuning is beneficial
```

### No LLM — parameter advisor only

```python
from qdrant_interpreter_plugin import UseCaseParamAdvisor

advisor = UseCaseParamAdvisor()
advice = advisor.advise(
    use_case="anomaly detection on sensor time-series with Euclidean distance",
    expected_count=1_000_000,
    priority="latency",
)
print(advice.distance)              # Distance.EUCLID
print(advice.quantization_mode)     # "binary"
print(advice.rationale["hnsw"])     # HNSW m=16, ef_construct=100: latency-optimised ...
```

### eval_interpreter — arbitrary Python against Qdrant

```python
from qdrant_interpreter_plugin import QdrantInterpreter

interp = QdrantInterpreter()

# All Qdrant model types are pre-imported in the exec namespace
interp.execute("""
client.create_collection(
    collection_name="demo",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
    ),
)
""")

result = interp.execute("""
hits = client.search("demo", query_vector=[0.1]*768, limit=5, with_payload=True)
for h in hits:
    print(h.id, h.score)
""")
print(result["stdout"])
print(result["success"])   # True
```

---

## Parameter Selection Logic

The `UseCaseParamAdvisor` is fully rule-based — no LLM needed, fully deterministic.

### Distance metric

| Signal in use-case text | Selected metric |
|---|---|
| `semantic`, `text`, `nlp`, `openai`, `bert`, `rag`, `recommendation` | **COSINE** (default) |
| `anomaly`, `outlier`, `cluster`, `euclidean`, `tabular`, `time-series` | **EUCLID** |
| `collaborative filtering`, `matrix factorization`, `dot product` | **DOT** |
| `sparse`, `tfidf`, `bm25`, `bag of words` | **MANHATTAN** |

### Vector size inference

Extracted from explicit dimension numbers (`"size 1536"`, `"dim=384"`) or model-family keywords:

| Keyword | Size |
|---|---|
| `all-minilm`, `e5-small`, `bge-small` | 384 |
| `bert-base`, `mpnet`, `e5-base`, `bge-base`, `gte-base` | 768 |
| `e5-large`, `bge-large`, `bert-large`, `cohere` | 1024 |
| `openai`, `ada-002`, `text-embedding-3-small` | 1536 |
| `text-embedding-3-large` | 3072 |
| `clip` | 512 |
| *(default)* | 768 |

### HNSW parameters

| Priority | Collection size | m | ef_construct |
|---|---|---|---|
| `recall` | < 100k | 16 | 100 |
| `recall` | 100k – 1M | 16 | 200 |
| `recall` | > 1M | 32 | 200 |
| `latency` | any | 16 | 100 |
| `memory` | any | 8 | 100 |

### Quantization

| Condition | Strategy |
|---|---|
| Small collection + `recall` | None |
| Large + `recall` | Scalar INT8 (~4× compression, <1% recall loss) |
| `latency` + large | Binary (fastest search, re-scoring restores accuracy) |
| `memory` or RAM < 4 GB | Product 16× (most aggressive compression) |

---

## Tools Reference

All 10 tools are available to the `QdrantAgentInterpreter` and can also be used standalone via `make_qdrant_tools(manager)`.

| Tool | Description |
|---|---|
| `create_qdrant_collection` | Create with auto-recommended parameters |
| `ensure_qdrant_collection` | Create only if absent (idempotent) |
| `upsert_qdrant_points` | Batch insert / update (JSON list of points) |
| `search_qdrant_collection` | ANN search with optional payload filter |
| `list_qdrant_collections` | List all collections |
| `describe_qdrant_collection` | Stats, vector config, and health recommendations |
| `create_qdrant_payload_index` | Index a payload field for fast filtered search |
| `recommend_qdrant_params` | Dry-run advisor — preview params without creating |
| `delete_qdrant_collection` | Permanently delete a collection |
| `optimize_qdrant_collection` | Update indexing/memmap thresholds on an existing collection |

### Using tools without an agent

```python
from qdrant_client import QdrantClient
from qdrant_interpreter_plugin import QdrantIndexManager
from qdrant_interpreter_plugin.tools import make_qdrant_tools
import json

manager = QdrantIndexManager(QdrantClient(":memory:"))
tools = {t.name: t for t in make_qdrant_tools(manager)}

tools["create_qdrant_collection"].invoke({
    "collection_name": "articles",
    "use_case": "semantic search over news articles using OpenAI embeddings",
    "expected_count": 1_000_000,
    "priority": "recall",
})

tools["create_qdrant_payload_index"].invoke({
    "collection_name": "articles",
    "field_name": "published_at",
    "field_type": "datetime",
})

tools["upsert_qdrant_points"].invoke({
    "collection_name": "articles",
    "points": json.dumps([
        {"id": 1, "vector": [0.1] * 1536, "payload": {"title": "Breaking News"}},
    ]),
})
```

---

## Remote Qdrant

```python
agent = QdrantAgentInterpreter(
    llm=ChatOpenAI(model="gpt-4o"),
    location="https://xyz.qdrant.tech",
    api_key="your-api-key",
)
```

Or use the index manager directly:

```python
from qdrant_client import QdrantClient
from qdrant_interpreter_plugin import QdrantIndexManager

client = QdrantClient(
    url="https://xyz.qdrant.tech",
    api_key="your-api-key",
)
manager = QdrantIndexManager(client)
```

---

## Streaming (agent)

```python
for chunk in agent.stream("Create a collection for image similarity search"):
    if "messages" in chunk:
        for msg in chunk["messages"]:
            print(msg.content, end="", flush=True)
```

---

## Package Structure

```
qdrant_interpreter_plugin/
├── __init__.py            Public API surface
├── agent.py               QdrantAgentInterpreter — LangGraph ReAct agent
├── index_manager.py       QdrantIndexManager — smart collection CRUD
├── param_advisor.py       UseCaseParamAdvisor — rule-based parameter selection
├── qdrant_interpreter.py  QdrantInterpreter — exec-based interpreter + eval_interpreter tool
├── tools.py               make_qdrant_tools() — 10 structured LangChain tools
└── examples/
    └── agent_demo.py      Runnable demo (param advisor, index manager, eval, full agent)
```

---

## License

Educational use. Adapt freely.
