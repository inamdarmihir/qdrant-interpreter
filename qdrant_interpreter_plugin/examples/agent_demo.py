"""
Qdrant Agent Interpreter — Demo
================================

This script showcases the full capabilities of :class:`QdrantAgentInterpreter`:

  1. Creating a collection with auto-selected parameters for a specific use case
  2. Adding payload indexes for fast filtered queries
  3. Inserting vectors and running searches
  4. Inspecting collection health and getting recommendations
  5. Falling back to the ``eval_interpreter`` for complex multi-step operations

Prerequisites
-------------
Install the full dependency set::

    pip install qdrant-client langchain-core langchain-openai langgraph pydantic

Then set your OpenAI API key::

    export OPENAI_API_KEY=sk-...

Run this demo::

    python agent_demo.py

What happens
------------
The demo creates three agents (all sharing the same in-memory Qdrant client)
and sends them increasingly complex queries to show how each tool type is used.

No actual network requests to Qdrant are made — everything runs in the local
in-process ``:memory:`` backend.
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Guard: fail fast with a clear message if dependencies are missing
# ---------------------------------------------------------------------------
try:
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
except ImportError as exc:
    print(
        f"Missing dependency: {exc}\n"
        "Install with: pip install langchain-openai langgraph pydantic",
        file=sys.stderr,
    )
    sys.exit(1)

from qdrant_interpreter_plugin import QdrantAgentInterpreter, QdrantIndexManager
from qdrant_interpreter_plugin.param_advisor import UseCaseParamAdvisor


# ---------------------------------------------------------------------------
# Demo 1: Parameter advisor (no LLM required)
# ---------------------------------------------------------------------------

def demo_param_advisor() -> None:
    """Show the rule-based parameter advisor in isolation."""
    print("\n" + "=" * 60)
    print("DEMO 1: UseCaseParamAdvisor (no LLM)")
    print("=" * 60)

    advisor = UseCaseParamAdvisor()

    cases = [
        (
            "Semantic search over product descriptions using OpenAI ada-002",
            None, 500_000, "recall",
        ),
        (
            "Anomaly detection on time-series sensor readings with Euclidean distance",
            64, 10_000, "latency",
        ),
        (
            "Collaborative filtering recommendations with matrix factorization embeddings",
            128, 2_000_000, "memory",
        ),
    ]

    for use_case, vec_size, count, priority in cases:
        advice = advisor.advise(
            use_case=use_case,
            vector_size=vec_size,
            expected_count=count,
            priority=priority,
        )
        print(f"\nUse case : {use_case[:60]}...")
        print(f"  Distance     : {advice.distance.value}")
        print(f"  Vector size  : {advice.vector_size}")
        print(f"  HNSW m/ef    : {advice.hnsw_config.m} / {advice.hnsw_config.ef_construct}")
        print(f"  Quantization : {advice.quantization_mode}")
        print(f"  On-disk      : {advice.on_disk_payload}")
        print("  Rationale:")
        for k, v in advice.rationale.items():
            print(f"    [{k}] {v}")


# ---------------------------------------------------------------------------
# Demo 2: QdrantIndexManager (no LLM required)
# ---------------------------------------------------------------------------

def demo_index_manager() -> None:
    """Show the QdrantIndexManager creating and inspecting collections."""
    print("\n" + "=" * 60)
    print("DEMO 2: QdrantIndexManager (no LLM)")
    print("=" * 60)

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    client = QdrantClient(":memory:")
    manager = QdrantIndexManager(client)

    # Create a collection
    result = manager.auto_create(
        collection_name="products",
        use_case="semantic search over e-commerce product descriptions using OpenAI ada-002",
        expected_count=200_000,
        priority="recall",
    )
    print(f"\nCreated collection: {result['collection_name']}")
    print(f"  distance    = {result['distance']}")
    print(f"  vector_size = {result['vector_size']}")
    print(f"  quant       = {result['quantization']}")

    # Create payload indexes
    for field, ftype in [("category", "keyword"), ("price", "float"), ("in_stock", "bool")]:
        manager.create_payload_index("products", field, ftype)
        print(f"  Indexed payload field: {field} ({ftype})")

    # Upsert a few dummy vectors
    vsize = result["vector_size"]
    points = [
        PointStruct(id=i, vector=[0.1 * i] * vsize, payload={"category": "shoes", "price": 49.99 + i})
        for i in range(1, 6)
    ]
    client.upsert("products", points=points)
    print(f"\n  Upserted {len(points)} points")

    # Health check
    health = manager.health("products")
    print(f"\nCollection health:")
    print(f"  points_count = {health['points_count']}")
    print(f"  status       = {health['status']}")
    rec = health.get("recommendation")
    if rec:
        print(f"  ⚠ {rec}")

    # ensure is idempotent
    result2 = manager.ensure("products", use_case="semantic search", vector_size=vsize)
    print(f"\n  ensure() action: {result2['action']}")


# ---------------------------------------------------------------------------
# Demo 3: Full agent (LLM required — needs OPENAI_API_KEY)
# ---------------------------------------------------------------------------

def demo_agent() -> None:
    """Run the full LangGraph agent with natural-language queries."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[SKIP] DEMO 3: set OPENAI_API_KEY to run the full agent demo.")
        return

    print("\n" + "=" * 60)
    print("DEMO 3: QdrantAgentInterpreter (LLM-powered)")
    print("=" * 60)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = QdrantAgentInterpreter(llm=llm)

    queries = [
        (
            "Recommend parameters for a collection that will store CLIP image embeddings "
            "for a visual product search engine. We expect 2 million images and need the "
            "lowest possible query latency. Do not create the collection yet.",
        ),
        (
            "Now create the collection called 'image_search' with those parameters.",
        ),
        (
            "Add payload indexes for 'brand' (keyword), 'price' (float), and "
            "'release_year' (integer) on the image_search collection.",
        ),
        (
            "List all collections and describe image_search.",
        ),
    ]

    for (query,) in queries:
        print(f"\n{'─' * 50}")
        print(f"Query: {query}")
        print("─" * 50)
        response = agent.run(query)
        print(response)


# ---------------------------------------------------------------------------
# Demo 4: eval_interpreter escape hatch
# ---------------------------------------------------------------------------

def demo_eval_interpreter() -> None:
    """Show the exec-based interpreter used for multi-step custom code."""
    print("\n" + "=" * 60)
    print("DEMO 4: eval_interpreter (exec-based, no LLM)")
    print("=" * 60)

    from qdrant_interpreter_plugin import QdrantInterpreter

    interp = QdrantInterpreter()

    # Step 1: create collection — all Qdrant types pre-imported in globals
    result = interp.execute("""
client.create_collection(
    collection_name="demo",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
    ),
)
print("Collection created with scalar quantization")
""")
    print(result["stdout"].strip())
    assert result["success"], result["stderr"]

    # Step 2: upsert points (using PointStruct available in globals)
    result = interp.execute("""
import random
random.seed(42)
points = [
    PointStruct(id=i, vector=[random.random() for _ in range(4)], payload={"label": f"item_{i}"})
    for i in range(20)
]
client.upsert("demo", points=points)
print(f"Upserted {len(points)} points")
""")
    print(result["stdout"].strip())
    assert result["success"], result["stderr"]

    # Step 3: search (Filter and FieldCondition available in globals)
    result = interp.execute("""
hits = client.search("demo", query_vector=[0.1, 0.2, 0.3, 0.4], limit=3, with_payload=True)
for h in hits:
    print(f"  id={h.id}  score={h.score:.4f}  label={h.payload['label']}")
""")
    print(result["stdout"].strip())
    assert result["success"], result["stderr"]

    # Persistent state: variable from step 2 is gone (new interp per demo)
    # but results_cache persists within the same interpreter instance
    result = interp.execute("""
results_cache["last_search"] = "done"
print("Stored in results_cache:", results_cache["last_search"])
""")
    print(result["stdout"].strip())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_param_advisor()
    demo_index_manager()
    demo_eval_interpreter()
    demo_agent()  # Skipped gracefully if OPENAI_API_KEY is not set


if __name__ == "__main__":
    main()
