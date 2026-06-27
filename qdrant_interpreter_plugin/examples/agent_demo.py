"""
Qdrant Agent Interpreter — Demo
================================

Demonstrates the interpreter pattern from the LangChain blog post
"Give Your Agents an Interpreter" applied to Qdrant vector database management.

The interpreter pattern
-----------------------
Instead of calling tools one-at-a-time (one model round-trip each), the
agent writes a short JavaScript program and calls the ``eval`` tool ONCE.
Inside the QuickJS sandbox, ``await tools.createQdrantCollection({...})``
crosses a PTC bridge and invokes the real Python Qdrant tool — no round-trip.
Multiple operations can be batched with Promise.all().

Prerequisites
-------------
Install full dependencies::

    pip install qdrant-client quickjs-rs deepagents[quickjs] langchain-quickjs \
                langchain-anthropic pydantic

Run this demo::

    ANTHROPIC_API_KEY=sk-ant-... python agent_demo.py

Or to only run the no-LLM demos (params advisor + standalone interpreter)::

    python agent_demo.py --no-llm
"""

from __future__ import annotations

import sys


# ──────────────────────────────────────────────────────────────────────────────
# DEMO 1: UseCaseParamAdvisor — no LLM, no sandbox
# ──────────────────────────────────────────────────────────────────────────────

def demo_param_advisor() -> None:
    """Rule-based parameter selection, zero dependencies beyond qdrant-client."""
    print("\n" + "=" * 60)
    print("DEMO 1: UseCaseParamAdvisor (rule-based, no LLM)")
    print("=" * 60)

    from qdrant_interpreter_plugin import UseCaseParamAdvisor

    advisor = UseCaseParamAdvisor()

    cases = [
        ("Semantic search over product descriptions using OpenAI ada-002", 500_000, "recall"),
        ("Anomaly detection on time-series sensor data with Euclidean distance", 10_000, "latency"),
        ("Collaborative filtering with matrix factorization embeddings", 2_000_000, "memory"),
    ]

    for use_case, count, priority in cases:
        advice = advisor.advise(use_case=use_case, expected_count=count, priority=priority)
        print(f"\n  use_case : {use_case[:55]}...")
        print(f"  distance : {advice.distance.value}")
        print(f"  size     : {advice.vector_size}")
        print(f"  HNSW m/ef: {advice.hnsw_config.m} / {advice.hnsw_config.ef_construct}")
        print(f"  quant    : {advice.quantization_mode}")
        for k, v in advice.rationale.items():
            print(f"    [{k}] {v}")


# ──────────────────────────────────────────────────────────────────────────────
# DEMO 2: QdrantInterpreter — QuickJS sandbox, no LLM
# ──────────────────────────────────────────────────────────────────────────────

def demo_quickjs_interpreter() -> None:
    """
    The core of the blog's pattern: a sandboxed JS runtime with Qdrant tools
    bridged as async callables. No LLM required — you drive the sandbox
    directly.
    """
    print("\n" + "=" * 60)
    print("DEMO 2: QdrantInterpreter — QuickJS sandbox (no LLM)")
    print("=" * 60)

    from qdrant_client import QdrantClient
    from qdrant_interpreter_plugin import QdrantIndexManager, QdrantInterpreter

    manager = QdrantIndexManager(QdrantClient(":memory:"))
    interp = QdrantInterpreter(manager, timeout=15.0, capture_console=True)

    # ── Step 1: create collection — param advisor runs in Python behind the bridge
    r = interp.eval("""
const col = await tools.createQdrantCollection({
    collection_name: "products",
    use_case: "semantic search over product descriptions using OpenAI ada-002",
    expected_count: 500000,
    priority: "recall",
});
console.log("Collection created");
col
""")
    assert r["success"], r["error"]
    print("\n[Step 1] Create collection")
    print("  stdout :", r["stdout"])
    print("  result :", r["result"][:200])

    # ── Step 2: batch payload indexes with Promise.all (single eval call)
    r = interp.eval("""
const [catIdx, priceIdx, stockIdx] = await Promise.all([
    tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "category",  field_type: "keyword" }),
    tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "price",     field_type: "float" }),
    tools.createQdrantPayloadIndex({ collection_name: "products", field_name: "in_stock",  field_type: "bool" }),
]);
console.log("3 indexes created in one eval call");
`${catIdx} | ${priceIdx} | ${stockIdx}`
""")
    assert r["success"], r["error"]
    print("\n[Step 2] Batch payload indexes via Promise.all")
    print("  stdout:", r["stdout"])

    # ── Step 3: persistent REPL state — variable from step 1 still accessible
    r = interp.eval("""
console.log("col still in scope:", typeof col);
const health = await tools.describeQdrantCollection({ collection_name: "products" });
health
""")
    assert r["success"], r["error"]
    print("\n[Step 3] Persistent state + health check")
    print("  stdout:", r["stdout"])
    print("  health:", r["result"][:300])

    # ── Step 4: recommend params (dry-run, no collection created)
    r = interp.eval("""
const rec = await tools.recommendQdrantParams({
    use_case: "anomaly detection on genomic sequences with Euclidean distance",
    expected_count: 1000000,
    priority: "latency",
});
rec
""")
    assert r["success"], r["error"]
    print("\n[Step 4] Dry-run param recommendations")
    print("  result:", r["result"][:400])

    # ── Snapshotting — capture REPL state for persistence
    snap = interp.snapshot()
    print(f"\n[Snapshot] Captured {len(snap):,} bytes of JS heap state")

    interp.close()
    print("\n  Interpreter closed cleanly.")


# ──────────────────────────────────────────────────────────────────────────────
# DEMO 3: QdrantAgentInterpreter — full agent with QuickJS + CodeInterpreterMiddleware
# ──────────────────────────────────────────────────────────────────────────────

def demo_agent() -> None:
    """Full agent: model writes JS → QuickJS sandbox → PTC bridge → Qdrant."""
    import os

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[SKIP] DEMO 3: set ANTHROPIC_API_KEY or OPENAI_API_KEY to run the full agent.")
        return

    print("\n" + "=" * 60)
    print("DEMO 3: QdrantAgentInterpreter (LLM + QuickJS + PTC)")
    print("=" * 60)

    from qdrant_interpreter_plugin import QdrantAgentInterpreter

    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model_name="claude-sonnet-4-6")
    else:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = QdrantAgentInterpreter(
        model=model,
        timeout=15.0,
        memory_limit_mb=64,
        max_ptc_calls=32,
    )

    queries = [
        # Recommend without creating — dry-run eval call
        "Recommend parameters for a collection that stores CLIP-ViT-L/14 image "
        "embeddings for a visual search engine. Expect 2 million images, lowest "
        "possible query latency. Do not create anything yet.",

        # Batch create + indexes in one eval call
        "Now create the collection called 'image_search' and also create payload "
        "indexes for 'brand' (keyword) and 'release_year' (integer).",

        # Health check
        "Describe image_search and tell me if there are any recommendations.",
    ]

    for query in queries:
        print(f"\n{'─' * 50}")
        print(f"Query: {query[:80]}...")
        print("─" * 50)
        response = agent.run(query, thread_id="demo-session")
        print(response)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    no_llm = "--no-llm" in sys.argv
    demo_param_advisor()
    demo_quickjs_interpreter()
    if not no_llm:
        demo_agent()
    print("\nDone.")


if __name__ == "__main__":
    main()
