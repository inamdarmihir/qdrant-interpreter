"""
Qdrant Agent Interpreter
=========================

A Deep Agent that manages Qdrant collections through a **sandboxed QuickJS
interpreter** — the exact pattern described in the LangChain blog post
*"Give Your Agents an Interpreter"*.

Architecture
------------

.. code-block:: text

    QdrantAgentInterpreter
    ├── create_deep_agent (deepagents)
    │   ├── model          — any LangChain BaseChatModel or model-string
    │   ├── tools          — 10 Qdrant management tools (direct calls)
    │   └── middleware
    │       └── CodeInterpreterMiddleware (langchain_quickjs)
    │           ├── eval tool — the agent writes JavaScript here
    │           ├── QuickJS sandbox per LangGraph thread
    │           ├── console.log capture
    │           └── PTC bridge — tools.createQdrantCollection(...) etc.
    └── QdrantIndexManager — param advisor + Qdrant client

How the agent works
--------------------
1. User sends a natural-language query.
2. The model reasons about which Qdrant operations are needed.
3. Instead of calling tools one-by-one (one model round-trip each), the
   model writes a short JavaScript program and calls the ``eval`` tool once.
4. Inside the QuickJS sandbox, JS calls like
   ``await tools.createQdrantCollection({...})`` cross the PTC bridge and
   invoke the real Python Qdrant tools.
5. Intermediate values stay in the JS runtime; only the final expression
   (and any ``console.log`` output) is returned to the model.
6. The model summarises the result and yields to the user.

This pattern uses up to 35% fewer tokens than serial tool-calling on
equivalent tasks (LangChain blog, May 2026).

Quick start::

    from langchain_anthropic import ChatAnthropic
    from qdrant_interpreter_plugin import QdrantAgentInterpreter

    agent = QdrantAgentInterpreter(
        model=ChatAnthropic(model_name="claude-sonnet-4-6"),
    )
    print(agent.run(
        "Create a collection for semantic search over product descriptions "
        "using OpenAI ada-002 embeddings. Expect 500k products. "
        "We need to filter by category and price."
    ))

Remote Qdrant::

    agent = QdrantAgentInterpreter(
        model="openai:gpt-4o",
        location="https://xyz.qdrant.tech",
        api_key="your-key",
    )
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from qdrant_client import QdrantClient

from .index_manager import QdrantIndexManager
from .tools import make_qdrant_tools


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert Qdrant vector-database agent. You create and manage Qdrant \
collections using a sandboxed JavaScript interpreter as your primary working \
environment.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 INTERPRETER TOOL  ("eval")
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The `eval` tool runs JavaScript inside a QuickJS sandbox. Key properties:

• State (variables, functions) persists across eval calls within a session.
• The value of the LAST expression is returned to you as the result.
• `console.log()` output appears in <stdout> blocks.
• No filesystem, network, or shell — only the bridged Qdrant tools below.
• Use `await` before every `tools.*` call — they are async.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 QDRANT TOOLS  (globalThis.tools.*)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All tool arguments are plain JavaScript objects.

  await tools.createQdrantCollection({
      collection_name,           // string — required
      use_case,                  // string — describe the workload in detail
      vector_size,               // number — inferred from use_case if omitted
      expected_count,            // number — default 100000
      priority,                  // "recall" | "latency" | "memory"
      sparse_vectors,            // boolean — true for hybrid search
      memory_gb_available,       // number — drives quantization aggression
  })

  await tools.ensureQdrantCollection({ collection_name, use_case, vector_size })
  await tools.listQdrantCollections()
  await tools.describeQdrantCollection({ collection_name })
  await tools.deleteQdrantCollection({ collection_name })

  await tools.upsertQdrantPoints({
      collection_name,
      points,   // JSON string: [{id, vector, payload?}, ...]
  })

  await tools.searchQdrantCollection({
      collection_name,
      query_vector,        // JSON string of float array
      limit,               // default 5
      score_threshold,     // optional float
      filter_conditions,   // optional JSON string
  })

  await tools.createQdrantPayloadIndex({
      collection_name,
      field_name,
      field_type,   // "keyword"|"integer"|"float"|"geo"|"text"|"bool"|"datetime"
  })

  await tools.recommendQdrantParams({
      use_case, vector_size, expected_count, priority
  })

  await tools.optimizeQdrantCollection({ collection_name, priority })

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Prefer ONE eval call that batches related operations with Promise.all().
• Keep intermediate data in JS variables — return only the final summary.
• Always call ensureQdrantCollection before assuming a collection exists.
• Proactively create payload indexes for fields the user intends to filter on.
• After creating a collection, explain the chosen parameters in plain English.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXAMPLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```javascript
// Batch create + index in one eval call
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

const health = await tools.describeQdrantCollection({ collection_name: "products" });
health   // ← last expression returned to you
```
"""


# ---------------------------------------------------------------------------
# QdrantAgentInterpreter
# ---------------------------------------------------------------------------

class QdrantAgentInterpreter:
    """Deep agent that manages Qdrant collections via a QuickJS interpreter.

    Combines :func:`deepagents.create_deep_agent` with
    :class:`langchain_quickjs.CodeInterpreterMiddleware` so the model writes
    JavaScript to batch Qdrant operations — instead of making one tool call
    per model round-trip.

    Parameters
    ----------
    model:
        LangChain ``BaseChatModel`` *or* a model-string like
        ``"openai:gpt-4o"`` / ``"anthropic:claude-sonnet-4-6"``.
    location:
        Qdrant backend. ``":memory:"`` (default) for in-process use.
        Pass a URL for a remote instance.
    api_key:
        Optional API key for Qdrant Cloud.
    system_prompt:
        Override the built-in system prompt. ``None`` uses the default.
    timeout:
        Per-eval wall-clock timeout in seconds (default 10).
    memory_limit_mb:
        QuickJS heap cap in megabytes (default 64).
    max_ptc_calls:
        Maximum ``tools.*`` bridge calls per single eval (default 256).
    max_result_chars:
        Truncation limit for result and stdout blocks (default 4000).
    capture_console:
        Whether to collect ``console.log`` output (default ``True``).
    ptc:
        Allowlist of Qdrant tool *names* to expose inside the interpreter
        via PTC. ``None`` exposes all 10 tools.
    mode:
        REPL state persistence: ``"thread"`` (across turns, default),
        ``"turn"`` (within a turn only), or ``"call"`` (fresh each eval).

    Attributes
    ----------
    client:
        The shared :class:`qdrant_client.QdrantClient`.
    manager:
        The :class:`~qdrant_interpreter_plugin.index_manager.QdrantIndexManager`.
    tools:
        All LangChain tools registered with the agent (Qdrant tools +
        the QuickJS ``eval`` tool from the middleware).
    """

    def __init__(
        self,
        model: str | Any,
        location: str = ":memory:",
        api_key: str | None = None,
        system_prompt: str | None = None,
        timeout: float = 10.0,
        memory_limit_mb: int = 64,
        max_ptc_calls: int | None = 256,
        max_result_chars: int = 4_000,
        capture_console: bool = True,
        ptc: list[str] | None = None,
        mode: Literal["thread", "turn", "call"] = "thread",
    ) -> None:
        from deepagents import create_deep_agent
        from langchain_quickjs import CodeInterpreterMiddleware

        # Shared Qdrant client
        client_kwargs: dict[str, Any] = {"location": location}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = QdrantClient(**client_kwargs)
        self._manager = QdrantIndexManager(self._client)

        # Build Qdrant tools
        self._qdrant_tools: list[BaseTool] = make_qdrant_tools(self._manager)

        # PTC allowlist — which tools are callable from inside the JS sandbox
        if ptc is None:
            ptc_tools: list[BaseTool] = self._qdrant_tools
        else:
            ptc_set = set(ptc)
            ptc_tools = [t for t in self._qdrant_tools if t.name in ptc_set]

        # Middleware — adds the `eval` tool and manages the QuickJS runtime
        middleware = CodeInterpreterMiddleware(
            ptc=ptc_tools,
            timeout=timeout,
            memory_limit=memory_limit_mb * 1024 * 1024,
            max_ptc_calls=max_ptc_calls,
            max_result_chars=max_result_chars,
            capture_console=capture_console,
            mode=mode,
        )

        # Assemble the deep agent
        self._agent = create_deep_agent(
            model=model,
            tools=self._qdrant_tools,
            middleware=[middleware],
            system_prompt=system_prompt or _SYSTEM_PROMPT,
        )

        # Expose middleware tools (includes the `eval` tool)
        self._all_tools: list[BaseTool] = self._qdrant_tools + middleware.tools

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        thread_id: str = "default",
        **invoke_kwargs: Any,
    ) -> str:
        """Run the agent with a natural-language query.

        Parameters
        ----------
        query:
            User instruction, e.g.
            ``"Create a collection for image similarity search with CLIP"``.
        thread_id:
            LangGraph thread identifier. Reusing the same ``thread_id``
            continues the same REPL session (state persists).
        **invoke_kwargs:
            Extra kwargs forwarded to ``agent.invoke()``.

        Returns
        -------
        str
            The agent's final response.
        """
        config = {"configurable": {"thread_id": thread_id}}
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            **invoke_kwargs,
        )
        last = result["messages"][-1]
        content = last.content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def stream(
        self,
        query: str,
        thread_id: str = "default",
        **stream_kwargs: Any,
    ):
        """Stream the agent's response.

        Yields LangGraph stream events. Useful for displaying progress in
        interactive applications.

        Parameters
        ----------
        query:
            User instruction.
        thread_id:
            LangGraph thread identifier.

        Yields
        ------
        dict
            LangGraph stream events.
        """
        config = {"configurable": {"thread_id": thread_id}}
        yield from self._agent.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            **stream_kwargs,
        )

    @property
    def client(self) -> QdrantClient:
        """The shared Qdrant client used by all tools."""
        return self._client

    @property
    def manager(self) -> QdrantIndexManager:
        """The index manager for direct programmatic access."""
        return self._manager

    @property
    def tools(self) -> list[BaseTool]:
        """All registered tools (Qdrant tools + QuickJS eval tool)."""
        return self._all_tools
