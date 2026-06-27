"""
Qdrant Agent Interpreter
=========================

A LangChain / LangGraph agent that combines:

* **10 structured Qdrant management tools** — create, upsert, search, index,
  optimise, and inspect collections without writing a single line of Python.
* **Auto-parameter selection** — the agent picks the best distance metric,
  HNSW settings, and quantization for any use case, then explains why.
* **``eval_interpreter`` escape hatch** — for complex multi-step operations
  the structured tools cannot express, the agent can fall back to executing
  arbitrary Python code against the same Qdrant client.

The agent is built on :func:`langgraph.prebuilt.create_react_agent`, which
implements the ReAct (Reason + Act) loop: it alternates between reasoning
steps and tool calls until it produces a final answer.

Quick start::

    from langchain_openai import ChatOpenAI
    from qdrant_interpreter_plugin import QdrantAgentInterpreter

    agent = QdrantAgentInterpreter(llm=ChatOpenAI(model="gpt-4o-mini"))
    response = agent.run(
        "Create a collection for semantic search over product descriptions "
        "using OpenAI ada-002 embeddings. Expect around 500k products. "
        "We will filter by category and price range."
    )
    print(response)

Connecting to a remote Qdrant server::

    agent = QdrantAgentInterpreter(
        llm=ChatOpenAI(model="gpt-4o"),
        location="https://my-cluster.qdrant.tech",
        api_key="my-api-key",
    )

Direct access to the underlying components::

    # The QdrantClient shared by all tools
    agent.client.get_collections()

    # The QdrantIndexManager for programmatic use
    agent.manager.health("my_collection")

    # The list of registered tools
    [t.name for t in agent.tools]
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient

from .index_manager import QdrantIndexManager
from .qdrant_interpreter import QdrantInterpreter, make_eval_interpreter
from .tools import make_qdrant_tools


_DEFAULT_SYSTEM_PROMPT = """\
You are an expert Qdrant vector-database agent. You help users create, manage,
and query Qdrant collections. You have access to the following tools:

COLLECTION MANAGEMENT
  create_qdrant_collection   — Create a collection with auto-selected parameters.
  ensure_qdrant_collection   — Create only if the collection doesn't exist yet.
  delete_qdrant_collection   — Permanently delete a collection.
  list_qdrant_collections    — List all existing collections.
  describe_qdrant_collection — Show stats and recommendations for a collection.
  optimize_qdrant_collection — Tune optimizer settings on an existing collection.

DATA OPS
  upsert_qdrant_points       — Insert or update vectors (JSON list of points).
  search_qdrant_collection   — ANN search with optional payload filters.

INDEXING
  create_qdrant_payload_index — Index a payload field for fast filtered search.

PLANNING / DRY-RUN
  recommend_qdrant_params    — Preview parameter recommendations without creating.

ESCAPE HATCH
  eval_interpreter           — Execute arbitrary Python against the Qdrant client.
                               Use this for complex multi-step operations.

GUIDELINES
- When creating a collection, always call create_qdrant_collection (not the \
eval_interpreter). Explain the chosen parameters in plain English.
- When the user mentions filtering by metadata fields (category, date, user_id, \
etc.), proactively create payload indexes on those fields after the collection.
- For large collections (> 100k vectors), consider recommending scalar INT8 \
quantization unless the user has already specified quantization preferences.
- When using eval_interpreter, use the globally available `client` (QdrantClient), \
`VectorParams`, `Distance`, `PointStruct`, `Filter`, `FieldCondition`, \
`MatchValue`, and `results_cache` objects.
- Always confirm successful operations and summarise what was done.
"""


class QdrantAgentInterpreter:
    """A LangChain agent with intelligent Qdrant index management.

    Wraps a LangGraph ReAct agent that has 10 structured Qdrant tools plus
    an ``eval_interpreter`` escape hatch. The agent uses
    :class:`~qdrant_interpreter_plugin.param_advisor.UseCaseParamAdvisor`
    to automatically select the best collection parameters for any use case.

    Parameters
    ----------
    llm:
        Any LangChain ``BaseChatModel`` that supports tool-calling
        (e.g. ``ChatOpenAI``, ``ChatAnthropic``, ``ChatGoogleGenerativeAI``).
    location:
        Qdrant backend. Default ``":memory:"`` for in-process use.
        Pass a URL (``"http://localhost:6333"`` or
        ``"https://xyz.qdrant.tech"``) for a remote instance.
    api_key:
        Optional API key for remote Qdrant Cloud instances.
    system_prompt:
        Override the default system prompt. ``None`` uses the built-in prompt.
    interpreter_timeout:
        Maximum seconds allowed per ``eval_interpreter`` call. Default 30.

    Attributes
    ----------
    client:
        The underlying :class:`qdrant_client.QdrantClient` shared across all
        tools and the ``eval_interpreter``.
    manager:
        The :class:`~qdrant_interpreter_plugin.index_manager.QdrantIndexManager`
        for direct programmatic access.
    tools:
        The list of LangChain tools registered with the agent.
    """

    def __init__(
        self,
        llm,
        location: str = ":memory:",
        api_key: str | None = None,
        system_prompt: str | None = None,
        interpreter_timeout: float = 30.0,
    ) -> None:
        # Shared Qdrant client
        client_kwargs: dict[str, Any] = {"location": location}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = QdrantClient(**client_kwargs)

        # Structured collection manager
        self._manager = QdrantIndexManager(self._client)

        # Exec-based interpreter — shares the same QdrantClient
        self._interpreter = QdrantInterpreter(timeout=interpreter_timeout)
        self._interpreter.client = self._client  # keep attribute for direct access
        self._interpreter.globals["client"] = self._client

        # Assemble tool list: 10 structured tools + eval_interpreter
        self._tools = make_qdrant_tools(self._manager)
        self._tools.append(make_eval_interpreter(self._interpreter))

        # System prompt
        prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

        # LangGraph ReAct agent
        from langgraph.prebuilt import create_react_agent

        self._agent = create_react_agent(
            model=llm,
            tools=self._tools,
            prompt=prompt,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str) -> str:
        """Run the agent with a natural-language query.

        Parameters
        ----------
        query:
            User instruction, e.g.
            ``"Create a collection for image similarity search with CLIP embeddings"``.

        Returns
        -------
        str
            The agent's final response after all tool calls have completed.
        """
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        last_msg = result["messages"][-1]
        content = last_msg.content
        # Handle list-form content (e.g. OpenAI structured output blocks)
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def stream(self, query: str):
        """Stream the agent's response token-by-token.

        Yields :class:`langchain_core.messages.BaseMessage` objects from the
        LangGraph streaming interface. Useful for displaying progress in
        interactive applications.

        Parameters
        ----------
        query:
            User instruction.

        Yields
        ------
        dict
            LangGraph stream events with ``"messages"`` or ``"updates"`` keys.
        """
        yield from self._agent.stream(
            {"messages": [HumanMessage(content=query)]}
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
    def tools(self) -> list:
        """All tools registered with the agent (structured + eval_interpreter)."""
        return self._tools
