"""Top-level package for the Qdrant interpreter plugin.

This package exposes:

* :class:`QdrantInterpreter` — stateful exec-based code runner against an
  in-memory (or remote) Qdrant client.
* :class:`QdrantIndexManager` — smart collection management with automatic
  parameter selection via :class:`UseCaseParamAdvisor`.
* :class:`QdrantAgentInterpreter` — a full LangChain / LangGraph agent that
  combines 10 structured Qdrant tools with the ``eval_interpreter`` escape
  hatch and auto-selects the best collection parameters for any use case.
* :class:`UseCaseParamAdvisor` — rule-based advisor that maps use-case
  descriptions to optimal Qdrant collection parameters.
* :func:`make_qdrant_tools` — factory that produces the 10 structured tools
  bound to a given :class:`QdrantIndexManager`.
* :func:`make_eval_interpreter` / ``eval_interpreter`` — LangChain tool
  wrapping the exec-based interpreter (requires ``langchain-core``).

Typical usage (full agent)::

    from langchain_openai import ChatOpenAI
    from qdrant_interpreter_plugin import QdrantAgentInterpreter

    agent = QdrantAgentInterpreter(llm=ChatOpenAI(model="gpt-4o-mini"))
    print(agent.run("Create a collection for semantic search over product docs"))

Low-level interpreter only::

    from qdrant_interpreter_plugin import QdrantInterpreter

    interp = QdrantInterpreter()
    result = interp.execute("print(client.get_collections())")
    print(result['stdout'])

Direct index manager (no LLM)::

    from qdrant_client import QdrantClient
    from qdrant_interpreter_plugin import QdrantIndexManager

    manager = QdrantIndexManager(QdrantClient(":memory:"))
    manager.auto_create("products", use_case="semantic search via OpenAI ada-002")
"""

from .qdrant_interpreter import QdrantInterpreter  # noqa: F401
from .index_manager import QdrantIndexManager  # noqa: F401
from .param_advisor import UseCaseParamAdvisor  # noqa: F401

__all__ = [
    "QdrantInterpreter",
    "QdrantIndexManager",
    "UseCaseParamAdvisor",
]

# -----------------------------------------------------------------------
# Optional: LangChain tool integration (requires langchain-core + pydantic)
# -----------------------------------------------------------------------
_has_langchain = False
try:
    from .qdrant_interpreter import (  # type: ignore  # noqa: F401
        eval_interpreter as eval_interpreter,
        make_eval_interpreter as make_eval_interpreter,
    )
    _has_langchain = True
except Exception:
    pass

if _has_langchain:
    __all__ += ["eval_interpreter", "make_eval_interpreter"]

# -----------------------------------------------------------------------
# Optional: structured tools factory (requires langchain-core + pydantic)
# -----------------------------------------------------------------------
_has_tools = False
try:
    from .tools import make_qdrant_tools  # type: ignore  # noqa: F401
    _has_tools = True
except Exception:
    pass

if _has_tools:
    __all__ += ["make_qdrant_tools"]

# -----------------------------------------------------------------------
# Optional: full agent (requires langchain-core + pydantic + langgraph)
# -----------------------------------------------------------------------
_has_agent = False
try:
    from .agent import QdrantAgentInterpreter  # type: ignore  # noqa: F401
    _has_agent = True
except Exception:
    pass

if _has_agent:
    __all__ += ["QdrantAgentInterpreter"]
