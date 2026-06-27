"""Top-level package for the Qdrant interpreter plugin.

This package implements the interpreter pattern from the LangChain blog post
*"Give Your Agents an Interpreter"* applied to Qdrant vector database management.

Public API
----------

**Full agent (requires deepagents + langchain-quickjs + an LLM)**::

    from langchain_anthropic import ChatAnthropic
    from qdrant_interpreter_plugin import QdrantAgentInterpreter

    agent = QdrantAgentInterpreter(
        model=ChatAnthropic(model_name="claude-sonnet-4-6"),
    )
    print(agent.run(
        "Create a collection for semantic search over product descriptions "
        "using OpenAI ada-002, ~500k products, filter by category and price."
    ))

**Standalone QuickJS interpreter (no LLM)**::

    from qdrant_client import QdrantClient
    from qdrant_interpreter_plugin import QdrantIndexManager, QdrantInterpreter

    manager = QdrantIndexManager(QdrantClient(":memory:"))
    interp = QdrantInterpreter(manager)

    result = interp.eval('''
        const col = await tools.createQdrantCollection({
            collection_name: "docs",
            use_case: "semantic search using bert-base",
        });
        console.log(col);
        col
    ''')
    print(result["result"])
    print(result["stdout"])

**Direct index manager (no LLM, no sandbox)**::

    from qdrant_client import QdrantClient
    from qdrant_interpreter_plugin import QdrantIndexManager

    manager = QdrantIndexManager(QdrantClient(":memory:"))
    manager.auto_create("products", use_case="semantic search with OpenAI ada-002")

**Parameter advisor (rule-based, no network)**::

    from qdrant_interpreter_plugin import UseCaseParamAdvisor

    advice = UseCaseParamAdvisor().advise(
        "semantic search using OpenAI ada-002", expected_count=500_000
    )
    print(advice.distance, advice.quantization_mode)
"""

from .index_manager import QdrantIndexManager  # noqa: F401
from .param_advisor import UseCaseParamAdvisor  # noqa: F401

__all__ = [
    "QdrantIndexManager",
    "UseCaseParamAdvisor",
]

# ---------------------------------------------------------------------------
# QuickJS interpreter (requires quickjs-rs)
# ---------------------------------------------------------------------------
_has_quickjs = False
try:
    from .qdrant_interpreter import QdrantInterpreter  # type: ignore  # noqa: F401
    _has_quickjs = True
except Exception:
    pass

if _has_quickjs:
    __all__ += ["QdrantInterpreter"]

# ---------------------------------------------------------------------------
# LangChain tool integration (requires langchain-core + pydantic + quickjs-rs)
# ---------------------------------------------------------------------------
_has_tool = False
try:
    from .qdrant_interpreter import make_eval_interpreter  # type: ignore  # noqa: F401
    _has_tool = True
except Exception:
    pass

if _has_tool:
    __all__ += ["make_eval_interpreter"]

# ---------------------------------------------------------------------------
# Structured Qdrant tools factory (requires langchain-core + pydantic)
# ---------------------------------------------------------------------------
_has_tools = False
try:
    from .tools import make_qdrant_tools  # type: ignore  # noqa: F401
    _has_tools = True
except Exception:
    pass

if _has_tools:
    __all__ += ["make_qdrant_tools"]

# ---------------------------------------------------------------------------
# Full agent (requires deepagents + langchain-quickjs)
# ---------------------------------------------------------------------------
_has_agent = False
try:
    from .agent import QdrantAgentInterpreter  # type: ignore  # noqa: F401
    _has_agent = True
except Exception:
    pass

if _has_agent:
    __all__ += ["QdrantAgentInterpreter"]
