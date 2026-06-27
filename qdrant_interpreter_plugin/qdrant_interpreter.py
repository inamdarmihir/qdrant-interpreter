"""
Qdrant Interpreter for LangChain / LangGraph Agents
===================================================

This module implements a simple stateful code interpreter that can be embedded
inside agent frameworks such as LangChain and LangGraph. The goal of the
interpreter is to give an LLM-based agent a controlled environment where it
can write and execute Python code against a Qdrant vector database client.

### Features

* **Persistent state** – Each instance of ``QdrantInterpreter`` maintains its
  own set of global variables. Code executed in one call can define
  functions or store variables that will be available on subsequent calls.
* **Allowlisted API** – Only a limited set of objects are exposed to the
  executed code. In this example those objects include:
  * ``client`` – An in-memory ``qdrant_client.QdrantClient`` instance.
  * ``VectorParams``, ``Distance`` and ``PointStruct`` – type helpers for
    creating collections and inserting points.
  * ``results_cache`` – A simple dictionary for agents to store
    intermediate state without printing it to the conversation.
* **Structured output** – The result of each execution returns a dict with
  keys ``success``, ``stdout``, ``stderr`` and ``stored_vars``. This makes
  it easy for calling code to surface output or detect errors.

The interpreter itself is just a Python wrapper around the ``exec`` builtin
function. It captures standard output and exceptions so that agent systems
receive clear feedback without raising unhandled errors in the host process.

Example::

    from qdrant_interpreter_plugin import QdrantInterpreter

    interp = QdrantInterpreter()
    code = '''
    client.create_collection(
        collection_name="my_index",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    '''
    result = interp.execute(code)
    print(result["stdout"])
    assert result["success"]

You can also integrate this interpreter as a LangChain tool, see
``eval_interpreter`` below.

"""

from __future__ import annotations

import io
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    BinaryQuantization,
    BinaryQuantizationConfig,
    CompressionRatio,
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    ProductQuantization,
    ProductQuantizationConfig,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


class QdrantInterpreter:
    """A stateful interpreter for running Python against Qdrant.

    Each instance exposes a rich global scope containing a Qdrant client
    and the full set of commonly used ``qdrant_client.models`` types so that
    agent-generated code rarely needs to import anything explicitly.
    """

    def __init__(self, location: str = ":memory:", timeout: float = 30.0) -> None:
        """Initialize the interpreter.

        Parameters
        ----------
        location : str, optional
            Location for the underlying Qdrant client. By default this uses
            in-memory storage. See Qdrant documentation for other
            possibilities, e.g. a path on disk or a remote URL.
        timeout : float, optional
            Maximum time in seconds for code execution. Defaults to 30.0.
            Set to None to disable timeout (not recommended for production).
        """
        self.client = QdrantClient(location=location)
        self.timeout = timeout
        self._lock = threading.Lock()
        # Expose the full allowlisted API to executed code.
        # All model types are included so agents rarely need explicit imports.
        self.globals: Dict[str, Any] = {
            # Client
            "client": self.client,
            # Core vector / collection types
            "VectorParams": VectorParams,
            "Distance": Distance,
            "PointStruct": PointStruct,
            "SparseVector": SparseVector,
            "SparseVectorParams": SparseVectorParams,
            "SparseIndexParams": SparseIndexParams,
            # Quantization
            "ScalarQuantization": ScalarQuantization,
            "ScalarQuantizationConfig": ScalarQuantizationConfig,
            "ScalarType": ScalarType,
            "ProductQuantization": ProductQuantization,
            "ProductQuantizationConfig": ProductQuantizationConfig,
            "CompressionRatio": CompressionRatio,
            "BinaryQuantization": BinaryQuantization,
            "BinaryQuantizationConfig": BinaryQuantizationConfig,
            # Index / optimizer config
            "HnswConfigDiff": HnswConfigDiff,
            "OptimizersConfigDiff": OptimizersConfigDiff,
            # Filtering
            "Filter": Filter,
            "FieldCondition": FieldCondition,
            "MatchValue": MatchValue,
            "MatchAny": MatchAny,
            "Range": Range,
            # Payload
            "PayloadSchemaType": PayloadSchemaType,
            "PointIdsList": PointIdsList,
            # Agent state
            "results_cache": {},
        }

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute a block of Python code within the interpreter context.

        The code is executed with ``exec`` in the context of the interpreter's
        stored globals. Standard output is captured and returned along with
        any errors that occur. Execution is thread-safe and time-limited.

        Parameters
        ----------
        code : str
            Python source code to execute.

        Returns
        -------
        Dict[str, Any]
            A dict with keys:
              - ``success``: boolean indicating if execution succeeded.
              - ``stdout``: captured standard output from the execution.
              - ``stderr``: traceback string if an exception was raised, else "".
              - ``stored_vars``: a short preview of the interpreter's globals.
        """
        error: str | None = None
        output: str = ""

        def _execute_code() -> None:
            nonlocal error, output
            # Redirect stdout so we can capture printed output
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output
            try:
                exec(code, self.globals)
            except Exception:
                error = traceback.format_exc()
            finally:
                # Restore stdout regardless of success or failure
                sys.stdout = old_stdout
                output = redirected_output.getvalue()

        # Execute code with thread safety and timeout
        with self._lock:
            if self.timeout is None:
                _execute_code()
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_execute_code)
                    try:
                        future.result(timeout=self.timeout)
                    except FuturesTimeoutError:
                        error = f"Code execution timed out after {self.timeout} seconds."

        # Only expose a stringified preview of stored variables to avoid
        # returning large data structures directly. Trim values to 100 chars.
        preview: Dict[str, str] = {}
        for k, v in self.globals.items():
            if k.startswith("__"):
                continue
            try:
                preview[k] = str(v)[:100]
            except Exception:
                preview[k] = "<unprintable>"

        return {
            "success": error is None,
            "stdout": output,
            "stderr": error or "",
            "stored_vars": preview,
        }


# Optional LangChain tool integration
try:
    from langchain_core.tools import tool  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore

    class EvalInput(BaseModel):
        """Schema for code execution input.

        Agents using this tool should populate the ``code`` field with the
        Python script they intend to run.
        """

        code: str = Field(
            description=(
                "Python code to execute inside the Qdrant interpreter.\n"
                "Use the `client` object to interact with Qdrant and print"
                " only the final result you want to surface."
            )
        )

    def make_eval_interpreter(interpreter: QdrantInterpreter | None = None):
        """Create a LangChain tool wrapping a QdrantInterpreter instance.

        Parameters
        ----------
        interpreter : QdrantInterpreter, optional
            The interpreter instance to use. If None, a new in-memory
            interpreter is created. Reusing an instance allows state to
            persist across multiple tool calls.

        Returns
        -------
        callable
            A LangChain tool that accepts code and returns formatted output.

        Example
        -------
        >>> interp = QdrantInterpreter()
        >>> my_tool = make_eval_interpreter(interp)
        >>> result = my_tool.invoke({"code": "print('hello')"})
        """
        inst = interpreter or QdrantInterpreter()

        @tool("eval_interpreter", args_schema=EvalInput)
        def eval_interpreter(code: str) -> str:
            """Run Python code inside the Qdrant interpreter.

            Parameters
            ----------
            code : str
                Source code to execute. The code may reference the ``client`` to
                create collections, insert vectors, or query the Qdrant database.

            Returns
            -------
            str
                A formatted string containing the captured stdout. If an error
                occurred, the error trace is appended under an ``--- ERROR ---``
                header.
            """
            res = inst.execute(code)
            response = f"--- STDOUT ---\n{res['stdout']}\n"
            if not res["success"]:
                response += f"--- ERROR ---\n{res['stderr']}\n"
            return response

        return eval_interpreter

    # Create a default module-level tool for backward compatibility
    eval_interpreter = make_eval_interpreter()

except Exception:
    # It is okay if langchain_core or pydantic are not installed when this
    # module is imported. Users can still instantiate and use the interpreter
    # directly.
    pass
