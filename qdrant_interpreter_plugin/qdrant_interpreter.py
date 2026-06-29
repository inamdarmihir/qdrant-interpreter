"""
Qdrant QuickJS Interpreter
===========================

A sandboxed JavaScript runtime for agent code — the interpreter pattern
described in the LangChain blog post *"Give Your Agents an Interpreter"*.

Architecture
------------
Unlike a Python ``exec()``-based approach, code runs inside a **QuickJS**
virtual machine (via ``quickjs-rs``) that has no filesystem, no network,
and no shell by default. Qdrant operations cross an explicit async bridge
into Python: JS calls ``await tools.createQdrantCollection({...})`` and the
host invokes the real LangChain tool.

Key properties (matching the blog spec):

* **Sandboxed JS runtime** — QuickJS; not the host Python process.
* **Persistent REPL state** — ``const``/``let``/``var`` survive across
  :meth:`eval` calls within the same instance (``TOP_LEVEL_CONST_TO_VAR``
  transform makes top-level ``const`` behave like ``var``).
* **Qdrant tools under ``globalThis.tools``** — each Qdrant management tool
  is exposed as an ``async`` function the agent can call with ``await``.
* **``console.log`` capture** — output accumulates in a buffer and is
  returned in the ``"stdout"`` field of :meth:`eval`\'s result.
* **Memory + timeout limits** — Runtime-level memory cap, per-call timeout.
* **Max PTC calls** — optional budget prevents unbounded tool-call loops.
* **Snapshotting** — :meth:`snapshot` / :meth:`restore` persist live JS heap
  state across turns.

Standalone usage::

    from qdrant_client import QdrantClient
    from qdrant_interpreter_plugin.index_manager import QdrantIndexManager
    from qdrant_interpreter_plugin.qdrant_interpreter import QdrantInterpreter

    manager = QdrantIndexManager(QdrantClient(":memory:"))
    interp = QdrantInterpreter(manager)

    result = interp.eval('''
        const info = await tools.createQdrantCollection({
            collection_name: "products",
            use_case: "semantic search using OpenAI ada-002",
            expected_count: 500000,
            priority: "recall",
        });
        console.log("Created:", info);
        info
    ''')
    print(result["result"])   # the string value of the final expression
    print(result["stdout"])   # what console.log printed

LangChain tool::

    from qdrant_interpreter_plugin.qdrant_interpreter import make_eval_interpreter

    tool = make_eval_interpreter(manager)
    response = tool.invoke({"code": "await tools.listQdrantCollections()"})
"""

from __future__ import annotations

import json
from typing import Any

from quickjs_rs import (
    JSError,
    MemoryLimitError,
    Runtime,
    SourceTransform,
    ThreadWorker,
    TimeoutError as QJSTimeoutError,
)

from .index_manager import QdrantIndexManager
from .tools import make_qdrant_tools


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_input(raw: Any) -> dict[str, Any]:
    """Coerce whatever the JS bridge passed to a Python dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return {} if raw is None else {"input": raw}


def _to_camel(snake: str) -> str:
    """``create_qdrant_collection`` → ``createQdrantCollection``."""
    parts = snake.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _js_tools_namespace(bridge_names: dict[str, str]) -> str:
    """Return JS that builds ``globalThis.tools`` from bridge symbol names."""
    props = ", ".join(
        f"{camel}: {sym}" for camel, sym in bridge_names.items()
    )
    return f"globalThis.tools = {{ {props} }}; undefined"


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class QdrantInterpreter:
    """Sandboxed QuickJS REPL with Qdrant tools bridged as async callables.

    One instance = one persistent JS context. Each :meth:`eval` call runs in
    the same QuickJS VM so variables, functions, and module state accumulate
    just like a REPL. Qdrant operations are available as:

    .. code-block:: javascript

        await tools.createQdrantCollection({ ... })
        await tools.upsertQdrantPoints({ ... })
        await tools.searchQdrantCollection({ ... })
        // ... and 7 more

    Parameters
    ----------
    manager:
        The :class:`~qdrant_interpreter_plugin.index_manager.QdrantIndexManager`
        whose tools will be bridged into the JS sandbox.
    timeout:
        Per-eval wall-clock timeout in seconds (default 10). Applied to every
        :meth:`eval` call independently.
    memory_limit_mb:
        QuickJS heap cap in megabytes (default 64). Exceeding this limit
        raises an ``OutOfMemory`` error in the sandbox.
    max_ptc_calls:
        Maximum number of ``tools.*`` bridge invocations allowed per eval.
        Exceeding this budget causes the eval to fail with a
        ``PTCCallBudgetExceeded`` error. ``None`` disables the budget.
    max_result_chars:
        Maximum characters returned in ``result`` and ``stdout`` fields.
        Longer values are truncated.
    capture_console:
        When ``True`` (default), install ``console.log/warn/error`` in the
        JS context and collect their output in the ``"stdout"`` result field.
    """

    def __init__(
        self,
        manager: QdrantIndexManager,
        timeout: float = 10.0,
        memory_limit_mb: int = 64,
        max_ptc_calls: int | None = 256,
        max_result_chars: int = 4_000,
        capture_console: bool = True,
    ) -> None:
        self._max_ptc_calls = max_ptc_calls
        self._max_result_chars = max_result_chars
        self._tools = {t.name: t for t in make_qdrant_tools(manager)}
        self._call_budget: int | None = None  # reset per eval

        # Dedicated OS thread with its own asyncio event loop — required
        # because quickjs_rs objects are !Send (cannot cross thread boundaries).
        self._worker = ThreadWorker(name="qdrant-quickjs")

        # All context setup happens on the worker thread.
        self._ctx = self._worker.run_sync(
            self._ainit(timeout, memory_limit_mb, capture_console)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def eval(self, code: str) -> dict[str, Any]:
        """Evaluate JavaScript code in the sandboxed QuickJS context.

        Parameters
        ----------
        code:
            JavaScript source. May use ``await`` at the top level to call
            bridged Qdrant tools. The value of the final expression is
            returned in ``result``.

        Returns
        -------
        dict
            Keys:

            * ``success`` — ``True`` if evaluation completed without error.
            * ``result`` — String representation of the final JS expression.
            * ``stdout`` — Captured ``console.log/warn/error`` output.
            * ``error`` — Error message string, or ``None`` on success.
            * ``error_type`` — One of ``"JSError"``, ``"Timeout"``,
              ``"OutOfMemory"``, ``"PTCCallBudgetExceeded"``, or ``None``.
        """
        # Reset per-eval call budget
        self._call_budget = self._max_ptc_calls

        return self._worker.run_sync(self._aeval(code))

    def snapshot(self) -> bytes:
        """Serialise the current JS heap to bytes for cross-turn persistence.

        The snapshot captures all live variables, functions, and prototype
        state. Host-function bridges (the ``tools.*`` callables) are
        *not* included — they are re-registered on :meth:`restore`.

        Returns
        -------
        bytes
            Opaque snapshot payload; pass to :meth:`restore`.
        """
        return self._worker.run_sync(self._asnapshot())

    def restore(self, payload: bytes) -> None:
        """Restore a previously captured snapshot.

        Parameters
        ----------
        payload:
            Bytes returned by :meth:`snapshot`.
        """
        self._worker.run_sync(self._arestore(payload))

    def close(self) -> None:
        """Shut down the QuickJS context and the worker thread."""
        self._worker.run_sync(self._aclose())
        self._worker.close()

    # ------------------------------------------------------------------
    # Internal — run on the worker thread
    # ------------------------------------------------------------------

    async def _ainit(
        self,
        timeout: float,
        memory_limit_mb: int,
        capture_console: bool,
    ):
        """Create Runtime + Context and install all bridges. Worker-thread only."""
        runtime = Runtime(
            memory_limit=memory_limit_mb * 1024 * 1024,
            # Converts top-level `const`/`let` to `var` so they survive
            # across separate eval() calls — the key REPL persistence trick.
            transform_flags=SourceTransform.TOP_LEVEL_CONST_TO_VAR,
        )
        ctx = runtime.new_context(timeout=timeout)

        # Install console capture
        if capture_console:
            self._console_buf: list[str] = []
            buf = self._console_buf

            async def _log(*args: Any) -> None:
                buf.append(" ".join(str(a) for a in args))

            async def _warn(*args: Any) -> None:
                buf.append("[warn] " + " ".join(str(a) for a in args))

            async def _err(*args: Any) -> None:
                buf.append("[error] " + " ".join(str(a) for a in args))

            ctx.register("__console_log", _log, is_async=True)
            ctx.register("__console_warn", _warn, is_async=True)
            ctx.register("__console_error", _err, is_async=True)
            ctx.eval(
                "globalThis.console = {"
                " log: (...a) => __console_log(...a),"
                " warn: (...a) => __console_warn(...a),"
                " error: (...a) => __console_error(...a),"
                "}; undefined"
            )
        else:
            self._console_buf = []

        # Bridge each Qdrant tool as an async callable
        bridge_names: dict[str, str] = {}
        for tool_name, tool in self._tools.items():
            camel = _to_camel(tool_name)
            sym = f"__bridge_{camel}"
            bridge_names[camel] = sym
            # Capture tool in closure
            _tool = tool

            async def _bridge(raw: Any = None, *, _t=_tool) -> str:
                if self._call_budget is not None:
                    if self._call_budget <= 0:
                        raise RuntimeError(
                            "PTCCallBudgetExceeded: maximum number of "
                            "tools.* calls reached for this eval."
                        )
                    self._call_budget -= 1
                args = _normalize_input(raw)
                result = _t.invoke(args)
                return str(result)

            ctx.register(sym, _bridge, is_async=True)

        # Build globalThis.tools namespace atomically
        ctx.eval(_js_tools_namespace(bridge_names))
        return ctx

    async def _aeval(self, code: str) -> dict[str, Any]:
        """Core eval — must run on the worker thread."""
        self._console_buf.clear()

        try:
            handle = await self._ctx.eval_handle_async(code)
            try:
                if handle.is_promise():
                    resolved = await handle.await_promise()
                else:
                    resolved = handle
                try:
                    value = resolved.to_python()
                    result_str = (
                        str(value) if value is not None else "undefined"
                    )[: self._max_result_chars]
                    return {
                        "success": True,
                        "result": result_str,
                        "stdout": self._drain_console(),
                        "error": None,
                        "error_type": None,
                    }
                except Exception:
                    result_str = str(resolved)[: self._max_result_chars]
                    return {
                        "success": True,
                        "result": result_str,
                        "stdout": self._drain_console(),
                        "error": None,
                        "error_type": None,
                    }
                finally:
                    if "resolved" in dir() and resolved is not handle:
                        resolved.dispose()
            finally:
                handle.dispose()

        except QJSTimeoutError as e:
            return self._error_result("Timeout", str(e))
        except MemoryLimitError as e:
            return self._error_result("OutOfMemory", str(e))
        except RuntimeError as e:
            msg = str(e)
            if "PTCCallBudgetExceeded" in msg:
                return self._error_result("PTCCallBudgetExceeded", msg)
            return self._error_result("RuntimeError", msg)
        except JSError as e:
            return self._error_result(e.name, f"{e.message}\n{e.stack or ''}".strip())
        except Exception as e:
            return self._error_result(type(e).__name__, str(e))

    async def _asnapshot(self) -> bytes:
        snapshot = self._ctx.create_snapshot()
        return snapshot.to_bytes()

    async def _arestore(self, payload: bytes) -> None:
        from quickjs_rs import Snapshot
        snap = Snapshot.from_bytes(payload)
        # Restore re-installs module scope; host bridges survive because
        # they're registered on the Runtime, not serialised into the snapshot.
        self._ctx.restore_snapshot(snap, inject_globals=True)

    async def _aclose(self) -> None:
        self._ctx.close()

    def _drain_console(self) -> str:
        out = "\n".join(self._console_buf)[: self._max_result_chars]
        self._console_buf.clear()
        return out

    def _error_result(self, error_type: str, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "result": None,
            "stdout": self._drain_console(),
            "error": message,
            "error_type": error_type,
        }


# ---------------------------------------------------------------------------
# LangChain tool integration (requires langchain-core + pydantic)
# ---------------------------------------------------------------------------

try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    class EvalInput(BaseModel):
        """Input schema for the QuickJS eval tool."""

        code: str = Field(
            description=(
                "JavaScript code to execute in the sandboxed QuickJS interpreter.\n"
                "\n"
                "The sandbox has no filesystem, network, or shell access. Qdrant\n"
                "operations are available as async functions under globalThis.tools:\n"
                "\n"
                "  await tools.createQdrantCollection({ collection_name, use_case, ... })\n"
                "  await tools.ensureQdrantCollection({ collection_name, use_case, ... })\n"
                "  await tools.upsertQdrantPoints({ collection_name, points })\n"
                "  await tools.searchQdrantCollection({ collection_name, query_vector, ... })\n"
                "  await tools.listQdrantCollections()\n"
                "  await tools.describeQdrantCollection({ collection_name })\n"
                "  await tools.createQdrantPayloadIndex({ collection_name, field_name, ... })\n"
                "  await tools.recommendQdrantParams({ use_case, ... })\n"
                "  await tools.deleteQdrantCollection({ collection_name })\n"
                "  await tools.optimizeQdrantCollection({ collection_name, priority })\n"
                "\n"
                "State (variables, functions) persists across calls within a session.\n"
                "The value of the LAST expression is returned as the result.\n"
                "Use console.log() for intermediate output."
            )
        )

    def make_eval_interpreter(
        manager: QdrantIndexManager | None = None,
        interpreter: QdrantInterpreter | None = None,
        **kwargs: Any,
    ):
        """Create a LangChain tool wrapping a :class:`QdrantInterpreter`.

        Parameters
        ----------
        manager:
            The index manager whose Qdrant tools are bridged into the sandbox.
            Required unless ``interpreter`` is given.
        interpreter:
            An existing :class:`QdrantInterpreter` instance to reuse. When
            provided, ``manager`` and ``kwargs`` are ignored.
        **kwargs:
            Forwarded to :class:`QdrantInterpreter` if a new one is created.

        Returns
        -------
        callable
            A LangChain ``@tool`` that accepts ``{"code": "..."}`` and returns
            a formatted string with ``RESULT``, ``STDOUT``, and ``ERROR``
            sections.
        """
        if interpreter is None:
            if manager is None:
                raise ValueError("Provide either manager= or interpreter=")
            inst = QdrantInterpreter(manager, **kwargs)
        else:
            inst = interpreter

        @tool("eval_interpreter", args_schema=EvalInput)
        def eval_interpreter(code: str) -> str:
            """Run JavaScript in the sandboxed QuickJS Qdrant interpreter."""
            res = inst.eval(code)
            parts = []
            if res["stdout"]:
                parts.append(f"<stdout>\n{res['stdout']}\n</stdout>")
            if res["success"]:
                parts.append(f"<result>\n{res['result']}\n</result>")
            else:
                parts.append(
                    f"<error type=\"{res['error_type']}\">\n{res['error']}\n</error>"
                )
            return "\n".join(parts) if parts else "<result>undefined</result>"

        return eval_interpreter

except ImportError:
    # langchain-core not installed — standalone QdrantInterpreter still works.
    pass
