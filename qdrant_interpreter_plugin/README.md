# Qdrant Interpreter Plugin

This repository provides a small but flexible interpreter designed to be
embedded inside agent frameworks such as LangChain and LangGraph. The
interpreter gives your agents a **Python code execution environment** with a
persistent context and a built‐in Qdrant vector database client.

## Why an interpreter?

In LangChain's **Deep Agents** concept, an agent usually calls tools one at a
time, inspecting the result and then choosing the next tool call. For tasks
that require multiple steps (e.g. creating a collection, inserting many
vectors, then querying), this pattern can be inefficient. An embedded
interpreter allows the agent to write a small program that orchestrates
multiple operations in a single turn. This reduces the number of tool calls
and avoids passing large intermediate data back through the LLM context. See
the LangChain blog for more details on interpreters in agents【173042851479231†L105-L110】.

## Features

* **In-memory Qdrant client** – The interpreter creates a local
  ``qdrant_client.QdrantClient`` instance using the ``:memory:`` backend. This
  means you can create collections and insert vectors without running an
  external Qdrant server. If you want to persist data or connect to a
  remote server, pass a different location when instantiating the
  interpreter.
* **Persistent state** – Variables defined in one execution persist to the
  next. This allows the agent to write helper functions or cache results
  across code blocks.
* **Allowlisted API** – Only a small set of objects are available to the
  executed code: ``client``, ``VectorParams``, ``Distance``, ``PointStruct`` and
  ``results_cache``. This keeps the environment controlled and predictable.
* **Optional LangChain tool** – If you have ``langchain-core`` and
  ``pydantic`` installed, the ``eval_interpreter`` function is automatically
  registered as a tool. This tool accepts a ``code`` field and returns
  captured output and error information.

## Installation

The interpreter relies on a few core dependencies:

```bash
pip install qdrant-client langchain langchain-openai langgraph pydantic
```

You can then clone this repository and import the interpreter:

```bash
from qdrant_interpreter_plugin import QdrantInterpreter

interp = QdrantInterpreter()
result = interp.execute("print(client.get_collections())")
print(result["stdout"])
```

When using it with LangChain, import the tool:

```bash
from qdrant_interpreter_plugin import eval_interpreter
```

## Running the Example from the Blog

The blog post demonstrates how to build a **recursive agent loop** that
iteratively writes code, inspects errors, and refines its logic. The
``eval_interpreter`` tool defined in this plugin is designed for that pattern.
For an example of how to build such a loop with LangGraph, see the code in
the blog or the `examples/` directory (if present).

## Repository Structure

* ``qdrant_interpreter.py`` – Implementation of the interpreter and optional
  LangChain tool.
* ``__init__.py`` – Re-exports the interpreter and tool for convenient import.
* ``README.md`` – This file.

## License

This plugin is provided for educational purposes. No specific license is
claimed; you may adapt the code as needed in your own projects.