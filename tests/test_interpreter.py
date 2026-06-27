"""Tests for the Qdrant interpreter."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from qdrant_interpreter_plugin import QdrantInterpreter


class TestQdrantInterpreter:
    """Tests for QdrantInterpreter.execute()."""

    def test_execute_simple_code(self):
        """Test executing simple valid code."""
        interp = QdrantInterpreter()
        result = interp.execute("x = 1 + 1\nprint(x)")
        assert result["success"]
        assert result["stdout"] == "2\n"
        assert result["stderr"] == ""

    def test_execute_with_error(self):
        """Test executing code that raises an exception."""
        interp = QdrantInterpreter()
        result = interp.execute("raise ValueError('test error')")
        assert not result["success"]
        assert "ValueError: test error" in result["stderr"]
        assert result["stdout"] == ""

    def test_execute_with_syntax_error(self):
        """Test executing code with syntax errors."""
        interp = QdrantInterpreter()
        result = interp.execute("def broken(")
        assert not result["success"]
        assert "SyntaxError" in result["stderr"]

    def test_persistent_state(self):
        """Test that variables persist across executions."""
        interp = QdrantInterpreter()
        result1 = interp.execute("x = 42")
        assert result1["success"]

        result2 = interp.execute("print(x)")
        assert result2["success"]
        assert result2["stdout"] == "42\n"

    def test_qdrant_client_available(self):
        """Test that the Qdrant client is available."""
        interp = QdrantInterpreter()
        result = interp.execute("print(type(client).__name__)")
        assert result["success"]
        assert "QdrantClient" in result["stdout"]

    def test_qdrant_models_available(self):
        """Test that Qdrant model classes are available."""
        interp = QdrantInterpreter()
        result = interp.execute(
            "print(VectorParams.__name__, Distance.__name__, PointStruct.__name__)"
        )
        assert result["success"]
        assert "VectorParams" in result["stdout"]
        assert "Distance" in result["stdout"]
        assert "PointStruct" in result["stdout"]

    def test_results_cache_available(self):
        """Test that results_cache is available."""
        interp = QdrantInterpreter()
        result = interp.execute("results_cache['key'] = 'value'\nprint(results_cache)")
        assert result["success"]
        assert "key" in result["stdout"]
        assert "value" in result["stdout"]

    def test_timeout_exceeded(self):
        """Test that code execution times out."""
        interp = QdrantInterpreter(timeout=0.1)
        result = interp.execute("import time; time.sleep(1)")
        assert not result["success"]
        assert "timed out" in result["stderr"].lower()

    def test_timeout_disabled(self):
        """Test that timeout can be disabled."""
        interp = QdrantInterpreter(timeout=None)
        # Quick execution should succeed even with timeout disabled
        result = interp.execute("x = 1")
        assert result["success"]

    def test_thread_safety(self):
        """Test that multiple threads can execute code concurrently."""
        interp = QdrantInterpreter(timeout=5.0)
        results = {}
        errors = {}

        def execute_in_thread(thread_id):
            try:
                code = f"results_cache['thread_{thread_id}'] = {thread_id}"
                result = interp.execute(code)
                results[thread_id] = result
            except Exception as e:
                errors[thread_id] = str(e)

        # Run 10 threads concurrently
        threads = []
        for i in range(10):
            t = threading.Thread(target=execute_in_thread, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # All threads should have succeeded
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        for result in results.values():
            assert result["success"], f"Execution failed: {result}"

    def test_stored_vars_preview(self):
        """Test that stored_vars returns a preview of globals."""
        interp = QdrantInterpreter()
        interp.execute("my_var = 'hello'")
        result = interp.execute("x = 1")
        assert "my_var" in result["stored_vars"]
        assert result["stored_vars"]["my_var"] == "hello"

    def test_stored_vars_truncation(self):
        """Test that stored_vars are truncated to 100 chars."""
        interp = QdrantInterpreter()
        long_string = "x" * 200
        interp.execute(f"long_var = '{long_string}'")
        result = interp.execute("y = 1")
        assert len(result["stored_vars"]["long_var"]) == 100

    def test_unprintable_vars_handled(self):
        """Test that unprintable variables are marked as such."""
        interp = QdrantInterpreter()
        # Create a class with a broken __str__
        interp.execute(
            "class Broken:\n"
            "    def __str__(self):\n"
            "        raise Exception('Cannot print')\n"
            "broken_obj = Broken()"
        )
        result = interp.execute("x = 1")
        assert result["stored_vars"]["broken_obj"] == "<unprintable>"

    def test_create_collection(self):
        """Test creating a Qdrant collection."""
        interp = QdrantInterpreter()
        code = """
client.create_collection(
    collection_name="test",
    vectors_config=VectorParams(size=4, distance=Distance.COSINE),
)
print("Collection created")
"""
        result = interp.execute(code)
        assert result["success"]
        assert "Collection created" in result["stdout"]

    def test_collection_persistence(self):
        """Test that collections persist across executions."""
        interp = QdrantInterpreter()
        # Create a collection
        interp.execute(
            "client.create_collection("
            "collection_name='test',"
            "vectors_config=VectorParams(size=4, distance=Distance.COSINE))"
        )
        # List collections in a separate execution
        result = interp.execute("collections = client.get_collections()\nprint(len(collections.collections))")
        assert result["success"]
        assert "1" in result["stdout"]


class TestMakeEvalInterpreter:
    """Tests for make_eval_interpreter factory."""

    def test_make_eval_interpreter_with_custom_instance(self):
        """Test creating a tool with a custom interpreter instance."""
        pytest.importorskip("langchain_core")
        from qdrant_interpreter_plugin import make_eval_interpreter

        custom_interp = QdrantInterpreter(timeout=60.0)
        tool = make_eval_interpreter(custom_interp)

        # The tool should have a name
        assert tool.name == "eval_interpreter"

        # The tool should have an invoke method
        assert hasattr(tool, "invoke")

    def test_make_eval_interpreter_default_instance(self):
        """Test creating a tool with the default instance."""
        pytest.importorskip("langchain_core")
        from qdrant_interpreter_plugin import make_eval_interpreter

        tool = make_eval_interpreter()

        # The tool should have a name
        assert tool.name == "eval_interpreter"

        # The tool should have an invoke method
        assert hasattr(tool, "invoke")

    def test_make_eval_interpreter_isolation(self):
        """Test that different tools have isolated interpreter instances."""
        pytest.importorskip("langchain_core")
        from qdrant_interpreter_plugin import make_eval_interpreter

        tool1 = make_eval_interpreter()
        tool2 = make_eval_interpreter()

        # Invoke tool1 to set a variable
        result1 = tool1.invoke({"code": "x = 42"})
        assert "--- STDOUT ---" in result1

        # tool2 should not see the variable from tool1
        result2 = tool2.invoke({"code": "print(x)"})
        assert "ERROR" in result2  # NameError: x is not defined
