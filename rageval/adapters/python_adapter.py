"""Query a RAG pipeline implemented as a Python function (sync or async)."""

import asyncio
import importlib
import inspect
import os
import sys

from rageval.adapters import AdapterError


class PythonAdapter:
    def __init__(self, target: str):
        """`target` is 'package.module:function'."""
        module_name, _, func_name = target.partition(":")
        if not module_name or not func_name:
            raise AdapterError(f"python adapter target must be 'module:function', got '{target}'")
        # Console scripts don't put the working directory on sys.path; add it so
        # `python:my_app.rag:answer` resolves from the project you run rageval in.
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise AdapterError(f"cannot import '{module_name}': {exc}") from exc
        self.func = getattr(module, func_name, None)
        if not callable(self.func):
            raise AdapterError(f"'{func_name}' is not a function in '{module_name}'")

    async def query(self, question: str) -> dict:
        if inspect.iscoroutinefunction(self.func):
            return await self.func(question)
        return await asyncio.to_thread(self.func, question)
