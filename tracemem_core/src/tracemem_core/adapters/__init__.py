"""Adapters for converting framework-specific messages to TraceMem's internal format.

Available adapters:
    - LangChainAdapter: Converts LangChain messages (HumanMessage, AIMessage, etc.)

Usage:
    ```python
    from tracemem_core.adapters.langchain import LangChainAdapter
    from langchain_core.messages import HumanMessage, AIMessage

    adapter = LangChainAdapter()
    messages = adapter.convert([
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ])
    ```
"""

from tracemem_core.adapters.protocol import TraceAdapter

__all__ = ["TraceAdapter"]
