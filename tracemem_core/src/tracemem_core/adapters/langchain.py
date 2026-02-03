"""LangChain message adapter.

Converts LangChain messages (HumanMessage, AIMessage, ToolMessage, etc.)
to TraceMem's internal Message format.
"""

from typing import TYPE_CHECKING

from tracemem_core.messages import Message, ToolCall

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class LangChainAdapter:
    """Converts LangChain messages to internal Message type.

    Usage:
        ```python
        from langchain_core.messages import HumanMessage, AIMessage
        from tracemem_core.adapters.langchain import LangChainAdapter

        adapter = LangChainAdapter()
        messages = adapter.convert([
            HumanMessage(content="Read auth.py"),
            AIMessage(content="Here's the file...", tool_calls=[...]),
        ])
        ```
    """

    def convert(self, messages: list["BaseMessage"]) -> list[Message]:
        """Convert a list of LangChain messages.

        Args:
            messages: List of LangChain BaseMessage objects.

        Returns:
            List of internal Message objects.
        """
        return [self.convert_single(msg) for msg in messages]

    def convert_single(self, message: "BaseMessage") -> Message:
        """Convert a single LangChain message.

        Args:
            message: A LangChain BaseMessage object.

        Returns:
            Internal Message object.
        """
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        content = self._extract_content(message)

        if isinstance(message, HumanMessage):
            return Message(role="user", content=content)

        elif isinstance(message, SystemMessage):
            return Message(role="system", content=content)

        elif isinstance(message, AIMessage):
            tool_calls = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    args=tc.get("args", {}),
                )
                for tc in (message.tool_calls or [])
            ]
            return Message(role="assistant", content=content, tool_calls=tool_calls)

        elif isinstance(message, ToolMessage):
            return Message(
                role="tool",
                content=content,
                tool_call_id=message.tool_call_id,
            )

        else:
            return Message(role="user", content=content)

    def _extract_content(self, message: "BaseMessage") -> str:
        """Extract text content from a message.

        Handles both simple string content and complex content lists
        (for multimodal messages).

        Args:
            message: A LangChain BaseMessage object.

        Returns:
            Text content as a string.
        """
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            texts = []
            for block in message.content:
                if isinstance(block, str):
                    texts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts)
        return str(message.content)
