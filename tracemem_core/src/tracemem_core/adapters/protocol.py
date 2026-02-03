"""Protocol for message adapters."""

from typing import Any, Protocol

from tracemem_core.messages import Message


class TraceAdapter(Protocol):
    """Protocol for converting framework messages to internal format.

    Implementations convert framework-specific message types (LangChain,
    OpenAI, etc.) to TraceMem's internal Message type.
    """

    def convert(self, messages: list[Any]) -> list[Message]:
        """Convert a list of framework-specific messages to internal Messages.

        Args:
            messages: List of framework-specific message objects.

        Returns:
            List of internal Message objects.
        """
        ...

    def convert_single(self, message: Any) -> Message:
        """Convert a single framework-specific message to internal Message.

        Args:
            message: A framework-specific message object.

        Returns:
            Internal Message object.
        """
        ...
