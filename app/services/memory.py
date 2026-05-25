from collections import defaultdict, deque
from dataclasses import dataclass


MAX_MESSAGES = 5


@dataclass
class ChatMessage:
    role: str
    content: str


class SessionMemory:
    def __init__(self) -> None:
        self._memory: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=MAX_MESSAGES)
        )

    def get_history(self, session_id: str) -> list[ChatMessage]:
        return list(self._memory[session_id])

    def format_history(self, session_id: str) -> str:
        messages = self.get_history(session_id)
        if not messages:
            return "No previous conversation."
        return "\n".join(f"{message.role}: {message.content}" for message in messages)

    def add_exchange(self, session_id: str, user_message: str, assistant_message: str) -> None:
        self._memory[session_id].append(ChatMessage(role="User", content=user_message))
        self._memory[session_id].append(ChatMessage(role="Assistant", content=assistant_message))


session_memory = SessionMemory()
